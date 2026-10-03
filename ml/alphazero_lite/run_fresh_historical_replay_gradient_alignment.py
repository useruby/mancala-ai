"""Frozen observational audit of fresh and historical replay gradient alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train
from ml.alphazero_lite.checkpoint_phase_selection import (
    active_pit_stones_from_encoded_state,
)
from ml.alphazero_lite.replay_gradient_mapping import (
    make_replay_mapping,
    partition_id,
    split_compact_rows,
)
from ml.alphazero_lite.replay_source_attribution import sha256_file
from ml.alphazero_lite.train import PolicyValueNet

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = ROOT / "docs/data/seed461-batch-order-sensitivity-confirmation.json"
RECOVERY = ROOT / "docs/data/seed461-o0-checkpoint-recovery-provenance.json"
SEED455 = (
    ROOT
    / ".tmp/seed48-nextgen-s455-default-value/runs/seed48-nextgen-s455-default-value-iter1/checkpoint.npz"
)
E4 = ROOT / ".tmp/seed461-order-confirmation/training/O0/E4.npz"
SEED455_SHA = "c18beeaa16ebaa038cd383cfd222705c636e2b6398c7270e6674d33231aa9dd1"
VALUE_WEIGHT, HUBER_DELTA, PARTITIONS, CHUNK = 0.3, 1.0, 4, 1024
DECOMPOSITION_ATOL, DECOMPOSITION_RTOL = 2e-5, 2e-5
COHORTS = ("all", ">32", "17-32", "<=16")
TRUNK_GROUPS = (
    "input_projection",
    "residual_block_0",
    "residual_block_1",
    "residual_block_2",
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return sha256_file(path)


def verify_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registration = _json(REGISTRATION)
    sources = registration["training"]["replays"]
    for source in sources:
        path = Path(source["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or _hash(path) != source["sha256"]:
            raise RuntimeError(f"registered_source_missing_or_changed:{source['name']}")
        source["resolved_path"] = path
    if _hash(SEED455) != SEED455_SHA:
        raise RuntimeError("registered_seed455_checkpoint_missing_or_changed")
    receipt = _json(RECOVERY)
    e4_record = next(row for row in receipt["checkpoint_files"] if row["epoch"] == "E4")
    if _hash(E4) != e4_record["expected_sha256"]:
        raise RuntimeError("registered_original_o0_e4_missing_or_changed")
    return registration, sources


def load_compact(
    sources: list[dict[str, Any]], seed: int, val_split: float
) -> tuple[Any, ...]:
    paths = [source["resolved_path"] for source in sources]
    weights = [int(source["weight"]) for source in sources]
    modes = [source["value_target_mode"] for source in sources]
    loaded = train.load_jsonl_replay(
        paths,
        weights,
        policy_target_mode="sharpened",
        value_target_mode="default",
        replay_value_target_modes=modes,
        include_policy_loss_weights=True,
    )
    x, p, v, replay_indexes, policy_weights = loaded
    source_names: list[str] = []
    source_rows: list[list[dict[str, Any]]] = []
    for source, path, mode in zip(sources, paths, modes):
        sx, sp, sv, sq = train.load_jsonl(
            path,
            policy_target_mode="sharpened",
            value_target_mode=mode,
            include_policy_loss_weights=True,
        )
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not (len(sx) == len(sp) == len(sv) == len(sq) == len(rows)):
            raise RuntimeError(f"compact_source_row_alignment_failed:{source['name']}")
        source_names.extend([source["name"]] * len(sx))
        source_rows.append(rows)
    if len(source_names) != len(x):
        raise RuntimeError("compact_source_label_count_mismatch")
    train_pos, val_pos = split_compact_rows(
        replay_indexes, seed=seed, validation_split=val_split
    )
    mapping = make_replay_mapping(source_names, replay_indexes, train_pos, val_pos)
    return x, p, v, policy_weights, replay_indexes, mapping, source_rows


def _flatten(
    grads: tuple[torch.Tensor | None, ...], params: tuple[torch.nn.Parameter, ...]
) -> torch.Tensor:
    chunks = []
    for grad, param in zip(grads, params):
        chunks.append(
            (torch.zeros_like(param) if grad is None else grad)
            .detach()
            .float()
            .reshape(-1)
            .cpu()
        )
    return torch.cat(chunks)


def _one_source_gradient(
    model: PolicyValueNet,
    trunk: tuple[torch.nn.Parameter, ...],
    x: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
    q: np.ndarray,
    m: np.ndarray,
    selected_ids: np.ndarray,
    denominator_policy: float,
    denominator_value: float,
    objective: str,
) -> tuple[torch.Tensor, int]:
    accum = torch.zeros(
        sum(parameter.numel() for parameter in trunk), dtype=torch.float32
    )
    missing_count = 0
    for offset in range(0, len(selected_ids), CHUNK):
        ids = selected_ids[offset : offset + CHUNK]
        bx, bp, bv = (torch.from_numpy(array[ids]) for array in (x, p, v))
        bq, bm = torch.from_numpy(q[ids]), torch.from_numpy(m[ids])
        logits, prediction = model(bx)
        legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[ids]))
        policy_rows = train.compute_policy_cross_entropy(
            logits.masked_fill(legal <= 0, -1e9), bp
        )
        value_rows = train.compute_value_loss_vector(
            prediction, bv, value_loss="huber", huber_delta=HUBER_DELTA
        )
        p_loss = (
            (policy_rows * bq * bm).sum() / denominator_policy
            if denominator_policy > 0
            else policy_rows.sum() * 0.0
        )
        v_loss = VALUE_WEIGHT * (value_rows * bm).sum() / denominator_value
        if objective == "policy":
            loss = p_loss
        elif objective == "weighted_value":
            loss = v_loss
        elif objective == "combined":
            loss = p_loss + v_loss
        else:
            raise ValueError(f"unknown_gradient_objective:{objective}")
        grad = torch.autograd.grad(loss, trunk, allow_unused=True)
        missing_count += sum(item is None for item in grad)
        accum += _flatten(grad, trunk)
        del logits, prediction, policy_rows, value_rows
    return accum, missing_count


def _norm(vec: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vec.double()))


def _geometry(fresh: torch.Tensor, history: torch.Tensor) -> dict[str, Any]:
    fresh64, history64 = fresh.double(), history.double()
    f2, h2 = float(torch.dot(fresh64, fresh64)), float(torch.dot(history64, history64))
    fn, hn = f2**0.5, h2**0.5
    dot = float(torch.dot(fresh64, history64))
    combined = fresh + history
    return {
        "dot": dot,
        "cosine": dot / (fn * hn) if fn and hn else None,
        "fresh_norm": fn,
        "historical_norm": hn,
        "historical_to_fresh_norm": hn / fn if fn else None,
        "combined_norm": _norm(combined),
        "cancellation": 1.0 - _norm(combined) / (fn + hn) if fn + hn else 0.0,
        "retained_fresh_projection": float(torch.dot(combined, fresh)) / f2
        if f2
        else None,
    }


def _state_cohorts(x: np.ndarray, train_mult: np.ndarray) -> dict[str, np.ndarray]:
    stones = np.asarray([active_pit_stones_from_encoded_state(row) for row in x])
    training = train_mult > 0
    return {
        "all": training,
        ">32": training & (stones > 32),
        "17-32": training & (stones >= 17) & (stones <= 32),
        "<=16": training & (stones <= 16),
    }


def _partition_masks(mapping: Any) -> np.ndarray:
    return np.asarray(
        [
            partition_id(name, int(local), PARTITIONS)
            for name, local in zip(
                mapping.compact_source_names, mapping.compact_local_row_ids
            )
        ],
        dtype=np.int64,
    )


def _frozen_source_accounting(
    sources: list[dict[str, Any]], arrays: tuple[Any, ...]
) -> dict[str, Any]:
    x, _p, _v, q, replay, mapping, rows_by_source = arrays
    cohorts = _state_cohorts(x, mapping.train_multiplicity)
    accounting: dict[str, Any] = {}
    for source_idx, source in enumerate(sources):
        ids = np.flatnonzero(mapping.compact_source_ids == source_idx)
        accounting[source["name"]] = {
            "sha256": source["sha256"],
            "weight": int(source["weight"]),
            "value_target_mode": source["value_target_mode"],
            "raw_rows": len(rows_by_source[source_idx]),
            "unique_compact_rows": int(len(ids)),
            "training_unique_rows": int(
                np.count_nonzero(mapping.train_multiplicity[ids])
            ),
            "training_weighted_replay_mass": int(mapping.train_multiplicity[ids].sum()),
            "training_policy_active_mass": float(
                np.dot(q[ids], mapping.train_multiplicity[ids])
            ),
            "validation_unique_rows": int(
                np.count_nonzero(mapping.validation_multiplicity[ids])
            ),
            "validation_weighted_replay_mass": int(
                mapping.validation_multiplicity[ids].sum()
            ),
            "cohort_coverage": {
                cohort: {
                    "training_unique_rows": int(np.count_nonzero(mask[ids])),
                    "weighted_replay_mass": int(
                        mapping.train_multiplicity[ids][mask[ids]].sum()
                    ),
                    "policy_active_mass": float(
                        np.dot(
                            q[ids][mask[ids]],
                            mapping.train_multiplicity[ids][mask[ids]],
                        )
                    ),
                }
                for cohort, mask in cohorts.items()
            },
        }
    return accounting


def _partition_accounting(
    sources: list[dict[str, Any]], arrays: tuple[Any, ...]
) -> dict[str, Any]:
    x, _p, _v, q, _replay, mapping, _rows = arrays
    assignments = _partition_masks(mapping)
    cohorts = _state_cohorts(x, mapping.train_multiplicity)
    result: dict[str, Any] = {}
    for partition in range(PARTITIONS):
        result[str(partition)] = {}
        for source_idx, source in enumerate(sources):
            ids = np.flatnonzero(
                (mapping.compact_source_ids == source_idx)
                & (assignments == partition)
                & (mapping.train_multiplicity > 0)
            )
            result[str(partition)][source["name"]] = {
                "unique_training_rows": len(ids),
                "weighted_replay_mass": int(mapping.train_multiplicity[ids].sum()),
                "policy_active_mass": float(
                    np.dot(q[ids], mapping.train_multiplicity[ids])
                ),
                "cohort_coverage": {
                    cohort: {
                        "unique_rows": int(np.count_nonzero(cohorts[cohort][ids])),
                        "weighted_replay_mass": int(
                            mapping.train_multiplicity[ids][cohorts[cohort][ids]].sum()
                        ),
                        "policy_active_mass": float(
                            np.dot(
                                q[ids][cohorts[cohort][ids]],
                                mapping.train_multiplicity[ids][cohorts[cohort][ids]],
                            )
                        ),
                    }
                    for cohort in COHORTS
                },
            }
    return result


def _audit_checkpoint(
    label: str,
    checkpoint: Path,
    expected_hash: str,
    arrays: tuple[Any, ...],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    x, p, v, q, replay, mapping, source_rows = arrays
    before_hash = _hash(checkpoint)
    if before_hash != expected_hash:
        raise RuntimeError(f"checkpoint_identity_changed:{label}")
    model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    train.load_checkpoint_into_model(model, checkpoint)
    model.eval()
    frozen = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    groups = __import__(
        "ml.alphazero_lite.policy_value_gradient_audit", fromlist=["parameter_groups"]
    ).parameter_groups(model)
    trunk = tuple(parameter for group in TRUNK_GROUPS for parameter in groups[group])
    group_sizes = {
        group: sum(parameter.numel() for parameter in groups[group])
        for group in TRUNK_GROUPS
    }
    group_ranges: dict[str, slice] = {}
    group_offset = 0
    for group in TRUNK_GROUPS:
        group_ranges[group] = slice(group_offset, group_offset + group_sizes[group])
        group_offset += group_sizes[group]
    cohorts = _state_cohorts(x, mapping.train_multiplicity)
    partition_ids = _partition_masks(mapping)
    source_ids = mapping.compact_source_ids
    output: dict[str, Any] = {
        "checkpoint_sha256": before_hash,
        "cohort_objective": "cohort_average; not original minibatch sequence or Adam update",
        "source_accounting": {},
        "results": {},
    }
    for idx, source in enumerate(sources):
        row_ids = np.flatnonzero(source_ids == idx)
        name = source["name"]
        output["source_accounting"][name] = {
            "raw_rows": len(source_rows[idx]),
            "unique_compact_rows": len(row_ids),
            "training_unique_rows": int(
                np.count_nonzero(mapping.train_multiplicity[row_ids])
            ),
            "training_weighted_replay_mass": int(
                mapping.train_multiplicity[row_ids].sum()
            ),
            "training_policy_active_mass": float(
                np.dot(q[row_ids], mapping.train_multiplicity[row_ids])
            ),
            "validation_unique_rows": int(
                np.count_nonzero(mapping.validation_multiplicity[row_ids])
            ),
            "validation_weighted_replay_mass": int(
                mapping.validation_multiplicity[row_ids].sum()
            ),
            "cohort_coverage": {
                cohort: {
                    "unique_rows": int(np.count_nonzero(mask[row_ids])),
                    "weighted_replay_mass": int(
                        mapping.train_multiplicity[row_ids][mask[row_ids]].sum()
                    ),
                    "policy_active_mass": float(
                        np.dot(
                            q[row_ids][mask[row_ids]],
                            mapping.train_multiplicity[row_ids][mask[row_ids]],
                        )
                    ),
                }
                for cohort, mask in cohorts.items()
            },
        }
    for cohort_name, cohort_mask in cohorts.items():
        output["results"][cohort_name] = {}
        for partition in (None, *range(PARTITIONS)):
            selected_mask = cohort_mask.copy()
            if partition is not None:
                selected_mask &= partition_ids == partition
            ids = np.flatnonzero(selected_mask)
            mult = mapping.train_multiplicity
            den_p = float(np.dot(q[ids], mult[ids]))
            den_v = float(mult[ids].sum())
            key = "all" if partition is None else f"partition_{partition}"
            partition_result: dict[str, Any] = {
                "selected_compact_rows": len(ids),
                "objectives": {},
            }
            for objective in ("policy", "weighted_value", "combined"):
                partition_result["objectives"][objective] = {}
                all_source_vectors: dict[str, torch.Tensor] = {}
                all_source_missing: dict[str, int] = {}
                for source_idx, source in enumerate(sources):
                    source_selected = ids[source_ids[ids] == source_idx]
                    vector, missing_count = _one_source_gradient(
                        model,
                        trunk,
                        x,
                        p,
                        v,
                        q,
                        mult,
                        source_selected,
                        den_p,
                        den_v,
                        objective,
                    )
                    all_source_vectors[source["name"]] = vector
                    all_source_missing[source["name"]] = missing_count
                decomposition: dict[str, Any] = {}
                if objective == "combined":
                    max_residual = 0.0
                    for source_idx, source in enumerate(sources):
                        source_selected = ids[source_ids[ids] == source_idx]
                        policy_vector, _ = _one_source_gradient(
                            model,
                            trunk,
                            x,
                            p,
                            v,
                            q,
                            mult,
                            source_selected,
                            den_p,
                            den_v,
                            "policy",
                        )
                        value_vector, _ = _one_source_gradient(
                            model,
                            trunk,
                            x,
                            p,
                            v,
                            q,
                            mult,
                            source_selected,
                            den_p,
                            den_v,
                            "weighted_value",
                        )
                        residual = (
                            all_source_vectors[source["name"]]
                            - policy_vector
                            - value_vector
                        )
                        residual_norm = _norm(residual)
                        tolerance = DECOMPOSITION_ATOL + DECOMPOSITION_RTOL * _norm(
                            policy_vector + value_vector
                        )
                        if residual_norm > tolerance:
                            raise RuntimeError(
                                f"gradient_decomposition_failed:{label}:{cohort_name}:{key}:{source['name']}"
                            )
                        max_residual = max(max_residual, residual_norm)
                    decomposition = {
                        "max_source_residual_l2": max_residual,
                        "absolute_tolerance": DECOMPOSITION_ATOL,
                        "relative_tolerance": DECOMPOSITION_RTOL,
                        "passed": True,
                    }
                for group_name in TRUNK_GROUPS:
                    per_source = {
                        name: vector[group_ranges[group_name]]
                        for name, vector in all_source_vectors.items()
                    }
                    fresh = per_source["fresh"]
                    history = sum(
                        (g for n, g in per_source.items() if n != "fresh"),
                        torch.zeros_like(fresh),
                    )
                    pairwise = {
                        left: {
                            right: float(torch.dot(gl.double(), gr.double()))
                            for right, gr in per_source.items()
                        }
                        for left, gl in per_source.items()
                    }
                    partition_result["objectives"][objective][group_name] = {
                        "fresh_historical": _geometry(fresh, history),
                        "source_norms": {
                            name: _norm(grad) for name, grad in per_source.items()
                        },
                        "zero_gradient_sources": [
                            name
                            for name, grad in per_source.items()
                            if _norm(grad) == 0.0
                        ],
                        "unused_trunk_parameter_count_by_source": all_source_missing,
                        "undefined_direction_metrics": [
                            metric
                            for metric in (
                                "cosine",
                                "historical_to_fresh_norm",
                                "retained_fresh_projection",
                            )
                            if _geometry(fresh, history)[metric] is None
                        ],
                        "source_pairwise_dots": pairwise,
                        "historical_contributions": {
                            name: {
                                "norm": _norm(grad),
                                "dot_fresh": float(
                                    torch.dot(fresh.double(), grad.double())
                                ),
                            }
                            for name, grad in per_source.items()
                            if name != "fresh"
                        },
                    }
                fresh = all_source_vectors["fresh"]
                history = sum(
                    (g for n, g in all_source_vectors.items() if n != "fresh"),
                    torch.zeros_like(fresh),
                )
                partition_result["objectives"][objective]["shared_trunk"] = {
                    "fresh_historical": _geometry(fresh, history),
                    "source_norms": {
                        name: _norm(grad) for name, grad in all_source_vectors.items()
                    },
                    "zero_gradient_sources": [
                        name
                        for name, grad in all_source_vectors.items()
                        if _norm(grad) == 0.0
                    ],
                    "unused_trunk_parameter_count_by_source": all_source_missing,
                    "undefined_direction_metrics": [
                        metric
                        for metric, value in _geometry(fresh, history).items()
                        if value is None
                    ],
                    "source_pairwise_dots": {
                        left: {
                            right: float(torch.dot(gl.double(), gr.double()))
                            for right, gr in all_source_vectors.items()
                        }
                        for left, gl in all_source_vectors.items()
                    },
                    "historical_contributions": {
                        name: {
                            "norm": _norm(grad),
                            "dot_fresh": float(
                                torch.dot(fresh.double(), grad.double())
                            ),
                        }
                        for name, grad in all_source_vectors.items()
                        if name != "fresh"
                    },
                }
                if objective == "combined":
                    partition_result["objectives"][objective]["shared_trunk"][
                        "decomposition_check"
                    ] = decomposition
            output["results"][cohort_name][key] = partition_result
    if any(
        not torch.equal(frozen[name], tensor)
        for name, tensor in model.state_dict().items()
    ):
        raise RuntimeError(f"checkpoint_model_mutated:{label}")
    if _hash(checkpoint) != before_hash:
        raise RuntimeError(f"checkpoint_file_mutated:{label}")
    output["immutability_checks"] = {
        "model_parameters_and_buffers_unchanged": True,
        "checkpoint_file_unchanged": True,
        "optimizer_steps": 0,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    registration, sources = verify_inputs()
    training = registration["training"]
    arrays = load_compact(
        sources, int(training["seed"]), float(training["validation_split"])
    )
    x, _p, _v, q, replay, mapping, _rows = arrays
    manifest = {
        "schema": "fresh_historical_replay_gradient_alignment_manifest_v1",
        "registration_sha256": _hash(REGISTRATION),
        "recovery_receipt_sha256": _hash(RECOVERY),
        "recovery_utility_sha256": _hash(
            ROOT / "ml/alphazero_lite/prepare_seed398_recovery_bundle.py"
        ),
        "replay_training_config_sha256": _hash(
            ROOT / "docs/data/alphazero-lite-replay-source-attribution/plan.json"
        ),
        "seed455_checkpoint_sha256": SEED455_SHA,
        "original_o0_e4_checkpoint_sha256": _hash(E4),
        "registered_split": {
            "seed": training["seed"],
            "validation_split": training["validation_split"],
        },
        "loaded_compact_rows": len(x),
        "weighted_replay_positions": len(replay),
        "source_config": [
            {k: s[k] for k in ("name", "sha256", "weight", "value_target_mode")}
            for s in sources
        ],
        "source_split_accounting": _frozen_source_accounting(sources, arrays),
        "partition_source_accounting": _partition_accounting(sources, arrays),
        "code_sha256": {
            f: _hash(ROOT / "ml/alphazero_lite" / f)
            for f in (
                "train.py",
                "policy_value_gradient_audit.py",
                "checkpoint_trajectory_diagnostic.py",
                "replay_gradient_mapping.py",
                Path(__file__).name,
                "verify_fresh_historical_replay_gradient_alignment.py",
                "test_replay_gradient_mapping.py",
            )
        },
        "partition_construction": "SHA256('fresh-historical-gradient-v1\\0' + source_name + '\\0' + source-local compact row id), first 8 bytes big-endian modulo 4; copies remain together",
        "partition_interpretation": "Disjoint source-row partitions measure consistency only; they are not independent games or statistical confidence intervals.",
        "partition_ids_sha256": hashlib.sha256(
            _partition_masks(mapping).tobytes()
        ).hexdigest(),
        "objective": {
            "type": "cohort-average",
            "policy": "sum(m*q*CE)/sum(m*q)",
            "value": "0.3*sum(m*Huber)/sum(m)",
            "huber": "smooth_l1 beta=1.0",
            "policy_target_mode": "sharpened",
            "legal_mask": "production train.legal_mask_matrix_for_encoded_states",
            "optimizer_steps": 0,
            "original_training_config": {
                "value_loss": "huber",
                "huber_delta": HUBER_DELTA,
                "value_loss_weight": VALUE_WEIGHT,
                "policy_target_mode": "sharpened",
                "exact_root_policy_loss_weight": 1.0,
            },
        },
        "decision_rule": "recommend only when seed455 primary >32 shared_trunk combined meets dot<0, H_norm>=F_norm, retained_projection<=0.5 and the same three conditions hold for both combined and policy-only in at least 3/4 partitions; otherwise no supported follow-up",
        "interpretation": "Observational diagnostic only; no evidence of improved playing strength.",
        "policy_active_rows": int(np.count_nonzero(q[mapping.train_multiplicity > 0])),
        "training_replay_mass": int(mapping.train_multiplicity.sum()),
    }
    manifest_path = args.out.with_name(args.out.stem + "-manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_digest = _hash(manifest_path)
    e4_record = next(
        row for row in _json(RECOVERY)["checkpoint_files"] if row["epoch"] == "E4"
    )
    result = {
        "schema": "fresh_historical_replay_gradient_alignment_correction_v1",
        "manifest": manifest,
        "manifest_sha256": manifest_digest,
        "seed455": _audit_checkpoint("seed455", SEED455, SEED455_SHA, arrays, sources),
        "original_o0_e4_secondary": _audit_checkpoint(
            "O0_E4", E4, e4_record["expected_sha256"], arrays, sources
        ),
    }
    result["classification"] = classify_result(result["seed455"])
    original_summary = (
        ROOT / "docs/data/seed455-fresh-historical-replay-gradient-alignment.json"
    )
    original = _json(original_summary)
    comparison: dict[str, Any] = {}
    for checkpoint_name in ("seed455", "original_o0_e4_secondary"):
        for objective in ("policy", "combined"):
            differences = []
            new_results = result[checkpoint_name]["results"]
            old_results = original[checkpoint_name]["results"]
            for cohort, partitions in new_results.items():
                for part, entry in partitions.items():
                    for group, metrics in entry["objectives"][objective].items():
                        old = old_results[cohort][part]["objectives"][objective][group]
                        for metric, value in metrics["fresh_historical"].items():
                            old_value = old["fresh_historical"][metric]
                            if value is None or old_value is None:
                                if value is not old_value:
                                    differences.append(
                                        {
                                            "cohort": cohort,
                                            "partition": part,
                                            "group": group,
                                            "metric": metric,
                                            "old": old_value,
                                            "corrected": value,
                                        }
                                    )
                            elif abs(value - old_value) > 1e-6 * max(
                                1.0, abs(old_value)
                            ):
                                differences.append(
                                    {
                                        "cohort": cohort,
                                        "partition": part,
                                        "group": group,
                                        "metric": metric,
                                        "old": old_value,
                                        "corrected": value,
                                        "absolute_difference": abs(value - old_value),
                                    }
                                )
            comparison[f"{checkpoint_name}:{objective}"] = {
                "tolerance": "abs(diff) <= 1e-6 * max(1, abs(original))",
                "within_tolerance": not differences,
                "differences": differences,
            }
    result["correction"] = {
        "label": "Post-publication correction to #402 weighted-value gradient audit",
        "observed_after_results": True,
        "original_summary_sha256": "53144cb5c8720f652fdd2d620c406d9b6a27a0898a4a1b8cd3652a8b27e1a8ad",
        "original_manifest_sha256": "3d20de91085db7e5e60319f3f8b0d7a8c4be18dbaa00ed2a6dd33ee90132b073",
        "original_files": [
            "seed455-fresh-historical-replay-gradient-alignment-original/summary.json",
            "seed455-fresh-historical-replay-gradient-alignment-original/manifest.json",
        ],
        "comparison_to_original_policy_and_combined": comparison,
        "weighted_value_correction": "Explicit weighted_value dispatch now returns only 0.3 times the Huber gradient; #402 accidentally returned policy plus weighted value.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _meets_rule(metrics: dict[str, Any]) -> bool:
    return (
        metrics["dot"] < 0.0
        and metrics["historical_norm"] >= metrics["fresh_norm"]
        and metrics["retained_fresh_projection"] is not None
        and metrics["retained_fresh_projection"] <= 0.5
    )


def classify_result(checkpoint: dict[str, Any]) -> dict[str, Any]:
    primary = checkpoint["results"][">32"]
    combined = primary["all"]["objectives"]["combined"]["shared_trunk"][
        "fresh_historical"
    ]
    policy_all = primary["all"]["objectives"]["policy"]["shared_trunk"][
        "fresh_historical"
    ]
    partition_keys = [f"partition_{index}" for index in range(PARTITIONS)]
    supporting = []
    for key in partition_keys:
        combined_metrics = primary[key]["objectives"]["combined"]["shared_trunk"][
            "fresh_historical"
        ]
        policy_metrics = primary[key]["objectives"]["policy"]["shared_trunk"][
            "fresh_historical"
        ]
        if _meets_rule(combined_metrics) and _meets_rule(policy_metrics):
            supporting.append(key)
    recommend = _meets_rule(combined) and len(supporting) >= 3
    return {
        "primary_combined_meets_rule": _meets_rule(combined),
        "primary_policy_meets_rule": _meets_rule(policy_all),
        "supporting_partitions": supporting,
        "supporting_partition_count": len(supporting),
        "recommend_later_replay_reweighting_experiment": recommend,
        "classification": "supports_separately_registered_replay_reweighting_experiment"
        if recommend
        else "audit_does_not_support_follow_up",
        "strength_claim": False,
    }


if __name__ == "__main__":
    main()
