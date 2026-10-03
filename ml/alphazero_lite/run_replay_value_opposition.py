"""Run the frozen full-gradient replay versus fresh-value diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import (
    run_fresh_historical_replay_gradient_alignment as reference,
)
from ml.alphazero_lite.policy_value_gradient_audit import parameter_groups
from ml.alphazero_lite.replay_value_opposition import geometry, verify_sums


ROOT = reference.ROOT
DECISION = (
    "Recommend a later value-target provenance/calibration audit only when seed455 "
    "primary >32 all-parameter cosine is <= -0.05 and this holds in at least "
    "three of four frozen partitions; undefined directions fail."
)
OBJECTIVES = ("policy", "weighted_value")
GROUPS = (
    "input_projection",
    "residual_block_0",
    "residual_block_1",
    "residual_block_2",
    "shared_trunk",
    "policy_head",
    "value_head",
    "all_parameters",
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_manifest() -> dict[str, Any]:
    registration, sources = reference.verify_inputs()
    training = registration["training"]
    source_rows = [
        {key: row[key] for key in ("name", "sha256", "weight", "value_target_mode")}
        for row in sources
    ]
    arrays = reference.load_compact(
        sources,
        int(training["seed"]),
        float(training["validation_split"]),
    )
    x, _policy, _value, _weights, replay, mapping, _rows = arrays
    partitions = _partition_ids(mapping)
    prior_manifest = json.loads(
        (
            ROOT
            / "docs/data/seed455-fresh-historical-replay-gradient-alignment-correction-manifest.json"
        ).read_text()
    )
    entry = {
        "schema": "replay_value_opposition_manifest_v1",
        "decision_rule": DECISION,
        "registration_sha256": _hash(reference.REGISTRATION),
        "recovery_receipt_sha256": _hash(reference.RECOVERY),
        "seed455_sha256": reference.SEED455_SHA,
        "original_o0_e4_sha256": _hash(reference.E4),
        "sources": source_rows,
        "split": {
            "seed": training["seed"],
            "validation_split": training["validation_split"],
            "implementation": "production split_replay_positions_by_source_row",
        },
        "partition": {
            "count": reference.PARTITIONS,
            "rule": "SHA256('fresh-historical-gradient-v1\\0'+source_name+'\\0'+source-local compact row id), first 8 bytes big-endian modulo 4; copies remain together",
            "assignments_sha256": hashlib.sha256(partitions.tobytes()).hexdigest(),
        },
        "loaded_compact_rows": len(x),
        "weighted_replay_positions": len(replay),
        "training_replay_mass": int(mapping.train_multiplicity.sum()),
        "source_split_accounting": prior_manifest["source_split_accounting"],
        "partition_source_accounting": prior_manifest["partition_source_accounting"],
        "objective": {
            "policy": "sum(m*q*CE)/global_sum(m*q)",
            "weighted_value": "0.3*sum(m*Huber)/global_sum(m)",
            "fresh_value": "fresh source weighted_value numerator/global_sum(m); phase cohort is not renormalized",
            "huber_delta": 1.0,
            "policy_target_mode": "sharpened",
            "optimizer_steps": 0,
        },
        "execution_source_sha256": {
            path: _hash(ROOT / path)
            for path in (
                "ml/alphazero_lite/run_replay_value_opposition.py",
                "ml/alphazero_lite/replay_value_opposition.py",
                "ml/alphazero_lite/run_fresh_historical_replay_gradient_alignment.py",
                "ml/alphazero_lite/replay_gradient_mapping.py",
                "ml/alphazero_lite/policy_value_gradient_audit.py",
                "ml/alphazero_lite/train.py",
                "ml/alphazero_lite/verify_replay_value_opposition.py",
                "ml/alphazero_lite/test_replay_value_opposition.py",
            )
        },
        "cohorts": list(reference.COHORTS),
        "chunk_size": reference.CHUNK,
    }
    return entry


def _group_indices(
    groups: dict[str, tuple[torch.nn.Parameter, ...]],
) -> dict[str, list[int]]:
    all_params = groups["all_parameters"]
    index = {id(parameter): i for i, parameter in enumerate(all_params)}
    return {
        name: [index[id(parameter)] for parameter in groups[name]] for name in GROUPS
    }


def _slice_vector(
    vector: torch.Tensor, sizes: list[int], indices: list[int]
) -> torch.Tensor:
    offsets = np.cumsum([0, *sizes])
    pieces = [vector[int(offsets[i]) : int(offsets[i + 1])] for i in indices]
    return torch.cat(pieces)


def _sizes(groups: dict[str, tuple[torch.nn.Parameter, ...]]) -> list[int]:
    return [parameter.numel() for parameter in groups["all_parameters"]]


def _gradient(
    model: torch.nn.Module,
    parameters: tuple[torch.nn.Parameter, ...],
    arrays: tuple[Any, ...],
    ids: np.ndarray,
    policy_denominator: float,
    value_denominator: float,
    objective: str,
) -> tuple[torch.Tensor, int]:
    # The corrected #403 primitive accepts an arbitrary parameter tuple and
    # explicitly turns unused gradients into zero coordinates.
    return reference._one_source_gradient(
        model,
        parameters,
        arrays[0],
        arrays[1],
        arrays[2],
        arrays[3],
        arrays[5].train_multiplicity,
        ids,
        policy_denominator,
        value_denominator,
        objective,
    )


def _partition_ids(mapping: Any) -> np.ndarray:
    return reference._partition_masks(mapping)


def _metrics_for_scope(
    components: dict[str, torch.Tensor],
    fresh: torch.Tensor,
    group_indices: dict[str, list[int]],
    sizes: list[int],
) -> dict[str, Any]:
    mixture = sum(components.values(), torch.zeros_like(fresh))
    verify_sums(components, mixture, fresh, atol=2e-5, rtol=2e-5)
    result: dict[str, Any] = {"groups": {}}
    for group in GROUPS:
        ix = group_indices[group]
        v = _slice_vector(fresh, sizes, ix)
        component_groups = {
            name: _slice_vector(vector, sizes, ix)
            for name, vector in components.items()
        }
        total = sum(component_groups.values(), torch.zeros_like(v))
        dots = {
            name: float(torch.dot(vector.double(), v.double()))
            for name, vector in component_groups.items()
        }
        metrics = geometry(total, v)
        metrics.update(
            {
                "source_objective_dot_contributions": dots,
                "contribution_dot_sum": sum(dots.values()),
                "component_norms": {
                    name: float(torch.linalg.vector_norm(vector.double()))
                    for name, vector in component_groups.items()
                },
                "cross_objective_dot_products": {
                    left: {
                        right: float(torch.dot(lvec.double(), rvec.double()))
                        for right, rvec in component_groups.items()
                    }
                    for left, lvec in component_groups.items()
                },
                "mixture_sum_verified": True,
            }
        )
        result["groups"][group] = metrics
    return result


def run_checkpoint(
    label: str,
    checkpoint: Path,
    expected_hash: str,
    arrays: tuple[Any, ...],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    x, p, v, q, _replay, mapping, _rows = arrays
    before = _hash(checkpoint)
    if before != expected_hash:
        raise RuntimeError(f"checkpoint_hash_mismatch:{label}")
    model = reference.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    reference.train.load_checkpoint_into_model(model, checkpoint)
    model.eval()
    frozen = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    groups = parameter_groups(model)
    layout = dict(groups)
    layout["shared_trunk"] = tuple(
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and name.startswith(("input_layer.", "residual_layers."))
    )
    layout["all_parameters"] = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    params = layout["all_parameters"]
    indices, sizes = _group_indices(layout), _sizes(layout)
    source_ids = mapping.compact_source_ids
    multiplicity = mapping.train_multiplicity
    train_ids = np.flatnonzero(multiplicity > 0)
    global_policy_den = float(np.dot(q[train_ids], multiplicity[train_ids]))
    global_value_den = float(multiplicity[train_ids].sum())
    source_vectors: dict[str, dict[str, torch.Tensor]] = {}
    missing: dict[str, dict[str, int]] = {}
    partition = _partition_ids(mapping)
    for source_idx, source in enumerate(sources):
        source_vectors[source["name"]], missing[source["name"]] = {}, {}
        for objective in OBJECTIVES:
            source_ids_selected = train_ids[source_ids[train_ids] == source_idx]
            vector, missing_count = _gradient(
                model,
                params,
                arrays,
                source_ids_selected,
                global_policy_den,
                global_value_den,
                objective,
            )
            source_vectors[source["name"]][objective] = vector
            missing[source["name"]][objective] = missing_count

    output: dict[str, Any] = {
        "checkpoint_sha256": before,
        "parameter_count": sum(p_.numel() for p_ in params),
        "source_gradient_missing_parameter_count": missing,
        "results": {},
    }
    cohorts = reference._state_cohorts(x, multiplicity)
    for cohort, cohort_mask in cohorts.items():
        output["results"][cohort] = {}
        for part in ("all", *(f"partition_{i}" for i in range(reference.PARTITIONS))):
            part_mask = (
                np.ones(len(x), dtype=bool)
                if part == "all"
                else partition == int(part.rsplit("_", 1)[1])
            )
            selected = cohort_mask & part_mask
            fresh_ids = np.flatnonzero(
                selected & (source_ids == 0) & (multiplicity > 0)
            )
            fresh_v, fresh_missing = _gradient(
                model,
                params,
                arrays,
                fresh_ids,
                global_policy_den,
                global_value_den,
                "weighted_value",
            )
            fresh_policy, fresh_policy_missing = _gradient(
                model,
                params,
                arrays,
                fresh_ids,
                global_policy_den,
                global_value_den,
                "policy",
            )
            components: dict[str, torch.Tensor] = {}
            # The mixture is the complete training subset; partitioned analyses
            # use mixture and fresh objective contributions from the same partition.
            for source in sources:
                for objective in OBJECTIVES:
                    key = f"{source['name']}:{objective}"
                    if part == "all":
                        vector = source_vectors[source["name"]][objective]
                    else:
                        source_idx = sources.index(source)
                        ids = train_ids[
                            (source_ids[train_ids] == source_idx) & part_mask[train_ids]
                        ]
                        vector, _ = _gradient(
                            model,
                            params,
                            arrays,
                            ids,
                            global_policy_den,
                            global_value_den,
                            objective,
                        )
                    components[key] = vector
            output["results"][cohort][part] = {
                "selected_fresh_rows": len(fresh_ids),
                "fresh_value_missing_parameter_count": fresh_missing,
                "fresh_policy_missing_parameter_count": fresh_policy_missing,
                **_metrics_for_scope(components, fresh_v, indices, sizes),
            }
            output["results"][cohort][part]["fresh_policy_alignment"] = {
                group: geometry(
                    _slice_vector(
                        sum(components.values(), torch.zeros_like(fresh_v)),
                        sizes,
                        indices[group],
                    ),
                    _slice_vector(fresh_policy, sizes, indices[group]),
                )
                for group in GROUPS
            }
    if any(
        not torch.equal(frozen[name], value)
        for name, value in model.state_dict().items()
    ):
        raise RuntimeError(f"model_state_changed:{label}")
    if _hash(checkpoint) != before:
        raise RuntimeError(f"checkpoint_changed:{label}")
    output["immutability"] = {
        "checkpoint_file_unchanged": True,
        "parameters_and_buffers_unchanged": True,
        "optimizer_steps": 0,
    }
    return output


def classify(seed455: dict[str, Any]) -> dict[str, Any]:
    def passes(entry: dict[str, Any]) -> bool:
        cosine = entry["groups"]["all_parameters"]["cosine"]
        return cosine is not None and cosine <= -0.05

    primary = seed455["results"][">32"]
    full = passes(primary["all"])
    partitions = [f"partition_{i}" for i in range(reference.PARTITIONS)]
    supporting = [name for name in partitions if passes(primary[name])]
    recommend = full and len(supporting) >= 3
    return {
        "primary_all_parameters_pass": full,
        "supporting_partitions": supporting,
        "supporting_partition_count": len(supporting),
        "recommend_value_target_provenance_calibration_audit": recommend,
        "classification": "recommend_later_value_target_provenance_calibration_audit"
        if recommend
        else "no_robust_net_opposition_signal",
        "strength_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, help="write frozen manifest and exit")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.freeze:
        args.freeze.write_text(
            json.dumps(freeze_manifest(), indent=2, sort_keys=True) + "\n"
        )
        return
    if not args.manifest or not args.out:
        parser.error("--manifest and --out are required for execution")
    manifest = json.loads(args.manifest.read_text())
    current = freeze_manifest()
    if current != manifest:
        raise RuntimeError("frozen_inputs_or_execution_sources_changed")
    registration, sources = reference.verify_inputs()
    training = registration["training"]
    arrays = reference.load_compact(
        sources, int(training["seed"]), float(training["validation_split"])
    )
    e4_entry = next(
        row
        for row in reference._json(reference.RECOVERY)["checkpoint_files"]
        if row["epoch"] == "E4"
    )
    result = {
        "schema": "replay_value_opposition_results_v1",
        "manifest_sha256": _hash(args.manifest),
        "manifest": manifest,
        "seed455": run_checkpoint(
            "seed455", reference.SEED455, reference.SEED455_SHA, arrays, sources
        ),
        "original_o0_e4_secondary": run_checkpoint(
            "original_o0_e4", reference.E4, e4_entry["expected_sha256"], arrays, sources
        ),
    }
    result["classification"] = classify(result["seed455"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
