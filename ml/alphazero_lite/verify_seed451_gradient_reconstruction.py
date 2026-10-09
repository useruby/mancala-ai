"""Independent frozen-input reconstruction for the seed451 gradient archive.

This completion is explicitly post-execution.  It never imports or invokes the
seed451 publisher's gradient routine and never writes publication evidence.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train
from ml.alphazero_lite.verify_seed434_census_source import reconstruct_sources
from ml.alphazero_lite.verify_seed450_seed449_correction import (
    _reconstruct_derivatives,
)

OUT = Path("docs/data/seed451-source-policy-alignment")
TOL = {"atol": 2e-5, "rtol": 2e-5}
SOURCES = (
    "fresh",
    "generic_bootstrap",
    "random_teacher",
    "opening_disagreement",
    "stability",
)
VECTOR_FIELDS = (
    "G_PF",
    "G_PH",
    "G_V",
    "G_P",
    "G_T",
    "U_F_exposure",
    "U_F_equal_exact_input",
    "U_H_exposure",
    "U_H_equal_exact_input",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _flat(
    grads: tuple[torch.Tensor | None, ...], params: tuple[torch.nn.Parameter, ...]
) -> np.ndarray:
    return np.concatenate(
        [
            np.zeros(p.numel(), dtype=np.float64)
            if g is None
            else g.detach().cpu().numpy().astype(np.float64).reshape(-1)
            for p, g in zip(params, grads, strict=True)
        ]
    )


def _grad(
    model: torch.nn.Module,
    params: tuple[torch.nn.Parameter, ...],
    x: np.ndarray,
    policy: np.ndarray,
    value: np.ndarray,
    compact: np.ndarray,
    weights: np.ndarray,
    kind: str,
    chunk: int,
    value_weights: np.ndarray | None = None,
) -> np.ndarray:
    result = np.zeros(sum(p.numel() for p in params), dtype=np.float64)
    for lo in range(0, len(compact), chunk):
        ix = compact[lo : lo + chunk]
        xb = torch.from_numpy(x[ix])
        logits, pred = model(xb)
        ce = None
        if kind in {"policy", "joint"}:
            legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[ix]))
            ce = train.compute_policy_cross_entropy(
                logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(policy[ix])
            )
        if kind == "policy":
            assert ce is not None
            objective = (
                ce * torch.from_numpy(weights[lo : lo + len(ix)].astype(np.float32))
            ).sum()
        elif kind == "value":
            huber = train.compute_value_loss_vector(
                pred, torch.from_numpy(value[ix]), value_loss="huber", huber_delta=1.0
            )
            objective = (
                huber * torch.from_numpy(weights[lo : lo + len(ix)].astype(np.float32))
            ).sum()
        elif kind == "joint":
            assert ce is not None
            huber = train.compute_value_loss_vector(
                pred, torch.from_numpy(value[ix]), value_loss="huber", huber_delta=1.0
            )
            if value_weights is None:
                raise ValueError("direct_joint_value_weights_missing")
            objective = (
                ce * torch.from_numpy(weights[lo : lo + len(ix)].astype(np.float32))
            ).sum() + 0.3 * (
                huber
                * torch.from_numpy(value_weights[lo : lo + len(ix)].astype(np.float32))
            ).sum()
        else:
            raise ValueError(f"unknown_gradient_kind:{kind}")
        result += _flat(
            torch.autograd.grad(objective, params, allow_unused=True), params
        )
    return result


def _identity_weights(identities: list[str]) -> np.ndarray:
    counts: dict[str, int] = {}
    for identity in identities:
        counts[identity] = counts.get(identity, 0) + 1
    return np.asarray(
        [1.0 / (len(counts) * counts[item]) for item in identities], dtype=np.float64
    )


def _expand_replay_rows(counts: list[int], weights: list[int]) -> np.ndarray:
    """Expand source-local compact row ranges without conflating copies and rows."""
    if (
        len(counts) != len(weights)
        or any(n < 0 for n in counts)
        or any(w < 1 for w in weights)
    ):
        raise ValueError("invalid_source_multiplicity_spec")
    parts = []
    offset = 0
    for count, weight in zip(counts, weights, strict=True):
        parts.append(np.tile(np.arange(offset, offset + count, dtype=np.int64), weight))
        offset += count
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def _close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, atol=TOL["atol"], rtol=TOL["rtol"]):
        raise ValueError(f"independent_metric_mismatch:{label}")


def _validate_source_mapping(
    ledger: list[dict[str, Any]],
    expected_sources: tuple[str, ...],
    expected_counts: list[int] | None = None,
) -> np.ndarray:
    if [row["compact_row"] for row in ledger] != list(range(len(ledger))):
        raise ValueError("compact_row_coverage_invalid")
    actual = np.asarray([row["source"] for row in ledger], dtype="U32")
    if expected_counts is not None:
        if len(expected_counts) != len(expected_sources) or sum(expected_counts) != len(
            ledger
        ):
            raise ValueError("source_mapping_count_mismatch")
        offset = 0
        for source, count in zip(expected_sources, expected_counts, strict=True):
            if not np.all(actual[offset : offset + count] == source):
                raise ValueError(f"source_order_mapping_invalid:{source}")
            offset += count
        return actual
    if set(actual.tolist()) - set(expected_sources):
        raise ValueError("unknown_source_mapping")
    offset = 0
    for source in expected_sources:
        count = int(np.count_nonzero(actual == source))
        if not np.all(actual[offset : offset + count] == source):
            raise ValueError(f"source_order_mapping_invalid:{source}")
        offset += count
    if offset != len(ledger):
        raise ValueError("source_mapping_coverage_invalid")
    return actual


def _validate_loader_materialization(
    x: np.ndarray,
    policy: np.ndarray,
    expected_policy: np.ndarray,
    value: np.ndarray,
    expected_value: np.ndarray,
    coefficients: np.ndarray,
    replay: np.ndarray,
    ledger: list[dict[str, Any]],
    source_registration: dict[str, Any],
    expected_source_counts: list[int] | None = None,
) -> np.ndarray:
    sources = _validate_source_mapping(ledger, SOURCES, expected_source_counts)
    expected_inputs = np.stack(
        [np.frombuffer(bytes.fromhex(row["input_hex"]), dtype="<f4") for row in ledger]
    )
    if not np.array_equal(x, expected_inputs):
        raise ValueError("loader_compact_input_mapping_mismatch")
    if not np.array_equal(policy, expected_policy):
        raise ValueError("loader_policy_target_mapping_mismatch")
    if not np.array_equal(value.reshape(-1), expected_value.reshape(-1)):
        raise ValueError("loader_value_target_mapping_mismatch")
    expected_coefficients = np.asarray(
        [row["policy_coefficient"] for row in ledger], dtype=np.float32
    )
    if not np.array_equal(coefficients, expected_coefficients):
        raise ValueError("loader_policy_coefficients_mismatch")
    counts = [int(np.count_nonzero(sources == source)) for source in SOURCES]
    weights = [int(item["weight"]) for item in source_registration["replays"]]
    if not np.array_equal(replay, _expand_replay_rows(counts, weights)):
        raise ValueError("loader_replay_multiplicity_or_source_order_mismatch")
    # The hash-checked lane-A derivative is the loader's row target source. This
    # comparison supplements production loader validation (including actual-mode
    # precedence and exact-root handling).
    return sources


def _validate_denominators(
    record: dict[str, Any], mass_f: float, mass_h: float, n: int
) -> None:
    total = mass_f + mass_h
    for key, expected in (
        ("rho_F", mass_f / total),
        ("rho_H", mass_h / total),
        ("policy_mass", total),
        ("policy_denominator", total),
        ("value_denominator", n),
        ("train_exposures", n),
    ):
        _close(float(record[key]), float(expected), key)


def _compare_gradient(name: str, stored: np.ndarray, rebuilt: np.ndarray) -> float:
    if stored.shape != rebuilt.shape or not np.isfinite(stored).all():
        raise ValueError(f"archived_vector_shape_or_finiteness:{name}")
    discrepancy = float(np.max(np.abs(stored.astype(np.float64) - rebuilt)))
    if not np.allclose(stored, rebuilt, atol=TOL["atol"], rtol=TOL["rtol"]):
        raise ValueError(f"independent_gradient_mismatch:{name}")
    return discrepancy


def _validate_geometry(
    unseen: np.ndarray, gradient: np.ndarray, metric: dict[str, Any], label: str
) -> None:
    dot = float(np.dot(unseen, gradient))
    norm_u, norm_g = float(np.linalg.norm(unseen)), float(np.linalg.norm(gradient))
    cosine = None if norm_u == 0 or norm_g == 0 else dot / (norm_u * norm_g)
    derivative = None if norm_g == 0 else -dot / norm_g
    _close(metric["dot"], dot, f"{label}/dot")
    _close(metric["norm"], norm_g, f"{label}/norm")
    for key, expected in (
        ("cosine", cosine),
        ("unit_direction_loss_derivative", derivative),
    ):
        actual = metric[key]
        if expected is None:
            if actual is not None:
                raise ValueError(f"zero_norm_geometry_mismatch:{label}/{key}")
        elif actual is None:
            raise ValueError(f"missing_geometry:{label}/{key}")
        else:
            _close(actual, expected, f"{label}/{key}")


def _validate_group_accounting(
    unseen: np.ndarray,
    fields: dict[str, np.ndarray],
    indices: list[int],
    rho_f: float,
    rho_h: float,
    published: dict[str, Any],
    label: str,
) -> None:
    ix = np.asarray(indices, dtype=np.int64)
    parts = {
        "fresh": rho_f * float(np.dot(unseen[ix], fields["G_PF"][ix])),
        "historical": rho_h * float(np.dot(unseen[ix], fields["G_PH"][ix])),
        "value": 0.3 * float(np.dot(unseen[ix], fields["G_V"][ix])),
        "joint": float(np.dot(unseen[ix], fields["G_T"][ix])),
    }
    for key, expected in parts.items():
        _close(published[key], expected, f"{label}/{key}")
    _close(
        sum(parts[k] for k in ("fresh", "historical", "value")),
        parts["joint"],
        f"{label}/reconciliation",
    )


def _validate_archive_names(names: list[str], expected: set[str]) -> None:
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("archived_gradient_coverage_invalid")


def _derive_classification(cosines: dict[str, dict[str, float | None]]) -> str:
    opposition = True
    positive = True
    for values in cosines.values():
        ph, pf, total = values["G_PH"], values["G_PF"], values["G_T"]
        opposition = (
            opposition
            and ph is not None
            and ph <= -0.10
            and pf is not None
            and pf >= 0.10
        )
        positive = positive and total is not None and total >= 0.10
    return (
        "historical_policy_opposition_present"
        if opposition
        else "fresh_first_order_alignment_positive"
        if positive
        else "mixed_or_checkpoint_dependent_alignment"
    )


def _validate_classification(
    cosines: dict[str, dict[str, float | None]], published: str
) -> str:
    derived = _derive_classification(cosines)
    if derived != published:
        raise ValueError("independently_reconstructed_classification_mismatch")
    return derived


def reconstruct(root: Path) -> dict[str, Any]:
    """Rebuild all input rows and gradients, then compare against archived vectors."""
    root = root.resolve()
    out = root / OUT
    registration = json.loads((out / "registration.json").read_text())
    results = json.loads((out / "results.json").read_text())
    publication_receipt = json.loads((out / "receipt.json").read_text())
    expected_publication_files = {
        "registration.json",
        "gradient-vectors.npz",
        "results.json",
        "results.md",
    }
    if set(publication_receipt["files_sha256"]) != expected_publication_files:
        raise ValueError("original_publication_receipt_coverage_invalid")
    for relative, expected in publication_receipt["files_sha256"].items():
        if _sha(out / relative) != expected:
            raise ValueError(f"original_publication_hash_mismatch:{relative}")
    # Bind historic execution, inputs, and checkpoint hashes exactly as originally frozen.
    for rel, digest in registration["source_sha256"].items():
        if _sha(root / rel) != digest:
            raise ValueError(f"frozen_source_hash_mismatch:{rel}")
    for rel, digest in registration["inputs_sha256"].items():
        if _sha(root / rel) != digest:
            raise ValueError(f"frozen_input_hash_mismatch:{rel}")
    source_reg_path = (
        root / "docs/data/seed416-policy-target-softening/registration-v3.json"
    )
    source_reg = json.loads(source_reg_path.read_text())
    if [item["name"] for item in source_reg["replays"]] != list(SOURCES):
        raise ValueError("source_order_mismatch")
    # Seed450 reconstructs and hash-checks actual lane-A derivatives beneath root;
    # production loader is then given only those local reconstructed files.
    (root / ".tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="seed451-verify-", dir=root / ".tmp"
    ) as scratch:
        paths = _reconstruct_derivatives(root, source_reg, Path(scratch))
        x, p, v, replay, coefficients = train.load_jsonl_replay(
            [paths[name] for name in SOURCES],
            [int(item["weight"]) for item in source_reg["replays"]],
            policy_target_mode="sharpened",
            value_target_mode="default",
            replay_value_target_modes=[
                item["value_target_mode"] for item in source_reg["replays"]
            ],
            include_policy_loss_weights=True,
        )
        derivative_rows = [
            json.loads(line)
            for source in SOURCES
            for line in paths[source].read_text(encoding="utf-8").splitlines()
        ]
    ledger = list(reconstruct_sources(root, source_reg))
    if len(ledger) != len(coefficients) or [r["compact_row"] for r in ledger] != list(
        range(len(ledger))
    ):
        raise ValueError("compact_row_coverage_invalid")
    expected_policy = np.stack(
        [np.asarray(row["target"], dtype=np.float32) for row in ledger]
    )
    expected_loader_policy = np.asarray(
        [row["policy"] for row in derivative_rows], dtype=np.float32
    )
    expected_loader_value = np.asarray(
        [row["value"] for row in derivative_rows], dtype=np.float32
    )
    if not np.array_equal(expected_policy, expected_loader_policy):
        raise ValueError("source_reconstruction_target_mismatch")
    compact_sources = _validate_loader_materialization(
        x,
        p,
        expected_loader_policy,
        v,
        expected_loader_value,
        coefficients,
        replay,
        ledger,
        source_reg,
        [int(source_reg["derivatives"][name]["A"]["rows"]) for name in SOURCES],
    )
    split_path = root / source_reg["training"]["source_row_split"]["path"]
    if _sha(split_path) != source_reg["training"]["source_row_split"]["sha256"]:
        raise ValueError("split_hash_mismatch")
    with gzip.open(split_path, "rt", encoding="utf-8") as stream:
        split = json.load(stream)
    train_positions = np.asarray(split["train_positions"], dtype=np.int64)
    valid_positions = np.asarray(split["validation_positions"], dtype=np.int64)
    if (
        hashlib.sha256(train_positions.astype("<i8").tobytes()).hexdigest()
        != source_reg["training"]["source_row_split"]["train_positions_sha256"]
    ):
        raise ValueError("training_position_hash_mismatch")
    expanded = len(replay)
    if (
        len(set(train_positions.tolist())) != len(train_positions)
        or len(set(valid_positions.tolist())) != len(valid_positions)
        or set(train_positions) & set(valid_positions)
        or set(train_positions) | set(valid_positions) != set(range(expanded))
    ):
        raise ValueError("frozen_split_coverage_invalid")
    train_compact = replay[train_positions]
    train_sources = compact_sources[train_compact]
    train_coeff = coefficients[train_compact].astype(np.float64)
    fresh = train_sources == "fresh"
    historical = ~fresh
    mass_f = float(train_coeff[fresh].sum())
    mass_h = float(train_coeff[historical].sum())
    mass = mass_f + mass_h
    rho_f, rho_h = mass_f / mass, mass_h / mass
    # Strict validation rows retain original order and exposure multiplicity.
    membership_path = (
        root
        / "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz"
    )
    unseen = []
    with gzip.open(membership_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["subset"] == "unseen" and row["active_stones"] > 32:
                unseen.append(row)
    if len(unseen) != 2607 or len({row["input_identity"] for row in unseen}) != 1242:
        raise ValueError("strict_unseen_population_invalid")
    unseen_compact = np.asarray(
        [int(row["compact_row"]) for row in unseen], dtype=np.int64
    )
    if np.any(unseen_compact < 0) or np.any(unseen_compact >= len(compact_sources)):
        raise ValueError("unseen_compact_row_out_of_range")
    for member, compact_row in zip(unseen, unseen_compact, strict=True):
        if member["input_identity"] != ledger[int(compact_row)]["input_hex"]:
            raise ValueError("unseen_ordered_membership_identity_mismatch")
    unseen_sources = compact_sources[unseen_compact]
    fresh_mask = unseen_sources == "fresh"
    cohorts = {
        "U_F": (
            unseen_compact[fresh_mask],
            [
                r["input_identity"]
                for r, yes in zip(unseen, fresh_mask, strict=True)
                if yes
            ],
        ),
        "U_H": (
            unseen_compact[~fresh_mask],
            [
                r["input_identity"]
                for r, yes in zip(unseen, fresh_mask, strict=True)
                if not yes
            ],
        ),
    }
    identity_sets = {name: set(ids) for name, (_, ids) in cohorts.items()}
    if len(identity_sets["U_F"] & identity_sets["U_H"]) != 3:
        raise ValueError("cross_source_identity_overlap_mismatch")
    # Enforce loader row arrays and source mapping reconstruction identities.
    parameter_layout = []
    group_indices: dict[str, list[int]] = {
        "shared_trunk": [],
        "policy_head": [],
        "value_head": [],
    }
    vectors_rebuilt: dict[str, dict[str, np.ndarray]] = {}
    errors: dict[str, float] = {}
    chunk = int(registration["chunk_size"])
    for checkpoint, spec in registration["checkpoints"].items():
        checkpoint_path = root / spec["path"]
        if _sha(checkpoint_path) != spec["sha256"]:
            raise ValueError(f"checkpoint_hash_mismatch:{checkpoint}")
        model = train.PolicyValueNet((96, 3), "residual_v3", 27)
        train.load_checkpoint_into_model(model, checkpoint_path)
        model.eval()
        named = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        group_indices = {"shared_trunk": [], "policy_head": [], "value_head": []}
        this_layout = []
        cursor = 0
        for name, param in named:
            group = (
                "shared_trunk"
                if name.startswith(("input_layer", "residual_layers"))
                else "policy_head"
                if name.startswith(("policy_hidden_layer", "policy_head"))
                else "value_head"
            )
            entry = {
                "name": name,
                "shape": list(param.shape),
                "start": cursor,
                "stop": cursor + param.numel(),
                "group": group,
            }
            this_layout.append(entry)
            group_indices[group].extend(range(cursor, cursor + param.numel()))
            cursor += param.numel()
        if parameter_layout and parameter_layout != this_layout:
            raise ValueError("checkpoint_parameter_layout_disagreement")
        parameter_layout = this_layout
        params = tuple(param for _, param in named)
        wp_f = train_coeff * fresh / mass_f
        wp_h = train_coeff * historical / mass_h
        wv = np.full(len(train_compact), 1.0 / len(train_compact), dtype=np.float64)
        wt = train_coeff / mass
        checkpoint_vectors = {
            "G_PF": _grad(model, params, x, p, v, train_compact, wp_f, "policy", chunk),
            "G_PH": _grad(model, params, x, p, v, train_compact, wp_h, "policy", chunk),
            "G_V": _grad(model, params, x, p, v, train_compact, wv, "value", chunk),
        }
        gp = rho_f * checkpoint_vectors["G_PF"] + rho_h * checkpoint_vectors["G_PH"]
        gt = gp + 0.3 * checkpoint_vectors["G_V"]
        direct_t = _grad(
            model, params, x, p, v, train_compact, wt, "joint", chunk, value_weights=wv
        )
        if not np.allclose(gt, direct_t, atol=TOL["atol"], rtol=TOL["rtol"]):
            raise ValueError(f"direct_joint_reconciliation_failed:{checkpoint}")
        checkpoint_vectors.update({"G_P": gp, "G_T": gt})
        for cohort_name, (rows, ids) in cohorts.items():
            for weighting in ("exposure", "equal_exact_input"):
                weights = (
                    np.full(len(rows), 1.0 / len(rows), dtype=np.float64)
                    if weighting == "exposure"
                    else _identity_weights(ids)
                )
                checkpoint_vectors[f"{cohort_name}_{weighting}"] = _grad(
                    model, params, x, p, v, rows, weights, "policy", chunk
                )
        vectors_rebuilt[checkpoint] = checkpoint_vectors
    if parameter_layout != registration["historical_seed438_parameter_layout"]:
        raise ValueError("registered_named_parameter_layout_mismatch")
    for group, indexes in group_indices.items():
        published = results["parameter_groups"][group]
        if published["indices"] != indexes or published["parameter_count"] != len(
            indexes
        ):
            raise ValueError(f"parameter_group_mapping_mismatch:{group}")
    source_accounting = {
        source: {
            "training_exposures": int(np.count_nonzero(train_sources == source)),
            "policy_coefficient_mass": float(
                train_coeff[train_sources == source].sum()
            ),
            "unique_compact_rows": int(
                len(np.unique(train_compact[train_sources == source]))
            ),
        }
        for source in SOURCES
    }
    if source_accounting != results["training_source_accounting"]:
        raise ValueError("training_source_accounting_mismatch")
    replay_accounting = {
        source: int(np.count_nonzero(compact_sources == source)) for source in SOURCES
    }
    if replay_accounting != results["replay_multiplicity_by_source"]:
        raise ValueError("compact_row_replay_accounting_mismatch")
    archive_path = out / "gradient-vectors.npz"
    expected_names = {
        f"{c}__{field}" for c in registration["checkpoints"] for field in VECTOR_FIELDS
    }
    with np.load(archive_path, allow_pickle=False) as archive:
        _validate_archive_names(archive.files, expected_names)
        for checkpoint, fields in vectors_rebuilt.items():
            checkpoint_result = results["checkpoints"][checkpoint]
            _validate_denominators(
                checkpoint_result, mass_f, mass_h, len(train_compact)
            )
            for field, rebuilt in fields.items():
                name = f"{checkpoint}__{field}"
                stored = np.asarray(archive[name])
                errors[name] = _compare_gradient(name, stored, rebuilt)
            for cohort in (
                "U_F_exposure",
                "U_F_equal_exact_input",
                "U_H_exposure",
                "U_H_equal_exact_input",
            ):
                u = fields[cohort]
                record = checkpoint_result["cohorts"][cohort]
                for objective in ("G_PF", "G_PH", "G_V", "G_T"):
                    _validate_geometry(
                        u,
                        fields[objective],
                        record["metrics"][objective],
                        f"{checkpoint}/{cohort}/{objective}",
                    )
                contribution = {
                    "fresh": rho_f * float(np.dot(u, fields["G_PF"])),
                    "historical": rho_h * float(np.dot(u, fields["G_PH"])),
                    "value": 0.3 * float(np.dot(u, fields["G_V"])),
                }
                joint_dot = float(np.dot(u, fields["G_T"]))
                for key, value_part in contribution.items():
                    _close(
                        record["signed_contributions"][key],
                        value_part,
                        f"{checkpoint}/{cohort}/contribution/{key}",
                    )
                _close(
                    sum(contribution.values()),
                    joint_dot,
                    f"{checkpoint}/{cohort}/contribution_reconciliation",
                )
                _close(
                    record["joint_dot"], joint_dot, f"{checkpoint}/{cohort}/joint_dot"
                )
                for group, indexes in group_indices.items():
                    ix = np.asarray(indexes, dtype=np.int64)
                    published_group = record["group_accounting"][group]
                    _validate_group_accounting(
                        u,
                        fields,
                        indexes,
                        rho_f,
                        rho_h,
                        published_group,
                        f"{checkpoint}/{cohort}/{group}",
                    )
                    if cohort in {"U_F_equal_exact_input", "U_H_equal_exact_input"}:
                        published_norm = results["parameter_groups"][group][
                            "norms_by_checkpoint"
                        ][checkpoint][cohort]
                        _close(
                            published_norm,
                            float(np.linalg.norm(u[ix])),
                            f"{checkpoint}/{cohort}/{group}/norm",
                        )
    # Recompute the fixed classification from reconstructed vectors, never from labels.
    cosines = {}
    for checkpoint, fields in vectors_rebuilt.items():
        u = fields["U_F_equal_exact_input"]
        cosines[checkpoint] = {}
        for objective in ("G_PF", "G_PH", "G_T"):
            g = fields[objective]
            denom = float(np.linalg.norm(u) * np.linalg.norm(g))
            cosines[checkpoint][objective] = (
                None if denom == 0 else float(np.dot(u, g) / denom)
            )
    classification = _validate_classification(cosines, results["classification"])
    return {
        "status": "valid",
        "classification": classification,
        "training": {
            "exposures": int(len(train_compact)),
            "fresh_mass": mass_f,
            "historical_mass": mass_h,
            "total_mass": mass,
            "rho_F": rho_f,
            "rho_H": rho_h,
        },
        "source_accounting": {
            source: {
                **source_accounting[source],
                "replay_copy_weight": int(source_reg["replays"][index]["weight"]),
                "value_target_mode": source_reg["replays"][index]["value_target_mode"],
            }
            for index, source in enumerate(SOURCES)
        },
        "unseen": {
            "U_F_exposures": int(len(cohorts["U_F"][0])),
            "U_F_identities": len(identity_sets["U_F"]),
            "U_H_exposures": int(len(cohorts["U_H"][0])),
            "U_H_identities": len(identity_sets["U_H"]),
            "cross_source_shared_identities": len(
                identity_sets["U_F"] & identity_sets["U_H"]
            ),
        },
        "reconstructed_cosines": cosines,
        "max_abs_vector_discrepancy": max(errors.values()),
        "vector_max_abs_discrepancies": errors,
        "chunk_size": chunk,
        "tolerance": TOL,
        "direct_joint_objective_reconciled": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
