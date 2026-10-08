"""Portable read-only semantic verifier for the corrected seed444 rerun."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train
from ml.alphazero_lite.seed444_minibatch_denominator import (
    A_SHA,
    AUDIT_DIR,
    BATCH,
    CORRECTED_EVIDENCE,
    CORRECTED_REGISTRATION,
    INIT_SHA,
    SCREEN,
    TOL,
    VALUE_WEIGHT,
    _compatible_unseen_vectors,
    _flatten_gradients,
    _flat_vector,
    _load_frozen_inputs,
    _norm,
    sha,
)
from ml.alphazero_lite.verify_seed434_census_source import reconstruct_sources

ORIGINAL_REGISTRATION_SHA = (
    "b915cc8175b75d6be83d1246103b93983d9307176f7cd5556ee89c8a9faade4a"
)
ORIGINAL_EVIDENCE_SHA = (
    "6fcebc90d7b31bc7e9f4b857445e98dde31ce38f042dfb9b08b520199e06cc5a"
)


def _same_hash(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}_mismatch")


def _close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(
        actual, expected, rel_tol=TOL["relative"], abs_tol=TOL["absolute"]
    ):
        raise ValueError(f"{label}_mismatch")


def _decision(norms: list[float], errors: list[float | None]) -> str:
    if any(norm <= 1e-12 for norm in norms):
        return "degenerate_reference_gradient"
    if any(error is not None and error >= SCREEN for error in errors):
        return "material_minibatch_denominator_mismatch"
    return "denominator_mismatch_below_screen_threshold"


def _validate_coefficients(actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.dtype != np.float32 or not np.array_equal(actual, expected):
        raise ValueError("coefficient_semantics_mismatch")


def _validate_split_and_order(
    split_positions: np.ndarray, positions: np.ndarray, order: np.ndarray
) -> None:
    if not np.array_equal(split_positions, positions):
        raise ValueError("split_semantics_mismatch")
    if len(order) != len(positions) or not np.array_equal(
        np.sort(order), np.arange(len(positions))
    ):
        raise ValueError("permutation_semantics_mismatch")


def _validate_denominators(
    n: np.ndarray, sums: np.ndarray, expected_ratios: list[float]
) -> None:
    N, Q = float(n.sum()), float(sums.sum())
    actual = [
        float(ni * Q / (N * si)) for ni, si in zip(n, sums, strict=True) if si > 0
    ]
    if not np.allclose(actual, expected_ratios, atol=1e-12, rtol=1e-12):
        raise ValueError("denominator_ratios_mismatch")


def _validate_layout(
    named_shapes: list[tuple[str, list[int]]], layout: list[dict[str, Any]]
) -> None:
    expected = [(item["name"], item["shape"]) for item in layout]
    if named_shapes != expected:
        raise ValueError("registered_layout_mismatch")


def _validate_component_vectors(
    global_policy: np.ndarray,
    batch_policy: np.ndarray,
    value_gradient: np.ndarray,
    global_total: np.ndarray,
    batch_total: np.ndarray,
) -> None:
    if not np.allclose(
        global_total,
        global_policy + VALUE_WEIGHT * value_gradient,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError("global_component_vector_mismatch")
    if not np.allclose(
        batch_total,
        batch_policy + VALUE_WEIGHT * value_gradient,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError("batch_component_vector_mismatch")


def _validate_decision_field(evidence: dict[str, Any], expected: str) -> None:
    if evidence.get("decision", {}).get("classification") != expected:
        raise ValueError("corrected_decision_mismatch")


def _load_model(checkpoint: Path) -> torch.nn.Module:
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, checkpoint)
    model.eval()
    return model


def _gradient(
    model: torch.nn.Module,
    x: np.ndarray,
    policy: np.ndarray,
    value: np.ndarray,
    coefficients: np.ndarray,
    compact_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    params = tuple(p for p in model.parameters() if p.requires_grad)
    states = torch.from_numpy(x[compact_rows])
    logits, prediction = model(states)
    legal = torch.from_numpy(
        train.legal_mask_matrix_for_encoded_states(x[compact_rows])
    )
    ce = train.compute_policy_cross_entropy(
        logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(policy[compact_rows])
    )
    batch_weights = torch.from_numpy(coefficients[compact_rows])
    if float(batch_weights.sum()) == 0.0:
        policy_components = tuple(
            np.zeros(tuple(p.shape), dtype=np.float64) for p in params
        )
    else:
        loss = train.weighted_policy_loss(ce, batch_weights)
        gradients = torch.autograd.grad(
            loss, params, retain_graph=True, allow_unused=True
        )
        policy_components = _flatten_gradients(params, gradients)
    value_loss = train.compute_value_loss_vector(
        prediction,
        torch.from_numpy(value[compact_rows]),
        value_loss="huber",
        huber_delta=1.0,
    ).mean()
    gradients = torch.autograd.grad(value_loss, params, allow_unused=True)
    value_components = _flatten_gradients(params, gradients)
    return _flat_vector(policy_components), _flat_vector(value_components)


def _verify_batch_record(
    batch: dict[str, Any],
    index: int,
    positions: np.ndarray,
    rows: np.ndarray,
    coefficients: np.ndarray,
    reconstructed: list[dict[str, Any]],
    x: np.ndarray,
) -> None:
    coeff = coefficients[rows]
    source_names = (
        "fresh",
        "generic_bootstrap",
        "random_teacher",
        "opening_disagreement",
        "stability",
    )
    source_counts = {
        source: sum(reconstructed[int(row)]["source"] == source for row in rows)
        for source in source_names
    }
    bucket_counts = {
        str(bucket): sum(
            min(4, int(np.sum(x[int(row), :12], dtype=np.float64) * 48) // 12) == bucket
            for row in rows
        )
        for bucket in range(5)
    }
    checks = {
        "index": index,
        "n": len(rows),
        "S": float(coeff.astype(np.float64).sum()),
        "positive_coefficients": int(np.count_nonzero(coeff > 0)),
        "zero_coefficients": int(np.count_nonzero(coeff == 0)),
        "expanded_positions_sha256": hashlib.sha256(
            np.asarray(positions, dtype="<i8").tobytes()
        ).hexdigest(),
        "compact_rows_sha256": hashlib.sha256(
            np.asarray(rows, dtype="<i8").tobytes()
        ).hexdigest(),
        "sources": source_counts,
        "active_stone_buckets": bucket_counts,
    }
    for field, expected in checks.items():
        if batch.get(field) != expected:
            raise ValueError(f"batch_{field}_mismatch:{index}")


def _verify_group_geometry(
    record: dict[str, Any],
    difference: np.ndarray,
    layout: list[dict[str, Any]],
    label: str,
) -> None:
    expected: dict[str, float] = {}
    for item in layout:
        start, stop, group = item["start"], item["stop"], item["group"]
        expected[group] = expected.get(group, 0.0) + float(
            np.dot(difference[start:stop], difference[start:stop])
        )
    actual = record["difference_squared_by_group"]
    if set(actual) != set(expected):
        raise ValueError(f"{label}_group_coverage_mismatch")
    for group in expected:
        _close(actual[group], expected[group], f"{label}_{group}_squared_norm")
    _close(
        record["difference_squared_total"],
        _norm(difference) ** 2,
        f"{label}_total_squared_norm",
    )


def _verify_receipt(root: Path, out: Path) -> None:
    receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
    for relative, expected in receipt["artifact_sha256"].items():
        _same_hash(sha(root / relative), expected, f"artifact_hash:{relative}")
    for relative, expected in receipt["source_sha256"].items():
        _same_hash(sha(root / relative), expected, f"publication_source:{relative}")
    _same_hash(
        sha(out / CORRECTED_EVIDENCE),
        receipt["corrected_evidence_sha256"],
        "receipt_evidence",
    )


def verify(root: Path) -> dict[str, Any]:
    """Reconstruct inputs, re-run all fixed gradients, and validate published geometry."""
    root = root.resolve()
    out = root / AUDIT_DIR
    original_path, original_evidence_path = (
        out / "registration.json",
        out / "evidence.json",
    )
    _same_hash(
        sha(original_path),
        ORIGINAL_REGISTRATION_SHA,
        "original_registration_immutability",
    )
    _same_hash(
        sha(original_evidence_path),
        ORIGINAL_EVIDENCE_SHA,
        "original_evidence_immutability",
    )
    corrected = json.loads((out / CORRECTED_REGISTRATION).read_text(encoding="utf-8"))
    evidence = json.loads((out / CORRECTED_EVIDENCE).read_text(encoding="utf-8"))
    receipt = json.loads((out / "correction-receipt.json").read_text(encoding="utf-8"))
    if corrected["status"] != "post_observation_sources_bound_before_corrected_rerun":
        raise ValueError("corrected_rerun_temporal_status_invalid")
    if receipt["correction_occurred_after_initial_measurements_observed"] is not True:
        raise ValueError("correction_history_missing")
    _same_hash(
        sha(original_path),
        corrected["original_registration_sha256"],
        "corrected_binding_original_registration",
    )
    _same_hash(
        sha(original_evidence_path),
        corrected["original_evidence_sha256"],
        "corrected_binding_original_evidence",
    )
    _same_hash(
        sha(out / CORRECTED_REGISTRATION),
        evidence["corrected_rerun_registration_sha256"],
        "corrected_rerun_registration",
    )
    _same_hash(
        sha(original_path),
        evidence["registration_sha256"],
        "original_registration_evidence_binding",
    )
    if evidence["interpretation"].find("post-observation") < 0:
        raise ValueError("post_observation_label_missing")
    for source, expected in corrected["corrected_source_sha256"].items():
        _same_hash(sha(root / source), expected, f"corrected_source:{source}")
    inputs = corrected["inputs"]
    for relative, expected in inputs["sources"].items():
        _same_hash(sha(root / relative), expected, f"snapshot:{relative}")
    for checkpoint_name, binding in inputs["checkpoints"].items():
        expected = INIT_SHA if checkpoint_name == "initializer" else A_SHA
        _same_hash(
            binding["sha256"],
            expected,
            f"authoritative_checkpoint_binding:{checkpoint_name}",
        )
        _same_hash(
            sha(root / binding["path"]),
            expected,
            f"authoritative_checkpoint:{checkpoint_name}",
        )
    _same_hash(
        sha(root / inputs["split"]["path"]), inputs["split"]["sha256"], "split_hash"
    )
    _same_hash(
        sha(root / inputs["permutation"]["path"]),
        inputs["permutation"]["sha256"],
        "permutation_hash",
    )
    source_registration = json.loads(
        (root / inputs["source_registration"]["path"]).read_text(encoding="utf-8")
    )
    reconstructed = list(reconstruct_sources(root, source_registration))
    arrays = _load_frozen_inputs(root, source_registration)
    x, policy, value, replay, coefficients, train_positions = arrays[:6]
    with gzip.open(root / inputs["split"]["path"], "rt", encoding="utf-8") as stream:
        split = json.load(stream)
    with gzip.open(
        root / inputs["permutation"]["path"], "rt", encoding="utf-8"
    ) as stream:
        order = np.asarray(json.load(stream)[0], dtype=np.int64)
    _validate_split_and_order(
        np.asarray(split["train_positions"], dtype=np.int64), train_positions, order
    )
    positions = train_positions[order]
    rows = replay[positions]
    coeff = coefficients[rows]
    _validate_coefficients(coeff, np.asarray(coeff, dtype=np.float32))
    coefficient_report = evidence["coefficient_audit"]
    expanded_hash = hashlib.sha256(np.asarray(coeff, dtype="<f4").tobytes()).hexdigest()
    if coefficient_report["expanded_float32_le_sha256"] != expanded_hash:
        raise ValueError("expanded_coefficients_hash_mismatch")
    if coefficient_report["N"] != len(rows) or coefficient_report["Q"] != float(
        coeff.astype(np.float64).sum()
    ):
        raise ValueError("coefficient_N_Q_mismatch")
    if coefficient_report["positive_count"] != int(
        np.count_nonzero(coeff > 0)
    ) or coefficient_report["zero_count"] != int(np.count_nonzero(coeff == 0)):
        raise ValueError("coefficient_counts_mismatch")
    if coefficient_report["minimum"] != float(coeff.min()) or coefficient_report[
        "maximum"
    ] != float(coeff.max()):
        raise ValueError("coefficient_extrema_mismatch")
    unique, counts = np.unique(coeff, return_counts=True)
    distribution = [
        {"float32_value": float(v), "exposures": int(c)}
        for v, c in zip(unique, counts, strict=True)
    ]
    if coefficient_report["expanded_distribution"] != distribution:
        raise ValueError("coefficient_distribution_mismatch")
    structural_identity = bool(np.all(coeff == np.float32(1.0)))
    if evidence["structural_denominator_identity"] is not structural_identity:
        raise ValueError("structural_identity_flag_mismatch")
    unseen_vectors, unseen_bindings = _compatible_unseen_vectors(
        root, inputs["parameter_layout"]
    )
    if coefficient_report["corrected_438_unseen_vector_reuse"] != unseen_bindings:
        raise ValueError("unseen_reuse_receipt_mismatch")
    vectors_path = out / "corrected-aggregate-vectors.npz"
    vectors_hash = sha(vectors_path)
    with np.load(vectors_path, allow_pickle=False) as archive:
        vectors = {
            key: np.asarray(archive[key], dtype=np.float64) for key in archive.files
        }
    ledger = json.loads(
        (out / "corrected-batch-ledger.json").read_text(encoding="utf-8")
    )
    measured_norms: list[float] = []
    measured_errors: list[float | None] = []
    for checkpoint_name, checkpoint in inputs["checkpoints"].items():
        model = _load_model(root / checkpoint["path"])
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        layout = inputs["parameter_layout"]
        _validate_layout(
            [(name, list(parameter.shape)) for name, parameter in named], layout
        )
        batches = ledger[checkpoint_name]
        if len(batches) != (len(rows) + BATCH - 1) // BATCH:
            raise ValueError("batch_coverage_mismatch")
        pg_values, vg_values, sizes, sums = [], [], [], []
        for index, batch in enumerate(batches):
            start, stop = index * BATCH, min((index + 1) * BATCH, len(rows))
            compact = rows[start:stop]
            _verify_batch_record(
                batch,
                index,
                positions[start:stop],
                compact,
                coefficients,
                reconstructed,
                x,
            )
            pg, vg = _gradient(model, x, policy, value, coefficients, compact)
            pg_values.append(pg)
            vg_values.append(vg)
            sizes.append(len(compact))
            sums.append(batch["S"])
        n, s = np.asarray(sizes, dtype=np.float64), np.asarray(sums, dtype=np.float64)
        pg_batches, vg_batches = np.stack(pg_values), np.stack(vg_values)
        N, Q = float(n.sum()), float(s.sum())
        global_policy = np.sum(pg_batches * (s / Q)[:, None], axis=0)
        batch_policy = np.sum(pg_batches * (n / N)[:, None], axis=0)
        global_value = np.sum(vg_batches * (n / N)[:, None], axis=0)
        global_total = global_policy + VALUE_WEIGHT * global_value
        batch_total = batch_policy + VALUE_WEIGHT * global_value
        direct_p, direct_v = _gradient(model, x, policy, value, coefficients, rows)
        direct = direct_p + VALUE_WEIGHT * direct_v
        if _norm(direct - global_total) > TOL["absolute"] + TOL["relative"] * _norm(
            global_total
        ):
            raise ValueError("direct_global_objective_reconciliation_failed")
        expected_vectors = {
            "global_policy": global_policy,
            "batch_policy": batch_policy,
            "value_gradient": global_value,
            "global_total": global_total,
            "batch_total": batch_total,
            "equal_update_policy": pg_batches.mean(axis=0),
            "equal_update_total": pg_batches.mean(axis=0)
            + VALUE_WEIGHT * vg_batches.mean(axis=0),
        }
        for key, expected_vector in expected_vectors.items():
            actual = vectors[f"{checkpoint_name}_{key}"]
            if not np.allclose(
                actual, expected_vector, atol=TOL["absolute"], rtol=TOL["relative"]
            ):
                raise ValueError(
                    f"aggregate_vector_recompute_mismatch:{checkpoint_name}:{key}"
                )
        _validate_component_vectors(
            vectors[f"{checkpoint_name}_global_policy"],
            vectors[f"{checkpoint_name}_batch_policy"],
            vectors[f"{checkpoint_name}_value_gradient"],
            vectors[f"{checkpoint_name}_global_total"],
            vectors[f"{checkpoint_name}_batch_total"],
        )
        record = evidence["checkpoints"][checkpoint_name]
        E = (
            _norm(batch_total - global_total) / _norm(global_total)
            if _norm(global_total) > 1e-12
            else None
        )
        if E is None:
            if record["E"] is not None:
                raise ValueError(f"primary_statistic:{checkpoint_name}_mismatch")
        else:
            _close(record["E"], E, f"primary_statistic:{checkpoint_name}")
        _close(
            record["direct_reconciliation_l2"],
            _norm(direct - global_total),
            f"direct_reconciliation:{checkpoint_name}",
        )
        for geom_name, a, b in (
            ("policy_global_vs_batch", batch_policy, global_policy),
            ("total_batch_vs_global", batch_total, global_total),
        ):
            geom = record[geom_name]
            _close(geom["norm_a"], _norm(a), f"{geom_name}_norm_a")
            _close(geom["norm_b"], _norm(b), f"{geom_name}_norm_b")
            _close(geom["difference_norm"], _norm(a - b), f"{geom_name}_difference")
            _verify_group_geometry(
                geom, a - b, layout, f"{checkpoint_name}_{geom_name}"
            )
        _validate_denominators(n, s, record["denominator_ratio_positive_batches"])
        measured_norms.append(_norm(global_total))
        measured_errors.append(E)
        # Check all unseen dot products against the reused, hash-bound corrected vectors.
        for weighting in ("exposure_weighted", "equal_input"):
            unseen_key = f"{checkpoint_name}_{weighting}_unseen_policy"
            if not np.array_equal(vectors[unseen_key], unseen_vectors[unseen_key]):
                raise ValueError(f"unseen_vector_mismatch:{unseen_key}")
            unseen = vectors[unseen_key]
            for objective_name, objective in (
                ("global_total", global_total),
                ("batch_denominator_total", batch_total),
            ):
                metrics = record["unseen_policy_alignment_descriptive_only"][weighting][
                    objective_name
                ]
                dot = float(np.dot(unseen, objective))
                _close(metrics["dot_product"], dot, "unseen_dot")
                _close(
                    metrics["unit_descent_alignment"],
                    -dot / _norm(objective),
                    "unseen_unit_descent",
                )
    classification = _decision(measured_norms, measured_errors)
    _validate_decision_field(evidence, classification)
    if (
        evidence["decision"]["original_classification_retained"]
        != "denominator_mismatch_below_screen_threshold"
    ):
        raise ValueError("original_classification_not_retained")
    _verify_receipt(root, out)
    return {
        "status": "valid",
        "classification": classification,
        "checkpoints": dict(zip(inputs["checkpoints"], measured_errors, strict=True)),
        "read_only": True,
        "corrected_vectors_sha256": vectors_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
