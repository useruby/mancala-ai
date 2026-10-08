"""Frozen-checkpoint minibatch policy-denominator audit (seed444)."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train
from ml.alphazero_lite.verify_seed434_census_source import reconstruct_sources

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed444-minibatch-denominator-audit"
INIT_SHA = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
A_SHA = "afb164603b34d0449f463f3c8e475f9bb4408e7a704c031a6f000572e6c4e9c7"
BATCH = 512
VALUE_WEIGHT = 0.3
TOL = {"absolute": 2e-5, "relative": 2e-5}
SCREEN = 0.05
AUDIT_DIR = "docs/data/seed444-minibatch-denominator-audit"
CORRECTION = "correction-receipt.json"
CORRECTED_REGISTRATION = "corrected-rerun-registration-v2.json"
CORRECTED_EVIDENCE = "corrected-rerun-evidence.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_hash(value: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = _norm(a), _norm(b)
    return None if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


def _flatten_gradients(
    parameters: tuple[torch.nn.Parameter, ...],
    gradients: tuple[torch.Tensor | None, ...],
) -> tuple[np.ndarray, ...]:
    """Keep one correctly shaped float64 vector for every trainable parameter."""
    if len(parameters) != len(gradients):
        raise ValueError("gradient_parameter_count_mismatch")
    return tuple(
        np.zeros(tuple(parameter.shape), dtype=np.float64)
        if gradient is None
        else gradient.detach().cpu().numpy().astype(np.float64)
        for parameter, gradient in zip(parameters, gradients, strict=True)
    )


def _flat_vector(components: tuple[np.ndarray, ...]) -> np.ndarray:
    return np.concatenate([component.reshape(-1) for component in components])


def _load_frozen_inputs(
    root: Path, registration: dict[str, Any]
) -> tuple[np.ndarray, ...]:
    """Rebuild loader arrays from the bound compressed snapshots, never .tmp paths."""
    specs = registration["replays"]
    scratch_root = root / ".tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="seed444-sources-", dir=scratch_root
    ) as scratch:
        paths = []
        for spec in specs:
            source = (
                root
                / "docs/data/seed426-canonical-overlap/sources"
                / f"{spec['name']}.jsonl.gz"
            )
            target = Path(scratch) / f"{spec['name']}.jsonl"
            with gzip.open(source, "rb") as reader, target.open("wb") as writer:
                while chunk := reader.read(1024 * 1024):
                    writer.write(chunk)
            paths.append(target)
        x, policy, value, replay, coefficients = train.load_jsonl_replay(
            paths,
            [int(spec["weight"]) for spec in specs],
            policy_target_mode="sharpened",
            value_target_mode="default",
            replay_value_target_modes=[spec["value_target_mode"] for spec in specs],
            include_policy_loss_weights=True,
        )
    split_path = root / registration["training"]["source_row_split"]["path"]
    with gzip.open(split_path, "rt", encoding="utf-8") as stream:
        split = json.load(stream)
    return (
        x,
        policy,
        value,
        replay,
        coefficients,
        np.asarray(split["train_positions"], dtype=np.int64),
        np.asarray(split["validation_positions"], dtype=np.int64),
    )


def register(root: Path) -> dict[str, Any]:
    """Freeze inputs and arithmetic before any model gradient is evaluated."""
    root = root.resolve()
    protocol = json.loads(
        (
            root / "docs/data/seed438-seed437-gradient-correction/protocol.json"
        ).read_text()
    )
    freeze = root / "docs/data/seed416-policy-target-softening/training-freeze-v3"
    perm_path = freeze / "epoch-permutations.json.gz"
    with gzip.open(perm_path, "rt", encoding="utf-8") as stream:
        first = [int(i) for i in json.load(stream)[0]]
    layout = protocol["bound_inputs"]["model"]["flat_parameter_layout"]
    out = root / "docs/data/seed444-minibatch-denominator-audit"
    out.mkdir(parents=True, exist_ok=True)
    registration = {
        "schema": "seed444-frozen-minibatch-denominator-registration-v1",
        "status": "frozen_before_model_gradient_computation",
        "checkpoints": {
            "initializer": {
                "path": "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
                "sha256": INIT_SHA,
            },
            "adam_a16": {
                "path": "docs/data/seed435-adam-direction-screen/A-final.npz",
                "sha256": A_SHA,
            },
        },
        "source_registration": {
            "path": "docs/data/seed416-policy-target-softening/registration-v3.json",
            "sha256": sha(
                root / "docs/data/seed416-policy-target-softening/registration-v3.json"
            ),
        },
        "split": {
            "path": str((freeze / "source-row-split.json.gz").relative_to(root)),
            "sha256": sha(freeze / "source-row-split.json.gz"),
        },
        "permutation": {
            "path": str(perm_path.relative_to(root)),
            "sha256": sha(perm_path),
            "first_epoch_sha256": hashlib.sha256(
                np.asarray(first, dtype="<i8").tobytes()
            ).hexdigest(),
        },
        "sources": {
            item["snapshot"]: item["snapshot_sha256"]
            for item in protocol["bound_inputs"]["sources"]
        },
        "loader_source_sha256": sha(root / "ml/alphazero_lite/train.py"),
        "reconstruction_source_sha256": sha(
            root / "ml/alphazero_lite/verify_seed434_census_source.py"
        ),
        "execution_source_sha256": {
            name: sha(root / name)
            for name in (
                "ml/alphazero_lite/seed444_minibatch_denominator.py",
                "ml/alphazero_lite/verify_seed444_minibatch_denominator.py",
                "ml/alphazero_lite/test_seed444_minibatch_denominator.py",
            )
            if (root / name).is_file()
        },
        "batch_size": BATCH,
        "parameter_layout": layout,
        "tolerance": TOL,
        "decision": {
            "statistic": "||G_batch-G_global||_2/||G_global||_2",
            "degenerate_reference_norm_lte": 1e-12,
            "material_threshold_inclusive": SCREEN,
        },
        "objective": "policy weighted_policy_loss per batch; mean Huber(delta=1); value coefficient 0.3 applied once",
        "semantics": "original float32 row coefficients and replay multiplicity; complete training positions in frozen epoch order; final partial batch included; fixed parameters; no clipping or updates",
    }
    path = out / "registration.json"
    if path.exists():
        raise ValueError("registration_is_immutable")
    path.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n")
    return registration


def bind_corrected_rerun(root: Path) -> dict[str, Any]:
    """Bind corrected sources after the initial result, explicitly as a rerun."""
    root = root.resolve()
    out = root / AUDIT_DIR
    original_path = out / "registration.json"
    evidence_path = out / "evidence.json"
    if (
        sha(original_path)
        != "b915cc8175b75d6be83d1246103b93983d9307176f7cd5556ee89c8a9faade4a"
    ):
        raise ValueError("original_registration_bytes_changed")
    if (
        sha(evidence_path)
        != "6fcebc90d7b31bc7e9f4b857445e98dde31ce38f042dfb9b08b520199e06cc5a"
    ):
        raise ValueError("original_evidence_bytes_changed")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    sources = (
        "ml/alphazero_lite/seed444_minibatch_denominator.py",
        "ml/alphazero_lite/verify_seed444_minibatch_denominator.py",
        "ml/alphazero_lite/test_seed444_minibatch_denominator.py",
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/verify_seed434_census_source.py",
        "ml/alphazero_lite/seed438_gradient_correction.py",
    )
    binding = {
        "schema": "seed444-post-observation-corrected-rerun-registration-v1",
        "status": "post_observation_sources_bound_before_corrected_rerun",
        "observation_order": "initial evidence E=0 was observed before this binding; corrected results are a transparently post-observation rerun",
        "original_registration_sha256": sha(original_path),
        "original_evidence_sha256": sha(evidence_path),
        "original_registration_source_sha256": original["execution_source_sha256"],
        "corrected_source_sha256": {name: sha(root / name) for name in sources},
        "inputs": {
            key: value
            for key, value in original.items()
            if key
            in {
                "checkpoints",
                "source_registration",
                "split",
                "permutation",
                "sources",
                "loader_source_sha256",
                "reconstruction_source_sha256",
                "batch_size",
                "parameter_layout",
                "tolerance",
                "decision",
                "objective",
                "semantics",
            }
        },
        "gradient_fix": "torch.autograd.grad(..., allow_unused=True); preserve each trainable parameter's original shape and order, inserting a correctly shaped float64 zero for every unused parameter before flattening",
        "supersedes": "original evidence.json as an admissible preregistered measurement; the original file remains byte-for-byte preserved as historical initial evidence",
        "coefficient_identity_rule": "inspect every float32 expanded training-exposure coefficient; never infer coefficient constancy from N=Q",
    }
    path = out / CORRECTED_REGISTRATION
    correction_path = out / "correction-receipt-amendment-1.json"
    if path.exists() or correction_path.exists():
        raise ValueError("corrected_rerun_registration_is_immutable")
    prior_correction = json.loads((out / CORRECTION).read_text(encoding="utf-8"))
    amendment = {
        "schema": "seed444-correction-receipt-amendment-v1",
        "original_correction_receipt_sha256": sha(out / CORRECTION),
        "correction_occurred_after_initial_measurements_observed": True,
        "earlier_corrected_binding_sha256": sha(
            out / "corrected-rerun-registration.json"
        ),
        "earlier_corrected_binding_attempt": "input reconstruction stopped on an incompatible #438 result-key assertion before any model gradient was evaluated",
        "corrected_source_sha256": binding["corrected_source_sha256"],
        "preserved_original_registration_sha256": prior_correction[
            "original_registration_sha256"
        ],
        "preserved_original_evidence_sha256": prior_correction[
            "original_evidence_sha256"
        ],
        "execution_stage": "new exact source hashes bound before the first corrected model gradient; all corrected results remain post-observation",
    }
    correction_path.write_text(
        json.dumps(amendment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    path.write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return binding


def _compatible_unseen_vectors(
    root: Path, layout: list[dict[str, Any]]
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Reuse only #438 unseen gradients whose frozen inputs match seed444 exactly."""
    base = Path("docs/data/seed438-seed437-gradient-correction")
    protocol_path = root / base / "protocol.json"
    receipt_path = root / base / "correction-receipt.json"
    vectors_path = root / base / "corrected-gradient-vectors.npz"
    results_path = root / base / "corrected-results.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    bindings = protocol["bound_inputs"]
    expected_checkpoints = {
        "initializer": INIT_SHA,
        "adam_a16": A_SHA,
    }
    for name, expected in expected_checkpoints.items():
        if bindings["checkpoints"][name]["sha256"] != expected:
            raise ValueError(f"unseen_checkpoint_incompatible:{name}")
    if (
        bindings["registration"]["sha256"]
        != "4839ed63d6a48945ea11ea4995570f11118adbd65c027f49c3c5f04066bbd066"
    ):
        raise ValueError("unseen_registration_incompatible")
    if (
        bindings["split"]["sha256"]
        != "f7b69ab8fd6c19128a29f9ae9a11a7ac42a5d26630368ff74e906f29ddb6be44"
    ):
        raise ValueError("unseen_split_incompatible")
    if bindings["model"]["flat_parameter_layout"] != layout:
        raise ValueError("unseen_layout_incompatible")
    if (
        bindings["counts"]["training_exposures"] != 134502
        or bindings["counts"]["compact_rows"] != 87625
    ):
        raise ValueError("unseen_partition_count_incompatible")
    membership = bindings["membership"]
    membership_path = root / membership["path"]
    if sha(membership_path) != membership["sha256"]:
        raise ValueError("unseen_membership_binding_invalid")
    if (
        membership["selected_exposures"] != 2607
        or membership["exact_input_identities"] != 1242
    ):
        raise ValueError("unseen_weighting_population_incompatible")
    archive_hash = sha(vectors_path)
    if receipt["corrected_vectors_sha256"] != archive_hash:
        raise ValueError("corrected_438_archive_hash_mismatch")
    if sha(protocol_path) != receipt["frozen_protocol_sha256"]:
        raise ValueError("corrected_438_protocol_hash_mismatch")
    if sha(results_path) != receipt["corrected_results_sha256"]:
        raise ValueError("corrected_438_published_results_hash_mismatch")
    for relative, expected in receipt["corrected_sources"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"corrected_438_source_hash_mismatch:{relative}")
    for name, checkpoint in expected_checkpoints.items():
        key = "adam_a16" if name == "adam_a16" else "initializer"
        published = results["checkpoints"][key]["checkpoint_sha256"]
        if published != checkpoint:
            raise ValueError(f"corrected_438_checkpoint_evidence_mismatch:{name}")
    vectors: dict[str, np.ndarray] = {}
    with np.load(vectors_path, allow_pickle=False) as archive:
        for checkpoint in ("initializer", "adam_a16"):
            for weighting, key in (
                ("exposure_weighted", "U_exposure"),
                ("equal_input", "U_equal_input"),
            ):
                vector = np.asarray(archive[f"{checkpoint}__{key}"], dtype=np.float64)
                if vector.shape != (
                    sum(item["stop"] - item["start"] for item in layout),
                ):
                    raise ValueError(
                        f"corrected_438_vector_layout_mismatch:{checkpoint}:{key}"
                    )
                vectors[f"{checkpoint}_{weighting}_unseen_policy"] = vector
    source_hashes = {
        "protocol": sha(protocol_path),
        "correction_receipt": sha(receipt_path),
        "corrected_results": sha(results_path),
        "corrected_gradient_vectors": archive_hash,
    }
    return vectors, source_hashes


def _model(checkpoint: Path) -> torch.nn.Module:
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, checkpoint)
    model.eval()
    return model


def _gradients(
    model: torch.nn.Module,
    x: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
    coeff: np.ndarray,
    compact_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model.zero_grad(set_to_none=True)
    states = torch.from_numpy(x[compact_rows])
    logits, prediction = model(states)
    legal = torch.from_numpy(
        train.legal_mask_matrix_for_encoded_states(x[compact_rows])
    )
    ce = train.compute_policy_cross_entropy(
        logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(p[compact_rows])
    )
    weights = torch.from_numpy(coeff[compact_rows])
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if float(weights.sum()) == 0.0:
        policy_components = tuple(
            np.zeros(tuple(parameter.shape), dtype=np.float64)
            for parameter in parameters
        )
    else:
        policy = train.weighted_policy_loss(ce, weights)
        grads = torch.autograd.grad(
            policy, parameters, retain_graph=True, allow_unused=True
        )
        policy_components = _flatten_gradients(parameters, grads)
    value = train.compute_value_loss_vector(
        prediction,
        torch.from_numpy(v[compact_rows]),
        value_loss="huber",
        huber_delta=1.0,
    ).mean()
    grads = torch.autograd.grad(value, parameters, allow_unused=True)
    value_components = _flatten_gradients(parameters, grads)
    return _flat_vector(policy_components), _flat_vector(value_components)


def compute(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / "docs/data/seed444-minibatch-denominator-audit"
    corrected = json.loads((out / CORRECTED_REGISTRATION).read_text())
    registration = corrected["inputs"]
    for source, expected in corrected["corrected_source_sha256"].items():
        if sha(root / source) != expected:
            raise ValueError(f"corrected_source_binding_invalid:{source}")
    reg416 = json.loads(
        (root / registration["source_registration"]["path"]).read_text()
    )
    if (
        sha(root / registration["source_registration"]["path"])
        != registration["source_registration"]["sha256"]
    ):
        raise ValueError("source_registration_hash_mismatch")
    for relative, expected in registration["sources"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"source_snapshot_hash_mismatch:{relative}")
    reconstructed = list(reconstruct_sources(root, reg416))
    arrays = _load_frozen_inputs(root, reg416)
    x, p, v, replay, coefficients, train_positions = arrays[:6]
    with gzip.open(
        root / registration["split"]["path"], "rt", encoding="utf-8"
    ) as stream:
        split = json.load(stream)
    if not np.array_equal(
        np.asarray(split["train_positions"], dtype=np.int64), train_positions
    ):
        raise ValueError("training_partition_reconstruction_mismatch")
    with gzip.open(
        root / registration["permutation"]["path"], "rt", encoding="utf-8"
    ) as stream:
        order = np.asarray(json.load(stream)[0], dtype=np.int64)
    if (
        hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()
        != registration["permutation"]["first_epoch_sha256"]
    ):
        raise ValueError("registered_first_epoch_hash_mismatch")
    if len(order) != len(train_positions) or set(order.tolist()) != set(
        range(len(train_positions))
    ):
        raise ValueError("first_epoch_permutation_invalid")
    positions = train_positions[order]
    rows = replay[positions]
    expanded_coefficients = coefficients[rows]
    coefficient_values, coefficient_counts = np.unique(
        expanded_coefficients, return_counts=True
    )
    compact_values, compact_counts = np.unique(coefficients, return_counts=True)
    coefficient_report = {
        "dtype": str(coefficients.dtype),
        "N": int(len(expanded_coefficients)),
        "Q": float(expanded_coefficients.astype(np.float64).sum()),
        "positive_count": int(np.count_nonzero(expanded_coefficients > 0)),
        "zero_count": int(np.count_nonzero(expanded_coefficients == 0)),
        "minimum": float(expanded_coefficients.min()),
        "maximum": float(expanded_coefficients.max()),
        "expanded_float32_le_sha256": _array_hash(expanded_coefficients, "<f4"),
        "expanded_distribution": [
            {"float32_value": float(value), "exposures": int(count)}
            for value, count in zip(coefficient_values, coefficient_counts, strict=True)
        ],
        "compact_row_distribution": [
            {"float32_value": float(value), "compact_rows": int(count)}
            for value, count in zip(compact_values, compact_counts, strict=True)
        ],
        "compact_rows": int(len(coefficients)),
        "unique_training_compact_rows": int(len(np.unique(rows))),
        "exposure_copies_beyond_unique_rows": int(len(rows) - len(np.unique(rows))),
        "expanded_positions": int(len(rows)),
        "replay_copies_retained": True,
    }
    structural_identity = bool(np.all(expanded_coefficients == np.float32(1.0)))
    coefficient_report["all_training_coefficients_exactly_one"] = structural_identity
    coefficient_report["structural_identity"] = (
        "S_b=n_b for every batch; Q=N; S_b/Q=n_b/N; hence policy and total aggregate gradients are identical at every parameter state"
        if structural_identity
        else None
    )
    unseen_vectors, unseen_bindings = _compatible_unseen_vectors(
        root, registration["parameter_layout"]
    )
    coefficient_report["corrected_438_unseen_vector_reuse"] = unseen_bindings
    layout = registration["parameter_layout"]
    results: dict[str, Any] = {}
    for label, checkpoint in registration["checkpoints"].items():
        path = root / checkpoint["path"]
        if sha(path) != checkpoint["sha256"]:
            raise ValueError(f"checkpoint_hash_mismatch:{label}")
        model = _model(path)
        params = list(model.named_parameters())
        if [(n, list(p.shape)) for n, p in params] != [
            (item["name"], item["shape"]) for item in layout
        ]:
            raise ValueError("parameter_layout_mismatch")
        n_total, q_total = len(rows), float(coefficients[rows].astype(np.float64).sum())
        batch_records, pg_list, vg_list = [], [], []
        for start in range(0, n_total, BATCH):
            stop = min(start + BATCH, n_total)
            pos, compact = positions[start:stop], rows[start:stop]
            batch_coeff = coefficients[compact]
            pg, vg = _gradients(model, x, p, v, coefficients, compact)
            pg_list.append(pg)
            vg_list.append(vg)
            batch_records.append(
                {
                    "index": len(batch_records),
                    "n": len(compact),
                    "S": float(batch_coeff.astype(np.float64).sum()),
                    "positive_coefficients": int(np.count_nonzero(batch_coeff > 0)),
                    "zero_coefficients": int(np.count_nonzero(batch_coeff == 0)),
                    "sources": {
                        name: sum(
                            reconstructed[int(r)]["source"] == name for r in compact
                        )
                        for name in (
                            "fresh",
                            "generic_bootstrap",
                            "random_teacher",
                            "opening_disagreement",
                            "stability",
                        )
                    },
                    "active_stone_buckets": {
                        key: int(
                            np.count_nonzero(
                                [
                                    min(
                                        4,
                                        int(
                                            np.sum(x[int(r), :12], dtype=np.float64)
                                            * 48
                                        )
                                        // 12,
                                    )
                                    == key
                                    for r in compact
                                ]
                            )
                        )
                        for key in range(5)
                    },
                    "expanded_positions_sha256": _array_hash(pos, "<i8"),
                    "compact_rows_sha256": _array_hash(compact, "<i8"),
                }
            )
        n = np.asarray([r["n"] for r in batch_records], dtype=np.float64)
        s = np.asarray([r["S"] for r in batch_records], dtype=np.float64)
        pg = np.stack(pg_list)
        vg = np.stack(vg_list)
        policy_global = np.sum(pg * (s / q_total)[:, None], axis=0)
        policy_batch = np.sum(pg * (n / n_total)[:, None], axis=0)
        value_global = np.sum(vg * (n / n_total)[:, None], axis=0)
        global_total = policy_global + VALUE_WEIGHT * value_global
        batch_total = policy_batch + VALUE_WEIGHT * value_global
        # Direct full objective gradient, independently evaluated as one weighted objective.
        direct_p, direct_v = _gradients(model, x, p, v, coefficients, rows)
        direct = direct_p + VALUE_WEIGHT * direct_v
        discrepancy = _norm(direct - global_total)
        if discrepancy > TOL["absolute"] + TOL["relative"] * _norm(global_total):
            raise ValueError("direct_global_gradient_reconciliation_failed")
        # Equal-update mean differs from exposure mean only in the short final batch.
        equal_pg = pg.mean(axis=0)
        equal_total = equal_pg + VALUE_WEIGHT * vg.mean(axis=0)
        group_ranges: dict[str, list[tuple[int, int]]] = {}
        for item in layout:
            group_ranges.setdefault(item["group"], []).append(
                (item["start"], item["stop"])
            )

        def geometry(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
            diff = a - b
            return {
                "norm_a": _norm(a),
                "norm_b": _norm(b),
                "cosine": _cosine(a, b),
                "difference_norm": _norm(diff),
                "relative_difference": None
                if _norm(b) <= 1e-12
                else _norm(diff) / _norm(b),
                "difference_squared_by_group": {
                    group: float(
                        sum(
                            float(np.dot(diff[start:stop], diff[start:stop]))
                            for start, stop in ranges
                        )
                    )
                    for group, ranges in group_ranges.items()
                },
                "difference_squared_total": float(np.dot(diff, diff)),
            }

        exposure_policy = np.sum(pg * (n / n_total)[:, None], axis=0)
        exposure_value = value_global
        equal_weight_effect = equal_total - (
            exposure_policy + VALUE_WEIGHT * exposure_value
        )
        unseen_alignment: dict[str, Any] = {}
        for weighting in ("exposure_weighted", "equal_input"):
            unseen = unseen_vectors[f"{label}_{weighting}_unseen_policy"]
            unseen_norm = _norm(unseen)
            unseen_alignment[weighting] = {}
            for objective_name, objective_gradient in (
                ("global_total", global_total),
                ("batch_denominator_total", batch_total),
            ):
                objective_norm = _norm(objective_gradient)
                dot = float(np.dot(unseen, objective_gradient))
                unseen_alignment[weighting][objective_name] = {
                    "dot_product": dot,
                    "unseen_gradient_norm": unseen_norm,
                    "objective_gradient_norm": objective_norm,
                    "cosine": _cosine(unseen, objective_gradient),
                    "unit_descent_alignment": None
                    if objective_norm <= 1e-12
                    else -dot / objective_norm,
                }
        results[label] = {
            "N": n_total,
            "Q": q_total,
            "batches": batch_records,
            "denominator_ratio_positive_batches": [
                float(ni * q_total / (n_total * si))
                for ni, si in zip(n, s, strict=True)
                if si > 0
            ],
            "policy_global_vs_batch": geometry(policy_batch, policy_global),
            "total_batch_vs_global": geometry(batch_total, global_total),
            "global_norm": _norm(global_total),
            "batch_norm": _norm(batch_total),
            "E": None
            if _norm(global_total) <= 1e-12
            else _norm(batch_total - global_total) / _norm(global_total),
            "direct_reconciliation_l2": discrepancy,
            "equal_update_policy": equal_pg.tolist(),
            "equal_update_total": equal_total.tolist(),
            "equal_update_final_partial_effect": {
                "vector_norm": _norm(equal_weight_effect),
                "relative_to_exposure_total": None
                if _norm(exposure_policy + VALUE_WEIGHT * exposure_value) <= 1e-12
                else _norm(equal_weight_effect)
                / _norm(exposure_policy + VALUE_WEIGHT * exposure_value),
            },
            "unseen_policy_alignment_descriptive_only": unseen_alignment,
            "global_policy": policy_global.tolist(),
            "batch_policy": policy_batch.tolist(),
            "value_gradient": value_global.tolist(),
            "global_total": global_total.tolist(),
            "batch_total": batch_total.tolist(),
        }
    norms = [result["global_norm"] for result in results.values()]
    if any(value <= 1e-12 for value in norms):
        classification = "degenerate_reference_gradient"
    elif any(result["E"] >= SCREEN for result in results.values()):
        classification = "material_minibatch_denominator_mismatch"
    else:
        classification = "denominator_mismatch_below_screen_threshold"
    vector_fields = {
        "global_policy",
        "batch_policy",
        "value_gradient",
        "global_total",
        "batch_total",
        "equal_update_policy",
        "equal_update_total",
    }
    vectors = {
        f"{name}_{key}": np.asarray(value, dtype=np.float64)
        for name, result in results.items()
        for key, value in result.items()
        if key in vector_fields
    }
    public_results = {
        name: {
            key: value
            for key, value in result.items()
            if key not in vector_fields | {"batches"}
        }
        for name, result in results.items()
    }
    evidence = {
        "schema": "seed444-minibatch-denominator-evidence-v1",
        "interpretation": "post-observation corrected rerun; fixed-checkpoint unclipped gradients only; not reconstructed optimizer updates or expectation over random permutations",
        "registration_sha256": sha(out / "registration.json"),
        "corrected_rerun_registration_sha256": sha(out / CORRECTED_REGISTRATION),
        "checkpoints": public_results,
        "coefficient_audit": coefficient_report,
        "structural_denominator_identity": structural_identity,
        "decision": {
            "classification": classification,
            "original_classification_retained": "denominator_mismatch_below_screen_threshold",
            "structural_no_op_finding": structural_identity,
        },
    }
    for label in results:
        for weighting in ("exposure_weighted", "equal_input"):
            vectors[f"{label}_{weighting}_unseen_policy"] = unseen_vectors[
                f"{label}_{weighting}_unseen_policy"
            ]
    np.savez_compressed(out / "corrected-aggregate-vectors.npz", **vectors)
    (out / "corrected-batch-ledger.json").write_text(
        json.dumps(
            {name: result["batches"] for name, result in results.items()},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (out / "corrected-coefficient-audit.json").write_text(
        json.dumps(coefficient_report, indent=2, sort_keys=True) + "\n"
    )
    (out / CORRECTED_EVIDENCE).write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--bind-corrected-rerun", action="store_true")
    parser.add_argument("--corrected-rerun", action="store_true")
    args = parser.parse_args()
    if args.bind_corrected_rerun:
        result = bind_corrected_rerun(args.root)
    elif args.corrected_rerun:
        result = compute(args.root)
    elif args.register:
        result = register(args.root)
    else:
        raise SystemExit(
            "select --register, --bind-corrected-rerun, or --corrected-rerun"
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
