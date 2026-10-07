"""Independent, read-only verification of supplemental seed435 evidence."""

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

from ml.alphazero_lite.seed435_adam_direction import (
    INIT_SHA,
    RECEIPTS,
    decide,
    sha,
)
from ml.alphazero_lite.train import (
    PolicyValueNet,
    compute_policy_cross_entropy,
    compute_value_loss_vector,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
    policy_loss_weight_for_row,
)
from ml.alphazero_lite.seed429_policy_normalization import (
    canonical_identity_from_encoded_state,
)

TOLERANCE = {"absolute": 2e-6, "relative": 2e-6, "prediction_absolute": 1e-7}
SOURCE_DIR = "docs/data/seed426-canonical-overlap/sources"
FROZEN_HASHES = {
    "docs/data/seed416-policy-target-softening/registration-v3.json": "4839ed63d6a48945ea11ea4995570f11118adbd65c027f49c3c5f04066bbd066",
    "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz": "f7b69ab8fd6c19128a29f9ae9a11a7ac42a5d26630368ff74e906f29ddb6be44",
    "docs/data/seed416-policy-target-softening/training-freeze-v3/epoch-permutations.json.gz": "d868edebd9ae1876cbc5b9edca6fa0df9d3e6184b97246847f9012dfe2d293d1",
    "docs/data/seed426-canonical-overlap/seed427-evaluation-manifest.json": "7a72f03438380174fa71c2689999a3ff9b03853c38d0cf71706580339a3fb640",
    "docs/data/seed426-canonical-overlap/seed427-evaluation-results.json": "4899e990f2ff119d77b4c9f3a2f7bd8e24e2402449b4d4d52d13ad917f279172",
    "docs/data/seed432-policy-target-compatibility/manifest.json": "5fcd9b28d0a6fa2fe99dbf5cc42a00c58c947684ac041218ae138067e3d62a1a",
    "docs/data/seed433-seed432-census-correction/correction-receipt.json": "d915211825a3851988ac37a6ac71b1f5ebca029e170bb50ec78473cd2ee86b73",
}


def _read(root: Path, relative: str) -> Any:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def _authoritative(root: Path) -> tuple[np.ndarray, ...]:
    registration = _read(
        root, "docs/data/seed416-policy-target-softening/registration-v3.json"
    )
    specs = registration["replays"]
    paths = [root / SOURCE_DIR / f"{spec['name']}.jsonl.gz" for spec in specs]
    from ml.alphazero_lite.verify_seed434_census_source import reconstruct_sources

    reconstructed = list(reconstruct_sources(root, registration))
    states: list[list[float]] = []
    policies: list[list[float]] = []
    values: list[list[float]] = []
    coefficients: list[float] = []
    chunks: list[np.ndarray] = []
    offset = 0
    for path, spec in zip(paths, specs, strict=True):
        source_rows = []
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            source_rows = [json.loads(line) for line in stream]
        for row in source_rows:
            states.append(row["state"])
            policies.append(row["policy"])
            values.append([row["value"]])
            coefficients.append(
                policy_loss_weight_for_row(row, exact_root_policy_loss_weight=1.0)
            )
        chunks.append(
            np.tile(
                np.arange(offset, offset + len(source_rows), dtype=np.int64),
                int(spec["weight"]),
            )
        )
        offset += len(source_rows)
    if len(states) != len(reconstructed):
        raise ValueError("source_reconstruction_population_mismatch")
    x = np.asarray(states, dtype=np.float32)
    policy = np.asarray(policies, dtype=np.float32)
    value = np.asarray(values, dtype=np.float32)
    coefficient = np.asarray(coefficients, dtype=np.float32)
    replay = np.concatenate(chunks)
    for index, row in enumerate(reconstructed):
        if (
            not np.array_equal(
                policy[index], np.asarray(row["target"], dtype=np.float32)
            )
            or x[index].astype("<f4", copy=False).tobytes().hex() != row["input_hex"]
            or not math.isclose(
                float(coefficient[index]),
                float(row["policy_coefficient"]),
                abs_tol=1e-7,
                rel_tol=0,
            )
        ):
            raise ValueError(f"reconstructed_training_row_mismatch:{index}")
    with gzip.open(
        root / registration["training"]["source_row_split"]["path"], "rt"
    ) as stream:
        split = json.load(stream)
    return (
        x,
        policy,
        value,
        coefficient,
        replay,
        np.asarray(split["train_positions"], dtype=np.int64),
        np.asarray(split["validation_positions"], dtype=np.int64),
    )


def _membership(root: Path) -> list[dict[str, Any]]:
    path = root / RECEIPTS["membership"][0]
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        published = [json.loads(line) for line in stream]
    from ml.alphazero_lite.seed427_validation_subsets import construct, read_evidence

    source = read_evidence(
        root / "docs/data/seed426-canonical-overlap/row-accounting.jsonl.gz"
    )
    rebuilt, counts = construct(source)
    if rebuilt != published or counts["unseen"][">32"]["weighted_positions"] != 2607:
        raise ValueError("independent_validation_membership_mismatch")
    return rebuilt


def _prediction_population(
    root: Path, arrays: tuple[np.ndarray, ...], membership: list[dict[str, Any]]
) -> np.ndarray:
    x, _policy, _value, _coeff, _replay, train_pos, _validation_pos = arrays
    training_rows = set(map(int, arrays[4][train_pos]))
    unseen_rows = {
        int(row["compact_row"])
        for row in membership
        if row["subset"] == "unseen" and row["active_stones"] > 32
    }
    return np.asarray(sorted(training_rows | unseen_rows), dtype=np.int64)


def _predict(
    root: Path, checkpoint: Path, population: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    load_checkpoint_into_model(model, checkpoint)
    model.eval()
    with torch.inference_mode():
        logits, values = model(torch.from_numpy(x[population]))
        masks = torch.from_numpy(legal_mask_matrix_for_encoded_states(x[population]))
        legal_logits = logits.masked_fill(masks <= 0, -1e9)
    return legal_logits.numpy().astype(np.float32), values.reshape(-1).numpy().astype(
        np.float32
    )


def _losses(
    archive: dict[str, np.ndarray],
    arrays: tuple[np.ndarray, ...],
    membership: list[dict[str, Any]],
) -> dict[str, float]:
    x, target_policy, target_value, coefficients, replay, train_pos, _val_pos = arrays
    rows = archive["compact_rows"].astype(np.int64)
    lookup = {int(row): i for i, row in enumerate(rows)}
    logits = torch.from_numpy(archive["legal_logits"])
    masks = torch.from_numpy(legal_mask_matrix_for_encoded_states(x[rows]))
    targets = torch.from_numpy(target_policy[rows])
    policy_losses = compute_policy_cross_entropy(
        logits.masked_fill(masks <= 0, -1e9), targets
    ).numpy()
    values = archive["value_predictions"].astype(np.float64)
    squared = np.square(values - target_value[rows, 0].astype(np.float64))
    huber = (
        compute_value_loss_vector(
            torch.from_numpy(values.astype(np.float32)).reshape(-1, 1),
            torch.from_numpy(target_value[rows]),
            value_loss="huber",
            huber_delta=1.0,
        )
        .numpy()
        .astype(np.float64)
    )
    train_rows = replay[train_pos]
    train_locs = np.asarray([lookup[int(row)] for row in train_rows], dtype=np.int64)
    train_coeff = coefficients[train_rows].astype(np.float64)
    objective = float(
        np.average(policy_losses[train_locs], weights=train_coeff)
        + 0.3 * np.mean(huber[train_locs])
    )
    selected = [
        r for r in membership if r["subset"] == "unseen" and r["active_stones"] > 32
    ]
    locs = np.asarray([lookup[int(r["compact_row"])] for r in selected], dtype=np.int64)
    exposure_policy = float(np.mean(policy_losses[locs]))
    exposure_mse = float(np.mean(squared[locs]))
    groups: dict[str, list[int]] = {}
    for row, location in zip(selected, locs, strict=True):
        groups.setdefault(row["input_identity"], []).append(int(location))
    equal_policy = float(
        np.mean([np.mean(policy_losses[group]) for group in groups.values()])
    )
    equal_mse = float(np.mean([np.mean(squared[group]) for group in groups.values()]))
    return {
        "full_training_objective": objective,
        "exposure_weighted_policy_ce": exposure_policy,
        "exposure_weighted_value_mse": exposure_mse,
        "equal_input_policy_ce": equal_policy,
        "equal_input_value_mse": equal_mse,
    }


def publish_predictions(root: Path) -> dict[str, Any]:
    data = root / "docs/data/seed435-adam-direction-screen"
    arrays = _authoritative(root)
    membership = _membership(root)
    population = _prediction_population(root, arrays, membership)
    models = {
        "initializer": root
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
        "A": data / "A-final.npz",
        "B": data / "B-final.npz",
    }
    for name, checkpoint in models.items():
        archive = data / f"{name}-predictions.npz"
        logits, values = _predict(root, checkpoint, population, arrays[0])
        np.savez_compressed(
            archive,
            compact_rows=population,
            input_sha256=np.asarray(
                [
                    hashlib.sha256(
                        arrays[0][row].astype("<f4", copy=False).tobytes()
                    ).hexdigest()
                    for row in population
                ]
            ),
            canonical_identity=np.asarray(
                [
                    canonical_identity_from_encoded_state(arrays[0][row].tolist())
                    for row in population
                ]
            ),
            legal_logits=logits,
            value_predictions=values,
        )
    return {"population_rows": int(population.size)}


def recompute(root: Path) -> dict[str, Any]:
    data = root / "docs/data/seed435-adam-direction-screen"
    arrays = _authoritative(root)
    membership = _membership(root)
    rows_expected = _prediction_population(root, arrays, membership)
    metrics: dict[str, Any] = {}
    for name in ("initializer", "A", "B"):
        with np.load(
            data / f"{name}-predictions.npz", allow_pickle=False
        ) as archive_file:
            archive = {key: archive_file[key] for key in archive_file.files}
        if not np.array_equal(archive["compact_rows"], rows_expected):
            raise ValueError(f"prediction_population_mismatch:{name}")
        expected_input_hashes = np.asarray(
            [
                hashlib.sha256(
                    arrays[0][row].astype("<f4", copy=False).tobytes()
                ).hexdigest()
                for row in rows_expected
            ]
        )
        expected_canonical = np.asarray(
            [
                canonical_identity_from_encoded_state(arrays[0][row].tolist())
                for row in rows_expected
            ]
        )
        if not np.array_equal(
            archive["input_sha256"], expected_input_hashes
        ) or not np.array_equal(archive["canonical_identity"], expected_canonical):
            raise ValueError(f"prediction_identity_mismatch:{name}")
        checkpoint = root / (
            "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
            if name == "initializer"
            else f"docs/data/seed435-adam-direction-screen/{name}-final.npz"
        )
        logits, values = _predict(root, checkpoint, rows_expected, arrays[0])
        if not np.allclose(
            archive["legal_logits"],
            logits,
            atol=TOLERANCE["prediction_absolute"],
            rtol=0,
        ):
            raise ValueError(f"altered_predictions:{name}:logits")
        if not np.allclose(
            archive["value_predictions"],
            values,
            atol=TOLERANCE["prediction_absolute"],
            rtol=0,
        ):
            raise ValueError(f"altered_predictions:{name}:values")
        metrics[name] = _losses(archive, arrays, membership)
    decision_inputs = {
        name: {
            "full_training_objective": metrics[name]["full_training_objective"],
            "exposure_weighted": {
                "policy_ce": metrics[name]["exposure_weighted_policy_ce"],
                "value_mse": metrics[name]["exposure_weighted_value_mse"],
            },
            "equal_input": {
                "policy_ce": metrics[name]["equal_input_policy_ce"],
                "value_mse": metrics[name]["equal_input_value_mse"],
            },
        }
        for name in metrics
    }
    original = _read(root, "docs/data/seed435-adam-direction-screen/results.json")
    recorded_arms = original["arms"]
    discrepancies = {}
    for name in metrics:
        comparisons = {
            "full_training_objective": (
                metrics[name]["full_training_objective"],
                recorded_arms[name]["full_training_objective"],
            )
        }
        for weighting, source_key in (
            ("exposure_weighted", "exposure_weighted"),
            ("equal_input", "equal_input"),
        ):
            for metric_key, recomputed_key in (
                ("policy_ce", f"{source_key}_policy_ce"),
                ("value_mse", f"{source_key}_value_mse"),
            ):
                comparisons[f"{weighting}.{metric_key}"] = (
                    metrics[name][recomputed_key],
                    recorded_arms[name][weighting][metric_key],
                )
        discrepancies[name] = {
            key: {
                "recomputed": actual,
                "original": recorded,
                "difference": actual - recorded,
            }
            for key, (actual, recorded) in comparisons.items()
        }
    classification = decide(decision_inputs)
    if classification != original["decision"]:
        raise ValueError("decision_mismatch")
    return {
        "metrics": metrics,
        "decision": classification,
        "original_aggregate_discrepancies": discrepancies,
    }


def _verify_telemetry(root: Path) -> dict[str, Any]:
    data = root / "docs/data/seed435-adam-direction-screen"
    telemetry = _read(
        root, "docs/data/seed435-adam-direction-screen/post-execution-telemetry.json"
    )
    if telemetry["final_parameter_arrays_match_original"] is not True:
        raise ValueError("audit_reproduction_not_checkpoint_identical")
    layout = telemetry["parameter_layout"]
    names = [item["name"] for item in layout]
    if len(names) != len(set(names)) or set(telemetry["group_assignment"]) != set(
        names
    ):
        raise ValueError("parameter_layout_invalid")
    groups = {
        key: [item for item in layout if item["group"] == key]
        for key in ("shared_trunk", "policy_head", "value_head")
    }
    if any(not groups[key] for key in groups) or sum(
        len(value) for value in groups.values()
    ) != len(layout):
        raise ValueError("parameter_groups_not_exhaustive")
    vectors = {}
    for arm in ("A", "B"):
        logs = telemetry["arms"][arm]
        with np.load(data / f"{arm}-audit-deltas.npz", allow_pickle=False) as archive:
            deltas = archive["deltas"].astype(np.float64)
        if (
            len(logs) != 16
            or deltas.shape[0] != 16
            or [row["step"] for row in logs] != list(range(1, 17))
        ):
            raise ValueError(f"audit_telemetry_count_invalid:{arm}")
        vectors[arm] = deltas
        for index, row in enumerate(logs):
            vector = deltas[index]
            squares = {}
            for group, members in groups.items():
                squares[group] = sum(
                    float(np.sum(vector[item["start"] : item["stop"]] ** 2))
                    for item in members
                )
                if not math.isclose(
                    math.sqrt(squares[group]),
                    row["group_norms"][group],
                    abs_tol=TOLERANCE["absolute"],
                    rel_tol=TOLERANCE["relative"],
                ):
                    raise ValueError(f"group_norm_mismatch:{arm}:{index + 1}:{group}")
            norm_sq = float(np.sum(vector**2))
            if not math.isclose(
                sum(squares.values()), norm_sq, abs_tol=1e-10, rel_tol=2e-6
            ):
                raise ValueError(f"group_norm_square_sum_mismatch:{arm}:{index + 1}")
            if not math.isclose(
                math.sqrt(norm_sq),
                row["update_norm"],
                abs_tol=TOLERANCE["absolute"],
                rel_tol=TOLERANCE["relative"],
            ):
                raise ValueError(f"update_vector_norm_mismatch:{arm}:{index + 1}")
    original_a = _read(root, "docs/data/seed435-adam-direction-screen/A-updates.json")
    original_b = _read(root, "docs/data/seed435-adam-direction-screen/B-updates.json")
    if len(original_a) != 16 or len(original_b) != 16:
        raise ValueError("original_update_count_invalid")
    arrays = _authoritative(root)
    train_replay = arrays[4][arrays[5]]
    registered_order = _read(
        root, "docs/data/seed435-adam-direction-screen/registration.json"
    )["first_epoch_permutation_prefix"]
    for index in range(16):
        expected_batch = registered_order[index * 512 : (index + 1) * 512]
        expected_rows = train_replay[
            np.asarray(expected_batch, dtype=np.int64)
        ].tolist()
        for arm, updates in (("A", original_a), ("B", original_b)):
            batch = updates[index]["batch"]
            if (
                updates[index]["step"] != index + 1
                or batch["expanded_positions"] != expected_batch
                or batch["compact_rows"] != expected_rows
            ):
                raise ValueError(f"original_batch_identity_mismatch:{arm}:{index + 1}")
        if not math.isclose(
            original_a[index]["stored_delta_norm"],
            original_b[index]["requested_radius"],
            abs_tol=TOLERANCE["absolute"],
            rel_tol=TOLERANCE["relative"],
        ):
            raise ValueError(f"paired_radius_mismatch:{index + 1}")
        a, b = vectors["A"][index], vectors["B"][index]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if not math.isclose(
            nb,
            original_a[index]["stored_delta_norm"],
            abs_tol=TOLERANCE["absolute"],
            rel_tol=TOLERANCE["relative"],
        ):
            raise ValueError(f"B_actual_update_radius_mismatch:{index + 1}")
        cosine = float(np.dot(a, b) / (na * nb)) if na and nb else None
        for arm in ("A", "B"):
            reported = telemetry["arms"][arm][index]["direction_cosine_vs_paired_arm"]
            if cosine is None and reported is not None:
                raise ValueError(f"zero_vector_cosine_not_undefined:{index + 1}")
        if cosine is not None and (
            reported is None
            or not math.isclose(reported, cosine, abs_tol=1e-10, rel_tol=1e-10)
        ):
            raise ValueError(f"paired_cosine_mismatch:{arm}:{index + 1}")
    initial = PolicyValueNet((96, 3), "residual_v3", 27)
    load_checkpoint_into_model(
        initial,
        root
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
    )
    final_a = PolicyValueNet((96, 3), "residual_v3", 27)
    final_b = PolicyValueNet((96, 3), "residual_v3", 27)
    load_checkpoint_into_model(final_a, data / "A-final.npz")
    load_checkpoint_into_model(final_b, data / "B-final.npz")
    for arm, final_model in (("A", final_a), ("B", final_b)):
        model = PolicyValueNet((96, 3), "residual_v3", 27)
        load_checkpoint_into_model(
            model,
            root
            / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
        )
        for step_vector in vectors[arm]:
            with torch.no_grad():
                for (name, parameter), entry in zip(
                    (
                        (name, parameter)
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                    ),
                    layout,
                    strict=True,
                ):
                    if name != entry["name"]:
                        raise ValueError("audit_parameter_order_mismatch")
                    parameter.add_(
                        torch.from_numpy(
                            step_vector[entry["start"] : entry["stop"]]
                            .astype(np.float32)
                            .reshape(entry["shape"])
                        )
                    )
        if any(
            not torch.equal(actual, expected)
            for actual, expected in zip(
                model.parameters(), final_model.parameters(), strict=True
            )
        ):
            raise ValueError(
                f"audit_delta_accumulation_final_checkpoint_mismatch:{arm}"
            )
    return {"status": "valid", "paired_updates": 16, "parameter_count": len(layout)}


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    data = root / "docs/data/seed435-adam-direction-screen"
    registration = _read(
        root, "docs/data/seed435-adam-direction-screen/registration.json"
    )
    if (
        sha(
            root
            / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
        )
        != INIT_SHA
    ):
        raise ValueError("initializer_identity_mismatch")
    for name, (path, expected) in RECEIPTS.items():
        if sha(root / path) != expected:
            raise ValueError(f"seed427_receipt_binding_mismatch:{name}")
    for relative, expected in FROZEN_HASHES.items():
        if sha(root / relative) != expected:
            raise ValueError(f"frozen_source_binding_mismatch:{relative}")
    membership = _membership(root)
    membership_path = root / RECEIPTS["membership"][0]
    receipt = _read(
        root, "docs/data/seed435-adam-direction-screen/supplemental-receipt.json"
    )
    if receipt["frozen_input_hashes"] != FROZEN_HASHES:
        raise ValueError("frozen_input_receipt_mismatch")
    if receipt["membership_sha256"] != sha(membership_path):
        raise ValueError("membership_binding_mismatch")
    if receipt["membership_row_count"] != len(membership):
        raise ValueError("membership_population_mismatch")
    for name in ("initializer", "A", "B"):
        path = data / f"{name}-predictions.npz"
        if receipt["prediction_archives"][name] != sha(path):
            raise ValueError(f"prediction_archive_binding_mismatch:{name}")
    for field, filename in (
        ("telemetry_json_sha256", "post-execution-telemetry.json"),
        ("A_deltas_sha256", "A-audit-deltas.npz"),
        ("B_deltas_sha256", "B-audit-deltas.npz"),
        ("cosines_sha256", "A-B-audit-delta-cosines.npz"),
    ):
        if receipt["audit_telemetry"][field] != sha(data / filename):
            raise ValueError(f"audit_telemetry_binding_mismatch:{field}")
    originals = receipt["original_evidence"]
    for relative, expected_hash in originals.items():
        if sha(root / relative) != expected_hash:
            raise ValueError(f"original_evidence_binding_mismatch:{relative}")
    sources = receipt["verification_sources"]
    for relative, expected_hash in sources.items():
        if sha(root / relative) != expected_hash:
            raise ValueError(f"verification_source_binding_mismatch:{relative}")
    telemetry_check = _verify_telemetry(root)
    if receipt["telemetry_verification"] != telemetry_check:
        raise ValueError("telemetry_receipt_mismatch")
    results = recompute(root)
    expected = receipt["recomputed_results"]
    if results != expected:
        raise ValueError("recomputed_results_receipt_mismatch")
    if len(membership) != 14946 or registration["source_reconstruction_count"] != 87625:
        raise ValueError("frozen_population_count_mismatch")
    return {
        "status": "valid",
        "classification": results["decision"]["classification"],
        "prediction_rows": receipt["prediction_population_rows"],
        "relocated_root": str(root),
    }


def publish_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    data = root / "docs/data/seed435-adam-direction-screen"
    results = recompute(root)
    telemetry = _verify_telemetry(root)
    original_names = (
        "registration.json",
        "initializer.npz",
        "A-final.npz",
        "B-final.npz",
        "A-updates.json",
        "B-updates.json",
        "results.json",
    )
    originals = {
        f"docs/data/seed435-adam-direction-screen/{name}": sha(data / name)
        for name in original_names
    }
    verification_sources = {
        relative: sha(root / relative)
        for relative in (
            "ml/alphazero_lite/verify_seed435_publication.py",
            "ml/alphazero_lite/seed435_telemetry_recovery.py",
            "ml/alphazero_lite/seed435_adam_direction.py",
            "ml/alphazero_lite/test_seed435_publication.py",
            "ml/alphazero_lite/test_seed435_adam_direction.py",
        )
    }
    membership_path = root / RECEIPTS["membership"][0]
    receipt = {
        "schema": "seed435-supplemental-verification-receipt-v1",
        "status": "post_execution_verification",
        "interpretation": "read-only predictions and deterministic audit reproduction; no training was extended or tuned",
        "original_evidence": originals,
        "frozen_input_hashes": FROZEN_HASHES,
        "membership_sha256": sha(membership_path),
        "membership_row_count": len(_membership(root)),
        "prediction_archives": {
            name: sha(data / f"{name}-predictions.npz")
            for name in ("initializer", "A", "B")
        },
        "prediction_population_rows": int(
            np.load(data / "A-predictions.npz")["compact_rows"].size
        ),
        "audit_telemetry": {
            "telemetry_json_sha256": sha(data / "post-execution-telemetry.json"),
            "A_deltas_sha256": sha(data / "A-audit-deltas.npz"),
            "B_deltas_sha256": sha(data / "B-audit-deltas.npz"),
            "cosines_sha256": sha(data / "A-B-audit-delta-cosines.npz"),
            "recovery_method": "replayed frozen 16-step protocols using original initializer, source targets, coefficients, split, permutation, and optimizer equations; both recovered final parameter arrays matched original final checkpoints bit-for-bit",
        },
        "telemetry_verification": telemetry,
        "verification_sources": verification_sources,
        "tolerances": TOLERANCE,
        "recomputed_results": results,
    }
    target = data / "supplemental-receipt.json"
    target.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "command",
        choices=("publish-predictions", "publish-receipt", "verify"),
        nargs="?",
        default="verify",
    )
    args = parser.parse_args()
    if args.command == "publish-predictions":
        result = publish_predictions(args.root.resolve())
    elif args.command == "publish-receipt":
        result = publish_receipt(args.root.resolve())
    else:
        result = verify(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
