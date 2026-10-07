"""Read-only semantic verification of seed435 direction telemetry (seed436)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.seed435_adam_direction import decide
from ml.alphazero_lite.train import PolicyValueNet
from ml.alphazero_lite.verify_seed435_publication import (
    TOLERANCE,
    verify as verify_seed435,
)

DATA = "docs/data/seed435-adam-direction-screen"
ARMS = ("A", "B")
GROUPS = ("shared_trunk", "policy_head", "value_head")
TOLERANCES = {
    "cosine_absolute": 1e-10,
    "cosine_relative": 1e-10,
    "norm_absolute": TOLERANCE["absolute"],
    "norm_relative": TOLERANCE["relative"],
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(root: Path, name: str) -> Any:
    return json.loads((root / DATA / name).read_text(encoding="utf-8"))


def _group(name: str) -> str:
    if name.startswith(("input_layer.", "residual_layers.")):
        return "shared_trunk"
    if name.startswith(("policy_hidden_layer.", "policy_head.")):
        return "policy_head"
    if name.startswith(("value_hidden_layer.", "value_head.")):
        return "value_head"
    raise ValueError(f"unassigned_architecture_parameter:{name}")


def validate_semantics(root: Path) -> dict[str, Any]:
    """Validate telemetry against archived vectors, architecture, and original logs."""
    root = root.resolve()
    data = root / DATA
    telemetry = _json(root, "post-execution-telemetry.json")
    if telemetry.get("final_parameter_arrays_match_original") is not True:
        raise ValueError("audit_reproduction_not_checkpoint_identical")

    model = PolicyValueNet((96, 3), "residual_v3", 27)
    parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    expected_layout = []
    offset = 0
    for name, parameter in parameters:
        size = parameter.numel()
        expected_layout.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "start": offset,
                "stop": offset + size,
                "group": _group(name),
            }
        )
        offset += size
    layout = telemetry.get("parameter_layout")
    if layout != expected_layout:
        raise ValueError("parameter_layout_architecture_mismatch")
    if (
        layout[0]["start"] != 0
        or layout[-1]["stop"] != offset
        or any(
            left["stop"] != right["start"] for left, right in zip(layout, layout[1:])
        )
    ):
        raise ValueError("parameter_layout_slices_not_contiguous_exhaustive")
    assignment = telemetry.get("group_assignment", {})
    expected_assignment = {entry["name"]: entry["group"] for entry in expected_layout}
    if assignment != expected_assignment:
        raise ValueError("group_assignment_architecture_mismatch")
    if set(assignment.values()) != set(GROUPS):
        raise ValueError("parameter_groups_not_exhaustive")

    vectors: dict[str, np.ndarray] = {}
    logs = {}
    for arm in ARMS:
        logs[arm] = telemetry["arms"][arm]
        with np.load(data / f"{arm}-audit-deltas.npz", allow_pickle=False) as archive:
            vectors[arm] = archive["deltas"].astype(np.float64)
        if vectors[arm].shape != (16, offset) or len(logs[arm]) != 16:
            raise ValueError(f"audit_telemetry_shape_invalid:{arm}")
        if [row.get("step") for row in logs[arm]] != list(range(1, 17)):
            raise ValueError(f"audit_telemetry_count_invalid:{arm}")
        if not np.isfinite(vectors[arm]).all():
            raise ValueError(f"nonfinite_update_vector:{arm}")

    with np.load(data / "A-B-audit-delta-cosines.npz", allow_pickle=False) as archive:
        cosine_archive = archive["cosines"]
    if cosine_archive.shape != (16,) or cosine_archive.dtype.kind not in "fiu":
        raise ValueError("cosine_archive_shape_invalid")

    originals = {arm: _json(root, f"{arm}-updates.json") for arm in ARMS}
    for step in range(16):
        a, b = vectors["A"][step], vectors["B"][step]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        undefined = na == 0 or nb == 0
        cosine = None if undefined else float(np.dot(a, b) / (na * nb))
        expected_cosine = 0.0 if cosine is None else cosine
        archive_value = float(cosine_archive[step])
        if undefined:
            if not math.isnan(archive_value):
                raise ValueError(f"zero_vector_archive_not_undefined:{step + 1}")
        elif not math.isfinite(archive_value) or not math.isclose(
            archive_value,
            expected_cosine,
            abs_tol=TOLERANCES["cosine_absolute"],
            rel_tol=TOLERANCES["cosine_relative"],
        ):
            raise ValueError(f"cosine_archive_mismatch:{step + 1}")
        for arm in ARMS:
            record = logs[arm][step]
            reported = record.get("direction_cosine_vs_paired_arm")
            flagged = record.get("cosine_undefined_zero_vector")
            if flagged is not undefined:
                raise ValueError(f"cosine_undefined_flag_mismatch:{arm}:{step + 1}")
            if undefined:
                if reported is not None:
                    raise ValueError(
                        f"zero_vector_cosine_not_undefined:{arm}:{step + 1}"
                    )
            elif (
                reported is None
                or not math.isfinite(float(reported))
                or not math.isclose(
                    float(reported),
                    expected_cosine,
                    abs_tol=TOLERANCES["cosine_absolute"],
                    rel_tol=TOLERANCES["cosine_relative"],
                )
            ):
                raise ValueError(f"paired_cosine_mismatch:{arm}:{step + 1}")
            group_norms = record.get("group_norms", {})
            if set(group_norms) != set(GROUPS):
                raise ValueError(f"group_norms_invalid:{arm}:{step + 1}")
            for group in GROUPS:
                members = [
                    entry for entry in expected_layout if entry["group"] == group
                ]
                norm = float(
                    np.linalg.norm(
                        np.concatenate(
                            [
                                vectors[arm][step, item["start"] : item["stop"]]
                                for item in members
                            ]
                        )
                    )
                )
                if not math.isfinite(float(group_norms[group])) or not math.isclose(
                    norm,
                    float(group_norms[group]),
                    abs_tol=TOLERANCES["norm_absolute"],
                    rel_tol=TOLERANCES["norm_relative"],
                ):
                    raise ValueError(f"group_norm_mismatch:{arm}:{step + 1}:{group}")
            norm = float(np.linalg.norm(vectors[arm][step]))
            if not math.isfinite(float(record["update_norm"])) or not math.isclose(
                norm,
                float(record["update_norm"]),
                abs_tol=TOLERANCES["norm_absolute"],
                rel_tol=TOLERANCES["norm_relative"],
            ):
                raise ValueError(f"update_vector_norm_mismatch:{arm}:{step + 1}")
            if not math.isfinite(
                float(record["delta_vector_norm"])
            ) or not math.isclose(
                norm,
                float(record["delta_vector_norm"]),
                abs_tol=TOLERANCES["norm_absolute"],
                rel_tol=TOLERANCES["norm_relative"],
            ):
                raise ValueError(f"delta_vector_norm_mismatch:{arm}:{step + 1}")
            if not math.isfinite(
                float(originals[arm][step]["stored_delta_norm"])
            ) or not math.isclose(
                norm,
                float(originals[arm][step]["stored_delta_norm"]),
                abs_tol=TOLERANCES["norm_absolute"],
                rel_tol=TOLERANCES["norm_relative"],
            ):
                raise ValueError(f"original_update_norm_mismatch:{arm}:{step + 1}")
        if not math.isfinite(
            float(originals["B"][step]["requested_radius"])
        ) or not math.isclose(
            float(originals["A"][step]["stored_delta_norm"]),
            float(originals["B"][step]["requested_radius"]),
            abs_tol=TOLERANCES["norm_absolute"],
            rel_tol=TOLERANCES["norm_relative"],
        ):
            raise ValueError(f"paired_radius_mismatch:{step + 1}")

    source_results = _json(root, "results.json")
    supplemental = _json(root, "supplemental-receipt.json")
    metrics = supplemental["recomputed_results"]["metrics"]
    decision_inputs = {
        arm: {
            "full_training_objective": metrics[arm]["full_training_objective"],
            "exposure_weighted": {
                "policy_ce": metrics[arm]["exposure_weighted_policy_ce"],
                "value_mse": metrics[arm]["exposure_weighted_value_mse"],
            },
            "equal_input": {
                "policy_ce": metrics[arm]["equal_input_policy_ce"],
                "value_mse": metrics[arm]["equal_input_value_mse"],
            },
        }
        for arm in ("initializer", "A", "B")
    }
    if (
        decide(decision_inputs) != source_results["decision"]
        or supplemental["recomputed_results"]["decision"] != source_results["decision"]
    ):
        raise ValueError("seed435_decision_changed")
    return {
        "status": "valid",
        "paired_updates": 16,
        "parameter_count": len(parameters),
        "classification": source_results["decision"]["classification"],
    }


def evidence_bindings(root: Path) -> dict[str, str]:
    """Return concrete hashes for all seed435 evidence and seed436 verifier sources."""
    data = root / DATA
    relatives = [
        f"{DATA}/{path.name}" for path in sorted(data.iterdir()) if path.is_file()
    ]
    source_dir = root / "docs/data/seed426-canonical-overlap/sources"
    relatives.extend(
        str(path.relative_to(root))
        for path in sorted(source_dir.iterdir())
        if path.is_file()
    )
    relatives.extend(
        (
            "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
            "docs/data/seed416-policy-target-softening/registration-v3.json",
            "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz",
            "docs/data/seed416-policy-target-softening/training-freeze-v3/epoch-permutations.json.gz",
            "docs/data/seed426-canonical-overlap/row-accounting.jsonl.gz",
            "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
            "docs/data/seed426-canonical-overlap/seed427-evaluation-verification-receipt.json",
            "docs/data/seed426-canonical-overlap/seed427-supplemental-verification-receipt.json",
            "docs/data/seed426-canonical-overlap/seed427-evaluation-manifest.json",
            "docs/data/seed426-canonical-overlap/seed427-evaluation-results.json",
            "docs/data/seed432-policy-target-compatibility/manifest.json",
            "docs/data/seed433-seed432-census-correction/correction-receipt.json",
            "ml/alphazero_lite/verify_seed435_publication.py",
            "ml/alphazero_lite/verify_seed434_census_source.py",
            "ml/alphazero_lite/seed435_telemetry_recovery.py",
            "ml/alphazero_lite/seed435_adam_direction.py",
            "ml/alphazero_lite/test_seed435_publication.py",
            "ml/alphazero_lite/test_seed435_adam_direction.py",
            "ml/alphazero_lite/verify_seed436_direction_telemetry.py",
            "ml/alphazero_lite/test_seed436_direction_telemetry.py",
        )
    )
    return {relative: _sha(root / relative) for relative in relatives}


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    original = verify_seed435(root)
    semantic = validate_semantics(root)
    correction_path = (
        root
        / "docs/data/seed436-direction-telemetry-verification/correction-receipt.json"
    )
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    if correction["schema"] != "seed436-direction-telemetry-correction-v1":
        raise ValueError("seed436_correction_receipt_mismatch")
    if correction["bindings"] != evidence_bindings(root):
        raise ValueError("seed436_evidence_binding_mismatch")
    if correction["tolerances"] != TOLERANCES:
        raise ValueError("seed436_tolerance_mismatch")
    if correction["semantic_verification"] != semantic:
        raise ValueError("seed436_semantic_receipt_mismatch")
    if original["classification"] != semantic["classification"]:
        raise ValueError("seed435_classification_changed")
    return {
        "status": "valid",
        "original_verification": original,
        "semantic_verification": semantic,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
