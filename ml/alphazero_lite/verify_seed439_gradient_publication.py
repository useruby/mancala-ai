"""Independent semantic verification of the seed438 gradient publication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.verify_seed437_training_unseen_gradient_alignment import (
    verify as verify_original,
)
from ml.alphazero_lite.verify_seed438_seed437_gradient_correction import (
    verify as verify_historical,
)

TOL = 2e-5
DECISION = "persistent_training_objective_opposition"
NO_DECISION = "no_persistent_training_objective_opposition"
CHECKPOINTS = {"initializer", "adam_a16"}
WEIGHTINGS = {"exposure", "equal_input"}
OBJECTIVES = {"P", "0.3V", "T"}
SOURCES = {
    "fresh",
    "generic_bootstrap",
    "random_teacher",
    "opening_disagreement",
    "stability",
}
BUCKETS = {"<=16", "17-21", "22-32", ">32"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(actual: float, expected: Any, label: str) -> None:
    if (
        expected is None
        or not np.isfinite(expected)
        or not np.isclose(actual, expected, atol=TOL, rtol=TOL)
    ):
        raise ValueError(f"metric_mismatch:{label}")


def validate_semantics(
    root: Path, result: dict[str, Any], archive_path: Path
) -> dict[str, Any]:
    """Validate published vectors and every derived semantic field without trusting hashes."""
    original_archive_path = (
        root
        / "docs/data/seed437-training-unseen-gradient-alignment/gradient-vectors.npz"
    )
    with (
        np.load(archive_path, allow_pickle=False) as archive,
        np.load(original_archive_path, allow_pickle=False) as original_archive,
    ):
        if set(result["checkpoints"]) != CHECKPOINTS:
            raise ValueError("checkpoint_coverage_mismatch")
        expected_arrays: set[str] = set()
        cosines: list[float] = []
        discrepancies: dict[str, float] = {}
        for checkpoint in sorted(CHECKPOINTS):
            record = result["checkpoints"][checkpoint]
            evidence = record["vector_evidence"]
            expected_vectors = {"P", "0.3V", "T", *(f"U_{w}" for w in WEIGHTINGS)}
            expected_vectors |= {
                f"{o}__{axis}__{part}"
                for o in OBJECTIVES
                for axis, labels in (("source", SOURCES), ("bucket", BUCKETS))
                for part in labels
            }
            expected_vectors |= {
                f"U_{w}__bucket__{b}" for w in WEIGHTINGS for b in BUCKETS
            }
            if set(evidence) != expected_vectors:
                raise ValueError(f"vector_coverage_mismatch:{checkpoint}")
            size = record["parameter_count"]
            manifest = json.loads(
                (
                    root
                    / "docs/data/seed437-training-unseen-gradient-alignment/manifest.json"
                ).read_text()
            )
            layout = manifest["model"]["flat_parameter_layout"]
            if record["parameter_names"] != [entry["name"] for entry in layout]:
                raise ValueError(f"parameter_order_mismatch:{checkpoint}")
            if manifest["model"]["parameter_count"] != size:
                raise ValueError(f"parameter_count_mismatch:{checkpoint}")
            for entry in layout:
                if int(np.prod(entry["shape"])) != entry["stop"] - entry["start"]:
                    raise ValueError(f"parameter_shape_mismatch:{entry['name']}")
            vectors: dict[str, np.ndarray] = {}
            for name, metadata in evidence.items():
                key = metadata["archive_key"]
                expected_arrays.add(key)
                vector = np.asarray(archive[key], dtype=np.float64)
                if vector.shape != (size,) or not np.isfinite(vector).all():
                    raise ValueError(f"vector_invalid:{checkpoint}:{name}")
                if (
                    hashlib.sha256(vector.astype("<f8").tobytes()).hexdigest()
                    != metadata["sha256"]
                ):
                    raise ValueError(f"vector_digest_mismatch:{checkpoint}:{name}")
                vectors[name] = vector
            if not np.allclose(
                vectors["T"], vectors["P"] + vectors["0.3V"], atol=TOL, rtol=TOL
            ):
                raise ValueError(f"combined_vector_mismatch:{checkpoint}")
            if set(record["metrics"]) != WEIGHTINGS or set(
                record["decomposition"]
            ) != OBJECTIVES | {f"U_{w}" for w in WEIGHTINGS}:
                raise ValueError(f"record_coverage_mismatch:{checkpoint}")
            for objective in OBJECTIVES:
                for axis, labels in (("source", SOURCES), ("bucket", BUCKETS)):
                    if set(record["decomposition"][objective][axis]) != labels:
                        raise ValueError("component_coverage_mismatch")
                    pieces = [
                        vectors[f"{objective}__{axis}__{label}"]
                        for label in sorted(labels)
                    ]
                    if not np.allclose(
                        np.sum(pieces, axis=0), vectors[objective], atol=TOL, rtol=TOL
                    ):
                        raise ValueError(
                            f"component_sum_mismatch:{checkpoint}:{objective}:{axis}"
                        )
            groups = record["parameter_group_indices"]
            if set(groups) != {"shared_trunk", "policy_head", "value_head"} or sorted(
                i for v in groups.values() for i in v
            ) != list(range(size)):
                raise ValueError(f"parameter_layout_invalid:{checkpoint}")
            for group in groups:
                indexes = [
                    i
                    for entry in layout
                    if entry["group"] == group
                    for i in range(entry["start"], entry["stop"])
                ]
                if groups[group] != indexes:
                    raise ValueError(
                        f"parameter_group_slice_mismatch:{checkpoint}:{group}"
                    )
            if set(record["dot_contributions"]) != {
                f"{w}/{o}" for w in WEIGHTINGS for o in OBJECTIVES
            }:
                raise ValueError("contribution_record_coverage_mismatch")
            for weighting in sorted(WEIGHTINGS):
                u = vectors[f"U_{weighting}"]
                if set(record["decomposition"][f"U_{weighting}"]["bucket"]) != BUCKETS:
                    raise ValueError("validation_bucket_coverage_mismatch")
                uparts = [
                    vectors[f"U_{weighting}__bucket__{b}"] for b in sorted(BUCKETS)
                ]
                if not np.allclose(sum(uparts), u, atol=TOL, rtol=TOL):
                    raise ValueError("validation_component_sum_mismatch")
                for objective in sorted(OBJECTIVES):
                    g = vectors[objective]
                    dot = float(np.dot(g, u))
                    metric = record["metrics"][weighting][objective]
                    gn, un = float(np.linalg.norm(g)), float(np.linalg.norm(u))
                    cosine = dot / (gn * un) if gn and un else None
                    unit = -dot / gn if gn else None
                    for field, value in (
                        ("dot", dot),
                        ("left_norm", gn),
                        ("right_norm", un),
                    ):
                        _close(
                            value,
                            metric[field],
                            f"{checkpoint}:{weighting}:{objective}:{field}",
                        )
                    for field, value in (
                        ("cosine", cosine),
                        ("unit_descent_ce_change", unit),
                    ):
                        if value is None:
                            if metric[field] is not None:
                                raise ValueError(f"zero_norm_mismatch:{field}")
                        else:
                            _close(
                                value,
                                metric[field],
                                f"{checkpoint}:{weighting}:{objective}:{field}",
                            )
                    details = record["dot_contributions"][f"{weighting}/{objective}"]
                    for axis, labels in (("source", SOURCES), ("bucket", BUCKETS)):
                        if set(details[axis]) != labels:
                            raise ValueError("contribution_coverage_mismatch")
                        for label in labels:
                            actual = float(
                                np.dot(u, vectors[f"{objective}__{axis}__{label}"])
                            )
                            _close(
                                actual,
                                details[axis][label],
                                f"contribution:{axis}:{label}",
                            )
                    if set(details["parameter_group"]) != set(groups):
                        raise ValueError("parameter_contribution_coverage_mismatch")
                    for group, indices in groups.items():
                        _close(
                            float(np.dot(u[indices], g[indices])),
                            details["parameter_group"][group],
                            f"parameter:{group}",
                        )
                    for axis, parts in details.items():
                        _close(sum(parts.values()), dot, f"contribution_sum:{axis}")
                    if objective == "T":
                        cosines.append(float(cosine) if cosine is not None else 0.0)
            for name in ("P", "U_exposure", "U_equal_input"):
                corrected = vectors[name]
                source = np.asarray(
                    original_archive[f"{checkpoint}__{name}"], dtype=np.float64
                )
                if not np.allclose(corrected, source, atol=TOL, rtol=TOL):
                    raise ValueError(
                        f"original_vector_comparison_mismatch:{checkpoint}:{name}"
                    )
                discrepancies[f"{checkpoint}/{name}"] = float(
                    np.max(np.abs(corrected - source))
                )
            mislabeled = np.asarray(
                original_archive[f"{checkpoint}__0.3V"], dtype=np.float64
            )
            if not np.allclose(
                vectors["0.3V"], (10.0 / 3.0) * mislabeled, atol=TOL, rtol=TOL
            ):
                raise ValueError(f"weighted_value_comparison_mismatch:{checkpoint}")
            discrepancies[f"{checkpoint}/0.3V_vs_scaled_original"] = float(
                np.max(np.abs(vectors["0.3V"] - (10.0 / 3.0) * mislabeled))
            )
        if set(archive.files) != expected_arrays:
            raise ValueError("archive_contents_mismatch")
    classification = (
        DECISION
        if len(cosines) == 4 and all(c <= -0.05 for c in cosines)
        else NO_DECISION
    )
    if result.get("classification") != classification:
        raise ValueError("classification_mismatch")
    return {
        "classification": classification,
        "recomputed_T_U_cosines": cosines,
        "max_abs_vector_discrepancies": discrepancies,
    }


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    verify_original(root)
    verify_historical(root)
    base = root / "docs/data/seed438-seed437-gradient-correction"
    protocol_path, result_path = base / "protocol.json", base / "corrected-results.json"
    protocol, result = (
        json.loads(protocol_path.read_text()),
        json.loads(result_path.read_text()),
    )
    manifest_path = (
        root / "docs/data/seed437-training-unseen-gradient-alignment/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads((base / "correction-receipt.json").read_text())
    original = root / "docs/data/seed437-training-unseen-gradient-alignment"
    required = {
        "original_evidence": {
            "manifest_sha256": sha(manifest_path),
            "gradient_vectors_sha256": sha(original / "gradient-vectors.npz"),
            "results_sha256": sha(original / "results.json"),
        },
        "frozen_protocol_sha256": sha(protocol_path),
        "corrected_sources": protocol["corrected_sources"],
        "corrected_vectors_sha256": sha(base / "corrected-gradient-vectors.npz"),
        "corrected_results_sha256": sha(result_path),
        "checkpoint_hashes_unchanged": {
            k: v["sha256"] for k, v in manifest["checkpoints"].items()
        },
        "component_comparison_tolerances": {"absolute": TOL, "relative": TOL},
        "classification": result["classification"],
    }
    for field, value in required.items():
        if receipt.get(field) != value:
            raise ValueError(f"receipt_field_mismatch:{field}")
    manifest_bound_fields = {
        "checkpoints",
        "counts",
        "denominators",
        "membership",
        "model",
        "reconstruction_evidence",
        "registration",
        "sources",
        "split",
    }
    if protocol["bound_inputs"] != {
        field: manifest[field] for field in manifest_bound_fields
    }:
        raise ValueError("protocol_checkpoint_binding_mismatch")
    geometry = validate_semantics(root, result, base / result["vector_archive"]["path"])
    receipt_path = (
        root
        / "docs/data/seed439-gradient-publication-verification/verification-receipt.json"
    )
    verification_receipt = json.loads(receipt_path.read_text())
    expected_receipt_hashes = {
        "seed437_manifest_sha256": sha(manifest_path),
        "seed437_vectors_sha256": sha(original / "gradient-vectors.npz"),
        "seed437_results_sha256": sha(original / "results.json"),
        "seed438_protocol_sha256": sha(protocol_path),
        "seed438_correction_receipt_sha256": sha(base / "correction-receipt.json"),
        "seed438_results_sha256": sha(result_path),
        "seed438_vectors_sha256": sha(base / "corrected-gradient-vectors.npz"),
    }
    if verification_receipt["historical_evidence"] != expected_receipt_hashes:
        raise ValueError("seed439_receipt_evidence_mismatch")
    for path, digest in verification_receipt["verification_sources"].items():
        if sha(root / path) != digest:
            raise ValueError(f"seed439_receipt_source_mismatch:{path}")
    if verification_receipt["classification"] != geometry["classification"]:
        raise ValueError("seed439_receipt_classification_mismatch")
    if (
        verification_receipt["recomputed_T_U_cosines"]
        != geometry["recomputed_T_U_cosines"]
    ):
        raise ValueError("seed439_receipt_geometry_mismatch")
    if (
        verification_receipt["max_abs_vector_discrepancies"]
        != geometry["max_abs_vector_discrepancies"]
    ):
        raise ValueError("seed439_receipt_comparison_mismatch")
    return {"status": "valid", **geometry}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
