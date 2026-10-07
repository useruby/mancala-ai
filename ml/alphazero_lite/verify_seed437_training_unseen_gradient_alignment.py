"""Portable independent verifier for seed437 vector and classification evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.seed427_validation_subsets import construct, read_evidence

TOL = 2e-5
DECISION = "persistent_training_objective_opposition"
NO_OPPOSITION = "no_persistent_training_objective_opposition"
CHECKPOINTS = {
    "initializer": (
        "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
        "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c",
    ),
    "adam_a16": (
        "docs/data/seed435-adam-direction-screen/A-final.npz",
        "afb164603b34d0449f463f3c8e475f9bb4408e7a704c031a6f000572e6c4e9c7",
    ),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(message: str) -> None:
    raise ValueError(message)


def _vector(record: dict[str, Any], key: str, size: int, archive: Any) -> np.ndarray:
    try:
        payload = record["vector_evidence"][key]
        value = np.asarray(archive[payload["archive_key"]], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"vector_missing_or_invalid:{key}") from error
    if value.shape != (size,) or not np.isfinite(value).all():
        fail(f"vector_shape_or_finiteness:{key}")
    digest = hashlib.sha256(value.astype("<f8").tobytes()).hexdigest()
    if digest != payload.get("sha256"):
        fail(f"vector_digest_mismatch:{key}")
    return value


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    data = root / "docs/data/seed437-training-unseen-gradient-alignment"
    manifest_path, result_path = data / "manifest.json", data / "results.json"
    manifest = json.loads(manifest_path.read_text())
    result = json.loads(result_path.read_text())
    if result.get("manifest_sha256") != sha(manifest_path):
        fail("result_manifest_binding_mismatch")
    for rel, digest in manifest["execution_sources"].items():
        if sha(root / rel) != digest:
            fail(f"execution_source_binding_mismatch:{rel}")
    if (
        sha(root / manifest["registration"]["path"])
        != manifest["registration"]["sha256"]
    ):
        fail("registration_binding_mismatch")
    split = manifest["split"]
    if sha(root / split["path"]) != split["sha256"]:
        fail("split_binding_mismatch")
    for source in manifest["sources"]:
        if sha(root / source["snapshot"]) != source["snapshot_sha256"]:
            fail(f"source_snapshot_mismatch:{source['name']}")
    reconstruction = manifest["reconstruction_evidence"]
    if (
        sha(root / reconstruction["row_accounting_path"])
        != reconstruction["row_accounting_sha256"]
    ):
        fail("row_accounting_binding_mismatch")
    if reconstruction["membership_reconstructed_from_row_accounting"] is not True:
        fail("membership_reconstruction_not_registered")
    for name, (relative, digest) in CHECKPOINTS.items():
        if (
            sha(root / relative) != digest
            or manifest["checkpoints"][name]["sha256"] != digest
        ):
            fail(f"checkpoint_binding_mismatch:{name}")
    member_path = root / manifest["membership"]["path"]
    if sha(member_path) != manifest["membership"]["sha256"]:
        fail("membership_hash_mismatch")
    with gzip.open(member_path, "rt", encoding="utf-8") as stream:
        members = [json.loads(line) for line in stream]
    accounting = read_evidence(root / reconstruction["row_accounting_path"])
    rebuilt_membership, _counts = construct(accounting)
    if members != rebuilt_membership:
        fail("membership_reconstruction_mismatch")
    selected = [
        row
        for row in members
        if row["subset"] == "unseen" and row["active_stones"] > 32
    ]
    if (
        len(selected) != manifest["membership"]["selected_exposures"]
        or len({r["input_identity"] for r in selected})
        != manifest["membership"]["exact_input_identities"]
    ):
        fail("membership_population_mismatch")
    if (
        result["checkpoint_immutability"]["before"]
        != result["checkpoint_immutability"]["after"]
    ):
        fail("checkpoint_immutability_mismatch")
    if not result["checkpoint_immutability"]["unchanged"]:
        fail("checkpoint_mutation_reported")
    vector_archive = result["vector_archive"]
    archive_path = data / vector_archive["path"]
    if sha(archive_path) != vector_archive["sha256"]:
        fail("vector_archive_hash_mismatch")
    archive = np.load(archive_path, allow_pickle=False)
    if len(archive.files) != vector_archive["arrays"]:
        fail("vector_archive_array_count_mismatch")
    cosine_values: list[float] = []
    reconstructed: dict[str, Any] = {}
    for name, record in result["checkpoints"].items():
        size = int(record["parameter_count"])
        names = record["parameter_names"]
        if len(names) != len(set(names)):
            fail(f"duplicate_parameter_names:{name}")
        groups = record["parameter_group_indices"]
        flat = [i for ids in groups.values() for i in ids]
        if sorted(flat) != list(range(size)):
            fail(f"parameter_accounting_invalid:{name}")
        declared_layout = manifest["model"]["flat_parameter_layout"]
        if manifest["model"]["parameter_count"] != size:
            fail(f"manifest_parameter_count_mismatch:{name}")
        for group, ids in groups.items():
            expected_indexes = [
                index
                for entry in declared_layout
                if entry["group"] == group
                for index in range(entry["start"], entry["stop"])
            ]
            if ids != expected_indexes:
                fail(f"parameter_group_layout_mismatch:{name}:{group}")
        vectors = {
            key: _vector(record, key, size, archive)
            for key in record["vector_evidence"]
        }
        metrics: dict[str, Any] = {}
        dots: dict[str, Any] = {}
        for weighting in ("exposure", "equal_input"):
            u = vectors[f"U_{weighting}"]
            metrics[weighting], dots[weighting] = {}, {}
            for objective in ("P", "0.3V", "T"):
                g = vectors[objective]
                dot = float(np.dot(g, u))
                gn, un = float(np.linalg.norm(g)), float(np.linalg.norm(u))
                cos = dot / (gn * un) if gn and un else None
                unit = -dot / gn if gn else None
                saved = record["metrics"][weighting][objective]
                for key, value in (
                    ("dot", dot),
                    ("cosine", cos),
                    ("left_norm", gn),
                    ("right_norm", un),
                    ("unit_descent_ce_change", unit),
                ):
                    expected = saved[key]
                    if value is None:
                        if expected is not None:
                            fail(
                                f"metric_zero_norm_mismatch:{name}:{weighting}:{objective}:{key}"
                            )
                    elif not np.isclose(value, expected, atol=TOL, rtol=TOL):
                        fail(
                            f"metric_recomputation_mismatch:{name}:{weighting}:{objective}:{key}"
                        )
                metrics[weighting][objective] = {
                    "dot": dot,
                    "cosine": cos,
                    "unit_descent_ce_change": unit,
                }
                if objective == "T":
                    cosine_values.append(cos if cos is not None else 0.0)
                detail = record["dot_contributions"][f"{weighting}/{objective}"]
                dots[weighting][objective] = detail
                for axis, values in detail.items():
                    if not np.isclose(sum(values.values()), dot, atol=TOL, rtol=TOL):
                        fail(
                            f"dot_decomposition_mismatch:{name}:{weighting}:{objective}:{axis}"
                        )
            if not np.allclose(
                vectors["T"], vectors["P"] + vectors["0.3V"], atol=TOL, rtol=TOL
            ):
                fail(f"combined_vector_mismatch:{name}")
            for objective in ("P", "0.3V", "T"):
                for axis in ("source", "bucket"):
                    ids = record["decomposition"][objective][axis]
                    pieces = [vectors[f"{objective}__{axis}__{part}"] for part in ids]
                    if not np.allclose(
                        np.sum(pieces, axis=0), vectors[objective], atol=TOL, rtol=TOL
                    ):
                        fail(
                            f"gradient_decomposition_mismatch:{name}:{objective}:{axis}"
                        )
            buckets = record["decomposition"][f"U_{weighting}"]["bucket"]
            validation_pieces = [
                vectors[f"U_{weighting}__bucket__{bucket}"] for bucket in buckets
            ]
            if not np.allclose(
                np.sum(validation_pieces, axis=0),
                vectors[f"U_{weighting}"],
                atol=TOL,
                rtol=TOL,
            ):
                fail(f"validation_gradient_decomposition_mismatch:{name}:{weighting}")
        reconstructed[name] = metrics
    classification = (
        DECISION
        if len(cosine_values) == 4 and all(value <= -0.05 for value in cosine_values)
        else NO_OPPOSITION
    )
    if result.get("classification") != classification:
        fail("classification_mismatch")
    archive.close()
    return {
        "status": "valid",
        "classification": classification,
        "checkpoint_count": len(reconstructed),
        "membership_exposures": len(selected),
        "membership_identities": len({r["input_identity"] for r in selected}),
        "metrics_recomputed": True,
        "decompositions_recomputed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
