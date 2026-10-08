"""Independent verifier for the seed438 correction publication."""

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

TOL = 2e-5
DECISION = "persistent_training_objective_opposition"
NO_DECISION = "no_persistent_training_objective_opposition"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    original = root / "docs/data/seed437-training-unseen-gradient-alignment"
    out = root / "docs/data/seed438-seed437-gradient-correction"
    protocol_path, result_path = out / "protocol.json", out / "corrected-results.json"
    protocol, result = (
        json.loads(protocol_path.read_text()),
        json.loads(result_path.read_text()),
    )
    if result["protocol_sha256"] != sha(protocol_path):
        raise ValueError("protocol_binding_mismatch")
    for path, digest in protocol["corrected_sources"].items():
        if sha(root / path) != digest:
            raise ValueError(f"corrected_source_binding_mismatch:{path}")
    if sha(original / "manifest.json") != protocol["original_manifest_sha256"]:
        raise ValueError("original_manifest_changed")
    if sha(original / "gradient-vectors.npz") != protocol["original_vectors_sha256"]:
        raise ValueError("original_vectors_changed")
    if sha(original / "results.json") != protocol["original_results_sha256"]:
        raise ValueError("original_results_changed")
    verify_original(root)
    archive_info = result["vector_archive"]
    archive_path = out / archive_info["path"]
    if sha(archive_path) != archive_info["sha256"]:
        raise ValueError("corrected_archive_hash_mismatch")
    with np.load(archive_path, allow_pickle=False) as archive:
        if len(archive.files) != archive_info["arrays"]:
            raise ValueError("corrected_archive_array_count_mismatch")
        cosine_values = []
        for checkpoint, record in result["checkpoints"].items():
            size = record["parameter_count"]
            vectors = {}
            for name, evidence in record["vector_evidence"].items():
                vector = np.asarray(archive[evidence["archive_key"]], dtype=np.float64)
                if vector.shape != (size,) or not np.isfinite(vector).all():
                    raise ValueError(f"vector_invalid:{checkpoint}:{name}")
                if (
                    hashlib.sha256(vector.astype("<f8").tobytes()).hexdigest()
                    != evidence["sha256"]
                ):
                    raise ValueError(f"vector_digest_mismatch:{checkpoint}:{name}")
                vectors[name] = vector
            if set(vectors) != set(record["vector_evidence"]):
                raise ValueError("vector_set_mismatch")
            if not np.allclose(
                vectors["T"], vectors["P"] + vectors["0.3V"], atol=TOL, rtol=TOL
            ):
                raise ValueError(f"combined_vector_mismatch:{checkpoint}")
            expected = {
                "P": {
                    "source": [
                        "fresh",
                        "generic_bootstrap",
                        "random_teacher",
                        "opening_disagreement",
                        "stability",
                    ],
                    "bucket": ["<=16", "17-21", "22-32", ">32"],
                },
                "0.3V": {
                    "source": [
                        "fresh",
                        "generic_bootstrap",
                        "random_teacher",
                        "opening_disagreement",
                        "stability",
                    ],
                    "bucket": ["<=16", "17-21", "22-32", ">32"],
                },
                "T": {
                    "source": [
                        "fresh",
                        "generic_bootstrap",
                        "random_teacher",
                        "opening_disagreement",
                        "stability",
                    ],
                    "bucket": ["<=16", "17-21", "22-32", ">32"],
                },
            }
            for objective, axes in expected.items():
                for axis, labels in axes.items():
                    if record["decomposition"][objective][axis] != labels:
                        raise ValueError(
                            f"component_set_mismatch:{checkpoint}:{objective}:{axis}"
                        )
                    parts = [
                        vectors[f"{objective}__{axis}__{label}"] for label in labels
                    ]
                    if not np.allclose(
                        np.sum(parts, axis=0), vectors[objective], atol=TOL, rtol=TOL
                    ):
                        raise ValueError(
                            f"component_sum_mismatch:{checkpoint}:{objective}:{axis}"
                        )
            groups = record["parameter_group_indices"]
            if set(groups) != {"shared_trunk", "policy_head", "value_head"} or sorted(
                i for ix in groups.values() for i in ix
            ) != list(range(size)):
                raise ValueError(f"parameter_layout_invalid:{checkpoint}")
            for weighting in ("exposure", "equal_input"):
                u = vectors[f"U_{weighting}"]
                for objective in ("P", "0.3V", "T"):
                    g = vectors[objective]
                    dot = float(np.dot(g, u))
                    metric = record["metrics"][weighting][objective]
                    if not np.isclose(dot, metric["dot"], atol=TOL, rtol=TOL):
                        raise ValueError(
                            f"metric_dot_mismatch:{checkpoint}:{weighting}:{objective}"
                        )
                    details = record["dot_contributions"][f"{weighting}/{objective}"]
                    for axis in ("source", "bucket"):
                        names = expected[objective][axis]
                        for label in names:
                            piece = vectors[f"{objective}__{axis}__{label}"]
                            actual = float(np.dot(u, piece))
                            if not np.isclose(
                                actual, details[axis][label], atol=TOL, rtol=TOL
                            ):
                                raise ValueError(
                                    f"individual_contribution_mismatch:{checkpoint}:{weighting}:{objective}:{axis}:{label}"
                                )
                    for group, indexes in groups.items():
                        actual = float(np.dot(u[indexes], g[indexes]))
                        if not np.isclose(
                            actual,
                            details["parameter_group"][group],
                            atol=TOL,
                            rtol=TOL,
                        ):
                            raise ValueError(
                                f"individual_contribution_mismatch:{checkpoint}:{weighting}:{objective}:parameter_group:{group}"
                            )
                    for axis, parts in details.items():
                        if not np.isclose(sum(parts.values()), dot, atol=TOL, rtol=TOL):
                            raise ValueError(
                                f"contribution_total_mismatch:{checkpoint}:{weighting}:{objective}:{axis}"
                            )
                cosine_values.append(record["metrics"][weighting]["T"]["cosine"])
        classification = (
            DECISION
            if len(cosine_values) == 4
            and all(value is not None and value <= -0.05 for value in cosine_values)
            else NO_DECISION
        )
        if result["classification"] != classification:
            raise ValueError("classification_mismatch")
    return {
        "status": "valid",
        "classification": result["classification"],
        "direct_contributions_recomputed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
