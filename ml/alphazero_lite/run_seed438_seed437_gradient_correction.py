"""Freeze and execute the append-only seed438 correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import run_seed437_training_unseen_gradient_alignment as legacy
from ml.alphazero_lite.seed438_gradient_correction import flat_grad

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed438-seed437-gradient-correction"
PROTOCOL = OUT / "protocol.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(root: Path) -> dict[str, Any]:
    """Bind the unchanged seed437 protocol and correction source before gradients."""
    original = json.loads(
        (
            root / "docs/data/seed437-training-unseen-gradient-alignment/manifest.json"
        ).read_text()
    )
    protocol = {
        "schema": "seed438-correction-protocol-v1",
        "status": "frozen_before_corrected_gradient_execution",
        "observation": "This correction follows observation of seed437 original results.",
        "original_manifest_sha256": sha(
            root / "docs/data/seed437-training-unseen-gradient-alignment/manifest.json"
        ),
        "original_vectors_sha256": sha(
            root
            / "docs/data/seed437-training-unseen-gradient-alignment/gradient-vectors.npz"
        ),
        "original_results_sha256": sha(
            root / "docs/data/seed437-training-unseen-gradient-alignment/results.json"
        ),
        "bound_inputs": {
            "registration": original["registration"],
            "sources": original["sources"],
            "split": original["split"],
            "membership": original["membership"],
            "reconstruction_evidence": original["reconstruction_evidence"],
            "checkpoints": original["checkpoints"],
            "model": original["model"],
            "denominators": original["denominators"],
            "counts": original["counts"],
        },
        "corrected_sources": {
            path: sha(root / path)
            for path in (
                "ml/alphazero_lite/seed438_gradient_correction.py",
                "ml/alphazero_lite/run_seed438_seed437_gradient_correction.py",
                "ml/alphazero_lite/verify_seed438_seed437_gradient_correction.py",
                "ml/alphazero_lite/test_seed438_gradient_correction.py",
            )
        },
        "value_gradient": "Compute unweighted V; apply 0.3 once in row weights; T=P+0.3V.",
        "decision_rule": original["decision_rule"],
        "tolerance": {"absolute": 2e-5, "relative": 2e-5},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    if PROTOCOL.exists():
        raise ValueError("correction_protocol_is_immutable")
    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return protocol


def execute(root: Path) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text())
    for path, digest in protocol["corrected_sources"].items():
        if sha(root / path) != digest:
            raise ValueError(f"correction_source_binding_mismatch:{path}")
    original_dir = root / "docs/data/seed437-training-unseen-gradient-alignment"
    for name, field in (
        ("manifest.json", "original_manifest_sha256"),
        ("gradient-vectors.npz", "original_vectors_sha256"),
        ("results.json", "original_results_sha256"),
    ):
        if sha(original_dir / name) != protocol[field]:
            raise ValueError(f"original_evidence_binding_mismatch:{name}")
    root = root.resolve()
    bound = protocol["bound_inputs"]
    evidence_paths = [
        bound["registration"]["path"],
        bound["split"]["path"],
        bound["membership"]["path"],
        bound["reconstruction_evidence"]["row_accounting_path"],
        *(source["snapshot"] for source in bound["sources"]),
        *(checkpoint["path"] for checkpoint in bound["checkpoints"].values()),
        "docs/data/seed437-training-unseen-gradient-alignment/manifest.json",
        "docs/data/seed437-training-unseen-gradient-alignment/gradient-vectors.npz",
        "docs/data/seed437-training-unseen-gradient-alignment/results.json",
    ]
    evidence_before = {path: sha(root / path) for path in evidence_paths}
    for source, binding in zip(bound["sources"], bound["sources"], strict=True):
        if sha(root / binding["snapshot"]) != source["snapshot_sha256"]:
            raise ValueError(f"source_binding_mismatch:{source['name']}")
    if (
        evidence_before[bound["registration"]["path"]]
        != bound["registration"]["sha256"]
    ):
        raise ValueError("registration_binding_mismatch")
    if evidence_before[bound["split"]["path"]] != bound["split"]["sha256"]:
        raise ValueError("split_binding_mismatch")
    if evidence_before[bound["membership"]["path"]] != bound["membership"]["sha256"]:
        raise ValueError("membership_binding_mismatch")
    if (
        evidence_before[bound["reconstruction_evidence"]["row_accounting_path"]]
        != bound["reconstruction_evidence"]["row_accounting_sha256"]
    ):
        raise ValueError("row_accounting_binding_mismatch")
    legacy.ROOT = root.resolve()
    legacy.OUT = original_dir
    legacy.DATA = root / "docs/data/seed426-canonical-overlap"
    legacy.REGISTRATION = (
        root / "docs/data/seed416-policy-target-softening/registration-v3.json"
    )
    legacy.INIT = root / protocol["bound_inputs"]["checkpoints"]["initializer"]["path"]
    legacy.ADAM = root / protocol["bound_inputs"]["checkpoints"]["adam_a16"]["path"]
    legacy.CHECKPOINTS = {
        name: (root / binding["path"], binding["sha256"])
        for name, binding in protocol["bound_inputs"]["checkpoints"].items()
    }
    (root / ".tmp").mkdir(exist_ok=True)
    arrays = legacy.prepare_arrays()
    legacy.flat_grad = flat_grad
    checkpoints = {
        name: legacy.run_checkpoint(name, path, digest, arrays, 512)
        for name, (path, digest) in legacy.CHECKPOINTS.items()
    }
    vector_arrays = {
        key: value
        for record in checkpoints.values()
        for key, value in record.pop("_vector_arrays").items()
    }
    vector_path = OUT / "corrected-gradient-vectors.npz"
    np.savez_compressed(vector_path, **vector_arrays)
    cosines = [
        checkpoints[checkpoint]["metrics"][weight]["T"]["cosine"]
        for checkpoint in checkpoints
        for weight in ("exposure", "equal_input")
    ]
    classification = (
        legacy.CLASSIFICATION
        if len(cosines) == 4
        and all(value is not None and value <= -0.05 for value in cosines)
        else legacy.NO_OPPOSITION
    )
    result = {
        "schema": "seed438-corrected-results-v1",
        "protocol_sha256": sha(PROTOCOL),
        "classification": classification,
        "strength_claim": False,
        "checkpoint_immutability": {
            name: sha(path) == digest
            for name, (path, digest) in legacy.CHECKPOINTS.items()
        },
        "bound_evidence_immutability": {
            "before": evidence_before,
            "after": {path: sha(root / path) for path in evidence_paths},
        },
        "vector_archive": {
            "path": vector_path.name,
            "sha256": sha(vector_path),
            "arrays": len(vector_arrays),
        },
        "checkpoints": checkpoints,
    }
    if (
        result["bound_evidence_immutability"]["before"]
        != result["bound_evidence_immutability"]["after"]
    ):
        raise ValueError("bound_evidence_immutability_failed")
    (OUT / "corrected-results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(args.root) if args.command == "freeze" else execute(args.root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
