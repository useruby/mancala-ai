"""Bind a completed seed451 reconstruction report after verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch


SOURCES = (
    "ml/alphazero_lite/verify_seed451_gradient_reconstruction.py",
    "ml/alphazero_lite/test_seed451_gradient_reconstruction.py",
    "ml/alphazero_lite/publish_seed451_verification_receipt.py",
    "ml/alphazero_lite/test_seed451_source_policy_alignment.py",
    "ml/alphazero_lite/verify_seed451_source_policy_alignment.py",
    "ml/alphazero_lite/seed451_source_policy_alignment.py",
    "ml/alphazero_lite/train.py",
    "ml/alphazero_lite/verify_seed434_census_source.py",
    "ml/alphazero_lite/verify_seed450_seed449_correction.py",
    "ml/alphazero_lite/seed416_policy_target_softening.py",
)
DEPENDENCY_FILES = ("requirements-dev.txt", "ml/alphazero_lite/requirements.txt")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(root: Path, report_path: Path) -> dict[str, object]:
    root = root.resolve()
    out = root / "docs/data/seed451-source-policy-alignment"
    report = json.loads(report_path.read_text())
    if (
        report.get("status") != "valid"
        or report.get("classification") != "mixed_or_checkpoint_dependent_alignment"
    ):
        raise ValueError("independent_report_not_valid_for_seed451")
    report_target = out / "independent-verification-report.json"
    receipt_target = out / "supplemental-verification-receipt.json"
    if report_target.exists() or receipt_target.exists():
        raise ValueError("supplemental_verification_publication_is_immutable")
    shutil.copyfile(report_path, report_target)
    registration = json.loads((out / "registration.json").read_text())
    historical = {
        **registration["inputs_sha256"],
        **{
            f"docs/data/seed451-source-policy-alignment/{name}": _sha(out / name)
            for name in (
                "registration.json",
                "receipt.json",
                "results.json",
                "results.md",
                "gradient-vectors.npz",
            )
        },
    }
    receipt = {
        "schema": "seed451-post-execution-verification-receipt-v1",
        "status": "independent_gradient_reconstruction_completed_post_execution",
        "observation_order": "original registration and gradient publication predate this supplemental verifier; no publisher rerun occurred",
        "canonical_command": ".venv-azlite/bin/python -m ml.alphazero_lite.verify_seed451_gradient_reconstruction --root CHECKOUT_ROOT",
        "classification": report["classification"],
        "independent_report_sha256": _sha(report_target),
        "verification_sources_sha256": {path: _sha(root / path) for path in SOURCES},
        "dependency_files_sha256": {
            path: _sha(root / path) for path in DEPENDENCY_FILES
        },
        "input_and_historical_evidence_sha256": historical,
        "runtime_dependencies": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "immutable_seed451_registration_sha256": _sha(out / "registration.json"),
        "immutable_seed451_publication_receipt_sha256": _sha(out / "receipt.json"),
        "verification_instructions_sha256": _sha(out / "verification.md"),
    }
    receipt_target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.root, args.report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
