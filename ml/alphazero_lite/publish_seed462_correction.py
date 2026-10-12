"""Append an independently verified seed462 correction publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import verify_seed462_correction as verifier

SOURCES = (
    "ml/alphazero_lite/verify_seed462_correction.py",
    "ml/alphazero_lite/publish_seed462_correction.py",
    "ml/alphazero_lite/test_seed462_correction.py",
)


def publish(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / verifier.CORRECTION
    if out.exists():
        raise ValueError("correction_publication_already_exists")
    result = verifier.verify(root)
    out.mkdir(parents=True)
    source_hashes = {name: verifier.sha(root / name) for name in SOURCES}
    report = {
        "schema": "seed462-correction-report-v1",
        "timing": "post_execution_append_only_correction",
        "classification": result["classification"],
        "original_bindings": result["original_bindings"],
        "original_publication_preservation": result["preservation"],
        "trajectory": result["trajectory"],
        "control": result["control"],
        "endpoint": result["endpoint"],
        "accounting": result["accounting"],
        "objective_erratum": {
            "original_label": "full_training_objectives",
            "actual_estimand": "all-replay occurrences including validation occurrences",
            "corrected_estimand": "expanded training occurrences replay[train_positions] only",
            "value_term": "mean Huber over expanded training occurrences only",
            "reduction_schedule": "float32 model inference in archived occurrence order; objective CE in consecutive 2048-row compact-array batches; float64 weighted-average accumulator; NumPy float64 mean for Huber values",
            "formal_decision": result["classification"],
        },
        "correction_sources_sha256": source_hashes,
    }
    report_path = out / "correction-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema": "seed462-correction-receipt-v1",
        "timing": "post_execution",
        "classification": result["classification"],
        "original_registration_sha256": result["original_bindings"][
            "registration_sha256"
        ],
        "original_recovery_receipt_sha256": result["original_bindings"][
            "recovery_receipt_sha256"
        ],
        "correction_sources_sha256": source_hashes,
        "correction_files_sha256": {
            "correction-report.json": verifier.sha(report_path)
        },
    }
    (out / "correction-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(publish(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
