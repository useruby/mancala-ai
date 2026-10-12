"""Create a separate post-execution recovery receipt for seed462."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.verify_seed462_recovery import verify_and_publish


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(root: Path) -> dict[str, Any]:
    root = root.resolve()
    result = verify_and_publish(root)
    out = root / "docs/data/seed462-historical-policy-removal"
    recovery = root / "docs/data/seed462-historical-policy-removal-recovery"
    registration = json.loads((out / "registration.json").read_text())
    restoration = json.loads((recovery / "source-restoration-report.json").read_text())
    baseline = json.loads((root / ".tmp/seed462-original-inventory.json").read_text())
    preservation = {}
    for relative, record in baseline.items():
        path = root / relative
        preservation[relative] = {
            "sha256_unchanged": sha(path) == record["sha256"],
            "mtime_unchanged": path.stat().st_mtime_ns == record["mtime_ns"],
        }
    if not all(
        item["sha256_unchanged"] and item["mtime_unchanged"]
        for item in preservation.values()
    ):
        raise ValueError("original_publication_modified")
    report = {
        "schema": "seed462-retrospective-recovery-report-v1",
        "timing": "post_execution_reconstruction_and_publication",
        "original_attempt_status": "invalid_unverified_attempt_preserved",
        "recovery_status": result,
        "registration_sha256": sha(out / "registration.json"),
        "registered_sources": registration["source_sha256"],
        "registered_inputs": registration["input_sha256"],
        "original_attempt_archives": {
            name: sha(out / name)
            for name in (
                "C-steps.npz",
                "F-steps.npz",
                "C-trajectory.json",
                "F-trajectory.json",
                "C-verified.json",
                "census.json",
            )
        },
        "source_restoration": restoration,
        "prospective_protocol_assessment": {
            "objective_and_mask": "bound by registration formulas and registered runner snapshot; fresh is reconstructed provenance.source == fresh",
            "minibatches_and_inputs": "hash-bound and reconstructed from registered frozen permutation/split/source snapshots",
            "cohort_estimands": "not fully prospectively specified: pooled/fresh >32 membership filters and exposure/equal-input weighting definitions are not enumerated in registration",
            "decision": "numeric gate arithmetic is reported retrospectively, but no valid advance/close classification is made because the gate estimands were not fully preregistered",
        },
        "original_publication_unchanged": preservation,
    }
    report_path = recovery / "recovery-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    source_names = (
        "ml/alphazero_lite/verify_seed462_recovery.py",
        "ml/alphazero_lite/publish_seed462_recovery.py",
        "ml/alphazero_lite/test_seed462_recovery.py",
    )
    evidence_names = [
        p.name
        for p in recovery.iterdir()
        if p.is_file() and p.name != "recovery-receipt.json"
    ]
    receipt = {
        "schema": "seed462-recovery-receipt-v1",
        "timing": "post_execution",
        "registration_sha256": sha(out / "registration.json"),
        "original_attempt_sha256": {
            name: sha(out / name)
            for name in (
                "C-steps.npz",
                "F-steps.npz",
                "C-trajectory.json",
                "F-trajectory.json",
                "census.json",
            )
        },
        "recovery_sources_sha256": {name: sha(root / name) for name in source_names},
        "recovery_files_sha256": {
            name: sha(recovery / name) for name in sorted(evidence_names)
        },
        "original_attempt_preservation": "validated_sha256_and_mtime",
        "classification": "no_experiment_decision_protocol_estimands_incomplete",
    }
    (recovery / "recovery-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def verify_receipt(root: Path) -> dict[str, Any]:
    """Check bound hashes and rerun semantic reconstruction from source evidence."""
    root = root.resolve()
    recovery = root / "docs/data/seed462-historical-policy-removal-recovery"
    receipt = json.loads((recovery / "recovery-receipt.json").read_text())
    for name, expected in receipt["recovery_sources_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError(f"recovery_source_hash_mismatch:{name}")
    semantic = verify_and_publish(root)
    for name, expected in receipt["recovery_files_sha256"].items():
        if sha(recovery / name) != expected:
            raise ValueError(f"recovery_evidence_semantic_or_hash_mismatch:{name}")
    return {"status": "valid", "semantic": semantic}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_receipt(args.root) if args.verify_only else publish(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
