"""Portable, read-only verifier for the seed422 accounting correction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed422_corrected_analysis import analyze
from ml.alphazero_lite.verify_seed422_adam_memory import DATA, verify as verify_original


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify() -> dict[str, Any]:
    """Validate unchanged evidence and the separately published correction."""
    original = verify_original()
    ledger_path = DATA / "outcome-ledger.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    result = analyze(rows)
    published_original = read_json(DATA / "analysis.json")
    for field in (
        "paired_delta",
        "paired_delta_ci95",
        "B_opening_cluster_score_ci95",
        "bootstrap_samples",
        "bootstrap_seed",
        "cluster_count",
        "opening_matrix",
        "thresholds",
        "decision",
    ):
        if result[field] != published_original[field]:
            raise ValueError(f"primary_analysis_changed:{field}")
    for lane in ("A", "B"):
        if result["lanes"][lane]["score"] != published_original["lanes"][lane]["score"]:
            raise ValueError(f"lane_score_changed:{lane}")
    correction_path = DATA / "corrected-analysis.json"
    if read_json(correction_path) != result:
        raise ValueError("corrected_analysis_mismatch")
    for lane, expected in {
        "A": {"0": 0.5322265625, "1": 0.43359375},
        "B": {"0": 0.533203125, "1": 0.4658203125},
    }.items():
        if result["lanes"][lane]["seat_scores"] != expected:
            raise ValueError(f"corrected_seat_score_mismatch:{lane}")
    amendment = read_json(DATA / "execution-accounting-amendment.json")
    amendment_path = DATA / "execution-accounting-amendment.json"
    if amendment.get("registration_sha256") != sha256(DATA / "registration.json"):
        raise ValueError("amendment_registration_identity_mismatch")
    recovery = amendment["supplemental_execution_source"]
    if sha256(DATA / recovery["snapshot"]) != recovery["snapshot_sha256"]:
        raise ValueError("amendment_recovery_snapshot_hash_mismatch")
    if not amendment_path.is_file():
        raise ValueError("accounting_amendment_missing")
    receipt = read_json(DATA / "correction-receipt.json")
    for path, digest in receipt["original_evidence_sha256"].items():
        if sha256(DATA / path) != digest:
            raise ValueError(f"original_evidence_hash_mismatch:{path}")
    for relative, digest in receipt["registered_source_sha256"].items():
        snapshot = DATA / "execution-source-snapshots" / relative
        if sha256(snapshot) != digest:
            raise ValueError(f"registered_source_snapshot_hash_mismatch:{relative}")
    for path, digest in receipt["corrected_artifacts"].items():
        if sha256(DATA / path) != digest:
            raise ValueError(f"corrected_artifact_hash_mismatch:{path}")
    for path, digest in receipt["correction_modules"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"correction_module_hash_mismatch:{path}")
    return {
        "verified": True,
        "original_evidence_verified": original["verified"],
        "seat_counts": {lane: {"0": 512, "1": 512} for lane in ("A", "B")},
        "seat_scores": {
            lane: result["lanes"][lane]["seat_scores"] for lane in ("A", "B")
        },
        "primary_estimates_unchanged": True,
        "decision": result["decision"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
