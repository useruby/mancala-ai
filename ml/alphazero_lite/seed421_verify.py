"""Artifact-independent public verifier for frozen seed421 evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ml.alphazero_lite.seed421_analysis import analyze, validate_ledger


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(data_dir: Path) -> dict:
    reg_path = data_dir / "registration.json"
    registration = json.loads(reg_path.read_text(encoding="utf-8"))
    if registration.get("schema") != "seed421-memoization-registration-v1":
        raise ValueError("registration_schema_mismatch")
    for relative, expected in registration["execution_source_hashes"].items():
        if sha(data_dir / "execution-source-snapshots" / relative) != expected:
            raise ValueError(f"execution_snapshot_hash_mismatch:{relative}")
    cohort_path = data_dir / "cohort.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if sha(cohort_path) != registration["cohort_sha256"]:
        raise ValueError("cohort_hash_mismatch")
    old_registration = (
        data_dir.parent / "seed420-artifact-evaluator-memoization" / "registration.json"
    )
    if sha(old_registration) != registration["seed420_registration_sha256"]:
        raise ValueError("seed420_registration_binding_mismatch")
    ledger_path = data_dir / "benchmark-ledger.jsonl"
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    registration_sha = sha(reg_path)
    validate_ledger(rows, cohort, registration_sha, complete=True)
    report = analyze(rows, cohort, registration_sha)
    published = json.loads((data_dir / "analysis.json").read_text(encoding="utf-8"))
    expected = {
        **report,
        "ledger_sha256": sha(ledger_path),
        "registration_sha256": registration_sha,
    }
    if published != expected:
        raise ValueError("published_analysis_mismatch")
    return {
        "valid": True,
        "cohort_roots": len(cohort),
        "ledger_pairs": len(rows),
        "searches": report["search_count"],
        "decision": report["decision"],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    print(
        json.dumps(
            verify(root / "docs/data/seed421-memoization-timing-correction"),
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
