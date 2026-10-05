"""Public verifier for frozen seed420 cohort, pair accounting, and decision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ml.alphazero_lite.seed420_analysis import analyze
from ml.alphazero_lite.seed420_cohort import derive_cohort


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(data_dir: Path) -> dict:
    registration_path = data_dir / "registration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    snapshots = data_dir / "execution-source-snapshots"
    for relative, expected in registration["execution_source_hashes"].items():
        frozen = snapshots / relative
        if _sha(frozen) != expected:
            raise ValueError(f"execution_snapshot_hash_mismatch:{relative}")
    source = registration["source_hashes"]
    input_dir = data_dir / "cohort-inputs"
    suite_path = input_dir / "seed416-openings.jsonl"
    games_path = input_dir / "seed416-outcome-ledger.jsonl"
    if _sha(suite_path) != source["suite"]:
        raise ValueError("cohort_suite_hash_mismatch")
    if _sha(games_path) != source["trajectory_ledger"]:
        raise ValueError("cohort_trajectory_hash_mismatch")
    if (
        _sha(input_dir / "seed416-registration-v3.json")
        != source["seed416_registration"]
    ):
        raise ValueError("seed416_registration_hash_mismatch")
    if (
        _sha(input_dir / "seed416-evaluation-binding.json")
        != source["evaluation_binding"]
    ):
        raise ValueError("seed416_binding_hash_mismatch")
    suite = [json.loads(line) for line in suite_path.read_text().splitlines()]
    games = [json.loads(line) for line in games_path.read_text().splitlines()]
    recomputed = derive_cohort(suite, games)
    cohort_path = data_dir / "cohort.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if recomputed != cohort:
        raise ValueError("cohort_selection_mismatch")
    if _sha(cohort_path) != registration["cohort_sha256"]:
        raise ValueError("cohort_binding_mismatch")
    ledger_path = data_dir / "benchmark-ledger.jsonl"
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    report = analyze(ledger, cohort)
    published = json.loads((data_dir / "analysis.json").read_text(encoding="utf-8"))
    if report != {
        key: value
        for key, value in published.items()
        if key not in {"ledger_sha256", "registration_sha256", "execution_order"}
    }:
        raise ValueError("published_analysis_mismatch")
    if published["ledger_sha256"] != _sha(ledger_path):
        raise ValueError("ledger_hash_mismatch")
    if published["registration_sha256"] != _sha(registration_path):
        raise ValueError("registration_hash_mismatch")
    return {
        "valid": True,
        "cohort_roots": len(cohort),
        "ledger_pairs": len(ledger),
        "searches": report["search_count"],
        "decision": report["decision"],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data = root / "docs/data/seed420-artifact-evaluator-memoization"
    print(json.dumps(verify(data), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
