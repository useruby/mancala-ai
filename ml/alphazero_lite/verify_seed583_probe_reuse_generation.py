"""Read-only verification of the frozen #583 runner-comparison evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import validate_varied_opponent_probe_reuse as validation


def evidence_inventory(root: Path) -> tuple[int, str]:
    lines = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n"
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    )
    return len(lines.splitlines()), hashlib.sha256(lines.encode()).hexdigest()


def verify(
    freeze_path: Path,
    evidence_root: Path,
    results_path: Path,
    *,
    expected_freeze_sha256: str,
) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("freeze_sha256") != expected_freeze_sha256:
        raise ValueError("external_validation_freeze_digest_mismatch")
    validation.verify_freeze(freeze)
    report = validation.verify_evidence(freeze, evidence_root)
    execution = json.loads((evidence_root / "execution.json").read_text())
    expected_order = [
        {"pair_id": job["game_specs"][0]["pair_id"], "variant": variant}
        for job, variant in validation.execution_plan(freeze["pairs"])
    ]
    if (
        execution.get("freeze_sha256") != expected_freeze_sha256
        or execution.get("execution_order") != expected_order
    ):
        raise ValueError("execution_order_or_identity_mismatch")
    results = json.loads(results_path.read_text())
    inventory_count, inventory_digest = evidence_inventory(evidence_root)
    inventory = results["raw_evidence"]
    if (
        inventory_count != inventory["files"]
        or inventory_digest
        != inventory["sha256_of_sorted_lines_sha256_two_spaces_relative_path_newline"]
    ):
        raise ValueError("raw_evidence_inventory_mismatch")
    if report["output_mismatches"] or not report["attempts_reconciled"]:
        raise ValueError("semantic_or_attempt_gate_failed")
    if (
        abs(
            report["v2_total_pair_cpu_reduction"]
            - results["v2_total_pair_cpu_reduction"]
        )
        > 1e-12
    ):
        raise ValueError("cpu_summary_mismatch")
    return {
        "status": "verified_read_only",
        "freeze_sha256": expected_freeze_sha256,
        "evidence_files": inventory_count,
        "evidence_inventory_sha256": inventory_digest,
        "pair_count": report["pair_count"],
        "game_count_per_runner": report["game_count_per_runner"],
        "output_mismatches": report["output_mismatches"],
        "attempts_reconciled": report["attempts_reconciled"],
        "accounting": report["accounting"],
        "v2_total_pair_cpu_reduction": report["v2_total_pair_cpu_reduction"],
        "adoption_gate_passed": report["adoption_gate_passed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.freeze,
        args.evidence,
        args.results,
        expected_freeze_sha256=args.expected_freeze_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["adoption_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
