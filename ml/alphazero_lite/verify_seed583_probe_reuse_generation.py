"""Read-only verification of the frozen #583 runner-comparison evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    expected_evidence_inventory_sha256: str,
) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("freeze_sha256") != expected_freeze_sha256:
        raise ValueError("external_validation_freeze_digest_mismatch")
    validation.verify_freeze(freeze, allow_historical_snapshot=True)
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
        inventory_digest != expected_evidence_inventory_sha256
        or inventory_count != inventory["files"]
        or inventory_digest
        != inventory["sha256_of_sorted_lines_sha256_two_spaces_relative_path_newline"]
    ):
        raise ValueError("raw_evidence_inventory_mismatch")
    if report["output_mismatches"] or not report["attempts_reconciled"]:
        raise ValueError("semantic_or_attempt_gate_failed")
    reconcile_results_report(report, results)
    return {
        "status": "verified_read_only",
        "freeze_sha256": expected_freeze_sha256,
        "evidence_files": inventory_count,
        "evidence_inventory_sha256": inventory_digest,
        "pair_count": report["pair_count"],
        "game_count_per_runner": report["game_count_per_runner"],
        "outputs": report["outputs"],
        "completed_receipts": report["completed_receipts"],
        "verified_checkpoints": report["verified_checkpoints"],
        "unreconciled_attempts": report["unreconciled_attempts"],
        "output_mismatches": report["output_mismatches"],
        "attempts_reconciled": report["attempts_reconciled"],
        "accounting": report["accounting"],
        "v2_total_pair_cpu_reduction": report["v2_total_pair_cpu_reduction"],
        "adoption_gate_passed": report["adoption_gate_passed"],
    }


def reconcile_results_report(report: dict[str, Any], results: dict[str, Any]) -> None:
    """Reconcile all published output, CPU, and gate metrics to checkpoints."""
    if results.get("freeze_sha256") != report.get("freeze_sha256"):
        raise ValueError("results_freeze_digest_mismatch")

    outputs = report["outputs"]
    published_outputs = results["outputs"]
    games = [outputs[variant]["games"] for variant in ("v1", "v2")]
    rows = [outputs[variant]["retained_training_rows"] for variant in ("v1", "v2")]
    moves = [outputs[variant]["trajectory_moves"] for variant in ("v1", "v2")]
    winners = [outputs[variant]["winner_counts_by_player"] for variant in ("v1", "v2")]
    if (
        games != [32, 32]
        or published_outputs.get("games_per_runner") != games[0]
        or published_outputs.get("games_total") != sum(games)
        or rows[0] != rows[1]
        or published_outputs.get("retained_training_rows_per_runner") != rows[0]
        or moves[0] != moves[1]
        or published_outputs.get("total_trajectory_moves_per_runner") != moves[0]
        or winners[0] != winners[1]
        or published_outputs.get("winner_counts_by_player_per_runner") != winners[0]
        or published_outputs.get("trajectory_outcome_training_row_mismatches")
        != len(report["output_mismatches"])
    ):
        raise ValueError("results_output_summary_mismatch")

    attempts = results["attempts"]
    if (
        attempts.get("completed_receipts") != report["completed_receipts"]
        or attempts.get("verified_checkpoints") != report["verified_checkpoints"]
        or attempts.get("unreconciled_attempts") != report["unreconciled_attempts"]
        or not report["attempts_reconciled"]
    ):
        raise ValueError("results_attempt_summary_mismatch")

    for variant in ("v1", "v2"):
        actual = report["accounting"][variant]
        published = results["cpu_time_seconds"][variant]
        for field, ns_field in (
            ("pair_process_cpu", "pair_process_cpu_time_ns"),
            ("reaped_native_child_cpu", "native_child_cpu_time_ns"),
            ("total_pair_cpu", "pair_cpu_time_ns"),
            ("sum_of_per_pair_wall_time", "wall_time_ns"),
        ):
            if not math.isclose(
                published[field],
                actual[ns_field] / 1_000_000_000,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(f"results_cpu_summary_mismatch:{variant}:{field}")

    reduction = report["v2_total_pair_cpu_reduction"]
    if not math.isclose(
        results["v2_total_pair_cpu_reduction"], reduction, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("results_cpu_reduction_mismatch")
    expected_gate = {
        "zero_output_mismatches": not report["output_mismatches"],
        "all_attempts_reconciled": report["attempts_reconciled"]
        and report["unreconciled_attempts"] == 0,
        "at_least_20_percent_lower_total_pair_cpu": reduction >= 0.20,
        "passed": report["adoption_gate_passed"],
    }
    if results["adoption_gate"] != expected_gate:
        raise ValueError("results_adoption_gate_mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-evidence-inventory-sha256", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.freeze,
        args.evidence,
        args.results,
        expected_freeze_sha256=args.expected_freeze_sha256,
        expected_evidence_inventory_sha256=args.expected_evidence_inventory_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["adoption_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
