#!/usr/bin/env python3
"""Produce the read-only quality and coverage report for exact forensic shadow mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.forensic_exact_references import (  # noqa: E402
    exact_regret,
    sha256_file,
)
from ml.alphazero_lite.forensic_suite import load_suite  # noqa: E402

SUITE = ROOT / "ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"
V1 = ROOT / "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
OVERRIDES = (
    ROOT
    / "ml/alphazero_lite/fixtures/incumbent_forensic_reference_audited_overrides_v1.json"
)
PR289 = ROOT / "docs/data/alphazero-lite-uniform1200-replay-distribution-audit.json"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return values[round((len(values) - 1) * fraction)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    suite = load_suite(SUITE)
    buckets = {row.id: row.bucket for row in suite}
    exact = json.loads(args.reference.read_text())["rows"]
    old = {row["canonical_state"]: row for row in json.loads(V1.read_text())["rows"]}
    solved = [row for row in exact if row["exact_status"] == "exact_solved"]
    details = []
    for row in solved:
        reference = old[row["canonical_state"]]
        move = reference.get("reference_move")
        if move is None:
            continue
        details.append(
            {
                "id": row["id"],
                "bucket": buckets[row["id"]],
                "reference_move": move,
                "exact_optimal": move in row["exact_optimal_actions"],
                "exact_regret": exact_regret(row, move),
                "multi_optimum": len(row["exact_optimal_actions"]) > 1,
            }
        )
    regrets = sorted(
        float(item["exact_regret"])
        for item in details
        if item["exact_regret"] is not None
    )
    regression_ids = set(
        json.loads(PR289.read_text())["forensics"]["consistent_uniform_regressions"]
    )
    regression_details = [item for item in details if item["id"] in regression_ids]
    coverage = {}
    for bucket in sorted(set(buckets.values())):
        rows = [row for row in exact if buckets[row["id"]] == bucket]
        coverage[bucket] = {
            "exact_covered_rows": sum(
                row["exact_status"] == "exact_solved" for row in rows
            ),
            "unresolved_rows": sum(row["exact_status"] == "unresolved" for row in rows),
            "total": len(rows),
        }
    classification = (
        "exact_forensic_reference_complete"
        if len(solved) == len(exact)
        else "exact_forensic_reference_partial"
    )
    report: dict[str, Any] = {
        "schema": "azlite_exact_forensic_shadow_v1",
        "read_only": {
            "training": False,
            "self_play": False,
            "promotion": False,
            "tablebase_generation": False,
        },
        "baseline_sha256": {
            "forensic_suite": sha256_file(SUITE),
            "current_reference": sha256_file(V1),
            "audited_overrides": sha256_file(OVERRIDES),
            "exact_v2": sha256_file(args.reference),
        },
        "coverage": {
            "overall": {
                "exact_covered_rows": len(solved),
                "unresolved_rows": len(exact) - len(solved),
                "total": len(exact),
            },
            "by_bucket": coverage,
        },
        "reference_quality": {
            "exact_covered_rows": len(solved),
            "unresolved_rows": len(exact) - len(solved),
            "current_reference_move_in_exact_optimal_set": sum(
                item["exact_optimal"] for item in details
            ),
            "exact_single_optimum_agreement": sum(
                item["exact_optimal"] and not item["multi_optimum"] for item in details
            ),
            "current_reference_strictly_suboptimal_rate": sum(
                not item["exact_optimal"] for item in details
            )
            / len(details)
            if details
            else None,
            "exact_multi_optimum_rate": sum(item["multi_optimum"] for item in details)
            / len(details)
            if details
            else None,
            "current_reference_exact_regret": {
                "p50": percentile(regrets, 0.5),
                "p90": percentile(regrets, 0.9),
                "max": max(regrets, default=None),
            },
            "rows": details,
        },
        "pr289_regression_anchors": {
            "registered_rows": len(regression_ids),
            "exact_covered_rows": len(regression_details),
            "unresolved_rows": len(regression_ids) - len(regression_details),
            "current_reference_strictly_suboptimal_rows": sum(
                not item["exact_optimal"] for item in regression_details
            ),
        },
        "pre_registered_status": classification,
        "shadow_exact_forensic_decision": "not_interpretable_on_partial_reference",
        "coverage_interpretation": "opening-only unresolved roots; both critical buckets are exact-complete",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Exact Forensic Shadow Reference",
        "",
        "## Coverage",
        "",
        f"Exact-covered roots: {len(solved)}/{len(exact)}; unresolved: {len(exact) - len(solved)}.",
        "",
        "| Bucket | Exact | Unresolved |",
        "| --- | ---: | ---: |",
    ]
    lines += [
        f"| {bucket} | {value['exact_covered_rows']} | {value['unresolved_rows']} |"
        for bucket, value in coverage.items()
    ]
    quality = report["reference_quality"]
    lines += [
        "",
        "## Frozen MCTS Reference",
        "",
        f"Strictly suboptimal exact-covered reference moves: {sum(not item['exact_optimal'] for item in details)}/{len(details)}. Exact multi-optimum rate: {quality['exact_multi_optimum_rate']!s}.",
        "",
        "## Coverage Interpretation",
        "",
        report["coverage_interpretation"],
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
