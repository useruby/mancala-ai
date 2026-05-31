#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


TARGET_FAILURE_STATUS = "fail_corrected_reference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forensic-report", required=True)
    parser.add_argument("--corrected-inventory", required=True)
    parser.add_argument("--out-artifact", required=True)
    parser.add_argument("--out-summary", required=True)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _entropy_from_child_stats(child_stats: list[dict[str, Any]]) -> float:
    total_visits = sum(int(child.get("visits", 0)) for child in child_stats)
    if total_visits <= 0:
        return 0.0
    entropy = 0.0
    for child in child_stats:
        probability = int(child.get("visits", 0)) / total_visits
        if probability > 0.0:
            entropy -= probability * math.log(probability, 2)
    return round(entropy, 4)


def _best_second_gap_from_child_stats(child_stats: list[dict[str, Any]]) -> float:
    win_rates = sorted(
        (float(child.get("win_rate", 0.0)) for child in child_stats), reverse=True
    )
    if len(win_rates) < 2:
        return 0.0
    return round(max(0.0, win_rates[0] - win_rates[1]), 4)


def build_corrected_reference_hard_state_artifact(
    *, forensic_report_path: str | Path, corrected_inventory_path: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    forensic_report = load_json(forensic_report_path)
    corrected_inventory = load_json(corrected_inventory_path)

    if forensic_report.get("schema") != "azlite_forensic_suite_v1":
        raise ValueError("forensic report must use schema azlite_forensic_suite_v1")
    if not isinstance(corrected_inventory, list):
        raise ValueError("corrected inventory must be a JSON list")

    challenger = dict(forensic_report.get("systems", {})).get("current")
    if challenger is None:
        challenger = dict(forensic_report.get("systems", {})).get("challenger")
    if not isinstance(challenger, dict):
        raise ValueError("forensic report must contain a current or challenger system")

    report_rows = challenger.get("rows")
    if not isinstance(report_rows, list):
        raise ValueError("forensic report system rows must be a list")

    allowed_rows = {
        str(row["row_id"]): dict(row)
        for row in corrected_inventory
        if isinstance(row, dict) and row.get("failure_status") == TARGET_FAILURE_STATUS
    }

    filtered_rows: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()

    for row in report_rows:
        if not isinstance(row, dict):
            raise ValueError("forensic report row must be a dictionary")
        row_id = str(row.get("id", ""))
        inventory_row = allowed_rows.get(row_id)
        if inventory_row is None:
            continue

        child_stats = row.get("child_stats")
        normalized_child_stats = child_stats if isinstance(child_stats, list) else []
        filtered_row = dict(row)
        filtered_row["entropy"] = _entropy_from_child_stats(normalized_child_stats)
        filtered_row["best_second_gap"] = _best_second_gap_from_child_stats(
            normalized_child_stats
        )
        filtered_row["corrected_failure_status"] = inventory_row["failure_status"]
        filtered_row["corrected_failure_severity"] = inventory_row["severity"]
        filtered_row["corrected_failure_family"] = inventory_row["family"]
        filtered_row["recommended_use"] = inventory_row["recommended_use"]
        filtered_rows.append(filtered_row)
        family_counts[str(inventory_row["family"])] += 1
        severity_counts[str(inventory_row["severity"])] += 1

    artifact = {
        "schema": "azlite_forensic_suite_v1",
        "kind": "corrected_reference_forensic_suite",
        "source_forensic_report": str(Path(forensic_report_path)),
        "source_corrected_inventory": str(Path(corrected_inventory_path)),
        "systems": {
            "challenger": {
                "artifact_path": challenger.get("artifact_path"),
                "rows": filtered_rows,
            }
        },
    }
    summary = {
        "schema": "azlite_corrected_reference_hard_state_artifact_v1",
        "forensic_report_path": str(Path(forensic_report_path)),
        "corrected_inventory_path": str(Path(corrected_inventory_path)),
        "row_count": len(filtered_rows),
        "family_counts": dict(sorted(family_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
    }
    return artifact, summary


def main() -> None:
    args = parse_args()
    artifact, summary = build_corrected_reference_hard_state_artifact(
        forensic_report_path=args.forensic_report,
        corrected_inventory_path=args.corrected_inventory,
    )
    write_json(args.out_artifact, artifact)
    write_json(args.out_summary, summary)


if __name__ == "__main__":
    main()
