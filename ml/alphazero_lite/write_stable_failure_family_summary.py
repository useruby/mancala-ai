#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_opening_capture_family_row(row: dict) -> bool:
    return (
        isinstance(row.get("id"), str)
        and row["id"].startswith("capture_available-")
        and row.get("bucket") == "capture_available"
        and row.get("phase") == "opening"
        and row.get("legal_moves") == [0, 1, 2, 3, 4]
        and not row.get("reference_unstable")
        and isinstance(row.get("reference_move"), int)
        and not isinstance(row.get("reference_move"), bool)
    )


def _regret(row: dict) -> float:
    raw_regret = row.get("regret")
    if raw_regret is None:
        return 0.0
    if isinstance(raw_regret, bool):
        raise ValueError(f"{row.get('id', '<unknown>')} has invalid regret")
    try:
        regret = float(raw_regret)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{row.get('id', '<unknown>')} has invalid regret") from error
    if not math.isfinite(regret):
        raise ValueError(f"{row.get('id', '<unknown>')} has invalid regret")
    return regret


def _average_regret(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return round(sum(_regret(row) for row in rows) / len(rows), 4)


def _blunder_ids(rows: list[dict]) -> list[str]:
    return [row["id"] for row in rows if _regret(row) >= 0.20]


def build_summary(*, candidate_forensics: dict, opening_family_report: dict) -> dict:
    challenger_rows = candidate_forensics["systems"]["challenger"]["rows"]
    capture_rows = [
        row for row in challenger_rows if _is_opening_capture_family_row(row)
    ]
    family_rows_by_id = {
        row["id"]: row
        for row in opening_family_report.get("rows", [])
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }

    search_flipped_ids = []
    for row in capture_rows:
        family_row = family_rows_by_id.get(row["id"])
        if family_row is None:
            continue
        prior = family_row.get("candidate_prior_summary", {})
        searched = family_row.get("candidate_searched_summary", {})
        reference_move = row.get("reference_move")
        if (
            prior.get("selected_move") == reference_move
            and searched.get("selected_move") != reference_move
        ):
            search_flipped_ids.append(row["id"])

    high_imbalance_rows = [
        row
        for row in challenger_rows
        if row.get("bucket") == "high_imbalance"
        and not row.get("reference_unstable")
        and isinstance(row.get("reference_move"), int)
        and not isinstance(row.get("reference_move"), bool)
    ]
    capture_blunder_ids = _blunder_ids(capture_rows)
    high_imbalance_blunder_ids = _blunder_ids(high_imbalance_rows)

    return {
        "schema": "azlite_stable_failure_family_summary_v1",
        "capture_available": {
            "tracked_rows": len(capture_rows),
            "average_regret": _average_regret(capture_rows),
            "blunder_rate_0_20": round(len(capture_blunder_ids) / len(capture_rows), 4)
            if capture_rows
            else 0.0,
            "blunder_ids": capture_blunder_ids,
            "search_flipped_rows": len(search_flipped_ids),
            "search_flipped_ids": search_flipped_ids,
            "missing_opening_family_rows": sorted(
                set(row["id"] for row in capture_rows) - set(family_rows_by_id)
            ),
        },
        "high_imbalance": {
            "stable_rows": len(high_imbalance_rows),
            "average_regret": _average_regret(high_imbalance_rows),
            "blunder_rate_0_20": round(
                len(high_imbalance_blunder_ids) / len(high_imbalance_rows), 4
            )
            if high_imbalance_rows
            else 0.0,
            "blunder_ids": high_imbalance_blunder_ids,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-forensics", required=True)
    parser.add_argument("--opening-family-report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = build_summary(
        candidate_forensics=load_json(Path(args.candidate_forensics)),
        opening_family_report=load_json(Path(args.opening_family_report)),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
