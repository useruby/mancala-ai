from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_opening_matrix_row(
    *,
    row_id: str,
    opening_row: dict,
    current_row: dict,
    original_row: dict,
    rebalanced_row: dict,
    original_opening_row: dict,
    rebalanced_opening_row: dict,
) -> dict:
    return {
        "row_id": row_id,
        "bucket": opening_row["bucket"],
        "phase": opening_row["phase"],
        "reference_move": opening_row["reference_move"],
        "current_prior_move": opening_row["current_prior_summary"]["selected_move"],
        "current_searched_move": opening_row["current_searched_summary"]["selected_move"],
        "original_challenger_prior_move": original_opening_row["candidate_prior_summary"]["selected_move"],
        "original_challenger_searched_move": original_opening_row["candidate_searched_summary"]["selected_move"],
        "rebalanced_challenger_prior_move": rebalanced_opening_row["candidate_prior_summary"]["selected_move"],
        "rebalanced_challenger_searched_move": rebalanced_opening_row["candidate_searched_summary"]["selected_move"],
        "teacher_value": current_row["teacher_value"],
        "current_system": {
            "value": current_row["system_value"],
            "value_error": current_row["value_error"],
        },
        "original_challenger_system": {
            "value": original_row["system_value"],
            "value_error": original_row["value_error"],
        },
        "rebalanced_challenger_system": {
            "value": rebalanced_row["system_value"],
            "value_error": rebalanced_row["value_error"],
        },
    }


def classify_mechanism(row: dict) -> str:
    teacher_value = row["teacher_value"]
    rebalanced_value = row["rebalanced_challenger_system"]["value"]
    rebalanced_error = row["rebalanced_challenger_system"]["value_error"]
    rebalanced_move = row["rebalanced_challenger_searched_move"]
    reference_move = row["reference_move"]
    rebalanced_prior_move = row.get("rebalanced_challenger_prior_move")
    original_move = row.get("original_challenger_searched_move")

    if row["phase"] == "late":
        return "persistent_late_game_weakness"

    if rebalanced_prior_move == reference_move and rebalanced_move != reference_move and rebalanced_error <= 0.1:
        return "search_overrides_prior"

    if original_move == reference_move and rebalanced_move != reference_move and rebalanced_error <= 0.1:
        return "search_overrides_prior"

    if teacher_value < 0 and rebalanced_value > 0 and original_move == reference_move and rebalanced_move != reference_move:
        return "value_sign_miscalibration"

    return "mixed"


def choose_next_branch(matrix: list[dict]) -> dict:
    search_rows = [row["row_id"] for row in matrix if row["mechanism"] == "search_overrides_prior"]
    value_rows = [row["row_id"] for row in matrix if row["mechanism"] == "value_sign_miscalibration"]
    endgame_rows = [row["row_id"] for row in matrix if row["mechanism"] == "persistent_late_game_weakness"]

    if search_rows:
        next_branch = "search_interaction_diagnostic"
    elif value_rows:
        next_branch = "value_calibration_diagnostic"
    else:
        next_branch = "endgame_isolation_diagnostic"

    return {
        "next_branch": next_branch,
        "priority_rows": search_rows,
        "followup_rows": value_rows,
        "separate_track_rows": endgame_rows,
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["id"]: row for row in rows}


TARGET_ROW_IDS_BY_BUCKET = {
    "capture_available": {"capture_available-002", "capture_available-003"},
    "high_imbalance": {"high_imbalance-010", "high_imbalance-011", "high_imbalance-019"},
    "incumbent_proxy_disagreement": {"incumbent_proxy_disagreement-031", "incumbent_proxy_disagreement-033"},
    "opening_plies_1_8": {"opening_plies_1_8-057"},
    "sparse_endgame": {"sparse_endgame-009"},
}


def _build_non_opening_matrix_row(
    *,
    row_id: str,
    current_row: dict,
    original_row: dict,
    rebalanced_row: dict,
) -> dict:
    return {
        "row_id": row_id,
        "bucket": current_row["bucket"],
        "phase": current_row["phase"],
        "reference_move": current_row["reference_move"],
        "current_prior_move": None,
        "current_searched_move": current_row["selected_move"],
        "original_challenger_prior_move": None,
        "original_challenger_searched_move": original_row["selected_move"],
        "rebalanced_challenger_prior_move": None,
        "rebalanced_challenger_searched_move": rebalanced_row["selected_move"],
        "teacher_value": current_row["teacher_value"],
        "current_system": {
            "value": current_row["system_value"],
            "value_error": current_row["value_error"],
        },
        "original_challenger_system": {
            "value": original_row["system_value"],
            "value_error": original_row["value_error"],
        },
        "rebalanced_challenger_system": {
            "value": rebalanced_row["system_value"],
            "value_error": rebalanced_row["value_error"],
        },
    }


def build_matrix_from_runs(*, original_run: Path, rebalanced_run: Path) -> list[dict]:
    original_forensics = _load_json(original_run / "final" / "selected_candidate_forensics.json")
    rebalanced_forensics = _load_json(rebalanced_run / "final" / "selected_candidate_forensics.json")
    original_opening_report = _load_json(original_run / "final" / "opening_capture_family_report.json")
    rebalanced_opening_report = _load_json(rebalanced_run / "final" / "opening_capture_family_report.json")

    current_rows = _rows_by_id(original_forensics["systems"]["current"]["rows"])
    original_rows = _rows_by_id(original_forensics["systems"]["challenger"]["rows"])
    rebalanced_rows = _rows_by_id(rebalanced_forensics["systems"]["challenger"]["rows"])
    original_opening_rows = _rows_by_id(original_opening_report["rows"])
    rebalanced_opening_rows = _rows_by_id(rebalanced_opening_report["rows"])

    matrix = []
    for row_id in sorted(current_rows):
        current_row = current_rows[row_id]
        if row_id not in TARGET_ROW_IDS_BY_BUCKET.get(current_row["bucket"], set()):
            continue

        original_row = original_rows[row_id]
        rebalanced_row = rebalanced_rows[row_id]

        if row_id in rebalanced_opening_rows and row_id in original_opening_rows:
            row = build_opening_matrix_row(
                row_id=row_id,
                opening_row=rebalanced_opening_rows[row_id],
                current_row=current_row,
                original_row=original_row,
                rebalanced_row=rebalanced_row,
                original_opening_row=original_opening_rows[row_id],
                rebalanced_opening_row=rebalanced_opening_rows[row_id],
            )
        else:
            row = _build_non_opening_matrix_row(
                row_id=row_id,
                current_row=current_row,
                original_row=original_row,
                rebalanced_row=rebalanced_row,
            )

        row["mechanism"] = classify_mechanism(row)
        if row["mechanism"] != "mixed":
            matrix.append(row)

    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-run", required=True)
    parser.add_argument("--rebalanced-run", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = build_matrix_from_runs(
        original_run=Path(args.original_run),
        rebalanced_run=Path(args.rebalanced_run),
    )
    payload = {
        "matrix": matrix,
        "summary": choose_next_branch(matrix),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
