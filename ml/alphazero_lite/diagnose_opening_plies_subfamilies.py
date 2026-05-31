#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.kalah_rules import KalahGame

FORENSIC_SCHEMA = "azlite_forensic_suite_v1"
REPORT_SCHEMA = "azlite_opening_plies_subfamily_diagnostic_v1"
OPENING_BUCKET = "opening_plies_1_8"
SUBFAMILY_EXTRA_TURN_OVERBIAS = "opening_extra_turn_overbias"
SUBFAMILY_MISSED_EXTRA_TURN = "opening_missed_extra_turn_continuation"
SUBFAMILY_EDGE_MOVE_5 = "opening_edge_move_5_preference"
SUBFAMILY_OTHER = "opening_other_mismatch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forensics", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def current_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        raise ValueError("forensics report must be a JSON object")
    if report.get("schema") != FORENSIC_SCHEMA:
        raise ValueError(f"forensics report must use schema {FORENSIC_SCHEMA}")
    systems = report.get("systems")
    if not isinstance(systems, dict):
        raise ValueError("forensics report must include systems object")
    current = systems.get("current")
    if not isinstance(current, dict):
        raise ValueError("forensics report must include systems.current object")
    rows = current.get("rows")
    if not isinstance(rows, list):
        raise ValueError("forensics report must include systems.current.rows list")
    return rows


def move_grants_extra_turn(state: dict[str, Any], move: int) -> bool:
    game = KalahGame.from_state(state)
    player = game.current_player
    if not game.move(game.pit_index(move)):
        raise ValueError(f"illegal move {move} for state {state}")
    return game.current_player == player and not game.over()


def classify_opening_failure(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("bucket") != OPENING_BUCKET:
        return None

    reference_move = row.get("reference_move")
    selected_move = row.get("selected_move")
    if (
        isinstance(reference_move, bool)
        or not isinstance(reference_move, int)
        or isinstance(selected_move, bool)
        or not isinstance(selected_move, int)
        or selected_move == reference_move
    ):
        return None

    state = row.get("state")
    if not isinstance(state, dict):
        raise ValueError("opening row must include state object")

    reference_extra_turn = move_grants_extra_turn(state, reference_move)
    selected_extra_turn = move_grants_extra_turn(state, selected_move)

    if reference_extra_turn and not selected_extra_turn:
        subfamily = SUBFAMILY_MISSED_EXTRA_TURN
    elif not reference_extra_turn and selected_extra_turn:
        subfamily = SUBFAMILY_EXTRA_TURN_OVERBIAS
    elif selected_move == 5 and reference_move != 5:
        subfamily = SUBFAMILY_EDGE_MOVE_5
    else:
        subfamily = SUBFAMILY_OTHER

    return {
        "row_id": row.get("id"),
        "reference_move": reference_move,
        "selected_move": selected_move,
        "regret": float(row.get("regret", 0.0)),
        "value_error": float(row.get("value_error", 0.0)),
        "reference_extra_turn": reference_extra_turn,
        "selected_extra_turn": selected_extra_turn,
        "subfamily": subfamily,
    }


def build_report(forensics: dict[str, Any], *, source_path: str) -> dict[str, Any]:
    rows = current_rows(forensics)
    opening_rows = [row for row in rows if row.get("bucket") == OPENING_BUCKET]
    failures = []
    for row in opening_rows:
        classified = classify_opening_failure(row)
        if classified is not None:
            failures.append(classified)

    subfamily_counts = Counter(row["subfamily"] for row in failures)
    move_pair_counts = Counter(
        (row["reference_move"], row["selected_move"]) for row in failures
    )

    dominant_subfamilies = [
        {"subfamily": name, "count": count}
        for name, count in sorted(
            subfamily_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    move_pairs = [
        {
            "reference_move": reference_move,
            "selected_move": selected_move,
            "count": count,
        }
        for (reference_move, selected_move), count in sorted(
            move_pair_counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
    ]

    return {
        "schema": REPORT_SCHEMA,
        "source_report": source_path,
        "opening_row_count": len(opening_rows),
        "opening_failure_count": len(failures),
        "dominant_subfamilies": dominant_subfamilies,
        "move_pair_counts": move_pairs,
        "rows": failures,
    }


def main() -> int:
    args = parse_args()
    report = build_report(
        load_json(Path(args.forensics)),
        source_path=args.forensics,
    )
    write_json(Path(args.out), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
