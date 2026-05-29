#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


DEFAULT_GUARD_SOURCE = Path("ml/alphazero_lite/tactical_opening_capture_family_replay.jsonl")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracked-artifact", required=True)
    parser.add_argument("--guard-source", default=str(DEFAULT_GUARD_SOURCE))
    parser.add_argument("--out", required=True)
    parser.add_argument("--out-summary", required=True)
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def build_rows(*, tracked_rows: list[dict], guard_rows: list[dict]) -> tuple[list[dict], dict]:
    preserved_guard_rows = [
        json.loads(json.dumps(row))
        for row in guard_rows
        if str(row.get("replay_role")) != "opening_capture_family"
    ]
    normalized_tracked_rows = []
    for row in tracked_rows:
        copied = json.loads(json.dumps(row))
        normalized_tracked_rows.append(copied)

    rows = [*preserved_guard_rows, *normalized_tracked_rows]
    replay_role_counts = Counter(str(row.get("replay_role", "")) for row in rows)
    teacher_move_counts = Counter(
        int(row["teacher_selected_move"]) for row in rows if "teacher_selected_move" in row
    )
    tracked_replay_role_counts = Counter(
        str(row.get("replay_role", "")) for row in normalized_tracked_rows
    )
    row_ids_by_replay_role: dict[str, list[str]] = {}
    teacher_move_counts_by_replay_role: dict[str, Counter[int]] = {}
    reference_extra_turn_by_row: dict[str, bool] = {}
    for row in normalized_tracked_rows:
        replay_role = str(row.get("replay_role", ""))
        row_id = row.get("source_runs", [{}])[0].get("id")
        if isinstance(row_id, str) and row_id:
            row_ids_by_replay_role.setdefault(replay_role, []).append(row_id)
            if "reference_move_extra_turn_available" in row:
                reference_extra_turn_by_row[row_id] = bool(
                    row["reference_move_extra_turn_available"]
                )
        if "teacher_selected_move" in row:
            teacher_move_counts_by_replay_role.setdefault(replay_role, Counter())[
                int(row["teacher_selected_move"])
            ] += 1

    summary = {
        "schema": "azlite_tracked_opening_capture_family_plus_guards_summary_v1",
        "row_count": len(rows),
        "replay_role_counts": dict(sorted(replay_role_counts.items())),
        "teacher_selected_move_distribution": dict(sorted(teacher_move_counts.items())),
        "tracked_row_ids": [
            row.get("source_runs", [{}])[0].get("id")
            for row in normalized_tracked_rows
        ],
        "tracked_replay_role_counts": dict(sorted(tracked_replay_role_counts.items())),
        "row_ids_by_replay_role": {
            role: row_ids_by_replay_role[role] for role in sorted(row_ids_by_replay_role)
        },
        "teacher_selected_move_distribution_by_replay_role": {
            role: dict(sorted(teacher_move_counts_by_replay_role[role].items()))
            for role in sorted(teacher_move_counts_by_replay_role)
        },
        "reference_extra_turn_by_row": dict(sorted(reference_extra_turn_by_row.items())),
        "guard_row_count": len(preserved_guard_rows),
    }
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tracked_rows = load_jsonl(Path(args.tracked_artifact))
    guard_rows = load_jsonl(Path(args.guard_source))
    rows, summary = build_rows(tracked_rows=tracked_rows, guard_rows=guard_rows)

    out_path = Path(args.out)
    write_jsonl(out_path, rows)
    out_summary_path = Path(args.out_summary)
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    out_summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
