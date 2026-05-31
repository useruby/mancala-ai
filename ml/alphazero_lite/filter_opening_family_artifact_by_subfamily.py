#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


OPENING_SUBFAMILY_DIAGNOSTIC_SCHEMA = "azlite_opening_plies_subfamily_diagnostic_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--artifact-summary", required=True)
    parser.add_argument("--opening-subfamily-diagnostic", required=True)
    parser.add_argument("--subfamily", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--out-summary", required=True)
    return parser.parse_args(argv)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def row_ids_by_subfamily(diagnostic: dict) -> dict[str, list[str]]:
    if diagnostic.get("schema") != OPENING_SUBFAMILY_DIAGNOSTIC_SCHEMA:
        raise SystemExit(
            "opening subfamily diagnostic must use schema "
            f"{OPENING_SUBFAMILY_DIAGNOSTIC_SCHEMA}"
        )
    rows = diagnostic.get("rows")
    if not isinstance(rows, list):
        raise SystemExit("opening subfamily diagnostic must contain rows list")
    grouped: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get("row_id")
        subfamily = row.get("subfamily")
        if isinstance(row_id, str) and isinstance(subfamily, str):
            grouped.setdefault(subfamily, []).append(row_id)
    return {name: sorted(ids) for name, ids in sorted(grouped.items())}


def filter_rows(
    *, artifact_rows: list[dict], selected_row_ids: set[str]
) -> tuple[list[dict], list[str]]:
    kept_rows: list[dict] = []
    kept_opening_row_ids: list[str] = []
    for row in artifact_rows:
        source_runs = row.get("source_runs") or []
        source_row_id = source_runs[0].get("id") if source_runs else None
        replay_role = str(row.get("replay_role", ""))
        if replay_role.startswith("rule_collision_"):
            kept_rows.append(json.loads(json.dumps(row)))
            continue
        if isinstance(source_row_id, str) and source_row_id in selected_row_ids:
            kept_rows.append(json.loads(json.dumps(row)))
            kept_opening_row_ids.append(source_row_id)
    return kept_rows, kept_opening_row_ids


def build_summary(
    *,
    base_summary: dict,
    rows: list[dict],
    kept_opening_row_ids: list[str],
    subfamily: str,
) -> dict:
    replay_role_counts = Counter(str(row.get("replay_role", "")) for row in rows)
    teacher_move_counts = Counter(
        int(row["teacher_selected_move"])
        for row in rows
        if "teacher_selected_move" in row
    )
    row_ids_by_replay_role: dict[str, list[str]] = {}
    teacher_move_counts_by_replay_role: dict[str, Counter[int]] = {}
    reference_extra_turn_by_row: dict[str, bool] = {}
    rule_collision_guard_row_ids: list[str] = []
    for row in rows:
        replay_role = str(row.get("replay_role", ""))
        source_runs = row.get("source_runs") or []
        row_id = source_runs[0].get("id") if source_runs else None
        if isinstance(row_id, str):
            row_ids_by_replay_role.setdefault(replay_role, []).append(row_id)
            if replay_role.startswith("rule_collision_"):
                rule_collision_guard_row_ids.append(row_id)
            if "reference_move_extra_turn_available" in row:
                reference_extra_turn_by_row[row_id] = bool(
                    row["reference_move_extra_turn_available"]
                )
        if "teacher_selected_move" in row:
            teacher_move_counts_by_replay_role.setdefault(replay_role, Counter())[
                int(row["teacher_selected_move"])
            ] += 1

    return {
        **base_summary,
        "row_count": len(rows),
        "filtered_opening_subfamily": subfamily,
        "filtered_opening_row_ids": kept_opening_row_ids,
        "replay_role_counts": dict(sorted(replay_role_counts.items())),
        "teacher_selected_move_distribution": dict(sorted(teacher_move_counts.items())),
        "row_ids_by_replay_role": {
            role: sorted(row_ids_by_replay_role[role])
            for role in sorted(row_ids_by_replay_role)
        },
        "teacher_selected_move_distribution_by_replay_role": {
            role: dict(sorted(teacher_move_counts_by_replay_role[role].items()))
            for role in sorted(teacher_move_counts_by_replay_role)
        },
        "reference_extra_turn_by_row": dict(
            sorted(reference_extra_turn_by_row.items())
        ),
        "tracked_opening_row_ids": kept_opening_row_ids,
        "rule_collision_guard_row_ids": rule_collision_guard_row_ids,
        "rule_collision_guard_count": len(rule_collision_guard_row_ids),
        "filtered": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_rows = load_jsonl(Path(args.artifact))
    base_summary = load_json(Path(args.artifact_summary))
    grouped = row_ids_by_subfamily(load_json(Path(args.opening_subfamily_diagnostic)))
    selected_row_ids = set(grouped.get(args.subfamily) or [])
    if not selected_row_ids:
        raise SystemExit(f"no opening rows found for subfamily: {args.subfamily}")

    rows, kept_opening_row_ids = filter_rows(
        artifact_rows=artifact_rows,
        selected_row_ids=selected_row_ids,
    )
    summary = build_summary(
        base_summary=base_summary,
        rows=rows,
        kept_opening_row_ids=kept_opening_row_ids,
        subfamily=args.subfamily,
    )

    write_jsonl(Path(args.out), rows)
    write_json(Path(args.out_summary), summary)
    print(json.dumps({"out": args.out, "rows": len(rows), "subfamily": args.subfamily}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
