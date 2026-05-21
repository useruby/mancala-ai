#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.build_tactical_balanced_replay import (
    DEFAULT_BALANCED_REPLAY_SOURCE,
    _load_jsonl,
    _role_labeled,
    _sanitize_source_artifacts,
    _select_capture_preservation_rows,
    _select_nearby_preservation_rows,
)
from ml.alphazero_lite.build_tactical_capture_protection import (
    build_regression_row,
    teacher_label_regression_row,
)


DEFAULT_OPENING_CAPTURE_FAMILY_REPLAY_SOURCE = DEFAULT_BALANCED_REPLAY_SOURCE
DEFAULT_REGRESSION_POSITIONS = (
    Path(__file__).resolve().parents[2]
    / "test/fixtures/ai/superhuman_regression_positions.json"
)
OPENING_FAMILY_SOURCE_IDS = {
    "capture_available-017",
    "capture_available-019",
    "capture_available-024",
}


def _has_opening_family_provenance(row: dict) -> bool:
    source_runs = row.get("source_runs")
    if not isinstance(source_runs, list):
        return False

    for source_run in source_runs:
        if (
            isinstance(source_run, dict)
            and source_run.get("id") in OPENING_FAMILY_SOURCE_IDS
        ):
            return True
    return False


def _opening_family_source_id(row: dict) -> str | None:
    source_runs = row.get("source_runs")
    if not isinstance(source_runs, list):
        return None

    for source_run in source_runs:
        if isinstance(source_run, dict):
            source_id = source_run.get("id")
            if source_id in OPENING_FAMILY_SOURCE_IDS:
                return str(source_id)
    return None


def _is_opening_phase(row: dict) -> bool:
    ply = row.get("ply")
    if ply is not None:
        return int(ply) <= 4

    move_number = row.get("move_number")
    if move_number is not None:
        return int(move_number) <= 4

    return _has_opening_family_provenance(row)


def _is_opening_capture_family_row(row: dict) -> bool:
    return (
        row.get("bucket") == "capture_available"
        and row.get("legal_moves") == [0, 1, 2, 3, 4]
        and row.get("teacher_selected_move") == 3
        and _is_opening_phase(row)
    )


def _select_opening_capture_family_rows(source_rows: list[dict]) -> list[dict]:
    selected_rows = [
        _sanitize_source_artifacts(dict(row))
        for row in source_rows
        if _is_opening_capture_family_row(row)
    ]
    selected_rows.sort(
        key=lambda row: (
            -float(row.get("priority_score", 0.0)),
            str(row.get("canonical_state", "")),
        )
    )

    deduped_rows = []
    seen_source_ids: set[str] = set()
    for row in selected_rows:
        source_id = _opening_family_source_id(row)
        if source_id is None:
            deduped_rows.append(row)
            continue
        if source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)
        deduped_rows.append(row)

    return [json.loads(json.dumps(row)) for row in deduped_rows]


def _build_capture_protection_row(regression_positions_path: Path) -> dict:
    regression_positions = json.loads(
        regression_positions_path.read_text(encoding="utf-8")
    )
    if not regression_positions:
        raise ValueError("regression fixture must contain at least one position")

    regression_position = regression_positions[0]
    row = build_regression_row(
        regression_position=regression_position,
        teacher_labeler=lambda raw_state: teacher_label_regression_row(
            raw_state,
            source_id=str(
                regression_position.get("id", "capture_protection_regression")
            ),
            move_number=regression_position.get("move_number"),
        ),
    )
    return json.loads(json.dumps(_sanitize_source_artifacts(row)))


def build_opening_capture_family_replay_dataset(
    *, tactical_replay_path: Path, out_path: Path
):
    if not tactical_replay_path.exists():
        raise FileNotFoundError(
            f"opening capture family replay source not found: {tactical_replay_path}"
        )

    source_rows = _load_jsonl(tactical_replay_path)
    protection_row = _build_capture_protection_row(DEFAULT_REGRESSION_POSITIONS)
    preservation_rows = _select_capture_preservation_rows(source_rows, [protection_row])
    opening_family_rows = _select_opening_capture_family_rows(source_rows)
    nearby_rows = _select_nearby_preservation_rows(source_rows)

    rows = [
        _role_labeled(protection_row, "capture_protection"),
        *(_role_labeled(row, "capture_preservation") for row in preservation_rows),
        *(_role_labeled(row, "opening_capture_family") for row in opening_family_rows),
        *(_role_labeled(row, "nearby_preservation") for row in nearby_rows),
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tactical-replay", default=str(DEFAULT_OPENING_CAPTURE_FAMILY_REPLAY_SOURCE)
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_opening_capture_family_replay_dataset(
        tactical_replay_path=Path(args.tactical_replay),
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
