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
    NEARBY_BUCKETS,
    NEARBY_PER_BUCKET,
    _load_jsonl,
    _normalized_capture_row,
    _role_labeled,
    _sanitize_source_artifacts,
)
from ml.alphazero_lite.build_tactical_capture_protection import (
    CAPTURE_PROTECTION_INPUT_ENCODING,
    CAPTURE_PROTECTION_POLICY_SIMULATIONS,
    CAPTURE_PROTECTION_POLICY_TARGET_MODE,
    CAPTURE_PROTECTION_SEED,
    CAPTURE_PROTECTION_VALUE_SIMULATIONS,
    CAPTURE_PROTECTION_VALUE_TARGET_MODE,
    build_regression_row,
    capture_shape_key,
    teacher_label_regression_row,
)
from ml.alphazero_lite.forensic_suite import canonical_state_key, load_suite
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite import label_tactical_states


DEFAULT_STABLE_FAILURE_REPLAY_SOURCE = DEFAULT_BALANCED_REPLAY_SOURCE
DEFAULT_FORENSIC_SUITE = (
    Path(__file__).resolve().parents[2]
    / "ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"
)
DEFAULT_REGRESSION_POSITIONS = (
    Path(__file__).resolve().parents[2]
    / "test/fixtures/ai/superhuman_regression_positions.json"
)
TARGET_ROLE_COUNTS = {
    "capture_protection": 1,
    "capture_preservation": 4,
    "opening_capture_family": 7,
    "high_imbalance_stable": 3,
    "nearby_preservation": 8,
}
PRIOR_OPENING_CAPTURE_FAMILY_COUNT = 13
PRIOR_NEARBY_PRESERVATION_COUNT = 6


def load_reference_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if payload.get("schema") != "azlite_forensic_references_v1" or not isinstance(
        rows, list
    ):
        raise ValueError(
            "reference artifact must use azlite_forensic_references_v1 with rows list"
        )
    return [row for row in rows if isinstance(row, dict)]


def reference_rows_by_canonical_state(reference_rows: list[dict]) -> dict[str, dict]:
    return {
        str(row["canonical_state"]): row
        for row in reference_rows
        if isinstance(row.get("canonical_state"), str)
    }


def stable_reference_row(
    suite_row: dict, references_by_state: dict[str, dict]
) -> dict | None:
    canonical_state = canonical_state_key(suite_row["state"])
    reference = references_by_state.get(canonical_state)
    reference_move = reference.get("reference_move") if reference is not None else None
    legal_moves = KalahGame.from_state(suite_row["state"]).possible_moves()
    if (
        reference is None
        or reference.get("reference_unstable")
        or isinstance(reference_move, bool)
        or not isinstance(reference_move, int)
        or reference_move not in legal_moves
    ):
        return None
    return {
        "id": suite_row["id"],
        "canonical_state": canonical_state,
        "reference_canonical_state": canonical_state,
        "state": suite_row["state"],
        "legal_moves": legal_moves,
        "bucket": suite_row["bucket"],
        "phase": suite_row["phase"],
        "reference_move": int(reference_move),
    }


def stable_reference_ply(suite_row: dict) -> int:
    if suite_row.get("phase") == "opening":
        return 4
    return 12


def build_stable_reference_replay_row(
    stable_row: dict, *, reference_path: Path
) -> dict:
    raw_state = dict(stable_row["state"])
    labeled = label_tactical_states.label_row(
        {
            "canonical_state": stable_row["reference_canonical_state"],
            "state": raw_state,
            "side_to_move": int(raw_state["current_player"]),
            "legal_moves": list(stable_row["legal_moves"]),
            "selection_reasons": ["stable_failure_family"],
            "source_artifacts": [reference_path.name],
            "source_runs": [{"kind": "forensic_reference", "id": stable_row["id"]}],
            "priority_score": 1.0,
            "ply": stable_reference_ply(stable_row),
        },
        policy_simulations=CAPTURE_PROTECTION_POLICY_SIMULATIONS,
        value_simulations=CAPTURE_PROTECTION_VALUE_SIMULATIONS,
        seed=CAPTURE_PROTECTION_SEED,
        policy_target_mode=CAPTURE_PROTECTION_POLICY_TARGET_MODE,
        value_target_mode=CAPTURE_PROTECTION_VALUE_TARGET_MODE,
        input_encoding=CAPTURE_PROTECTION_INPUT_ENCODING,
    )
    return {
        **labeled,
        "id": stable_row["id"],
        "phase": stable_row["phase"],
        "bucket": stable_row["bucket"],
        "reference_move": stable_row["reference_move"],
        "reference_unstable": False,
    }


def select_opening_capture_family_rows(
    *, suite_rows: list[dict], reference_rows: list[dict]
) -> list[dict]:
    references_by_state = reference_rows_by_canonical_state(reference_rows)
    selected = []
    for row in suite_rows:
        if not (
            isinstance(row.get("id"), str)
            and row["id"].startswith("capture_available-")
            and row.get("bucket") == "capture_available"
            and row.get("phase") == "opening"
        ):
            continue
        stable_row = stable_reference_row(row, references_by_state)
        if stable_row is not None:
            selected.append(_role_labeled(stable_row, "opening_capture_family"))
    return selected


def select_high_imbalance_stable_rows(
    *,
    suite_rows: list[dict],
    reference_rows: list[dict],
    selected_ids: set[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    references_by_state = reference_rows_by_canonical_state(reference_rows)
    selected = []
    for row in suite_rows:
        if row.get("bucket") != "high_imbalance":
            continue
        if selected_ids is not None and row.get("id") not in selected_ids:
            continue
        stable_row = stable_reference_row(row, references_by_state)
        if stable_row is not None:
            selected.append(_role_labeled(stable_row, "high_imbalance_stable"))
        if limit is not None and len(selected) == limit:
            break
    return selected


def select_capture_preservation_rows(
    *, source_rows: list[dict], protection_rows: list[dict]
) -> list[dict]:
    protected_states = {str(row.get("canonical_state")) for row in protection_rows}
    protected_shapes = set()
    for row in protection_rows:
        normalized = _normalized_capture_row(row)
        if normalized is None:
            continue
        try:
            protected_shapes.add(capture_shape_key(normalized))
        except (KeyError, TypeError, ValueError):
            continue

    candidates = []
    for row in source_rows:
        if row.get("bucket") != "capture_available":
            continue
        normalized = _normalized_capture_row(row)
        if normalized is None:
            continue
        canonical_state = str(normalized.get("canonical_state"))
        if canonical_state in protected_states:
            continue
        try:
            shape_key = capture_shape_key(normalized)
        except (KeyError, TypeError, ValueError):
            continue
        if shape_key in protected_shapes:
            continue
        normalized["capture_shape_key"] = shape_key
        candidates.append(normalized)

    candidates.sort(
        key=lambda row: (
            -float(row.get("priority_score", 0.0)),
            str(row.get("canonical_state", "")),
        )
    )

    selected = []
    seen_shapes = set()
    remaining = []
    for row in candidates:
        shape_key = row["capture_shape_key"]
        if shape_key in seen_shapes:
            remaining.append(row)
            continue
        selected.append(row)
        seen_shapes.add(shape_key)
        if len(selected) == TARGET_ROLE_COUNTS["capture_preservation"]:
            return selected

    for row in remaining:
        selected.append(row)
        if len(selected) == TARGET_ROLE_COUNTS["capture_preservation"]:
            break

    return selected


def select_nearby_preservation_rows(*, source_rows: list[dict]) -> list[dict]:
    selected_rows = []
    remaining_rows = []
    for bucket in NEARBY_BUCKETS:
        bucket_rows = [
            _sanitize_source_artifacts(dict(row))
            for row in source_rows
            if row.get("bucket") == bucket
        ]
        bucket_rows.sort(
            key=lambda row: (
                -float(row.get("priority_score", 0.0)),
                str(row.get("canonical_state", "")),
            )
        )
        base_rows = bucket_rows[:NEARBY_PER_BUCKET]
        selected_rows.extend(base_rows)
        remaining_rows.extend(bucket_rows[NEARBY_PER_BUCKET:])

    remaining_rows.sort(
        key=lambda row: (
            -float(row.get("priority_score", 0.0)),
            str(row.get("canonical_state", "")),
        )
    )
    extra_needed = max(
        0, TARGET_ROLE_COUNTS["nearby_preservation"] - len(selected_rows)
    )
    selected_rows.extend(remaining_rows[:extra_needed])
    return selected_rows


def selected_ids_by_role(rows: list[dict]) -> dict[str, list[str]]:
    selected = {role: [] for role in TARGET_ROLE_COUNTS}
    id_field_by_role = {
        "capture_protection": "id",
        "capture_preservation": "canonical_state",
        "opening_capture_family": "id",
        "high_imbalance_stable": "id",
        "nearby_preservation": "canonical_state",
    }
    for row in rows:
        role = row.get("replay_role")
        if role not in selected:
            continue
        value = row.get(id_field_by_role[role])
        if isinstance(value, str):
            selected[role].append(value)
    return selected


def actual_role_counts(rows: list[dict]) -> dict[str, int]:
    counts = {role: 0 for role in TARGET_ROLE_COUNTS}
    for row in rows:
        role = row.get("replay_role")
        if role in counts:
            counts[role] += 1
    return counts


def capture_preservation_shape_counts(rows: list[dict]) -> dict[str, int]:
    shape_counts: dict[str, int] = {}
    for row in rows:
        if row.get("replay_role") != "capture_preservation":
            continue
        normalized = _normalized_capture_row(row)
        if normalized is None:
            continue
        try:
            shape_key = repr(capture_shape_key(normalized))
        except (KeyError, TypeError, ValueError):
            continue
        shape_counts[shape_key] = shape_counts.get(shape_key, 0) + 1
    return shape_counts


def nearby_bucket_counts(rows: list[dict]) -> dict[str, int]:
    counts = {bucket: 0 for bucket in NEARBY_BUCKETS}
    for row in rows:
        if row.get("replay_role") != "nearby_preservation":
            continue
        bucket = row.get("bucket")
        if bucket in counts:
            counts[bucket] += 1
    return counts


def invalid_reasons_for_counts(
    role_counts: dict[str, int],
    *,
    capture_shape_counts: dict[str, int],
    nearby_bucket_counts: dict[str, int],
) -> list[str]:
    reasons = []
    opening_count = role_counts["opening_capture_family"]
    nearby_count = role_counts["nearby_preservation"]
    other_counts = [
        count for role, count in role_counts.items() if role != "opening_capture_family"
    ]
    missing_nearby_buckets = [
        bucket for bucket in NEARBY_BUCKETS if nearby_bucket_counts.get(bucket, 0) <= 0
    ]

    if opening_count >= PRIOR_OPENING_CAPTURE_FAMILY_COUNT:
        reasons.append("opening_capture_family not capped below prior 13")
    if nearby_count <= PRIOR_NEARBY_PRESERVATION_COUNT:
        reasons.append("nearby_preservation does not exceed prior 6")
    if other_counts and opening_count > max(other_counts):
        reasons.append("opening_capture_family remains dominant")
    if (
        role_counts["nearby_preservation"] >= TARGET_ROLE_COUNTS["nearby_preservation"]
        and missing_nearby_buckets
    ):
        reasons.append(
            "nearby_preservation missing bucket participation: "
            + ", ".join(missing_nearby_buckets)
        )
    return reasons


def build_summary(*, rows: list[dict], summary_out_path: Path) -> dict:
    role_counts = actual_role_counts(rows)
    capture_shape_count_map = capture_preservation_shape_counts(rows)
    nearby_bucket_count_map = nearby_bucket_counts(rows)
    capture_shape_shortfall = max(
        0, TARGET_ROLE_COUNTS["capture_preservation"] - len(capture_shape_count_map)
    )
    nearby_missing_buckets = [
        bucket
        for bucket in NEARBY_BUCKETS
        if nearby_bucket_count_map.get(bucket, 0) <= 0
    ]
    return {
        "schema": "azlite_tactical_stable_failure_family_replay_summary_v1",
        "summary_artifact_path": str(summary_out_path),
        "target_counts": dict(TARGET_ROLE_COUNTS),
        "role_counts": role_counts,
        "selected_ids_by_role": selected_ids_by_role(rows),
        "shortfalls_by_role": {
            role: max(0, TARGET_ROLE_COUNTS[role] - role_counts[role])
            for role in TARGET_ROLE_COUNTS
        },
        "capture_preservation_shape_counts": capture_shape_count_map,
        "capture_preservation_distinct_shape_count": len(capture_shape_count_map),
        "capture_preservation_shape_shortfall": capture_shape_shortfall,
        "nearby_bucket_counts": nearby_bucket_count_map,
        "nearby_missing_buckets": nearby_missing_buckets,
        "invalid_reasons": invalid_reasons_for_counts(
            role_counts,
            capture_shape_counts=capture_shape_count_map,
            nearby_bucket_counts=nearby_bucket_count_map,
        ),
        "total_rows": len(rows),
    }


def write_summary(summary: dict, *, summary_out_path: Path) -> None:
    summary_out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_out_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_capture_protection_row(regression_positions_path: Path) -> dict:
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
    sanitized = json.loads(json.dumps(_sanitize_source_artifacts(row)))
    sanitized["id"] = str(
        regression_position.get("id", "capture_protection_regression")
    )
    return sanitized


def build_stable_failure_family_replay_dataset(
    *,
    tactical_replay_path: Path,
    suite_path: Path,
    reference_path: Path,
    regression_positions_path: Path,
    high_imbalance_ids: set[str] | None = None,
    out_path: Path,
    summary_out_path: Path,
) -> tuple[list[dict], dict]:
    source_rows = _load_jsonl(tactical_replay_path)
    suite_rows = [position.__dict__ for position in load_suite(suite_path)]
    reference_rows = load_reference_rows(reference_path)

    protection_row = build_capture_protection_row(regression_positions_path)
    preservation_rows = select_capture_preservation_rows(
        source_rows=source_rows, protection_rows=[protection_row]
    )
    opening_rows = [
        build_stable_reference_replay_row(row, reference_path=reference_path)
        for row in select_opening_capture_family_rows(
            suite_rows=suite_rows, reference_rows=reference_rows
        )
    ][: TARGET_ROLE_COUNTS["opening_capture_family"]]
    imbalance_rows = [
        build_stable_reference_replay_row(row, reference_path=reference_path)
        for row in select_high_imbalance_stable_rows(
            suite_rows=suite_rows,
            reference_rows=reference_rows,
            selected_ids=high_imbalance_ids,
            limit=TARGET_ROLE_COUNTS["high_imbalance_stable"],
        )
    ]
    nearby_rows = select_nearby_preservation_rows(source_rows=source_rows)

    rows = [
        _role_labeled(protection_row, "capture_protection"),
        *(_role_labeled(row, "capture_preservation") for row in preservation_rows),
        *(_role_labeled(row, "opening_capture_family") for row in opening_rows),
        *(_role_labeled(row, "high_imbalance_stable") for row in imbalance_rows),
        *(_role_labeled(row, "nearby_preservation") for row in nearby_rows),
    ]

    summary = build_summary(rows=rows, summary_out_path=summary_out_path)
    summary["replay_artifact_path"] = str(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    write_summary(summary, summary_out_path=summary_out_path)
    if summary["invalid_reasons"]:
        raise ValueError("; ".join(summary["invalid_reasons"]))
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tactical-replay", default=str(DEFAULT_STABLE_FAILURE_REPLAY_SOURCE)
    )
    parser.add_argument("--suite", default=str(DEFAULT_FORENSIC_SUITE))
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--regression-positions", default=str(DEFAULT_REGRESSION_POSITIONS)
    )
    parser.add_argument(
        "--high-imbalance-id", action="append", dest="high_imbalance_ids", default=[]
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    high_imbalance_ids = (
        None if not args.high_imbalance_ids else set(args.high_imbalance_ids)
    )
    build_stable_failure_family_replay_dataset(
        tactical_replay_path=Path(args.tactical_replay),
        suite_path=Path(args.suite),
        reference_path=Path(args.reference),
        regression_positions_path=Path(args.regression_positions),
        high_imbalance_ids=high_imbalance_ids,
        out_path=Path(args.out),
        summary_out_path=Path(args.summary_out),
    )


if __name__ == "__main__":
    main()
