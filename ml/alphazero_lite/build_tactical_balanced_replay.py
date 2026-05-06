#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.build_tactical_capture_protection import (
    build_regression_row,
    capture_shape_key,
    extract_regression_motif_signature,
    raw_state_from_canonical_state,
    score_candidate_support_row,
    teacher_label_regression_row,
    CAPTURE_PROTECTION_SOURCE_ARTIFACT,
    validate_tactical_capture_row,
)


DEFAULT_BALANCED_REPLAY_SOURCE = Path(__file__).resolve().with_name("tactical_balanced_replay_source.jsonl")
CAPTURE_PROTECTION_COUNT = 2
CAPTURE_PRESERVATION_TARGET = 2
CAPTURE_PRESERVATION_MINIMUM = 2
NEARBY_BUCKETS = ("high_imbalance", "high_value_swing", "starvation_pressure")
NEARBY_PER_BUCKET = 2


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _require_source_counts(rows: list[dict]) -> None:
    counts = {"capture_available": 0, **{bucket: 0 for bucket in NEARBY_BUCKETS}}
    for row in rows:
        bucket = row.get("bucket")
        if bucket in counts:
            counts[bucket] += 1

    missing = []
    required_capture_available = 1 + CAPTURE_PRESERVATION_MINIMUM
    if counts["capture_available"] < required_capture_available:
        missing.append(f"capture_available={counts['capture_available']}")
    for bucket in NEARBY_BUCKETS:
        if counts[bucket] < NEARBY_PER_BUCKET:
            missing.append(f"{bucket}={counts[bucket]}")
    if missing:
        raise ValueError(f"balanced replay source lacks required bucket counts: {', '.join(missing)}")


def _normalized_capture_row(row: dict) -> dict | None:
    try:
        normalized = validate_tactical_capture_row(row)
    except ValueError:
        return None
    normalized.setdefault(
        "raw_state",
        row.get("raw_state") or raw_state_from_canonical_state(normalized.get("canonical_state")),
    )
    if not normalized.get("raw_state"):
        return None
    return normalized


def _role_labeled(row: dict, role: str) -> dict:
    labeled = dict(row)
    labeled["replay_role"] = role
    return json.loads(json.dumps(labeled))


def _sanitize_source_artifacts(row: dict) -> dict:
    source_artifacts = row.get("source_artifacts")
    if not isinstance(source_artifacts, list):
        return row

    sanitized = []
    for item in source_artifacts:
        if isinstance(item, str) and item:
            sanitized.append(Path(item).name if Path(item).is_absolute() else item)
    row["source_artifacts"] = list(dict.fromkeys(sanitized)) or [CAPTURE_PROTECTION_SOURCE_ARTIFACT]
    return row


def _select_capture_protection_rows(regression_position: dict, source_rows: list[dict]) -> list[dict]:
    exact_row = build_regression_row(
        regression_position=regression_position,
        teacher_labeler=lambda raw_state: teacher_label_regression_row(
            raw_state,
            source_id=str(regression_position.get("id", "capture_protection_regression")),
            move_number=regression_position.get("move_number"),
        ),
    )
    signature = extract_regression_motif_signature(
        regression_position["state"],
        expected_move=int(regression_position["expected_move"]),
    )

    support_rows = []
    for row in source_rows:
        if row.get("bucket") != "capture_available":
            continue
        normalized = _normalized_capture_row(row)
        if normalized is None or normalized.get("canonical_state") == exact_row.get("canonical_state"):
            continue
        try:
            scored = score_candidate_support_row(normalized, signature)
            normalized["capture_shape_key"] = capture_shape_key(normalized)
        except (KeyError, TypeError, ValueError):
            continue
        if not scored["motif_protective"]:
            continue
        normalized["motif_support_score"] = scored["score"]
        normalized["motif_protective"] = True
        support_rows.append(normalized)

    if not support_rows:
        raise ValueError("balanced replay source lacks capture_protection support rows")

    support_rows.sort(
        key=lambda row: (
            -float(row.get("motif_support_score", 0.0)),
            -float(row.get("priority_score", 0.0)),
            str(row.get("canonical_state", "")),
        )
    )
    return [exact_row, support_rows[0]]


def _select_capture_preservation_rows(source_rows: list[dict], protection_rows: list[dict]) -> list[dict]:
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

    distinct_rows = []
    seen_shapes = set()
    for row in candidates:
        shape_key = row["capture_shape_key"]
        if shape_key in seen_shapes:
            continue
        distinct_rows.append(row)
        seen_shapes.add(shape_key)
        if len(distinct_rows) == CAPTURE_PRESERVATION_TARGET:
            break

    if len(distinct_rows) < CAPTURE_PRESERVATION_MINIMUM:
        raise ValueError(
            f"capture_preservation distinct shapes below minimum: {len(distinct_rows)}"
        )

    if len(distinct_rows) <= CAPTURE_PRESERVATION_TARGET:
        print(
            f"balanced replay source supports only {len(distinct_rows)} distinct capture-preservation shapes",
            file=sys.stderr,
        )

    return distinct_rows


def _select_nearby_preservation_rows(source_rows: list[dict]) -> list[dict]:
    selected_rows = []
    for bucket in NEARBY_BUCKETS:
        bucket_rows = [_sanitize_source_artifacts(dict(row)) for row in source_rows if row.get("bucket") == bucket]
        bucket_rows.sort(
            key=lambda row: (
                -float(row.get("priority_score", 0.0)),
                str(row.get("canonical_state", "")),
            )
        )
        selected = bucket_rows[:NEARBY_PER_BUCKET]
        if len(selected) < NEARBY_PER_BUCKET:
            raise ValueError(
                f"balanced replay source lacks nearby_preservation rows for {bucket}: selected={len(selected)}"
            )
        selected_rows.extend(selected)
    return selected_rows


def build_balanced_replay_dataset(
    *, regression_positions_path: Path, tactical_replay_path: Path, out_path: Path
):
    if not tactical_replay_path.exists():
        raise FileNotFoundError(f"balanced replay source not found: {tactical_replay_path}")

    regression_positions = _load_json(regression_positions_path)
    if not regression_positions:
        raise ValueError("regression fixture must contain at least one position")
    regression_position = regression_positions[0]
    source_rows = _load_jsonl(tactical_replay_path)

    _require_source_counts(source_rows)

    protection_rows = _select_capture_protection_rows(regression_position, source_rows)
    preservation_rows = _select_capture_preservation_rows(source_rows, protection_rows)
    nearby_rows = _select_nearby_preservation_rows(source_rows)

    rows = [
        *(_role_labeled(row, "capture_protection") for row in protection_rows),
        *(_role_labeled(row, "capture_preservation") for row in preservation_rows),
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
    parser.add_argument("--regression-positions", required=True)
    parser.add_argument("--tactical-replay", default=str(DEFAULT_BALANCED_REPLAY_SOURCE))
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_balanced_replay_dataset(
        regression_positions_path=Path(args.regression_positions),
        tactical_replay_path=Path(args.tactical_replay),
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
