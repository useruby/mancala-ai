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


DEFAULT_BALANCED_REPLAY_SOURCE = (
    Path(__file__).resolve().with_name("tactical_balanced_replay_source.jsonl")
)
CAPTURE_PROTECTION_COUNT = 2
CAPTURE_PRESERVATION_TARGET = 2
CAPTURE_PRESERVATION_MINIMUM = 2
NEARBY_BUCKETS = ("high_imbalance", "high_value_swing", "starvation_pressure")
NEARBY_PER_BUCKET = 2
TARGETED_SOURCE_ROW_ID = "incumbent_proxy_disagreement-033"
TARGETED_SOURCE_BUCKET = "incumbent_proxy_disagreement"
TARGETED_SOURCE_REPLAY_ROLE = "targeted_source_coverage"
TARGETED_SOURCE_SUMMARY_FILENAME = (
    "targeted_source_coverage_for_incumbent_proxy_033_summary.json"
)
SUPPORTED_VARIANTS = {
    "capped_11": "targeted-source-coverage-033-capped-11",
    "expanded_12_guard_reinforced": "targeted-source-coverage-033-expanded-12-guard-002",
}
TARGETED_SOURCE_VARIANT_ROW_COUNTS = {
    "capped_11": 11,
    "expanded_12_guard_reinforced": 12,
}
GUARD_REINFORCEMENT_ROW_ID = "capture_available-002"
GUARD_REINFORCEMENT_REPLAY_ROLE = "guard_reinforcement"
REPLAY_ROW_REQUIRED_FIELDS = (
    "canonical_state",
    "state",
    "raw_state",
    "legal_moves",
    "policy",
    "value",
    "bucket",
    "bucket_group",
    "input_encoding",
    "policy_target_mode",
    "value_target_mode",
    "replay_role",
)
TARGETED_SOURCE_CANONICAL_STATE = json.dumps(
    {
        "current_player": 0,
        "opponent_pits": [2, 9, 7, 7, 1, 0],
        "opponent_store": 2,
        "player_pits": [2, 2, 2, 1, 9, 0],
        "player_store": 4,
    },
    separators=(",", ":"),
    sort_keys=True,
)
TARGETED_SOURCE_GUARD_CANONICAL_STATES = {
    "capture_available-002": json.dumps(
        {
            "current_player": 1,
            "opponent_pits": [5, 4, 4, 4, 4, 0],
            "opponent_store": 1,
            "player_pits": [1, 0, 7, 6, 6, 5],
            "player_store": 1,
        },
        separators=(",", ":"),
        sort_keys=True,
    ),
    "capture_available-003": json.dumps(
        {
            "current_player": 1,
            "opponent_pits": [5, 5, 4, 4, 4, 0],
            "opponent_store": 1,
            "player_pits": [1, 6, 0, 6, 6, 5],
            "player_store": 1,
        },
        separators=(",", ":"),
        sort_keys=True,
    ),
}
TARGETED_SOURCE_STATE_STRUCTURE = json.loads(TARGETED_SOURCE_CANONICAL_STATE)
TARGETED_SOURCE_GUARD_STATE_STRUCTURES = {
    row_id: json.loads(canonical_state)
    for row_id, canonical_state in TARGETED_SOURCE_GUARD_CANONICAL_STATES.items()
}


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


def _write_summary(summary: dict, *, summary_out_path: Path) -> None:
    summary_out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_out_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _targeted_source_summary_out_path(out_path: Path) -> Path:
    return out_path.parent / TARGETED_SOURCE_SUMMARY_FILENAME


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
        raise ValueError(
            f"balanced replay source lacks required bucket counts: {', '.join(missing)}"
        )


def _normalized_capture_row(row: dict) -> dict | None:
    try:
        normalized = validate_tactical_capture_row(row)
    except ValueError:
        return None
    normalized.setdefault(
        "raw_state",
        row.get("raw_state")
        or raw_state_from_canonical_state(normalized.get("canonical_state")),
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
    row["source_artifacts"] = list(dict.fromkeys(sanitized)) or [
        CAPTURE_PROTECTION_SOURCE_ARTIFACT
    ]
    return row


def _normalized_canonical_state_structure(candidate_state):
    if isinstance(candidate_state, str):
        try:
            return json.loads(candidate_state)
        except json.JSONDecodeError:
            return None
    return candidate_state


def _targeted_source_state_matches(candidate_state) -> bool:
    return (
        _normalized_canonical_state_structure(candidate_state)
        == TARGETED_SOURCE_STATE_STRUCTURE
    )


def _normalized_targeted_source_row_id(row_id) -> str:
    if isinstance(row_id, str):
        row_id = row_id.strip()
    if not row_id:
        return TARGETED_SOURCE_ROW_ID
    return str(row_id)


def _select_capture_protection_rows(
    regression_position: dict, source_rows: list[dict]
) -> list[dict]:
    exact_row = build_regression_row(
        regression_position=regression_position,
        teacher_labeler=lambda raw_state: teacher_label_regression_row(
            raw_state,
            source_id=str(
                regression_position.get("id", "capture_protection_regression")
            ),
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
        if normalized is None or normalized.get("canonical_state") == exact_row.get(
            "canonical_state"
        ):
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


def _select_capture_preservation_rows(
    source_rows: list[dict], protection_rows: list[dict]
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


def _select_capture_preservation_rows_excluding(
    source_rows: list[dict],
    protection_rows: list[dict],
    *,
    excluded_row_ids: set[str] | None = None,
) -> list[dict]:
    selected_rows = _select_capture_preservation_rows(source_rows, protection_rows)
    if not excluded_row_ids:
        return selected_rows

    filtered_rows = [
        row for row in selected_rows if str(row.get("id", "")) not in excluded_row_ids
    ]
    if len(filtered_rows) == len(selected_rows):
        return selected_rows

    protected_states = {str(row.get("canonical_state")) for row in protection_rows}
    selected_states = {str(row.get("canonical_state")) for row in filtered_rows}
    selected_shapes = {
        row.get("capture_shape_key")
        for row in filtered_rows
        if row.get("capture_shape_key") is not None
    }

    candidates = []
    for row in source_rows:
        if row.get("bucket") != "capture_available":
            continue
        if str(row.get("id", "")) in excluded_row_ids:
            continue
        normalized = _normalized_capture_row(row)
        if normalized is None:
            continue
        canonical_state = str(normalized.get("canonical_state"))
        if canonical_state in protected_states or canonical_state in selected_states:
            continue
        try:
            shape_key = capture_shape_key(normalized)
        except (KeyError, TypeError, ValueError):
            continue
        if shape_key in selected_shapes:
            continue
        normalized["capture_shape_key"] = shape_key
        candidates.append(normalized)

    candidates.sort(
        key=lambda row: (
            -float(row.get("priority_score", 0.0)),
            str(row.get("canonical_state", "")),
        )
    )

    for row in candidates:
        filtered_rows.append(row)
        selected_states.add(str(row.get("canonical_state")))
        selected_shapes.add(row.get("capture_shape_key"))
        if len(filtered_rows) == CAPTURE_PRESERVATION_TARGET:
            break

    if len(filtered_rows) < CAPTURE_PRESERVATION_MINIMUM:
        raise ValueError(
            f"capture_preservation distinct shapes below minimum: {len(filtered_rows)}"
        )

    return filtered_rows


def _select_nearby_preservation_rows(source_rows: list[dict]) -> list[dict]:
    selected_rows = []
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
        selected = bucket_rows[:NEARBY_PER_BUCKET]
        if len(selected) < NEARBY_PER_BUCKET:
            raise ValueError(
                f"balanced replay source lacks nearby_preservation rows for {bucket}: selected={len(selected)}"
            )
        selected_rows.extend(selected)
    return selected_rows


def _select_targeted_source_coverage_row(source_rows: list[dict]) -> dict | None:
    candidates = []
    for row in source_rows:
        if row.get("bucket") != TARGETED_SOURCE_BUCKET:
            continue
        if not _targeted_source_state_matches(row.get("canonical_state")):
            continue
        if _normalized_targeted_source_row_id(row.get("id")) != TARGETED_SOURCE_ROW_ID:
            continue
        candidate = _sanitize_source_artifacts(dict(row))
        candidate["id"] = _normalized_targeted_source_row_id(candidate.get("id"))
        candidates.append(candidate)

    if not candidates:
        return None

    candidates.sort(
        key=lambda row: (
            str(row.get("id") != TARGETED_SOURCE_ROW_ID),
            -float(row.get("priority_score", 0.0)),
            str(row.get("canonical_state", "")),
        )
    )
    return candidates[0]


def _select_guard_reinforcement_row(source_rows: list[dict]) -> dict:
    candidates = []
    for row in source_rows:
        if str(row.get("id")) != GUARD_REINFORCEMENT_ROW_ID:
            continue
        if row.get("bucket") != "capture_available":
            continue
        normalized = _normalized_capture_row(row)
        if normalized is None:
            continue
        candidates.append(_sanitize_source_artifacts(dict(normalized)))

    if not candidates:
        raise ValueError(
            f"guard reinforcement row missing for {GUARD_REINFORCEMENT_ROW_ID}"
        )

    candidates.sort(
        key=lambda row: (
            -float(row.get("priority_score", 0.0)),
            str(row.get("canonical_state", "")),
        )
    )
    selected = candidates[0]
    selected["id"] = GUARD_REINFORCEMENT_ROW_ID
    return selected


def _guard_canonical_states_present(source_rows: list[dict]) -> dict[str, bool]:
    present_states = {
        json.dumps(normalized_state, separators=(",", ":"), sort_keys=True)
        for row in source_rows
        if (
            normalized_state := _normalized_canonical_state_structure(
                row.get("canonical_state")
            )
        )
        is not None
    }
    return {
        row_id: json.dumps(canonical_state, separators=(",", ":"), sort_keys=True)
        in present_states
        for row_id, canonical_state in TARGETED_SOURCE_GUARD_STATE_STRUCTURES.items()
    }


def _guard_reinforcement_matches_expected(rows: list[dict]) -> bool:
    guard_rows = [
        row for row in rows if row.get("replay_role") == GUARD_REINFORCEMENT_REPLAY_ROLE
    ]
    if len(guard_rows) != 1:
        return False

    expected_canonical_state = TARGETED_SOURCE_GUARD_CANONICAL_STATES[
        GUARD_REINFORCEMENT_ROW_ID
    ]
    actual_canonical_state = guard_rows[0].get("canonical_state")
    normalized_actual_state = _normalized_canonical_state_structure(
        actual_canonical_state
    )
    if normalized_actual_state is None:
        return False
    return (
        json.dumps(normalized_actual_state, separators=(",", ":"), sort_keys=True)
        == expected_canonical_state
    )


def _replay_row_training_ready(row: dict) -> bool:
    if any(field not in row for field in REPLAY_ROW_REQUIRED_FIELDS):
        return False
    if _normalized_canonical_state_structure(row.get("canonical_state")) is None:
        return False
    if not isinstance(row.get("state"), list) or not row["state"]:
        return False
    if not isinstance(row.get("raw_state"), dict) or not row["raw_state"]:
        return False
    if not isinstance(row.get("legal_moves"), list):
        return False
    if not isinstance(row.get("policy"), list) or len(row["policy"]) != 6:
        return False
    if not str(row.get("replay_role", "")):
        return False
    return True


def _replay_rows_training_ready(rows: list[dict]) -> bool:
    return all(_replay_row_training_ready(row) for row in rows)


def _build_targeted_source_summary(
    *, rows: list[dict], source_rows: list[dict], out_path: Path, variant: str
) -> dict:
    summary_out_path = _targeted_source_summary_out_path(out_path)
    targeted_rows = [
        row for row in rows if row.get("replay_role") == TARGETED_SOURCE_REPLAY_ROLE
    ]
    guard_rows = [
        row for row in rows if row.get("replay_role") == GUARD_REINFORCEMENT_REPLAY_ROLE
    ]
    guard_presence = _guard_canonical_states_present(source_rows)
    role_order = [str(row.get("replay_role", "")) for row in rows]
    role_counts = {}
    for role in role_order:
        role_counts[role] = role_counts.get(role, 0) + 1
    expected_row_count = TARGETED_SOURCE_VARIANT_ROW_COUNTS[variant]
    guard_reinforcement_required = variant == "expanded_12_guard_reinforced"
    guard_reinforcement_present = (
        _guard_reinforcement_matches_expected(rows)
        if guard_reinforcement_required
        else len(guard_rows) == 0
    )
    pass_flags = {
        "targeted_row_selected": len(targeted_rows) == 1,
        "guard_canonical_states_present": all(guard_presence.values()),
        "output_row_count_matches_variant": len(rows) == expected_row_count,
        "guard_reinforcement_present": guard_reinforcement_present,
        "replay_rows_training_ready": _replay_rows_training_ready(rows),
    }
    pass_flags["structurally_valid"] = all(pass_flags.values())

    return {
        "schema": "azlite_targeted_source_coverage_for_incumbent_proxy_033_summary_v1",
        "variant": variant,
        "variant_run_id": SUPPORTED_VARIANTS[variant],
        "summary_artifact_path": str(summary_out_path),
        "replay_artifact_path": str(out_path),
        "targeted_row_id": TARGETED_SOURCE_ROW_ID,
        "targeted_bucket": TARGETED_SOURCE_BUCKET,
        "targeted_canonical_state": TARGETED_SOURCE_CANONICAL_STATE,
        "selected_row_count": len(targeted_rows),
        "selected_replay_role": TARGETED_SOURCE_REPLAY_ROLE,
        "selected_ids": [str(row.get("id", "")) for row in targeted_rows],
        "guard_reinforcement_ids": [str(row.get("id", "")) for row in guard_rows],
        "output_row_count": len(rows),
        "role_order": role_order,
        "role_counts": role_counts,
        "guard_canonical_states_present": guard_presence,
        "pass_flags": pass_flags,
    }


def build_balanced_replay_dataset(
    *,
    regression_positions_path: Path,
    tactical_replay_path: Path,
    out_path: Path,
    variant: str = "capped_11",
):
    if variant not in SUPPORTED_VARIANTS:
        supported = ", ".join(sorted(SUPPORTED_VARIANTS))
        raise ValueError(
            f"unsupported targeted source coverage variant: {variant}; supported variants: {supported}"
        )

    if not tactical_replay_path.exists():
        raise FileNotFoundError(
            f"balanced replay source not found: {tactical_replay_path}"
        )

    regression_positions = _load_json(regression_positions_path)
    if not regression_positions:
        raise ValueError("regression fixture must contain at least one position")
    regression_position = regression_positions[0]
    source_rows = _load_jsonl(tactical_replay_path)

    _require_source_counts(source_rows)

    protection_rows = _select_capture_protection_rows(regression_position, source_rows)
    excluded_preservation_ids = (
        {GUARD_REINFORCEMENT_ROW_ID}
        if variant == "expanded_12_guard_reinforced"
        else None
    )
    preservation_rows = _select_capture_preservation_rows_excluding(
        source_rows,
        protection_rows,
        excluded_row_ids=excluded_preservation_ids,
    )
    nearby_rows = _select_nearby_preservation_rows(source_rows)
    targeted_row = _select_targeted_source_coverage_row(source_rows)
    guard_reinforcement_row = (
        _select_guard_reinforcement_row(source_rows)
        if variant == "expanded_12_guard_reinforced"
        else None
    )

    rows = [
        *(_role_labeled(row, "capture_protection") for row in protection_rows),
        *(_role_labeled(row, "capture_preservation") for row in preservation_rows),
        *(_role_labeled(row, "nearby_preservation") for row in nearby_rows),
    ]
    if targeted_row is not None:
        rows.append(_role_labeled(targeted_row, TARGETED_SOURCE_REPLAY_ROLE))
    if (
        variant == "expanded_12_guard_reinforced"
        and guard_reinforcement_row is not None
    ):
        rows.append(
            _role_labeled(guard_reinforcement_row, GUARD_REINFORCEMENT_REPLAY_ROLE)
        )

    expected_row_count = TARGETED_SOURCE_VARIANT_ROW_COUNTS[variant]
    if len(rows) != expected_row_count:
        raise ValueError(
            f"balanced replay dataset row count mismatch for variant {variant}: expected={expected_row_count}, actual={len(rows)}"
        )

    summary_out_path = _targeted_source_summary_out_path(out_path)
    summary = _build_targeted_source_summary(
        rows=rows,
        source_rows=source_rows,
        out_path=out_path,
        variant=variant,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    _write_summary(summary, summary_out_path=summary_out_path)
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regression-positions", required=True)
    parser.add_argument(
        "--tactical-replay", default=str(DEFAULT_BALANCED_REPLAY_SOURCE)
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--variant", choices=sorted(SUPPORTED_VARIANTS), default="capped_11"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_balanced_replay_dataset(
        regression_positions_path=Path(args.regression_positions),
        tactical_replay_path=Path(args.tactical_replay),
        out_path=Path(args.out),
        variant=args.variant,
    )


if __name__ == "__main__":
    main()
