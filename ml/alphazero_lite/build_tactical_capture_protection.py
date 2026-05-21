#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.input_encodings import SUPPORTED_INPUT_ENCODINGS
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite import label_tactical_states, train

CAPTURE_PROTECTION_INPUT_ENCODING = "kalah_v3"
CAPTURE_PROTECTION_POLICY_TARGET_MODE = "sharpened"
CAPTURE_PROTECTION_VALUE_TARGET_MODE = "sharpened"
CAPTURE_PROTECTION_POLICY_SIMULATIONS = 1200
CAPTURE_PROTECTION_VALUE_SIMULATIONS = 1200
CAPTURE_PROTECTION_SEED = 42
CAPTURE_PROTECTION_SOURCE_ARTIFACT = "ml/alphazero_lite/tactical_replay_train.jsonl"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def raw_state_from_canonical_state(canonical_state: str | None) -> dict | None:
    if not isinstance(canonical_state, str) or not canonical_state:
        return None
    try:
        parsed = json.loads(canonical_state)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def copied_row_policy_peak(row: dict) -> int:
    policy = row.get("policy")
    if not isinstance(policy, list) or not policy:
        raise ValueError("policy is required for tactical capture protection rows")
    return max(range(len(policy)), key=lambda index: float(policy[index]))


def _side_view(state: dict, *, player: int) -> tuple[list[int], list[int], int, int]:
    if player == 0:
        return (
            list(state["player_pits"]),
            list(state["opponent_pits"]),
            int(state["player_store"]),
            int(state["opponent_store"]),
        )
    return (
        list(state["opponent_pits"]),
        list(state["player_pits"]),
        int(state["opponent_store"]),
        int(state["player_store"]),
    )


def move_outcome_features(raw_state: dict, move: int) -> dict[str, int | bool | None]:
    game = KalahGame.from_state(raw_state)
    original_player = game.current_player
    before_store = game.captured_seeds[original_player]
    absolute_move = game.pit_index(move)
    seeds = game.pits[absolute_move]
    if seeds <= 0:
        raise ValueError(f"illegal move {move} for motif features")

    landing_absolute = absolute_move
    pit_owner = original_player
    lands_in_store = False
    for _ in range(seeds):
        next_index = (landing_absolute + 1) % 12
        next_owner = game.pit_owner(next_index)
        if pit_owner != next_owner:
            need_take_seed_out = pit_owner == original_player
            pit_owner = next_owner
            if need_take_seed_out:
                lands_in_store = True
                continue
        lands_in_store = False
        landing_absolute = next_index

    simulated = KalahGame.from_state(raw_state)
    if not simulated.move(simulated.pit_index(move)):
        raise ValueError(f"illegal move {move} for motif features")
    state_after = simulated.to_state()
    side_pits_after, _, side_store_after, _ = _side_view(
        state_after, player=original_player
    )
    store_gain = side_store_after - before_store
    capture = store_gain > 1
    captured_opposite_pit = None
    if capture:
        captured_absolute = simulated.opposite_pit_index(landing_absolute)
        captured_opposite_pit = captured_absolute - ((1 - original_player) * 6)

    return {
        "move": move,
        "capture": capture,
        "extra_turn": simulated.current_player == original_player
        and not simulated.over(),
        "store_gain": store_gain,
        "landing_pit": None if lands_in_store else landing_absolute % 6,
        "lands_in_store": lands_in_store,
        "captured_opposite_pit": captured_opposite_pit,
        "post_move_empty_pits": sum(1 for value in side_pits_after if value == 0),
        "terminal": simulated.over(),
        "side_to_move_change": simulated.current_player != original_player,
    }


def legal_move_features(raw_state: dict) -> list[dict[str, int | bool | None]]:
    game = KalahGame.from_state(raw_state)
    return [move_outcome_features(raw_state, move) for move in game.possible_moves()]


def capture_store_gain_bucket(store_gain: int) -> str:
    if store_gain >= 6:
        return "gain_6_plus"
    if store_gain >= 4:
        return "gain_4_5"
    return "gain_2_3"


def landing_representation(feature: dict[str, int | bool | None]) -> tuple:
    if bool(feature["lands_in_store"]):
        return ("store",)
    return ("pit", int(feature["landing_pit"]))


def choose_candidate_conflict_moves(row: dict) -> dict[str, int]:
    raw_state = row["raw_state"]
    policy = row["policy"]
    features = legal_move_features(raw_state)
    captures = [feature for feature in features if feature["capture"]]
    extra_turns = [feature for feature in features if feature["extra_turn"]]
    if not captures or not extra_turns:
        raise ValueError("candidate row must expose both capture and extra-turn moves")

    capture = min(
        captures,
        key=lambda feature: (
            -int(feature["store_gain"]),
            -float(policy[int(feature["move"])]),
            int(feature["move"]),
        ),
    )
    extra_turn = min(
        extra_turns,
        key=lambda feature: (
            -float(policy[int(feature["move"])]),
            -int(feature["store_gain"]),
            int(feature["move"]),
        ),
    )
    return {
        "capture_move": int(capture["move"]),
        "extra_turn_move": int(extra_turn["move"]),
    }


def capture_shape_key(row: dict) -> tuple:
    conflict = choose_candidate_conflict_moves(row)
    features = {
        feature["move"]: feature for feature in legal_move_features(row["raw_state"])
    }
    capture = features[conflict["capture_move"]]
    extra_turn = features[conflict["extra_turn_move"]]
    return (
        landing_representation(capture),
        int(capture["captured_opposite_pit"]),
        capture_store_gain_bucket(int(capture["store_gain"])),
        "extra_turn",
        landing_representation(extra_turn),
    )


def extract_regression_motif_signature(raw_state: dict, *, expected_move: int) -> dict:
    features = {feature["move"]: feature for feature in legal_move_features(raw_state)}
    capture = features[expected_move]
    if not capture["capture"]:
        raise ValueError("expected regression move must be a capture")
    extra_turns = [feature for feature in features.values() if feature["extra_turn"]]
    if not extra_turns:
        raise ValueError("regression motif requires an extra-turn competitor")
    extra_turn = min(
        extra_turns,
        key=lambda feature: (-int(feature["store_gain"]), int(feature["move"])),
    )
    return {
        "target_capture_move": int(expected_move),
        "tempting_extra_turn_move": int(extra_turn["move"]),
        "legal_moves": sorted(features.keys()),
        "capture_store_gain": int(capture["store_gain"]),
        "extra_turn_store_gain": int(extra_turn["store_gain"]),
        "capture_landing_pit": capture["landing_pit"],
        "capture_lands_in_store": bool(capture["lands_in_store"]),
        "capture_opposite_pit": int(capture["captured_opposite_pit"]),
        "extra_turn_landing_pit": extra_turn["landing_pit"],
        "extra_turn_lands_in_store": bool(extra_turn["lands_in_store"]),
        "teacher_prefers_capture": True,
    }


MOTIF_SCORE_CAPTURE_EXTRA_TURN_PRESENT = 3
MOTIF_SCORE_PREFERENCE_SHAPE_MATCH = 3
MOTIF_SCORE_CAPTURE_GAIN_SUPERIOR = 2
MOTIF_SCORE_CAPTURE_LANDING_MATCH = 1
MOTIF_SCORE_CAPTURE_STORE_FLAG_MATCH = 1
MOTIF_SCORE_CAPTURE_OPPOSITE_MATCH = 1
MOTIF_SCORE_EXTRA_TURN_LANDING_MATCH = 1
MOTIF_SCORE_EXTRA_TURN_STORE_FLAG_MATCH = 1
MOTIF_SCORE_TEACHER_SELECTED_IS_EXTRA_TURN = 3
MOTIF_PROTECTIVE_SCORE_THRESHOLD = 11


def score_candidate_support_row(row: dict, signature: dict) -> dict:
    conflict = choose_candidate_conflict_moves(row)
    features = {
        feature["move"]: feature for feature in legal_move_features(row["raw_state"])
    }
    capture = features[conflict["capture_move"]]
    extra_turn = features[conflict["extra_turn_move"]]
    policy = row["policy"]
    score = 0

    if capture["capture"] and extra_turn["extra_turn"]:
        score += MOTIF_SCORE_CAPTURE_EXTRA_TURN_PRESENT
    if float(policy[extra_turn["move"]]) > float(policy[capture["move"]]):
        score += MOTIF_SCORE_PREFERENCE_SHAPE_MATCH
    if int(capture["store_gain"]) > int(extra_turn["store_gain"]):
        score += MOTIF_SCORE_CAPTURE_GAIN_SUPERIOR
    if bool(capture["lands_in_store"]) == bool(signature["capture_lands_in_store"]):
        score += MOTIF_SCORE_CAPTURE_STORE_FLAG_MATCH
    if int(capture["landing_pit"]) == int(signature["capture_landing_pit"]):
        score += MOTIF_SCORE_CAPTURE_LANDING_MATCH
    if int(capture["captured_opposite_pit"]) == int(signature["capture_opposite_pit"]):
        score += MOTIF_SCORE_CAPTURE_OPPOSITE_MATCH
    if bool(extra_turn["lands_in_store"]) == bool(
        signature["extra_turn_lands_in_store"]
    ):
        score += MOTIF_SCORE_EXTRA_TURN_STORE_FLAG_MATCH
    if (
        extra_turn["landing_pit"] is not None
        and extra_turn["landing_pit"] == signature["extra_turn_landing_pit"]
    ):
        score += MOTIF_SCORE_EXTRA_TURN_LANDING_MATCH
    if int(row.get("teacher_selected_move", -1)) == int(extra_turn["move"]):
        score += MOTIF_SCORE_TEACHER_SELECTED_IS_EXTRA_TURN

    return {
        "score": score,
        "motif_protective": score >= MOTIF_PROTECTIVE_SCORE_THRESHOLD,
        "capture_move": int(capture["move"]),
        "extra_turn_move": int(extra_turn["move"]),
    }


def build_forensic_reference_moves(rows: list[dict]) -> dict[str, int]:
    reference_moves: dict[str, int] = {}
    for row in rows:
        canonical_state = row.get("canonical_state")
        if not isinstance(canonical_state, str):
            raw_state = row.get("state")
            if isinstance(raw_state, dict):
                canonical_state = canonical_state_key(raw_state)
        reference_move = row.get("reference_move")
        bucket = row.get("bucket")
        if bucket != "capture_available" or not isinstance(canonical_state, str):
            continue
        if isinstance(reference_move, int) and not isinstance(reference_move, bool):
            reference_moves[canonical_state] = reference_move
    return reference_moves


def load_forensic_reference_moves(path: Path | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    systems = payload.get("systems") if isinstance(payload, dict) else None
    if not isinstance(systems, dict):
        return {}
    challenger = systems.get("challenger")
    if not isinstance(challenger, dict):
        return {}
    rows = challenger.get("rows")
    if not isinstance(rows, list):
        return {}
    return build_forensic_reference_moves(
        [row for row in rows if isinstance(row, dict)]
    )


def select_capture_rows(
    rows: list[dict],
    *,
    limit: int = 4,
    excluded_canonical_states: set[str] | None = None,
    forensic_reference_moves: dict[str, int] | None = None,
    regression_signature: dict | None = None,
) -> list[dict]:
    excluded_canonical_states = excluded_canonical_states or set()
    forensic_reference_moves = forensic_reference_moves or {}
    capture_rows = []
    for row in rows:
        if row.get("bucket") != "capture_available":
            continue
        normalized = validate_tactical_capture_row(row)
        if normalized["canonical_state"] in excluded_canonical_states:
            continue
        normalized.setdefault(
            "raw_state",
            row.get("raw_state")
            or raw_state_from_canonical_state(normalized["canonical_state"]),
        )
        scored = None
        if (
            regression_signature is not None
            and normalized.get("raw_state")
            and "teacher_selected_move" in normalized
        ):
            try:
                scored = score_candidate_support_row(normalized, regression_signature)
            except (KeyError, TypeError, ValueError):
                continue
            normalized["motif_support_score"] = scored["score"]
            normalized["motif_protective"] = scored["motif_protective"]
            if not scored["motif_protective"]:
                continue
        reference_move = forensic_reference_moves.get(normalized["canonical_state"])
        contradictory = (
            reference_move is not None
            and copied_row_policy_peak(normalized) != reference_move
        )
        if contradictory:
            if scored is None:
                continue
        try:
            normalized["capture_shape_key"] = capture_shape_key(normalized)
        except (KeyError, TypeError, ValueError):
            continue
        capture_rows.append(normalized)

    def support_row_sort_key(row: dict) -> tuple:
        return (
            -float(row.get("motif_support_score", 0.0)),
            -float(row.get("priority_score", 0.0)),
            str(row.get("canonical_state", "")),
        )

    def teacher_capture_margin(row: dict) -> float:
        try:
            policy = row["policy"]
            conflict = choose_candidate_conflict_moves(row)
        except (KeyError, TypeError, ValueError):
            return 0.0
        return float(policy[conflict["capture_move"]]) - float(
            policy[conflict["extra_turn_move"]]
        )

    def copied_row_sort_key(row: dict) -> tuple:
        return (
            -float(row.get("motif_support_score", 0.0)),
            -float(row.get("priority_score", 0.0)),
            -teacher_capture_margin(row),
            str(row.get("canonical_state", "")),
        )

    eligible_rows = sorted(capture_rows, key=support_row_sort_key)
    support_row = eligible_rows[:1]
    remaining_rows = sorted(eligible_rows[1:], key=copied_row_sort_key)

    selected = list(support_row)
    used_states = {row["canonical_state"] for row in selected}
    used_shapes = {row["capture_shape_key"] for row in selected}

    while len(selected) < limit:
        added = False
        for row in remaining_rows:
            if row["canonical_state"] in used_states:
                continue
            shape_key = row["capture_shape_key"]
            if shape_key in used_shapes:
                continue
            selected.append(row)
            used_states.add(row["canonical_state"])
            used_shapes.add(shape_key)
            added = True
            if len(selected) == limit:
                break
        if len(selected) == limit or not added:
            break

    for row in remaining_rows:
        if len(selected) == limit:
            break
        if row["canonical_state"] in used_states:
            continue
        selected.append(row)
        used_states.add(row["canonical_state"])

    return selected[:limit]


def build_regression_row(*, regression_position: dict, teacher_labeler) -> dict:
    built = teacher_labeler(regression_position["state"])
    if built is None:
        raise ValueError(
            "teacher_labeler must return a labeled replay row for capture protection"
        )
    if int(built.get("teacher_selected_move", -1)) != int(
        regression_position["expected_move"]
    ):
        raise ValueError(
            "teacher_selected_move must equal expected_move for capture protection row"
        )

    row = dict(built)
    bucket = row.get("bucket")
    if bucket is None:
        row["bucket"] = "capture_available"
    elif bucket != "capture_available":
        raise ValueError("regression row bucket must equal capture_available")

    bucket_group = row.get("bucket_group")
    if bucket_group is None:
        row["bucket_group"] = "tactical"
    elif bucket_group != "tactical":
        raise ValueError("regression row bucket_group must equal tactical")

    row["capture_protection_source"] = regression_position["id"]
    return row


def _require_present(row: dict, field: str) -> object:
    if field not in row:
        raise ValueError(f"{field} is required for tactical capture protection rows")
    return row[field]


def _sanitize_source_artifacts(row: dict) -> dict:
    source_artifacts = row.get("source_artifacts")
    if not isinstance(source_artifacts, list):
        return row

    sanitized = []
    for item in source_artifacts:
        if isinstance(item, str) and item:
            sanitized.append(Path(item).name if Path(item).is_absolute() else item)
    if sanitized:
        row["source_artifacts"] = list(dict.fromkeys(sanitized))
    else:
        row["source_artifacts"] = [CAPTURE_PROTECTION_SOURCE_ARTIFACT]
    return row


def validate_tactical_capture_row(row: dict) -> dict:
    normalized = dict(row)
    canonical_state = _require_present(normalized, "canonical_state")
    if not canonical_state:
        raise ValueError(
            "canonical_state is required for tactical capture protection rows"
        )

    if _require_present(normalized, "bucket") != "capture_available":
        raise ValueError("bucket must equal capture_available")
    if _require_present(normalized, "bucket_group") != "tactical":
        raise ValueError("bucket_group must equal tactical")
    if (
        _require_present(normalized, "input_encoding")
        != CAPTURE_PROTECTION_INPUT_ENCODING
    ):
        raise ValueError(
            f"input_encoding must equal {CAPTURE_PROTECTION_INPUT_ENCODING}"
        )
    if (
        _require_present(normalized, "policy_target_mode")
        != CAPTURE_PROTECTION_POLICY_TARGET_MODE
    ):
        raise ValueError(
            f"policy_target_mode must equal {CAPTURE_PROTECTION_POLICY_TARGET_MODE}"
        )
    if (
        _require_present(normalized, "value_target_mode")
        != CAPTURE_PROTECTION_VALUE_TARGET_MODE
    ):
        raise ValueError(
            f"value_target_mode must equal {CAPTURE_PROTECTION_VALUE_TARGET_MODE}"
        )

    _require_present(normalized, "state")
    _require_present(normalized, "policy")
    _require_present(normalized, "value")
    return _sanitize_source_artifacts(normalized)


def build_capture_protection_dataset(
    *,
    regression_positions_path: Path,
    tactical_replay_path: Path,
    out_path: Path,
    teacher_labeler,
    forensic_suite_path: Path | None = None,
) -> list[dict]:
    regression_positions = load_json(regression_positions_path)
    if not regression_positions:
        raise ValueError("regression fixture must contain at least one position")
    regression_position = regression_positions[0]
    tactical_rows = load_jsonl(tactical_replay_path)
    forensic_reference_moves = load_forensic_reference_moves(forensic_suite_path)
    if not forensic_reference_moves:
        forensic_reference_moves = build_forensic_reference_moves(tactical_rows)

    protected_rows = [
        build_regression_row(
            regression_position=regression_position, teacher_labeler=teacher_labeler
        )
    ]
    regression_signature = extract_regression_motif_signature(
        regression_position["state"],
        expected_move=int(regression_position["expected_move"]),
    )

    protected_rows.extend(
        select_capture_rows(
            tactical_rows,
            excluded_canonical_states={
                row.get("canonical_state", "") for row in protected_rows
            },
            forensic_reference_moves=forensic_reference_moves,
            regression_signature=regression_signature,
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(row) for row in protected_rows)
        + ("\n" if protected_rows else ""),
        encoding="utf-8",
    )
    return protected_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regression-positions", required=True)
    parser.add_argument("--tactical-replay", required=True)
    parser.add_argument("--forensic-suite")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _build_regression_mined_row(
    raw_state: dict, *, source_id: str, move_number: int | None
) -> dict:
    game = KalahGame.from_state(raw_state)
    return {
        "canonical_state": canonical_state_key(raw_state),
        "state": raw_state,
        "side_to_move": int(raw_state["current_player"]),
        "legal_moves": game.possible_moves(),
        "selection_reasons": ["capture_protection_regression"],
        "source_artifacts": ["test/fixtures/ai/superhuman_regression_positions.json"],
        "source_runs": [{"kind": "capture_protection_regression", "id": source_id}],
        "priority_score": 1.0,
        "ply": int(move_number) if move_number is not None else 1,
    }


def _prove_capture_available_bucket(raw_state: dict) -> tuple[str, str]:
    game = KalahGame.from_state(raw_state)
    legal_moves = game.possible_moves()
    if not legal_moves:
        raise ValueError("regression state must have at least one legal move")
    if not any(
        bool(label_tactical_states._move_features(game, move)["capture"])
        for move in legal_moves
    ):
        raise ValueError(
            "default regression labeling must prove a capture_available move"
        )
    bucket = "capture_available"
    return bucket, label_tactical_states.bucket_group(bucket)


def teacher_label_regression_row(
    raw_state: dict,
    *,
    source_id: str = "capture_protection_regression",
    move_number: int | None = None,
) -> dict:
    policy_target_mode = train.normalize_policy_target_mode(
        CAPTURE_PROTECTION_POLICY_TARGET_MODE
    )
    value_target_mode = train.normalize_value_target_mode(
        CAPTURE_PROTECTION_VALUE_TARGET_MODE
    )
    if CAPTURE_PROTECTION_INPUT_ENCODING not in SUPPORTED_INPUT_ENCODINGS:
        raise ValueError(
            f"unsupported input encoding: {CAPTURE_PROTECTION_INPUT_ENCODING}"
        )

    labeled = label_tactical_states.label_row(
        _build_regression_mined_row(
            raw_state, source_id=source_id, move_number=move_number
        ),
        policy_simulations=CAPTURE_PROTECTION_POLICY_SIMULATIONS,
        value_simulations=CAPTURE_PROTECTION_VALUE_SIMULATIONS,
        seed=CAPTURE_PROTECTION_SEED,
        policy_target_mode=policy_target_mode,
        value_target_mode=value_target_mode,
        input_encoding=CAPTURE_PROTECTION_INPUT_ENCODING,
    )
    bucket, bucket_group = _prove_capture_available_bucket(raw_state)
    labeled["bucket"] = bucket
    labeled["bucket_group"] = bucket_group
    return labeled


def main() -> None:
    args = parse_args()
    regression_positions = load_json(Path(args.regression_positions))
    regression_position = regression_positions[0] if regression_positions else {}
    build_capture_protection_dataset(
        regression_positions_path=Path(args.regression_positions),
        tactical_replay_path=Path(args.tactical_replay),
        out_path=Path(args.out),
        teacher_labeler=lambda raw_state: teacher_label_regression_row(
            raw_state,
            source_id=str(
                regression_position.get("id", "capture_protection_regression")
            ),
            move_number=regression_position.get("move_number"),
        ),
        forensic_suite_path=Path(args.forensic_suite) if args.forensic_suite else None,
    )


if __name__ == "__main__":
    main()
