from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.alphazero_lite.kalah_rules import KalahGame, PITS_PER_PLAYER


SCHEMA = "azlite_capture_002_003_rule_collision_diagnostic_v1"
REFERENCE_SCHEMA = "azlite_forensic_references_v1"
ARBITRATION_SCHEMA = "azlite_capture_002_003_search_policy_arbitration_v1"
ROW_IDS = ["capture_available-002", "capture_available-003"]
RULE_COMPARISON_FIELDS = [
    "capture_legal",
    "extra_turn_available",
    "store_gain",
    "landing_pit",
    "lands_in_store",
    "captured_opposite_pit",
    "post_move_empty_pit_pattern",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a move-level rule-collision diagnostic for capture_available-002/003"
    )
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--tracked-arbitration", type=Path, required=True)
    parser.add_argument("--broader-arbitration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_id(rows: list[dict]) -> dict[str, dict]:
    return {str(row["id"]): row for row in rows}


def validate_reference_artifact(payload: dict) -> dict[str, dict]:
    if payload.get("schema") != REFERENCE_SCHEMA:
        raise ValueError(
            f"reference artifact must use schema {REFERENCE_SCHEMA}: {payload.get('schema')}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("reference artifact rows must be a list")
    resolved_rows = rows_by_id(rows)
    for row_id in ROW_IDS:
        if row_id not in resolved_rows:
            raise ValueError(f"reference artifact missing row {row_id}")
    return resolved_rows


def validate_arbitration(payload: dict, *, label: str) -> dict[str, dict]:
    if payload.get("schema") != ARBITRATION_SCHEMA:
        raise ValueError(
            f"{label} arbitration artifact must use schema {ARBITRATION_SCHEMA}: {payload.get('schema')}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, dict):
        raise ValueError(f"{label} arbitration rows must be an object")
    for row_id in ROW_IDS:
        if row_id not in rows:
            raise ValueError(f"{label} arbitration missing row {row_id}")
    return rows


def _relative_index_for_player(*, absolute_index: int, player: int) -> int | None:
    start = player * PITS_PER_PLAYER
    end = start + PITS_PER_PLAYER
    if start <= absolute_index < end:
        return absolute_index - start
    return None


def _empty_pits(pits: list[int]) -> list[int]:
    return [index for index, seeds in enumerate(pits) if seeds == 0]


def simulate_move_rule_features(*, state: dict, move: int) -> dict:
    game = KalahGame.from_state(state)
    original_player = game.current_player
    original_store = game.captured_seeds[original_player]
    absolute_index = game.pit_index(move)
    if move not in game.possible_moves():
        raise ValueError(f"move {move} must be legal for provided state")

    pit_seeds = game.pits[absolute_index]
    game.pits[absolute_index] = 0
    last_pit_index, extra_turn = game._seeding(absolute_index, pit_seeds, original_player)

    landing_pit = None
    lands_in_store = bool(extra_turn)
    if not extra_turn:
        landing_pit = _relative_index_for_player(
            absolute_index=last_pit_index,
            player=game.pit_owner(last_pit_index),
        )

    captured_opposite_pit = None
    capture_legal = False
    if not extra_turn and game.pit_owner(last_pit_index) == original_player:
        if game.pits[last_pit_index] == 1:
            opposite_index = game.opposite_pit_index(last_pit_index)
            if game.pits[opposite_index] > 0:
                capture_legal = True
                captured_opposite_pit = _relative_index_for_player(
                    absolute_index=opposite_index,
                    player=game.pit_owner(opposite_index),
                )

    if not extra_turn:
        game._capture(last_pit_index)
        game.current_player = game.opposite_player()

    if game.over():
        game._after_game_over()

    final_state = game.to_state()
    if original_player == 0:
        attacking_side = list(final_state["player_pits"])
        defending_side = list(final_state["opponent_pits"])
    else:
        attacking_side = list(final_state["opponent_pits"])
        defending_side = list(final_state["player_pits"])

    return {
        "move": move,
        "capture_legal": capture_legal,
        "extra_turn_available": extra_turn,
        "store_gain": game.captured_seeds[original_player] - original_store,
        "landing_pit": landing_pit,
        "lands_in_store": lands_in_store,
        "captured_opposite_pit": captured_opposite_pit,
        "post_move_empty_pit_pattern": {
            "attacking_side": _empty_pits(attacking_side),
            "defending_side": _empty_pits(defending_side),
        },
        "post_move_state": final_state,
    }


def search_run_summary(*, arbitration_row: dict) -> dict:
    search_view = arbitration_row.get("search_view") or {}
    value_view = arbitration_row.get("value_view") or {}
    policy_view = arbitration_row.get("policy_view") or {}
    return {
        "searched_selected_move": search_view.get("searched_selected_move"),
        "reference_move_visit_share": search_view.get("reference_move_visit_share"),
        "selected_move_visit_share": search_view.get("selected_move_visit_share"),
        "visit_distribution": search_view.get("visit_distribution"),
        "child_stats": search_view.get("child_stats"),
        "policy_top_move": policy_view.get("top_move"),
        "reference_move_probability": policy_view.get("reference_move_probability"),
        "selected_minus_reference_margin": policy_view.get(
            "selected_minus_reference_margin"
        ),
        "reference_move_q_value": value_view.get("reference_move_q_value"),
        "selected_move_q_value": value_view.get("selected_move_q_value"),
        "selected_minus_reference_q_margin": value_view.get(
            "selected_minus_reference_q_margin"
        ),
    }


def build_row_payload(
    *,
    row_id: str,
    reference_row: dict,
    tracked_row: dict,
    broader_row: dict,
) -> dict:
    raw_state = dict(reference_row["state"])
    reference_move = int(reference_row["reference_move"])
    tracked_selected_move = int(
        (tracked_row.get("search_view") or {}).get("searched_selected_move")
    )
    broader_selected_move = int(
        (broader_row.get("search_view") or {}).get("searched_selected_move")
    )

    move_features = {
        "reference_move": simulate_move_rule_features(
            state=raw_state, move=reference_move
        ),
        "tracked_selected_move": simulate_move_rule_features(
            state=raw_state, move=tracked_selected_move
        ),
        "broader_selected_move": simulate_move_rule_features(
            state=raw_state, move=broader_selected_move
        ),
    }

    return {
        "row_id": row_id,
        "resolved_raw_state": raw_state,
        "legal_moves": list(tracked_row["legal_moves"]),
        "reference_move": reference_move,
        "teacher_value": reference_row.get("teacher_value"),
        "teacher_child_stats": list(reference_row.get("child_stats") or []),
        "search_runs": {
            "tracked": search_run_summary(arbitration_row=tracked_row),
            "broader": search_run_summary(arbitration_row=broader_row),
        },
        "move_rule_features": move_features,
    }


def _side_by_side(
    *,
    left_label: str,
    left_features: dict,
    right_label: str,
    right_features: dict,
) -> dict:
    return {
        field: {
            left_label: left_features.get(field),
            right_label: right_features.get(field),
        }
        for field in RULE_COMPARISON_FIELDS
    }


def build_paired_comparison(*, rows: dict[str, dict]) -> dict:
    row_002 = rows[ROW_IDS[0]]
    row_003 = rows[ROW_IDS[1]]
    return {
        "reference_move_side_by_side": _side_by_side(
            left_label=ROW_IDS[0],
            left_features=row_002["move_rule_features"]["reference_move"],
            right_label=ROW_IDS[1],
            right_features=row_003["move_rule_features"]["reference_move"],
        ),
        "tracked_selected_move_side_by_side": _side_by_side(
            left_label=ROW_IDS[0],
            left_features=row_002["move_rule_features"]["tracked_selected_move"],
            right_label=ROW_IDS[1],
            right_features=row_003["move_rule_features"]["tracked_selected_move"],
        ),
        "broader_selected_move_side_by_side": _side_by_side(
            left_label=ROW_IDS[0],
            left_features=row_002["move_rule_features"]["broader_selected_move"],
            right_label=ROW_IDS[1],
            right_features=row_003["move_rule_features"]["broader_selected_move"],
        ),
        "row_002_reference_vs_tracked_selected": _side_by_side(
            left_label="reference_move",
            left_features=row_002["move_rule_features"]["reference_move"],
            right_label="tracked_selected_move",
            right_features=row_002["move_rule_features"]["tracked_selected_move"],
        ),
        "row_002_reference_vs_broader_selected": _side_by_side(
            left_label="reference_move",
            left_features=row_002["move_rule_features"]["reference_move"],
            right_label="broader_selected_move",
            right_features=row_002["move_rule_features"]["broader_selected_move"],
        ),
        "row_003_reference_vs_tracked_selected": _side_by_side(
            left_label="reference_move",
            left_features=row_003["move_rule_features"]["reference_move"],
            right_label="tracked_selected_move",
            right_features=row_003["move_rule_features"]["tracked_selected_move"],
        ),
        "row_003_reference_vs_broader_selected": _side_by_side(
            left_label="reference_move",
            left_features=row_003["move_rule_features"]["reference_move"],
            right_label="broader_selected_move",
            right_features=row_003["move_rule_features"]["broader_selected_move"],
        ),
    }


def infer_diagnosis(*, rows: dict[str, dict]) -> dict:
    row_002 = rows[ROW_IDS[0]]
    row_003 = rows[ROW_IDS[1]]
    row_002_reference = row_002["move_rule_features"]["reference_move"]
    row_002_tracked_selected = row_002["move_rule_features"]["tracked_selected_move"]
    row_002_broader_selected = row_002["move_rule_features"]["broader_selected_move"]
    row_003_reference = row_003["move_rule_features"]["reference_move"]
    row_003_tracked_selected = row_003["move_rule_features"]["tracked_selected_move"]
    row_003_broader_selected = row_003["move_rule_features"]["broader_selected_move"]
    row_002_tracked = row_002["search_runs"]["tracked"]
    row_002_broader = row_002["search_runs"]["broader"]

    row_002_wrong_move_is_extra_turn = (
        row_002_tracked_selected["extra_turn_available"]
        and row_002_broader_selected["extra_turn_available"]
    )
    row_002_reference_is_not_extra_turn = not row_002_reference[
        "extra_turn_available"
    ]
    row_003_reference_is_extra_turn = row_003_reference["extra_turn_available"]
    row_003_stays_reference = (
        row_003["search_runs"]["tracked"]["searched_selected_move"]
        == row_003["reference_move"]
        and row_003["search_runs"]["broader"]["searched_selected_move"]
        == row_003["reference_move"]
        and row_003_tracked_selected["extra_turn_available"]
        and row_003_broader_selected["extra_turn_available"]
    )
    row_002_reference_is_not_capture = not row_002_reference["capture_legal"]
    broader_q_prefers_reference = (
        (row_002_broader.get("selected_minus_reference_q_margin") or 0.0) < 0.0
    )
    tracked_q_is_near_neutral = abs(
        row_002_tracked.get("selected_minus_reference_q_margin") or 0.0
    ) <= 0.02

    if (
        row_002_wrong_move_is_extra_turn
        and row_002_reference_is_not_extra_turn
        and row_003_reference_is_extra_turn
        and row_003_stays_reference
        and row_002_reference_is_not_capture
    ):
        evidence = [
            "Row 002 fails only when the search-selected move changes from the no-extra-turn reference move 4 to extra-turn move 2.",
            "Row 003 remains stable because its reference move 1 is itself an extra-turn move, so the same bias does not create a conflict there.",
        ]
        if broader_q_prefers_reference:
            evidence.append(
                "In the broader run, child Q already prefers the reference move on row 002, but visits still collapse onto move 2, which points to extra-turn prior pressure surviving search rather than a hidden no-extra-turn benefit being missed."
            )
        elif tracked_q_is_near_neutral:
            evidence.append(
                "In the tracked run, child Q is nearly neutral between the two moves on row 002, so an extra-turn-biased prior is sufficient to tip search toward move 2."
            )
        return {
            "best_explanation": "extra_turn_overvaluation",
            "recommendation": "rule_conditioned_policy_artifact_redesign",
            "evidence": evidence,
        }

    if row_002_reference["capture_legal"] and not row_002_tracked_selected["capture_legal"]:
        return {
            "best_explanation": "no_extra_turn_capture_undervaluation",
            "recommendation": "row_pair_diagnostic_replay_artifact",
            "evidence": [
                "Row 002 reference move keeps a capture branch that the selected move drops."
            ],
        }

    if (
        row_002_reference["extra_turn_available"]
        == row_002_tracked_selected["extra_turn_available"]
        and row_003_reference["extra_turn_available"]
        == row_003_tracked_selected["extra_turn_available"]
    ):
        return {
            "best_explanation": "capture_shape_aliasing",
            "recommendation": "search_selection_instrumentation_pass",
            "evidence": [
                "The row split persists without an extra-turn polarity change, so the collision is more likely tied to move-shape aliasing inside search telemetry."
            ],
        }

    return {
        "best_explanation": "another_rule_conditioned_mechanism",
        "recommendation": "search_selection_instrumentation_pass",
        "evidence": [
            "The current row-pair features do not isolate one of the narrower rule-conditioned mechanisms with enough confidence."
        ],
    }


def build_payload(
    *,
    reference_artifact_path: Path,
    tracked_arbitration_path: Path,
    broader_arbitration_path: Path,
) -> dict:
    reference_artifact = load_json(reference_artifact_path)
    tracked_arbitration = load_json(tracked_arbitration_path)
    broader_arbitration = load_json(broader_arbitration_path)

    reference_rows = validate_reference_artifact(reference_artifact)
    tracked_rows = validate_arbitration(tracked_arbitration, label="tracked")
    broader_rows = validate_arbitration(broader_arbitration, label="broader")

    rows = {
        row_id: build_row_payload(
            row_id=row_id,
            reference_row=reference_rows[row_id],
            tracked_row=tracked_rows[row_id],
            broader_row=broader_rows[row_id],
        )
        for row_id in ROW_IDS
    }

    return {
        "schema": SCHEMA,
        "source_artifacts": {
            "reference_artifact": str(reference_artifact_path),
            "tracked_arbitration": str(tracked_arbitration_path),
            "broader_arbitration": str(broader_arbitration_path),
        },
        "rows": rows,
        "paired_comparison": build_paired_comparison(rows=rows),
        "diagnosis": infer_diagnosis(rows=rows),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(
        reference_artifact_path=args.reference_artifact,
        tracked_arbitration_path=args.tracked_arbitration,
        broader_arbitration_path=args.broader_arbitration,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact_path": str(args.out),
                "schema": SCHEMA,
                "best_explanation": payload["diagnosis"]["best_explanation"],
                "recommendation": payload["diagnosis"]["recommendation"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
