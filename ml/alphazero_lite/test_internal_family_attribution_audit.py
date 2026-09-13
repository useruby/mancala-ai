from __future__ import annotations

from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_internal_family_attribution_audit import (
    family_signature,
    first_degradation,
    legal_bucket,
    root_utility,
    select_family,
    stage,
)


def _label(value: int, utilities: dict[int, int]) -> dict:
    return {
        "exact_status": "exact_solved",
        "exact_root_value": value,
        "exact_outcome_utilities": {
            str(key): value for key, value in utilities.items()
        },
        "exact_outcome_optimal_actions": [
            key for key, value in utilities.items() if value == 1
        ],
    }


def test_original_root_perspective_uses_player_identity() -> None:
    assert root_utility(1, root_player=0, state_player=0) == 1
    assert root_utility(1, root_player=0, state_player=1) == -1


def test_first_degradation_filters_non_winning_root_action_and_one_per_path() -> None:
    game = KalahGame.from_state(
        {
            "player_pits": [1, 1, 1, 1, 1, 1],
            "opponent_pits": [1, 1, 1, 1, 1, 1],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    state = game.to_state()
    action = 5
    game.move(game.pit_index(action))
    middle = game.to_state()
    degrading = game.possible_moves()[0]
    game.move(game.pit_index(degrading))
    post = game.to_state()
    labels = {
        canonical_state_key(state): _label(1, {action: 1}),
        canonical_state_key(middle): _label(1, {degrading: 0}),
        canonical_state_key(post): _label(0, {}),
    }
    path = {
        "simulation": 3,
        "path": [action, degrading],
        "nodes": [
            {"state": state, "chosen_action": action, "depth": 0, "decision": {}},
            {"state": middle, "chosen_action": degrading, "depth": 1, "decision": {}},
        ],
    }
    event = first_degradation(path, labels, 0)
    assert event is not None
    assert event["simulation"] == 3
    assert event["utility_before"] == 1 and event["utility_after"] == 0


def test_signature_and_buckets_are_rules_only_and_deterministic() -> None:
    event = {
        "utility_after": -1,
        "tactical_type": "capture",
        "optimal_tactical_types": ["neither", "extra_turn"],
        "legal_action_count": 4,
    }
    assert legal_bucket(2) == "2"
    assert family_signature(event) == "win_to_loss|capture|extra_turn+neither|3-4"


def test_family_selection_requires_ten_events_and_prefers_both_roots() -> None:
    base = {"family": "ignored", "opportunities": 2}
    events = [
        {
            **base,
            "case": "44:uniform1200:capture_available-025",
            "utility_after": 0,
            "tactical_type": "capture",
            "optimal_tactical_types": ["neither"],
            "legal_action_count": 3,
        }
        for _ in range(10)
    ]
    events += [
        {
            **base,
            "case": "45:uniform1200:capture_available-018",
            "utility_after": 0,
            "tactical_type": "capture",
            "optimal_tactical_types": ["neither"],
            "legal_action_count": 3,
        }
        for _ in range(10)
    ]
    assert select_family(events) == "win_to_draw|capture|neither|3-4"


def test_stage_classifies_raw_prior_vs_search_without_neural_inputs() -> None:
    event = {"selected_action": 1, "optimal_actions": [0]}
    assert stage({"top_is_optimal": False}, event) == "raw_prior_bad_search_bad"
    assert stage({"top_is_optimal": True}, event) == "raw_prior_good_search_bad"
