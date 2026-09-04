from __future__ import annotations

import pytest

from ml.alphazero_lite.gumbel_root_search import run_puct_root_search
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state
from ml.alphazero_lite.self_play import HeuristicEvaluator, standard_start_state
from ml.alphazero_lite.turn_completion_puct import run_turn_completion_puct


def game(player: int = 0) -> KalahGame:
    return KalahGame.from_state(standard_start_state() | {"current_player": player})


@pytest.mark.parametrize("player", [0, 1])
@pytest.mark.parametrize("budget", [16, 48])
def test_disabled_search_is_byte_equivalent_to_exact_budget_puct(player, budget):
    baseline = run_puct_root_search(
        game(player), HeuristicEvaluator(), seed=275101, budget=budget
    )
    disabled = run_turn_completion_puct(
        game(player), HeuristicEvaluator(), seed=275101, budget=budget, enabled=False
    )
    assert disabled.selected_move == baseline.selected_move
    assert disabled.visits == baseline.visits
    assert disabled.q_values == baseline.q_values
    assert disabled.evaluator_calls == baseline.evaluator_calls == budget
    assert disabled.budget_padding_calls == baseline.budget_padding_calls


def test_non_extra_turn_leaf_stops_at_normal_evaluation():
    state = {
        "player_pits": [1, 1, 0, 0, 0, 0],
        "opponent_pits": [1, 1, 1, 1, 1, 1],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    result = run_turn_completion_puct(
        KalahGame.from_state(state), HeuristicEvaluator(), seed=275101, budget=2
    )
    assert result.extensions_started == 0
    assert result.evaluator_calls == 2


def test_extra_turn_continues_and_discards_intermediate_value():
    # Pit 5 lands in the store, then the remaining pit ends the turn.
    state = {
        "player_pits": [1, 0, 0, 0, 0, 1],
        "opponent_pits": [1, 1, 1, 1, 1, 1],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    result = run_turn_completion_puct(
        KalahGame.from_state(state), HeuristicEvaluator(), seed=275101, budget=4
    )
    assert result.extensions_started >= 1
    assert result.extensions_completed >= 1
    assert result.extra_turn_actions >= 1
    assert result.discarded_intermediate_values
    assert result.final_backed_up_values


def test_terminal_move_is_not_an_extra_turn_and_capture_metadata_is_canonical():
    terminal = {
        "player_pits": [1, 0, 0, 0, 0, 0],
        "opponent_pits": [0] * 6,
        "player_store": 23,
        "opponent_store": 24,
        "current_player": 0,
    }
    consequence = move_consequence_for_state(terminal, 0)
    assert consequence["game_over_after_move"]
    assert not consequence["gives_extra_turn"]
    capture = {
        "player_pits": [1, 0, 0, 0, 0, 0],
        "opponent_pits": [0, 0, 0, 0, 1, 0],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    assert move_consequence_for_state(capture, 0)["produces_capture"]


def test_candidate_is_deterministic_legal_and_never_exceeds_budget():
    for player in (0, 1):
        first = run_turn_completion_puct(
            game(player), HeuristicEvaluator(), seed=275102, budget=32
        )
        second = run_turn_completion_puct(
            game(player), HeuristicEvaluator(), seed=275102, budget=32
        )
        assert first == second
        assert first.selected_move in game(player).possible_moves()
        assert first.evaluator_calls == 32
        assert first.incomplete_due_to_budget <= 1
        assert first.defensive_cap_events == first.repeated_state_events == 0
