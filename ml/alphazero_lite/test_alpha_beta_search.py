from __future__ import annotations

import numpy as np
import pytest

from ml.alphazero_lite.alpha_beta_search import (
    ArtifactValueEvaluator,
    SearchBudgetExceeded,
    run_alpha_beta_search,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import standard_start_state


class FlatScalar:
    lane = "test"

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def evaluate_value(self, _game: KalahGame) -> float:
        return self.value


class PolicyNoiseEvaluator:
    def __init__(self, policy: list[float]) -> None:
        self.policy = np.asarray(policy, dtype=np.float32)

    def evaluate(self, _game: KalahGame):
        return self.policy, 0.125


def tiny_game() -> KalahGame:
    return KalahGame(
        pits=[1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        captured_seeds=[20, 20],
        current_player=0,
    )


def brute_force(game: KalahGame, root_player: int) -> float:
    if game.over():
        return (
            0.0
            if game.winner is None
            else (1.0 if game.winner == root_player else -1.0)
        )
    values = []
    for move in game.possible_moves():
        child = game.clone()
        child.move(child.pit_index(move))
        values.append(brute_force(child, root_player))
    return max(values) if game.current_player == root_player else min(values)


def test_legal_move_and_deterministic_tie_break() -> None:
    game = KalahGame.from_state(standard_start_state())
    result = run_alpha_beta_search(game, FlatScalar(), leaf_evaluation_budget=32)
    assert result.selected_move in game.possible_moves()
    assert result.selected_move == min(game.possible_moves())


def test_policy_arrays_cannot_change_scalar_search() -> None:
    game = tiny_game()
    left = run_alpha_beta_search(
        game,
        ArtifactValueEvaluator(PolicyNoiseEvaluator([1, 0, 0, 0, 0, 0])),
        leaf_evaluation_budget=128,
    )
    right = run_alpha_beta_search(
        game,
        ArtifactValueEvaluator(PolicyNoiseEvaluator([0, 0, 0, 0, 0, 1])),
        leaf_evaluation_budget=128,
    )
    assert left.selected_move == right.selected_move
    assert left.root_action_values == right.root_action_values


def test_budget_depth_and_terminal_contracts() -> None:
    game = KalahGame.from_state(standard_start_state())
    result = run_alpha_beta_search(game, FlatScalar(), leaf_evaluation_budget=12)
    assert result.leaf_evaluator_calls <= 12
    assert result.completed_depth == 1
    assert result.deeper_iteration_abandoned
    with pytest.raises(SearchBudgetExceeded):
        run_alpha_beta_search(game, FlatScalar(), leaf_evaluation_budget=1)
    for winner, expected in ((0, 1.0), (None, 0.0), (1, -1.0)):
        terminal = KalahGame(
            pits=[0] * 12,
            captured_seeds=[24, 24],
            current_player=0,
            winner=winner,
            _over=True,
        )
        search = __import__(
            "ml.alphazero_lite.alpha_beta_search", fromlist=["AlphaBetaSearch"]
        ).AlphaBetaSearch(FlatScalar(), leaf_evaluation_budget=1)
        search.root_player = 0
        assert search._terminal_value(terminal) == expected


def test_extra_turn_opponent_minimization_tt_and_oracle() -> None:
    game = tiny_game()
    left = run_alpha_beta_search(game, FlatScalar(), leaf_evaluation_budget=256)
    right = run_alpha_beta_search(game, FlatScalar(), leaf_evaluation_budget=256)
    assert left.__dict__ | {"runtime_seconds": 0.0} == right.__dict__ | {
        "runtime_seconds": 0.0
    }
    assert left.root_action_values[left.selected_move] == brute_force(
        game.clone(), game.current_player
    )
    assert left.transposition_table_hits > 0
    from ml.alphazero_lite.alpha_beta_search import AlphaBetaSearch

    without_tt = AlphaBetaSearch(
        FlatScalar(), leaf_evaluation_budget=256, use_transposition_table=False
    ).search(game)
    assert left.selected_move == without_tt.selected_move
    assert left.root_action_values == without_tt.root_action_values
