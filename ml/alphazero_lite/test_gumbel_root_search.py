from __future__ import annotations

import numpy as np

from ml.alphazero_lite.gumbel_root_search import (
    CountingEvaluator,
    run_gumbel_root_search,
    run_puct_root_search,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import Evaluator, standard_start_state


class FlatEvaluator(Evaluator):
    def evaluate(self, _game: KalahGame) -> tuple[np.ndarray, float]:
        return np.full(6, 1 / 6, dtype=np.float32), 0.0


def test_both_lanes_use_exact_neural_budget_and_legal_move() -> None:
    game = KalahGame.from_state(standard_start_state())
    for run in (run_puct_root_search, run_gumbel_root_search):
        result = run(game, FlatEvaluator(), seed=267101, budget=32)
        assert result.evaluator_calls == 32
        assert result.selected_move in game.possible_moves()
        assert result.all_legal_actions_initially_visited


def test_gumbel_lane_is_seed_deterministic() -> None:
    game = KalahGame.from_state(standard_start_state())
    left = run_gumbel_root_search(game, FlatEvaluator(), seed=267102, budget=32)
    right = run_gumbel_root_search(game, FlatEvaluator(), seed=267102, budget=32)
    assert left == right


def test_counting_evaluator_counts_each_call() -> None:
    evaluator = CountingEvaluator(FlatEvaluator())
    evaluator.evaluate(KalahGame.from_state(standard_start_state()))
    assert evaluator.calls == 1
