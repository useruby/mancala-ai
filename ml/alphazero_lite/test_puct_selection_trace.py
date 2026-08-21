from __future__ import annotations

import random

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import Evaluator, PUCT


class FixedEvaluator(Evaluator):
    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        policy = np.asarray([0.31, 0.19, 0.17, 0.13, 0.11, 0.09], dtype=np.float32)
        return policy, float((sum(game.captured_seeds) % 5 - 2) / 10)


def _game() -> KalahGame:
    return KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )


def test_selection_trace_is_behavior_neutral_and_matches_live_selection() -> None:
    baseline = PUCT(FixedEvaluator(), 48, 1.25, random.Random(9))
    baseline_visits, baseline_root = baseline.run(_game())
    trace: list[dict] = []
    traced = PUCT(
        FixedEvaluator(),
        48,
        1.25,
        random.Random(9),
        selection_trace=trace,
        trace_checkpoints={1, 2, 4, 8, 16, 32, 48},
    )
    traced_visits, traced_root = traced.run(_game())

    assert np.array_equal(traced_visits, baseline_visits)
    assert traced.select_root_move(
        traced_root, sorted(traced_root.children)
    ) == baseline.select_root_move(baseline_root, sorted(baseline_root.children))
    assert traced_root.visit_count == baseline_root.visit_count
    assert traced_root.value_sum == baseline_root.value_sum
    assert (
        traced.root_summary()["terminal_leaf_count"]
        == baseline.root_summary()["terminal_leaf_count"]
    )
    assert (
        traced.root_summary()["nonterminal_leaf_count"]
        == baseline.root_summary()["nonterminal_leaf_count"]
    )
    for move in traced_root.children:
        assert (
            traced_root.children[move].visit_count
            == baseline_root.children[move].visit_count
        )
        assert (
            traced_root.children[move].q_value == baseline_root.children[move].q_value
        )
    for simulation in trace:
        for decision in simulation["selection_path"]:
            winner = max(
                decision["children"], key=lambda entry: entry["selection_score"]
            )
            assert winner["move"] == decision["chosen_move"]
