from __future__ import annotations

import random
import unittest

import numpy as np

from ml.alphazero_lite.arena import BlendedArtifactEvaluator
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import PUCT


class FixedEvaluator:
    def __init__(self, value: float):
        self.value = value

    def evaluate(self, game):
        priors = np.zeros(6, dtype=np.float32)
        priors[game.possible_moves()] = 1.0 / len(game.possible_moves())
        return priors, self.value


class ValueDeltaBlendTest(unittest.TestCase):
    def setUp(self):
        self.game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )

    def test_endpoints_and_midpoint_keep_current_policy(self):
        current = FixedEvaluator(-0.4)
        candidate = FixedEvaluator(0.8)
        current_policy, current_value = current.evaluate(self.game)
        candidate_policy, candidate_value = candidate.evaluate(self.game)
        for alpha, expected in ((0.0, current_value), (0.5, 0.2), (1.0, candidate_value)):
            policy, value = BlendedArtifactEvaluator(current, candidate, alpha).evaluate(
                self.game
            )
            np.testing.assert_array_equal(policy, current_policy)
            self.assertAlmostEqual(value, expected)
        self.assertFalse(np.array_equal(current_policy, candidate_policy) is False)

    def test_terminal_value_bypasses_evaluator_and_perspective_is_once(self):
        terminal = KalahGame(
            pits=[0] * 12, captured_seeds=[30, 18], current_player=0
        )
        evaluator = BlendedArtifactEvaluator(FixedEvaluator(-0.3), FixedEvaluator(0.9), 0.5)
        search = PUCT(evaluator, simulations=1, c_puct=1.25, rng=random.Random(7))
        visits, root = search.run(terminal)
        self.assertEqual(float(np.sum(visits)), 0.0)
        self.assertEqual(root.visit_count, 1)
        self.assertEqual(root.q_value, 0.0)

    def test_deterministic_search_repeats(self):
        evaluator = BlendedArtifactEvaluator(FixedEvaluator(-0.2), FixedEvaluator(0.6), 0.5)
        def run_once():
            search = PUCT(evaluator, simulations=32, c_puct=1.25, rng=random.Random(9))
            visits, root = search.run(self.game)
            return visits, search.select_root_move(root, self.game.possible_moves())
        first_visits, first_move = run_once()
        second_visits, second_move = run_once()
        np.testing.assert_array_equal(first_visits, second_visits)
        self.assertEqual(first_move, second_move)
