import unittest

import numpy as np

from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.exact_root_decision import (
    ExactRootCoverageGap,
    exact_root_decision,
    exact_root_profile_fields,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import build_eval_search_options, build_search_profile


class FixedEvaluator:
    def __init__(self, priors):
        self.priors = np.asarray(priors, dtype=np.float32)
        self.calls = 0

    def evaluate(self, _game):
        self.calls += 1
        return self.priors, 0.0


class MissingMarginTablebase(EndgameTablebase):
    def final_margin(self, _game, _perspective_player):
        return None


class FixedActionOracle:
    def __init__(self, margins):
        self.margins = margins
        self.calls = 0

    def root_action_margins(self, _game, _perspective_player):
        self.calls += 1
        return self.margins


class ExactRootDecisionTest(unittest.TestCase):
    def test_sparse_endgame_023_uses_exact_best_margin(self):
        game = KalahGame.from_state(
            {
                "player_pits": [0, 6, 0, 0, 1, 2],
                "opponent_pits": [0, 0, 3, 0, 0, 4],
                "player_store": 13,
                "opponent_store": 19,
                "current_player": 0,
            }
        )
        decision = exact_root_decision(
            game,
            FixedEvaluator([0.0, 0.2, 0.0, 0.0, 0.3, 0.5]),
            tablebase=EndgameTablebase(),
            threshold=16,
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual({1: -12, 4: 0, 5: -6}, decision.action_margins)
        self.assertEqual([4], decision.optimal_moves)
        self.assertEqual(4, decision.selected_move)
        self.assertEqual(0, decision.root_player)
        self.assertEqual(16, decision.active_pit_stones)

    def test_ineligible_root_does_not_evaluate_network(self):
        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        evaluator = FixedEvaluator([1 / 6] * 6)

        self.assertIsNone(
            exact_root_decision(
                game, evaluator, tablebase=EndgameTablebase(), threshold=16
            )
        )
        self.assertEqual(0, evaluator.calls)

    def test_ties_use_prior_then_lowest_move(self):
        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 1],
                "opponent_pits": [0, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 20,
                "current_player": 0,
            }
        )
        decision = exact_root_decision(
            game,
            FixedEvaluator([0, 0, 0, 0, 0.5, 0.5]),
            tablebase=EndgameTablebase(),
            threshold=16,
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual([4, 5], decision.optimal_moves)
        self.assertEqual(4, decision.selected_move)

    def test_missing_action_margin_fails_closed(self):
        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 0],
                "opponent_pits": [0, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 19,
                "current_player": 0,
            }
        )
        with self.assertRaisesRegex(ExactRootCoverageGap, "exact_root_coverage_gap"):
            exact_root_decision(
                game,
                FixedEvaluator([1 / 6] * 6),
                tablebase=MissingMarginTablebase(),
                threshold=16,
            )

    def test_native_action_oracle_requires_complete_legal_coverage(self):
        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 1],
                "opponent_pits": [0, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 20,
                "current_player": 0,
            }
        )
        oracle = FixedActionOracle({4: 0})
        with self.assertRaisesRegex(ExactRootCoverageGap, "exact_root_coverage_gap"):
            exact_root_decision(
                game, FixedEvaluator([1 / 6] * 6), tablebase=oracle, threshold=16
            )
        self.assertEqual(1, oracle.calls)

    def test_below_threshold_uses_root_player_one_perspective(self):
        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 0],
                "opponent_pits": [0, 0, 0, 0, 1, 1],
                "player_store": 19,
                "opponent_store": 20,
                "current_player": 1,
            }
        )
        decision = exact_root_decision(
            game,
            FixedEvaluator([0, 0, 0, 0, 0.1, 0.9]),
            tablebase=EndgameTablebase(),
            threshold=16,
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(1, decision.root_player)
        self.assertEqual(2, decision.active_pit_stones)
        self.assertIn(decision.selected_move, decision.optimal_moves)

    def test_extra_turn_action_is_solved_as_a_root_action(self):
        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 1],
                "opponent_pits": [0, 0, 0, 0, 1, 0],
                "player_store": 22,
                "opponent_store": 23,
                "current_player": 0,
            }
        )
        decision = exact_root_decision(
            game,
            FixedEvaluator([0, 0, 0, 0, 0.9, 0.1]),
            tablebase=EndgameTablebase(),
            threshold=16,
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(5, decision.selected_move)

    def test_profile_identity_contains_root_semantics(self):
        profile = exact_root_profile_fields(16)
        self.assertEqual(16, profile["exact_root_solve_threshold"])
        self.assertEqual("final_score_margin", profile["exact_root_objective"])
        self.assertEqual("disabled", profile["exact_leaf_solve_mode"])
        baseline = build_search_profile(
            kind="test",
            player_mode="puct",
            simulations=384,
            c_puct=1.25,
            search_options=build_eval_search_options(),
            extra_fields=exact_root_profile_fields(None),
        )
        handoff = build_search_profile(
            kind="test",
            player_mode="puct",
            simulations=384,
            c_puct=1.25,
            search_options=build_eval_search_options(),
            extra_fields=profile,
        )
        self.assertNotEqual(baseline["hash"], handoff["hash"])
