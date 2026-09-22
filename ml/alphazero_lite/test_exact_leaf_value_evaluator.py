import random
import unittest

import numpy as np

from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import (
    Evaluator,
    ExactLeafValueEvaluator,
    ExactSolveCoverageGap,
    PUCT,
)


class FixedEvaluator(Evaluator):
    def evaluate(self, _game):
        return np.asarray([0.05, 0.10, 0.15, 0.20, 0.20, 0.30], dtype=np.float32), 0.25


class MissingTablebase:
    def lookup_cached(self, _game, _perspective_player):
        return None

    def lookup(self, _game, _perspective_player):
        return None


def game(*, pits, stores=(20, 19), current_player=0):
    return KalahGame(
        pits=list(pits), captured_seeds=list(stores), current_player=current_player
    )


class ExactLeafValueEvaluatorTest(unittest.TestCase):
    def test_preserves_network_priors_and_replaces_only_value(self):
        position = game(pits=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        network = FixedEvaluator()
        priors, _ = network.evaluate(position)
        wrapped_priors, value = ExactLeafValueEvaluator(
            network, endgame_tablebase=EndgameTablebase(), stone_threshold=1
        ).evaluate(position)
        np.testing.assert_array_equal(priors, wrapped_priors)
        self.assertEqual(1.0, value)

    def test_threshold_uses_pit_stones_and_is_inclusive(self):
        at_threshold = game(pits=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        above_threshold = game(pits=[0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0])
        wrapped = ExactLeafValueEvaluator(
            FixedEvaluator(), endgame_tablebase=EndgameTablebase(), stone_threshold=1
        )
        self.assertEqual(1.0, wrapped.evaluate(at_threshold)[1])
        self.assertEqual(0.25, wrapped.evaluate(above_threshold)[1])
        self.assertEqual(1, wrapped.telemetry["qualifying_leaf_evaluations"])

    def test_terminal_does_not_probe_tablebase(self):
        terminal = game(pits=[0] * 12)
        wrapped = ExactLeafValueEvaluator(
            FixedEvaluator(), endgame_tablebase=EndgameTablebase(), stone_threshold=0
        )
        self.assertEqual(0.25, wrapped.evaluate(terminal)[1])
        self.assertEqual(0, wrapped.telemetry["tablebase_lookup_count"])

    def test_exact_value_uses_side_to_move_perspective_with_extra_turn_state(self):
        position = game(pits=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0], stores=(22, 23))
        extra_turn = position.clone()
        extra_turn.move(extra_turn.pit_index(5))
        self.assertEqual(0, extra_turn.current_player)
        wrapped = ExactLeafValueEvaluator(
            FixedEvaluator(), endgame_tablebase=EndgameTablebase(), stone_threshold=3
        )
        self.assertEqual(0.0, wrapped.evaluate(extra_turn)[1])

    def test_missing_exact_value_fails_closed(self):
        position = game(pits=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        wrapped = ExactLeafValueEvaluator(
            FixedEvaluator(),
            endgame_tablebase=MissingTablebase(),
            stone_threshold=1,
            fail_closed=True,
        )
        with self.assertRaisesRegex(
            ExactSolveCoverageGap, "puct_exact_solve_coverage_gap"
        ):
            wrapped.evaluate(position)
        self.assertEqual(1, len(wrapped.telemetry["coverage_gaps"]))

    def test_disabled_wrapper_reproduces_puct_output(self):
        position = game(pits=[1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
        baseline = PUCT(FixedEvaluator(), 24, 1.25, random.Random(9))
        wrapped = PUCT(
            ExactLeafValueEvaluator(FixedEvaluator()), 24, 1.25, random.Random(9)
        )
        baseline_visits, baseline_root = baseline.run(position)
        wrapped_visits, wrapped_root = wrapped.run(position)
        np.testing.assert_array_equal(baseline_visits, wrapped_visits)
        self.assertEqual(
            baseline.select_root_move(baseline_root, position.possible_moves()),
            wrapped.select_root_move(wrapped_root, position.possible_moves()),
        )

    def test_puct_reports_deterministic_exact_telemetry(self):
        position = game(pits=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        evaluator = ExactLeafValueEvaluator(
            FixedEvaluator(), endgame_tablebase=EndgameTablebase(), stone_threshold=2
        )
        search = PUCT(evaluator, 8, 1.25, random.Random(4))
        search.run(position)
        telemetry = search.root_summary()["exact_leaf_value_telemetry"]
        self.assertGreater(telemetry["tablebase_lookup_count"], 0)
        self.assertEqual(
            telemetry["tablebase_lookup_count"], telemetry["successful_exact_lookups"]
        )
        self.assertEqual(
            telemetry["successful_exact_lookups"], telemetry["exact_leaf_value_count"]
        )
