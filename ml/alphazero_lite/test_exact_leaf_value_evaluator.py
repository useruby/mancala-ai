import random
import unittest

import numpy as np

from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import (
    build_eval_search_options,
    build_search_profile,
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


class FixedTablebase:
    def __init__(self, value, margin):
        self.value = value
        self.margin = margin

    def lookup_cached(self, _game, _perspective_player):
        return self.value

    def lookup(self, _game, _perspective_player):
        return self.value

    def final_margin(self, _game, _perspective_player):
        return self.margin


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

    def test_default_mode_remains_flat_wdl(self):
        position = game(pits=[1] + [0] * 11, stores=(20, 27))
        value = ExactLeafValueEvaluator(
            FixedEvaluator(),
            endgame_tablebase=FixedTablebase(1.0, 1),
            stone_threshold=1,
        ).evaluate(position)[1]
        self.assertEqual(1.0, value)

    def test_wdl_margin_mapping_is_lexicographic_and_bounded(self):
        position = game(pits=[1] + [0] * 11, stores=(20, 27))
        cases = (
            (48, 1.0),
            (1, 0.5104166666666666),
            (0, 0.0),
            (-1, -0.5104166666666666),
            (-48, -1.0),
        )
        values = []
        for margin, expected in cases:
            wdl = 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
            value = ExactLeafValueEvaluator(
                FixedEvaluator(),
                endgame_tablebase=FixedTablebase(wdl, margin),
                stone_threshold=1,
                value_mode="wdl_margin",
            ).evaluate(position)[1]
            self.assertAlmostEqual(expected, value)
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)
            values.append(value)
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], 0.5)
        self.assertEqual(0.0, values[2])
        self.assertLess(values[3], -0.5)
        self.assertGreater(values[3], values[4])

    def test_wdl_margin_fails_closed_for_missing_or_inconsistent_margin(self):
        position = game(pits=[1] + [0] * 11, stores=(20, 27))
        for tablebase, error in (
            (FixedTablebase(1.0, None), "exact_margin_coverage_gap"),
            (FixedTablebase(1.0, -1), "exact_wdl_margin_inconsistent"),
        ):
            evaluator = ExactLeafValueEvaluator(
                FixedEvaluator(),
                endgame_tablebase=tablebase,
                stone_threshold=1,
                value_mode="wdl_margin",
            )
            with self.assertRaisesRegex(ExactSolveCoverageGap, error):
                evaluator.evaluate(position)

    def test_wdl_margin_preserves_priors_and_profile_identity(self):
        position = game(pits=[1] + [0] * 11, stores=(20, 27))
        priors, _ = FixedEvaluator().evaluate(position)
        wrapped_priors, _ = ExactLeafValueEvaluator(
            FixedEvaluator(),
            endgame_tablebase=FixedTablebase(1.0, 1),
            stone_threshold=1,
            value_mode="wdl_margin",
        ).evaluate(position)
        np.testing.assert_array_equal(priors, wrapped_priors)
        options = build_eval_search_options()
        wdl = build_search_profile(
            kind="test",
            player_mode="puct",
            simulations=384,
            c_puct=1.25,
            search_options=options,
            extra_fields={"exact_solve_value_mode": "wdl"},
        )
        margin = build_search_profile(
            kind="test",
            player_mode="puct",
            simulations=384,
            c_puct=1.25,
            search_options=options,
            extra_fields={"exact_solve_value_mode": "wdl_margin"},
        )
        self.assertNotEqual(wdl["hash"], margin["hash"])

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

    def test_bounded_root_child_telemetry_preserves_search_and_checkpoints(self):
        position = game(pits=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        baseline = PUCT(
            ExactLeafValueEvaluator(
                FixedEvaluator(),
                endgame_tablebase=EndgameTablebase(),
                stone_threshold=2,
            ),
            8,
            1.25,
            random.Random(4),
        )
        telemetry = {}
        observed = PUCT(
            ExactLeafValueEvaluator(
                FixedEvaluator(),
                endgame_tablebase=EndgameTablebase(),
                stone_threshold=2,
            ),
            8,
            1.25,
            random.Random(4),
            root_child_telemetry=telemetry,
            root_snapshot_checkpoints={2, 8},
        )
        baseline_visits, baseline_root = baseline.run(position)
        observed_visits, observed_root = observed.run(position)
        np.testing.assert_array_equal(baseline_visits, observed_visits)
        self.assertEqual(
            baseline.select_root_move(baseline_root, position.possible_moves()),
            observed.select_root_move(observed_root, position.possible_moves()),
        )
        self.assertEqual(
            [2, 8], [row["simulation"] for row in telemetry["checkpoints"]]
        )
        self.assertEqual({"0"}, set(telemetry["children"]))
        self.assertEqual(
            8, sum(row["visits"] for row in telemetry["checkpoints"][-1]["children"])
        )
        self.assertEqual(4, telemetry["children"]["0"]["exact_backup_count"])
