from __future__ import annotations

import unittest

from ml.alphazero_lite.run_seed48_target_quality_audit import (
    _selection_change_class,
    canonical_key,
    exact_action_summary,
    paired_bootstrap,
    quality,
)


class Seed48TargetQualityAuditTest(unittest.TestCase):
    def test_tied_exact_actions_are_all_optimal(self) -> None:
        exact = exact_action_summary({0: 3, 1: 3, 2: 1}, root_player=0, stores=[20, 20])
        self.assertEqual([0, 1], exact["exact_optimal_moves"])
        self.assertEqual(0, exact["exact_regret_by_move"][1])
        self.assertEqual(2, exact["exact_regret_by_move"][2])

    def test_player_one_score_is_negated_after_store_conversion(self) -> None:
        exact = exact_action_summary({0: 2}, root_player=1, stores=[12, 8])
        self.assertEqual(-6, exact["exact_score_by_move"][0])

    def test_mass_and_expected_regret_use_all_tied_optima(self) -> None:
        exact = exact_action_summary({0: 2, 1: 2, 2: 0}, root_player=0, stores=[0, 0])
        result = quality([0.25, 0.5, 0.25, 0, 0, 0], exact, [0, 1, 2])
        self.assertEqual(0.75, result["optimal_mass"])
        self.assertEqual(0.5, result["expected_regret"])

    def test_canonical_deduplication_key_is_order_independent(self) -> None:
        state = {
            "opponent_pits": [0] * 6,
            "player_pits": [1] + [0] * 5,
            "current_player": 0,
            "opponent_store": 23,
            "player_store": 24,
        }
        self.assertEqual(
            canonical_key(state), canonical_key(dict(reversed(list(state.items()))))
        )

    def test_paired_bootstrap_is_deterministic(self) -> None:
        first = paired_bootstrap(
            [0.5, 0.7, 0.2], [0.4, 0.2, 0.3], seed=340, samples=100
        )
        self.assertEqual(
            first,
            paired_bootstrap([0.5, 0.7, 0.2], [0.4, 0.2, 0.3], seed=340, samples=100),
        )

    def test_selection_change_classifier_preserves_tied_optimal_moves(self) -> None:
        exact = exact_action_summary({0: 4, 1: 4, 2: 2}, root_player=0, stores=[0, 0])
        self.assertEqual("exact_tie", _selection_change_class(exact, 0, 1))

    def test_selection_change_classifier_distinguishes_wdl_and_margin(self) -> None:
        exact = exact_action_summary({0: 4, 1: 2, 2: -1}, root_player=0, stores=[0, 0])
        self.assertEqual(
            "same_outcome_margin_regression", _selection_change_class(exact, 0, 1)
        )
        self.assertEqual("outcome_regression", _selection_change_class(exact, 1, 2))


if __name__ == "__main__":
    unittest.main()
