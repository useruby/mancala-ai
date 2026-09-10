"""Focused contracts for the evaluation-only exact neighborhood audit."""

from __future__ import annotations

import unittest

from ml.alphazero_lite.run_uniform1200_exact_neighborhood_audit import (
    family_classification,
    generate_neighborhood,
    load_anchor_ids,
    sample_radius2,
)


class ExactNeighborhoodAuditTest(unittest.TestCase):
    def test_pr289_anchor_loading_is_locked_to_expected_regression_count(self) -> None:
        regressions, improvements = load_anchor_ids()
        self.assertEqual(38, len(regressions))
        self.assertTrue(improvements)

    def test_radius_generation_uses_legal_game_transitions_and_marks_ineligible(
        self,
    ) -> None:
        state = {
            "player_pits": [0, 0, 0, 0, 0, 1],
            "opponent_pits": [0, 0, 0, 0, 0, 1],
            "player_store": 23,
            "opponent_store": 23,
            "current_player": 0,
        }
        rows = generate_neighborhood("test-001", state)
        self.assertEqual(1, sum(row["radius"] == 1 for row in rows))
        self.assertTrue(all(not row["training_eligible"] for row in rows))
        radius1 = next(row for row in rows if row["radius"] == 1)
        self.assertTrue(radius1["provenance"][0]["extra_turn"])

    def test_radius2_sampling_is_sha_deterministic(self) -> None:
        rows = [{"canonical_state_key": str(index), "radius": 2} for index in range(20)]
        first, counts = sample_radius2(rows, cap=7)
        second, _ = sample_radius2(list(reversed(rows)), cap=7)
        self.assertEqual(first, second)
        self.assertEqual(20, counts["radius2_full"])
        self.assertEqual(7, counts["radius2_sampled"])

    def test_family_thresholds_are_fixed(self) -> None:
        self.assertEqual(
            "stable_regression_family",
            family_classification(
                original_regression_seeds=2,
                local_rates_by_seed=[0.6, 0.7, 0.5],
                solved_states=10,
                same_direction_seeds=2,
            ),
        )
        self.assertEqual(
            "anchor_specific_regression",
            family_classification(
                original_regression_seeds=2,
                local_rates_by_seed=[0.1, 0.2, 0.3],
                solved_states=10,
                same_direction_seeds=2,
            ),
        )
        self.assertEqual(
            "mixed",
            family_classification(
                original_regression_seeds=2,
                local_rates_by_seed=[0.4, 0.5, 0.3],
                solved_states=10,
                same_direction_seeds=2,
            ),
        )


if __name__ == "__main__":
    unittest.main()
