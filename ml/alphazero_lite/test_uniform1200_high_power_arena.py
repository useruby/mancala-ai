"""Focused contracts for the frozen uniform1200 high-power arena."""

from __future__ import annotations

import unittest

from ml.alphazero_lite.run_uniform1200_high_power_arena import (
    EXPECTED_SEARCH_OPTIONS,
    EXPECTED_SHA256,
    GAMES_PER_OPENING,
    OPENING_COUNT,
    OPENING_PLIES,
    SEEDS,
    SIMULATIONS,
    aggregate_opening_pairs,
    bootstrap_ci,
    build_opening_suite,
    hierarchical_ci,
    suite_preflight,
)


class Uniform1200HighPowerArenaTest(unittest.TestCase):
    def test_frozen_checkpoint_contract_has_six_pairs(self) -> None:
        self.assertEqual((47, 48, 49, 50, 51, 52), SEEDS)
        self.assertEqual(set(SEEDS), set(EXPECTED_SHA256))
        self.assertTrue(
            all(
                set(EXPECTED_SHA256[seed]) == {"control", "uniform1200"}
                for seed in SEEDS
            )
        )

    def test_opening_suite_is_four_ply_and_diverse(self) -> None:
        suite = build_opening_suite()
        self.assertEqual(OPENING_COUNT, len(suite))
        self.assertTrue(all(len(row["prefix_moves"]) == OPENING_PLIES for row in suite))
        self.assertEqual(
            233, len({row["canonical_resulting_state_hash"] for row in suite})
        )
        with self.assertRaisesRegex(
            ValueError, "high_power_opening_suite_low_diversity"
        ):
            suite_preflight(suite)

    def test_frozen_suite_serialization_is_deterministic(self) -> None:
        self.assertEqual(build_opening_suite(), build_opening_suite())

    def test_pairing_requires_two_opposite_seats(self) -> None:
        entries = []
        for index in range(OPENING_COUNT):
            for within, seat in enumerate((0, 1)):
                entries.append(
                    {
                        "opening_index": index,
                        "game_within_opening": within,
                        "challenger_player": seat,
                        "winner": "challenger" if seat == 0 else "current",
                    }
                )
        pairs = aggregate_opening_pairs(entries)
        self.assertEqual(OPENING_COUNT, len(pairs))
        self.assertEqual(0.5, pairs[0]["pair_score"])

    def test_bootstraps_are_deterministic(self) -> None:
        scores = [0.0, 0.5, 1.0] * (OPENING_COUNT // 3) + [0.5]
        self.assertEqual(
            bootstrap_ci(scores, seed=90417), bootstrap_ci(scores, seed=90417)
        )
        self.assertEqual(hierarchical_ci([scores] * 6), hierarchical_ci([scores] * 6))

    def test_search_is_locked_to_pr325_identity(self) -> None:
        self.assertEqual(384, SIMULATIONS)
        self.assertEqual(2, GAMES_PER_OPENING)
        self.assertEqual(
            {
                "fpu_mode": "zero",
                "reuse_subtree": False,
                "normalize_values": False,
                "root_policy_mode": "deterministic",
                "tactical_root_bias": 0.0,
                "root_temperature": 0.0,
            },
            EXPECTED_SEARCH_OPTIONS,
        )


if __name__ == "__main__":
    unittest.main()
