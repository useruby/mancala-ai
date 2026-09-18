"""Focused contracts for the frozen uniform1200 high-power arena."""

from __future__ import annotations

import unittest
from pathlib import Path

from ml.alphazero_lite.run_uniform1200_high_power_arena import (
    EXPECTED_SEARCH_OPTIONS,
    EXPECTED_SHA256,
    GAMES_PER_OPENING,
    OPENING_COUNT,
    OPENING_PLIES,
    SEEDS,
    SIMULATIONS,
    aggregate_opening_pairs,
    arena_command,
    bootstrap_ci,
    build_opening_suite,
    canonical_unique_opening_population,
    checkpoint_preflight,
    enumerate_four_ply_openings,
    hierarchical_ci,
    initial_game,
    select_unique_opening_suite,
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

    def test_opening_suite_is_four_ply_and_canonically_unique(self) -> None:
        suite, _openings, _excluded = build_opening_suite()
        self.assertEqual(OPENING_COUNT, len(suite))
        self.assertTrue(all(len(row["prefix_moves"]) == OPENING_PLIES for row in suite))
        self.assertEqual(
            OPENING_COUNT, len({row["canonical_resulting_state_hash"] for row in suite})
        )
        self.assertEqual(1.0, suite_preflight(suite)["canonical_unique_fraction"])

    def test_four_ply_enumeration_uses_kalah_turn_transitions(self) -> None:
        openings, excluded = enumerate_four_ply_openings()
        self.assertEqual(942, len(openings))
        self.assertEqual(0, excluded)
        self.assertEqual({0, 1}, {row["current_player"] for row in openings})
        for row in openings:
            game = initial_game()
            self.assertEqual(
                OPENING_PLIES,
                sum(
                    int(game.move(game.pit_index(move))) for move in row["prefix_moves"]
                ),
            )

    def test_canonical_grouping_selects_lexicographic_representative(self) -> None:
        state_hash = "same-state"
        population = canonical_unique_opening_population(
            [
                {
                    "prefix_moves": [2, 0, 0, 0],
                    "canonical_resulting_state_hash": state_hash,
                },
                {
                    "prefix_moves": [1, 5, 5, 5],
                    "canonical_resulting_state_hash": state_hash,
                },
            ]
        )
        self.assertEqual([1, 5, 5, 5], population[0]["prefix_moves"])
        self.assertEqual(2, population[0]["prefix_multiplicity"])

    def test_unique_selection_is_deterministic_and_model_independent(self) -> None:
        suite, _openings, _excluded = build_opening_suite()
        self.assertEqual(build_opening_suite(), build_opening_suite())
        self.assertEqual(
            suite,
            select_unique_opening_suite(
                canonical_unique_opening_population(enumerate_four_ply_openings()[0])
            ),
        )
        self.assertTrue(all("selection_key" in row for row in suite))

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

    def test_checkpoint_shas_are_frozen_and_v1_artifact_is_immutable(self) -> None:
        self.assertEqual(
            EXPECTED_SHA256,
            {int(seed): values for seed, values in checkpoint_preflight().items()},
        )
        v1 = Path("docs/data/alphazero-lite-uniform1200-high-power-openings.jsonl")
        self.assertEqual(
            "bf087acb340904387f4cb2edb8197b950d91f3acc827ffc682e15a285837eb4d",
            __import__("hashlib").sha256(v1.read_bytes()).hexdigest(),
        )

    def test_arena_remains_seat_reversed_and_evaluation_only(self) -> None:
        command = arena_command(
            47,
            Path("docs/data/alphazero-lite-uniform1200-high-power-openings-v2.jsonl"),
            Path("output"),
        )
        self.assertIn("--games-per-opening", command)
        self.assertEqual("2", command[command.index("--games-per-opening") + 1])
        self.assertEqual("384", command[command.index("--challenger-simulations") + 1])
        self.assertEqual("384", command[command.index("--current-simulations") + 1])


if __name__ == "__main__":
    unittest.main()
