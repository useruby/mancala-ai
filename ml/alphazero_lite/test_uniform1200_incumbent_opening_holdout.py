"""Focused contracts for the seed48 incumbent opening holdout."""

import unittest
from pathlib import Path

from ml.alphazero_lite.run_uniform1200_incumbent_opening_holdout import (
    CANDIDATE_SHA256,
    CHALLENGER_SIMULATIONS,
    CURRENT_SIMULATIONS,
    GAMES_PER_OPENING,
    HOLDOUT_SUITE_SHA256,
    INCUMBENT_SHA256,
    OPENING_COUNT,
    PR327_SUITE_SHA256,
    SEARCH_OPTIONS,
    SELECTION_SEED,
    SUITE_VERSION,
    arena_command,
    bootstrap,
    build_suite,
    candidate_preflight,
    classify,
    incumbent_preflight,
    load_pr327_hashes,
    opening_pairs,
    sha256,
)


class Uniform1200IncumbentOpeningHoldoutTest(unittest.TestCase):
    def test_frozen_artifacts_and_excluded_suite_match(self) -> None:
        self.assertEqual(CANDIDATE_SHA256, candidate_preflight()["weights_sha256"])
        self.assertEqual(INCUMBENT_SHA256, incumbent_preflight()["weights_sha256"])
        self.assertEqual(
            PR327_SUITE_SHA256,
            sha256(
                Path(
                    "docs/data/alphazero-lite-uniform1200-high-power-openings-v2.jsonl"
                )
            ),
        )

    def test_population_and_selection_are_canonical_and_deterministic(self) -> None:
        suite, preflight = build_suite()
        self.assertEqual(942, preflight["total_legal_four_ply_prefixes"])
        self.assertEqual(942, preflight["total_canonical_population"])
        self.assertEqual(686, preflight["remaining_population_count"])
        self.assertEqual(OPENING_COUNT, len(suite))
        self.assertEqual(
            OPENING_COUNT, len({row["canonical_resulting_state_hash"] for row in suite})
        )
        self.assertFalse(
            {row["canonical_resulting_state_hash"] for row in suite}
            & load_pr327_hashes()
        )
        self.assertEqual(suite, build_suite()[0])
        self.assertEqual(
            HOLDOUT_SUITE_SHA256,
            sha256(
                Path(
                    "docs/data/alphazero-lite-uniform1200-incumbent-holdout-openings-v1.jsonl"
                )
            ),
        )
        self.assertEqual(42, SELECTION_SEED)
        self.assertEqual("uniform1200_incumbent_holdout_unique_v1", SUITE_VERSION)

    def test_arena_is_single_candidate_production_profile(self) -> None:
        command = arena_command(
            Path(
                "docs/data/alphazero-lite-uniform1200-incumbent-holdout-openings-v1.jsonl"
            ),
            Path("out"),
        )
        self.assertEqual(1, command.count("--challenger"))
        self.assertEqual("384", command[command.index("--challenger-simulations") + 1])
        self.assertEqual("256", command[command.index("--current-simulations") + 1])
        self.assertEqual("2", command[command.index("--games-per-opening") + 1])
        self.assertEqual(384, CHALLENGER_SIMULATIONS)
        self.assertEqual(256, CURRENT_SIMULATIONS)
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
            SEARCH_OPTIONS,
        )
        self.assertNotIn("train", " ".join(command))
        self.assertNotIn("promote", " ".join(command))

    def test_pair_aggregation_bootstrap_and_classification_contracts(self) -> None:
        entries = [
            {
                "opening_index": index,
                "game_within_opening": seat,
                "challenger_player": seat,
                "winner": "challenger" if seat == 0 else "current",
            }
            for index in range(OPENING_COUNT)
            for seat in (0, 1)
        ]
        self.assertEqual(0.5, opening_pairs(entries)[0]["pair_score"])
        scores = [0.0, 0.5, 1.0] * 85 + [0.5]
        self.assertEqual(bootstrap(scores), bootstrap(scores))
        base = {
            "mean_opening_pair_score": 0.56,
            "opening_pair_ci95": {"lower": 0.51, "upper": 0.61},
            "opening_strength_distribution": {
                "candidate_favored": 130,
                "neutral": 80,
                "incumbent_favored": 46,
            },
            "score_by_challenger_seat": {"player_0": 0.5, "player_1": 0.5},
        }
        self.assertEqual(
            "production_start_state_arena_disagreement_confirmed", classify(base, True)
        )
        self.assertEqual(
            "incumbent_holdout_positive_below_production_threshold",
            classify({**base, "mean_opening_pair_score": 0.53}, True),
        )
        self.assertEqual(
            "incumbent_holdout_strength_uncertain",
            classify(
                {
                    **base,
                    "mean_opening_pair_score": 0.53,
                    "opening_pair_ci95": {"lower": 0.49, "upper": 0.57},
                },
                True,
            ),
        )
        self.assertEqual(
            "incumbent_holdout_candidate_weaker",
            classify(
                {
                    **base,
                    "mean_opening_pair_score": 0.42,
                    "opening_pair_ci95": {"lower": 0.37, "upper": 0.48},
                },
                True,
            ),
        )
        self.assertEqual(
            "incumbent_holdout_opening_specific",
            classify(
                {
                    **base,
                    "opening_strength_distribution": {
                        "candidate_favored": 20,
                        "neutral": 220,
                        "incumbent_favored": 16,
                    },
                },
                True,
            ),
        )
        self.assertEqual(
            "incumbent_holdout_no_strength_gain",
            classify(
                {
                    **base,
                    "mean_opening_pair_score": 0.5,
                    "opening_strength_distribution": {
                        "candidate_favored": 50,
                        "neutral": 156,
                        "incumbent_favored": 50,
                    },
                },
                True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
