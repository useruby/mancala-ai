"""Focused contracts for the production arena prefilter calibration."""

from __future__ import annotations

import unittest
from pathlib import Path

from ml.alphazero_lite.run_production_arena_prefilter_calibration import (
    CHALLENGER_SIMULATIONS,
    CURRENT_SIMULATIONS,
    GAMES_PER_OPENING,
    OPENING_COUNT,
    PRODUCTION_THRESHOLD,
    START_GAMES,
    arena_command,
    bootstrap,
    build_suite,
    classify,
    frozen_pairs,
    opening_pairs,
    production_contract,
    verify_manifest,
)


class ProductionArenaPrefilterCalibrationTest(unittest.TestCase):
    def test_frozen_manifest_has_seven_positive_pairs_and_two_controls(self) -> None:
        manifest = verify_manifest(frozen_pairs())
        pairs = manifest["pairs"]
        self.assertTrue(manifest["frozen_before_calibration_results"])
        self.assertEqual(
            7, sum(pair["frozen_label"] == "known_positive" for pair in pairs)
        )
        self.assertEqual(2, sum(pair["identity_control"] for pair in pairs))
        self.assertEqual(
            {"P47", "P48", "P49", "P50", "P51", "P52", "P_INC", "I_INC", "I_48"},
            {pair["pair_id"] for pair in pairs},
        )

    def test_contract_suite_and_commands_are_frozen(self) -> None:
        contract = production_contract()
        self.assertEqual(START_GAMES, contract["arena_games"])
        self.assertEqual(PRODUCTION_THRESHOLD, contract["minimum_score"])
        self.assertEqual(0, contract["random_opening_plies"])
        suite, preflight = build_suite()
        self.assertEqual(942, preflight["total_canonical_population"])
        self.assertEqual(430, preflight["remaining_population"])
        self.assertEqual(OPENING_COUNT, len(suite))
        self.assertEqual(0, preflight["pr327_overlap"])
        self.assertEqual(0, preflight["pr329_overlap"])
        self.assertEqual(suite, build_suite()[0])
        pair = frozen_pairs()[0]
        start = arena_command(pair, Path("output/start"))
        canonical = arena_command(
            pair,
            Path("output/canonical"),
            suite=Path(
                "docs/data/alphazero-lite-uniform1200-high-power-openings-v2.jsonl"
            ),
        )
        self.assertEqual(str(START_GAMES), start[start.index("--games") + 1])
        self.assertNotIn("--opening-prefixes-jsonl", start)
        self.assertEqual(
            str(OPENING_COUNT * GAMES_PER_OPENING),
            canonical[canonical.index("--games") + 1],
        )
        self.assertEqual(
            str(GAMES_PER_OPENING),
            canonical[canonical.index("--games-per-opening") + 1],
        )
        for command in (start, canonical):
            self.assertEqual(
                str(CHALLENGER_SIMULATIONS),
                command[command.index("--challenger-simulations") + 1],
            )
            self.assertEqual(
                str(CURRENT_SIMULATIONS),
                command[command.index("--current-simulations") + 1],
            )
            self.assertNotIn("train", " ".join(command))
            self.assertNotIn("promote", " ".join(command))

    def test_pair_scoring_bootstrap_and_classification_precedence(self) -> None:
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
        self.assertEqual(
            bootstrap([0.0, 0.5, 1.0] * 85 + [0.5]),
            bootstrap([0.0, 0.5, 1.0] * 85 + [0.5]),
        )
        base = {
            "known_positive": {
                "positive_evidence_count": 7,
                "canonical_pass_count": 7,
                "start_pass_count": 0,
                "matching_decision_count": 0,
                "meaningful_start_diversity": False,
            },
            "primary_rule": {"passes": True},
            "identity_controls": [{"identity_material_budget_advantage": True}],
            "pair_inc_strong_disagreement": True,
            "uniform_matching_decisions": 6,
        }
        self.assertEqual(
            "production_prefilter_calibration_invalid",
            classify({**base, "invalid": True}),
        )
        self.assertEqual(
            "production_prefilter_start_state_systematic_false_negative", classify(base)
        )
        self.assertEqual(
            "production_prefilter_budget_advantage_material",
            classify({**base, "primary_rule": {"passes": False}}),
        )


if __name__ == "__main__":
    unittest.main()
