"""Contracts for the PR #330 384/256 versus 384/384 budget experiment."""

from __future__ import annotations

import unittest
from pathlib import Path

from ml.alphazero_lite.run_production_arena_budget_calibration import (
    ASYMMETRIC_SIMULATIONS,
    BOOTSTRAP_REPLICATES,
    CHALLENGER_SIMULATIONS,
    EQUAL_CURRENT_SIMULATIONS,
    GAMES_PER_OPENING,
    OPENING_COUNT,
    SUITE_SHA256,
    arena_command,
    bootstrap,
    budget_effect,
    classify,
    command_delta,
    identity_rule,
    load_asymmetric_results,
    load_frozen_suite,
)
from ml.alphazero_lite.run_production_arena_prefilter_calibration import (
    frozen_pairs,
    verify_manifest,
)


class ProductionArenaBudgetCalibrationTest(unittest.TestCase):
    def test_frozen_pr330_inputs_are_exact_and_immutable(self) -> None:
        manifest = verify_manifest(frozen_pairs())
        self.assertEqual(
            ["P47", "P48", "P49", "P50", "P51", "P52", "P_INC", "I_INC", "I_48"],
            [pair["pair_id"] for pair in manifest["pairs"]],
        )
        self.assertEqual(OPENING_COUNT, len(load_frozen_suite()))
        self.assertEqual(
            SUITE_SHA256,
            "811feb7d208467cdfa9a42244eb75adcc8877db5bc03f87dacbd3379e08bd4bf",
        )
        inherited = load_asymmetric_results(manifest)
        self.assertEqual(
            0.697265625, inherited["P47"]["metrics"]["canonical_pair_score"]
        )
        self.assertEqual(
            0.8896484375, inherited["P_INC"]["metrics"]["canonical_pair_score"]
        )

    def test_only_current_budget_changes(self) -> None:
        pair = frozen_pairs()[0]
        command = arena_command(pair, Path("output"))
        self.assertEqual(
            str(OPENING_COUNT * GAMES_PER_OPENING),
            command[command.index("--games") + 1],
        )
        self.assertEqual(
            str(CHALLENGER_SIMULATIONS),
            command[command.index("--challenger-simulations") + 1],
        )
        self.assertEqual(
            str(EQUAL_CURRENT_SIMULATIONS),
            command[command.index("--current-simulations") + 1],
        )
        self.assertEqual(
            str(GAMES_PER_OPENING), command[command.index("--games-per-opening") + 1]
        )
        self.assertNotIn("train", " ".join(command))
        self.assertNotIn("promote", " ".join(command))
        proof = command_delta(pair)
        self.assertEqual(
            ASYMMETRIC_SIMULATIONS, proof["asymmetric_current_simulations"]
        )
        self.assertEqual(EQUAL_CURRENT_SIMULATIONS, proof["equal_current_simulations"])

    def test_paired_effect_bootstrap_and_identity_rule(self) -> None:
        asymmetric = [
            {"opening_index": index, "pair_score": 0.75}
            for index in range(OPENING_COUNT)
        ]
        equal = [
            {"opening_index": index, "pair_score": 0.5}
            for index in range(OPENING_COUNT)
        ]
        effect, rows = budget_effect(asymmetric, equal)
        self.assertEqual(OPENING_COUNT, len(rows))
        self.assertEqual(0.25, effect["mean"])
        self.assertEqual(
            BOOTSTRAP_REPLICATES, effect["budget_effect_ci95"]["replicates"]
        )
        self.assertEqual(331, effect["budget_effect_ci95"]["rng_seed"])
        self.assertEqual(
            bootstrap([0.5] * OPENING_COUNT, 330), bootstrap([0.5] * OPENING_COUNT, 330)
        )
        self.assertTrue(
            identity_rule(
                0.6,
                {
                    "canonical_pair_score": 0.5,
                    "canonical_pair_ci95": {"lower": 0.49, "upper": 0.51},
                },
                effect,
            )
        )

    def test_hard_classification_precedence(self) -> None:
        base = {
            "identity_failed": False,
            "budget_causal_rule": True,
            "strength_retention_rule": True,
            "positive_evidence_count": 7,
            "asymmetry_dependent_055_pass_count": 2,
            "identity_budget_effect_material": True,
        }
        self.assertEqual(
            "budget_calibration_artifact_mismatch",
            classify({**base, "artifact_mismatch": True}),
        )
        self.assertEqual(
            "budget_calibration_suite_mismatch",
            classify({**base, "suite_mismatch": True}),
        )
        self.assertEqual(
            "budget_calibration_inconclusive",
            classify({**base, "execution_failed": True}),
        )
        self.assertEqual(
            "equal_budget_identity_control_failed",
            classify({**base, "identity_failed": True}),
        )
        self.assertEqual(
            "production_prefilter_budget_asymmetry_confound_confirmed_strength_retained",
            classify(base),
        )


if __name__ == "__main__":
    unittest.main()
