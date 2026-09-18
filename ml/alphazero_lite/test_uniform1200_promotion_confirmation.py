"""Contracts for the single-candidate uniform1200 promotion confirmation."""

import json
import unittest
from pathlib import Path

from ml.alphazero_lite.run_uniform1200_promotion_confirmation import (
    EXACT,
    EXPECTED_CANDIDATE_SHA256,
    EXPECTED_CURRENT_SHA256,
    HIGH_POWER,
    SEEDS,
    classify,
    gate_command,
    gate_defaults,
    selection_candidates,
    verify_incumbent,
)


class Uniform1200PromotionConfirmationTest(unittest.TestCase):
    def setUp(self):
        self.high = json.loads(HIGH_POWER.read_text())
        self.exact = json.loads(EXACT.read_text())

    def test_selection_uses_exactly_the_six_registered_seeds(self):
        ranked = selection_candidates(self.high, self.exact)
        self.assertEqual(SEEDS, tuple(sorted(row["seed"] for row in ranked)))

    def test_selection_is_deterministic_and_selects_seed48(self):
        ranked = selection_candidates(self.high, self.exact)
        self.assertEqual(ranked, selection_candidates(self.high, self.exact))
        self.assertEqual(48, ranked[0]["seed"])
        self.assertEqual(
            EXPECTED_CANDIDATE_SHA256, ranked[0]["checkpoint_weights_json_sha256"]
        )

    def test_failure_never_falls_through_to_another_seed(self):
        self.assertEqual(
            "uniform1200_promotion_arena_failed",
            classify(
                {"failure_reasons": [{"code": "arena_score_below_threshold"}]},
                {"passed": True},
            ),
        )

    def test_incumbent_and_production_defaults_are_frozen(self):
        self.assertEqual(
            EXPECTED_CURRENT_SHA256, verify_incumbent()["weights_json_sha256"]
        )
        defaults = gate_defaults()
        self.assertEqual(120, defaults["arena_games"])
        self.assertEqual(40, defaults["mcts_games"])
        self.assertFalse(defaults["skip_mcts_relative_check"])

    def test_gate_command_has_one_candidate_and_no_relaxations_or_stubs(self):
        command = gate_command(Path("candidate"), Path("report.json"))
        self.assertEqual(1, command.count("--candidate-path"))
        self.assertNotIn("--skip-mcts-relative-check", command)
        self.assertFalse(any(flag.startswith("--stub-") for flag in command))
        self.assertNotIn("promote_superhuman_candidate", " ".join(command))

    def test_hard_classification_precedence(self):
        self.assertEqual(
            "uniform1200_mcts1200_relative_failed",
            classify(
                {"failure_reasons": [{"code": "candidate_mcts_below_current"}]},
                {"passed": True},
            ),
        )
        self.assertEqual(
            "uniform1200_production_forensic_only_blocker",
            classify(
                {"failure_reasons": [{"code": "forensic_overall_regressed"}]},
                {"passed": True},
            ),
        )
        self.assertEqual(
            "uniform1200_outcome_shadow_regression",
            classify({"passed": True, "failure_reasons": []}, {"passed": False}),
        )


if __name__ == "__main__":
    unittest.main()
