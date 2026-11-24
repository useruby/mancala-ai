"""Regression checks for exact-oracle state/action handling."""

from __future__ import annotations

import copy
import unittest

from ml.alphazero_lite.seed633_exact_oracle_decision_diagnostic import derive


class ExactOracleDecisionDiagnosticTests(unittest.TestCase):
    """Protect tied-optimum handling, regret, and cohort integrity."""

    def setUp(self) -> None:
        self.cohort = [
            {
                "canonical_identity": "state-a",
                "teacher_type": "exact_root_tablebase",
                "teacher_evidence": {"exact_action_margins": {"0": 5, "1": 5, "2": -1}},
                "legal_moves": [0, 1, 2],
                "exact_optimal_actions": [0, 1],
                "source_opening_group": "source:opening",
                "membership_class": "neither_training_membership",
                "phase": "at_most_16",
            }
        ]
        self.metrics = [
            {
                "canonical_identity": "state-a",
                "teacher_entropy_nats": 0.5,
                "B512": {
                    "legal_policy": [0.4, 0.3, 0.3, 0.0, 0.0, 0.0],
                    "teacher_target_cross_entropy_nats": 0.5,
                },
                "S615": {
                    "legal_policy": [0.3, 0.4, 0.3, 0.0, 0.0, 0.0],
                    "teacher_target_cross_entropy_nats": 0.4,
                },
            }
        ]

    def test_all_tied_optimal_actions_count_as_correct(self) -> None:
        result = derive(self.cohort, self.metrics)[0]
        self.assertEqual(result["B512"]["exact_optimal_top_action"], True)
        self.assertEqual(result["S615"]["exact_optimal_top_action"], True)
        self.assertEqual(result["S615"]["top_policy_action_regret"], 0)

    def test_altered_oracle_label_rejected(self) -> None:
        cohort = copy.deepcopy(self.cohort)
        cohort[0]["exact_optimal_actions"] = [0]
        with self.assertRaisesRegex(
            ValueError, "oracle_action_value_correspondence_invalid"
        ):
            derive(cohort, self.metrics)

    def test_altered_legal_action_mask_rejected(self) -> None:
        cohort = copy.deepcopy(self.cohort)
        cohort[0]["legal_moves"] = [0, 1]
        with self.assertRaisesRegex(
            ValueError, "oracle_action_value_correspondence_invalid"
        ):
            derive(cohort, self.metrics)

    def test_model_policy_identity_order_mismatch_rejected(self) -> None:
        metrics = copy.deepcopy(self.metrics)
        metrics[0]["canonical_identity"] = "other-state"
        with self.assertRaisesRegex(ValueError, "cohort_metric_identity_mismatch"):
            derive(self.cohort, metrics)

    def test_cohort_membership_is_preserved(self) -> None:
        cohort = copy.deepcopy(self.cohort)
        cohort[0]["membership_class"] = "S615_only"
        self.assertEqual(
            derive(cohort, self.metrics)[0]["membership_class"], "S615_only"
        )


if __name__ == "__main__":
    unittest.main()
