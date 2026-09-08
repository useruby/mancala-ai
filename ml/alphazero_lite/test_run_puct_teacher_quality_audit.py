"""Focused contracts for the evaluation-only PUCT teacher audit."""

from __future__ import annotations

import unittest

from ml.alphazero_lite.run_puct_teacher_quality_audit import (
    audit_seed,
    normalize_policy,
    optimal_metrics,
    transition,
)


class PuctTeacherQualityAuditTest(unittest.TestCase):
    def test_seed_is_reproducible_and_budget_independent(self) -> None:
        self.assertEqual(audit_seed("state", 96), audit_seed("state", 96))
        self.assertNotEqual(audit_seed("state", 96), audit_seed("state", 192))

    def test_policy_normalization_masks_illegal_actions(self) -> None:
        self.assertEqual(
            [0.0, 0.25, 0.0, 0.75, 0.0, 0.0],
            normalize_policy([9, 1, 9, 3, 9, 9], [1, 3]),
        )

    def test_optimal_set_metrics_accept_exact_ties(self) -> None:
        metric = optimal_metrics([0.0, 0.6, 0.4, 0.0, 0.0, 0.0], [1, 2], [1, 2])
        self.assertTrue(metric["correct"])
        self.assertEqual(1.0, metric["optimal_mass"])

    def test_correction_transitions(self) -> None:
        self.assertEqual("wrong_to_correct", transition(False, True))
        self.assertEqual("correct_to_wrong", transition(True, False))


if __name__ == "__main__":
    unittest.main()
