"""Focused contracts for the PR #182 prior-amplification diagnostic."""

from __future__ import annotations

import unittest

import numpy as np

from ml.alphazero_lite.run_policy_prior_search_amplification_audit import (
    InterpolatedPolicyEvaluator,
    aggregate,
    bootstrap,
    classify,
    js,
    legal_distribution,
    ranked,
)


class PolicyPriorSearchAmplificationAuditTest(unittest.TestCase):
    def test_legal_distribution_removes_illegal_mass(self) -> None:
        policy = legal_distribution(np.array([1, 100, 1, 100, 1, 100]), [0, 2, 4])
        self.assertTrue(np.allclose(policy, [1 / 3, 0, 1 / 3, 0, 1 / 3, 0]))

    def test_rank_is_stable_on_equal_priors(self) -> None:
        self.assertEqual([1, 2, 4], ranked(np.ones(6), [4, 2, 1]))

    def test_js_is_zero_for_same_legal_distribution(self) -> None:
        policy = legal_distribution(np.ones(6), [0, 1])
        self.assertAlmostEqual(0.0, js(policy, policy, [0, 1]))

    def test_aggregate_keeps_candidate_budget_groups(self) -> None:
        result = aggregate(
            [
                {
                    "candidate": "D384",
                    "budget": 128,
                    "state_hash": "a",
                    "visit_js": 0.1,
                },
                {
                    "candidate": "D384",
                    "budget": 128,
                    "state_hash": "b",
                    "visit_js": 0.3,
                },
            ],
            ("candidate", "budget"),
        )
        self.assertEqual(2, result["D384|128"]["n"])
        self.assertAlmostEqual(0.2, result["D384|128"]["visit_js"])

    def test_bootstrap_preserves_zero_difference(self) -> None:
        result = bootstrap([0.0, 0.0], seed=1)
        self.assertEqual(2, result["n"])
        self.assertEqual(0.0, result["mean"])

    def test_interpolation_rejects_non_diagnostic_alpha(self) -> None:
        with self.assertRaises(ValueError):
            InterpolatedPolicyEvaluator(None, None, 0.1)  # type: ignore[arg-type]

    def test_bifurcation_classification_uses_amplification(self) -> None:
        labels, _action = classify(
            {"D384|128": {"amplification_statistic": 6.0}}, {}, {}
        )
        self.assertIn("search_prior_bifurcation_confirmed", labels)


if __name__ == "__main__":
    unittest.main()
