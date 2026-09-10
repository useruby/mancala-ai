"""Contracts for the missingness-robust exact population audit."""

from __future__ import annotations

import unittest

from ml.alphazero_lite.run_uniform1200_exact_population_audit import (
    anchor_weighted_comparison,
    bounds,
    classify_anchor,
    combine_oracles,
    whole_classification,
)


class ExactPopulationAuditTest(unittest.TestCase):
    @staticmethod
    def row(key: str, status: str = "exact_timeout") -> dict:
        return {
            "canonical_state_key": key,
            "oracle_status": status,
            "exact_value": 1,
            "exact_action_values": {0: 1},
            "exact_optimal_actions": [0],
            "exact_root_value": 1.0,
        }

    def test_exact_label_precedence_never_overwrites_earlier_tier(self) -> None:
        base = self.row("a", "exact_solved")
        base["exact_value"] = 18
        tier19, tier20, tier21 = (self.row("a", "exact_solved") for _ in range(3))
        tier19["exact_value"], tier20["exact_value"], tier21["exact_value"] = 19, 20, 21
        merged = combine_oracles([base], [tier19], [tier20], [tier21])
        self.assertEqual(18, merged[0]["exact_value"])
        self.assertEqual("pr291_292", merged[0]["oracle_label_source"])

    def test_later_tiers_are_used_in_order(self) -> None:
        base = self.row("a")
        tier19, tier20, tier21 = (self.row("a", "exact_solved") for _ in range(3))
        for tier in (tier19, tier20, tier21):
            tier["status"] = "exact_solved"
        tier19["exact_value"] = 19
        self.assertEqual(
            19, combine_oracles([base], [tier19], [tier20], [tier21])[0]["exact_value"]
        )

    def test_bounds_retain_the_registered_denominator(self) -> None:
        result = bounds(6, 2, 10)
        self.assertEqual(10, result["denominator"])
        self.assertEqual(0.75, result["solved_only"])
        self.assertEqual(0.6, result["pessimistic"])
        self.assertEqual(0.8, result["optimistic"])
        self.assertEqual(2, result["unresolved_capable_of_changing"])

    def test_reference_bounds_cover_all_registered_anchors(self) -> None:
        result = bounds(4, 2, 38)
        self.assertEqual(4 / 38, result["pessimistic"])
        self.assertEqual(6 / 38, result["optimistic"])

    def test_stable_family_requires_lower_bound(self) -> None:
        self.assertEqual(
            "missingness_or_model_mixed",
            classify_anchor(
                original_regression_seeds=2,
                solved=10,
                lower=[0.10, 0.59, 0.90],
                upper=[0.90, 0.90, 0.90],
                direction=2,
            ),
        )
        self.assertEqual(
            "robust_stable_regression_family",
            classify_anchor(
                original_regression_seeds=2,
                solved=10,
                lower=[0.60, 0.70, 0.80],
                upper=[0.80, 0.80, 0.80],
                direction=2,
            ),
        )

    def test_anchor_specific_family_requires_upper_bound(self) -> None:
        self.assertEqual(
            "missingness_or_model_mixed",
            classify_anchor(
                original_regression_seeds=2,
                solved=10,
                lower=[0.0] * 3,
                upper=[0.1, 0.35, 0.35],
                direction=2,
            ),
        )
        self.assertEqual(
            "robust_anchor_specific_regression",
            classify_anchor(
                original_regression_seeds=2,
                solved=10,
                lower=[0.0] * 3,
                upper=[0.1, 0.34, 0.1],
                direction=2,
            ),
        )

    def test_whole_cohort_cannot_call_stable_from_ambiguous_anchors(self) -> None:
        anchors = {
            str(index): {"classification": "missingness_or_model_mixed"}
            for index in range(38)
        }
        classification, decision = whole_classification(
            {"classification": "forensic_exact_reference_valid_robust"}, anchors
        )
        self.assertEqual("forensic_partial_exact_audit_inconclusive", classification)
        self.assertIn("do NOT automatically build tier 22", decision["next_experiment"])

    def test_anchor_primary_aggregation_does_not_let_large_neighborhood_dominate(
        self,
    ) -> None:
        def metric(rate: float, denominator: int) -> dict:
            return {
                "solved_only": rate,
                "pessimistic": rate,
                "optimistic": rate,
                "denominator": denominator,
                "unresolved_capable_of_changing": 0,
            }

        anchors = {
            "small": {
                "by_seed": {
                    str(seed): {"exact_regret_regression": metric(1.0, 1)}
                    for seed in (44, 45, 46)
                }
            },
            "large": {
                "by_seed": {
                    str(seed): {"exact_regret_regression": metric(0.0, 99)}
                    for seed in (44, 45, 46)
                }
            },
        }
        result = anchor_weighted_comparison(anchors)["44"]
        self.assertEqual(0.5, result["equal_anchor_primary"]["pessimistic"])
        self.assertEqual(0.01, result["state_weighted_secondary"]["pessimistic"])


if __name__ == "__main__":
    unittest.main()
