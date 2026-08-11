"""Focused contracts for the Q-versus-visit causal audit."""

from __future__ import annotations

import unittest

from ml.alphazero_lite.run_puct_visit_allocation_causal_audit import (
    _best_move,
    _bootstrap,
    classify,
)


class PuctVisitAllocationCausalAuditTest(unittest.TestCase):
    def test_q_best_uses_q_then_stable_move_tie_break(self) -> None:
        teacher = {
            "child_stats": [
                {"move": 2, "q_value": 0.4, "visits": 9},
                {"move": 1, "q_value": 0.4, "visits": 1},
            ]
        }
        self.assertEqual(1, _best_move(teacher, field="q_value"))

    def test_visit_best_uses_visit_then_q_tie_break(self) -> None:
        teacher = {
            "child_stats": [
                {"move": 2, "q_value": 0.1, "visits": 9},
                {"move": 1, "q_value": 0.4, "visits": 9},
            ]
        }
        self.assertEqual(1, _best_move(teacher, field="visits"))

    def test_bootstrap_reports_direct_paired_difference(self) -> None:
        result = _bootstrap([0.1, 0.2, 0.3], seed=1)
        self.assertEqual(3, result["n"])
        self.assertGreater(result["mean"], 0)

    def test_confirmed_requires_direct_and_paired_lower_bounds(self) -> None:
        direct = {
            f"D{budget}": {
                str(cont): {
                    "normalized_store_margin_delta": {"mean": 0.1, "lower": 0.01}
                }
                for cont in (768, 1200)
            }
            for budget in (384, 768, 1200)
        }
        paired = {
            f"D{budget}": {
                str(cont): {
                    "q_minus_visit_concordance": {"lower": 0.01},
                    "visit_minus_q_regret": {"lower": 0.01},
                }
                for cont in (768, 1200)
            }
            for budget in (384, 768, 1200)
        }
        rows = [
            {
                "teacher_budget": 768,
                "state_hash": str(index),
                "normalized_store_margin_delta": 0.1,
                "store_margin_delta": 4.8,
                "outcome_delta": 1.0,
                "prior_advantage_for_visit": 0.1,
            }
            for index in range(64)
        ]
        labels, _action = classify(direct, paired, rows)
        self.assertIn("visit_allocation_bottleneck_confirmed", labels)


if __name__ == "__main__":
    unittest.main()
