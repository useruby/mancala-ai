"""Focused invariants for the denoised PUCT convergence audit."""

from __future__ import annotations

import unittest

from ml.alphazero_lite.run_denoised_puct_convergence_audit import (
    pair_metrics,
    q_calibration,
    stability_summary,
    teacher_seed_identity,
)


def teacher(policy: list[float], move: int) -> dict:
    return {
        "policy": policy,
        "legal_moves": [0, 1, 2],
        "top_move": move,
        "entropy": 0.5,
        "top1_top2_visit_margin": 0.2,
        "root_value": 0.1,
        "child_q_values": {"0": 0.1, "1": 0.3, "2": -0.2},
    }


def record(moves: list[int]) -> dict:
    return {
        "state_hash": "state-" + "-".join(map(str, moves)),
        "player": 0,
        "phase": "opening",
        "source_domain": "diagnostic",
        "legal_move_count": 3,
        "current_policy_entropy": 0.7,
        "current_network_top_move": moves[-1],
        "teachers": {
            f"D{budget}": teacher([0.6, 0.3, 0.1], move)
            for budget, move in zip((128, 384, 768, 1200), moves)
        },
    }


class DenoisedPuctConvergenceAuditTest(unittest.TestCase):
    def test_teacher_seed_is_budget_invariant(self) -> None:
        seed, context_hash = teacher_seed_identity(
            state_hash="state-a", experiment_seed=42
        )
        repeated_seed, repeated_context_hash = teacher_seed_identity(
            state_hash="state-a", experiment_seed=42
        )
        self.assertEqual((seed, context_hash), (repeated_seed, repeated_context_hash))

    def test_pair_metrics_reports_both_kl_directions_and_decision_change(self) -> None:
        lower = teacher([0.7, 0.2, 0.1], 0)
        higher = teacher([0.1, 0.8, 0.1], 1)
        metrics = pair_metrics(lower, higher)
        self.assertTrue(metrics["top_move_change"])
        self.assertFalse(metrics["top1_agreement"])
        self.assertGreater(metrics["kl_lower_to_higher"], 0)
        self.assertGreater(metrics["kl_higher_to_lower"], 0)

    def test_stability_summary_detects_alternating_oscillation(self) -> None:
        summary = stability_summary([record([0, 1, 0, 1]), record([2, 2, 2, 2])])
        self.assertEqual(1, summary["alternating_128_384_768_1200_count"])
        self.assertEqual(1, summary["categories"]["stable_from_128"]["count"])
        self.assertEqual(1, summary["categories"]["unstable_through_1200"]["count"])

    def test_q_calibration_uses_realized_forced_deltas_not_search_q(self) -> None:
        rows = [
            {
                "lower_move": 0,
                "higher_move": 1,
                "higher_outcome": 1.0,
                "lower_outcome": 0.0,
                "teacher_child_q_values": {
                    f"D{budget}": {"0": 0.0, "1": 0.5}
                    for budget in (128, 384, 768, 1200)
                },
            },
            {
                "lower_move": 0,
                "higher_move": 1,
                "higher_outcome": -1.0,
                "lower_outcome": 0.0,
                "teacher_child_q_values": {
                    f"D{budget}": {"0": 0.0, "1": 0.5}
                    for budget in (128, 384, 768, 1200)
                },
            },
        ]
        calibration = q_calibration(rows)
        self.assertEqual(2, calibration["D128"]["n"])
        self.assertEqual(
            0.5, calibration["D128"]["high_confidence_causally_wrong_fraction"]
        )


if __name__ == "__main__":
    unittest.main()
