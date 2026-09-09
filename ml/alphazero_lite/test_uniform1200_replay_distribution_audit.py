"""Focused contracts for the read-only uniform1200 replay audit."""

from __future__ import annotations

import unittest

from ml.alphazero_lite.run_uniform1200_replay_distribution_audit import (
    MIN_BUCKET_ROWS,
    consistent_positions,
    direction_consistency,
    entropy,
    forensic_change,
    jensen_shannon,
    matched_target_audit,
    replay_features,
    total_variation,
)


class Uniform1200ReplayDistributionAuditTest(unittest.TestCase):
    def test_entropy_tv_and_js_are_deterministic(self) -> None:
        self.assertEqual(0.0, entropy([1.0, 0.0]))
        self.assertEqual(1.0, total_variation([1.0, 0.0], [0.0, 1.0]))
        self.assertAlmostEqual(1.0, jensen_shannon([1.0, 0.0], [0.0, 1.0]))

    def test_extra_turn_and_capture_features_use_rule_engine(self) -> None:
        # Encoding has 12 pits, two stores, current-player, then auxiliary planes.
        row = {
            "state": [0, 0, 0, 0, 0, 1 / 48, 0, 0, 0, 0, 0, 1 / 48, 0, 0, 0] + [0] * 12,
            "policy": [0, 0, 0, 0, 0, 1],
            "value": 0.0,
            "move_index": 9,
            "winner": 0,
        }
        _key, feature = replay_features(row)
        self.assertTrue(feature["extra_turn_available"])
        row["state"] = [1 / 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 48, 0, 0, 0, 0] + [
            0
        ] * 12
        _key, feature = replay_features(row)
        self.assertTrue(feature["capture_available"])

    def test_shared_state_audit_aggregates_duplicates(self) -> None:
        feature = {
            "phase": "early",
            "legal_move_count": 2,
            "extra_turn_available": False,
            "capture_available": False,
            "outcome": "0",
            "current_player": 0,
        }
        control = {
            "states": {
                "x": {
                    "count": 2,
                    "feature": feature,
                    "policy": [1, 0, 0, 0, 0, 0],
                    "value": 0.5,
                    "policy_varies": True,
                    "value_varies": True,
                }
            }
        }
        uniform = {
            "states": {
                "x": {
                    "count": 1,
                    "feature": feature,
                    "policy": [0, 1, 0, 0, 0, 0],
                    "value": -0.5,
                    "policy_varies": False,
                    "value_varies": False,
                }
            }
        }
        audit = matched_target_audit(control, uniform)
        self.assertEqual(1, audit["aggregate"]["overall"]["states"])
        self.assertEqual(
            1, audit["duplicate_target_variation"]["control_policy_varying_states"]
        )

    def test_correct_to_wrong_and_two_of_three_rule(self) -> None:
        change = forensic_change(
            {"p": {"agrees_top1": True, "regret": 0.0}},
            {"p": {"agrees_top1": False, "regret": 0.1}},
        )
        self.assertEqual(["p"], change["regressions"])
        self.assertEqual(
            ["p"],
            consistent_positions(
                {
                    "44": change,
                    "45": change,
                    "46": {"regressions": [], "improvements": []},
                },
                "regressions",
            ),
        )

    def test_registered_support_threshold_is_not_arbitrary_at_runtime(self) -> None:
        self.assertEqual(200, MIN_BUCKET_ROWS)

    def test_cross_seed_direction_consistency_requires_all_three(self) -> None:
        result = direction_consistency([0.2, 0.1, 0.3])
        self.assertTrue(result["all_three_agree"])
        self.assertFalse(direction_consistency([0.2, -0.1, 0.3])["all_three_agree"])


if __name__ == "__main__":
    unittest.main()
