"""Focused guardrail tests for the D384-versus-D1200 policy-teacher runner."""

from __future__ import annotations

import unittest

from ml.alphazero_lite.run_stronger_policy_teacher_ablation import (
    classify,
    paired_invariants,
    policy_head_only,
)
from ml.alphazero_lite.train import PolicyValueNet, input_size_for_encoding


def row(policy: list[float]) -> dict:
    return {
        "game_id": 1,
        "game_index": 1,
        "ply": 0,
        "move_index": 0,
        "state": [0.0] * 27,
        "player": 0,
        "chosen_gameplay_move": 2,
        "winner": 0,
        "terminal_outcome_target": 1.0,
        "value": 1.0,
        "trajectory_hash": "trajectory",
        "policy": policy,
        "policy_teacher": "D384",
        "policy_teacher_telemetry": {"target_hash": "a"},
    }


class StrongerPolicyTeacherAblationTest(unittest.TestCase):
    def test_policy_head_scope_excludes_trunk_and_value_head(self) -> None:
        model = PolicyValueNet(
            (96, 3), "residual_v3", input_size_for_encoding("kalah_v3")
        )
        names = policy_head_only(model)
        self.assertEqual(4, len(names))
        self.assertTrue(all(name.startswith("policy_") for name in names))
        self.assertTrue(
            all(
                parameter.requires_grad == (name in names)
                for name, parameter in model.named_parameters()
            )
        )

    def test_policy_and_teacher_telemetry_are_the_only_lane_differences(self) -> None:
        left, right = row([1, 0, 0, 0, 0, 0]), row([0, 1, 0, 0, 0, 0])
        right["policy_teacher"] = "D1200"
        right["policy_teacher_telemetry"] = {"target_hash": "b"}
        result = paired_invariants([left], [right])
        self.assertTrue(result["passes"])
        self.assertIn("D384_policies", result["hashes"])
        self.assertIn("D1200_policies", result["hashes"])

    def test_non_policy_mismatch_invalidates_ablation(self) -> None:
        left, right = row([1, 0, 0, 0, 0, 0]), row([0, 1, 0, 0, 0, 0])
        right["policy_teacher"] = "D1200"
        right["chosen_gameplay_move"] = 3
        self.assertFalse(paired_invariants([left], [right])["passes"])

    def test_invalid_and_nondeterministic_results_have_priority(self) -> None:
        self.assertEqual(
            "paired_teacher_ablation_invalid",
            classify(invariants={"passes": False}, reproducible=False),
        )
        self.assertEqual(
            "full_scale_training_nondeterministic",
            classify(invariants={"passes": True}, reproducible=False),
        )


if __name__ == "__main__":
    unittest.main()
