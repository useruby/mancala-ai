import unittest

import numpy as np

from ml.alphazero_lite import arena
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_joint_heads_arena_failure_attribution import (
    LANES,
    composition_hash,
)


class _Evaluator:
    def __init__(self, policy, value):
        self.policy = np.asarray(policy, dtype=np.float32)
        self.value = value

    def evaluate(self, _game):
        return self.policy.copy(), self.value


class JointHeadsArenaFailureAttributionTests(unittest.TestCase):
    def setUp(self):
        self.game = KalahGame(pits=[4] * 12, captured_seeds=[0, 0], current_player=0)
        self.current = _Evaluator([0.1, 0.2, 0.3, 0.4, 0.0, 0.0], 0.25)
        self.candidate = _Evaluator([0.4, 0.3, 0.2, 0.1, 0.0, 0.0], -0.5)

    def test_compositions_select_requested_outputs_and_preserve_masks(self):
        outputs = {}
        for name, (policy_source, value_source) in LANES.items():
            evaluator = arena.ComposedArtifactEvaluator(
                self.current,
                self.candidate,
                policy_source=policy_source,
                value_source=value_source,
            )
            outputs[name] = evaluator.evaluate(self.game)
        np.testing.assert_array_equal(
            outputs["current_policy_current_value"][0], self.current.policy
        )
        self.assertEqual(outputs["current_policy_current_value"][1], self.current.value)
        np.testing.assert_array_equal(
            outputs["candidate_policy_candidate_value"][0], self.candidate.policy
        )
        self.assertEqual(
            outputs["candidate_policy_candidate_value"][1], self.candidate.value
        )
        np.testing.assert_array_equal(
            outputs["candidate_policy_current_value"][0], self.candidate.policy
        )
        self.assertEqual(
            outputs["candidate_policy_current_value"][1], self.current.value
        )
        np.testing.assert_array_equal(
            outputs["current_policy_candidate_value"][0], self.current.policy
        )
        self.assertEqual(
            outputs["current_policy_candidate_value"][1], self.candidate.value
        )

    def test_terminal_values_are_not_composed(self):
        terminal = KalahGame(pits=[0] * 12, captured_seeds=[24, 24], current_player=0)
        evaluator = arena.ComposedArtifactEvaluator(
            self.current,
            self.candidate,
            policy_source="candidate",
            value_source="current",
        )
        policy, value = evaluator.evaluate(terminal)
        np.testing.assert_array_equal(policy, np.zeros(6, dtype=np.float32))
        self.assertEqual(value, 0.0)

    def test_profile_hash_includes_both_artifacts_and_lane(self):
        first = composition_hash(
            "current-a", "candidate-a", "candidate_policy_current_value"
        )
        self.assertNotEqual(
            first,
            composition_hash(
                "current-b", "candidate-a", "candidate_policy_current_value"
            ),
        )
        self.assertNotEqual(
            first,
            composition_hash(
                "current-a", "candidate-b", "candidate_policy_current_value"
            ),
        )
        self.assertNotEqual(
            first,
            composition_hash(
                "current-a", "candidate-a", "current_policy_candidate_value"
            ),
        )


if __name__ == "__main__":
    unittest.main()
