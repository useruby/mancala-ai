"""Frozen residual-v3 trunk extraction contracts."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.kalah_rules import KalahGame


class FrozenTrunkValueSufficiencyAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluator = ArtifactEvaluator(Path("model-artifact/current"))
        cls.game = KalahGame([4] * 12, [0, 0], 0)

    def test_extraction_is_deterministic_and_does_not_change_predictions(self) -> None:
        before_policy, before_value = self.evaluator.evaluate(self.game)
        first = self.evaluator.extract_trunk(self.game)
        second = self.evaluator.extract_trunk(self.game)
        after_policy, after_value = self.evaluator.evaluate(self.game)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(before_policy, after_policy)
        self.assertEqual(before_value, after_value)

    def test_reapplying_normal_heads_exactly_reproduces_evaluator(self) -> None:
        trunk = self.evaluator.extract_trunk(self.game)
        expected_policy, expected_value = self.evaluator.evaluate(self.game)
        actual_policy, actual_value = self.evaluator.apply_heads_to_trunk(
            trunk, self.game
        )
        np.testing.assert_array_equal(expected_policy, actual_policy)
        self.assertEqual(expected_value, actual_value)


if __name__ == "__main__":
    unittest.main()
