"""Unit tests for deterministic PR #242 target entropy factorization."""

from __future__ import annotations

import math
import unittest

import numpy as np

from ml.alphazero_lite.run_pr242_target_entropy_factorization import (
    entropy,
    match_entropy,
)


class TargetEntropyFactorizationTest(unittest.TestCase):
    def test_two_legal_moves_increases_entropy(self) -> None:
        policy = [0.9, 0.1, 0, 0, 0, 0]
        result = match_entropy(policy, [0, 1], math.log(2.0) * 0.9)
        self.assertAlmostEqual(entropy(result, [0, 1]), math.log(2.0) * 0.9, places=8)
        self.assertEqual(0, int(np.argmax(result)))

    def test_six_legal_moves_decreases_entropy(self) -> None:
        policy = [0.3, 0.2, 0.15, 0.14, 0.11, 0.1]
        result = match_entropy(policy, list(range(6)), 0.4)
        self.assertAlmostEqual(entropy(result, list(range(6))), 0.4, places=8)
        self.assertEqual(0, int(np.argmax(result)))

    def test_zero_probability_actions_remain_legal(self) -> None:
        policy = [1.0, 0.0, 0.0, 0, 0, 0]
        result = match_entropy(policy, [0, 1, 2], 0.8)
        self.assertAlmostEqual(entropy(result, [0, 1, 2]), 0.8, places=8)
        self.assertEqual([0.0, 0.0, 0.0], result[3:])

    def test_equal_entropy_returns_source_exactly(self) -> None:
        policy = [0.7, 0.3, 0, 0, 0, 0]
        self.assertEqual(policy, match_entropy(policy, [0, 1], entropy(policy, [0, 1])))

    def test_ties_are_deterministic(self) -> None:
        policy = [0.5, 0.5, 0, 0, 0, 0]
        first = match_entropy(policy, [0, 1], 0.2)
        second = match_entropy(policy, [0, 1], 0.2)
        self.assertEqual(first, second)
        self.assertEqual(0, int(np.argmax(first)))


if __name__ == "__main__":
    unittest.main()
