"""Regression coverage for frozen checkpoint trajectory diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from ml.alphazero_lite.checkpoint_trajectory_diagnostic import js, phase_masks


class CheckpointTrajectoryDiagnosticTest(unittest.TestCase):
    def test_phase_masks_use_active_stones_and_move_index(self) -> None:
        rows = [
            {
                "move_index": 12,
                "state": {
                    "player_pits": [6, 6, 6, 6, 6, 6],
                    "opponent_pits": [0, 0, 0, 0, 0, 0],
                    "player_store": 0,
                    "opponent_store": 12,
                    "current_player": 0,
                },
            },
            {
                "move_index": 13,
                "state": {
                    "player_pits": [1, 1, 1, 1, 1, 1],
                    "opponent_pits": [0, 0, 0, 0, 0, 0],
                    "player_store": 0,
                    "opponent_store": 42,
                    "current_player": 0,
                },
            },
        ]

        masks = phase_masks(rows, np.asarray([0, 1]))

        self.assertEqual([True, False], masks["high_stone"].tolist())
        self.assertEqual([True, False], masks["opening"].tolist())
        self.assertEqual([True, False], masks["opening_high_stone"].tolist())
        self.assertEqual([False, True], masks["solver_owned"].tolist())

    def test_policy_js_is_deterministic_and_zero_for_identical_policies(self) -> None:
        policy = np.asarray([0.7, 0.3, 0.0])

        self.assertEqual(0.0, js(policy, policy))
        self.assertAlmostEqual(
            0.06665370714512762, js(policy, np.asarray([0.4, 0.6, 0.0]))
        )


if __name__ == "__main__":
    unittest.main()
