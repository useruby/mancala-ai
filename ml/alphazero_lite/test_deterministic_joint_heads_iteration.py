"""Focused tests for the deterministic joint-head iteration protocol."""

from __future__ import annotations

import unittest

import numpy as np

from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    compare_runs,
    freeze_joint_heads,
    game_split,
)
from ml.alphazero_lite.train import PolicyValueNet, input_size_for_encoding


class DeterministicJointHeadsIterationTest(unittest.TestCase):
    def _rows(self) -> list[dict]:
        rows = []
        for game in range(20):
            for ply in range(3):
                rows.append(
                    {
                        "game_index": game,
                        "value": [-1.0, 0.0, 1.0][game % 3],
                        "seat_context": f"player_{game % 2}",
                        "game_length": 20 + game,
                        "state": [0.0] * 27,
                        "policy": [1.0, 0, 0, 0, 0, 0],
                    }
                )
        return rows

    def test_game_split_is_seeded_disjoint_and_game_level(self) -> None:
        train, validation = game_split(self._rows(), 42)
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertFalse(set(train) & set(validation))
        self.assertEqual((train, validation), game_split(self._rows(), 42))

    def test_freeze_joint_heads_leaves_only_specialized_heads_trainable(self) -> None:
        model = PolicyValueNet(
            (96, 3), "residual_v3", input_size_for_encoding("kalah_v3")
        )
        names = freeze_joint_heads(model)
        self.assertEqual(8, len(names))
        self.assertTrue(all("residual_layers" not in name for name in names))
        self.assertTrue(
            all(
                parameter.requires_grad == (name in names)
                for name, parameter in model.named_parameters()
            )
        )

    def test_reproduction_accepts_matching_hashes_and_tolerant_predictions(
        self,
    ) -> None:
        base = {
            "parameter_sha256": {"x": "a"},
            "checkpoint_sha256": "one",
            "artifact_weights_sha256": "two",
            "captures": {
                "initialization": {
                    "parameter_sha256": {"x": "a"},
                    "optimizer_state_sha256": "optimizer",
                    "checkpoint_sha256": "checkpoint",
                }
            },
            "validation_logits": np.array([[1.0, 0.0]]),
            "validation_values": np.array([[0.2]]),
        }
        identical = {
            **base,
            "validation_logits": base["validation_logits"].copy(),
            "validation_values": base["validation_values"].copy(),
        }
        self.assertTrue(compare_runs(base, identical)["passes"])
        close = {
            **identical,
            "parameter_sha256": {"x": "b"},
            "validation_values": np.array([[0.20000001]]),
        }
        self.assertTrue(compare_runs(base, close)["passes"])
        divergent = {**close, "validation_logits": np.array([[0.0, 1.0]])}
        self.assertFalse(compare_runs(base, divergent)["passes"])


if __name__ == "__main__":
    unittest.main()
