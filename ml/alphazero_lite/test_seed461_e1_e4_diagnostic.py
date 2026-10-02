"""Frozen-input checks for the original O0 epoch diagnostic."""

from __future__ import annotations

import unittest

from ml.alphazero_lite.seed461_e1_e4_diagnostic import (
    EPOCHS,
    registered_inputs,
)


class FrozenOriginalEpochTests(unittest.TestCase):
    def test_original_epoch_checkpoints_and_exports_match_published_binding(
        self,
    ) -> None:
        inputs = registered_inputs()
        for epoch, expected in EPOCHS.items():
            self.assertEqual(
                inputs["epochs"][epoch]["source_checkpoint_sha256"],
                expected["checkpoint_sha256"],
            )
            self.assertEqual(
                inputs["epochs"][epoch]["artifact_sha256"]["model.npz"],
                expected["checkpoint_sha256"],
            )
        self.assertEqual(
            inputs["runtime_contract"]["runtime_search_policy_sha256"],
            "b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d",
        )


if __name__ == "__main__":
    unittest.main()
