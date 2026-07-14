"""Focused tests for immutable reproducibility-audit inputs and serialization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.alphazero_lite.run_training_reproducibility_audit import (
    FIXTURE_TRAIN_ROWS,
    FIXTURE_VALIDATION_ROWS,
    build_fixture,
    fixed_npz_bytes,
    read_jsonl,
    sha256_file,
)


class TrainingReproducibilityAuditTest(unittest.TestCase):
    def _rows(self) -> list[dict]:
        return [
            {
                "game_index": index // 8,
                "state": [0.0] * 27,
                "policy": [1.0, 0, 0, 0, 0, 0],
                "value": 0.0,
            }
            for index in range(FIXTURE_TRAIN_ROWS + FIXTURE_VALIDATION_ROWS)
        ]

    def test_fixture_has_fixed_row_membership_and_two_epoch_batch_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture_dir = Path(temporary) / "fixture"
            manifest = build_fixture(self._rows(), fixture_dir, seed=42)
            self.assertEqual(
                FIXTURE_TRAIN_ROWS, len(read_jsonl(fixture_dir / "train.jsonl"))
            )
            self.assertEqual(
                FIXTURE_VALIDATION_ROWS,
                len(read_jsonl(fixture_dir / "validation.jsonl")),
            )
            plan = np.load(fixture_dir / "batch_indexes.npy", allow_pickle=False)
            self.assertEqual((128, 32), plan.shape)
            self.assertEqual(FIXTURE_TRAIN_ROWS, len(np.unique(plan[:64])))
            self.assertEqual(3, len(manifest["files"]))
            self.assertEqual(
                manifest["files"]["train.jsonl"],
                sha256_file(fixture_dir / "train.jsonl"),
            )

    def test_fixed_npz_serialization_is_byte_identical_and_sorted(self) -> None:
        arrays = {
            "z": np.asarray([2], dtype=np.float32),
            "a": np.asarray([1], dtype=np.float32),
        }
        first = fixed_npz_bytes(arrays)
        second = fixed_npz_bytes(dict(reversed(list(arrays.items()))))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
