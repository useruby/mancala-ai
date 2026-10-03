"""Fail-closed checks for restored seed398 bundle inputs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.prepare_seed398_recovery_bundle import verify_inventory


class RecoveryBundleTests(unittest.TestCase):
    def test_inventory_rejects_missing_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "bundle_file_missing"):
                verify_inventory(
                    Path(temporary_directory),
                    [{"bundle_path": "missing.bin", "sha256": "0" * 64}],
                )

    def test_inventory_rejects_changed_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "artifact.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "bundle_file_hash_mismatch"):
                verify_inventory(
                    root,
                    [{"bundle_path": "artifact.bin", "sha256": "0" * 64}],
                )


if __name__ == "__main__":
    unittest.main()
