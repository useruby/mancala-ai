"""Focused tamper tests for seed500/#504 source authentication."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.seed500_504_exclusion_inventory import require_digest


class Seed500504AuthenticationTests(unittest.TestCase):
    def test_changed_shard_bytes_are_rejected_with_both_digests(self) -> None:
        scratch_root = Path(__file__).resolve().parents[2] / ".tmp"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_root) as scratch:
            path = Path(scratch) / "shard.jsonl"
            path.write_bytes(b'{"state":[1]}\n')
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            path.write_bytes(b'{"state":[2]}\n')
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(
                ValueError,
                f"digest_mismatch:shard.jsonl:expected={expected}:observed={observed}",
            ):
                require_digest(path, expected, "shard.jsonl")

    def test_changed_split_assignment_bytes_are_rejected(self) -> None:
        scratch_root = Path(__file__).resolve().parents[2] / ".tmp"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_root) as scratch:
            path = Path(scratch) / "split.json"
            path.write_bytes(b'{"training":[],"development":[]}\n')
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            path.write_bytes(b'{"training":[1],"development":[]}\n')
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(
                ValueError,
                f"digest_mismatch:split.json:expected={expected}:observed={observed}",
            ):
                require_digest(path, expected, "split.json")


if __name__ == "__main__":
    unittest.main()
