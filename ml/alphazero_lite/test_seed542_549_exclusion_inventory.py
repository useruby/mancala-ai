"""Corruption tests for the state and membership source bindings."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.seed542_549_exclusion_inventory import authenticate

ROOT = Path(__file__).resolve().parents[2]


class Seed542549AuthenticationTests(unittest.TestCase):
    def _assert_mutation_rejected(
        self, filename: str, before: bytes, after: bytes
    ) -> None:
        scratch_root = ROOT / ".tmp"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_root) as scratch:
            path = Path(scratch) / filename
            path.write_bytes(before)
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            path.write_bytes(after)
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            label = f"tamper-fixture/{filename}"
            with self.assertRaisesRegex(
                ValueError,
                f"source_digest_mismatch:{label}:expected={expected}:observed={observed}",
            ):
                authenticate(path, expected, label)

    def test_changed_seed542_shard_rejected(self) -> None:
        self._assert_mutation_rejected("shard-542048.jsonl", b"row-a\n", b"row-b\n")

    def test_changed_seed542_membership_rejected(self) -> None:
        self._assert_mutation_rejected(
            "development-membership.json",
            b'{"included":false}\n',
            b'{"included":true}\n',
        )

    def test_changed_seed549_corpus_shard_rejected(self) -> None:
        self._assert_mutation_rejected(
            "seed549-control.jsonl", b"state-a\n", b"state-b\n"
        )

    def test_changed_seed549_shared_membership_rejected(self) -> None:
        self._assert_mutation_rejected(
            "shared-development-membership.jsonl",
            b'{"included":false}\n',
            b'{"included":true}\n',
        )


if __name__ == "__main__":
    unittest.main()
