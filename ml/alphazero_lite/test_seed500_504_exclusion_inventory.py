"""Focused tamper tests for seed500/#504 source authentication."""

from __future__ import annotations

import hashlib
import json
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

    def test_updated_local_manifest_cannot_rebind_changed_shard(self) -> None:
        scratch_root = Path(__file__).resolve().parents[2] / ".tmp"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_root) as scratch:
            root = Path(scratch)
            shard = root / "shard.jsonl"
            manifest = root / "manifest.json"
            shard.write_bytes(b"published-row\n")
            manifest.write_bytes(
                json.dumps(
                    {"shard_sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}
                ).encode()
            )
            pinned_manifest = hashlib.sha256(manifest.read_bytes()).hexdigest()

            # Simulate an attacker changing the source and rewriting the local
            # manifest and derived report to agree with the replacement bytes.
            shard.write_bytes(b"replacement-row\n")
            replacement_sha = hashlib.sha256(shard.read_bytes()).hexdigest()
            manifest.write_bytes(json.dumps({"shard_sha256": replacement_sha}).encode())
            (root / "identity-index.jsonl").write_bytes(
                json.dumps(
                    {"row": shard.read_text(), "sha256": replacement_sha}
                ).encode()
            )
            (root / "count-report.json").write_text(
                json.dumps({"source_rows": 1, "source_sha256": replacement_sha})
            )
            with self.assertRaisesRegex(ValueError, "digest_mismatch:manifest.json"):
                require_digest(manifest, pinned_manifest, "manifest.json")


if __name__ == "__main__":
    unittest.main()
