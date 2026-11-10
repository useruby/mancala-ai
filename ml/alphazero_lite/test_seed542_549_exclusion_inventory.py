"""Corruption tests for the state and membership source bindings."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from ml.alphazero_lite.seed542_549_exclusion_inventory import (
    authenticate,
    derivative_state_pairs,
    identity_record,
    legacy_canonical_sha256,
    require_state_membership,
)
from ml.alphazero_lite.self_play import encode_state

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

    def test_changed_seed554_derivative_rejected_by_bound_digest(self) -> None:
        scratch_root = ROOT / ".tmp"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_root) as scratch:
            path = Path(scratch) / "control-shard-000.jsonl"
            path.write_bytes(b'{"state": [4]}\n')
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            path.write_bytes(b'{"state": [5]}\n')
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(
                ValueError,
                f"source_digest_mismatch:fixture-554-shard:expected={expected}:observed={observed}",
            ):
                derivative_state_pairs(path, expected, "fixture-554-shard")

    def test_rebound_derivative_hash_still_rejects_new_states(self) -> None:
        scratch_root = ROOT / ".tmp"
        scratch_root.mkdir(exist_ok=True)
        start = {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        changed = {**start, "player_pits": [3, 5, 4, 4, 4, 4]}
        with tempfile.TemporaryDirectory(dir=scratch_root) as scratch:
            path = Path(scratch) / "control-shard-000.jsonl"
            path.write_text(
                json.dumps({"state": encode_state(start, input_encoding="kalah_v3")})
                + "\n"
            )
            first_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            observed, _ = derivative_state_pairs(path, first_hash, "fixture-554-shard")
            start_ids = identity_record(encode_state(start, input_encoding="kalah_v3"))
            start_pair = (legacy_canonical_sha256(start_ids[2]), start_ids[1])
            require_state_membership(observed, Counter({start_pair: 1}), "fixture-554")

            # Even if a local binding is rewritten to the changed shard digest,
            # comparing decoded inputs with authenticated #549 rejects it.
            path.write_text(
                json.dumps({"state": encode_state(changed, input_encoding="kalah_v3")})
                + "\n"
            )
            updated_local_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            changed_pairs, _ = derivative_state_pairs(
                path, updated_local_hash, "fixture-554-shard"
            )
            with self.assertRaisesRegex(
                ValueError, "seed554_derivative_state_membership_changed:fixture-554"
            ):
                require_state_membership(
                    changed_pairs, Counter({start_pair: 1}), "fixture-554"
                )


if __name__ == "__main__":
    unittest.main()
