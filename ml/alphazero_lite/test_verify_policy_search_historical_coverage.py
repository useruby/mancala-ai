"""Regression tests for the policy/search retrospective coverage verifier."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from ml.alphazero_lite.verify_policy_search_historical_coverage import (
    index_unique,
    validate_search_pair,
    verify_bytes_identity,
)


class HistoricalCoverageVerifierTests(unittest.TestCase):
    def test_source_substitution_is_rejected(self) -> None:
        original = b"registered source bytes"
        expected = hashlib.sha256(original).hexdigest()
        with self.assertRaisesRegex(ValueError, "historical_source_identity_mismatch"):
            verify_bytes_identity(
                Path("evidence.jsonl"), b"substituted bytes", expected
            )

    def test_incorrect_state_identity_does_not_join(self) -> None:
        policy = index_unique(
            [{"canonical_state_identity": "position-player-0"}],
            "canonical_state_identity",
        )
        exact = index_unique(
            [{"canonical_identity": "position-player-1"}], "canonical_identity"
        )
        self.assertEqual(set(), policy.keys() & exact.keys())

    def test_player_perspective_mismatch_is_rejected(self) -> None:
        row = {
            "b512_move": 1,
            "s615_move": 2,
            "player_to_move": 0,
            "b512_seed": 11,
            "s615_seed": 12,
            "state_before_divergence": {"current_player": 1},
            "policy_prior_comparison": {"canonical_state_identity": "state"},
            "canonical_state_identity": "state",
        }
        with self.assertRaisesRegex(ValueError, "player_perspective_mismatch"):
            validate_search_pair(row)

    def test_altered_action_margins_are_rejected_by_source_anchor(self) -> None:
        original = b'{"exact_action_margins":{"0":-8}}\n'
        expected = hashlib.sha256(original).hexdigest()
        altered = b'{"exact_action_margins":{"0":8}}\n'
        with self.assertRaisesRegex(ValueError, "historical_source_identity_mismatch"):
            verify_bytes_identity(Path("exact-results.jsonl"), altered, expected)

    def test_substituted_search_action_is_rejected_by_source_anchor(self) -> None:
        original = b'{"b512_move":4,"s615_move":2}\n'
        expected = hashlib.sha256(original).hexdigest()
        substituted = b'{"b512_move":3,"s615_move":2}\n'
        with self.assertRaisesRegex(ValueError, "historical_source_identity_mismatch"):
            verify_bytes_identity(
                Path("first-divergences.jsonl"), substituted, expected
            )


if __name__ == "__main__":
    unittest.main()
