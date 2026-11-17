"""Synthetic failure-path tests for Seed 604 cohort verification."""

from __future__ import annotations

import unittest

from ml.alphazero_lite.seed604_bounded_exclusions import (
    _validate_candidate_identities,
)
from ml.alphazero_lite.verify_seed604_opening_cohort import (
    _independent_replay,
    require_digest,
    require_distinct_identities,
)


class Seed604CohortNegativeTests(unittest.TestCase):
    def test_altered_source_hash_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "generator_source_hash_mismatch"):
            require_digest(b"altered source", "0" * 64, "generator_source")

    def test_manifest_digest_rejects_changed_bytes(self) -> None:
        with self.assertRaisesRegex(ValueError, "manifest_hash_mismatch"):
            require_digest(b"changed manifest", "0" * 64, "manifest")

    def test_modified_exclusion_binding_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exclusion_policy_hash_mismatch"):
            require_digest(b"modified policy", "0" * 64, "exclusion_policy")

    def test_altered_prefix_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "illegal_prefix_move"):
            _independent_replay([99] + [0] * 11)

    def test_incomplete_prefix_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "prefix_length_mismatch"):
            _independent_replay([0] * 11)

    def test_duplicate_root_rejected(self) -> None:
        row = {"canonical_identity": "a" * 64, "float32_identity": "b" * 64}
        with self.assertRaisesRegex(ValueError, "duplicate_canonical_root"):
            require_distinct_identities([row, dict(row)])

    def test_canonical_only_exclusion_collision_rejected(self) -> None:
        excluded = {"canonical": {"a" * 64}, "float32": set()}
        with self.assertRaisesRegex(
            ValueError, "candidate_canonical_exclusion_collision"
        ):
            _validate_candidate_identities("a" * 64, "b" * 64, excluded)

    def test_float32_only_exclusion_collision_rejected(self) -> None:
        excluded = {"canonical": set(), "float32": {"b" * 64}}
        with self.assertRaisesRegex(
            ValueError, "candidate_float32_exclusion_collision"
        ):
            _validate_candidate_identities("a" * 64, "b" * 64, excluded)


if __name__ == "__main__":
    unittest.main()
