"""Synthetic checks for the bounded Seed 604 exclusion contract."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.seed604_bounded_exclusions import (
    identity_set_sha256,
    require_complete_population_manifest,
    validate_candidate_identities,
    verify_identity_union_bindings,
    verify_published_policy,
    verify_source_bytes,
)


class Seed604BoundedExclusionTests(unittest.TestCase):
    def test_published_policy_recomputes_from_registered_source_bytes(self) -> None:
        result = verify_published_policy()
        self.assertTrue(result["passed"])
        self.assertEqual(result["identity_unions"]["canonical"]["distinct"], 537698)
        self.assertEqual(result["identity_unions"]["float32"]["distinct"], 537698)
        self.assertFalse(result["evaluation_openings_generated"])

    def test_missing_required_seed599_population_fails_closed(self) -> None:
        partial = {
            "seed599_training_A": {},
            "seed599_training_B": {},
            "seed599_heldout": {},
            "seed561_historical_inventory": {},
        }
        with self.assertRaisesRegex(
            ValueError, "required_exclusion_population_missing:seed599_evaluation_roots"
        ):
            require_complete_population_manifest(partial)
        del partial["seed561_historical_inventory"]
        partial["seed599_evaluation_roots"] = {}
        with self.assertRaisesRegex(
            ValueError,
            "required_exclusion_population_missing:seed561_historical_inventory",
        ):
            require_complete_population_manifest(partial)

    def test_altered_or_missing_source_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.json"
            source.write_bytes(b'{"states": [1]}')
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(
                verify_source_bytes(source, expected, "fixture"), source.read_bytes()
            )

            source.write_bytes(b'{"states": [2]}')
            with self.assertRaisesRegex(
                ValueError, "exclusion_source_hash_mismatch:fixture"
            ):
                verify_source_bytes(source, expected, "fixture")
            source.unlink()
            with self.assertRaisesRegex(ValueError, "exclusion_source_missing:fixture"):
                verify_source_bytes(source, expected, "fixture")

    def test_canonical_only_and_float32_only_collisions_are_rejected(self) -> None:
        excluded = {"canonical": {"a" * 64}, "float32": {"b" * 64}}
        with self.assertRaisesRegex(
            ValueError, "candidate_canonical_exclusion_collision"
        ):
            validate_candidate_identities("a" * 64, "c" * 64, excluded)
        with self.assertRaisesRegex(
            ValueError, "candidate_float32_exclusion_collision"
        ):
            validate_candidate_identities("c" * 64, "b" * 64, excluded)
        validate_candidate_identities("c" * 64, "d" * 64, excluded)

    def test_altered_identity_manifest_is_rejected(self) -> None:
        excluded = {"canonical": {"a" * 64}, "float32": {"b" * 64}}
        policy = {
            "identity_unions": {
                "canonical": {
                    "distinct": 1,
                    "identity_set_sha256": identity_set_sha256(excluded["canonical"]),
                },
                "float32": {
                    "distinct": 1,
                    "identity_set_sha256": "0" * 64,
                },
            }
        }
        with self.assertRaisesRegex(
            ValueError, "exclusion_identity_manifest_mismatch:float32"
        ):
            verify_identity_union_bindings(policy, excluded)


if __name__ == "__main__":
    unittest.main()
