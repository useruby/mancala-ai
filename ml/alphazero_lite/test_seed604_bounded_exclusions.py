"""Synthetic checks for the bounded Seed 604 exclusion contract."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.seed604_bounded_exclusions import (
    EXECUTION_STATUS_KEYS,
    REQUIRED_POLICY_POPULATIONS,
    load_authenticated_published_policy,
    require_complete_population_manifest,
    _validate_candidate_identities,
    validate_candidate_against_published_policy,
    validate_published_policy_document,
    verify_source_bytes,
)


def _published_document_fixture() -> dict:
    sources = {
        name: {
            "path": f"fixture/{name}.json",
            "source_commit": "1" * 40,
            "registration_sha256": "2" * 64,
            "file_sha256": "3" * 64,
            "records": 1,
            "canonical_identity_set": {"distinct": 1, "sha256": "4" * 64},
            "production_float32_identity_set": {
                "distinct": 1,
                "sha256": "5" * 64,
            },
            "verification_status": "passed",
        }
        for name in REQUIRED_POLICY_POPULATIONS
    }
    sources["seed561_historical_inventory"].update(
        {
            "canonical_index_sha256": "6" * 64,
            "float32_index_sha256": "7" * 64,
        }
    )
    for arm in ("A", "B"):
        sources[f"seed599_training_{arm}"].update({"training_freeze_sha256": "8" * 64})
    return {
        "schema": "seed604-bounded-exclusion-policy-v1",
        "status": "proposed-awaiting-project-owner-review",
        "sources": sources,
        "execution_status": {name: False for name in EXECUTION_STATUS_KEYS},
    }


class Seed604BoundedExclusionTests(unittest.TestCase):
    def test_published_policy_recomputes_from_registered_source_bytes(self) -> None:
        policy = load_authenticated_published_policy()
        accounting = policy.document["observed_overlap_accounting"]
        self.assertEqual(accounting["bounded_union_per_scheme"], 537698)
        self.assertEqual(
            accounting["bounded_union_identity_set_sha256"]["canonical"],
            "cd84d06ac97ed9918ad1727a9abe19d7432272a1f24140ba4ca9317bbcd72aa0",
        )
        self.assertEqual(
            accounting["bounded_union_identity_set_sha256"]["production_float32"],
            "e369414d72ab36d2c7e9be2280c89a4ecde61fb360c1771879ac8cdc2e8b708a",
        )
        self.assertFalse(
            policy.document["execution_status"]["evaluation_openings_generated"]
        )
        with self.assertRaisesRegex(
            AttributeError, "authenticated_bounded_policy_is_immutable"
        ):
            policy._excluded = {}

        training_rows = json.loads(
            Path(
                "docs/data/seed599-replay-coverage/control-membership.json"
            ).read_bytes()
        )
        with self.assertRaisesRegex(
            ValueError, "candidate_canonical_exclusion_collision"
        ):
            validate_candidate_against_published_policy(
                training_rows[0]["canonical_state"], policy
            )

    def test_candidate_validation_requires_authenticated_loader_result(self) -> None:
        with self.assertRaisesRegex(
            TypeError, "candidate_validation_requires_authenticated_published_policy"
        ):
            validate_candidate_against_published_policy(
                {"canonical_state": {}}, {"sources": {}}
            )  # type: ignore[arg-type]

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
            _validate_candidate_identities("a" * 64, "c" * 64, excluded)
        with self.assertRaisesRegex(
            ValueError, "candidate_float32_exclusion_collision"
        ):
            _validate_candidate_identities("c" * 64, "b" * 64, excluded)
        _validate_candidate_identities("c" * 64, "d" * 64, excluded)

    def test_altered_identity_manifest_is_rejected(self) -> None:
        expected = _published_document_fixture()
        changed = json.loads(json.dumps(expected))
        changed["sources"]["seed599_training_A"]["canonical_identity_set"]["sha256"] = (
            "0" * 64
        )
        with self.assertRaisesRegex(
            ValueError, "bounded_exclusion_policy_source_binding_mismatch"
        ):
            validate_published_policy_document(changed, expected)

    def test_seed561_index_hash_changes_are_rejected(self) -> None:
        expected = _published_document_fixture()
        for field in ("canonical_index_sha256", "float32_index_sha256"):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(expected))
                changed["sources"]["seed561_historical_inventory"][field] = "f" * 64
                with self.assertRaisesRegex(
                    ValueError, "seed561_policy_index_hash_mismatch"
                ):
                    validate_published_policy_document(changed, expected)

    def test_published_identity_set_hash_changes_are_rejected(self) -> None:
        expected = _published_document_fixture()
        changed = json.loads(json.dumps(expected))
        changed["sources"]["seed599_training_A"]["canonical_identity_set"]["sha256"] = (
            "f" * 64
        )
        with self.assertRaisesRegex(
            ValueError, "bounded_exclusion_policy_source_binding_mismatch"
        ):
            validate_published_policy_document(changed, expected)

    def test_seed561_canonical_index_hash_tampering_is_rejected(self) -> None:
        self._assert_seed561_index_mutation_rejected("canonical_index_sha256")

    def test_seed561_float32_index_hash_tampering_is_rejected(self) -> None:
        self._assert_seed561_index_mutation_rejected("float32_index_sha256")

    def test_missing_execution_status_flag_fails_closed(self) -> None:
        expected = _published_document_fixture()
        changed = json.loads(json.dumps(expected))
        del changed["execution_status"]["training_started"]
        with self.assertRaisesRegex(
            ValueError, "bounded_exclusion_policy_execution_status_schema_mismatch"
        ):
            validate_published_policy_document(changed, expected)

    def test_true_execution_status_flag_fails_closed(self) -> None:
        expected = _published_document_fixture()
        changed = json.loads(json.dumps(expected))
        changed["execution_status"]["training_started"] = True
        with self.assertRaisesRegex(
            ValueError, "bounded_exclusion_policy_execution_status_invalid"
        ):
            validate_published_policy_document(changed, expected)

    def test_missing_source_registration_fails_closed(self) -> None:
        expected = _published_document_fixture()
        changed = json.loads(json.dumps(expected))
        del changed["sources"]["seed599_evaluation_roots"]
        with self.assertRaisesRegex(
            ValueError, "bounded_exclusion_policy_source_registration_missing"
        ):
            validate_published_policy_document(changed, expected)

    def test_changed_training_freeze_reference_is_rejected(self) -> None:
        expected = _published_document_fixture()
        changed = json.loads(json.dumps(expected))
        changed["sources"]["seed599_training_A"]["training_freeze_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            ValueError, "bounded_exclusion_policy_source_binding_mismatch"
        ):
            validate_published_policy_document(changed, expected)

    def _assert_seed561_index_mutation_rejected(self, field: str) -> None:
        expected = _published_document_fixture()
        changed = json.loads(json.dumps(expected))
        changed["sources"]["seed561_historical_inventory"][field] = "f" * 64
        with self.assertRaisesRegex(ValueError, "seed561_policy_index_hash_mismatch"):
            validate_published_policy_document(changed, expected)


if __name__ == "__main__":
    unittest.main()
