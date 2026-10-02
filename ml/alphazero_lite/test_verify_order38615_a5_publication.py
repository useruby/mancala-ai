"""Mutation tests for the amendment-anchored A5 publication verifier."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest

from ml.alphazero_lite.verify_order38615_a5_publication import (
    validate_public_artifact_hashes,
    validate_public_bindings,
)


def json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class PublicationBindingMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.amendment = {
            "original_hashes": {
                "registration_sha256": "registration",
                "manifest_sha256": "manifest",
                "evaluation_binding_sha256": "binding",
                "opening_score_matrix_sha256": "matrix",
                "game_outcome_accounting_sha256": "accounting",
                "public_results_binding": {
                    "registration_sha256": "registration",
                    "exclusion_manifest_sha256": "manifest",
                    "evaluation_binding_sha256": "binding",
                },
            }
        }
        self.results = {
            "registration_sha256": "registration",
            "evaluation_binding_sha256": "binding",
            "exclusion_manifest_sha256": "manifest",
            "opening_score_matrix_sha256": "matrix",
            "game_outcome_accounting_sha256": "accounting",
            "per_suite_and_pooled_pass": False,
        }
        self.matrix = {
            "exclusion_manifest_sha256": "manifest",
            "game_outcome_accounting_sha256": "accounting",
            "opening_scores": {"395": [0.5]},
        }
        self.results_bytes = b'{"per_suite_and_pooled_pass":false}'
        self.matrix_bytes = b'{"opening_scores":{"395":[0.5]}}'
        self.accounting_bytes = b'{"winner":"challenger"}\n'
        self.amendment["original_hashes"].update(
            {
                "results_sha256": hashlib.sha256(self.results_bytes).hexdigest(),
                "opening_score_matrix_sha256": hashlib.sha256(
                    self.matrix_bytes
                ).hexdigest(),
                "game_outcome_accounting_sha256": hashlib.sha256(
                    self.accounting_bytes
                ).hexdigest(),
            }
        )
        self.results.update(
            {
                "opening_score_matrix_sha256": self.amendment["original_hashes"][
                    "opening_score_matrix_sha256"
                ],
                "game_outcome_accounting_sha256": self.amendment["original_hashes"][
                    "game_outcome_accounting_sha256"
                ],
            }
        )
        self.matrix["game_outcome_accounting_sha256"] = self.amendment[
            "original_hashes"
        ]["game_outcome_accounting_sha256"]

    def test_unchanged_public_child_bindings_pass(self) -> None:
        validate_public_bindings(self.amendment, self.results, self.matrix)

    def test_changed_pass_flag_fails_even_with_child_hash_reference(self) -> None:
        changed = copy.deepcopy(self.results)
        changed["per_suite_and_pooled_pass"] = True
        changed["opening_score_matrix_sha256"] = hashlib.sha256(
            b"new matrix"
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "results_hash_mismatch"):
            validate_public_artifact_hashes(
                self.amendment["original_hashes"],
                json_bytes(changed),
                self.matrix_bytes,
                self.accounting_bytes,
            )
        with self.assertRaisesRegex(ValueError, "opening_score_matrix_sha256"):
            validate_public_bindings(self.amendment, changed, self.matrix)

    def test_changed_score_and_resealed_matrix_hash_fails_anchor(self) -> None:
        changed_bytes = b'{"opening_scores":{"395":[1.0]}}'
        resealed = hashlib.sha256(changed_bytes).hexdigest()
        changed_results = copy.deepcopy(self.results)
        changed_results["opening_score_matrix_sha256"] = resealed
        with self.assertRaisesRegex(ValueError, "results_hash_mismatch"):
            validate_public_artifact_hashes(
                self.amendment["original_hashes"],
                json_bytes(changed_results),
                changed_bytes,
                self.accounting_bytes,
            )

    def test_changed_outcome_and_resealed_child_reference_fails_anchor(self) -> None:
        changed = copy.deepcopy(self.results)
        changed_bytes = b'{"winner":"current"}\n'
        resealed = hashlib.sha256(changed_bytes).hexdigest()
        changed["game_outcome_accounting_sha256"] = resealed
        changed_matrix = copy.deepcopy(self.matrix)
        changed_matrix["game_outcome_accounting_sha256"] = resealed
        with self.assertRaisesRegex(ValueError, "results_hash_mismatch"):
            validate_public_artifact_hashes(
                self.amendment["original_hashes"],
                json_bytes(changed),
                json_bytes(changed_matrix),
                changed_bytes,
            )
        with self.assertRaisesRegex(ValueError, "game_outcome_accounting_sha256"):
            validate_public_bindings(self.amendment, changed, self.matrix)


if __name__ == "__main__":
    unittest.main()
