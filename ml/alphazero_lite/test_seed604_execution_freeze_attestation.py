"""Focused rejection tests for the append-only Seed 604 load attestation."""

from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite import seed604_training_executor as executor
from ml.alphazero_lite import verify_seed604_execution_freeze_attestation as attestation


def fixture_record() -> dict[str, object]:
    return {
        "schema": "seed604-execution-freeze-validation-attestation-v1",
        "execution_freeze_sha256": hashlib.sha256(b"freeze").hexdigest(),
        "execution_readiness_evidence_sha256": hashlib.sha256(b"readiness").hexdigest(),
        "source_commit": attestation.SOURCE_COMMIT,
        "certified_executor_source_sha256": attestation.EXECUTOR_SHA256,
        "direct_load_freeze_validation": {
            "passed": True,
            "schema": "seed604-training-execution-freeze-v1",
            "status": "frozen-awaiting-owner-authorization",
        },
        "independent_freeze_verifier": {"freeze_integrity_verified": True},
        "runtime_certification_independently_verified": {"passed": True},
        "training_authorization_artifact_absent": True,
        "production_B2048_output_directory_absent": True,
        "b512_reproduction_occurred": False,
    }


class AttestationNegativeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = fixture_record()

    def test_original_freeze_change_rejected(self) -> None:
        changed = copy.deepcopy(self.expected)
        changed["execution_freeze_sha256"] = hashlib.sha256(
            b"changed freeze"
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "validation_attestation_mismatch"):
            attestation.validate_record(changed, self.expected)

    def test_original_readiness_change_rejected(self) -> None:
        changed = copy.deepcopy(self.expected)
        changed["execution_readiness_evidence_sha256"] = hashlib.sha256(
            b"changed readiness"
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "validation_attestation_mismatch"):
            attestation.validate_record(changed, self.expected)

    def test_executor_source_hash_mismatch_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "certified_executor_source_hash_mismatch"
        ):
            attestation.validate_executor_source(b"modified executor")

    def test_attested_pass_inconsistent_with_validation_rejected(self) -> None:
        changed = copy.deepcopy(self.expected)
        changed["direct_load_freeze_validation"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "validation_attestation_mismatch"):
            attestation.validate_record(changed, self.expected)

    def test_unexpected_authorization_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authorization = root / executor.AUTHORIZATION
            authorization.parent.mkdir(parents=True)
            authorization.write_bytes(b"{}")
            with self.assertRaisesRegex(
                ValueError, "training_authorization_artifact_exists"
            ):
                attestation._require_no_execution(root)

    def test_unexpected_production_output_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / executor.OUTPUT).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "production_B2048_output_exists"):
                attestation._require_no_execution(root)

    def test_create_only_destination_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / attestation.ATTESTATION
            target.parent.mkdir(parents=True)
            target.write_bytes(b"immutable")
            with self.assertRaisesRegex(
                FileExistsError, "validation_attestation_destination_exists"
            ):
                attestation.publish(root)


if __name__ == "__main__":
    unittest.main()
