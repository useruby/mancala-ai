"""Negative tests for Seed 604 freeze and supplemental readiness integrity."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite import verify_seed604_training_freeze as verifier


def fixture_freeze() -> dict[str, object]:
    return {
        "schema": "seed604-training-execution-freeze-v1",
        "status": "frozen-awaiting-owner-authorization",
        "execution_authorized": False,
        "bindings": {
            "parent_checkpoint_sha256": "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
            "b512_checkpoint_sha256": "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
            "membership_sha256": "95ecc180e28945702e08815de7a5210e1bce036afdaf76d194bbd774589a4516",
            "b512_schedule_sha256": "62edb14805d20af1cc7988d73aba95dcdf9e67842b6ea1d61a12c98a87c28599",
            "b2048_schedule_sha256": "397c9c84388fad2cfe6a38d549f1b2a9b52909907e6e19c1bcb81a418551063b",
            "cohort_manifest_sha256": "445052b818b36f84953f73f280b20f7421c441fac6a74a0e2bb309cf907fb410",
        },
        "runtime": {"python": "3.14.7", "numpy": "2.5.2", "torch": "2.13.0+cu130"},
        "source_sha256": {"source.py": hashlib.sha256(b"source").hexdigest()},
        "output_contract": {
            "directory": "docs/data/seed604-replay-optimization/training-run-B2048",
            "create_only": True,
            "updates": 2048,
            "result_schema": "seed604-b2048-training-result-v1",
        },
        "b512_reproduction_gate": copy.deepcopy(verifier.GATE),
    }


class FreezeDocumentNegativeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = fixture_freeze()

    def reject_change(self, section: str, key: str, value: object) -> None:
        changed = copy.deepcopy(self.expected)
        if section == "top":
            changed[key] = value
        else:
            changed[section][key] = value  # type: ignore[index]
        with self.assertRaises(ValueError):
            verifier._validate_freeze(json.dumps(changed).encode(), self.expected)

    def test_modified_execution_freeze_bytes_rejected(self) -> None:
        raw = json.dumps(self.expected).encode() + b" "
        with self.assertRaisesRegex(ValueError, "serialization_not_canonical"):
            verifier._validate_freeze(raw, self.expected)
        altered = json.loads(raw)
        altered["execution_authorized"] = True
        with self.assertRaises(ValueError):
            verifier._validate_freeze(json.dumps(altered).encode(), self.expected)

    def test_additional_incompatible_field_rejected(self) -> None:
        self.reject_change("top", "unauthorized_extra", True)

    def test_wrong_schema_or_status_rejected(self) -> None:
        self.reject_change("top", "schema", "wrong-schema")
        self.reject_change("top", "status", "authorized")

    def test_incorrect_parent_or_b512_identity_rejected(self) -> None:
        self.reject_change("bindings", "parent_checkpoint_sha256", "0" * 64)
        self.reject_change("bindings", "b512_checkpoint_sha256", "0" * 64)

    def test_modified_membership_or_schedule_rejected(self) -> None:
        self.reject_change("bindings", "membership_sha256", "0" * 64)
        self.reject_change("bindings", "b512_schedule_sha256", "0" * 64)
        self.reject_change("bindings", "b2048_schedule_sha256", "0" * 64)

    def test_changed_runtime_or_cohort_rejected(self) -> None:
        self.reject_change("runtime", "torch", "2.13.0+cpu")
        self.reject_change("bindings", "cohort_manifest_sha256", "0" * 64)

    def test_source_code_hash_mismatch_rejected(self) -> None:
        self.reject_change("source_sha256", "source.py", "0" * 64)

    def test_incorrect_output_directory_rejected(self) -> None:
        self.reject_change("output_contract", "directory", "/tmp/wrong")

    def test_weakened_B512_gate_rejected(self) -> None:
        self.reject_change("b512_reproduction_gate", "continue_same_optimizer", False)
        self.reject_change(
            "b512_reproduction_gate", "compare_every_model_tensor", False
        )


class ReadinessNegativeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = {
            "schema": "seed604-execution-readiness-evidence-v1",
            "execution_freeze_sha256": "a" * 64,
            "runtime_certification_v3": {
                "artifact_sha256": {"runtime-identity.json": "b" * 64},
                "verification_result": {"passed": True},
            },
            "source_sha256": {"certified_source.py": "c" * 64},
            "execution_freeze_validation": {"passed": True},
            "owner_authorization_present": False,
        }

    def test_modified_runtime_certification_artifact_rejected(self) -> None:
        changed = copy.deepcopy(self.expected)
        changed["runtime_certification_v3"]["artifact_sha256"][
            "runtime-identity.json"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "execution_readiness_binding_mismatch"):
            verifier._validate_readiness(changed, self.expected)

    def test_altered_certification_to_freeze_binding_rejected(self) -> None:
        changed = copy.deepcopy(self.expected)
        changed["execution_freeze_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            verifier._validate_readiness(changed, self.expected)

    def test_missing_or_unauthorized_readiness_rejected(self) -> None:
        for key in ("source_sha256", "runtime_certification_v3"):
            changed = copy.deepcopy(self.expected)
            del changed[key]
            with self.assertRaises(ValueError):
                verifier._validate_readiness(changed, self.expected)
        changed = copy.deepcopy(self.expected)
        changed["owner_authorization_present"] = True
        with self.assertRaises(ValueError):
            verifier._validate_readiness(changed, self.expected)

    def test_existing_publication_destination_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / verifier.FREEZE_PATH).parent.mkdir(parents=True)
            (root / verifier.FREEZE_PATH).write_bytes(b"prior")
            with self.assertRaisesRegex(FileExistsError, "destination_exists"):
                verifier.publish(root)

    def test_authorization_artifact_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authorization = (
                root
                / "docs/data/seed604-replay-optimization/training-authorization.json"
            )
            authorization.parent.mkdir(parents=True)
            authorization.write_text("{}")
            with self.assertRaisesRegex(
                ValueError, "training_authorization_artifact_exists"
            ):
                verifier._require_preparation_only(root)

    def test_production_output_directory_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "docs/data/seed604-replay-optimization/training-run-B2048"
            output.mkdir(parents=True)
            with self.assertRaisesRegex(
                ValueError, "production_training_output_exists"
            ):
                verifier._require_preparation_only(root)


if __name__ == "__main__":
    unittest.main()
