"""Synthetic and mutation checks for Seed 599 training readiness."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite import seed599_training_executor as executor
from ml.alphazero_lite import verify_seed599_training_result as verifier
from ml.alphazero_lite.seed599_replay_coverage import canonical_bytes


def _registered_fixture(root: Path) -> Path:
    data = root / "docs/data/seed599-replay-coverage"
    data.mkdir(parents=True)
    membership = [
        {"canonical_state": {}, "target": [1.0, 0, 0, 0, 0, 0]} for _ in range(32)
    ]
    schedule = [list(range(32)) for _ in range(512)]
    config = {
        "schema": "seed599_training_configuration_v1",
        "status": "frozen-pretraining-not-authorized",
        "arm": "A",
        "states": 32,
        "membership_sha256": hashlib.sha256(canonical_bytes(membership)).hexdigest(),
        "batch_schedule_sha256": hashlib.sha256(canonical_bytes(schedule)).hexdigest(),
    }
    config_path = data / "training-A.json"
    config_path.write_text(json.dumps(config))
    (data / "control-membership.json").write_text(json.dumps(membership))
    (data / "batch-schedule-A.json").write_text(json.dumps(schedule))
    (data / "training-execution-freeze.json").write_text(
        json.dumps(
            {
                "schema": "seed599-training-execution-freeze-v1",
                "status": "frozen-training-not-authorized",
                "arms": {
                    "A": {
                        "configuration_sha256": hashlib.sha256(
                            config_path.read_bytes()
                        ).hexdigest()
                    }
                },
            }
        )
    )
    return data


class Seed599TrainingGuardTests(unittest.TestCase):
    def test_membership_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = _registered_fixture(root)
            rows = json.loads((data / "control-membership.json").read_bytes())
            rows[0]["target"][0] = 0.5
            (data / "control-membership.json").write_text(json.dumps(rows))
            with self.assertRaisesRegex(
                ValueError, "training_membership_identity_mismatch"
            ):
                executor._load_registered_arm("A", root)

    def test_schedule_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = _registered_fixture(root)
            batches = json.loads((data / "batch-schedule-A.json").read_bytes())
            batches[0][0] = 32
            (data / "batch-schedule-A.json").write_text(json.dumps(batches))
            with self.assertRaisesRegex(
                ValueError, "training_batch_schedule_identity_mismatch"
            ):
                executor._load_registered_arm("A", root)

    def test_missing_authorization_rejects_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "docs/data/seed599-replay-coverage"
            data.mkdir(parents=True)
            freeze = {
                "schema": "seed599-training-execution-freeze-v1",
                "status": "frozen-training-not-authorized",
                "arms": {},
            }
            (data / "training-execution-freeze.json").write_text(json.dumps(freeze))
            output = root / "run"
            with self.assertRaises(FileNotFoundError):
                executor.train_arm("A", output, root=root)
            self.assertFalse(output.exists())

    def test_authorization_must_bind_exact_freeze_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "docs/data/seed599-replay-coverage"
            data.mkdir(parents=True)
            freeze_path = data / "training-execution-freeze.json"
            freeze_path.write_text(
                json.dumps(
                    {
                        "schema": "seed599-training-execution-freeze-v1",
                        "status": "frozen-training-not-authorized",
                    }
                )
            )
            (data / "training-authorization.json").write_text(
                json.dumps(
                    {
                        "schema": "seed599-training-owner-authorization-v1",
                        "status": "authorized-for-training",
                        "authorization": {
                            "type": "explicit-project-owner-authorization",
                            "training_may_start": True,
                            "authorized_by": "synthetic owner fixture",
                        },
                        "execution_freeze_sha256": "0" * 64,
                    }
                )
            )
            with self.assertRaisesRegex(
                ValueError, "authorization_freeze_binding_mismatch"
            ):
                executor.verify_authorization(root)

    def test_exact_freeze_binding_is_accepted_only_with_owner_assertion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "docs/data/seed599-replay-coverage"
            data.mkdir(parents=True)
            path = data / "training-execution-freeze.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "seed599-training-execution-freeze-v1",
                        "status": "frozen-training-not-authorized",
                    }
                )
            )
            binding = hashlib.sha256(path.read_bytes()).hexdigest()
            (data / "training-authorization.json").write_text(
                json.dumps(
                    {
                        "schema": "seed599-training-owner-authorization-v1",
                        "status": "authorized-for-training",
                        "authorization": {
                            "type": "explicit-project-owner-authorization",
                            "training_may_start": True,
                            "authorized_by": "synthetic owner fixture",
                        },
                        "execution_freeze_sha256": binding,
                    }
                )
            )
            self.assertEqual(
                executor.verify_authorization(root)["execution_freeze_sha256"], binding
            )

    def test_freeze_mutation_invalidates_authorization_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "docs/data/seed599-replay-coverage"
            data.mkdir(parents=True)
            path = data / "training-execution-freeze.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "seed599-training-execution-freeze-v1",
                        "status": "frozen-training-not-authorized",
                    }
                )
            )
            binding = hashlib.sha256(path.read_bytes()).hexdigest()
            path.write_text(path.read_text() + " ")
            (data / "training-authorization.json").write_text(
                json.dumps(
                    {
                        "schema": "seed599-training-owner-authorization-v1",
                        "status": "authorized-for-training",
                        "authorization": {
                            "type": "explicit-project-owner-authorization",
                            "training_may_start": True,
                            "authorized_by": "project owner",
                        },
                        "execution_freeze_sha256": binding,
                    }
                )
            )
            with self.assertRaisesRegex(
                ValueError, "authorization_freeze_binding_mismatch"
            ):
                executor.verify_authorization(root)


class Seed599ResultMutationTests(unittest.TestCase):
    def test_incomplete_output_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "incomplete_training_output"):
                verifier.verify_result("A", Path(temporary))

    def test_invalid_arm_rejected_before_file_reads(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid_training_arm"):
            verifier.verify_result("C", Path("does-not-exist"))


if __name__ == "__main__":
    unittest.main()
