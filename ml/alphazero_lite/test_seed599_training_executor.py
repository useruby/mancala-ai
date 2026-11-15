"""Synthetic and mutation checks for Seed 599 training readiness."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from ml.alphazero_lite import seed599_training_executor as executor
from ml.alphazero_lite import verify_seed599_training_result as verifier
from ml.alphazero_lite import train
from ml.alphazero_lite.seed599_replay_coverage import ROOT, canonical_bytes


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


def _dependency_fixture(root: Path) -> dict[str, str]:
    hashes = {}
    for name in sorted(
        executor.TRAINING_DEPENDENCIES
        | {
            "ml/alphazero_lite/seed599_training_executor.py",
            "ml/alphazero_lite/verify_seed599_training_result.py",
            "ml/alphazero_lite/seed599_replay_coverage.py",
        }
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic source:{name}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _successful_result_fixture(root: Path, output: Path) -> None:
    data = root / executor.DATA
    data.mkdir(parents=True)
    source_data = ROOT / executor.DATA
    rows = [{"canonical_state": {}, "target": [1.0, 0, 0, 0, 0, 0]} for _ in range(32)]
    schedule = [list(range(32)) for _ in range(512)]
    membership_sha = hashlib.sha256(canonical_bytes(rows)).hexdigest()
    schedule_sha = hashlib.sha256(canonical_bytes(schedule)).hexdigest()
    config = {
        "schema": "seed599_training_configuration_v1",
        "status": "frozen-pretraining-not-authorized",
        "arm": "A",
        "states": 32,
        "membership_sha256": membership_sha,
        "batch_schedule_sha256": schedule_sha,
    }
    config_path = data / "training-A.json"
    config_path.write_text(json.dumps(config))
    (data / "control-membership.json").write_text(json.dumps(rows))
    (data / "batch-schedule-A.json").write_text(json.dumps(schedule))
    parent = root / executor.PARENT
    parent.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / executor.PARENT, parent)
    runtime = json.loads((source_data / "training-execution-freeze.json").read_bytes())[
        "runtime"
    ]
    freeze_path = data / "training-execution-freeze.json"
    freeze_path.write_text(
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
                "runtime": runtime,
            }
        )
    )
    model = train.PolicyValueNet(
        hidden_sizes=(96, 3), model_type="residual_v3", input_size=27
    ).cpu()
    if train.load_checkpoint_into_model(model, parent):
        raise AssertionError("synthetic_parent_checkpoint_mismatch")
    output.mkdir()
    checkpoint = output / "seed599-arm-A-update-512.npz"
    import numpy as np

    np.savez(checkpoint, **train.checkpoint_from_model(model))
    trace = output / "training-trace.jsonl"
    with trace.open("w") as stream:
        for update, indexes in enumerate(schedule, 1):
            stream.write(
                json.dumps(
                    {
                        "update": update,
                        "indexes": indexes,
                        "loss": 1.0,
                        "gradient_norm": 0.5,
                        "seed": 599,
                        "runtime": runtime,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    receipt = {
        "schema": "seed599-training-result-v1",
        "arm": "A",
        "states": 32,
        "updates": 512,
        "execution_freeze_sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
        "membership_sha256": membership_sha,
        "batch_schedule_sha256": schedule_sha,
        "parent_checkpoint_sha256": executor.PARENT_SHA256,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "trace": trace.name,
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "frozen_parameter_names": sorted(
            name
            for name, _ in model.named_parameters()
            if name.startswith(executor.FROZEN)
        ),
        "frozen_parameters_unchanged": True,
    }
    (output / "training-receipt.json").write_text(json.dumps(receipt))


class Seed599TrainingGuardTests(unittest.TestCase):
    def test_changed_training_dependency_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hashes = _dependency_fixture(root)
            freeze_path = root / executor.FREEZE
            freeze_path.parent.mkdir(parents=True, exist_ok=True)
            freeze_path.write_text(
                json.dumps(
                    {
                        "schema": "seed599-training-execution-freeze-v1",
                        "status": "frozen-training-not-authorized",
                        "source_sha256": hashes,
                    }
                )
            )
            dependency = root / "ml/alphazero_lite/train.py"
            dependency.write_text("mutated dependency")
            with self.assertRaisesRegex(ValueError, "execution_source_hash_mismatch"):
                executor.load_freeze(root)

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
            output = root / executor.DATA / "training-run-A"
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

    def test_alternate_output_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            alternate = Path(temporary) / "alternate"
            with self.assertRaisesRegex(ValueError, "noncanonical_output_directory"):
                executor.train_arm("A", alternate)
            self.assertFalse(alternate.exists())

    def test_existing_canonical_output_blocks_arm_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / executor.DATA / "training-run-B"
            output.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "output_directory_exists"):
                executor.train_arm("B", output, root=root)


class Seed599ResultMutationTests(unittest.TestCase):
    def test_incomplete_output_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "incomplete_training_output"):
                verifier.verify_result("A", Path(temporary))

    def test_invalid_arm_rejected_before_file_reads(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid_training_arm"):
            verifier.verify_result("C", Path("does-not-exist"))

    def test_successful_result_reaches_tensor_load_and_frozen_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run-A"
            _successful_result_fixture(root, output)
            with patch.object(
                verifier.train,
                "load_checkpoint_into_model",
                wraps=train.load_checkpoint_into_model,
            ) as load_checkpoint:
                with patch.object(
                    verifier.torch, "equal", wraps=torch.equal
                ) as tensor_equal:
                    result = verifier.verify_result("A", output, root=root)
            self.assertTrue(result["passed"])
            self.assertEqual(load_checkpoint.call_count, 2)
            self.assertEqual(
                tensor_equal.call_count, len(result["parameter_groups"]["frozen"])
            )


if __name__ == "__main__":
    unittest.main()
