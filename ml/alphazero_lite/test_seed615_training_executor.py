"""Synthetic guard tests for the Seed 615 training freeze and executor."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ml.alphazero_lite import seed615_training_executor as executor


class RegisteredInputGuards(unittest.TestCase):
    def test_wrong_parent_checkpoint_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (executor.PARENT, executor.B512):
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                (root / relative).write_bytes(b"wrong")
            with self.assertRaisesRegex(
                ValueError, "parent_checkpoint_identity_mismatch"
            ):
                executor.validate_checkpoints(root)

    def test_wrong_membership_hash_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / executor.MEMBERSHIP
            path.parent.mkdir(parents=True)
            path.write_text("[]\n")
            with self.assertRaisesRegex(ValueError, "membership_identity_mismatch"):
                executor.membership_rows(root)

    def test_changed_teacher_target_rejected(self) -> None:
        phases = [("above_32", 512), ("17_to_32", 1024), ("at_most_16", 512)]
        rows = []
        index = 0
        for phase, count in phases:
            for _ in range(count):
                rows.append(
                    {
                        "phase": phase,
                        "canonical_identity": f"c{index:04d}",
                        "float32_identity": f"f{index:04d}",
                        "target": [0.5, 0.5],
                    }
                )
                index += 1
        rows.sort(key=lambda row: (row["phase"], row["canonical_identity"]))
        rows[0]["target"] = [0.4, 0.6]
        raw = b"\n".join(json.dumps(row).encode() for row in rows) + b"\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / executor.MEMBERSHIP
            path.parent.mkdir(parents=True)
            path.write_bytes(raw)
            with (
                patch.object(
                    executor,
                    "MEMBERSHIP_SHA256",
                    hashlib.sha256(executor.canonical_bytes(rows)).hexdigest(),
                ),
                patch(
                    "ml.alphazero_lite.seed599_replay_coverage.validate_target_record",
                    side_effect=lambda row: (
                        (_ for _ in ()).throw(ValueError("teacher target mismatch"))
                        if row["target"] != [0.5, 0.5]
                        else None
                    ),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "teacher target mismatch"):
                    executor.membership_rows(Path(directory))

    def test_wrong_batch_schedule_digest_rejected(self) -> None:
        schedule = [[0] * 32 for _ in range(512)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / executor.SCHEDULE
            path.parent.mkdir(parents=True)
            path.write_bytes(executor.canonical_bytes(schedule))
            with patch.object(executor, "SCHEDULE_SHA256", "0" * 64):
                with self.assertRaisesRegex(
                    ValueError, "batch_schedule_identity_mismatch"
                ):
                    executor.batch_schedule(Path(directory))

    def test_duplicate_minibatch_ordering_rejected(self) -> None:
        schedule = [[0] * 32 for _ in range(512)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / executor.SCHEDULE
            path.parent.mkdir(parents=True)
            path.write_bytes(executor.canonical_bytes(schedule))
            with patch.object(
                executor,
                "SCHEDULE_SHA256",
                hashlib.sha256(executor.canonical_bytes(schedule)).hexdigest(),
            ):
                with self.assertRaisesRegex(ValueError, "invalid_batch_schedule"):
                    executor.batch_schedule(Path(directory))

    def test_incorrect_update_count_rejected(self) -> None:
        schedule = [[index for index in range(32)] for _ in range(511)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / executor.SCHEDULE
            path.parent.mkdir(parents=True)
            path.write_bytes(executor.canonical_bytes(schedule))
            with patch.object(
                executor,
                "SCHEDULE_SHA256",
                hashlib.sha256(executor.canonical_bytes(schedule)).hexdigest(),
            ):
                with self.assertRaisesRegex(
                    ValueError, "batch_schedule_identity_mismatch"
                ):
                    executor.batch_schedule(Path(directory))

    def test_changed_optimizer_settings_rejected_by_frozen_contract(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.Adam([parameter], lr=2e-5)
        with self.assertRaisesRegex(
            ValueError, "optimizer_hyperparameter_or_parameter_mismatch"
        ):
            executor.validate_optimizer(optimizer, [parameter], torch)

    def test_unexpected_trainable_parameter_prefix_rejected(self) -> None:
        class SyntheticModel:
            def named_parameters(self):
                yield "unexpected.weight", torch.nn.Parameter(torch.ones(1))

        with self.assertRaisesRegex(ValueError, "unregistered_parameter_group"):
            executor.classify_parameters(SyntheticModel())


class IntegrityAndAuthorizationGuards(unittest.TestCase):
    def test_changed_frozen_parameter_is_detected(self) -> None:
        before = torch.tensor([1.0])
        after = torch.nn.Parameter(torch.tensor([2.0]))
        with self.assertRaisesRegex(ValueError, "frozen_parameter_changed"):
            executor.validate_frozen_parameters(
                {"value_head.weight": after}, {"value_head.weight": before}, torch
            )

    def test_incomplete_adam_state_rejected(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.Adam([parameter])
        with self.assertRaisesRegex(ValueError, "adam_state_parameter_set_mismatch"):
            executor._verify_adam(optimizer, {"input_layer.weight": parameter}, torch)

    def test_nonfinite_gradient_and_parameter_are_detectable(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([float("nan")]))
        self.assertFalse(torch.isfinite(parameter).all())
        model = torch.nn.Module()
        model.register_parameter("weight", parameter)
        optimizer = torch.optim.Adam([parameter])
        with self.assertRaisesRegex(ValueError, "nonfinite_model_or_optimizer_state:1"):
            executor.validate_finite_state(model, optimizer, torch, 1)
        parameter.grad = torch.tensor([float("nan")])
        with self.assertRaises(RuntimeError):
            torch.nn.utils.clip_grad_norm_([parameter], 1.0, error_if_nonfinite=True)

    def test_runtime_identity_rejected(self) -> None:
        class WrongRuntime:
            @staticmethod
            def set_num_threads(_count: int) -> None:
                pass

            @staticmethod
            def use_deterministic_algorithms(_enabled: bool) -> None:
                pass

            @staticmethod
            def get_num_threads() -> int:
                return 2

            @staticmethod
            def are_deterministic_algorithms_enabled() -> bool:
                return False

            __version__ = "wrong"

        with self.assertRaisesRegex(ValueError, "runtime_identity_mismatch"):
            executor.check_runtime(WrongRuntime)

    def test_missing_authorization_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(executor, "load_freeze", return_value=({}, b"freeze")):
                with self.assertRaisesRegex(
                    ValueError, "training_authorization_missing"
                ):
                    executor.verify_authorization(Path(directory))

    def test_wrong_authorization_freeze_binding_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / executor.AUTHORIZATION
            path.parent.mkdir(parents=True)
            record = {
                "schema": "seed615-training-owner-authorization-v1",
                "status": "authorized-for-training",
                "execution_freeze_sha256": "0" * 64,
                "authorization": {
                    "type": "explicit-project-owner-authorization",
                    "authorized_by": "synthetic",
                    "training_may_start": True,
                },
            }
            path.write_text(json.dumps(record))
            with patch.object(executor, "load_freeze", return_value=({}, b"freeze")):
                with self.assertRaisesRegex(
                    ValueError, "training_authorization_or_freeze_binding_invalid"
                ):
                    executor.verify_authorization(root)

    def test_existing_output_and_repeated_attempt_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / executor.OUTPUT
            output.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "output_directory_exists"):
                executor.train_s615(output, root=root)
            with self.assertRaisesRegex(ValueError, "output_directory_exists"):
                executor.train_s615(output, root=root)

    def test_failure_recovery_is_not_automatic(self) -> None:
        self.assertIn("no automatic retry", executor.EXECUTION_POLICY)

    def test_b512_tensor_equality_is_diagnostic_only(self) -> None:
        final = {"w_input": np.array([1.0], dtype=np.float32)}
        parent = {"w_input": np.array([0.0], dtype=np.float32)}
        b512 = {"w_input": np.array([1.0], dtype=np.float32)}
        diagnostics = executor.checkpoint_diagnostics(
            final, parent, b512, {"input_layer.weight": object()}
        )
        self.assertEqual(diagnostics["b512_equal_tensor_names"], ["w_input"])
        self.assertTrue(diagnostics["b512_exact_tensor_equality"])
        self.assertFalse(diagnostics["b512_equality_used_for_selection"])
        self.assertTrue(
            diagnostics["trainable_parameter_movement_diagnostic"]["input_layer.weight"]
        )

    def test_modified_result_artifact_rejected_by_digest(self) -> None:
        from ml.alphazero_lite import verify_seed615_training_result as verifier

        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "trace.jsonl"
            artifact.write_text("original")
            expected = verifier.digest(artifact)
            artifact.write_text("modified")
            with self.assertRaisesRegex(ValueError, "training_artifact_hash_mismatch"):
                verifier.require_digest(expected, artifact, "trace")


if __name__ == "__main__":
    unittest.main()
