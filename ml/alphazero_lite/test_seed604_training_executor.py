"""Dependency-light synthetic guard tests for Seed 604 training readiness."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import seed604_training_executor as executor


def _freeze(
    root: Path, sources: dict[str, str] | None = None
) -> tuple[bytes, dict[str, object]]:
    path = root / executor.FREEZE
    path.parent.mkdir(parents=True, exist_ok=True)
    if sources is None:
        sources = {}
        for relative in executor.FREEZE_SOURCE_PATHS:
            dependency = root / relative
            dependency.parent.mkdir(parents=True, exist_ok=True)
            dependency.write_text("synthetic source")
            sources[relative] = hashlib.sha256(dependency.read_bytes()).hexdigest()
    expected: dict[str, object] = {
        "bindings": {"seed604_registration_sha256": "1" * 64},
        "runtime": dict(executor.RUNTIME),
        "source_sha256": sources,
        "output_contract": {
            "directory": str(executor.OUTPUT),
            "create_only": True,
            "updates": executor.UPDATES,
            "result_schema": "seed604-b2048-training-result-v1",
        },
        "b512_reproduction_gate": {
            "stop_after_update": 512,
            "compare_every_model_tensor": True,
            "exact_names_shapes_dtypes_values": True,
            "require_complete_adam_state": True,
            "require_adam_step": 512,
            "continue_same_optimizer": True,
            "resume_at_update": 513,
            "fail_closed": True,
        },
    }
    value = {
        "schema": "seed604-training-execution-freeze-v1",
        "status": "frozen-awaiting-owner-authorization",
        "execution_authorized": False,
        **expected,
    }
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return raw, expected


def _verify_with_contract(root: Path, expected: dict[str, object]) -> dict[str, object]:
    with patch.object(executor, "expected_freeze_contract", return_value=expected):
        return executor.verify_authorization(root)


def _authorization(root: Path, freeze_raw: bytes, **updates: object) -> None:
    path = root / executor.AUTHORIZATION
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "seed604-training-owner-authorization-v1",
        "status": "authorized-for-training",
        "execution_freeze_sha256": hashlib.sha256(freeze_raw).hexdigest(),
        "authorization": {
            "type": "explicit-project-owner-authorization",
            "authorized_by": "synthetic owner fixture",
            "training_may_start": True,
        },
    }
    record.update(updates)
    path.write_text(json.dumps(record))


class AuthorizationAndSourceGuards(unittest.TestCase):
    def test_missing_authorization_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, expected = _freeze(root)
            with self.assertRaises((FileNotFoundError, ValueError)):
                _verify_with_contract(root, expected)

    def test_forged_or_stale_authorization_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze, expected = _freeze(root)
            _authorization(root, freeze, execution_freeze_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "training_not_authorized"):
                _verify_with_contract(root, expected)

    def test_modified_freeze_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, expected = _freeze(root)
            _authorization(root, raw)
            changed = json.loads(raw)
            changed["execution_authorized"] = True
            (root / executor.FREEZE).write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "execution_freeze_invalid"):
                _verify_with_contract(root, expected)

    def test_source_hash_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = sorted(executor.FREEZE_SOURCE_PATHS)[0]
            dependency = root / relative
            dependency.parent.mkdir(parents=True, exist_ok=True)
            dependency.write_text("original")
            hashes = {}
            for source in executor.FREEZE_SOURCE_PATHS:
                source_path = root / source
                source_path.parent.mkdir(parents=True, exist_ok=True)
                if source_path != dependency:
                    source_path.write_text("fixture")
                hashes[source] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            raw, expected = _freeze(root, hashes)
            _authorization(root, raw)
            dependency.write_text("changed")
            with self.assertRaisesRegex(ValueError, "execution_source_hash_mismatch"):
                _verify_with_contract(root, expected)

    def test_missing_required_freeze_binding_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, expected = _freeze(root)
            _authorization(root, raw)
            changed = json.loads(raw)
            changed["bindings"] = {}
            (root / executor.FREEZE).write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "binding_mismatch:bindings"):
                _verify_with_contract(root, expected)

    def test_changed_cohort_or_checkpoint_identity_rejected(self) -> None:
        for binding in ("cohort_manifest_sha256", "b512_checkpoint_sha256"):
            with (
                self.subTest(binding=binding),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                raw, expected = _freeze(root)
                _authorization(root, raw)
                changed = json.loads(raw)
                changed["bindings"][binding] = "f" * 64
                (root / executor.FREEZE).write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "binding_mismatch:bindings"):
                    _verify_with_contract(root, expected)

    def test_freeze_cannot_define_a_different_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, expected = _freeze(root)
            _authorization(root, raw)
            changed = json.loads(raw)
            changed["runtime"]["numpy"] = "2.5.3"
            (root / executor.FREEZE).write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "binding_mismatch:runtime"):
                _verify_with_contract(root, expected)

    def test_changed_output_and_update_contract_rejected(self) -> None:
        for key, value in (("directory", "elsewhere"), ("updates", 512)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                raw, expected = _freeze(root)
                _authorization(root, raw)
                changed = json.loads(raw)
                changed["output_contract"][key] = value
                (root / executor.FREEZE).write_text(json.dumps(changed))
                with self.assertRaisesRegex(
                    ValueError, "binding_mismatch:output_contract"
                ):
                    _verify_with_contract(root, expected)

    def test_unknown_status_cannot_authorize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, expected = _freeze(root)
            _authorization(root, raw, status="authorized")
            with self.assertRaisesRegex(ValueError, "training_not_authorized"):
                _verify_with_contract(root, expected)


class OutputAndRuntimeGuards(unittest.TestCase):
    def test_output_overwrite_rejected_before_torch_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / executor.OUTPUT
            output.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "output_directory_exists"):
                executor.train_b2048(output, root=root)

    def test_wrong_output_path_rejected_before_torch_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "noncanonical_output_directory"):
                executor.train_b2048(
                    Path(directory) / "elsewhere", root=Path(directory)
                )

    def test_runtime_mismatch_helper_rejects_different_runtime(self) -> None:
        class FakeTorch:
            __version__ = "wrong"

            @staticmethod
            def get_num_threads() -> int:
                return 1

            @staticmethod
            def are_deterministic_algorithms_enabled() -> bool:
                return True

            @staticmethod
            def set_num_threads(_count: int) -> None:
                return None

            @staticmethod
            def use_deterministic_algorithms(_enabled: bool) -> None:
                return None

        with self.assertRaisesRegex(ValueError, "runtime_mismatch"):
            executor.check_runtime(executor.RUNTIME, FakeTorch)


try:
    import torch as _torch  # noqa: F401
except ImportError:
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True


@unittest.skipUnless(
    TORCH_AVAILABLE, "requires PyTorch; exact runtime certification remains separate"
)
class PyTorchSyntheticGateTests(unittest.TestCase):
    def test_exact_tensor_comparison_gate_is_strict(self) -> None:
        import torch

        expected = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([[3.0]])}
        passed = executor.compare_tensor_mappings(
            {key: value.clone() for key, value in expected.items()}, expected, torch
        )
        self.assertTrue(passed["passed"])
        changed = {key: value.clone() for key, value in expected.items()}
        changed["b"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "b512_exact_tensor_mismatch"):
            executor.compare_tensor_mappings(changed, expected, torch)

    def test_tensor_name_shape_and_dtype_mutations_are_rejected(self) -> None:
        import torch

        expected = {"weight": torch.ones((2, 3), dtype=torch.float32)}
        variants = (
            {"altered": expected["weight"]},
            {"weight": torch.ones((3, 2), dtype=torch.float32)},
            {"weight": torch.ones((2, 3), dtype=torch.float64)},
        )
        for actual in variants:
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                executor.compare_tensor_mappings(actual, expected, torch)

    @staticmethod
    def _optimizer_after_steps(torch, count: int):
        model = torch.nn.Linear(2, 1, bias=True)
        named = dict(model.named_parameters())
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
        )
        for _ in range(count):
            optimizer.zero_grad(set_to_none=True)
            loss = model(torch.tensor([[1.0, -2.0]])).square().mean()
            loss.backward()
            optimizer.step()
        return optimizer, named

    def test_adam_state_is_complete_and_counters_equal_512(self) -> None:
        import torch

        optimizer, named = self._optimizer_after_steps(torch, 512)
        evidence = executor.optimizer_state_evidence(
            optimizer, named, torch, expected_step=512
        )
        self.assertEqual(evidence["parameter_count"], len(named))
        self.assertEqual({item["step"] for item in evidence["step_counters"]}, {512})
        self.assertEqual(len(evidence["state_sha256"]), 64)

    def test_optimizer_continues_same_live_state_across_512_to_513(self) -> None:
        import torch

        optimizer, named = self._optimizer_after_steps(torch, 512)
        before = executor.optimizer_state_evidence(
            optimizer, named, torch, expected_step=512
        )
        identity = id(optimizer)
        continuous = executor.verify_optimizer_continuity(
            optimizer,
            identity,
            named,
            torch,
            expected_state_sha256=before["state_sha256"],
            expected_step=512,
        )
        self.assertEqual(continuous["state_sha256"], before["state_sha256"])
        optimizer.zero_grad(set_to_none=True)
        loss = next(iter(named.values())).square().sum()
        loss.backward()
        optimizer.step()
        after = executor.optimizer_state_evidence(
            optimizer, named, torch, expected_step=513
        )
        self.assertEqual(id(optimizer), identity)
        self.assertEqual({item["step"] for item in after["step_counters"]}, {513})

    def test_optimizer_reset_or_altered_state_is_rejected(self) -> None:
        import torch

        optimizer, named = self._optimizer_after_steps(torch, 512)
        evidence = executor.optimizer_state_evidence(
            optimizer, named, torch, expected_step=512
        )
        with self.assertRaisesRegex(ValueError, "optimizer_object_reset"):
            executor.verify_optimizer_continuity(
                torch.optim.Adam(named.values()),
                id(optimizer),
                named,
                torch,
                expected_state_sha256=evidence["state_sha256"],
                expected_step=512,
            )
        first = next(iter(named.values()))
        optimizer.state[first]["exp_avg"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "optimizer_state_reset_or_altered"):
            executor.verify_optimizer_continuity(
                optimizer,
                id(optimizer),
                named,
                torch,
                expected_state_sha256=evidence["state_sha256"],
                expected_step=512,
            )


if __name__ == "__main__":
    unittest.main()
