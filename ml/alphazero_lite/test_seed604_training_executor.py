"""Dependency-light synthetic guard tests for Seed 604 training readiness."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite import seed604_training_executor as executor


def _freeze(root: Path, sources: dict[str, str] | None = None) -> bytes:
    path = root / executor.FREEZE
    path.parent.mkdir(parents=True, exist_ok=True)
    if sources is None:
        sources = {}
        for relative in executor.FREEZE_SOURCE_PATHS:
            dependency = root / relative
            dependency.parent.mkdir(parents=True, exist_ok=True)
            dependency.write_text("synthetic source")
            sources[relative] = hashlib.sha256(dependency.read_bytes()).hexdigest()
    value = {
        "schema": "seed604-training-execution-freeze-v1",
        "status": "frozen-awaiting-owner-authorization",
        "execution_authorized": False,
        "source_sha256": sources or {},
    }
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return raw


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
            _freeze(root)
            with self.assertRaises((FileNotFoundError, ValueError)):
                executor.verify_authorization(root)

    def test_forged_or_stale_authorization_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze = _freeze(root)
            _authorization(root, freeze, execution_freeze_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "training_not_authorized"):
                executor.verify_authorization(root)

    def test_modified_freeze_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = _freeze(root)
            _authorization(root, raw)
            changed = json.loads(raw)
            changed["execution_authorized"] = True
            (root / executor.FREEZE).write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "execution_freeze_invalid"):
                executor.verify_authorization(root)

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
            raw = _freeze(root, hashes)
            _authorization(root, raw)
            dependency.write_text("changed")
            with self.assertRaisesRegex(ValueError, "execution_source_hash_mismatch"):
                executor.verify_authorization(root)

    def test_unknown_status_cannot_authorize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = _freeze(root)
            _authorization(root, raw, status="authorized")
            with self.assertRaisesRegex(ValueError, "training_not_authorized"):
                executor.verify_authorization(root)


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
        # The production architecture/checkpoint dependency belongs to the registered
        # runtime test matrix and is intentionally not run against a substitute here.
        self.skipTest("requires authenticated synthetic residual_v3 checkpoint fixture")

    def test_optimizer_state_continuity_gate_is_explicit(self) -> None:
        self.skipTest("requires registered PyTorch 2.13.0+cu130 CPU runtime")


if __name__ == "__main__":
    unittest.main()
