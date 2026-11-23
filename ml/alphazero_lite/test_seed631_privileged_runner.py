"""Unit tests for the Seed 631 synthetic-only runner boundary."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import seed630_synthetic_harness as harness
from ml.alphazero_lite import seed631_privileged_runner as runner


class Seed631RunnerTests(unittest.TestCase):
    def _freeze(self, root: Path) -> Path:
        freeze = {
            "schema": "seed630-synthetic-freeze-v1",
            "namespace": harness.SYNTHETIC_NAMESPACE,
            "execution_class": "synthetic_only",
            "worker_module": harness.WORKER_MODULE,
            "worker_source_sha256": harness._digest(
                harness._worker_script().read_bytes()
            ),
            "production_authorization": False,
            "worker_mode": "success",
            "run_id": "a150e840-e7c7-4a64-bd6e-174ed25b5e4a",
            "result_path": str(root / "result.json"),
            "quota_project_id": 63101,
            "cgroup_path": "/sys/fs/cgroup/seed631/worker",
            "output_path": "/mnt/seed631/output",
            "scratch_path": "/mnt/seed631/scratch",
            "seed631_attempt_record_path": str(root / "attempt.json"),
            "lock_path": "/run/seed631/slot",
            "worker_uid": 1001,
            "worker_gid": 1001,
            "limits": {
                "wall_seconds": 1,
                "cpu_seconds": 1,
                "memory_bytes": 4096,
                "concurrency": 1,
                "storage_bytes": 8192,
            },
        }
        path = root / "freeze.json"
        path.write_text(json.dumps(freeze, sort_keys=True, separators=(",", ":")))
        return path

    def test_arbitrary_modes_are_rejected_before_freeze_or_worker_access(self) -> None:
        with self.assertRaisesRegex(
            PermissionError, "unregistered_synthetic_worker_mode"
        ):
            runner.run_once(
                Path("missing-freeze"), Path("missing-config"), "python", "0" * 64
            )

    def test_exact_freeze_digest_is_required_before_harness_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            freeze_path = self._freeze(root)
            with patch.object(harness, "run_synthetic") as run:
                with self.assertRaisesRegex(PermissionError, "unique_seed631"):
                    runner.run_once(
                        freeze_path, root / "config.json", "success", "0" * 64
                    )
                run.assert_not_called()
            self.assertFalse((root / "attempt.json").exists())


if __name__ == "__main__":
    unittest.main()
