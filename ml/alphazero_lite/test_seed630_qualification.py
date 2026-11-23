"""Unit tests for Seed 630's fail-closed qualification boundaries."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import seed629_execution_supervisor as supervisor
from ml.alphazero_lite import seed630_synthetic_harness as harness
from ml.alphazero_lite import seed630_xfs_project_quota as quota


class Seed630QualificationTests(unittest.TestCase):
    def test_quota_report_requires_one_finite_exact_project_row(self) -> None:
        report = "Project ID Used Soft Hard Warn/Grace\n#63001 0 0 2048 00 [------]\n"
        self.assertEqual(quota._project_hard_limit(report, 63001), 2048 * 512)
        for invalid in (
            "Project ID Used Soft Hard Warn/Grace\n#63001 0 0 0 00 [------]\n",
            "Project ID Used Soft Hard Warn/Grace\n#63001 0 0 2 00 [------]\n#63001 0 0 2 00 [------]\n",
            "Project ID Used Soft Hard Warn/Grace\n#63001 0 0 unlimited 00 [------]\n",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(PermissionError):
                quota._project_hard_limit(invalid, 63001)

    def test_synthetic_freeze_requires_strict_namespace_and_limits(self) -> None:
        limits = {
            "wall_seconds": 1,
            "cpu_seconds": 1,
            "memory_bytes": 1024,
            "concurrency": 1,
            "storage_bytes": 4096,
        }
        valid = {
            "schema": "seed630-synthetic-freeze-v1",
            "namespace": harness.SYNTHETIC_NAMESPACE,
            "execution_class": "synthetic_only",
            "worker_module": harness.WORKER_MODULE,
            "production_authorization": False,
            "run_id": "unit-test",
            "quota_project_id": 63001,
            "limits": limits,
        }
        with tempfile.TemporaryDirectory() as temporary:
            freeze = Path(temporary) / "freeze.json"
            freeze.write_text(json.dumps(valid))
            loaded, digest = harness._load_synthetic_freeze(freeze)
            self.assertEqual(loaded["run_id"], "unit-test")
            self.assertEqual(len(digest), 64)
            for mutation in (
                {"namespace": "production"},
                {"worker_module": "arbitrary"},
                {"production_authorization": True},
                {"limits": {**limits, "storage_bytes": 0}},
            ):
                invalid = {**valid, **mutation}
                freeze.write_text(json.dumps(invalid))
                with (
                    self.subTest(mutation=mutation),
                    self.assertRaisesRegex(
                        PermissionError, "invalid_seed630_synthetic_freeze"
                    ),
                ):
                    harness._load_synthetic_freeze(freeze)

    def test_harness_rejects_arbitrary_worker_modes_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            freeze = Path(temporary) / "not-read.json"
            with self.assertRaisesRegex(
                PermissionError, "unregistered_synthetic_worker_mode"
            ):
                harness.run_synthetic(freeze, {}, "python -c arbitrary")

    def test_production_launch_guard_is_still_unconditional(self) -> None:
        with patch.object(supervisor.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(
                PermissionError, "launch_path_unavailable_pending_verified_hard_quota"
            ):
                supervisor.run_authorized({}, {}, ["anything"])
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
