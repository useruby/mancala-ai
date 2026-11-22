"""Seed 629 fail-closed and copied-evidence regression tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite import seed629_execution_supervisor as supervisor
from ml.alphazero_lite import verify_seed629_supervisor as verifier


class SupervisorTests(unittest.TestCase):
    def test_missing_cgroup_and_quota_support_fail_closed(self) -> None:
        freeze = {"production_resource_limits": {name: 1 for name in supervisor.LIMITS}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "cgroup_path": str(root / "missing"),
                "output_path": str(root / "out"),
                "temporary_path": str(root / "scratch"),
                "quota_enforced": True,
                "quota_identity": "synthetic",
                "quota_bytes": 1,
            }
            with self.assertRaises((PermissionError, FileNotFoundError)):
                supervisor._preflight(freeze, config)

    def test_independent_verifier_rejects_tampered_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            freeze = {
                "authorized_run_identity": "synthetic",
                "production_resource_limits": {name: 1 for name in supervisor.LIMITS},
            }
            (root / "freeze.json").write_text(json.dumps(freeze))
            result = {
                "schema": "seed629-supervisor-execution-result-v1",
                "execution_freeze_sha256": "wrong",
                "authorized_run_identity": "synthetic",
            }
            (root / "result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "freeze_digest"):
                verifier.verify(root / "result.json", root / "freeze.json")


if __name__ == "__main__":
    unittest.main()
