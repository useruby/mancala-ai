"""Seed 629 fail-closed and copied-evidence regression tests."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from ml.alphazero_lite import seed629_execution_supervisor as supervisor
from ml.alphazero_lite import verify_seed629_supervisor as verifier


class SupervisorTests(unittest.TestCase):
    def _fake_cgroup(self, root: Path) -> Path:
        cgroup = root / "cgroup"
        cgroup.mkdir()
        for name, contents in {
            "cgroup.procs": "",
            "cgroup.kill": "0",
            "memory.max": "max",
            "memory.oom.group": "0",
            "cpu.max": "max 100000",
            "pids.max": "max",
            "cgroup.events": "populated 0\nfrozen 0\n",
            "cgroup.controllers": "cpu memory pids\n",
        }.items():
            (cgroup / name).write_text(contents)
        return cgroup

    def test_run_authorized_refuses_without_verified_quota_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "quota_enforced": True,
                "quota_verified": True,
                "quota_identity": "caller-asserted",
                "quota_bytes": 1,
            }
            with patch.object(supervisor.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(PermissionError, "launch_path_unavailable"):
                    supervisor.run_authorized({}, config, ["synthetic-worker"])
                popen.assert_not_called()
            self.assertFalse((root / "out").exists())

    def test_cgroup_preflight_accepts_only_empty_supported_writable_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cgroup = self._fake_cgroup(Path(temporary))
            with patch.object(
                supervisor, "_cgroup2_mount_for", return_value=Path(temporary)
            ):
                observed = supervisor._validate_cgroup(cgroup)
            self.assertEqual(observed["populated"], "0")

    def test_cgroup_preflight_rejects_missing_kill_nonempty_and_bad_controllers(
        self,
    ) -> None:
        cases = (
            ("cgroup.kill", None, "cgroup_interface_unavailable:cgroup.kill"),
            (None, ("cgroup.procs", "123\n"), "dedicated_cgroup_must_be_empty"),
            (
                None,
                ("cgroup.controllers", "memory pids\n"),
                "required_cgroup_controllers_unavailable",
            ),
        )
        for missing, replacement, message in cases:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as temporary,
            ):
                cgroup = self._fake_cgroup(Path(temporary))
                if missing:
                    (cgroup / missing).unlink()
                if replacement:
                    (cgroup / replacement[0]).write_text(replacement[1])
                with patch.object(
                    supervisor, "_cgroup2_mount_for", return_value=Path(temporary)
                ):
                    with self.assertRaisesRegex(PermissionError, message):
                        supervisor._validate_cgroup(cgroup)

    def test_final_accounting_classifies_boundary_and_failure_cases(self) -> None:
        limits = {"cpu_seconds": 2, "wall_seconds": 3, "process_rss_bytes": 1000}
        common = {
            "memory_peak_bytes": 1,
            "memory_cap_bytes": 800,
            "memory_events": {},
            "wall_seconds": 1,
            "limits": limits,
            "exit_status": 0,
            "termination_signal": None,
            "residual_processes": False,
        }
        self.assertIsNone(
            supervisor.classify_final_accounting(cpu_usage_usec=1_999_999, **common)
        )
        self.assertEqual(
            supervisor.classify_final_accounting(cpu_usage_usec=2_000_000, **common),
            "cpu_exhaustion",
        )
        self.assertEqual(
            supervisor.classify_final_accounting(
                cpu_usage_usec=0, **{**common, "wall_seconds": 3}
            ),
            "wall_time_expiration",
        )
        self.assertEqual(
            supervisor.classify_final_accounting(
                cpu_usage_usec=0, **{**common, "memory_events": {"oom_kill": 1}}
            ),
            "memory_oom",
        )
        self.assertEqual(
            supervisor.classify_final_accounting(
                cpu_usage_usec=0, **{**common, "memory_peak_bytes": 800}
            ),
            "memory_limit",
        )
        self.assertEqual(
            supervisor.classify_final_accounting(
                cpu_usage_usec=0, **{**common, "residual_processes": True}
            ),
            "residual_descendants",
        )
        self.assertEqual(
            supervisor.classify_final_accounting(
                cpu_usage_usec=0, **{**common, "termination_signal": 9}
            ),
            "external_termination_signal_9",
        )

    def test_coordination_lock_descriptor_released_on_success_and_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "coordination"
            descriptor = supervisor._acquire_slot(lock_path, 1)
            supervisor._release_slot(descriptor)
            with self.assertRaises(OSError):
                os.fstat(descriptor)
            opened: list[int] = []
            original_open = os.open

            def tracked_open(*args: Any, **kwargs: Any) -> int:
                fd = original_open(*args, **kwargs)
                opened.append(fd)
                return fd

            with (
                patch.object(
                    supervisor.fcntl, "flock", side_effect=RuntimeError("rejected")
                ),
                patch.object(supervisor.os, "open", side_effect=tracked_open),
            ):
                with self.assertRaisesRegex(RuntimeError, "rejected"):
                    supervisor._acquire_slot(lock_path, 1)
            self.assertEqual(len(opened), 1)
            with self.assertRaises(OSError):
                os.fstat(opened[0])
            descriptor = supervisor._acquire_slot(lock_path, 1)
            supervisor._release_slot(descriptor)

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

    def test_verifier_requires_receipt_for_resource_termination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            limits = {name: 1 for name in supervisor.LIMITS}
            freeze = {
                "authorized_run_identity": "synthetic",
                "production_resource_limits": limits,
            }
            freeze_raw = json.dumps(freeze).encode()
            (root / "freeze.json").write_bytes(freeze_raw)
            result = {
                "schema": "seed629-supervisor-execution-result-v1",
                "execution_freeze_sha256": supervisor._sha(freeze_raw),
                "authorized_run_identity": "synthetic",
                "limits": limits,
                "measurements": {
                    "cpu_usage_usec": 1_000_000,
                    "wall_seconds": 0.1,
                    "memory_peak_bytes": 1,
                    "memory_events": {},
                },
                "cgroup": {"memory_max": 1},
                "breach_category": "cpu_exhaustion",
                "status": "failure",
                "partial_artifacts": [],
                "receipt_sha256": "unused",
            }
            result["evidence_sha256"] = supervisor._sha(
                json.dumps(
                    result, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                ).encode()
            )
            (root / "result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "frozen_failure_receipt_required"):
                verifier.verify(root / "result.json", root / "freeze.json")


if __name__ == "__main__":
    unittest.main()
