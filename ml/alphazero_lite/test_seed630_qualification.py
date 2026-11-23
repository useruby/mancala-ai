"""Unit tests for Seed 630's fail-closed qualification boundaries."""

from __future__ import annotations

import json
import errno
import contextlib
import io
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import seed629_execution_supervisor as supervisor
from ml.alphazero_lite import seed630_synthetic_harness as harness
from ml.alphazero_lite import seed630_xfs_project_quota as quota
from ml.alphazero_lite import verify_seed630_os_evidence as evidence_verifier
from ml.alphazero_lite import seed629_synthetic_worker as worker


class Seed630QualificationTests(unittest.TestCase):
    def test_quota_report_requires_one_finite_exact_project_row(self) -> None:
        # xfs_quota(8): report block counts are 1K units by default.
        report = (
            "Project quota on /mnt/seed630 (/dev/loop0)\n"
            "                           Blocks\n"
            "Project ID       Used   Soft   Hard Warn/Grace\n"
            "---------- -------- -------- -------- ----------------\n"
            "#63001          1024        0     2048 00 [------]\n"
        )
        self.assertEqual(quota._project_hard_limit(report, 63001), 2048 * 1024)
        self.assertEqual(quota._project_usage_bytes(report, 63001), 1024 * 1024)
        one_kib_report = (
            "Project ID Used Soft Hard Warn/Grace\n#63001 1 0 1 00 [------]\n"
        )
        self.assertEqual(quota._project_hard_limit(one_kib_report, 63001), 1024)
        self.assertEqual(quota._project_usage_bytes(one_kib_report, 63001), 1024)
        for invalid in (
            "Project ID Used Soft Hard Warn/Grace\n#63001 0 0 0 00 [------]\n",
            "Project ID Used Soft Hard Warn/Grace\n#63001 0 0 2 00 [------]\n#63001 0 0 2 00 [------]\n",
            "Project ID Used Soft Hard Warn/Grace\n#63001 0 0 unlimited 00 [------]\n",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(PermissionError):
                quota._project_hard_limit(invalid, 63001)

    def test_project_quota_state_rejects_accounting_only_and_ambiguous(self) -> None:
        active = (
            "Project quota state on /mnt/seed630 (/dev/loop0)\n"
            "  Accounting: ON\n"
            "  Enforcement: ON\n"
            "  Inode: #128 (1 blocks, 0 extents)\n"
        )
        quota._verify_project_quota_state(active)
        for invalid in (
            active.replace("Enforcement: ON", "Enforcement: OFF"),
            active.replace("Project quota state on", "Group quota state on"),
            active + "  Enforcement: OFF\n",
            active.replace("Accounting: ON", "Accounting: OFF"),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(PermissionError):
                quota._verify_project_quota_state(invalid)

    def test_storage_source_excludes_user_group_and_inode_quota_limits(self) -> None:
        disabled = (
            "User quota state on /mnt/seed630 (/dev/loop0)\n"
            "  Accounting: OFF\n"
            "  Enforcement: OFF\n"
            "Group quota state on /mnt/seed630 (/dev/loop0)\n"
            "  Accounting: ON\n"
            "  Enforcement: OFF\n"
        )
        quota._verify_nonproject_quotas_disabled(disabled)
        for enabled in (
            disabled.replace("Enforcement: OFF", "Enforcement: ON", 1),
            disabled.replace("Group quota state", "Project quota state", 1),
        ):
            with self.subTest(enabled=enabled), self.assertRaises(PermissionError):
                quota._verify_nonproject_quotas_disabled(enabled)
        evidence_verifier._assert_project_inode_limit_unlimited(
            "Project ID Used Soft Hard Warn/Grace\n#63001 1 0 0 00 [------]\n",
            63001,
        )
        with self.assertRaisesRegex(ValueError, "inode_limit_must_be_unlimited"):
            evidence_verifier._assert_project_inode_limit_unlimited(
                "Project ID Used Soft Hard Warn/Grace\n#63001 1 0 2 00 [------]\n",
                63001,
            )

    def test_project_directory_identity_record_binds_paths_and_inheritance(
        self,
    ) -> None:
        xattr = bytearray(28)
        xattr[0:4] = (0x200).to_bytes(4, "little")
        xattr[12:16] = (63001).to_bytes(4, "little")
        freeze = {
            "output_path": "/mnt/seed630/run/output",
            "scratch_path": "/mnt/seed630/run/scratch",
        }
        records = [
            {
                "path": path,
                "project_id": 63001,
                "xflags": 0x200,
                "uid": 0,
                "mode": 0o770,
                "fsxattr_hex": bytes(xattr).hex(),
            }
            for path in (freeze["output_path"], freeze["scratch_path"])
        ]
        payload = json.dumps(records).encode()
        evidence_verifier._assert_project_directory_identities(payload, freeze, 63001)
        records[1]["project_id"] = 63002
        with self.assertRaisesRegex(ValueError, "kernel_identity_mismatch"):
            evidence_verifier._assert_project_directory_identities(
                json.dumps(records).encode(), freeze, 63001
            )

    def test_structured_storage_failure_distinguishes_edquot_from_enospc(self) -> None:
        edquot = json.dumps(
            {
                "schema": "seed630-worker-failure-v1",
                "reason": "project_quota_exceeded",
                "errno": errno.EDQUOT,
                "errno_name": "EDQUOT",
            }
        )
        enospc = json.dumps(
            {
                "schema": "seed630-worker-failure-v1",
                "reason": "filesystem_space_exhausted",
                "errno": errno.ENOSPC,
                "errno_name": "ENOSPC",
            }
        )
        self.assertEqual(
            harness._storage_failure(edquot, 74)["reason"], "project_quota_exceeded"
        )
        self.assertEqual(
            harness._storage_failure(enospc, 75)["reason"], "filesystem_space_exhausted"
        )
        self.assertIsNone(harness._storage_failure("No space left on device", 1))
        self.assertIsNone(harness._storage_failure(edquot, 75))
        self.assertEqual(
            evidence_verifier._structured_worker_failure(
                {"worker_stderr": edquot, "exit_status": 74}
            ),
            json.loads(edquot),
        )
        self.assertIsNone(
            evidence_verifier._structured_worker_failure(
                {"worker_stderr": enospc, "exit_status": 75}
            )
        )

    def test_worker_emits_registered_edquot_failure_record(self) -> None:
        captured = io.StringIO()
        with (
            patch.object(
                worker.sys,
                "argv",
                ["seed629_synthetic_worker", "write", "/out/file", "/scratch/file"],
            ),
            patch.object(Path, "open", side_effect=OSError(errno.EDQUOT, "quota")),
            contextlib.redirect_stderr(captured),
            self.assertRaises(SystemExit) as raised,
        ):
            worker.main()
        self.assertEqual(raised.exception.code, 74)
        record = json.loads(captured.getvalue())
        self.assertEqual(record["schema"], "seed630-worker-failure-v1")
        self.assertEqual(record["reason"], "project_quota_exceeded")
        self.assertEqual(record["errno"], errno.EDQUOT)

    def test_counter_deltas_use_nonzero_historical_baselines(self) -> None:
        self.assertEqual(
            harness._counter_deltas(
                {"usage_usec": 120, "oom_kill": 8},
                {"usage_usec": 100, "oom_kill": 7},
            ),
            {"usage_usec": 20, "oom_kill": 1},
        )
        with self.assertRaisesRegex(RuntimeError, "counter_decreased"):
            harness._counter_deltas({"oom": 2}, {"oom": 3})
        self.assertEqual(
            evidence_verifier._counter_delta(1_000_000, 0, "cpu"), 1_000_000
        )
        self.assertEqual(evidence_verifier._counter_delta(999_999, 0, "cpu"), 999_999)
        self.assertEqual(
            evidence_verifier._event_deltas({"oom_kill": "9"}, {"oom_kill": 7}),
            {"oom_kill": 2},
        )
        self.assertEqual(
            evidence_verifier._event_deltas({"oom_kill": "7"}, {"oom_kill": 7}),
            {"oom_kill": 0},
        )
        with self.assertRaisesRegex(ValueError, "counter_decreased"):
            evidence_verifier._event_deltas({"oom_kill": "6"}, {"oom_kill": 7})

    def test_signed_incomplete_manifest_is_rejected(self) -> None:
        if not shutil.which("openssl"):
            self.skipTest("openssl is required by the evidence signer/verifier")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence"
            raw_directory = evidence / "raw"
            raw_directory.mkdir(parents=True)
            for name in evidence_verifier.EXPECTED_RAW_FILES:
                (raw_directory / name).write_text(f"fixture:{name}")
            freeze = {
                "run_id": "signed-incomplete-test",
                "quota_project_id": 63001,
                "cgroup_path": "/sys/fs/cgroup/test",
                "output_path": "/mnt/test/output",
                "scratch_path": "/mnt/test/scratch",
                "limits": {
                    "wall_seconds": 1,
                    "cpu_seconds": 1,
                    "memory_bytes": 1024,
                    "concurrency": 1,
                    "storage_bytes": 1024,
                },
            }
            freeze_path = root / "freeze.json"
            freeze_raw = json.dumps(freeze, sort_keys=True).encode()
            freeze_path.write_bytes(freeze_raw)
            source_identity = {
                source_name: "a" * 64
                for source_name in sorted(evidence_verifier.EXPECTED_SOURCE_FILES)
            }
            source_digest = evidence_verifier._source_bundle_digest(source_identity)
            manifest = {
                "schema": "seed630-os-evidence-manifest-v1",
                "collector_version": "test",
                "execution_class": "synthetic_only",
                "collector_source_sha256": source_digest,
                "source_identity": source_identity,
                "freeze_sha256": evidence_verifier._sha(freeze_raw),
                "synthetic_run_id": freeze["run_id"],
                "worker_exit_status": 0,
                "cgroup_path": "/sys/fs/cgroup/test",
                "quota_mount": "/mnt/test",
                "quota_source": "/dev/loop0",
                "project_id": 63001,
                "host": {"uid": 0},
                "raw_files": [
                    {
                        "path": f"raw/{name}",
                        "sha256": evidence_verifier._sha(
                            (raw_directory / name).read_bytes()
                        ),
                    }
                    for name in sorted(evidence_verifier.EXPECTED_RAW_FILES)[:-1]
                ],
            }
            manifest_path = evidence / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            private_key = root / "collector.pem"
            public_key = root / "collector.pub"
            subprocess.run(
                [
                    "openssl",
                    "genpkey",
                    "-algorithm",
                    "ED25519",
                    "-out",
                    str(private_key),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "openssl",
                    "pkey",
                    "-in",
                    str(private_key),
                    "-pubout",
                    "-out",
                    str(public_key),
                ],
                check=True,
                capture_output=True,
            )
            signature = evidence / "manifest.sig"

            def sign_manifest() -> None:
                subprocess.run(
                    [
                        "openssl",
                        "pkeyutl",
                        "-sign",
                        "-rawin",
                        "-inkey",
                        str(private_key),
                        "-in",
                        str(manifest_path),
                        "-out",
                        str(signature),
                    ],
                    check=True,
                    capture_output=True,
                )

            sign_manifest()
            with self.assertRaisesRegex(ValueError, "inventory_incomplete"):
                evidence_verifier.verify(
                    evidence, freeze_path, public_key, source_digest
                )
            manifest["raw_files"] = [
                {
                    "path": f"raw/{name}",
                    "sha256": evidence_verifier._sha(
                        (raw_directory / name).read_bytes()
                    ),
                }
                for name in sorted(evidence_verifier.EXPECTED_RAW_FILES)
            ]
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            sign_manifest()
            (raw_directory / "cpu.stat").write_text("modified after signature")
            with self.assertRaisesRegex(ValueError, "raw_hash_mismatch"):
                evidence_verifier.verify(
                    evidence, freeze_path, public_key, source_digest
                )

    def test_worker_process_success_and_forced_termination_are_reaped(self) -> None:
        stdout_fd = os.memfd_create("seed630-test-out", os.MFD_CLOEXEC)
        stderr_fd = os.memfd_create("seed630-test-err", os.MFD_CLOEXEC)
        success = subprocess.Popen(
            [sys.executable, "-I", "-c", "import sys;print('done');sys.exit(7)"],
            stdin=subprocess.PIPE,
            stdout=stdout_fd,
            stderr=stderr_fd,
        )
        assert success.stdin is not None
        success.stdin.close()
        success.stdin = None
        self.assertEqual(
            harness._wait_and_reap(success, timeout=2, terminate=success.kill), 7
        )
        self.assertEqual(success.returncode, 7)
        self.assertIn(b"done", harness._read_memfd(stdout_fd))
        os.close(stdout_fd)
        os.close(stderr_fd)

        forced = subprocess.Popen(
            [sys.executable, "-I", "-c", "import time;time.sleep(60)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert forced.stdin is not None
        forced.stdin.close()
        forced.stdin = None
        status = harness._wait_and_reap(
            forced, timeout=0.05, terminate=forced.terminate
        )
        self.assertEqual(status, -signal.SIGTERM)
        self.assertIsNotNone(forced.returncode)

    def test_gated_worker_handle_releases_waits_terminates_and_reaps(self) -> None:
        def start_blocked_child(sleep_after_release: bool) -> harness._WorkerProcess:
            gate_read, gate_write = os.pipe2(os.O_CLOEXEC)
            pid = os.fork()
            if pid == 0:
                os.close(gate_write)
                released = os.read(gate_read, 1)
                os.close(gate_read)
                if released != b"1":
                    os._exit(126)
                if sleep_after_release:
                    import time

                    time.sleep(60)
                os._exit(0 if sleep_after_release else 9)
            os.close(gate_read)
            return harness._WorkerProcess(pid, gate_write)

        completed = start_blocked_child(sleep_after_release=False)
        completed.release()
        self.assertEqual(completed.wait(timeout=2), 9)
        self.assertEqual(completed.poll(), 9)

        interrupted = start_blocked_child(sleep_after_release=True)
        interrupted.release()
        self.assertEqual(
            harness._wait_and_reap(
                interrupted, timeout=0.05, terminate=interrupted.terminate
            ),
            -signal.SIGTERM,
        )
        self.assertEqual(interrupted.poll(), -signal.SIGTERM)

    def test_synthetic_worker_environment_ignores_pythonpath(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            with patch.dict(os.environ, {"PYTHONPATH": "/caller/redirect"}):
                environment = harness._worker_environment(scratch)
            self.assertNotIn("PYTHONPATH", environment)
            self.assertEqual(
                harness._worker_script().name, "seed629_synthetic_worker.py"
            )
            self.assertEqual(environment["TMPDIR"], str(scratch.resolve()))

    def test_raw_inventory_rejects_incomplete_duplicate_tampered_and_extra_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            raw_directory = evidence / "raw"
            raw_directory.mkdir()
            (evidence / "manifest.json").write_text("signed-integration-fixture")
            (evidence / "manifest.sig").write_bytes(b"signature-fixture")
            inventory = []
            for name in sorted(evidence_verifier.EXPECTED_RAW_FILES):
                content = f"raw:{name}".encode()
                (raw_directory / name).write_bytes(content)
                inventory.append(
                    {
                        "path": f"raw/{name}",
                        "sha256": evidence_verifier._sha(content),
                    }
                )
            manifest = {"raw_files": inventory}
            self.assertEqual(
                set(evidence_verifier._verify_raw_inventory(evidence, manifest)),
                evidence_verifier.EXPECTED_RAW_FILES,
            )
            with self.assertRaisesRegex(ValueError, "inventory_duplicated"):
                evidence_verifier._verify_raw_inventory(
                    evidence, {"raw_files": [*inventory, inventory[0]]}
                )
            changed = dict(manifest)
            changed["raw_files"] = inventory[:-1]
            with self.assertRaisesRegex(ValueError, "inventory_incomplete"):
                evidence_verifier._verify_raw_inventory(evidence, changed)
            (raw_directory / "cpu.stat").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "raw_hash_mismatch"):
                evidence_verifier._verify_raw_inventory(evidence, manifest)
            (raw_directory / "cpu.stat").write_text("raw:cpu.stat")
            (raw_directory / "unexpected").write_text("extra")
            with self.assertRaisesRegex(
                ValueError, "inventory_incomplete_or_unexpected"
            ):
                evidence_verifier._verify_raw_inventory(evidence, manifest)

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
            "worker_source_sha256": harness._digest(
                harness._worker_script().read_bytes()
            ),
            "production_authorization": False,
            "run_id": "unit-test",
            "result_path": "/root/seed630/unit-test-result.json",
            "quota_project_id": 63001,
            "cgroup_path": "/sys/fs/cgroup/seed630/unit-test",
            "output_path": "/mnt/seed630/unit-test/output",
            "scratch_path": "/mnt/seed630/unit-test/scratch",
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
                {"worker_source_sha256": "0" * 64},
                {"production_authorization": True},
                {"result_path": "relative/result.json"},
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
