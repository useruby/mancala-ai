"""Unit tests for Seed 631's runner trust and immutable-record boundaries."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import seed630_synthetic_harness as harness
from ml.alphazero_lite import seed631_privileged_runner as runner


class Seed631RunnerTests(unittest.TestCase):
    def _tempdir(self) -> tempfile.TemporaryDirectory[str]:
        return tempfile.TemporaryDirectory(
            dir=Path(__file__).parents[2] / ".tmp", prefix="seed631-test-"
        )

    def _freeze(self, root: Path, *, run_id: str | None = None) -> tuple[Path, dict]:
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
            "run_id": run_id or "a150e840-e7c7-4a64-bd6e-174ed25b5e4a",
            "result_path": str(root / "result.json"),
            "quota_project_id": 63101,
            "cgroup_path": "/sys/fs/cgroup/seed631/worker",
            "output_path": "/mnt/seed631/output",
            "scratch_path": "/mnt/seed631/scratch",
            "seed631_attempt_record_path": str(root / "attempt.json"),
            "worker_uid": os.geteuid() + 1,
            "worker_gid": os.getegid() + 1,
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
        return path, freeze

    def _config(self, freeze: dict) -> dict:
        return {
            key: freeze[key]
            for key in (
                "cgroup_path",
                "output_path",
                "scratch_path",
                "result_path",
                "worker_uid",
                "worker_gid",
            )
        }

    def _lock_directory(self, path: Path, slots: int) -> Path:
        path.mkdir(mode=0o700)
        capacity = path / runner.LOCK_CAPACITY_FILE
        capacity.write_text(f"{slots}\n")
        capacity.chmod(0o400)
        return path

    def _run_with_test_ownership(self, root: Path, freeze_path: Path, freeze: dict):
        config_path = root / "config.json"
        config = self._config(freeze)
        config_path.write_text(json.dumps(config))
        freeze_digest = harness._digest(freeze_path.read_bytes())
        real_record = runner._attempt_record

        def test_record(path, record, *, worker_uid, owner_uid=0):
            return real_record(
                path,
                record,
                worker_uid=worker_uid,
                owner_uid=os.geteuid(),
            )

        slot_fd = os.open(root / "mock-slot", os.O_CREAT | os.O_RDWR, 0o600)
        with (
            patch.object(runner, "_attempt_record", side_effect=test_record),
            patch.object(
                runner, "_acquire_shared_slot", return_value=slot_fd
            ) as acquire,
        ):
            result = runner.run_once(freeze_path, config_path, "success", freeze_digest)
        with self.assertRaises(OSError):
            os.fstat(slot_fd)
        return result, acquire

    def test_arbitrary_modes_are_rejected_before_freeze_or_worker_access(self) -> None:
        with self.assertRaisesRegex(
            PermissionError, "unregistered_synthetic_worker_mode"
        ):
            runner.run_once(
                Path("missing-freeze"), Path("missing-config"), "python", "0" * 64
            )

    def test_freeze_and_config_identity_mismatch_blocks_before_launch(self) -> None:
        with self._tempdir() as temporary:
            root = Path(temporary)
            freeze_path, freeze = self._freeze(root)
            for key, expected_error in (
                ("worker_uid", "worker_identity_not_bound"),
                ("output_path", "config_not_bound"),
            ):
                with self.subTest(key=key):
                    config = self._config(freeze)
                    config[key] = (
                        config[key] + 50 if key == "worker_uid" else "/wrong/output"
                    )
                    config_path = root / f"config-{key}.json"
                    config_path.write_text(json.dumps(config))
                    with (
                        patch.object(harness, "run_synthetic") as run,
                        patch.object(runner, "_acquire_shared_slot") as acquire,
                    ):
                        with self.assertRaisesRegex(PermissionError, expected_error):
                            runner.run_once(
                                freeze_path,
                                config_path,
                                "success",
                                harness._digest(freeze_path.read_bytes()),
                            )
                        run.assert_not_called()
                        acquire.assert_not_called()
            valid_config = root / "config-valid.json"
            valid_config.write_text(json.dumps(self._config(freeze)))
            with (
                patch.object(harness, "run_synthetic") as run,
                patch.object(runner, "_acquire_shared_slot") as acquire,
            ):
                with self.assertRaisesRegex(PermissionError, "unique_seed631"):
                    runner.run_once(freeze_path, valid_config, "success", "0" * 64)
                run.assert_not_called()
                acquire.assert_not_called()
            self.assertFalse((root / "attempt.json").exists())

    def test_freeze_cannot_supply_lock_namespace_through_config(self) -> None:
        with self._tempdir() as temporary:
            root = Path(temporary)
            freeze_path, freeze = self._freeze(root)
            freeze["lock_namespace"] = "/different/namespace"
            freeze_path.write_text(json.dumps(freeze))
            config = self._config(freeze)
            config["lock_path"] = "/different/namespace/slot"
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            with (
                patch.object(harness, "run_synthetic") as run,
                patch.object(runner, "_acquire_shared_slot") as acquire,
            ):
                with self.assertRaisesRegex(PermissionError, "config_schema_invalid"):
                    runner.run_once(
                        freeze_path,
                        config_path,
                        "success",
                        harness._digest(freeze_path.read_bytes()),
                    )
                run.assert_not_called()
                acquire.assert_not_called()

    def test_unsafe_and_symlinked_lock_directories_are_rejected(self) -> None:
        with self._tempdir() as temporary:
            root = Path(temporary)
            unsafe = root / "unsafe"
            unsafe.mkdir(mode=0o777)
            unsafe.chmod(0o777)
            with self.assertRaisesRegex(PermissionError, "writable_by_group_or_other"):
                runner._acquire_shared_slot(
                    1,
                    worker_uid=os.geteuid() + 1,
                    directory=unsafe,
                    owner_uid=os.geteuid(),
                )

            target = root / "target"
            self._lock_directory(target, 1)
            symlink = root / "linked"
            symlink.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(PermissionError, "not_canonical"):
                runner._acquire_shared_slot(
                    1,
                    worker_uid=os.geteuid() + 1,
                    directory=symlink,
                    owner_uid=os.geteuid(),
                )

    def test_slot_symlink_nonregular_and_unsafe_modes_are_rejected(self) -> None:
        worker_uid = os.geteuid() + 1
        owner_uid = os.geteuid()
        with self._tempdir() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.write_text("not a lock")
            link_dir = root / "link-dir"
            self._lock_directory(link_dir, 1)
            (link_dir / "slot-0.lock").symlink_to(target)
            with self.assertRaisesRegex(PermissionError, "slot_file_unsafe"):
                runner._acquire_shared_slot(
                    1,
                    worker_uid=worker_uid,
                    directory=link_dir,
                    owner_uid=owner_uid,
                )

            nonregular = root / "nonregular"
            self._lock_directory(nonregular, 1)
            (nonregular / "slot-0.lock").mkdir()
            with self.assertRaises(OSError):
                runner._acquire_shared_slot(
                    1,
                    worker_uid=worker_uid,
                    directory=nonregular,
                    owner_uid=owner_uid,
                )

            unsafe = root / "unsafe-file"
            self._lock_directory(unsafe, 1)
            slot = unsafe / "slot-0.lock"
            slot.touch(mode=0o644)
            slot.chmod(0o644)
            with self.assertRaisesRegex(PermissionError, "slot_file_unsafe"):
                runner._acquire_shared_slot(
                    1,
                    worker_uid=worker_uid,
                    directory=unsafe,
                    owner_uid=owner_uid,
                )

    def test_independent_run_ids_share_lock_namespace_and_contention_cleans_fd(
        self,
    ) -> None:
        worker_uid = os.geteuid() + 1
        with self._tempdir() as temporary:
            root = Path(temporary)
            first_freeze = root / "freeze-a.json"
            second_freeze = root / "freeze-b.json"
            first = {
                "run_id": "first",
                "limits": {"concurrency": 1},
                "lock_dir": "/one",
            }
            second = {
                "run_id": "second",
                "limits": {"concurrency": 1},
                "lock_dir": "/another",
            }
            first_freeze.write_text(json.dumps(first))
            second_freeze.write_text(json.dumps(second))
            lock_directory = root / "shared"
            self._lock_directory(lock_directory, 1)

            def acquire_for_freeze(_freeze):
                # The run/freeze cannot supply this host-global directory.
                return runner._acquire_shared_slot(
                    1,
                    worker_uid=worker_uid,
                    directory=lock_directory,
                    owner_uid=os.geteuid(),
                )

            descriptor = acquire_for_freeze(first)
            opened: list[int] = []
            real_open = runner.os.open

            def tracked_open(*args, **kwargs):
                opened_fd = real_open(*args, **kwargs)
                opened.append(opened_fd)
                return opened_fd

            try:
                with patch.object(runner.os, "open", side_effect=tracked_open):
                    with self.assertRaisesRegex(
                        BlockingIOError, "execution_concurrency_limit_reached"
                    ):
                        acquire_for_freeze(second)
                self.assertTrue(opened)
                with self.assertRaises(OSError):
                    os.fstat(opened[-1])
            finally:
                os.close(descriptor)
            recovered = acquire_for_freeze(second)
            os.close(recovered)
            self.assertNotEqual(first["lock_dir"], second["lock_dir"])

    def test_freeze_cannot_expand_root_registered_shared_slot_capacity(self) -> None:
        with self._tempdir() as temporary:
            lock_directory = self._lock_directory(Path(temporary) / "shared", 1)
            with self.assertRaisesRegex(
                PermissionError, "lock_capacity_freeze_mismatch"
            ):
                runner._acquire_shared_slot(
                    2,
                    worker_uid=os.geteuid() + 1,
                    directory=lock_directory,
                    owner_uid=os.geteuid(),
                )

    def test_forked_worker_does_not_keep_supervisor_slot_open(self) -> None:
        with self._tempdir() as temporary:
            root = Path(temporary)
            lock_directory = self._lock_directory(root / "shared", 1)
            lock_fd = runner._acquire_shared_slot(
                1,
                worker_uid=os.geteuid() + 1,
                directory=lock_directory,
                owner_uid=os.geteuid(),
            )
            ready_read, ready_write = os.pipe()
            gate_read, gate_write = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(ready_read)
                os.close(gate_write)
                harness._close_inherited_fds((lock_fd,))
                os.write(ready_write, b"R")
                os.read(gate_read, 1)
                os._exit(0)

            os.close(ready_write)
            os.close(gate_read)
            try:
                self.assertEqual(os.read(ready_read, 1), b"R")
                os.close(lock_fd)
                recovered = runner._acquire_shared_slot(
                    1,
                    worker_uid=os.geteuid() + 1,
                    directory=lock_directory,
                    owner_uid=os.geteuid(),
                )
                os.close(recovered)
            finally:
                os.write(gate_write, b"X")
                os.close(gate_write)
                os.close(ready_read)
                _pid, status = os.waitpid(pid, 0)
            self.assertEqual(os.waitstatus_to_exitcode(status), 0)

    def test_success_record_binds_exact_result_hash(self) -> None:
        with self._tempdir() as temporary:
            root = Path(temporary)
            freeze_path, freeze = self._freeze(root)
            result = {
                "schema": "seed630-synthetic-run-v1",
                "run_id": freeze["run_id"],
                "freeze_sha256": harness._digest(freeze_path.read_bytes()),
                "worker_mode": "success",
                "exit_status": 0,
            }
            result_raw = json.dumps(result, sort_keys=True).encode()

            def complete_worker(_freeze_path, _config, _mode, **_kwargs):
                (root / "result.json").write_bytes(result_raw)
                return result

            with patch.object(harness, "run_synthetic", side_effect=complete_worker):
                self._run_with_test_ownership(root, freeze_path, freeze)
            completion = json.loads((root / "attempt.json.success.json").read_bytes())
            self.assertEqual(completion["result_sha256"], harness._digest(result_raw))
            self.assertEqual(completion["freeze_sha256"], result["freeze_sha256"])

    def test_failure_record_collision_preserves_original_exception(self) -> None:
        with self._tempdir() as temporary:
            root = Path(temporary)
            freeze_path, freeze = self._freeze(root)
            (root / "attempt.json.failure.json").write_text("preexisting")
            original = RuntimeError("worker failed")
            config_path = root / "config.json"
            config_path.write_text(json.dumps(self._config(freeze)))
            real_record = runner._attempt_record

            def test_record(path, record, *, worker_uid, owner_uid=0):
                return real_record(
                    path,
                    record,
                    worker_uid=worker_uid,
                    owner_uid=os.geteuid(),
                )

            slot_fd = os.open(root / "failure-slot", os.O_CREAT | os.O_RDWR, 0o600)
            with (
                patch.object(runner, "_attempt_record", side_effect=test_record),
                patch.object(
                    runner,
                    "_acquire_shared_slot",
                    return_value=slot_fd,
                ),
                patch.object(harness, "run_synthetic", side_effect=original) as run,
            ):
                with self.assertRaisesRegex(RuntimeError, "worker failed") as raised:
                    runner.run_once(
                        freeze_path,
                        config_path,
                        "success",
                        harness._digest(freeze_path.read_bytes()),
                    )
                self.assertIs(raised.exception, original)
                self.assertEqual(run.call_count, 1)
            with self.assertRaises(OSError):
                os.fstat(slot_fd)
            self.assertTrue((root / "attempt.json").exists())

    def test_worker_failure_publishes_separate_failure_record_once(self) -> None:
        with self._tempdir() as temporary:
            root = Path(temporary)
            freeze_path, freeze = self._freeze(root)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(self._config(freeze)))
            original = RuntimeError("synthetic worker failed")
            real_record = runner._attempt_record

            def test_record(path, record, *, worker_uid, owner_uid=0):
                return real_record(
                    path,
                    record,
                    worker_uid=worker_uid,
                    owner_uid=os.geteuid(),
                )

            slot_fd = os.open(
                root / "worker-failure-slot", os.O_CREAT | os.O_RDWR, 0o600
            )
            with (
                patch.object(runner, "_attempt_record", side_effect=test_record),
                patch.object(
                    runner,
                    "_acquire_shared_slot",
                    return_value=slot_fd,
                ),
                patch.object(harness, "run_synthetic", side_effect=original) as run,
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic worker failed"):
                    runner.run_once(
                        freeze_path,
                        config_path,
                        "success",
                        harness._digest(freeze_path.read_bytes()),
                    )
                self.assertEqual(run.call_count, 1)
            with self.assertRaises(OSError):
                os.fstat(slot_fd)
            failure = json.loads((root / "attempt.json.failure.json").read_bytes())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["error_type"], "RuntimeError")
            self.assertTrue((root / "attempt.json").exists())

    def test_success_record_collision_fails_without_retry_or_overwrite(self) -> None:
        with self._tempdir() as temporary:
            root = Path(temporary)
            freeze_path, freeze = self._freeze(root)
            completion_path = root / "attempt.json.success.json"
            completion_path.write_text("keep-me")
            result = {
                "schema": "seed630-synthetic-run-v1",
                "run_id": freeze["run_id"],
                "freeze_sha256": harness._digest(freeze_path.read_bytes()),
                "worker_mode": "success",
                "exit_status": 0,
            }

            def complete_worker(_freeze_path, _config, _mode, **_kwargs):
                (root / "result.json").write_text(json.dumps(result))
                return result

            config_path = root / "config.json"
            config_path.write_text(json.dumps(self._config(freeze)))
            record = runner._attempt_record

            def test_record(path, payload, *, worker_uid, owner_uid=0):
                return record(
                    path,
                    payload,
                    worker_uid=worker_uid,
                    owner_uid=os.geteuid(),
                )

            with (
                patch.object(runner, "_attempt_record", side_effect=test_record),
                patch.object(
                    runner,
                    "_acquire_shared_slot",
                    return_value=os.open(root / "slot", os.O_CREAT | os.O_RDWR, 0o600),
                ),
                patch.object(
                    harness, "run_synthetic", side_effect=complete_worker
                ) as run,
            ):
                with self.assertRaises(FileExistsError):
                    runner.run_once(
                        freeze_path,
                        config_path,
                        "success",
                        harness._digest(freeze_path.read_bytes()),
                    )
                self.assertEqual(run.call_count, 1)
            self.assertEqual(completion_path.read_text(), "keep-me")
            self.assertTrue((root / "attempt.json.failure.json").exists())


if __name__ == "__main__":
    unittest.main()
