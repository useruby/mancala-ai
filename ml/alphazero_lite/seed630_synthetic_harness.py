"""Privileged, test-only Seed 629 worker harness.

This is intentionally independent of Seed 624 production authorization and can
launch exactly the registered synthetic worker module, never arbitrary argv.
It does not relax ``seed629_execution_supervisor.run_authorized``.
"""

from __future__ import annotations

import hashlib
import errno
import json
import os
import pwd
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed629_execution_supervisor as supervisor
from ml.alphazero_lite.seed630_xfs_project_quota import verify_shared_project_quota

SYNTHETIC_NAMESPACE = "seed630.synthetic.v1"
WORKER_MODULE = "ml.alphazero_lite.seed629_synthetic_worker"
MODES = {"success", "cpu", "memory", "sleep", "write", "child"}
_SEED631_COORDINATION_TOKEN = object()


class _WorkerProcess:
    """Minimal waitable child handle for a pre-exec cgroup-attachment gate."""

    def __init__(self, pid: int, gate_fd: int) -> None:
        self.pid = pid
        self._gate_fd: int | None = gate_fd
        self.returncode: int | None = None

    def release(self) -> None:
        if self._gate_fd is None:
            raise RuntimeError("synthetic_worker_gate_already_closed")
        try:
            if os.write(self._gate_fd, b"1") != 1:
                raise RuntimeError("synthetic_worker_gate_release_failed")
        finally:
            os.close(self._gate_fd)
            self._gate_fd = None

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            return self.returncode
        if pid == 0:
            return None
        self.returncode = os.waitstatus_to_exitcode(status)
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while (status := self.poll()) is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(
                    "seed630-synthetic-worker", timeout or 0.0
                )
            time.sleep(0.005)
        return status

    def terminate(self) -> None:
        self._signal(signal.SIGTERM)

    def kill(self) -> None:
        self._close_gate()
        self._signal(signal.SIGKILL)

    def _close_gate(self) -> None:
        if self._gate_fd is not None:
            os.close(self._gate_fd)
            self._gate_fd = None

    def _signal(self, signum: int) -> None:
        if self.poll() is None:
            try:
                os.kill(self.pid, signum)
            except ProcessLookupError:
                pass


def _fork_blocked_worker(
    *,
    worker_script: Path,
    worker_args: list[str],
    worker_uid: int,
    worker_gid: int,
    stdout_fd: int,
    stderr_fd: int,
    working_directory: Path,
    environment: dict[str, str],
    close_fds: tuple[int, ...] = (),
) -> _WorkerProcess:
    """Fork a blocked child so no Python worker code runs before cgroup attach."""
    gate_read, gate_write = os.pipe2(os.O_CLOEXEC)
    try:
        pid = os.fork()
    except BaseException:
        os.close(gate_read)
        os.close(gate_write)
        raise
    if pid == 0:
        try:
            os.close(gate_write)
            _close_inherited_fds(close_fds)
            os.setgroups([worker_gid])
            os.setgid(worker_gid)
            os.setuid(worker_uid)
            os.setsid()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.chdir(working_directory)
            if os.read(gate_read, 1) != b"1":
                os._exit(126)
            os.execve(
                sys.executable,
                [sys.executable, "-I", str(worker_script), *worker_args],
                environment,
            )
        except BaseException as error:
            try:
                os.write(
                    2,
                    f"synthetic_worker_exec_failure:{type(error).__name__}\n".encode(),
                )
            finally:
                os._exit(127)
    os.close(gate_read)
    return _WorkerProcess(pid, gate_write)


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_memfd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while chunk := os.read(descriptor, 65536):
        chunks.append(chunk)
    return b"".join(chunks)


def _worker_script() -> Path:
    return Path(__file__).with_name("seed629_synthetic_worker.py").resolve(strict=True)


def _close_inherited_fds(descriptors: tuple[int, ...]) -> None:
    """Drop supervisor-only descriptors in a forked worker before its gate."""
    for descriptor in descriptors:
        os.close(descriptor)


def _worker_environment(scratch: Path) -> dict[str, str]:
    """Build a minimal environment without inherited Python import controls."""
    return {
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        "TMPDIR": str(scratch.resolve(strict=True)),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _validate_result_path(path: Path) -> None:
    if path.exists() or path.is_symlink() or not path.is_absolute():
        raise FileExistsError("synthetic_result_must_be_absolute_and_create_only")
    if path.parent.resolve(strict=True) / path.name != path:
        raise PermissionError("synthetic_result_path_not_canonical")
    for parent in (path.parent, *path.parent.parents):
        metadata = parent.stat()
        if metadata.st_uid != 0 or metadata.st_mode & 0o022:
            raise PermissionError("synthetic_result_parent_not_root_private")


def _create_result(path: Path, payload: dict[str, Any]) -> None:
    """Durably publish create-only root evidence outside worker-writable trees."""
    _validate_result_path(path)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o400,
    )
    try:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("synthetic_result_short_write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _counter_deltas(final: dict[str, int], baseline: dict[str, int]) -> dict[str, int]:
    if set(final) != set(baseline) or any(
        value < baseline[key] for key, value in final.items()
    ):
        raise RuntimeError("cgroup_counter_decreased_during_synthetic_run")
    return {key: value - baseline[key] for key, value in final.items()}


def _wait_and_reap(
    process: Any,
    *,
    timeout: float,
    terminate: Any,
) -> int:
    """Bound the wait, run cgroup cleanup on timeout, and always reap the leader."""
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        termination_error: BaseException | None = None
        try:
            terminate()
        except BaseException as error:
            termination_error = error
        try:
            status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            status = process.wait()
        if termination_error is not None:
            raise termination_error
        return status


def _storage_failure(stderr: str, exit_status: int) -> dict[str, Any] | None:
    """Accept only the worker's structured errno record and matching exit code."""
    try:
        records = [json.loads(line) for line in stderr.splitlines() if line.strip()]
    except json.JSONDecodeError:
        return None
    if len(records) != 1:
        return None
    record = records[0]
    expected = {
        ("project_quota_exceeded", errno.EDQUOT, 74),
        ("filesystem_space_exhausted", errno.ENOSPC, 75),
    }
    observed = (record.get("reason"), record.get("errno"), exit_status)
    if (
        record.get("schema") != "seed630-worker-failure-v1"
        or observed not in expected
        or record.get("errno_name") != errno.errorcode.get(record["errno"])
    ):
        return None
    return record


def _load_synthetic_freeze(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    freeze = json.loads(raw)
    limits = freeze.get("limits")
    if (
        freeze.get("schema") != "seed630-synthetic-freeze-v1"
        or freeze.get("namespace") != SYNTHETIC_NAMESPACE
        or freeze.get("execution_class") != "synthetic_only"
        or not isinstance(limits, dict)
        or set(limits)
        != {
            "wall_seconds",
            "cpu_seconds",
            "memory_bytes",
            "concurrency",
            "storage_bytes",
        }
        or any(type(value) is not int or value <= 0 for value in limits.values())
        or freeze.get("worker_module") != WORKER_MODULE
        or freeze.get("worker_source_sha256") != _digest(_worker_script().read_bytes())
        or not isinstance(freeze.get("result_path"), str)
        or not Path(freeze["result_path"]).is_absolute()
        or not isinstance(freeze.get("output_path"), str)
        or not Path(freeze["output_path"]).is_absolute()
        or not isinstance(freeze.get("scratch_path"), str)
        or not Path(freeze["scratch_path"]).is_absolute()
        or not isinstance(freeze.get("cgroup_path"), str)
        or not Path(freeze["cgroup_path"]).is_absolute()
        or type(freeze.get("quota_project_id")) is not int
        or freeze["quota_project_id"] <= 0
        or not isinstance(freeze.get("run_id"), str)
        or not freeze["run_id"]
        or freeze.get("production_authorization") is not False
    ):
        raise PermissionError("invalid_seed630_synthetic_freeze")
    return freeze, _digest(raw)


def run_synthetic(
    freeze_path: Path,
    config: dict[str, Any],
    mode: str,
    *,
    coordination_lock_fd: int | None = None,
    coordination_lock_token: object | None = None,
) -> dict[str, Any]:
    """Run one frozen worker mode after verifying kernel cgroup and XFS limits."""
    if mode not in MODES:
        raise PermissionError("unregistered_synthetic_worker_mode")
    freeze, freeze_digest = _load_synthetic_freeze(freeze_path)
    limits = freeze["limits"]
    result_path = Path(config.get("result_path", ""))
    if str(result_path) != freeze["result_path"]:
        raise PermissionError("synthetic_result_path_not_frozen")
    if result_path.exists() or result_path.is_symlink():
        raise FileExistsError("synthetic_result_must_be_create_only")
    _validate_result_path(result_path)
    if os.geteuid() != 0:
        raise PermissionError("privileged_synthetic_harness_required")
    worker_uid = config.get("worker_uid")
    worker_gid = config.get("worker_gid")
    if (
        type(worker_uid) is not int
        or type(worker_gid) is not int
        or worker_uid <= 0
        or worker_gid <= 0
    ):
        raise PermissionError("dedicated_unprivileged_synthetic_account_required")
    try:
        account = pwd.getpwuid(worker_uid)
    except KeyError:
        raise PermissionError("synthetic_worker_account_missing") from None
    if account.pw_gid != worker_gid:
        raise PermissionError("synthetic_worker_primary_group_mismatch")
    cgroup = Path(config["cgroup_path"])
    output = Path(config["output_path"])
    scratch = Path(config["scratch_path"])
    if (
        str(cgroup) != freeze["cgroup_path"]
        or str(output) != freeze["output_path"]
        or str(scratch) != freeze["scratch_path"]
        or cgroup.resolve(strict=True) != cgroup
        or output.resolve(strict=True) != output
        or scratch.resolve(strict=True) != scratch
    ):
        raise PermissionError("synthetic_resource_paths_not_frozen")
    canonical_result_path = result_path.parent.resolve(strict=True) / result_path.name
    if any(
        canonical_result_path == root or root in canonical_result_path.parents
        for root in (output.resolve(strict=True), scratch.resolve(strict=True))
    ):
        raise PermissionError("synthetic_result_inside_worker_writable_artifacts")
    quota = verify_shared_project_quota(
        output,
        scratch,
        expected_project_id=freeze["quota_project_id"],
        expected_hard_limit_bytes=limits["storage_bytes"],
    )
    observed_cgroup = supervisor._validate_cgroup(cgroup)
    if observed_cgroup["populated"] != "0":
        raise PermissionError("synthetic_cgroup_not_empty")
    if (
        output.exists()
        and any(output.iterdir())
        or scratch.exists()
        and any(scratch.iterdir())
    ):
        raise FileExistsError("synthetic_artifact_directories_must_be_empty")
    if coordination_lock_fd is not None and (
        coordination_lock_token is not _SEED631_COORDINATION_TOKEN
        or type(coordination_lock_fd) is not int
    ):
        raise PermissionError("seed631_shared_coordination_lock_required")
    owns_slot = type(coordination_lock_fd) is not int
    if owns_slot:
        lock_path = Path(config["lock_path"])
        slot = supervisor._acquire_slot(lock_path, limits["concurrency"])
    else:
        slot = coordination_lock_fd
    process: _WorkerProcess | None = None
    stdout_fd = -1
    stderr_fd = -1
    start = time.monotonic()
    category: str | None = None
    termination_mechanisms: list[str] = []
    try:
        stdout_fd = os.memfd_create("seed630-worker-stdout", os.MFD_CLOEXEC)
        stderr_fd = os.memfd_create("seed630-worker-stderr", os.MFD_CLOEXEC)
        memory_max = int(limits["memory_bytes"])
        cpu_usage_baseline = supervisor._read_int(cgroup / "cpu.stat", "usage_usec")
        memory_events_baseline = dict(
            line.split() for line in (cgroup / "memory.events").read_text().splitlines()
        )
        (cgroup / "memory.max").write_text(str(memory_max))
        (cgroup / "memory.oom.group").write_text("1")
        (cgroup / "cpu.max").write_text("max 100000")
        (cgroup / "pids.max").write_text("64")
        (cgroup / "memory.peak").write_text("0")
        memory_peak_baseline = int((cgroup / "memory.peak").read_text())
        if (cgroup / "memory.max").read_text().strip() != str(memory_max):
            raise PermissionError("synthetic_memory_limit_readback_mismatch")
        output.mkdir(mode=0o700, exist_ok=True)
        scratch.mkdir(mode=0o700, exist_ok=True)
        os.chown(output, 0, worker_gid)
        os.chown(scratch, 0, worker_gid)
        os.chmod(output, 0o770)
        os.chmod(scratch, 0o770)
        destination = output / ("worker.bin" if mode == "write" else "worker.txt")
        worker_script = _worker_script()
        worker_args = [mode, str(destination)]
        if mode == "write":
            worker_args.append(str(scratch / "worker.bin"))

        # No worker interpreter initialization runs before cgroup attachment.
        process = _fork_blocked_worker(
            worker_script=worker_script,
            worker_args=worker_args,
            worker_uid=worker_uid,
            worker_gid=worker_gid,
            stdout_fd=stdout_fd,
            stderr_fd=stderr_fd,
            working_directory=worker_script.parent,
            environment=_worker_environment(scratch),
            close_fds=(slot,),
        )
        (cgroup / "cgroup.procs").write_text(str(process.pid))
        process.release()
        deadline = start + limits["wall_seconds"]
        cpu_budget = limits["cpu_seconds"] * 1_000_000
        while process.poll() is None:
            usage = (
                supervisor._read_int(cgroup / "cpu.stat", "usage_usec")
                - cpu_usage_baseline
            )
            events = dict(
                line.split()
                for line in (cgroup / "memory.events").read_text().splitlines()
            )
            if int(events.get("oom_kill", 0)) > int(
                memory_events_baseline.get("oom_kill", 0)
            ):
                category = "memory_oom"
            elif usage >= cpu_budget:
                category = "cpu_exhaustion"
            elif time.monotonic() >= deadline:
                category = "wall_time_expiration"
            if category:
                if category == "memory_oom":
                    termination_mechanisms.append("memory.oom.group")
                else:
                    termination_mechanisms.append("supervisor_cgroup.kill")
                supervisor._kill_cgroup(cgroup)
                if category == "memory_oom":
                    termination_mechanisms.append("supervisor_cgroup.kill")
                break
            time.sleep(0.01)
        if process.poll() is None:
            exit_status = _wait_and_reap(
                process,
                timeout=10,
                terminate=lambda: supervisor._kill_cgroup(cgroup),
            )
        else:
            exit_status = process.wait()
        if (cgroup / "cgroup.procs").read_text().strip():
            supervisor._kill_cgroup(cgroup)
        if (cgroup / "cgroup.procs").read_text().strip():
            raise RuntimeError("synthetic_process_tree_not_empty_after_kill")
        stdout = _read_memfd(stdout_fd)
        stderr = _read_memfd(stderr_fd)
        cpu_stat = (cgroup / "cpu.stat").read_text()
        cpu_usage_usec = (
            supervisor._read_int(cgroup / "cpu.stat", "usage_usec") - cpu_usage_baseline
        )
        final_events = dict(
            line.split() for line in (cgroup / "memory.events").read_text().splitlines()
        )
        memory_event_delta = _counter_deltas(
            {key: int(value) for key, value in final_events.items()},
            {key: int(value) for key, value in memory_events_baseline.items()},
        )
        elapsed = time.monotonic() - start
        storage_failure = _storage_failure(stderr.decode(errors="replace"), exit_status)
        # Reclassify from final post-exit counters, even when polling saw no breach.
        if memory_event_delta.get("oom_kill", 0) and exit_status == -signal.SIGKILL:
            category = "memory_oom"
            if "memory.oom.group" not in termination_mechanisms:
                termination_mechanisms.append("memory.oom.group")
        elif (
            cpu_usage_usec >= cpu_budget
            and exit_status == -signal.SIGKILL
            and "supervisor_cgroup.kill" in termination_mechanisms
        ):
            category = "cpu_exhaustion"
        elif (
            elapsed >= limits["wall_seconds"]
            and exit_status == -signal.SIGKILL
            and "supervisor_cgroup.kill" in termination_mechanisms
        ):
            category = "wall_time_expiration"
        elif storage_failure and storage_failure["reason"] == "project_quota_exceeded":
            category = "storage_project_quota_exhaustion"
        elif (
            storage_failure
            and storage_failure["reason"] == "filesystem_space_exhausted"
        ):
            category = "storage_filesystem_exhaustion"
        memory_peak = int((cgroup / "memory.peak").read_text())
        result = {
            "schema": "seed630-synthetic-run-v1",
            "namespace": SYNTHETIC_NAMESPACE,
            "run_id": freeze["run_id"],
            "freeze_sha256": freeze_digest,
            "worker_mode": mode,
            "exit_status": exit_status,
            "worker_stdout": stdout.decode(errors="replace"),
            "worker_stderr": stderr.decode(errors="replace"),
            "storage_failure": storage_failure,
            "termination_category": category,
            "termination_mechanisms": sorted(set(termination_mechanisms)),
            "elapsed_seconds": elapsed,
            "cgroup": str(cgroup),
            "cgroup_populated_after_wait": (cgroup / "cgroup.events").read_text(),
            "cpu_stat": cpu_stat,
            "cpu_usage_baseline_usec": cpu_usage_baseline,
            "cpu_usage_usec_delta": cpu_usage_usec,
            "memory_peak_baseline_bytes": memory_peak_baseline,
            "memory_peak_bytes": memory_peak,
            "memory_events_baseline": memory_events_baseline,
            "memory_events": final_events,
            "memory_event_delta": memory_event_delta,
            "quota": {
                "filesystem": "xfs",
                "mountpoint": str(quota.mountpoint),
                "source": quota.source,
                "project_id": quota.project_id,
                "hard_limit_bytes": quota.hard_limit_bytes,
                "native_report": quota.quota_report,
            },
            "artifacts": [str(output), str(scratch)],
            "captured_at_unix_ns": time.time_ns(),
        }
        _create_result(result_path, result)
        return result
    finally:
        cleanup_error: BaseException | None = None
        try:
            if process is not None:
                if process.poll() is None:
                    _wait_and_reap(
                        process,
                        timeout=10,
                        terminate=lambda: supervisor._kill_cgroup(cgroup),
                    )
                if (cgroup / "cgroup.procs").read_text().strip():
                    supervisor._kill_cgroup(cgroup)
                if (cgroup / "cgroup.procs").read_text().strip():
                    raise RuntimeError("synthetic_process_tree_not_reaped_in_finally")
        except BaseException as error:
            cleanup_error = error
        finally:
            if stdout_fd >= 0:
                os.close(stdout_fd)
            if stderr_fd >= 0:
                os.close(stderr_fd)
            if owns_slot:
                os.close(slot)
        if cleanup_error is not None:
            raise cleanup_error


def main() -> None:
    raise SystemExit("Synthetic runs require an operator-provisioned isolated host")


if __name__ == "__main__":
    main()
