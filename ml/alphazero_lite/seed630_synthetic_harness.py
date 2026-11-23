"""Privileged, test-only Seed 629 worker harness.

This is intentionally independent of Seed 624 production authorization and can
launch exactly the registered synthetic worker module, never arbitrary argv.
It does not relax ``seed629_execution_supervisor.run_authorized``.
"""

from __future__ import annotations

import hashlib
import json
import os
import pwd
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


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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
        or freeze.get("production_authorization") is not False
    ):
        raise PermissionError("invalid_seed630_synthetic_freeze")
    return freeze, _digest(raw)


def run_synthetic(
    freeze_path: Path, config: dict[str, Any], mode: str
) -> dict[str, Any]:
    """Run one frozen worker mode after verifying kernel cgroup and XFS limits."""
    if mode not in MODES:
        raise PermissionError("unregistered_synthetic_worker_mode")
    freeze, freeze_digest = _load_synthetic_freeze(freeze_path)
    limits = freeze["limits"]
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
    lock_path = Path(config["lock_path"])
    slot = supervisor._acquire_slot(lock_path, limits["concurrency"])
    process: subprocess.Popen[bytes] | None = None
    start = time.monotonic()
    category: str | None = None
    try:
        memory_max = int(limits["memory_bytes"])
        cpu_usage_baseline = supervisor._read_int(cgroup / "cpu.stat", "usage_usec")
        memory_events_baseline = dict(
            line.split() for line in (cgroup / "memory.events").read_text().splitlines()
        )
        (cgroup / "memory.max").write_text(str(memory_max))
        (cgroup / "memory.oom.group").write_text("1")
        (cgroup / "cpu.max").write_text("max 100000")
        (cgroup / "pids.max").write_text("64")
        if (cgroup / "memory.max").read_text().strip() != str(memory_max):
            raise PermissionError("synthetic_memory_limit_readback_mismatch")
        output.mkdir(mode=0o700, exist_ok=True)
        scratch.mkdir(mode=0o700, exist_ok=True)
        os.chown(output, 0, worker_gid)
        os.chown(scratch, 0, worker_gid)
        os.chmod(output, 0o770)
        os.chmod(scratch, 0o770)
        destination = output / ("worker.bin" if mode == "write" else "worker.txt")
        worker_args = [WORKER_MODULE, mode, str(destination)]
        if mode == "write":
            worker_args.append(str(scratch / "worker.bin"))

        def drop_worker_privileges() -> None:
            os.setgroups([worker_gid])
            os.setgid(worker_gid)
            os.setuid(worker_uid)

        # Child starts blocked before exec, allowing cgroup attachment first.
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os,sys;sys.stdin.buffer.read(1);os.execv(sys.executable,[sys.executable,'-m',*sys.argv[1:]])",
                *worker_args,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            cwd=Path(config["working_directory"]),
            env={**os.environ, "TMPDIR": str(scratch)},
            preexec_fn=drop_worker_privileges,
        )
        (cgroup / "cgroup.procs").write_text(str(process.pid))
        assert process.stdin is not None
        process.stdin.write(b"1")
        process.stdin.close()
        deadline = start + limits["wall_seconds"]
        cpu_budget = limits["cpu_seconds"] * 1_000_000
        while process.poll() is None:
            usage = supervisor._read_int(cgroup / "cpu.stat", "usage_usec")
            events = dict(
                line.split()
                for line in (cgroup / "memory.events").read_text().splitlines()
            )
            if int(events.get("oom_kill", 0)) > 0:
                category = "memory_oom"
            elif usage >= cpu_budget:
                category = "cpu_exhaustion"
            elif time.monotonic() >= deadline:
                category = "wall_time_expiration"
            if category:
                supervisor._kill_cgroup(cgroup)
                break
            time.sleep(0.01)
        if process.poll() is None:
            supervisor._kill_cgroup(cgroup)
        exit_status = process.wait(timeout=10)
        stdout, stderr = process.communicate(timeout=10)
        if mode == "write" and b"No space left on device" in stderr:
            category = "storage_exhaustion"
        if (cgroup / "cgroup.procs").read_text().strip():
            supervisor._kill_cgroup(cgroup)
            if process.poll() is None:
                process.wait(timeout=10)
            raise RuntimeError("synthetic_process_tree_not_empty_after_kill")
        cpu_stat = (cgroup / "cpu.stat").read_text()
        memory_events = (cgroup / "memory.events").read_text()
        return {
            "schema": "seed630-synthetic-run-v1",
            "namespace": SYNTHETIC_NAMESPACE,
            "run_id": freeze["run_id"],
            "freeze_sha256": freeze_digest,
            "worker_mode": mode,
            "exit_status": exit_status,
            "worker_stdout": stdout.decode(errors="replace"),
            "worker_stderr": stderr.decode(errors="replace"),
            "termination_category": category,
            "elapsed_seconds": time.monotonic() - start,
            "cgroup": str(cgroup),
            "cgroup_populated_after_wait": (cgroup / "cgroup.events").read_text(),
            "cpu_stat": cpu_stat,
            "cpu_usage_baseline_usec": cpu_usage_baseline,
            "memory_events_baseline": memory_events_baseline,
            "memory_events": memory_events,
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
    finally:
        if process is not None and process.poll() is None:
            supervisor._kill_cgroup(cgroup)
            process.wait(timeout=10)
        os.close(slot)


def main() -> None:
    raise SystemExit("Synthetic runs require an operator-provisioned isolated host")


if __name__ == "__main__":
    main()
