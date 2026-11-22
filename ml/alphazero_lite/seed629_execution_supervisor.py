"""Fail-closed Linux cgroup-v2 supervisor for an already authenticated Seed 624 run.

This module deliberately accepts no caller-supplied trust anchors. Production
authorization is delegated to Seed 624's existing freeze/receipt contract.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed624_prospective_benchmark as seed624

VERSION = "seed629-cgroup-v2-v1"
LIMITS = (
    "wall_seconds",
    "cpu_seconds",
    "process_rss_bytes",
    "execution_concurrency",
    "output_and_temporary_storage_bytes",
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _durable_create(path: Path, payload: dict[str, Any]) -> None:
    seed624._create_durable_json(path, payload)


def _read_int(path: Path, field: str) -> int:
    values = dict(line.split() for line in path.read_text().splitlines())
    return int(values[field])


def _preflight(freeze: dict[str, Any], config: dict[str, Any]) -> tuple[Path, Path]:
    """Validate pre-provisioned cgroup/quota paths without creating them."""
    limits = freeze.get("production_resource_limits")
    if not isinstance(limits, dict) or any(
        type(limits.get(key)) is not int or limits[key] <= 0 for key in LIMITS
    ):
        raise PermissionError("five_positive_frozen_limits_required")
    cgroup = Path(config["cgroup_path"])
    output = Path(config["output_path"])
    scratch = Path(config["temporary_path"])
    contract = freeze.get("supervisor_contract", {})
    if (
        contract.get("implementation") != VERSION
        or contract.get("source_file")
        != "ml/alphazero_lite/seed629_execution_supervisor.py"
        or contract.get("source_sha256") != _sha(Path(__file__).read_bytes())
        or str(output) not in freeze.get("create_only_output_paths", [])
        or str(scratch) != freeze.get("temporary_path")
        or str(config.get("result_path")) != freeze.get("supervisor_result_path")
        or config.get("required_verification_commands")
        != freeze.get("verification_commands")
    ):
        raise PermissionError("supervisor_configuration_not_bound_by_freeze")
    if not Path("/sys/fs/cgroup/cgroup.controllers").is_file():
        raise PermissionError("cgroup_v2_unavailable")
    if not cgroup.is_dir() or not (cgroup / "cgroup.procs").is_file():
        raise PermissionError("precreated_delegated_cgroup_required")
    controllers = (cgroup / "cgroup.controllers").read_text().split()
    if not {"cpu", "memory", "pids"}.issubset(controllers):
        raise PermissionError("required_cgroup_controllers_unavailable")
    if not os.access(cgroup / "cgroup.procs", os.W_OK):
        raise PermissionError("cgroup_write_privilege_required")
    if output.exists() or scratch.exists():
        raise FileExistsError("output_and_temporary_paths_must_be_create_only")
    for path in (output.parent, scratch.parent):
        if not path.is_dir() or path.is_symlink():
            raise PermissionError("dedicated_path_parent_invalid")
    if config.get("quota_enforced") is not True or not config.get("quota_identity"):
        raise PermissionError("hard_combined_storage_quota_required")
    if config.get("quota_bytes") != limits["output_and_temporary_storage_bytes"]:
        raise PermissionError("storage_quota_limit_mismatch")
    return cgroup, Path(config["coordination_lock"])


def _kill_cgroup(cgroup: Path) -> None:
    kill = cgroup / "cgroup.kill"
    if not kill.is_file():
        raise RuntimeError("cgroup_kill_unavailable")
    kill.write_text("1")
    deadline = time.monotonic() + 10
    while (cgroup / "cgroup.procs").read_text().strip():
        if time.monotonic() >= deadline:
            raise RuntimeError("cgroup_descendants_not_reaped")
        time.sleep(0.02)


def run_authorized(
    authorization: dict[str, Any],
    config: dict[str, Any],
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one command under the signed freeze's external resource limits.

    Callers must pass the authenticated Seed 624 authorization record. The
    worker executes with new-session isolation and is moved into its dedicated
    cgroup before it is allowed to execute user code.
    """
    binding = seed624._failure_receipt_binding(authorization)
    freeze = binding["freeze"]
    if not isinstance(command, list) or command != freeze.get("worker_command"):
        raise PermissionError("worker_command_not_bound_by_freeze")
    cgroup, lock_path = _preflight(freeze, config)
    limits = freeze["production_resource_limits"]
    result_path = Path(config["result_path"])
    if result_path.exists():
        raise FileExistsError("execution_result_must_be_create_only")
    output = Path(config["output_path"])
    scratch = Path(config["temporary_path"])
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    category: str | None = None
    cpu_usage_usec = 0
    memory_peak = 0
    memory_events: dict[str, int] = {}
    exit_status: int | None = None
    term_signal: int | None = None
    try:
        slots = config.get("lock_slots")
        if type(slots) is not int or slots != limits["execution_concurrency"]:
            raise PermissionError("coordination_slot_count_mismatch")
        lock_fd = -1
        for slot in range(slots):
            candidate = os.open(f"{lock_path}.{slot}", os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_fd = candidate
                break
            except BlockingIOError:
                os.close(candidate)
        if lock_fd < 0:
            raise BlockingIOError("execution_concurrency_limit_reached")
        memory_cap = int(
            limits["process_rss_bytes"] * config.get("memory_safety_factor", 0.8)
        )
        if memory_cap <= 0 or memory_cap >= limits["process_rss_bytes"]:
            raise PermissionError("conservative_cgroup_memory_ceiling_required")
        (cgroup / "memory.max").write_text(str(memory_cap))
        (cgroup / "memory.oom.group").write_text("1")
        (cgroup / "cpu.max").write_text("max 100000")
        (cgroup / "pids.max").write_text(str(config.get("tasks_max", 4096)))
        if int((cgroup / "memory.max").read_text()) != memory_cap:
            raise PermissionError("memory_ceiling_not_applied")
        output.mkdir(mode=0o700)
        scratch.mkdir(mode=0o700)
        env = os.environ.copy()
        env.update(environment or {})
        env["TMPDIR"] = str(scratch.resolve())
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os,sys; sys.stdin.buffer.read(1); os.execvpe(sys.argv[1],sys.argv[1:],os.environ)",
                *command,
            ],
            cwd=Path(config["working_directory"]),
            env=env,
            stdin=subprocess.PIPE,
            start_new_session=True,
        )
        (cgroup / "cgroup.procs").write_text(str(process.pid))
        assert process.stdin is not None
        process.stdin.write(b"1")
        process.stdin.close()
        deadline = started + limits["wall_seconds"]
        while process.poll() is None:
            elapsed = time.monotonic() - started
            cpu_usage_usec = _read_int(cgroup / "cpu.stat", "usage_usec")
            memory_peak = int((cgroup / "memory.peak").read_text())
            memory_events = {
                key: int(value)
                for key, value in (
                    line.split()
                    for line in (cgroup / "memory.events").read_text().splitlines()
                )
            }
            if memory_events.get("oom_kill", 0):
                category = "memory_oom"
            elif cpu_usage_usec >= limits["cpu_seconds"] * 1_000_000:
                category = "cpu_exhaustion"
            elif time.monotonic() >= deadline:
                category = "wall_time_expiration"
            elif memory_peak >= limits["process_rss_bytes"]:
                category = "memory_oom"
            if category:
                _kill_cgroup(cgroup)
                break
            time.sleep(min(0.05, max(0.005, deadline - time.monotonic())))
        if process.poll() is None:
            _kill_cgroup(cgroup)
        exit_status = process.wait()
        if exit_status < 0:
            term_signal = -exit_status
        status = "success" if exit_status == 0 and category is None else "failure"
        cpu_usage_usec = _read_int(cgroup / "cpu.stat", "usage_usec")
        memory_peak = int((cgroup / "memory.peak").read_text())
        elapsed = time.monotonic() - started
        payload = {
            "schema": "seed629-supervisor-execution-result-v1",
            "status": status,
            "execution_freeze_sha256": binding["freeze_sha256"],
            "authorized_run_identity": binding["authorized_run_identity"],
            "limits": limits,
            "supervisor": {
                "version": VERSION,
                "sha256": _sha(Path(__file__).read_bytes()),
            },
            "host": {"platform": platform.platform(), "kernel": platform.release()},
            "cgroup": {
                "path": str(cgroup),
                "memory_max": memory_cap,
                "cpu_max": (cgroup / "cpu.max").read_text().strip(),
                "pids_max": (cgroup / "pids.max").read_text().strip(),
            },
            "measurements": {
                "cpu_usage_usec": cpu_usage_usec,
                "memory_peak_bytes": memory_peak,
                "memory_events": memory_events,
                "wall_seconds": elapsed,
                "storage_quota_identity": config["quota_identity"],
                "storage_quota_bytes": config["quota_bytes"],
            },
            "exit_status": exit_status,
            "termination_signal": term_signal,
            "breach_category": category,
            "partial_artifacts": [str(output), str(scratch)],
            "receipt_sha256": None,
        }
        if category:
            try:
                receipt = seed624.write_failure_receipt(
                    authorization=authorization,
                    category="external_resource_limit_termination",
                    diagnostic={
                        "termination_category": category,
                        "supervisor_version": VERSION,
                        "execution_result_path": str(result_path),
                    },
                )
                payload["receipt_sha256"] = _sha(receipt.read_bytes())
            except BaseException as error:
                payload["receipt_error"] = f"{type(error).__name__}:{error}"
        payload["evidence_sha256"] = _sha(seed624.canonical_bytes(payload))
        _durable_create(result_path, payload)
        return payload
    except BaseException:
        if process is not None and process.poll() is None:
            _kill_cgroup(cgroup)
            process.wait()
        raise
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)


def main() -> None:
    raise SystemExit(
        "Use the authenticated library API; CLI authorization is intentionally absent"
    )


if __name__ == "__main__":
    main()
