"""Fail-closed Linux cgroup-v2 supervisor for an already authenticated Seed 624 run.

This module deliberately accepts no caller-supplied trust anchors. Production
authorization is delegated to Seed 624's existing freeze/receipt contract.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import platform
import stat
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


def _acquire_slot(lock_path: Path, slots: int) -> int:
    """Acquire one cross-process slot; caller owns and must close the descriptor."""
    for slot in range(slots):
        candidate = os.open(f"{lock_path}.{slot}", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return candidate
        except BlockingIOError:
            os.close(candidate)
        except BaseException:
            os.close(candidate)
            raise
    raise BlockingIOError("execution_concurrency_limit_reached")


def _release_slot(descriptor: int) -> None:
    if descriptor >= 0:
        os.close(descriptor)


def _cgroup2_mount_for(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, separator, right = line.partition(" - ")
        fields = right.split()
        if separator and fields and fields[0] == "cgroup2":
            mountpoint = Path(left.split()[4].replace("\\040", " ")).resolve()
            if resolved == mountpoint or mountpoint in resolved.parents:
                return mountpoint
    raise PermissionError("cgroup2_hierarchy_mount_not_found")


def _validate_cgroup(cgroup: Path) -> dict[str, str]:
    """Verify an empty dedicated cgroup and writable control interfaces."""
    if cgroup.is_symlink() or cgroup.resolve(strict=True) != cgroup.absolute():
        raise PermissionError("cgroup_path_symlink_or_noncanonical")
    _cgroup2_mount_for(cgroup)
    owner = cgroup.stat().st_uid
    if owner not in {0, os.geteuid()}:
        raise PermissionError("unsafe_cgroup_owner")
    for parent in (cgroup, *cgroup.parents):
        if parent == Path("/"):
            break
        parent_mode = parent.stat().st_mode
        if parent_mode & stat.S_IWOTH and not parent_mode & stat.S_ISVTX:
            raise PermissionError("world_writable_cgroup_path_component")
    required = (
        "cgroup.procs",
        "cgroup.kill",
        "memory.max",
        "memory.oom.group",
        "cpu.max",
        "pids.max",
    )
    for name in required:
        interface = cgroup / name
        if not interface.is_file() or not os.access(interface, os.W_OK):
            raise PermissionError(f"cgroup_interface_unavailable:{name}")
        descriptor = os.open(interface, os.O_WRONLY)
        os.close(descriptor)
    procs = (cgroup / "cgroup.procs").read_text().strip()
    events = dict(
        line.split() for line in (cgroup / "cgroup.events").read_text().splitlines()
    )
    if procs or events.get("populated") != "0":
        raise PermissionError("dedicated_cgroup_must_be_empty")
    (cgroup / "cgroup.kill").write_text("1")
    events_after_probe = dict(
        line.split() for line in (cgroup / "cgroup.events").read_text().splitlines()
    )
    if (cgroup / "cgroup.procs").read_text().strip() or events_after_probe.get(
        "populated"
    ) != "0":
        raise PermissionError("cgroup_kill_empty_probe_failed")
    controllers = set((cgroup / "cgroup.controllers").read_text().split())
    if not {"cpu", "memory", "pids"}.issubset(controllers):
        raise PermissionError("required_cgroup_controllers_unavailable")
    return {
        "controllers": " ".join(sorted(controllers)),
        "populated": events["populated"],
    }


def classify_final_accounting(
    *,
    cpu_usage_usec: int,
    memory_peak_bytes: int,
    memory_cap_bytes: int,
    memory_events: dict[str, int],
    wall_seconds: float,
    limits: dict[str, int],
    exit_status: int,
    termination_signal: int | None,
    residual_processes: bool,
) -> str | None:
    """Classify from final cgroup/clock observations, not just polling samples."""
    if any(
        memory_events.get(event, 0) for event in ("oom", "oom_kill", "oom_group_kill")
    ):
        return "memory_oom"
    if cpu_usage_usec >= limits["cpu_seconds"] * 1_000_000:
        return "cpu_exhaustion"
    if wall_seconds >= limits["wall_seconds"]:
        return "wall_time_expiration"
    if memory_peak_bytes >= memory_cap_bytes:
        return "memory_limit"
    if residual_processes:
        return "residual_descendants"
    if termination_signal is not None:
        return f"external_termination_signal_{termination_signal}"
    if exit_status != 0:
        return "ordinary_worker_failure"
    return None


def _memory_event_delta(
    current: dict[str, int], baseline: dict[str, int]
) -> dict[str, int]:
    return {key: max(0, value - baseline.get(key, 0)) for key, value in current.items()}


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
    _validate_cgroup(cgroup)
    if output.exists() or scratch.exists():
        raise FileExistsError("output_and_temporary_paths_must_be_create_only")
    for path in (output.parent, scratch.parent):
        if not path.is_dir() or path.is_symlink():
            raise PermissionError("dedicated_path_parent_invalid")
    # Caller assertions and environment/config flags are not a trusted quota backend.
    raise PermissionError("verified_hard_quota_backend_unavailable_launch_disabled")


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
    raise PermissionError("seed629_launch_path_unavailable_pending_verified_hard_quota")
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
    lock_fd = -1
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    category: str | None = None
    cpu_usage_usec = 0
    memory_peak = 0
    memory_events: dict[str, int] = {}
    memory_events_baseline: dict[str, int] = {}
    exit_status: int | None = None
    term_signal: int | None = None
    try:
        slots = config.get("lock_slots")
        if type(slots) is not int or slots != limits["execution_concurrency"]:
            raise PermissionError("coordination_slot_count_mismatch")
        lock_fd = _acquire_slot(lock_path, slots)
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
        memory_events_baseline = {
            key: int(value)
            for key, value in (
                line.split()
                for line in (cgroup / "memory.events").read_text().splitlines()
            )
        }
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
        cpu_usage_usec = _read_int(cgroup / "cpu.stat", "usage_usec")
        memory_peak = int((cgroup / "memory.peak").read_text())
        memory_events = {
            key: int(value)
            for key, value in (
                line.split()
                for line in (cgroup / "memory.events").read_text().splitlines()
            )
        }
        memory_events = _memory_event_delta(memory_events, memory_events_baseline)
        elapsed = time.monotonic() - started
        residual_processes = bool((cgroup / "cgroup.procs").read_text().strip())
        if residual_processes:
            _kill_cgroup(cgroup)
            residual_processes = bool((cgroup / "cgroup.procs").read_text().strip())
        category = classify_final_accounting(
            cpu_usage_usec=cpu_usage_usec,
            memory_peak_bytes=memory_peak,
            memory_cap_bytes=memory_cap,
            memory_events=memory_events,
            wall_seconds=elapsed,
            limits=limits,
            exit_status=exit_status,
            termination_signal=term_signal,
            residual_processes=residual_processes,
        )
        resource_breach = category in {
            "memory_oom",
            "memory_limit",
            "cpu_exhaustion",
            "wall_time_expiration",
        }
        status = "success" if exit_status == 0 and category is None else "failure"
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
        if resource_breach:
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
        _release_slot(lock_fd)


def main() -> None:
    raise SystemExit(
        "Use the authenticated library API; CLI authorization is intentionally absent"
    )


if __name__ == "__main__":
    main()
