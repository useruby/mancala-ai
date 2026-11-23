"""One-shot, synthetic-only entry point for Seed 631 privileged qualification."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import stat
import sys
import time
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed630_synthetic_harness as harness

SCHEMA = "seed631-synthetic-attempt-v1"
LOCK_DIRECTORY = Path("/run/seed631-locks")
LOCK_CAPACITY_FILE = "slot-count"


def _check_trusted_path(
    path: Path, *, owner_uid: int, worker_uid: int, require_directory: bool
) -> None:
    """Reject symlinks, writable ancestors, and worker-controlled ownership."""
    if not path.is_absolute() or path != path.resolve(strict=True):
        raise PermissionError("seed631_trusted_path_not_canonical")
    components = (Path("/"), *path.parents[::-1], path)
    for index, component in enumerate(components):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise PermissionError("seed631_trusted_path_symlink")
        is_final = index == len(components) - 1
        if is_final:
            if require_directory and not stat.S_ISDIR(info.st_mode):
                raise PermissionError("seed631_lock_directory_not_directory")
            if info.st_uid != owner_uid:
                raise PermissionError("seed631_trusted_directory_owner_mismatch")
        elif info.st_uid == worker_uid:
            raise PermissionError("seed631_trusted_path_worker_owned_ancestor")
        if info.st_mode & 0o022:
            raise PermissionError("seed631_trusted_path_writable_by_group_or_other")


def _open_trusted_directory(path: Path, *, owner_uid: int, worker_uid: int) -> int:
    """Open a trusted directory component-by-component without following links."""
    _check_trusted_path(
        path, owner_uid=owner_uid, worker_uid=worker_uid, require_directory=True
    )
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        components = path.parts[1:]
        for index, component in enumerate(components):
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
            metadata = os.fstat(descriptor)
            is_final = index == len(components) - 1
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or (is_final and metadata.st_uid != owner_uid)
                or (not is_final and metadata.st_uid == worker_uid)
                or metadata.st_mode & 0o022
            ):
                raise PermissionError("seed631_lock_directory_descriptor_invalid")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _acquire_shared_slot(
    slots: int,
    *,
    worker_uid: int,
    directory: Path = LOCK_DIRECTORY,
    owner_uid: int = 0,
) -> int:
    """Acquire a slot in the single host-level Seed 631 lock namespace."""
    if type(slots) is not int or slots <= 0:
        raise ValueError("seed631_positive_concurrency_required")
    if os.geteuid() != 0 and owner_uid == 0:
        raise PermissionError("seed631_privileged_lock_acquisition_required")
    directory_fd = _open_trusted_directory(
        directory, owner_uid=owner_uid, worker_uid=worker_uid
    )
    try:
        capacity_path_stat = os.stat(
            LOCK_CAPACITY_FILE, dir_fd=directory_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(capacity_path_stat.st_mode)
            or capacity_path_stat.st_uid != owner_uid
            or capacity_path_stat.st_mode & 0o022
            or capacity_path_stat.st_nlink != 1
        ):
            raise PermissionError("seed631_lock_capacity_file_unsafe")
        capacity_fd = os.open(
            LOCK_CAPACITY_FILE,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_fd,
        )
        try:
            capacity_stat = os.fstat(capacity_fd)
            if (
                not stat.S_ISREG(capacity_stat.st_mode)
                or capacity_stat.st_uid != owner_uid
                or capacity_stat.st_mode & 0o022
                or capacity_stat.st_nlink != 1
                or (capacity_stat.st_dev, capacity_stat.st_ino)
                != (capacity_path_stat.st_dev, capacity_path_stat.st_ino)
            ):
                raise PermissionError("seed631_lock_capacity_file_unsafe")
            capacity_raw = os.read(capacity_fd, 33)
            if len(capacity_raw) > 32 or not capacity_raw.endswith(b"\n"):
                raise PermissionError("seed631_lock_capacity_file_invalid")
            capacity = capacity_raw[:-1].decode("ascii")
            if not capacity.isdecimal() or int(capacity) != slots:
                raise PermissionError("seed631_lock_capacity_freeze_mismatch")
        finally:
            os.close(capacity_fd)
        for index in range(slots):
            name = f"slot-{index}.lock"
            try:
                slot_path_stat = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                slot_path_stat = None
            if slot_path_stat is not None:
                if (
                    not stat.S_ISREG(slot_path_stat.st_mode)
                    or slot_path_stat.st_uid != owner_uid
                    or slot_path_stat.st_mode & 0o077
                    or slot_path_stat.st_nlink != 1
                ):
                    raise PermissionError("seed631_lock_slot_file_unsafe")
                descriptor = os.open(
                    name,
                    os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=directory_fd,
                )
            else:
                descriptor = os.open(
                    name,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600,
                    dir_fd=directory_fd,
                )
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != owner_uid
                    or metadata.st_mode & 0o077
                    or metadata.st_nlink != 1
                ):
                    raise PermissionError("seed631_lock_slot_file_unsafe")
                path_metadata = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
                if (path_metadata.st_dev, path_metadata.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    raise PermissionError("seed631_lock_slot_path_changed")
                if slot_path_stat is not None and (
                    path_metadata.st_dev,
                    path_metadata.st_ino,
                ) != (slot_path_stat.st_dev, slot_path_stat.st_ino):
                    raise PermissionError("seed631_lock_slot_path_changed")
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return descriptor
                except BlockingIOError:
                    os.close(descriptor)
            except BaseException:
                if not _descriptor_is_closed(descriptor):
                    os.close(descriptor)
                raise
        raise BlockingIOError("execution_concurrency_limit_reached")
    finally:
        os.close(directory_fd)


def _descriptor_is_closed(descriptor: int) -> bool:
    try:
        os.fstat(descriptor)
    except OSError:
        return True
    return False


def _attempt_record(
    path: Path,
    record: dict[str, Any],
    *,
    worker_uid: int,
    owner_uid: int = 0,
) -> None:
    """Create a durable root-private record without replacing prior attempts."""
    if path.exists() or path.is_symlink() or not path.is_absolute():
        raise FileExistsError("seed631_attempt_record_must_be_absolute_and_create_only")
    _check_trusted_path(
        path.parent, owner_uid=owner_uid, worker_uid=worker_uid, require_directory=True
    )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o400,
    )
    descriptor_open = True
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != owner_uid
            or metadata.st_mode & 0o077
            or metadata.st_nlink != 1
        ):
            raise PermissionError("seed631_attempt_record_file_unsafe")
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        stream = os.fdopen(descriptor, "wb", closefd=True)
        descriptor_open = False
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        # Exclusive-create artifacts are deliberately retained on partial failure.
        if descriptor_open:
            os.close(descriptor)
        raise


def run_once(
    freeze_path: Path,
    config_path: Path,
    mode: str,
    expected_freeze_sha256: str,
) -> dict[str, Any]:
    """Run one registered synthetic mode, preserving one attempt record."""
    if mode not in harness.MODES:
        raise PermissionError("unregistered_synthetic_worker_mode")
    freeze, freeze_digest = harness._load_synthetic_freeze(freeze_path)
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_freeze_sha256)
        or freeze_digest != expected_freeze_sha256
        or not re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            freeze["run_id"],
        )
        or freeze.get("production_authorization") is not False
        or freeze.get("execution_class") != "synthetic_only"
        or freeze.get("namespace") != harness.SYNTHETIC_NAMESPACE
        or freeze.get("worker_mode") != mode
    ):
        raise PermissionError("unique_seed631_synthetic_freeze_required")
    config = json.loads(config_path.read_bytes())
    if not isinstance(config, dict) or set(config) != {
        "cgroup_path",
        "output_path",
        "scratch_path",
        "result_path",
        "worker_uid",
        "worker_gid",
    }:
        raise PermissionError("seed631_config_schema_invalid")
    for key in (
        "cgroup_path",
        "output_path",
        "scratch_path",
        "result_path",
    ):
        if str(config[key]) != freeze.get(key):
            raise PermissionError("seed631_config_not_bound_by_freeze")
    if any(
        type(config[key]) is not int or config[key] != freeze.get(key)
        for key in ("worker_uid", "worker_gid")
    ):
        raise PermissionError("seed631_worker_identity_not_bound_by_freeze")

    evidence_path = Path(freeze.get("seed631_attempt_record_path", ""))
    if not evidence_path.is_absolute():
        raise PermissionError("seed631_attempt_record_path_not_frozen")
    # Reserve the evidence destination before any worker can start. If it exists,
    # this is a collision and no experiment is launched or retried.
    placeholder = {
        "schema": SCHEMA,
        "run_id": freeze["run_id"],
        "freeze_sha256": freeze_digest,
        "mode": mode,
        "status": "launching",
        "started_monotonic_ns": time.monotonic_ns(),
    }
    _attempt_record(evidence_path, placeholder, worker_uid=freeze["worker_uid"])

    lock_fd = -1
    try:
        lock_fd = _acquire_shared_slot(
            freeze["limits"]["concurrency"], worker_uid=freeze["worker_uid"]
        )
        result = harness.run_synthetic(
            freeze_path,
            config,
            mode,
            coordination_lock_fd=lock_fd,
            coordination_lock_token=harness._SEED631_COORDINATION_TOKEN,
        )
        result_path = Path(freeze["result_path"])
        result_bytes = result_path.read_bytes()
        result_record = json.loads(result_bytes)
        if (
            result_record.get("schema") != "seed630-synthetic-run-v1"
            or result_record.get("run_id") != freeze["run_id"]
            or result_record.get("freeze_sha256") != freeze_digest
            or result_record.get("worker_mode") != mode
            or result_record.get("exit_status") != result.get("exit_status")
        ):
            raise ValueError("seed631_result_binding_mismatch")
        completion = {
            "schema": "seed631-synthetic-completion-v1",
            "run_id": freeze["run_id"],
            "freeze_sha256": freeze_digest,
            "mode": mode,
            "result_path": str(result_path),
            "result_sha256": harness._digest(result_bytes),
            "completed_monotonic_ns": time.monotonic_ns(),
        }
        _attempt_record(
            evidence_path.with_name(evidence_path.name + ".success.json"),
            completion,
            worker_uid=freeze["worker_uid"],
        )
        return result
    except BaseException as error:
        # Do not retry. The original record is immutable; a second exclusive file
        # captures failure details without replacing evidence from this attempt.
        failure_path = evidence_path.with_name(evidence_path.name + ".failure.json")
        failure = {
            **placeholder,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "finished_monotonic_ns": time.monotonic_ns(),
        }
        try:
            _attempt_record(failure_path, failure, worker_uid=freeze["worker_uid"])
        except BaseException as record_error:
            error.add_note(
                "seed631_failure_record_error:"
                f"{type(record_error).__name__}:{record_error}"
            )
        raise
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("freeze", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("mode", choices=sorted(harness.MODES))
    parser.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    try:
        result = run_once(args.freeze, args.config, args.mode, args.freeze_sha256)
    except (OSError, ValueError, PermissionError) as error:
        print(f"seed631_run_blocked:{type(error).__name__}:{error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
