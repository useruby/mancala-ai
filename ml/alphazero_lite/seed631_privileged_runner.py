"""One-shot, synthetic-only entry point for Seed 631 privileged qualification."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed630_synthetic_harness as harness

SCHEMA = "seed631-synthetic-attempt-v1"


def _attempt_record(path: Path, record: dict[str, Any]) -> None:
    """Create a durable root-private record without replacing prior attempts."""
    if path.exists() or path.is_symlink() or not path.is_absolute():
        raise FileExistsError("seed631_attempt_record_must_be_absolute_and_create_only")
    for parent in (path.parent, *path.parent.parents):
        info = parent.stat()
        if info.st_uid != 0 or info.st_mode & 0o022:
            raise PermissionError("seed631_attempt_parent_not_root_private")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o400,
    )
    try:
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
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
        "lock_path",
        "worker_uid",
        "worker_gid",
    }:
        raise PermissionError("seed631_config_schema_invalid")
    for key in (
        "cgroup_path",
        "output_path",
        "scratch_path",
        "result_path",
        "lock_path",
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
    _attempt_record(evidence_path, placeholder)

    try:
        result = harness.run_synthetic(freeze_path, config, mode)
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
        _attempt_record(failure_path, failure)
        raise
    return result


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
