"""Verify copied Seed 630 evidence against a separately pinned public key."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed630_xfs_project_quota import _project_hard_limit


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify(
    evidence: Path,
    freeze_path: Path,
    pinned_public_key: Path,
    pinned_collector_sha256: str,
) -> dict[str, Any]:
    """Verify signature, raw-file hashes, freeze binding, and measured limits."""
    manifest_path = evidence / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    subprocess.run(
        [
            "openssl",
            "pkeyutl",
            "-verify",
            "-pubin",
            "-inkey",
            str(pinned_public_key),
            "-rawin",
            "-in",
            str(manifest_path),
            "-sigfile",
            str(evidence / "manifest.sig"),
        ],
        check=True,
        capture_output=True,
        timeout=15,
    )
    freeze_raw = freeze_path.read_bytes()
    freeze = json.loads(freeze_raw)
    if (
        manifest.get("schema") != "seed630-os-evidence-manifest-v1"
        or manifest.get("execution_class") != "synthetic_only"
        or manifest.get("collector_source_sha256") != pinned_collector_sha256
        or manifest.get("freeze_sha256") != _sha(freeze_raw)
        or manifest.get("synthetic_run_id") != freeze.get("run_id")
        or manifest.get("host", {}).get("uid") != 0
    ):
        raise ValueError("os_evidence_freeze_or_classification_mismatch")
    for relative, expected in manifest.get("raw_sha256", {}).items():
        name = (
            "supervisor-run-record.json"
            if relative == "supervisor/run-record"
            else Path(relative).name
        )
        raw_path = evidence / "raw" / name
        if raw_path.is_symlink() or _sha(raw_path.read_bytes()) != expected:
            raise ValueError(f"os_evidence_raw_hash_mismatch:{relative}")
    limits = freeze["limits"]
    raw = evidence / "raw"
    cpu_stat = dict(
        line.split() for line in (raw / "cpu.stat").read_text().splitlines()
    )
    memory_events = dict(
        line.split() for line in (raw / "memory.events").read_text().splitlines()
    )
    memory_max = (raw / "memory.max").read_text().strip()
    if memory_max != str(limits["memory_bytes"]):
        raise ValueError("os_memory_limit_does_not_match_freeze")
    run = json.loads((raw / "supervisor-run-record.json").read_bytes())
    mode = run.get("worker_mode")
    if mode not in {"memory", "cpu", "sleep", "write", "success", "child"}:
        raise ValueError("unknown_synthetic_experiment_classification")
    baseline_events = run.get("memory_events_baseline", {})
    event_delta = {
        key: max(0, int(value) - int(baseline_events.get(key, 0)))
        for key, value in memory_events.items()
    }
    cpu_usec = int(cpu_stat["usage_usec"]) - int(run["cpu_usage_baseline_usec"])
    cgroup_events = dict(
        line.split() for line in (raw / "cgroup.events").read_text().splitlines()
    )
    if (
        cgroup_events.get("populated") != "0"
        or (raw / "cgroup.procs").read_text().strip()
    ):
        raise ValueError("os_cgroup_process_tree_not_terminated")
    if mode == "memory" and (
        int(event_delta.get("oom_kill", 0)) == 0
        or run.get("termination_category") != "memory_oom"
    ):
        raise ValueError("memory_experiment_has_no_oom_evidence")
    if mode == "cpu" and (
        cpu_usec < limits["cpu_seconds"] * 1_000_000
        or run.get("termination_category") != "cpu_exhaustion"
    ):
        raise ValueError("cpu_experiment_did_not_reach_frozen_budget")
    if mode == "sleep" and (
        run.get("elapsed_seconds", 0) < limits["wall_seconds"]
        or run.get("termination_category") != "wall_time_expiration"
    ):
        raise ValueError("wall_experiment_did_not_reach_frozen_deadline")
    if mode == "write" and (
        run.get("termination_category") != "storage_exhaustion"
        or "No space left on device" not in run.get("worker_stderr", "")
    ):
        raise ValueError("storage_experiment_has_no_filesystem_denial")
    quota_limit = _project_hard_limit(
        (raw / "xfs-project-report").read_text(), manifest["project_id"]
    )
    if quota_limit != limits["storage_bytes"]:
        raise ValueError("os_storage_limit_does_not_match_freeze")
    if (
        run.get("run_id") != manifest["synthetic_run_id"]
        or run.get("exit_status") != manifest.get("worker_exit_status")
        or run.get("freeze_sha256") != manifest["freeze_sha256"]
    ):
        raise ValueError("signed_worker_exit_or_run_binding_mismatch")
    return {
        "status": "signed_os_records_verified",
        "os_enforcement_verified": False,
        "reason": "run evidence is not sufficient to qualify the complete five-limit matrix",
        "run_id": manifest["synthetic_run_id"],
        "freeze_sha256": manifest["freeze_sha256"],
        "experiment": mode,
        "memory_events_delta": event_delta,
        "cpu_usage_usec": cpu_usec,
        "quota_hard_limit_bytes": quota_limit,
        "worker_exit_status": manifest["worker_exit_status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("freeze", type=Path)
    parser.add_argument("pinned_public_key", type=Path)
    parser.add_argument("pinned_collector_sha256")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                args.evidence,
                args.freeze,
                args.pinned_public_key,
                args.pinned_collector_sha256,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
