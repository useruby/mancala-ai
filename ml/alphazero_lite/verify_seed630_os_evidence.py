"""Verify copied Seed 630 OS evidence against independent trust anchors."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import re
import signal
import subprocess
from pathlib import Path
from typing import Any

CGROUP_RAW_FILES = {
    "cgroup.controllers",
    "cgroup.events",
    "cgroup.procs",
    "cpu.max",
    "cpu.stat",
    "memory.current",
    "memory.events",
    "memory.max",
    "memory.oom.group",
    "memory.peak",
    "memory.stat",
    "pids.current",
    "pids.max",
}
EXPECTED_RAW_FILES = CGROUP_RAW_FILES | {
    "mountinfo",
    "proc-version",
    "xfs-project-report",
    "xfs-project-state",
    "xfs-project-inode-report",
    "xfs-nonproject-quota-state",
    "xfs-project-directory-identities.json",
    "supervisor-run-record.json",
}
EXPECTED_SOURCE_FILES = {
    "collect_seed630_os_evidence.py",
    "seed630_xfs_project_quota.py",
    "seed630_synthetic_harness.py",
    "seed629_synthetic_worker.py",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _source_bundle_digest(source_identity: dict[str, str]) -> str:
    if set(source_identity) != EXPECTED_SOURCE_FILES or any(
        not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in source_identity.values()
    ):
        raise ValueError("collector_source_identity_inventory_invalid")
    canonical = json.dumps(source_identity, sort_keys=True, separators=(",", ":"))
    return _sha(canonical.encode())


def _project_record(report: str, project_id: int) -> tuple[int, int]:
    rows = []
    for line in report.splitlines():
        fields = line.split()
        if fields and fields[0].lstrip("#") == str(project_id):
            rows.append(fields)
    if len(rows) != 1 or len(rows[0]) < 4:
        raise ValueError("xfs_project_record_missing_or_ambiguous")
    used, hard = rows[0][1], rows[0][3]
    if not re.fullmatch(r"\d+", used) or not re.fullmatch(r"\d+", hard):
        raise ValueError("xfs_project_record_units_unparseable")
    # xfs_quota(8) documents report block counts in 1K units.
    return int(used) * 1024, int(hard) * 1024


def _counter_delta(final: int, baseline: int, label: str) -> int:
    if final < baseline:
        raise ValueError(f"{label}_counter_decreased")
    return final - baseline


def _event_deltas(final: dict[str, str], baseline: dict[str, Any]) -> dict[str, int]:
    if set(final) != set(baseline):
        raise ValueError("memory_event_counter_inventory_changed")
    return {
        key: _counter_delta(int(value), int(baseline[key]), f"memory_event_{key}")
        for key, value in final.items()
    }


def _structured_worker_failure(run: dict[str, Any]) -> dict[str, Any] | None:
    try:
        rows = [
            json.loads(line)
            for line in run.get("worker_stderr", "").splitlines()
            if line.strip()
        ]
    except (AttributeError, json.JSONDecodeError):
        return None
    if len(rows) != 1:
        return None
    record = rows[0]
    if (
        record.get("schema") != "seed630-worker-failure-v1"
        or record.get("errno") != errno.EDQUOT
        or record.get("errno_name") != "EDQUOT"
        or record.get("reason") != "project_quota_exceeded"
        or run.get("exit_status") != 74
    ):
        return None
    return record


def _assert_project_enforcement(state: str) -> None:
    values: dict[str, list[str]] = {"Accounting": [], "Enforcement": []}
    headers = []
    for line in state.splitlines():
        if line.strip().startswith("Project quota state on "):
            headers.append(line.strip())
        key, separator, value = line.strip().partition(":")
        if not separator:
            continue
        if key in values:
            values[key].append(value.strip().upper())
    if (
        len(headers) != 1
        or values["Accounting"] != ["ON"]
        or values["Enforcement"] != ["ON"]
    ):
        raise ValueError("xfs_project_accounting_or_enforcement_inactive")


def _assert_other_quota_enforcement_off(state: str) -> None:
    lines = state.splitlines()
    for kind in ("User", "Group"):
        headers = [
            index
            for index, line in enumerate(lines)
            if line.strip().startswith(f"{kind} quota state on ")
        ]
        if len(headers) != 1:
            raise ValueError("xfs_other_quota_state_missing_or_ambiguous")
        start = headers[0]
        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if any(
                    lines[index].strip().startswith(f"{quota_kind} quota state on ")
                    for quota_kind in ("User", "Group", "Project")
                )
            ),
            len(lines),
        )
        accounting = [
            line.partition(":")[2].strip().upper()
            for line in lines[start:end]
            if line.strip().startswith("Accounting:")
        ]
        enforcement = [
            line.partition(":")[2].strip().upper()
            for line in lines[start:end]
            if line.strip().startswith("Enforcement:")
        ]
        if (
            len(accounting) != 1
            or accounting[0] not in {"ON", "OFF"}
            or enforcement != ["OFF"]
        ):
            raise ValueError("xfs_other_quota_enforcement_must_be_off")


def _assert_project_inode_limit_unlimited(report: str, project_id: int) -> None:
    rows = [
        line.split()
        for line in report.splitlines()
        if line.split() and line.split()[0].lstrip("#") == str(project_id)
    ]
    if len(rows) != 1 or len(rows[0]) < 4 or rows[0][3] not in {"0", "-"}:
        raise ValueError("xfs_project_inode_limit_must_be_unlimited")


def _assert_project_directory_identities(
    raw_identity: bytes, freeze: dict[str, Any], project_id: int
) -> None:
    try:
        records = json.loads(raw_identity)
    except json.JSONDecodeError as error:
        raise ValueError("xfs_project_directory_identity_record_invalid") from error
    expected = [freeze.get("output_path"), freeze.get("scratch_path")]
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("xfs_project_directory_identity_inventory_invalid")
    if [record.get("path") for record in records] != expected:
        raise ValueError("xfs_project_directory_path_not_frozen")
    for record in records:
        if set(record) != {
            "path",
            "project_id",
            "xflags",
            "uid",
            "mode",
            "fsxattr_hex",
        }:
            raise ValueError("xfs_project_directory_identity_shape_invalid")
        try:
            raw_xattr = bytes.fromhex(record["fsxattr_hex"])
        except (TypeError, ValueError) as error:
            raise ValueError("xfs_project_fsxattr_encoding_invalid") from error
        if (
            len(raw_xattr) != 28
            or int.from_bytes(raw_xattr[0:4], "little") != record["xflags"]
            or int.from_bytes(raw_xattr[12:16], "little") != record["project_id"]
            or record["project_id"] != project_id
            or record["xflags"] & 0x00000200 == 0
            or record["uid"] != 0
            or record["mode"] & 0o002
        ):
            raise ValueError("xfs_project_directory_kernel_identity_mismatch")


def _assert_quota_mount(mountinfo: str, mountpoint: str, expected_source: str) -> None:
    matches = []
    for line in mountinfo.splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        before, after = left.split(), right.split()
        if len(before) < 6 or len(after) < 3:
            continue
        observed_mount = before[4].replace("\\040", " ")
        if observed_mount == mountpoint:
            matches.append((before, after))
    if len(matches) != 1:
        raise ValueError("xfs_quota_mount_missing_or_ambiguous")
    before, after = matches[0]
    options = set(before[5].split(",")) | set(after[2].split(","))
    if (
        after[0] != "xfs"
        or after[1] != expected_source
        or "pqnoenforce" in options
        or not options.intersection({"pquota", "prjquota"})
    ):
        raise ValueError("xfs_project_quota_mount_identity_or_enforcement_mismatch")


def _verify_raw_inventory(evidence: Path, manifest: dict[str, Any]) -> dict[str, bytes]:
    if {entry.name for entry in evidence.iterdir()} != {
        "raw",
        "manifest.json",
        "manifest.sig",
    }:
        raise ValueError("signed_evidence_directory_inventory_mismatch")
    raw_directory = evidence / "raw"
    if raw_directory.is_symlink() or not raw_directory.is_dir():
        raise ValueError("signed_evidence_raw_directory_invalid")
    actual_names = {path.name for path in raw_directory.iterdir()}
    if actual_names != EXPECTED_RAW_FILES or any(
        path.is_symlink() or not path.is_file() for path in raw_directory.iterdir()
    ):
        raise ValueError("raw_os_evidence_inventory_incomplete_or_unexpected")
    entries = manifest.get("raw_files")
    if not isinstance(entries, list):
        raise ValueError("signed_raw_evidence_inventory_missing")
    paths = [entry.get("path") for entry in entries if isinstance(entry, dict)]
    expected_paths = {f"raw/{name}" for name in EXPECTED_RAW_FILES}
    if len(paths) != len(entries) or len(paths) != len(set(paths)):
        raise ValueError("signed_raw_evidence_inventory_duplicated_or_invalid")
    if set(paths) != expected_paths:
        raise ValueError("signed_raw_evidence_inventory_incomplete_or_unexpected")
    records: dict[str, bytes] = {}
    for entry in entries:
        if set(entry) != {"path", "sha256"}:
            raise ValueError("signed_raw_evidence_entry_shape_invalid")
        relative = entry["path"]
        if not re.fullmatch(r"raw/[A-Za-z0-9_.-]+", relative):
            raise ValueError("signed_raw_evidence_path_invalid")
        raw = (evidence / relative).read_bytes()
        if _sha(raw) != entry["sha256"]:
            raise ValueError(f"os_evidence_raw_hash_mismatch:{relative}")
        records[Path(relative).name] = raw
    return records


def verify(
    evidence: Path,
    freeze_path: Path,
    pinned_public_key: Path,
    pinned_collector_sha256: str,
) -> dict[str, Any]:
    """Verify signed evidence inventory, OS provenance, and one run's limits."""
    manifest_path = evidence / "manifest.json"
    signature_path = evidence / "manifest.sig"
    if manifest_path.is_symlink() or signature_path.is_symlink():
        raise ValueError("signed_evidence_control_file_symlink")
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
            str(signature_path),
        ],
        check=True,
        capture_output=True,
        timeout=15,
    )
    manifest = json.loads(manifest_path.read_bytes())
    freeze_raw = freeze_path.read_bytes()
    freeze = json.loads(freeze_raw)
    manifest_fields = {
        "schema",
        "collector_version",
        "collector_source_sha256",
        "source_identity",
        "host",
        "synthetic_run_id",
        "execution_class",
        "freeze_sha256",
        "worker_exit_status",
        "cgroup_path",
        "quota_mount",
        "quota_source",
        "project_id",
        "raw_files",
    }
    if (
        set(manifest) != manifest_fields
        or manifest.get("schema") != "seed630-os-evidence-manifest-v1"
        or manifest.get("execution_class") != "synthetic_only"
        or manifest.get("collector_source_sha256") != pinned_collector_sha256
        or manifest.get("freeze_sha256") != _sha(freeze_raw)
        or manifest.get("synthetic_run_id") != freeze.get("run_id")
        or manifest.get("host", {}).get("uid") != 0
    ):
        raise ValueError("os_evidence_freeze_provenance_or_classification_mismatch")
    if (
        _source_bundle_digest(manifest["source_identity"]) != pinned_collector_sha256
        or manifest.get("collector_source_sha256") != pinned_collector_sha256
    ):
        raise ValueError("collector_source_bundle_digest_mismatch")
    records = _verify_raw_inventory(evidence, manifest)
    required_limits = {
        "wall_seconds",
        "cpu_seconds",
        "memory_bytes",
        "concurrency",
        "storage_bytes",
    }
    limits = freeze.get("limits")
    if (
        not isinstance(limits, dict)
        or set(limits) != required_limits
        or any(type(value) is not int or value <= 0 for value in limits.values())
    ):
        raise ValueError("frozen_synthetic_limits_invalid")
    frozen_paths = [
        freeze.get("cgroup_path"),
        freeze.get("output_path"),
        freeze.get("scratch_path"),
        freeze.get("result_path"),
    ]
    if any(
        not isinstance(value, str)
        or not Path(value).is_absolute()
        or ".." in Path(value).parts
        for value in frozen_paths
    ):
        raise ValueError("frozen_synthetic_path_invalid")
    output_path = Path(freeze["output_path"])
    scratch_path = Path(freeze["scratch_path"])
    if (
        output_path == scratch_path
        or output_path in scratch_path.parents
        or scratch_path in output_path.parents
    ):
        raise ValueError("frozen_output_and_scratch_must_be_disjoint")
    result_path = Path(freeze["result_path"])
    if any(
        root == result_path or root in result_path.parents
        for root in (output_path, scratch_path)
    ):
        raise ValueError("frozen_result_path_inside_worker_artifacts")
    project_id = manifest.get("project_id")
    if (
        type(project_id) is not int
        or project_id <= 0
        or freeze.get("quota_project_id") != project_id
        or manifest.get("cgroup_path") != freeze.get("cgroup_path")
    ):
        raise ValueError("synthetic_resource_identity_not_frozen")
    mountpoint = manifest.get("quota_mount")
    _assert_quota_mount(
        records["mountinfo"].decode(), mountpoint, manifest.get("quota_source")
    )
    _assert_project_enforcement(records["xfs-project-state"].decode())
    _assert_other_quota_enforcement_off(records["xfs-nonproject-quota-state"].decode())
    used_bytes, hard_bytes = _project_record(
        records["xfs-project-report"].decode(), project_id
    )
    if hard_bytes != limits["storage_bytes"]:
        raise ValueError("os_storage_limit_does_not_match_freeze")
    _assert_project_inode_limit_unlimited(
        records["xfs-project-inode-report"].decode(), project_id
    )
    _assert_project_directory_identities(
        records["xfs-project-directory-identities.json"], freeze, project_id
    )

    run = json.loads(records["supervisor-run-record.json"])
    expected_run_quota = {
        "project_id": project_id,
        "mountpoint": mountpoint,
        "source": manifest.get("quota_source"),
        "hard_limit_bytes": hard_bytes,
    }
    observed_run_quota = run.get("quota", {})
    if any(
        observed_run_quota.get(key) != value
        for key, value in expected_run_quota.items()
    ):
        raise ValueError("signed_run_quota_identity_mismatch")
    mode = run.get("worker_mode")
    if mode not in {"memory", "cpu", "sleep", "write", "success", "child"}:
        raise ValueError("unknown_synthetic_experiment_classification")
    cpu_stat = dict(line.split() for line in records["cpu.stat"].decode().splitlines())
    memory_events = dict(
        line.split() for line in records["memory.events"].decode().splitlines()
    )
    baseline_events = run.get("memory_events_baseline", {})
    event_delta = _event_deltas(memory_events, baseline_events)
    cpu_delta = _counter_delta(
        int(cpu_stat["usage_usec"]),
        int(run["cpu_usage_baseline_usec"]),
        "cpu_usage_usec",
    )
    cgroup_events = dict(
        line.split() for line in records["cgroup.events"].decode().splitlines()
    )
    if (
        cgroup_events.get("populated") != "0"
        or records["cgroup.procs"].decode().strip()
    ):
        raise ValueError("os_cgroup_process_tree_not_terminated")
    if records["memory.max"].decode().strip() != str(limits["memory_bytes"]):
        raise ValueError("os_memory_limit_does_not_match_freeze")
    if records["memory.oom.group"].decode().strip() != "1":
        raise ValueError("os_memory_oom_group_not_enabled")
    if int(records["memory.peak"].decode().strip()) != int(run["memory_peak_bytes"]):
        raise ValueError("final_memory_peak_binding_mismatch")
    if records["cpu.max"].decode().strip() != "max 100000":
        raise ValueError("os_cpu_configuration_mismatch")
    if mode == "memory" and (
        event_delta.get("oom_kill", 0) <= 0
        or run.get("termination_category") != "memory_oom"
        or run.get("exit_status") != -signal.SIGKILL
        or "memory.oom.group" not in run.get("termination_mechanisms", [])
    ):
        raise ValueError("memory_experiment_has_no_oom_evidence")
    if mode == "cpu" and (
        cpu_delta < limits["cpu_seconds"] * 1_000_000
        or run.get("termination_category") != "cpu_exhaustion"
        or run.get("exit_status") != -signal.SIGKILL
        or "supervisor_cgroup.kill" not in run.get("termination_mechanisms", [])
    ):
        raise ValueError("cpu_experiment_did_not_reach_frozen_budget")
    if mode == "sleep" and (
        run.get("elapsed_seconds", 0) < limits["wall_seconds"]
        or run.get("termination_category") != "wall_time_expiration"
        or run.get("exit_status") != -signal.SIGKILL
        or "supervisor_cgroup.kill" not in run.get("termination_mechanisms", [])
    ):
        raise ValueError("wall_experiment_did_not_reach_frozen_deadline")
    if mode == "write":
        failure = _structured_worker_failure(run)
        if (
            not isinstance(failure, dict)
            or run.get("termination_category") != "storage_project_quota_exhaustion"
            or used_bytes <= 0
            or used_bytes > hard_bytes
            or used_bytes < hard_bytes - max(64 * 1024, hard_bytes // 100)
            or run.get("storage_failure") != failure
        ):
            raise ValueError("storage_project_quota_denial_not_proven")
    if (
        run.get("run_id") != manifest["synthetic_run_id"]
        or run.get("exit_status") != manifest.get("worker_exit_status")
        or run.get("freeze_sha256") != manifest["freeze_sha256"]
        or run.get("cgroup") != manifest.get("cgroup_path")
        or run.get("artifacts")
        != [freeze.get("output_path"), freeze.get("scratch_path")]
        or run.get("cpu_usage_usec_delta") != cpu_delta
        or run.get("memory_event_delta") != event_delta
    ):
        raise ValueError("signed_worker_exit_or_final_accounting_binding_mismatch")
    return {
        "status": "signed_os_records_verified",
        "os_enforcement_verified": False,
        "reason": "full five-limit matrix and privileged independent qualification remain required",
        "run_id": manifest["synthetic_run_id"],
        "freeze_sha256": manifest["freeze_sha256"],
        "experiment": mode,
        "memory_events_delta": event_delta,
        "cpu_usage_usec_delta": cpu_delta,
        "storage_used_bytes": used_bytes,
        "storage_hard_limit_bytes": hard_bytes,
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
