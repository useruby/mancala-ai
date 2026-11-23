"""Privileged collector for raw Seed 630 cgroup and XFS quota observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import stat
import subprocess
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed630_xfs_project_quota import (
    _fsxattr_record,
    _verify_nonproject_quotas_disabled,
    _verify_project_quota_state,
    verify_shared_project_quota,
)

VERSION = "seed630-root-os-collector-v1"
CGROUP_FILES = (
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
)
RAW_FILES = {
    **{f"cgroup/{name}": name for name in CGROUP_FILES},
    "host/mountinfo": "mountinfo",
    "host/proc-version": "proc-version",
    "filesystem/xfs-project-report": "xfs-project-report",
    "filesystem/xfs-project-state": "xfs-project-state",
    "filesystem/xfs-project-inode-report": "xfs-project-inode-report",
    "filesystem/xfs-nonproject-quota-state": "xfs-nonproject-quota-state",
    "filesystem/xfs-project-directory-identities": "xfs-project-directory-identities.json",
    "supervisor/run-record": "supervisor-run-record.json",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def collect(
    *,
    run_record_path: Path,
    freeze_path: Path,
    cgroup_path: Path,
    quota_mount: Path,
    project_id: int,
    destination: Path,
    signing_key: Path,
) -> dict[str, Any]:
    """Capture raw kernel/filesystem records and sign their manifest as root."""
    if os.geteuid() != 0:
        raise PermissionError("root_os_evidence_collector_required")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("evidence_destination_must_be_create_only")

    def require_root_readonly_file(path: Path, label: str) -> None:
        metadata = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_mode & 0o022
        ):
            raise PermissionError(
                f"{label}_must_be_root_owned_and_not_group_other_writable"
            )
        for parent in (path.parent, *path.parent.parents):
            parent_stat = parent.stat()
            if parent_stat.st_uid != 0 or parent_stat.st_mode & 0o022:
                raise PermissionError(f"{label}_parent_path_not_root_private")

    require_root_readonly_file(run_record_path, "supervisor_run_record")
    require_root_readonly_file(freeze_path, "synthetic_freeze")
    require_root_readonly_file(signing_key, "collector_signing_key")
    key_stat = signing_key.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(key_stat.st_mode)
        or key_stat.st_uid != 0
        or key_stat.st_mode & 0o077
    ):
        raise PermissionError("collector_signing_key_must_be_root_private")
    run = json.loads(run_record_path.read_bytes())
    freeze_raw = freeze_path.read_bytes()
    freeze = json.loads(freeze_raw)
    if (
        run.get("namespace") != "seed630.synthetic.v1"
        or run.get("schema") != "seed630-synthetic-run-v1"
        or run.get("freeze_sha256") != _sha(freeze_raw)
        or run.get("run_id") != freeze.get("run_id")
        or str(run_record_path) != freeze.get("result_path")
        or freeze.get("quota_project_id") != project_id
        or str(cgroup_path) != freeze.get("cgroup_path")
    ):
        raise ValueError("synthetic_run_freeze_binding_mismatch")
    if cgroup_path.is_symlink() or not cgroup_path.is_dir():
        raise PermissionError("collector_cgroup_path_invalid")
    for parent in (destination.parent, *destination.parent.parents):
        parent_stat = parent.stat()
        if parent_stat.st_uid != 0 or parent_stat.st_mode & 0o022:
            raise PermissionError("evidence_destination_parent_not_root_private")
    artifact_paths = run.get("artifacts")
    if not isinstance(artifact_paths, list) or artifact_paths != [
        freeze.get("output_path"),
        freeze.get("scratch_path"),
    ]:
        raise ValueError("synthetic_output_and_scratch_paths_missing")
    run_path = run_record_path.resolve(strict=True)
    freeze_file_path = freeze_path.resolve(strict=True)
    artifact_roots = tuple(Path(value).resolve(strict=True) for value in artifact_paths)
    if any(
        file_path == root or root in file_path.parents
        for file_path in (run_path, freeze_file_path)
        for root in artifact_roots
    ):
        raise PermissionError("signed_control_input_inside_worker_writable_artifacts")
    quota_observation = verify_shared_project_quota(
        Path(artifact_paths[0]),
        Path(artifact_paths[1]),
        expected_project_id=project_id,
        expected_hard_limit_bytes=freeze["limits"]["storage_bytes"],
    )
    if (
        run.get("quota", {}).get("project_id") != project_id
        or run.get("quota", {}).get("hard_limit_bytes")
        != quota_observation.hard_limit_bytes
        or quota_mount.resolve(strict=True) != quota_observation.mountpoint
    ):
        raise ValueError("collector_quota_binding_mismatch")
    destination.mkdir(mode=0o700, parents=False)
    raw_root = destination / "raw"
    raw_root.mkdir(mode=0o700)
    captured: dict[str, str] = {}
    try:
        for name in CGROUP_FILES:
            raw = (cgroup_path / name).read_bytes()
            (raw_root / name).write_bytes(raw)
            captured[f"cgroup/{name}"] = _sha(raw)
        mountinfo = Path("/proc/self/mountinfo").read_bytes()
        proc_version = Path("/proc/version").read_bytes()
        (raw_root / "mountinfo").write_bytes(mountinfo)
        (raw_root / "proc-version").write_bytes(proc_version)
        captured["host/mountinfo"] = _sha(mountinfo)
        captured["host/proc-version"] = _sha(proc_version)
        report = subprocess.run(
            ["xfs_quota", "-x", "-c", "report -p -n -b", str(quota_mount)],
            check=True,
            capture_output=True,
            timeout=15,
        ).stdout
        (raw_root / "xfs-project-report").write_bytes(report)
        captured["filesystem/xfs-project-report"] = _sha(report)
        state = subprocess.run(
            ["xfs_quota", "-x", "-c", "state -p", str(quota_mount)],
            check=True,
            capture_output=True,
            timeout=15,
        ).stdout
        _verify_project_quota_state(state.decode())
        (raw_root / "xfs-project-state").write_bytes(state)
        captured["filesystem/xfs-project-state"] = _sha(state)
        nonproject_state = subprocess.run(
            ["xfs_quota", "-x", "-c", "state -u -g", str(quota_mount)],
            check=True,
            capture_output=True,
            timeout=15,
        ).stdout
        _verify_nonproject_quotas_disabled(nonproject_state.decode())
        (raw_root / "xfs-nonproject-quota-state").write_bytes(nonproject_state)
        captured["filesystem/xfs-nonproject-quota-state"] = _sha(nonproject_state)
        inode_report = subprocess.run(
            ["xfs_quota", "-x", "-c", "report -p -n -i", str(quota_mount)],
            check=True,
            capture_output=True,
            timeout=15,
        ).stdout
        (raw_root / "xfs-project-inode-report").write_bytes(inode_report)
        captured["filesystem/xfs-project-inode-report"] = _sha(inode_report)
        directory_identities = []
        for supplied_path in artifact_paths:
            artifact_path = Path(supplied_path)
            xflags, observed_id, raw_fsxattr = _fsxattr_record(artifact_path)
            metadata = artifact_path.stat(follow_symlinks=False)
            directory_identities.append(
                {
                    "path": str(artifact_path.resolve(strict=True)),
                    "project_id": observed_id,
                    "xflags": xflags,
                    "uid": metadata.st_uid,
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "fsxattr_hex": raw_fsxattr.hex(),
                }
            )
        identity_raw = json.dumps(
            directory_identities, sort_keys=True, separators=(",", ":")
        ).encode()
        (raw_root / "xfs-project-directory-identities.json").write_bytes(identity_raw)
        captured["filesystem/xfs-project-directory-identities"] = _sha(identity_raw)
        result_raw = run_record_path.read_bytes()
        (raw_root / "supervisor-run-record.json").write_bytes(result_raw)
        captured["supervisor/run-record"] = _sha(result_raw)
        source_paths = (
            Path(__file__),
            Path(__file__).with_name("seed630_xfs_project_quota.py"),
            Path(__file__).with_name("seed630_synthetic_harness.py"),
            Path(__file__).with_name("seed629_synthetic_worker.py"),
        )
        source_identity = {
            source_path.name: _sha(source_path.read_bytes())
            for source_path in source_paths
        }
        source_bundle_digest = _sha(
            json.dumps(source_identity, sort_keys=True, separators=(",", ":")).encode()
        )
        manifest = {
            "schema": "seed630-os-evidence-manifest-v1",
            "collector_version": VERSION,
            "collector_source_sha256": source_bundle_digest,
            "source_identity": source_identity,
            "host": {
                "platform": platform.platform(),
                "kernel": platform.release(),
                "machine": platform.machine(),
                "uid": os.geteuid(),
            },
            "synthetic_run_id": run["run_id"],
            "execution_class": "synthetic_only",
            "freeze_sha256": _sha(freeze_raw),
            "worker_exit_status": run.get("exit_status"),
            "cgroup_path": str(cgroup_path.resolve(strict=True)),
            "quota_mount": str(quota_mount.resolve(strict=True)),
            "quota_source": quota_observation.source,
            "project_id": project_id,
            "raw_files": [
                {"path": f"raw/{RAW_FILES[key]}", "sha256": digest}
                for key, digest in sorted(captured.items())
            ],
        }
        os.chmod(raw_root, 0o500)
        manifest_raw = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode()
        manifest_path = destination / "manifest.json"
        manifest_path.write_bytes(manifest_raw)
        signature_path = destination / "manifest.sig"
        subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-sign",
                "-rawin",
                "-inkey",
                str(signing_key),
                "-in",
                str(manifest_path),
                "-out",
                str(signature_path),
            ],
            check=True,
            capture_output=True,
            timeout=15,
        )
        os.chmod(manifest_path, 0o400)
        os.chmod(signature_path, 0o400)
        return manifest
    except BaseException:
        # Preserve the partial evidence directory for diagnosis; never replace it.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_record", type=Path)
    parser.add_argument("freeze", type=Path)
    parser.add_argument("cgroup", type=Path)
    parser.add_argument("quota_mount", type=Path)
    parser.add_argument("project_id", type=int)
    parser.add_argument("destination", type=Path)
    parser.add_argument("signing_key", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            collect(
                run_record_path=args.run_record,
                freeze_path=args.freeze,
                cgroup_path=args.cgroup,
                quota_mount=args.quota_mount,
                project_id=args.project_id,
                destination=args.destination,
                signing_key=args.signing_key,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
