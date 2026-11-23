"""Read-only verification of a shared XFS project quota for synthetic runs.

Provisioning is deliberately external and privileged. This module never changes
mounts, project maps, quota limits, or directory ownership.
"""

from __future__ import annotations

import fcntl
import os
import re
import shlex
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

FSGETXATTR = 0x801C581F
FSX_PROJINHERIT = 0x00000200


@dataclass(frozen=True)
class VerifiedProjectQuota:
    mountpoint: Path
    source: str
    project_id: int
    hard_limit_bytes: int
    output_path: Path
    scratch_path: Path
    quota_report: str


def _mount_for(path: Path) -> tuple[Path, str, str]:
    resolved = path.resolve(strict=True)
    candidates: list[tuple[int, Path, str, str]] = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        before, after = left.split(), right.split()
        if len(before) < 6 or len(after) < 3:
            continue
        mountpoint = Path(before[4].replace("\\040", " ")).resolve()
        if resolved == mountpoint or mountpoint in resolved.parents:
            candidates.append((len(mountpoint.parts), mountpoint, after[0], after[1]))
    if not candidates:
        raise PermissionError("quota_mount_not_found")
    _, mountpoint, filesystem, source = max(candidates, key=lambda item: item[0])
    return mountpoint, filesystem, source


def _fsxattr(path: Path) -> tuple[int, int]:
    # struct fsxattr: five uint32 fields followed by two padding uint32 fields.
    buffer = bytearray(28)
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.ioctl(descriptor, FSGETXATTR, buffer, True)
    finally:
        os.close(descriptor)
    xflags = int.from_bytes(buffer[0:4], "little")
    project_id = int.from_bytes(buffer[12:16], "little")
    return xflags, project_id


def _project_hard_limit(report: str, project_id: int) -> int:
    """Parse xfs_quota's numeric project report; reject ambiguous output."""
    rows = []
    for line in report.splitlines():
        fields = shlex.split(line)
        if fields and fields[0].lstrip("#") == str(project_id):
            rows.append(fields)
    if len(rows) != 1:
        raise PermissionError("xfs_project_quota_record_missing_or_ambiguous")
    fields = rows[0]
    # `report -p -n -b` emits ID, used, soft, hard, warn/grace. Units are 512-byte
    # blocks on XFS; zero means unlimited and is never accepted.
    if len(fields) < 4 or not re.fullmatch(r"\d+", fields[3]):
        raise PermissionError("xfs_project_hard_limit_unparseable")
    hard_bytes = int(fields[3]) * 512
    if hard_bytes <= 0:
        raise PermissionError("xfs_project_hard_limit_unlimited")
    return hard_bytes


def verify_shared_project_quota(
    output_path: Path,
    scratch_path: Path,
    *,
    expected_project_id: int,
    expected_hard_limit_bytes: int,
) -> VerifiedProjectQuota:
    """Verify both directories share one inherited, finite kernel quota."""
    if os.geteuid() != 0:
        raise PermissionError("privileged_quota_verification_required")
    if expected_project_id <= 0 or expected_hard_limit_bytes <= 0:
        raise ValueError("positive_project_id_and_hard_limit_required")
    for supplied in (output_path, scratch_path):
        candidate = Path(os.path.abspath(supplied))
        for component in (candidate, *candidate.parents):
            if component.is_symlink():
                raise PermissionError("quota_path_symlink_rejected")
    output = output_path.resolve(strict=True)
    scratch = scratch_path.resolve(strict=True)
    if output == scratch or output in scratch.parents or scratch in output.parents:
        raise PermissionError("output_and_scratch_must_be_disjoint")
    mount_out, fs_out, source_out = _mount_for(output)
    mount_tmp, fs_tmp, source_tmp = _mount_for(scratch)
    if (
        fs_out != "xfs"
        or fs_tmp != "xfs"
        or mount_out != mount_tmp
        or source_out != source_tmp
    ):
        raise PermissionError("shared_xfs_test_filesystem_required")
    mountinfo = next(
        line
        for line in Path("/proc/self/mountinfo").read_text().splitlines()
        if line.partition(" - ")[0].split()[4].replace("\\040", " ") == str(mount_out)
    )
    left, _, right = mountinfo.partition(" - ")
    mount_options = set(left.split()[5].split(",")) | set(right.split()[2].split(","))
    if not mount_options.intersection({"prjquota", "pquota", "prjquota=1"}):
        raise PermissionError("xfs_project_quota_mount_option_missing")
    # A nested mount beneath either controlled tree could redirect writes away
    # from the project quota. Reject every mountpoint at or below either root.
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.partition(" - ")[0].split()
        if len(fields) < 5:
            continue
        nested = Path(fields[4].replace("\\040", " ")).resolve()
        controlled_paths = (output, scratch)
        if (
            (nested == mount_out or mount_out in nested.parents)
            and any(
                nested == root or nested in root.parents or root in nested.parents
                for root in controlled_paths
            )
            and nested != mount_out
        ):
            raise PermissionError("bind_or_nested_mount_near_quota_directory")
    identity: tuple[int, int] | None = None
    for path in (output, scratch):
        if path.is_symlink() or not path.is_dir():
            raise PermissionError("quota_path_must_be_real_directory")
        mode = path.stat().st_mode
        if path.stat().st_uid != 0:
            raise PermissionError("quota_directory_must_be_root_owned")
        if mode & stat.S_IWOTH:
            raise PermissionError("world_writable_quota_directory")
        xflags, project_id = _fsxattr(path)
        if not xflags & FSX_PROJINHERIT or project_id != expected_project_id:
            raise PermissionError("quota_project_identity_or_inheritance_mismatch")
        current = (project_id, xflags & FSX_PROJINHERIT)
        if identity is not None and identity != current:
            raise PermissionError("output_scratch_project_identity_mismatch")
        identity = current
    executable = shutil.which("xfs_quota")
    if not executable:
        raise PermissionError("xfs_quota_readback_tool_unavailable")
    completed = subprocess.run(
        [executable, "-x", "-c", "report -p -n -b", str(mount_out)],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    observed_limit = _project_hard_limit(completed.stdout, expected_project_id)
    if observed_limit != expected_hard_limit_bytes:
        raise PermissionError("xfs_project_hard_limit_mismatch")
    return VerifiedProjectQuota(
        mountpoint=mount_out,
        source=source_out,
        project_id=expected_project_id,
        hard_limit_bytes=observed_limit,
        output_path=output,
        scratch_path=scratch,
        quota_report=completed.stdout,
    )
