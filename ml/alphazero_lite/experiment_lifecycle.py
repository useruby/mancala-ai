"""Small worktree-based experiment lifecycle and artifact utilities.

This module handles process/artifact bookkeeping only. Scientific validation,
resource policy, and experiment-specific authorization remain with each study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "mancala-experiment-attempt-v1"
REGISTRATION_SCHEMA = "mancala-experiment-registration-v1"
WORKTREE_REGISTRATION_SCHEMA = "mancala-experiment-worktree-registration-v1"
DEFAULT_WORKTREE_ROOT = Path(
    os.environ.get("MANCALA_EXPERIMENT_WORKTREE_ROOT", ".worktrees")
)
DEFAULT_ARCHIVE_ROOT = Path.home() / "mancala-ai-experiment-archives"


class _PrelaunchAuthenticationError(RuntimeError):
    """A frozen dependency no longer matches immediately before launch."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _write_create_only(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_sha256(path: Path) -> str:
    """Hash one regular input file or a directory's sorted file inventory."""
    if path.is_symlink():
        raise ValueError(f"input_artifact_symlink_not_allowed:{path}")
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise ValueError(f"input_artifact_not_regular_file_or_directory:{path}")
    digest = hashlib.sha256()
    files = []
    for member in sorted(path.rglob("*")):
        if member.is_symlink():
            raise ValueError(f"input_artifact_symlink_not_allowed:{member}")
        if member.is_file():
            files.append(member)
        elif not member.is_dir():
            raise ValueError(f"input_artifact_special_file_not_allowed:{member}")
    for member in files:
        relative = member.relative_to(path).as_posix().encode()
        member_digest = sha256_file(member).encode()
        digest.update(relative + b"\0" + member_digest + b"\n")
    return digest.hexdigest()


def _git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
    ).stdout


def _safe_identity(value: str, label: str) -> str:
    if not value or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for char in value
    ):
        raise ValueError(f"invalid_{label}")
    if value in {".", ".."}:
        raise ValueError(f"invalid_{label}")
    return value


def registration_repository_path(experiment_id: str) -> str:
    """Return the permanent, reviewable path for one experiment registration."""
    experiment_id = _safe_identity(experiment_id, "experiment_id")
    return f"docs/experiments/{experiment_id}/registration.json"


def create_worktree(
    repo_root: Path,
    experiment_id: str,
    branch: str,
    base_commit: str,
    *,
    hypothesis: str,
    objective: str,
    experiment_type: str,
    authorization_reference: str | None = None,
    random_seeds: list[int],
    input_artifacts: dict[str, str],
    checkpoint_identity: dict[str, str] | None,
    acceptance_criteria: list[str],
    worktree_root: Path = DEFAULT_WORKTREE_ROOT,
    registration_metadata: dict[str, Any] | None = None,
) -> Path:
    """Create and register a dedicated worktree at an exact base commit."""
    experiment_id = _safe_identity(experiment_id, "experiment_id")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (branch, hypothesis, objective, experiment_type)
    ) or (
        authorization_reference is not None
        and (
            not isinstance(authorization_reference, str)
            or not authorization_reference.strip()
        )
    ):
        raise ValueError("experiment_registration_text_fields_required")
    if not isinstance(random_seeds, list) or any(
        type(seed) is not int for seed in random_seeds
    ):
        raise ValueError("registered_random_seeds_invalid")
    if any(
        not isinstance(path, str)
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        for path, digest in input_artifacts.items()
    ):
        raise ValueError("registered_input_artifact_hashes_invalid")
    if not acceptance_criteria or any(
        not isinstance(item, str) or not item for item in acceptance_criteria
    ):
        raise ValueError("registered_acceptance_criteria_required")
    repo_root = repo_root.resolve(strict=True)
    _git(repo_root, "rev-parse", "--git-dir")
    resolved_base = _git(
        repo_root, "rev-parse", "--verify", f"{base_commit}^{{commit}}"
    )
    worktree_root = (
        worktree_root if worktree_root.is_absolute() else repo_root / worktree_root
    )
    worktree_root.mkdir(parents=True, exist_ok=True)
    registry_root = worktree_root / ".experiment-registry"
    registry_root.mkdir(exist_ok=True)
    registry_path = registry_root / f"{experiment_id}.json"
    if registry_path.exists():
        raise FileExistsError("experiment_identity_already_registered")
    target = worktree_root / experiment_id
    if target.exists():
        raise FileExistsError("experiment_worktree_path_already_exists")
    registration = {
        "schema": REGISTRATION_SCHEMA,
        "experiment_id": experiment_id,
        "experiment_type": experiment_type,
        "authorization_reference": authorization_reference,
        "hypothesis": hypothesis,
        "objective": objective,
        "base_commit": resolved_base,
        "branch": branch,
        "random_seeds": random_seeds,
        "input_artifacts": input_artifacts,
        "checkpoint_identity": checkpoint_identity,
        "acceptance_criteria": acceptance_criteria,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    if registration_metadata is not None:
        if not isinstance(registration_metadata, dict) or set(
            registration_metadata
        ).intersection(registration):
            raise ValueError("experiment_registration_metadata_invalid")
        registration.update(registration_metadata)
    _write_create_only(registry_path, registration)
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "worktree",
                "add",
                "-b",
                branch,
                str(target),
                resolved_base,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        operational_metadata = target / ".experiment"
        permanent_relative = registration_repository_path(experiment_id)
        permanent_registration = target / permanent_relative
        _write_create_only(permanent_registration, registration)
        _write_create_only(
            operational_metadata / "registration.json",
            {
                "schema": WORKTREE_REGISTRATION_SCHEMA,
                "experiment_id": experiment_id,
                "registration_path": permanent_relative,
                "base_commit": resolved_base,
            },
        )
        (operational_metadata / ".gitignore").write_text(
            "/runtime/\n/registration.json\n/base-commit.json\n", encoding="utf-8"
        )
        (target / ".experiment" / "runtime" / "attempts").mkdir(
            parents=True, exist_ok=True
        )
        _write_create_only(
            target / ".experiment" / "base-commit.json", {"commit": resolved_base}
        )
    except BaseException:
        if target.exists():
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "worktree",
                    "remove",
                    "--force",
                    str(target),
                ],
                check=False,
                capture_output=True,
            )
        registry_path.unlink(missing_ok=True)
        raise
    return target


def inspect_worktrees(repo_root: Path) -> list[dict[str, Any]]:
    """List registered experiment worktrees and report their working-tree state."""
    raw = _git(repo_root, "worktree", "list", "--porcelain")
    records = []
    for stanza in raw.split("\n\n"):
        fields = dict(line.split(" ", 1) for line in stanza.splitlines() if " " in line)
        path_text = fields.get("worktree")
        if not path_text:
            continue
        path = Path(path_text)
        metadata_path = path / ".experiment" / "registration.json"
        if not metadata_path.is_file():
            continue
        registration, registration_path = load_worktree_registration(path)
        records.append(
            {
                **registration,
                "registration_path": registration_path,
                "worktree_path": str(path),
                "head": fields.get("HEAD"),
                "dirty": bool(_git(path, "status", "--porcelain")),
            }
        )
    return records


def load_worktree_registration(worktree: Path) -> tuple[dict[str, Any], str]:
    """Resolve local worktree metadata to its permanent registration file.

    Historical worktrees may still contain the old committed registration at
    `.experiment/registration.json`; new worktrees store a pointer there and
    keep the reviewable registration at `docs/experiments/<id>/registration.json`.
    """
    metadata_path = worktree / ".experiment" / "registration.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError("experiment_worktree_registration_missing")
    metadata = json.loads(metadata_path.read_bytes())
    if metadata.get("schema") == REGISTRATION_SCHEMA:
        return metadata, ".experiment/registration.json"
    if metadata.get("schema") != WORKTREE_REGISTRATION_SCHEMA:
        raise ValueError("experiment_worktree_registration_invalid")
    experiment_id = _safe_identity(metadata.get("experiment_id", ""), "experiment_id")
    registration_path = registration_repository_path(experiment_id)
    if metadata.get("registration_path") != registration_path:
        raise ValueError("experiment_worktree_registration_path_invalid")
    permanent_path = worktree / registration_path
    if permanent_path.is_symlink() or not permanent_path.is_file():
        raise ValueError("permanent_experiment_registration_missing")
    registration = json.loads(permanent_path.read_bytes())
    if (
        registration.get("schema") != REGISTRATION_SCHEMA
        or registration.get("experiment_id") != experiment_id
        or registration.get("base_commit") != metadata.get("base_commit")
    ):
        raise ValueError("permanent_experiment_registration_identity_mismatch")
    return registration, registration_path


def _validate_experiment_worktree(worktree: Path) -> dict[str, Any]:
    """Require a registered linked worktree rather than the primary checkout."""
    worktree = worktree.resolve(strict=True)
    entries = [
        Path(line.split(" ", 1)[1]).resolve()
        for line in _git(worktree, "worktree", "list", "--porcelain").splitlines()
        if line.startswith("worktree ")
    ]
    if not entries or worktree not in entries or worktree == entries[0]:
        raise ValueError("experiment_execution_requires_registered_linked_worktree")
    registration, registration_path = load_worktree_registration(worktree)
    try:
        _git(worktree, "ls-files", "--error-unmatch", registration_path)
    except subprocess.CalledProcessError as error:
        raise ValueError("permanent_registration_not_committed") from error
    return registration


def _input_identity(inputs: dict[str, str]) -> dict[str, str]:
    checked: dict[str, str] = {}
    for raw_path, expected in sorted(inputs.items()):
        supplied_path = Path(raw_path).absolute()
        if (
            supplied_path.is_symlink()
            or supplied_path.resolve(strict=True) != supplied_path
        ):
            raise ValueError(f"input_artifact_path_not_canonical:{raw_path}")
        path = supplied_path
        observed = artifact_sha256(path)
        if observed != expected:
            raise ValueError(f"input_artifact_hash_mismatch:{raw_path}")
        checked[str(path)] = observed
    return checked


def _authenticate_frozen_attempt(attempt_dir: Path, manifest: dict[str, Any]) -> None:
    """Authenticate current source, registration, IDs, and all inputs prelaunch."""
    worktree = Path(manifest["worktree_path"]).resolve(strict=True)
    registration, registration_path = load_worktree_registration(worktree)
    _validate_experiment_worktree(worktree)
    if (
        attempt_dir.resolve(strict=True)
        != (
            worktree / ".experiment" / "runtime" / "attempts" / manifest["attempt_id"]
        ).resolve()
        or attempt_dir.name != manifest["attempt_id"]
        or registration.get("experiment_id") != manifest.get("experiment_id")
        or registration.get("experiment_type") != manifest.get("experiment_type")
        or registration.get("base_commit") != manifest.get("base_commit")
        or registration.get("authorization_reference")
        != manifest.get("authorization_reference")
        or registration.get("hypothesis") != manifest.get("hypothesis")
        or registration.get("objective") != manifest.get("objective")
        or registration.get("random_seeds") != manifest.get("random_seeds")
        or registration.get("input_artifacts") is None
    ):
        raise _PrelaunchAuthenticationError("attempt_registration_identity_mismatch")
    registration_digest = sha256_file(worktree / registration_path)
    if (
        manifest.get("registration_path", registration_path) != registration_path
        or manifest.get("registration_sha256", registration_digest)
        != registration_digest
    ):
        raise _PrelaunchAuthenticationError(
            "attempt_registration_path_or_digest_mismatch"
        )
    if _git(worktree, "status", "--porcelain", "--untracked-files=all"):
        raise _PrelaunchAuthenticationError("execution_worktree_not_clean")
    current_commit = _git(worktree, "rev-parse", "HEAD")
    current_tree = _git(worktree, "rev-parse", "HEAD^{tree}")
    if current_commit != manifest.get("source_commit") or current_tree != manifest.get(
        "execution_source_identity"
    ):
        raise _PrelaunchAuthenticationError("frozen_source_commit_or_tree_changed")
    try:
        registered_inputs = _input_identity(registration["input_artifacts"])
    except (OSError, ValueError) as error:
        raise _PrelaunchAuthenticationError(
            f"registered_input_invalid:{error}"
        ) from error
    if registered_inputs != manifest.get("input_artifacts"):
        raise _PrelaunchAuthenticationError("registered_input_identity_changed")
    if manifest.get("checkpoint_identity") != registration.get("checkpoint_identity"):
        raise _PrelaunchAuthenticationError("checkpoint_registration_changed")


def _verify_historical_source(attempt_dir: Path, manifest: dict[str, Any]) -> None:
    """Verify the recorded source commit/tree in Git, independent of current HEAD."""
    worktree = attempt_dir.parents[3]
    registration, registration_path = load_worktree_registration(worktree)
    _validate_experiment_worktree(worktree)
    if (
        registration.get("experiment_id") != manifest.get("experiment_id")
        or registration.get("experiment_type") != manifest.get("experiment_type")
        or registration.get("authorization_reference")
        != manifest.get("authorization_reference")
        or registration.get("hypothesis") != manifest.get("hypothesis")
        or registration.get("objective") != manifest.get("objective")
        or registration.get("base_commit") != manifest.get("base_commit")
        or registration.get("random_seeds") != manifest.get("random_seeds")
        or registration.get("checkpoint_identity")
        != manifest.get("checkpoint_identity")
        or attempt_dir.name != manifest.get("attempt_id")
        or attempt_dir.resolve(strict=True)
        != (
            worktree
            / ".experiment"
            / "runtime"
            / "attempts"
            / manifest.get("attempt_id", "")
        ).resolve()
    ):
        raise ValueError("historical_attempt_identity_mismatch")
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40,64}", source_commit
    ):
        raise ValueError("historical_source_commit_invalid")
    _git(worktree, "cat-file", "-e", f"{source_commit}^{{commit}}")
    historical_tree = _git(worktree, "rev-parse", f"{source_commit}^{{tree}}")
    if historical_tree != manifest.get("execution_source_identity"):
        raise ValueError("historical_source_tree_identity_mismatch")
    manifested_registration_path = manifest.get("registration_path")
    if manifested_registration_path is not None:
        if manifested_registration_path != registration_path:
            raise ValueError("historical_registration_path_mismatch")
        registered_bytes = (worktree / registration_path).read_bytes()
        if (
            _git_bytes(
                worktree, "show", f"{source_commit}:{manifested_registration_path}"
            )
            != registered_bytes
            or manifest.get("registration_sha256")
            != hashlib.sha256(registered_bytes).hexdigest()
        ):
            raise ValueError("historical_permanent_registration_mismatch")
    if subprocess.run(
        [
            "git",
            "-C",
            str(worktree),
            "merge-base",
            "--is-ancestor",
            registration["base_commit"],
            source_commit,
        ],
        capture_output=True,
    ).returncode:
        raise ValueError("historical_source_not_descended_from_registered_base")
    registered_inputs = {
        str(Path(path).absolute()): digest
        for path, digest in registration["input_artifacts"].items()
    }
    if registered_inputs != manifest.get("input_artifacts"):
        raise ValueError("historical_input_registration_mismatch")


def freeze_attempt(
    worktree: Path,
    *,
    attempt_id: str,
    command: list[str],
    configuration: dict[str, Any],
    runtime_versions: dict[str, str],
    device: dict[str, Any],
    worker_count: int,
    expected_runtime_seconds: int,
    wall_time_limit_seconds: int,
    minimum_free_disk_bytes: int,
    outputs: list[str],
    input_artifacts: dict[str, str],
    random_seeds: list[int],
    additional_resource_limits: dict[str, int] | None = None,
    publication_outputs: list[str] | None = None,
    environment: dict[str, str] | None = None,
) -> Path:
    """Freeze attempt identity, exact argv/config, source, inputs, and budgets."""
    attempt_id = _safe_identity(attempt_id, "attempt_id")
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("command_must_be_nonempty_argv")
    if (
        not isinstance(configuration, dict)
        or not isinstance(runtime_versions, dict)
        or not isinstance(device, dict)
    ):
        raise ValueError("attempt_configuration_runtime_and_device_must_be_objects")
    if environment is not None and any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ValueError("attempt_environment_must_map_strings_to_strings")
    if len(set(outputs)) != len(outputs):
        raise ValueError("attempt_outputs_must_be_unique")
    for output in outputs:
        _safe_member_path(output)
    publication_outputs = publication_outputs or []
    if len(set(publication_outputs)) != len(publication_outputs) or not set(
        publication_outputs
    ).issubset(outputs):
        raise ValueError("publication_outputs_must_be_unique_declared_outputs")
    if (
        worker_count < 1
        or min(
            expected_runtime_seconds, wall_time_limit_seconds, minimum_free_disk_bytes
        )
        < 0
    ):
        raise ValueError("invalid_resource_budget")
    additional_resource_limits = additional_resource_limits or {}
    if any(
        type(value) is not int or value < 0
        for value in additional_resource_limits.values()
    ):
        raise ValueError("additional_resource_limits_invalid")
    status = _git(worktree, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ValueError("source_worktree_must_be_committed_before_freeze")
    registration, registration_path = load_worktree_registration(worktree)
    validated_registration = _validate_experiment_worktree(worktree)
    if validated_registration["experiment_id"] != registration["experiment_id"]:
        raise ValueError("experiment_worktree_registration_mismatch")
    if (
        input_artifacts != registration["input_artifacts"]
        or random_seeds != registration["random_seeds"]
    ):
        raise ValueError("attempt_inputs_or_seeds_differ_from_registration")
    attempt_dir = worktree / ".experiment" / "runtime" / "attempts" / attempt_id
    if attempt_dir.exists():
        raise FileExistsError("attempt_identity_already_exists")
    checked_inputs = _input_identity(registration["input_artifacts"])
    checkpoint = registration.get("checkpoint_identity")
    if isinstance(checkpoint, dict) and {"path", "sha256"}.issubset(checkpoint):
        if (
            checked_inputs.get(str(Path(checkpoint["path"]).absolute()))
            != checkpoint["sha256"]
        ):
            raise ValueError("checkpoint_identity_not_bound_by_input_hash")
    source_commit = _git(worktree, "rev-parse", "HEAD")
    manifest = {
        "schema": SCHEMA,
        "experiment_id": registration["experiment_id"],
        "attempt_id": attempt_id,
        "experiment_type": registration["experiment_type"],
        "authorization_reference": registration.get("authorization_reference"),
        "registration_path": registration_path,
        "registration_sha256": sha256_file(worktree / registration_path),
        "hypothesis": registration["hypothesis"],
        "objective": registration["objective"],
        "source_commit": source_commit,
        "execution_source_identity": _git(worktree, "rev-parse", "HEAD^{tree}"),
        "worktree_path": str(worktree.resolve(strict=True)),
        "base_commit": registration["base_commit"],
        "input_artifacts": checked_inputs,
        "checkpoint_identity": registration["checkpoint_identity"],
        "runtime_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            **runtime_versions,
        },
        "command": command,
        "environment": {
            key: os.environ[key]
            for key in ("PATH", "HOME", "LANG", "LC_ALL")
            if key in os.environ
        }
        | (environment or {}),
        "configuration": configuration,
        "random_seeds": random_seeds,
        "device": device,
        "worker_count": worker_count,
        "resource_budget": {
            "expected_runtime_seconds": expected_runtime_seconds,
            "wall_time_limit_seconds": wall_time_limit_seconds,
            "minimum_free_disk_bytes": minimum_free_disk_bytes,
            "additional_limits": additional_resource_limits,
        },
        "acceptance_criteria": registration["acceptance_criteria"],
        "output_paths": outputs,
        "publication_outputs": publication_outputs,
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    attempt_dir.mkdir(parents=True)
    try:
        _write_create_only(attempt_dir / "manifest.json", manifest)
        (attempt_dir / "manifest.json").chmod(0o444)
    except BaseException:
        attempt_dir.rmdir()
        raise
    return attempt_dir


def _process_usage() -> dict[str, float | int]:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "max_rss_platform_units": usage.ru_maxrss,
    }


def _launch_command(
    command: list[str],
    *,
    cwd: Path,
    stdout: Any,
    stderr: Any,
    environment: dict[str, str],
) -> subprocess.Popen[bytes]:
    """Start one frozen command in a new process group."""
    return subprocess.Popen(
        command,
        cwd=cwd,
        stdout=stdout,
        stderr=stderr,
        env=environment,
        start_new_session=True,
    )


def _stop_process_group(
    process: subprocess.Popen[bytes], grace_seconds: float = 3.0
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        process.poll()
        time.sleep(0.02)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.wait()


def _stop_orphaned_group(process_group_id: int, grace_seconds: float = 1.0) -> None:
    """Terminate any descendants that outlive a successfully exited leader."""
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass


def execute_attempt(
    attempt_dir: Path,
    *,
    expected_manifest_sha256: str,
    minimum_free_disk_bytes: int | None = None,
) -> dict[str, Any]:
    """Run the frozen argv without a shell; persist logs and terminal status."""
    manifest_path = attempt_dir / "manifest.json"
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256)
        or sha256_file(manifest_path) != expected_manifest_sha256
    ):
        raise ValueError("attempt_freeze_digest_mismatch")
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "frozen":
        raise ValueError("attempt_manifest_not_frozen")
    if manifest.get("worktree_path") != str(attempt_dir.parents[3]):
        raise ValueError("attempt_worktree_path_binding_mismatch")
    _validate_experiment_worktree(Path(manifest["worktree_path"]))
    required_disk = (
        minimum_free_disk_bytes
        if minimum_free_disk_bytes is not None
        else manifest["resource_budget"]["minimum_free_disk_bytes"]
    )
    if required_disk < manifest["resource_budget"]["minimum_free_disk_bytes"]:
        raise ValueError("runtime_disk_budget_cannot_be_lowered_below_freeze")
    started_path = attempt_dir / "started.json"
    _write_create_only(
        started_path,
        {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns(),
        },
    )
    free_space = shutil.disk_usage(attempt_dir).free
    stdout_path, stderr_path = attempt_dir / "stdout.log", attempt_dir / "stderr.log"
    start = time.monotonic()
    usage_before = _process_usage()
    exit_status: int | None = None
    status = "failed"
    reason: str | None = None
    with (
        stdout_path.open("xb") as stdout_stream,
        stderr_path.open("xb") as stderr_stream,
    ):
        if free_space < required_disk:
            reason = f"disk_space_preflight_failed:{free_space}<{required_disk}"
        else:
            process: subprocess.Popen[bytes] | None = None
            try:
                try:
                    _authenticate_frozen_attempt(attempt_dir, manifest)
                except _PrelaunchAuthenticationError:
                    raise
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    subprocess.CalledProcessError,
                ) as error:
                    raise _PrelaunchAuthenticationError(str(error)) from error
                process = _launch_command(
                    manifest["command"],
                    cwd=Path(manifest["worktree_path"]),
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    environment=manifest["environment"],
                )
                exit_status = process.wait(
                    timeout=manifest["resource_budget"]["wall_time_limit_seconds"]
                )
                _stop_orphaned_group(process.pid)
                status = "completed" if exit_status == 0 else "failed"
                if exit_status != 0:
                    reason = "nonzero_exit_status"
            except subprocess.TimeoutExpired:
                reason = "wall_time_limit_exceeded"
                if process is not None:
                    _stop_process_group(process)
                    exit_status = process.returncode
            except KeyboardInterrupt:
                reason = "execution_interrupted"
                if process is not None:
                    _stop_process_group(process)
                    exit_status = process.returncode
                status = "interrupted"
            except _PrelaunchAuthenticationError as error:
                reason = f"prelaunch_authentication_failed:{error}"
                status = "failed"
            except BaseException as error:
                if process is not None:
                    _stop_process_group(process)
                reason = (
                    f"worker_launch_or_execution_error:{type(error).__name__}:{error}"
                )
                status = "failed"
    output_hashes: dict[str, str] = {}
    if status == "completed":
        for relative in manifest["output_paths"]:
            safe_relative = _safe_member_path(relative)
            output_path = attempt_dir.joinpath(*safe_relative.parts)
            if output_path.is_symlink() or not output_path.exists():
                status = "failed"
                reason = f"declared_output_missing:{relative}"
                break
            try:
                output_hashes[relative] = artifact_sha256(output_path)
            except (OSError, ValueError) as error:
                status = "failed"
                reason = f"declared_output_invalid:{relative}:{error}"
                break
    finished = time.monotonic()
    usage_after = _process_usage()
    usage_delta = {
        "user_cpu_seconds": usage_after["user_cpu_seconds"]
        - usage_before["user_cpu_seconds"],
        "system_cpu_seconds": usage_after["system_cpu_seconds"]
        - usage_before["system_cpu_seconds"],
        "max_rss_platform_units": usage_after["max_rss_platform_units"],
        "max_rss_scope": "supervisor_children_high_water_including_prior_children",
    }
    result = {
        "schema": "mancala-experiment-attempt-result-v1",
        "experiment_id": manifest["experiment_id"],
        "attempt_id": manifest["attempt_id"],
        "source_commit": manifest["source_commit"],
        "manifest_sha256": sha256_file(manifest_path),
        "status": status,
        "reason": reason,
        "exit_status": exit_status,
        "started_at": json.loads(started_path.read_bytes())["started_at"],
        "started_sha256": sha256_file(started_path),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": finished - start,
        "free_disk_bytes_before": free_space,
        "resource_usage_before": usage_before,
        "resource_usage_after": usage_after,
        "resource_usage_delta": usage_delta,
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
        "output_sha256": output_hashes,
    }
    _write_create_only(attempt_dir / f"{status}.json", result)
    return result


def verify_attempt(
    attempt_dir: Path, *, allow_changed_inputs_for_failed_attempt: bool = False
) -> dict[str, Any]:
    """Recheck frozen input, source, log, output, and terminal-record identities."""
    manifest_path = attempt_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    terminal_paths = [
        attempt_dir / f"{status}.json"
        for status in ("completed", "failed", "interrupted")
    ]
    terminals = [path for path in terminal_paths if path.is_file()]
    if len(terminals) != 1:
        raise ValueError("attempt_terminal_record_missing_or_ambiguous")
    terminal = json.loads(terminals[0].read_bytes())
    if (
        manifest.get("schema") != SCHEMA
        or terminal.get("schema") != "mancala-experiment-attempt-result-v1"
        or terminal.get("manifest_sha256") != sha256_file(manifest_path)
        or terminal.get("experiment_id") != manifest.get("experiment_id")
        or terminal.get("attempt_id") != manifest.get("attempt_id")
        or terminal.get("source_commit") != manifest.get("source_commit")
        or terminal.get("status") != terminals[0].stem
    ):
        raise ValueError("attempt_terminal_binding_mismatch")
    _verify_historical_source(attempt_dir, manifest)
    started_path = attempt_dir / "started.json"
    if (
        not started_path.is_file()
        or sha256_file(started_path) != terminal.get("started_sha256")
        or json.loads(started_path.read_bytes()).get("started_at")
        != terminal.get("started_at")
    ):
        raise ValueError("attempt_start_record_binding_mismatch")
    input_mismatches = []
    for path, expected_hash in manifest["input_artifacts"].items():
        try:
            observed_hash = artifact_sha256(Path(path))
        except (OSError, ValueError):
            observed_hash = None
        if observed_hash != expected_hash:
            input_mismatches.append(path)
    if input_mismatches and not (
        allow_changed_inputs_for_failed_attempt and terminal["status"] == "failed"
    ):
        raise ValueError(f"input_artifact_hash_mismatch:{input_mismatches[0]}")
    if sha256_file(attempt_dir / "stdout.log") != terminal["stdout_sha256"]:
        raise ValueError("attempt_stdout_hash_mismatch")
    if sha256_file(attempt_dir / "stderr.log") != terminal["stderr_sha256"]:
        raise ValueError("attempt_stderr_hash_mismatch")
    for relative, expected_hash in terminal.get("output_sha256", {}).items():
        target = attempt_dir.joinpath(*_safe_member_path(relative).parts)
        if (
            target.is_symlink()
            or not target.exists()
            or artifact_sha256(target) != expected_hash
        ):
            raise ValueError(f"attempt_output_hash_mismatch:{relative}")
    if terminal["status"] == "completed" and set(
        terminal.get("output_sha256", {})
    ) != set(manifest["output_paths"]):
        raise ValueError("attempt_declared_output_inventory_mismatch")
    return {
        "verified": not input_mismatches,
        "integrity_verified": True,
        "experiment_id": manifest["experiment_id"],
        "attempt_id": manifest["attempt_id"],
        "status": terminal["status"],
        "manifest_sha256": sha256_file(manifest_path),
        "outputs_verified": len(terminal.get("output_sha256", {})),
        "input_artifacts_verified": not input_mismatches,
        "input_artifact_mismatches": input_mismatches,
    }


def attempt_inventory(attempt_dir: Path) -> list[dict[str, Any]]:
    """Inventory regular attempt files using relative paths, sizes, and hashes."""
    files = []
    for path in sorted(attempt_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact_symlink_not_archivable:{path}")
        if (
            path.is_file()
            and path.relative_to(attempt_dir).as_posix() != "archive-reference.json"
        ):
            files.append(
                {
                    "path": path.relative_to(attempt_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return files


def _artifact_inventory_digest(files: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_json_bytes(files)).hexdigest()


def _safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("unsafe_archive_member_path")
    return path


def _write_tar(attempt_dir: Path, files: list[dict[str, Any]], output: Path) -> str:
    if shutil.which("zstd"):
        with tempfile.NamedTemporaryFile(
            dir=output.parent, suffix=".tar", delete=False
        ) as stream:
            tar_path = Path(stream.name)
        try:
            with tarfile.open(tar_path, "w") as archive:
                for item in files:
                    archive.add(
                        attempt_dir / item["path"],
                        arcname=item["path"],
                        recursive=False,
                    )
            subprocess.run(
                ["zstd", "-q", "-f", str(tar_path), "-o", str(output)],
                check=True,
                capture_output=True,
            )
        finally:
            tar_path.unlink(missing_ok=True)
        return "tar.zst"
    with tarfile.open(output, "w:gz") as archive:
        for item in files:
            archive.add(
                attempt_dir / item["path"], arcname=item["path"], recursive=False
            )
    return "tar.gz"


def _verify_archive_contents(
    archive_path: Path, manifest: dict[str, Any], temporary_root: Path
) -> None:
    with tempfile.NamedTemporaryFile(
        dir=temporary_root, suffix=".tar", delete=False
    ) as stream:
        tar_path = Path(stream.name)
    try:
        if archive_path.name.endswith(".tar.zst"):
            subprocess.run(
                ["zstd", "-q", "-d", "-f", str(archive_path), "-o", str(tar_path)],
                check=True,
                capture_output=True,
            )
        else:
            with (
                tarfile.open(archive_path, "r:gz") as source,
                tarfile.open(tar_path, "w") as target,
            ):
                for member in source.getmembers():
                    _safe_member_path(member.name)
                    if not member.isfile():
                        raise ValueError("archive_contains_nonregular_member")
                    data = source.extractfile(member)
                    if data is None:
                        raise ValueError("archive_member_unreadable")
                    target.addfile(member, data)
        expected = {item["path"]: item for item in manifest["files"]}
        with tarfile.open(tar_path, "r") as archive:
            members = archive.getmembers()
            if {member.name for member in members} != set(expected):
                raise ValueError("archive_member_inventory_mismatch")
            for member in members:
                _safe_member_path(member.name)
                if not member.isfile():
                    raise ValueError("archive_contains_nonregular_member")
                target = temporary_root.joinpath(*PurePosixPath(member.name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("archive_member_unreadable")
                with target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
                if (
                    target.stat().st_size != expected[member.name]["size_bytes"]
                    or sha256_file(target) != expected[member.name]["sha256"]
                ):
                    raise ValueError(f"archive_member_hash_mismatch:{member.name}")
    finally:
        tar_path.unlink(missing_ok=True)


def archive_attempt(attempt_dir: Path, archive_root: Path) -> dict[str, Any]:
    """Create a create-only compressed archive and verify archive plus extraction."""
    manifest_path = attempt_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    attempt_verification = verify_attempt(
        attempt_dir, allow_changed_inputs_for_failed_attempt=True
    )
    archive_root.mkdir(parents=True, exist_ok=True)
    files = attempt_inventory(attempt_dir)
    stem = f"{manifest['experiment_id']}--{manifest['attempt_id']}"
    extension = ".tar.zst" if shutil.which("zstd") else ".tar.gz"
    final_path = archive_root / f"{stem}{extension}"
    sidecar = archive_root / f"{stem}.archive-manifest.json"
    reference_path = attempt_dir / "archive-reference.json"
    if final_path.exists() or sidecar.exists() or reference_path.exists():
        raise FileExistsError("archive_publication_is_create_only")
    with tempfile.NamedTemporaryFile(
        dir=archive_root, prefix=f".{stem}.", suffix=extension, delete=False
    ) as stream:
        staged_path = Path(stream.name)
    staged_path.unlink()
    try:
        archive_format = _write_tar(attempt_dir, files, staged_path)
        with tempfile.TemporaryDirectory(
            dir=archive_root, prefix=f".{stem}.verify-"
        ) as verify_dir:
            archive_manifest = {
                "schema": "mancala-experiment-archive-v1",
                "experiment_id": manifest["experiment_id"],
                "attempt_id": manifest["attempt_id"],
                "source_commit": manifest["source_commit"],
                "manifest_sha256": sha256_file(manifest_path),
                "attempt_verification": attempt_verification,
                "required_artifact_inventory_sha256": _artifact_inventory_digest(files),
                "archive_format": archive_format,
                "archive_size_bytes": staged_path.stat().st_size,
                "archive_sha256": sha256_file(staged_path),
                "files": files,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _verify_archive_contents(staged_path, archive_manifest, Path(verify_dir))
        os.link(staged_path, final_path)
        directory_fd = os.open(archive_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        staged_path.unlink()
        _write_create_only(sidecar, archive_manifest)
        reference = {
            "schema": "mancala-experiment-archive-reference-v1",
            "experiment_id": manifest["experiment_id"],
            "attempt_id": manifest["attempt_id"],
            "source_commit": manifest["source_commit"],
            "attempt_manifest_sha256": archive_manifest["manifest_sha256"],
            "required_artifact_inventory_sha256": archive_manifest[
                "required_artifact_inventory_sha256"
            ],
            "archive_path": str(final_path.resolve()),
            "archive_manifest_path": str(sidecar.resolve()),
            "archive_sha256": archive_manifest["archive_sha256"],
            "archive_manifest_sha256": sha256_file(sidecar),
            "verified": True,
        }
        _write_create_only(reference_path, reference)
        return reference
    finally:
        staged_path.unlink(missing_ok=True)


def verify_archive(archive_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify outer archive digest/size and safely extract/hash all members."""
    if archive_path.is_symlink() or manifest_path.is_symlink():
        raise ValueError("archive_or_manifest_symlink_not_allowed")
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("schema") != "mancala-experiment-archive-v1" or not isinstance(
        manifest.get("files"), list
    ):
        raise ValueError("archive_manifest_schema_invalid")
    if (
        not isinstance(manifest.get("experiment_id"), str)
        or not isinstance(manifest.get("attempt_id"), str)
        or not isinstance(manifest.get("source_commit"), str)
        or not isinstance(manifest.get("manifest_sha256"), str)
        or not isinstance(manifest.get("required_artifact_inventory_sha256"), str)
        or not isinstance(manifest.get("archive_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{40,64}", manifest.get("source_commit", ""))
        or not re.fullmatch(r"[0-9a-f]{64}", manifest.get("manifest_sha256", ""))
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            manifest.get("required_artifact_inventory_sha256", ""),
        )
        or not re.fullmatch(r"[0-9a-f]{64}", manifest.get("archive_sha256", ""))
        or type(manifest.get("archive_size_bytes")) is not int
        or manifest["archive_size_bytes"] < 0
    ):
        raise ValueError("archive_manifest_identity_invalid")
    archive_files = manifest["files"]
    if any(
        not isinstance(item, dict)
        or set(item) != {"path", "size_bytes", "sha256"}
        or not isinstance(item.get("path"), str)
        or not isinstance(item.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", ""))
        or type(item.get("size_bytes")) is not int
        or item["size_bytes"] < 0
        for item in archive_files
    ) or len({item["path"] for item in archive_files}) != len(archive_files):
        raise ValueError("archive_manifest_file_inventory_invalid")
    for item in archive_files:
        _safe_member_path(item["path"])
    if manifest.get("required_artifact_inventory_sha256") != _artifact_inventory_digest(
        manifest["files"]
    ):
        raise ValueError("archive_manifest_artifact_inventory_digest_mismatch")
    expected_format = "tar.zst" if archive_path.name.endswith(".tar.zst") else "tar.gz"
    if manifest.get("archive_format") != expected_format:
        raise ValueError("archive_manifest_format_mismatch")
    if (
        archive_path.stat().st_size != manifest["archive_size_bytes"]
        or sha256_file(archive_path) != manifest["archive_sha256"]
    ):
        raise ValueError("archive_digest_or_size_mismatch")
    with tempfile.TemporaryDirectory(
        dir=archive_path.parent, prefix=".mancala-archive-verify-"
    ) as temporary:
        _verify_archive_contents(archive_path, manifest, Path(temporary))
    return {
        "verified": True,
        "experiment_id": manifest["experiment_id"],
        "attempt_id": manifest["attempt_id"],
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "required_artifact_inventory_sha256": manifest[
            "required_artifact_inventory_sha256"
        ],
        "archive_sha256": manifest["archive_sha256"],
        "files_verified": len(manifest["files"]),
        "files": manifest["files"],
    }


def prepare_publication(attempt_dir: Path, destination: Path) -> dict[str, Any]:
    """Copy selected records, results, logs, and archive receipt into a PR bundle."""
    if destination.exists():
        raise FileExistsError("publication_directory_must_be_create_only")
    verification = verify_attempt(
        attempt_dir, allow_changed_inputs_for_failed_attempt=True
    )
    destination.mkdir(parents=True)
    manifest = json.loads((attempt_dir / "manifest.json").read_bytes())
    terminal = attempt_dir / f"{verification['status']}.json"
    files = [
        attempt_dir / "manifest.json",
        terminal,
    ]
    files.extend(attempt_dir / output for output in manifest["publication_outputs"])
    files.extend(
        path
        for path in (attempt_dir / "stdout.log", attempt_dir / "stderr.log")
        if path.stat().st_size <= 1024 * 1024
    )
    archive_reference = attempt_dir / "archive-reference.json"
    if archive_reference.is_file():
        files.append(archive_reference)
        reference = json.loads(archive_reference.read_bytes())
        sidecar_path = Path(reference["archive_manifest_path"])
        verify_archive(Path(reference["archive_path"]), sidecar_path)
        files.append(sidecar_path)
    inventory = []
    for source in files:
        if (
            source.is_symlink()
            or not source.exists()
            or not (source.is_file() or source.is_dir())
        ):
            raise ValueError(f"publication_file_missing_or_unsafe:{source}")
        target = destination / source.name
        if target.exists():
            raise FileExistsError("publication_filename_collision")
        if source.is_dir():
            shutil.copytree(source, target, symlinks=True)
            for member in sorted(target.rglob("*")):
                if member.is_symlink() or not member.is_file():
                    if member.is_symlink():
                        raise ValueError(f"publication_symlink_not_allowed:{member}")
                    continue
                inventory.append(
                    {
                        "path": member.relative_to(destination).as_posix(),
                        "size_bytes": member.stat().st_size,
                        "sha256": sha256_file(member),
                    }
                )
        else:
            shutil.copy2(source, target)
            inventory.append(
                {
                    "path": target.name,
                    "size_bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                }
            )
    publication = {
        "schema": "mancala-experiment-publication-v1",
        "experiment_id": manifest["experiment_id"],
        "attempt_id": manifest["attempt_id"],
        "source_commit": manifest["source_commit"],
        "attempt_verification": verification,
        "files": inventory,
    }
    _write_create_only(destination / "publication-manifest.json", publication)
    return publication


def safe_remove_worktree(repo_root: Path, worktree: Path, *, merged_into: str) -> None:
    """Remove only a clean, merged worktree whose required attempts are archived."""
    repo_root = repo_root.resolve(strict=True)
    worktree = worktree.resolve(strict=True)
    if worktree == repo_root:
        raise ValueError("cannot_remove_primary_repository_worktree")
    registration, _registration_path = load_worktree_registration(worktree)
    registered_worktrees = {
        Path(line.split(" ", 1)[1]).resolve()
        for line in _git(repo_root, "worktree", "list", "--porcelain").splitlines()
        if line.startswith("worktree ")
    }
    if worktree not in registered_worktrees:
        raise ValueError("path_is_not_a_registered_git_worktree")
    status = _git(worktree, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ValueError("worktree_has_unpublished_changes")
    if subprocess.run(
        [
            "git",
            "-C",
            str(worktree),
            "merge-base",
            "--is-ancestor",
            "HEAD",
            merged_into,
        ],
        capture_output=True,
    ).returncode:
        raise ValueError("worktree_commit_not_merged")
    runtime_root = worktree / ".experiment" / "runtime"
    attempts_root = runtime_root / "attempts"
    if runtime_root.is_symlink() or not runtime_root.is_dir():
        raise ValueError("experiment_runtime_directory_invalid")
    for runtime_entry in runtime_root.iterdir():
        if runtime_entry != attempts_root:
            raise ValueError("unaccounted_runtime_artifact_outside_attempt_directory")
    if attempts_root.is_symlink() or not attempts_root.is_dir():
        raise ValueError("experiment_attempts_directory_invalid")
    for attempt_dir in sorted(attempts_root.iterdir()):
        if attempt_dir.is_symlink() or not attempt_dir.is_dir():
            raise ValueError("unaccounted_runtime_artifact_outside_attempt_directory")
        try:
            _safe_identity(attempt_dir.name, "attempt_id")
        except ValueError as error:
            raise ValueError("attempt_directory_identity_invalid") from error
        manifest_path = attempt_dir / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("attempt_manifest_missing_or_unsafe")
        frozen_attempt = json.loads(manifest_path.read_bytes())
        if (
            frozen_attempt.get("schema") != SCHEMA
            or frozen_attempt.get("status") != "frozen"
            or frozen_attempt.get("experiment_id") != registration["experiment_id"]
            or frozen_attempt.get("attempt_id") != attempt_dir.name
            or frozen_attempt.get("worktree_path") != str(worktree)
        ):
            raise ValueError("attempt_directory_identity_invalid")
        _verify_historical_source(attempt_dir, frozen_attempt)
        started_path = attempt_dir / "started.json"
        log_paths = (attempt_dir / "stdout.log", attempt_dir / "stderr.log")
        terminal_paths = [
            attempt_dir / f"{name}.json"
            for name in ("completed", "failed", "interrupted")
        ]
        terminals = [path for path in terminal_paths if path.exists()]
        logs_present = [path.exists() for path in log_paths]
        if started_path.exists():
            if (
                started_path.is_symlink()
                or len(terminals) != 1
                or not all(logs_present)
            ):
                raise ValueError("attempt_lifecycle_records_inconsistent")
            if any(path.is_symlink() for path in (*log_paths, *terminals)):
                raise ValueError("attempt_lifecycle_records_inconsistent")
            attempt_verification = verify_attempt(
                attempt_dir, allow_changed_inputs_for_failed_attempt=True
            )
            if attempt_verification["experiment_id"] != registration["experiment_id"]:
                raise ValueError("attempt_experiment_identity_mismatch")
        elif terminals or any(logs_present):
            raise ValueError("attempt_lifecycle_records_inconsistent")
        else:
            raise ValueError("attempt_has_no_execution_or_terminal_record")
        reference_path = attempt_dir / "archive-reference.json"
        if not reference_path.is_file():
            raise ValueError("required_attempt_artifacts_not_archived")
        if reference_path.is_symlink():
            raise ValueError("archive_reference_symlink_not_allowed")
        reference = json.loads(reference_path.read_bytes())
        verified = verify_archive(
            Path(reference["archive_path"]), Path(reference["archive_manifest_path"])
        )
        current_inventory = attempt_inventory(attempt_dir)
        if (
            verified["experiment_id"] != reference["experiment_id"]
            or verified["attempt_id"] != reference["attempt_id"]
            or verified["source_commit"] != reference["source_commit"]
            or reference["experiment_id"] != registration["experiment_id"]
            or reference["attempt_id"] != frozen_attempt["attempt_id"]
            or reference["source_commit"] != frozen_attempt["source_commit"]
            or reference["attempt_manifest_sha256"] != sha256_file(manifest_path)
            or verified["manifest_sha256"] != sha256_file(manifest_path)
            or verified["files"] != current_inventory
            or reference["required_artifact_inventory_sha256"]
            != _artifact_inventory_digest(current_inventory)
            or verified["required_artifact_inventory_sha256"]
            != _artifact_inventory_digest(current_inventory)
            or verified["archive_sha256"] != reference["archive_sha256"]
            or sha256_file(Path(reference["archive_manifest_path"]))
            != reference["archive_manifest_sha256"]
        ):
            raise ValueError("archive_reference_identity_mismatch")
        tracked_paths = _git(worktree, "ls-files", "-z").split("\0")
        published_references = []
        published_manifests = []
        for relative in tracked_paths:
            if not relative:
                continue
            tracked_path = worktree / relative
            if tracked_path.name == "archive-reference.json":
                if tracked_path.is_symlink():
                    raise ValueError("published_archive_reference_symlink")
                try:
                    published_references.append(json.loads(tracked_path.read_bytes()))
                except (OSError, json.JSONDecodeError):
                    continue
            elif tracked_path.name.endswith(".archive-manifest.json"):
                if tracked_path.is_symlink():
                    raise ValueError("published_archive_manifest_symlink")
                if sha256_file(tracked_path) == reference["archive_manifest_sha256"]:
                    published_manifests.append(json.loads(tracked_path.read_bytes()))
        if (
            reference not in published_references
            or len(published_manifests) != 1
            or published_manifests[0].get("experiment_id")
            != registration["experiment_id"]
            or published_manifests[0].get("attempt_id") != frozen_attempt["attempt_id"]
            or published_manifests[0].get("source_commit")
            != frozen_attempt["source_commit"]
        ):
            raise ValueError("archive_publication_anchor_missing_or_substituted")
    subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "remove", str(worktree)],
        check=True,
        capture_output=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--repo", type=Path, default=Path.cwd())
    create.add_argument("--registration", type=Path, required=True)
    create.add_argument("--branch", required=True)
    create.add_argument("--base-commit", required=True)
    create.add_argument("--worktree-root", type=Path, default=DEFAULT_WORKTREE_ROOT)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--worktree", type=Path, required=True)
    freeze.add_argument("--spec", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("attempt_dir", type=Path)
    run.add_argument("--manifest-sha256", required=True)
    run.add_argument("--minimum-free-disk-bytes", type=int)
    archive_attempt_parser = subparsers.add_parser("archive")
    archive_attempt_parser.add_argument("attempt_dir", type=Path)
    archive_attempt_parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path(
            os.environ.get("MANCALA_EXPERIMENT_ARCHIVE_ROOT", DEFAULT_ARCHIVE_ROOT)
        ),
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("attempt_dir", type=Path)
    publish = subparsers.add_parser("prepare-publication")
    publish.add_argument("attempt_dir", type=Path)
    publish.add_argument("destination", type=Path)
    listing = subparsers.add_parser("list")
    listing.add_argument("--repo", type=Path, default=Path.cwd())
    archive = subparsers.add_parser("verify-archive")
    archive.add_argument("archive", type=Path)
    archive.add_argument("manifest", type=Path)
    remove = subparsers.add_parser("remove")
    remove.add_argument("--repo", type=Path, default=Path.cwd())
    remove.add_argument("--worktree", type=Path, required=True)
    remove.add_argument("--merged-into", default="origin/main")
    args = parser.parse_args()
    if args.action == "create":
        spec = json.loads(args.registration.read_bytes())
        path = create_worktree(
            args.repo,
            spec["experiment_id"],
            args.branch,
            args.base_commit,
            hypothesis=spec["hypothesis"],
            objective=spec["objective"],
            experiment_type=spec["experiment_type"],
            authorization_reference=spec.get("authorization_reference"),
            random_seeds=spec["random_seeds"],
            input_artifacts=spec["input_artifacts"],
            checkpoint_identity=spec.get("checkpoint_identity"),
            acceptance_criteria=spec["acceptance_criteria"],
            worktree_root=args.worktree_root,
            registration_metadata=spec.get("registration_metadata"),
        )
        print(path)
    elif args.action == "freeze":
        spec = json.loads(args.spec.read_bytes())
        attempt_dir = freeze_attempt(args.worktree, **spec)
        print(
            json.dumps(
                {
                    "attempt_dir": str(attempt_dir),
                    "manifest_sha256": sha256_file(attempt_dir / "manifest.json"),
                },
                sort_keys=True,
            )
        )
    elif args.action == "run":

        def terminate_as_interruption(_signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt("experiment_process_terminated")

        signal.signal(signal.SIGTERM, terminate_as_interruption)
        print(
            json.dumps(
                execute_attempt(
                    args.attempt_dir,
                    expected_manifest_sha256=args.manifest_sha256,
                    minimum_free_disk_bytes=args.minimum_free_disk_bytes,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.action == "archive":
        print(
            json.dumps(
                archive_attempt(args.attempt_dir, args.archive_root),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.action == "verify":
        print(json.dumps(verify_attempt(args.attempt_dir), indent=2, sort_keys=True))
    elif args.action == "prepare-publication":
        print(
            json.dumps(
                prepare_publication(args.attempt_dir, args.destination),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.action == "remove":
        safe_remove_worktree(args.repo, args.worktree, merged_into=args.merged_into)
    elif args.action == "list":
        print(json.dumps(inspect_worktrees(args.repo), indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                verify_archive(args.archive, args.manifest), indent=2, sort_keys=True
            )
        )


if __name__ == "__main__":
    main()
