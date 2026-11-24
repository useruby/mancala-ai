"""GitHub-owner authorization for Seed 636 research execution."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = "seed636-github-owner-approval-v1"
REPOSITORY = "useruby/mancala-ai"
PROJECT_OWNER = "useruby"
EXECUTION_SCOPE = "model-independent-generation-and-screening-only"
APPROVAL_MARKER = "SEED636-APPROVAL-V1"
APPROVED_DECISIONS = [
    "seed625-finite-cohort-estimand-amendment",
    "seed635-worktree-execution-contract-amendment",
    "known-exposure-screened-inherited-PE4-unresolved",
    "model-independent-population-generation-and-screening-only",
]
REQUIRED_PROTOCOL_DOCUMENTS = {
    "docs/seed624-estimand-amendment-proposal.md",
    "docs/seed623-631-execution-contract-amendment.md",
    "docs/experiment-execution-protocol.md",
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def fetch_github_comment(comment_id: str) -> dict[str, Any]:
    """Fetch the authoritative GitHub issue-comment record using the gh CLI."""
    result = subprocess.run(
        ["gh", "api", f"repos/{REPOSITORY}/issues/comments/{comment_id}"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("github_approval_comment_record_invalid")
    return value


def validate_approval(
    registration_path: Path,
    approval_path: Path,
    *,
    comment_fetcher=fetch_github_comment,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify approval identity, scope, documents and registration binding."""
    if registration_path.is_symlink() or not registration_path.is_file():
        raise PermissionError("experiment_registration_file_invalid")
    registration_raw = registration_path.read_bytes()
    registration = json.loads(registration_raw)
    if approval_path.is_symlink() or not approval_path.is_file():
        raise PermissionError("github_owner_approval_missing_or_invalid")
    approval = json.loads(approval_path.read_bytes())
    if approval.get("schema") != SCHEMA:
        raise PermissionError("github_owner_approval_missing_or_invalid")
    if approval.get("project_owner_identity") != PROJECT_OWNER:
        raise PermissionError("github_project_owner_identity_mismatch")
    if approval.get("execution_scope") != EXECUTION_SCOPE:
        raise PermissionError("github_approval_execution_scope_mismatch")
    if approval.get("experiment_id") != registration.get("experiment_id"):
        raise PermissionError("github_approval_experiment_identity_mismatch")
    if approval.get("registration_sha256") != sha256(registration_raw):
        raise PermissionError("github_approval_registration_identity_mismatch")
    if approval.get("approved_protocol_revision") != registration.get(
        "approved_protocol_revision"
    ):
        raise PermissionError("github_approval_protocol_revision_mismatch")
    if approval.get("protocol_documents") != registration.get("protocol_documents"):
        raise PermissionError("github_approval_protocol_document_mismatch")
    if (
        registration.get("approved_decisions") != APPROVED_DECISIONS
        or approval.get("approved_decisions") != APPROVED_DECISIONS
    ):
        raise PermissionError("github_approval_decision_scope_mismatch")
    if (
        not isinstance(approval.get("protocol_documents"), dict)
        or not approval["protocol_documents"]
    ):
        raise PermissionError("github_approval_protocol_documents_missing")
    if not REQUIRED_PROTOCOL_DOCUMENTS.issubset(approval["protocol_documents"]):
        raise PermissionError("github_approval_protocol_documents_incomplete")
    if not re.fullmatch(
        r"[0-9a-f]{40,64}", str(approval.get("approved_protocol_revision", ""))
    ):
        raise PermissionError("github_approval_protocol_revision_invalid")
    repository_root = registration_path.resolve().parents[1]
    for name, digest in approval["protocol_documents"].items():
        document = (repository_root / name).resolve(strict=True)
        if (
            (repository_root / name).is_symlink()
            or not document.is_relative_to(repository_root)
            or sha256(document.read_bytes()) != digest
        ):
            raise PermissionError("github_approval_protocol_document_hash_mismatch")
    url = approval.get("approval_url")
    match = re.fullmatch(
        r"https://github\.com/useruby/mancala-ai/(?:issues|pull)/(\d+)#issuecomment-(\d+)",
        str(url),
    )
    if not match:
        raise PermissionError("github_approval_comment_url_invalid")
    comment = comment_fetcher(match.group(2))
    expected_body = {
        "marker": APPROVAL_MARKER,
        "experiment_id": registration["experiment_id"],
        "registration_sha256": sha256(registration_raw),
        "approved_protocol_revision": registration["approved_protocol_revision"],
        "protocol_documents": registration["protocol_documents"],
        "approved_decisions": APPROVED_DECISIONS,
        "execution_scope": EXECUTION_SCOPE,
    }
    body = str(comment.get("body", ""))
    prefix = APPROVAL_MARKER + "\n"
    if not body.startswith(prefix):
        raise PermissionError("github_approval_comment_text_invalid")
    try:
        body_record = json.loads(body[len(prefix) :])
    except json.JSONDecodeError as error:
        raise PermissionError("github_approval_comment_text_invalid") from error
    if body_record != expected_body:
        raise PermissionError("github_approval_comment_binding_mismatch")
    if comment.get("user", {}).get("login") != PROJECT_OWNER:
        raise PermissionError("github_approval_comment_author_mismatch")
    if comment.get("html_url") != url:
        raise PermissionError("github_approval_comment_url_mismatch")
    if comment.get("created_at") != approval.get("approval_timestamp"):
        raise PermissionError("github_approval_timestamp_mismatch")
    try:
        parsed_timestamp = datetime.fromisoformat(
            str(approval["approval_timestamp"]).replace("Z", "+00:00")
        )
    except (KeyError, ValueError) as error:
        raise PermissionError("github_approval_timestamp_invalid") from error
    if parsed_timestamp.tzinfo is None:
        raise PermissionError("github_approval_timestamp_invalid")
    approval["github_comment_evidence"] = {
        "id": comment.get("id"),
        "author": comment.get("user", {}).get("login"),
        "created_at": comment.get("created_at"),
        "html_url": comment.get("html_url"),
        "body": body,
        "api_record_sha256": sha256(canonical_bytes(comment)),
    }
    return registration, approval


def validate_attempt(
    attempt_dir: Path, *, comment_fetcher=fetch_github_comment
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require lifecycle execution context and verify the frozen argv binding."""
    attempt_dir = attempt_dir.resolve(strict=True)
    manifest_path = attempt_dir / "manifest.json"
    started_path = attempt_dir / "started.json"
    if not manifest_path.is_file() or not started_path.is_file():
        raise PermissionError("lifecycle_frozen_started_attempt_required")
    manifest = json.loads(manifest_path.read_bytes())
    worktree = Path(manifest.get("worktree_path", "")).resolve(strict=True)
    if (
        worktree == worktree.parents[0]
        or attempt_dir
        != (
            worktree
            / ".experiment"
            / "runtime"
            / "attempts"
            / manifest.get("attempt_id", "")
        ).resolve()
    ):
        raise PermissionError("lifecycle_attempt_path_binding_mismatch")
    if (
        manifest.get("status") != "frozen"
        or manifest.get("experiment_id") != "seed636-prospective-population"
    ):
        raise PermissionError("lifecycle_attempt_identity_mismatch")
    command = manifest.get("command", [])
    if not any("seed624_prospective_benchmark" in item for item in command):
        raise PermissionError("lifecycle_command_binding_mismatch")
    registration_path = worktree / ".experiment" / "registration.json"
    approval_path = worktree / ".experiment" / "seed636-owner-approval.json"
    registration, approval = validate_approval(
        registration_path, approval_path, comment_fetcher=comment_fetcher
    )
    from ml.alphazero_lite.experiment_lifecycle import (
        _validate_experiment_worktree,
    )

    registered_worktree = _validate_experiment_worktree(worktree)
    if registered_worktree.get("experiment_id") != registration.get("experiment_id"):
        raise PermissionError("lifecycle_worktree_registration_mismatch")
    try:
        _git(
            worktree,
            "ls-files",
            "--error-unmatch",
            ".experiment/seed636-owner-approval.json",
        )
    except subprocess.CalledProcessError as error:
        raise PermissionError("lifecycle_approval_record_not_committed") from error
    if manifest.get("source_commit") != _git(worktree, "rev-parse", "HEAD"):
        raise PermissionError("lifecycle_source_commit_mismatch")
    if manifest.get("execution_source_identity") != _git(
        worktree, "rev-parse", "HEAD^{tree}"
    ):
        raise PermissionError("lifecycle_source_tree_mismatch")
    if manifest.get("configuration", {}).get("execution_scope") != EXECUTION_SCOPE:
        raise PermissionError("lifecycle_configuration_scope_mismatch")
    if manifest.get("configuration", {}).get("registration_sha256") != sha256(
        registration_path.read_bytes()
    ):
        raise PermissionError("lifecycle_registration_identity_mismatch")
    registered_inputs = {
        str(Path(path).absolute()): digest
        for path, digest in registration.get("input_artifacts", {}).items()
    }
    if registered_inputs != manifest.get("input_artifacts"):
        raise PermissionError("lifecycle_input_registration_mismatch")
    if _git(worktree, "status", "--porcelain", "--untracked-files=all"):
        raise PermissionError("lifecycle_source_worktree_dirty")
    from ml.alphazero_lite.experiment_lifecycle import artifact_sha256

    for raw_path, expected in manifest.get("input_artifacts", {}).items():
        if artifact_sha256(Path(raw_path)) != expected:
            raise PermissionError("lifecycle_input_identity_mismatch")
    if not manifest.get("output_paths"):
        raise PermissionError("lifecycle_output_paths_missing")
    if manifest.get("worker_count") != 1:
        raise PermissionError("lifecycle_worker_count_mismatch")
    return manifest, {"registration": registration, "approval": approval}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
