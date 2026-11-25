"""Verify Seed 636 authorization from a project-owner-merged registration PR."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

SCHEMA = "seed636-registration-pr-authorization-v1"
REPOSITORY = "useruby/mancala-ai"
PROJECT_OWNER = "useruby"
EXECUTION_SCOPE = "model-independent-generation-and-screening-only"
EXPERIMENT_ID = "seed636-prospective-population"
APPROVED_DECISIONS = [
    "seed625-finite-cohort-estimand-amendment",
    "seed635-worktree-execution-contract-amendment",
    "known-exposure-screened-inherited-PE4-unresolved",
    "model-independent-population-generation-and-screening-only",
]
REQUIRED_PROTOCOL_DOCUMENTS = {
    "docs/seed624-estimand-amendment-proposal.md",
    "docs/seed623-prospective-exact-quality-benchmark.md",
    "docs/seed623-631-execution-contract-amendment.md",
    "docs/experiment-execution-protocol.md",
    "docs/seed636-owner-approval.md",
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
    ).stdout


def _git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            capture_output=True,
        ).returncode
        == 0
    )


def _github_json(endpoint: str) -> Any:
    result = subprocess.run(
        ["gh", "api", endpoint], check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout)


def _validate_registration(registration: dict[str, Any]) -> None:
    if (
        registration.get("schema") != "mancala-experiment-registration-v1"
        or registration.get("experiment_id") != EXPERIMENT_ID
        or registration.get("execution_scope") != EXECUTION_SCOPE
        or registration.get("authorization_contract") != SCHEMA
        or registration.get("random_seeds") != [62320261121]
        or registration.get("approved_decisions") != APPROVED_DECISIONS
        or not REQUIRED_PROTOCOL_DOCUMENTS.issubset(
            registration.get("protocol_documents", {})
        )
    ):
        raise PermissionError("merged_registration_scope_or_protocol_invalid")

    budgets = registration.get("resource_budgets", {})
    if (
        not budgets
        or any(type(value) is not int or value <= 0 for value in budgets.values())
        or budgets.get("execution_concurrency") != 1
        or budgets.get("selection_optimizer_workers") != 1
    ):
        raise PermissionError("merged_registration_resource_budget_invalid")

    criteria = registration.get("scientific_acceptance", {})
    if (
        criteria.get("openings") != 256
        or criteria.get("continuation_games") != 2048
        or criteria.get("selected_states") != 512
        or criteria.get("stone_bin_quotas")
        != {"17": 103, "18": 103, "19": 102, "20": 102, "21": 102}
        or criteria.get("minimum_eligible_candidate_supply")
        != {"17": 206, "18": 206, "19": 204, "20": 204, "21": 204}
        or criteria.get("minimum_distinct_opening_clusters") != 128
        or criteria.get("max_states_per_opening") != 4
        or criteria.get("max_states_per_continuation_game") != 2
        or criteria.get("deduplication")
        != ["canonical_state", "production_float32_input"]
        or criteria.get("authenticated_known_exposure_screening") is not True
        or criteria.get(
            "no_inference_mcts_exact_solving_tablebase_probes_training_or_promotion"
        )
        is not True
        or criteria.get("exposure_classification")
        != "known-exposure-screened; inherited P-E4 exposure unresolved"
        or criteria.get("independent_reproduction_required") is not True
        or criteria.get("failure_policy")
        != "preserve_evidence_and_stop_without_parameter_change_or_automatic_retry"
    ):
        raise PermissionError("merged_registration_scientific_criteria_invalid")


def authorize_registration(
    registration_path: Path,
    source_commit: str,
    *,
    associated_prs: list[dict[str, Any]] | None = None,
    pr_fetcher: Callable[[int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a create-time receipt for the unique owner-merged registration PR."""
    if registration_path.is_symlink() or not registration_path.is_file():
        raise PermissionError("experiment_registration_file_invalid")
    repository_root = Path(
        _git_bytes(registration_path.parent, "rev-parse", "--show-toplevel")
        .decode()
        .strip()
    )
    registration_raw = registration_path.read_bytes()
    registration = json.loads(registration_raw)
    _validate_registration(registration)
    registration_digest = sha256(registration_raw)
    from ml.alphazero_lite.experiment_lifecycle import registration_repository_path

    permanent_registration_path = registration_repository_path(
        registration["experiment_id"]
    )
    permanent_registration = repository_root / permanent_registration_path
    if (
        permanent_registration.is_symlink()
        or not permanent_registration.is_file()
        or permanent_registration.read_bytes() != registration_raw
    ):
        raise PermissionError("permanent_registration_snapshot_mismatch")

    resolved_commit = (
        _git_bytes(
            repository_root, "rev-parse", "--verify", f"{source_commit}^{{commit}}"
        )
        .decode()
        .strip()
    )
    if resolved_commit != source_commit:
        raise PermissionError("registration_pr_source_commit_invalid")

    candidate_prs = associated_prs
    if candidate_prs is None:
        candidate_prs = _github_json(
            f"repos/{REPOSITORY}/commits/{source_commit}/pulls"
        )
    fetch_pr = pr_fetcher or (
        lambda number: _github_json(f"repos/{REPOSITORY}/pulls/{number}")
    )

    matches: list[dict[str, Any]] = []
    for association in candidate_prs:
        number = association.get("number")
        if type(number) is not int:
            continue
        pull = fetch_pr(number)
        base = pull.get("base", {})
        merge_commit = pull.get("merge_commit_sha")
        if (
            pull.get("number") != number
            or pull.get("state") != "closed"
            or not pull.get("merged_at")
            or pull.get("merged_by", {}).get("login") != PROJECT_OWNER
            or base.get("ref") != "main"
            or base.get("repo", {}).get("full_name") != REPOSITORY
            or merge_commit != source_commit
        ):
            continue
        try:
            committed_registration = _git_bytes(
                repository_root,
                "show",
                f"{merge_commit}:{permanent_registration_path}",
            )
        except subprocess.CalledProcessError:
            continue
        if committed_registration != registration_raw:
            continue
        base_commit = base.get("sha")
        if not isinstance(base_commit, str):
            continue
        try:
            resolved_base = (
                _git_bytes(
                    repository_root,
                    "rev-parse",
                    "--verify",
                    f"{base_commit}^{{commit}}",
                )
                .decode()
                .strip()
            )
        except subprocess.CalledProcessError:
            continue
        if resolved_base != base_commit:
            continue
        if not _git_is_ancestor(repository_root, base_commit, source_commit):
            continue
        try:
            base_registration = _git_bytes(
                repository_root,
                "show",
                f"{base_commit}:{permanent_registration_path}",
            )
        except subprocess.CalledProcessError:
            base_registration = None
        if base_registration == committed_registration:
            continue
        for path, expected in registration["protocol_documents"].items():
            if path.startswith("/") or ".." in Path(path).parts:
                raise PermissionError("registration_protocol_path_invalid")
            try:
                blob = _git_bytes(repository_root, "show", f"{merge_commit}:{path}")
            except subprocess.CalledProcessError as error:
                raise PermissionError("registration_protocol_blob_missing") from error
            if sha256(blob) != expected:
                raise PermissionError("registration_protocol_blob_hash_mismatch")
        matches.append(pull)

    if len(matches) != 1:
        raise PermissionError("owner_merged_registration_pr_not_unique")
    pull = matches[0]
    return {
        "schema": SCHEMA,
        "repository": REPOSITORY,
        "experiment_id": EXPERIMENT_ID,
        "registration_path": permanent_registration_path,
        "execution_scope": EXECUTION_SCOPE,
        "approved_decisions": APPROVED_DECISIONS,
        "registration_sha256": registration_digest,
        "pr_number": pull["number"],
        "pr_url": pull["html_url"],
        "merge_commit": pull["merge_commit_sha"],
        "merged_by": pull["merged_by"]["login"],
        "merge_timestamp": pull["merged_at"],
    }


def validate_attempt(attempt_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require a frozen lifecycle attempt and verify its merged registration PR."""
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
        or manifest.get("experiment_id") != EXPERIMENT_ID
    ):
        raise PermissionError("lifecycle_attempt_identity_mismatch")
    if not any(
        "seed624_prospective_benchmark" in item for item in manifest.get("command", [])
    ):
        raise PermissionError("lifecycle_command_binding_mismatch")

    from ml.alphazero_lite.experiment_lifecycle import (
        _validate_experiment_worktree,
        artifact_sha256,
        load_worktree_registration,
    )

    registration, registration_relative_path = load_worktree_registration(worktree)
    registered_worktree = _validate_experiment_worktree(worktree)
    if registered_worktree.get("experiment_id") != EXPERIMENT_ID:
        raise PermissionError("lifecycle_worktree_registration_mismatch")
    registration_path = worktree / registration_relative_path
    registration_digest = sha256(registration_path.read_bytes())
    if (
        manifest.get("registration_path") != registration_relative_path
        or manifest.get("registration_sha256") != registration_digest
    ):
        raise PermissionError("lifecycle_permanent_registration_binding_mismatch")
    if manifest.get("source_commit") != _git(
        worktree, "rev-parse", "HEAD"
    ) or manifest.get("execution_source_identity") != _git(
        worktree, "rev-parse", "HEAD^{tree}"
    ):
        raise PermissionError("lifecycle_source_commit_or_tree_mismatch")
    if manifest.get("configuration", {}).get("execution_scope") != EXECUTION_SCOPE:
        raise PermissionError("lifecycle_configuration_scope_mismatch")
    if (
        manifest.get("configuration", {}).get("registration_sha256")
        != registration_digest
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
    for raw_path, expected in manifest.get("input_artifacts", {}).items():
        if artifact_sha256(Path(raw_path)) != expected:
            raise PermissionError("lifecycle_input_identity_mismatch")
    if not manifest.get("output_paths") or manifest.get("worker_count") != 1:
        raise PermissionError("lifecycle_output_or_worker_binding_invalid")

    receipt = authorize_registration(registration_path, manifest["source_commit"])
    return manifest, {"registration": registration, "authorization": receipt}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
