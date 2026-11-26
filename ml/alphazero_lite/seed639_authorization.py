"""Merge-based authorization and lifecycle binding for Seed 639 execution."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

REPOSITORY = "useruby/mancala-ai"
PROJECT_OWNER = "useruby"
EXPERIMENT_ID = "seed639-prospective-policy-prior-classification"
SELECTED_INVENTORY_SHA256 = (
    "6c6b1f680b2177cb67c79a9a642216703f160131034fbf8fe86e8130910ed077"
)
EXECUTION_SCOPE = (
    "B512/S615 deterministic legal-masked prior inference, descriptive policy "
    "comparison, and frozen secondary subset selection on exactly the authenticated "
    "512-state Seed 636 population; no additional states or primary-population modification."
)
REGISTRATION_PATH = f"docs/experiments/{EXPERIMENT_ID}/registration.json"
REQUIRED_PROTOCOLS = {
    "docs/experiment-execution-protocol.md": "69a77cdc24e2828f92d8d9eb96c75f619aa594240910c1684835fc1245519d6b",
    "docs/seed623-prospective-exact-quality-benchmark.md": "f01df9467dfc4a1088796f583551e203f8b045a00498363f4038b3ffac87f400",
    "docs/seed623-631-execution-contract-amendment.md": "633742139061227f1543ff7c1360f5d8e5b97eafb50a4755b2a7b47afcc033eb",
}
RESOURCE_BUDGETS = {
    "wall_time_limit_seconds": 14_400,
    "cpu_time_limit_seconds": 10_800,
    "process_rss_limit_bytes": 8_589_934_592,
    "output_and_temporary_storage_limit_bytes": 5_368_709_120,
    "execution_concurrency": 1,
    "inference_batch_size": 1,
}
REGISTERED_ENVIRONMENT = {
    "PYTHONHASHSEED": "615",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}
HISTORICAL_RUNTIME_LOCK_PATH = (
    "/home/alex/Mancala/ai/.worktrees/seed639-prospective-policy-prior-classification/docs/experiments/"
    "seed639-prospective-policy-prior-classification/historical-verifier-runtime-lock.txt"
)
HISTORICAL_RUNTIME_LOCK_SHA256 = (
    "4ac15303ded8e4a2616c89eb5108ca700c6e5382bf3257904aba5c20632194c6"
)
SEED615_DEPENDENCY_LOCK_SHA256 = (
    "5950f891ea746be6058af5fae4c414ac0aca872bec064a7d28c383f795e2fa79"
)
HISTORICAL_VERIFIER_RUNTIME = {
    "interpreter": "/home/alex/Mancala/ai/.tmp/seed639-historical-verifier-venv/bin/python",
    "python_version": "3.14.7",
    "numpy_version": "2.5.3",
    "torch_version": "2.14.0+cu130",
    "device": "cpu",
    "cuda_available": False,
    "thread_environment": REGISTERED_ENVIRONMENT,
    "torch_threads": 1,
    "deterministic_algorithms": True,
    "runtime_lock_path": HISTORICAL_RUNTIME_LOCK_PATH,
    "runtime_lock_sha256": HISTORICAL_RUNTIME_LOCK_SHA256,
    "seed615_dependency_lock_path": "/home/alex/Mancala/ai-seed615-authorized-eval/ml/alphazero_lite/requirements-seed599-evaluation.txt",
    "seed615_dependency_lock_sha256": SEED615_DEPENDENCY_LOCK_SHA256,
    "worktree_dependency_lock": "ml/alphazero_lite/requirements-seed599-evaluation.txt",
    "worktree_dependency_lock_sha256": SEED615_DEPENDENCY_LOCK_SHA256,
    "entrypoint_module": "ml.alphazero_lite.seed639_historical_authentication",
    "execution_mode": "separate_read_only_python_subprocess",
}
EXECUTION_AMENDMENT = {
    "schema": "seed639-execution-amendment-v1",
    "amendment_id": "attempt-0002-historical-verifier-runtime-correction",
    "reason": "attempt-0001 stopped before inference because the historical Seed 615 verifier dependency was missing from the inference runtime",
    "preserved_attempt_ids": ["attempt-0001"],
    "authorized_attempt_ids": ["attempt-0002"],
    "historical_verifier_runtime": HISTORICAL_VERIFIER_RUNTIME,
    "historical_authentication_preflight_required_before_freeze": True,
    "prospective_inference_runtime_unchanged": True,
}
AMENDMENT_RUNTIME_INPUTS = {
    HISTORICAL_RUNTIME_LOCK_PATH: HISTORICAL_RUNTIME_LOCK_SHA256,
    "/home/alex/Mancala/ai/.worktrees/seed639-prospective-policy-prior-classification/ml/alphazero_lite/requirements-seed599-evaluation.txt": SEED615_DEPENDENCY_LOCK_SHA256,
    HISTORICAL_VERIFIER_RUNTIME[
        "seed615_dependency_lock_path"
    ]: SEED615_DEPENDENCY_LOCK_SHA256,
}
CHECKPOINT_IDENTITY = {
    "B512": "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
    "S615": "cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780",
    "historical_seed615_evaluation_freeze_sha256": "ba30f5e3efadf3358db9f03c7b4eba4f69b23e2f95dcb4089723a1fb9664f32b",
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True
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
        or registration.get("random_seeds") != []
        or registration.get("resource_budgets") != RESOURCE_BUDGETS
        or registration.get("checkpoint_identity") != CHECKPOINT_IDENTITY
        or registration.get("approved_model_implementation", {}).get("input_encoding")
        != "kalah_v3"
        or registration.get("approved_model_implementation", {}).get("evaluation_cache")
        is not False
        or registration.get("approved_model_implementation", {}).get("arena_py_sha256")
        != "c16a668e8f90bd87b59dbc08f6a1ddb3b5ffc912e21cc0564efaceba37e03fe2"
        or registration.get("approved_model_implementation", {}).get(
            "self_play_py_sha256"
        )
        != "76718efde7ccfb708f4bb0a4314f812bab746d1364cb75ce5c8b3551b635fc0c"
        or registration.get("approved_model_implementation", {}).get(
            "input_encodings_py_sha256"
        )
        != "0a4f56ed60f0c2ed6b93a673dbc0850a966a907912dc26edd092bfd1ac209dbe"
        or registration.get("approved_model_implementation", {}).get(
            "kalah_rules_py_sha256"
        )
        != "6e5338ef9a85b690f51b6d0d184c8bc0f54c2b28ac084a4a17a88f9f552ad036"
        or registration.get("approved_model_implementation", {}).get("python_version")
        != "3.14.7"
        or registration.get("approved_model_implementation", {}).get("numpy_version")
        != "2.5.3"
        or registration.get("approved_model_implementation", {}).get(
            "numerical_backend"
        )
        != "NumPy ArtifactEvaluator; capture NumPy build/BLAS configuration in runtime receipt"
        or registration.get("approved_model_implementation", {}).get(
            "thread_configuration"
        )
        != {"environment": REGISTERED_ENVIRONMENT, "threads_per_pool": 1}
        or registration.get("scientific_acceptance", {}).get(
            "selected_inventory_sha256"
        )
        != SELECTED_INVENTORY_SHA256
    ):
        raise PermissionError("seed639_registration_scope_or_budget_invalid")
    if registration.get("protocol_documents") != REQUIRED_PROTOCOLS:
        raise PermissionError("seed639_registered_protocol_bindings_invalid")
    if set(registration.get("input_artifacts", {})) == set():
        raise PermissionError("seed639_registered_inputs_missing")
    amendment = registration.get("execution_amendment")
    if amendment is not None:
        if amendment != EXECUTION_AMENDMENT or any(
            registration.get("input_artifacts", {}).get(path) != digest
            for path, digest in AMENDMENT_RUNTIME_INPUTS.items()
        ):
            raise PermissionError("seed639_execution_amendment_invalid")


def authorize_registration(
    registration_path: Path,
    source_commit: str,
    *,
    expected_registration_sha256: str | None = None,
    associated_prs: list[dict[str, Any]] | None = None,
    pr_fetcher: Callable[[int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Authenticate the owner-merged registration revision bound to a source."""
    if registration_path.is_symlink() or not registration_path.is_file():
        raise PermissionError("seed639_registration_file_invalid")
    repo = Path(
        _git_bytes(registration_path.parent, "rev-parse", "--show-toplevel")
        .decode()
        .strip()
    )
    current_raw = registration_path.read_bytes()
    current_registration = json.loads(current_raw)
    _validate_registration(current_registration)
    permanent = repo / REGISTRATION_PATH
    if (
        permanent.is_symlink()
        or not permanent.is_file()
        or permanent.read_bytes() != current_raw
    ):
        raise PermissionError("seed639_permanent_registration_mismatch")
    resolved_source = (
        _git_bytes(repo, "rev-parse", "--verify", f"{source_commit}^{{commit}}")
        .decode()
        .strip()
    )
    if resolved_source != source_commit:
        raise PermissionError("seed639_source_commit_invalid")
    try:
        source_raw = _git_bytes(repo, "show", f"{source_commit}:{REGISTRATION_PATH}")
    except subprocess.CalledProcessError as error:
        raise PermissionError("seed639_source_registration_missing") from error
    source_digest = sha256(source_raw)
    _validate_registration(json.loads(source_raw))
    if (
        expected_registration_sha256 is not None
        and source_digest != expected_registration_sha256
    ):
        raise PermissionError("seed639_source_registration_digest_mismatch")

    prs = associated_prs
    if prs is None:
        prs = _github_json(f"repos/{REPOSITORY}/commits/{source_commit}/pulls")
    fetch_pr = pr_fetcher or (
        lambda number: _github_json(f"repos/{REPOSITORY}/pulls/{number}")
    )
    accepted: list[dict[str, Any]] = []
    for association in prs:
        number = association.get("number")
        if type(number) is not int:
            continue
        pr = fetch_pr(number)
        base = pr.get("base", {})
        merge_commit = pr.get("merge_commit_sha")
        if (
            pr.get("number") != number
            or pr.get("state") != "closed"
            or not pr.get("merged_at")
            or pr.get("merged_by", {}).get("login") != PROJECT_OWNER
            or base.get("ref") != "main"
            or base.get("repo", {}).get("full_name") != REPOSITORY
            or merge_commit != source_commit
        ):
            continue
        try:
            committed_registration = _git_bytes(
                repo, "show", f"{merge_commit}:{REGISTRATION_PATH}"
            )
        except subprocess.CalledProcessError:
            continue
        if committed_registration != source_raw:
            continue
        base_commit = base.get("sha")
        if not isinstance(base_commit, str):
            continue
        try:
            resolved_base = (
                _git_bytes(repo, "rev-parse", "--verify", f"{base_commit}^{{commit}}")
                .decode()
                .strip()
            )
        except subprocess.CalledProcessError:
            continue
        if resolved_base != base_commit or not _git_is_ancestor(
            repo, base_commit, source_commit
        ):
            continue
        try:
            base_registration = _git_bytes(
                repo, "show", f"{base_commit}:{REGISTRATION_PATH}"
            )
        except subprocess.CalledProcessError:
            base_registration = None
        if base_registration == committed_registration:
            continue
        for path, expected in REQUIRED_PROTOCOLS.items():
            try:
                protocol_blob = _git_bytes(repo, "show", f"{merge_commit}:{path}")
            except subprocess.CalledProcessError as error:
                raise PermissionError("seed639_protocol_blob_missing") from error
            if sha256(protocol_blob) != expected:
                raise PermissionError("seed639_protocol_blob_hash_mismatch")
        accepted.append(pr)
    if len(accepted) != 1:
        raise PermissionError("seed639_owner_merged_registration_not_unique")
    pr = accepted[0]
    return {
        "repository": REPOSITORY,
        "experiment_id": EXPERIMENT_ID,
        "registration_path": REGISTRATION_PATH,
        "registration_sha256": source_digest,
        "current_registration_sha256": sha256(current_raw),
        "pr_number": pr["number"],
        "pr_url": pr["html_url"],
        "merge_commit": pr["merge_commit_sha"],
        "merged_by": pr["merged_by"]["login"],
        "merge_timestamp": pr["merged_at"],
    }


def validate_attempt(
    attempt_dir: Path,
    *,
    associated_prs: list[dict[str, Any]] | None = None,
    pr_fetcher: Callable[[int], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the frozen/started attempt and owner-merged registration."""
    from ml.alphazero_lite.experiment_lifecycle import (
        _validate_experiment_worktree,
        artifact_sha256,
        load_worktree_registration,
    )

    attempt_dir = attempt_dir.resolve(strict=True)
    manifest_path = attempt_dir / "manifest.json"
    if not manifest_path.is_file() or not (attempt_dir / "started.json").is_file():
        raise PermissionError("seed639_frozen_started_attempt_required")
    manifest = json.loads(manifest_path.read_bytes())
    worktree = Path(manifest.get("worktree_path", "")).resolve(strict=True)
    expected_runtime = manifest.get("resource_budget", {}).get(
        "expected_runtime_seconds"
    )
    if (
        attempt_dir
        != (
            worktree
            / ".experiment"
            / "runtime"
            / "attempts"
            / manifest.get("attempt_id", "")
        ).resolve()
    ):
        raise PermissionError("seed639_lifecycle_attempt_path_mismatch")
    registration, registration_path = load_worktree_registration(worktree)
    _validate_registration(registration)
    _validate_experiment_worktree(worktree)
    amendment = registration.get("execution_amendment")
    attempt_id = manifest.get("attempt_id")
    if attempt_id == "attempt-0002" and (
        amendment != EXECUTION_AMENDMENT
        or attempt_id not in amendment.get("authorized_attempt_ids", [])
        or manifest.get("configuration", {}).get("historical_verifier_runtime")
        != HISTORICAL_VERIFIER_RUNTIME
    ):
        raise PermissionError("seed639_attempt_0002_runtime_amendment_mismatch")
    has_terminal = any(
        (attempt_dir / f"{status}.json").is_file()
        for status in ("completed", "failed", "interrupted")
    )
    source_tree = (
        _git_bytes(worktree, "rev-parse", f"{manifest.get('source_commit')}^{{tree}}")
        .decode()
        .strip()
    )
    if (
        manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("status") != "frozen"
        or manifest.get("registration_path") != registration_path
        or (
            not has_terminal
            and manifest.get("source_commit")
            != _git_bytes(worktree, "rev-parse", "HEAD").decode().strip()
        )
        or (
            not has_terminal
            and manifest.get("execution_source_identity")
            != _git_bytes(worktree, "rev-parse", "HEAD^{tree}").decode().strip()
        )
        or manifest.get("execution_source_identity") != source_tree
        or manifest.get("configuration", {}).get("execution_scope") != EXECUTION_SCOPE
        or manifest.get("configuration", {}).get("input_encoding") != "kalah_v3"
        or manifest.get("configuration", {}).get("inference_batch_size") != 1
        or manifest.get("configuration", {}).get("evaluation_cache") is not False
        or manifest.get("configuration", {}).get("execution_concurrency") != 1
        or manifest.get("configuration", {}).get("environment")
        != REGISTERED_ENVIRONMENT
        or manifest.get("configuration", {}).get("numerical_backend")
        != "NumPy ArtifactEvaluator"
        or manifest.get("configuration", {}).get("thread_configuration")
        != {"environment": REGISTERED_ENVIRONMENT, "threads_per_pool": 1}
        or manifest.get("device", {}).get("type") != "cpu"
        or manifest.get("configuration", {}).get("registration_sha256")
        != manifest.get("registration_sha256")
        or manifest.get("worker_count") != 1
        or manifest.get("resource_budget", {}).get("wall_time_limit_seconds")
        != RESOURCE_BUDGETS["wall_time_limit_seconds"]
        or type(expected_runtime) is not int
        or not 1 <= expected_runtime <= RESOURCE_BUDGETS["wall_time_limit_seconds"]
        or manifest.get("resource_budget", {}).get("minimum_free_disk_bytes", 0)
        < RESOURCE_BUDGETS["output_and_temporary_storage_limit_bytes"]
        or manifest.get("resource_budget", {})
        .get("additional_limits", {})
        .get("cpu_time_limit_seconds")
        != RESOURCE_BUDGETS["cpu_time_limit_seconds"]
        or manifest.get("resource_budget", {})
        .get("additional_limits", {})
        .get("process_rss_limit_bytes")
        != RESOURCE_BUDGETS["process_rss_limit_bytes"]
        or manifest.get("resource_budget", {})
        .get("additional_limits", {})
        .get("output_and_temporary_storage_limit_bytes")
        != RESOURCE_BUDGETS["output_and_temporary_storage_limit_bytes"]
        or manifest.get("runtime_versions", {}).get("numpy") != "2.5.3"
        or manifest.get("runtime_versions", {}).get("python_version") != "3.14.7"
        or manifest.get("runtime_versions", {}).get("numerical_backend")
        != "NumPy ArtifactEvaluator"
        or "seed639-output" not in manifest.get("output_paths", [])
        or manifest.get("random_seeds") != []
    ):
        raise PermissionError("seed639_lifecycle_manifest_binding_invalid")
    command = manifest.get("command", [])
    if not any("seed639_policy_prior_classification" in token for token in command):
        raise PermissionError("seed639_frozen_command_invalid")
    dirty_status = _git_bytes(
        worktree, "status", "--porcelain", "--untracked-files=all"
    ).decode()
    if dirty_status:
        raise PermissionError(f"seed639_source_worktree_dirty:{dirty_status}")
    registered_inputs = {
        str(Path(path).absolute()): digest
        for path, digest in registration["input_artifacts"].items()
    }
    if attempt_id == "attempt-0001" and amendment == EXECUTION_AMENDMENT:
        for path in AMENDMENT_RUNTIME_INPUTS:
            registered_inputs.pop(str(Path(path).absolute()), None)
    if registered_inputs != manifest.get("input_artifacts"):
        raise PermissionError("seed639_lifecycle_input_binding_invalid")
    for path, expected in manifest["input_artifacts"].items():
        if artifact_sha256(Path(path)) != expected:
            raise PermissionError("seed639_registered_input_identity_mismatch")
    if any(
        manifest.get("environment", {}).get(key) != value
        for key, value in REGISTERED_ENVIRONMENT.items()
    ):
        raise PermissionError("seed639_frozen_thread_environment_mismatch")
    authorization = authorize_registration(
        worktree / registration_path,
        manifest["source_commit"],
        expected_registration_sha256=manifest.get("registration_sha256"),
        associated_prs=associated_prs,
        pr_fetcher=pr_fetcher,
    )
    if authorization["registration_sha256"] != manifest.get("registration_sha256"):
        raise PermissionError("seed639_authorization_registration_digest_mismatch")
    return manifest, {"registration": registration, "authorization": authorization}
