"""Portable runtime search-policy contract stored with an incumbent artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "azlite_runtime_search_policy_v1"
EXACT_ROOT_MODE = "puct_exact_root_hybrid"


class RuntimeSearchPolicyError(ValueError):
    """Raised when an incumbent search-policy sidecar is incomplete or altered."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runtime_search_policy(artifact_dir: str | Path) -> dict[str, Any] | None:
    """Load and validate the optional generic policy attached to an artifact."""
    artifact_dir = Path(artifact_dir)
    policy_path = artifact_dir / "search_policy.json"
    if not policy_path.is_file():
        return None
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeSearchPolicyError("runtime_search_policy_invalid") from error
    if not isinstance(policy, dict) or policy.get("schema") != SCHEMA:
        raise RuntimeSearchPolicyError("runtime_search_policy_invalid")
    required = {
        "mode": EXACT_ROOT_MODE,
        "exact_root_enabled": True,
        "exact_root_objective": "final_score_margin",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
        "exact_leaf_solve": "disabled",
        "solver_implementation_identity": "native_kvtb_root_action_probe_v1",
    }
    if any(policy.get(name) != value for name, value in required.items()):
        raise RuntimeSearchPolicyError("runtime_search_policy_invalid")
    if not isinstance(policy.get("exact_root_threshold"), int):
        raise RuntimeSearchPolicyError("runtime_search_policy_invalid")
    for artifact_name in ("native_probe", "tablebase"):
        artifact = policy.get(artifact_name)
        if not isinstance(artifact, dict):
            raise RuntimeSearchPolicyError("runtime_search_policy_invalid")
        reference, expected_hash = artifact.get("path"), artifact.get("sha256")
        if not isinstance(reference, str) or Path(reference).is_absolute():
            raise RuntimeSearchPolicyError("runtime_search_policy_invalid")
        if not isinstance(expected_hash, str):
            raise RuntimeSearchPolicyError("runtime_search_policy_invalid")
        artifact_path = Path(__file__).resolve().parents[2] / reference
        if not artifact_path.is_file() or sha256_file(artifact_path) != expected_hash:
            raise RuntimeSearchPolicyError("runtime_search_policy_artifact_mismatch")
        artifact["resolved_path"] = str(artifact_path)
    return policy


def resolve_strength_comparison_runtime_contract(
    *,
    current_artifact: str | Path,
    challenger_artifact: str | Path,
    exact_root_threshold: int | None = None,
    native_probe: str | Path | None = None,
    tablebase: str | Path | None = None,
) -> dict[str, Any] | None:
    """Bind a strength comparison to the incumbent's validated runtime policy.

    A challenger without a copied sidecar explicitly inherits the incumbent
    policy.  This is intentionally a resolution layer, not another policy
    validator: ``load_runtime_search_policy`` remains authoritative.
    """
    try:
        current = load_runtime_search_policy(current_artifact)
        challenger = load_runtime_search_policy(challenger_artifact)
    except RuntimeSearchPolicyError as error:
        raise RuntimeSearchPolicyError(
            "arena_runtime_search_policy_mismatch"
        ) from error
    if current is None:
        return None

    semantic_fields = (
        "mode",
        "exact_root_threshold",
        "solver_implementation_identity",
        "exact_root_objective",
        "exact_root_tie_rule",
        "exact_leaf_solve",
    )
    if challenger is not None and any(
        challenger[field] != current[field] for field in semantic_fields
    ):
        raise RuntimeSearchPolicyError("arena_runtime_search_policy_mismatch")
    if challenger is not None and any(
        challenger[name]["sha256"] != current[name]["sha256"]
        for name in ("native_probe", "tablebase")
    ):
        raise RuntimeSearchPolicyError("arena_runtime_search_policy_mismatch")

    expected_threshold = current["exact_root_threshold"]
    expected_probe = Path(current["native_probe"]["resolved_path"])
    expected_tablebase = Path(current["tablebase"]["resolved_path"])
    if exact_root_threshold not in (None, expected_threshold):
        raise RuntimeSearchPolicyError("arena_runtime_search_policy_mismatch")
    if (
        native_probe is not None
        and Path(native_probe).resolve() != expected_probe.resolve()
    ):
        raise RuntimeSearchPolicyError("arena_runtime_search_policy_mismatch")
    if (
        tablebase is not None
        and Path(tablebase).resolve() != expected_tablebase.resolve()
    ):
        raise RuntimeSearchPolicyError("arena_runtime_search_policy_mismatch")

    policy_path = Path(current_artifact) / "search_policy.json"
    return {
        "runtime_search_policy_mode": current["mode"],
        "exact_root_solve_threshold": expected_threshold,
        "exact_root_solver": current["solver_implementation_identity"],
        "exact_root_native_probe": str(expected_probe),
        "exact_root_native_probe_sha256": current["native_probe"]["sha256"],
        "exact_root_tablebase": str(expected_tablebase),
        "exact_root_tablebase_sha256": current["tablebase"]["sha256"],
        "exact_root_objective": current["exact_root_objective"],
        "exact_root_tie_rule": current["exact_root_tie_rule"],
        "exact_leaf_solve_mode": current["exact_leaf_solve"],
        "runtime_search_policy_sha256": sha256_file(policy_path),
        "challenger_runtime_policy_inherited": challenger is None,
    }
