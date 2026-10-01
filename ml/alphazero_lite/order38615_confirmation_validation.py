"""Integrity checks shared by the frozen order 38615 confirmation tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.frozen_opponent_identity import validate_frozen_opponent_identity


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_confirmation_inputs(
    registration_path: Path,
    candidate_binding_path: Path,
    evaluation_binding_path: Path,
    registration: dict[str, Any],
    candidate_binding: dict[str, Any],
    evaluation_binding: dict[str, Any],
) -> None:
    """Validate every immutable header and artifact before using/publishing evidence."""
    registration_sha = sha256_file(registration_path)
    candidate_sha = sha256_file(candidate_binding_path)
    if candidate_binding.get("registration_sha256") != registration_sha:
        raise ValueError("candidate_binding_registration_hash_mismatch")
    if evaluation_binding.get("registration_sha256") != registration_sha:
        raise ValueError("evaluation_binding_registration_hash_mismatch")
    if evaluation_binding.get("candidate_binding_sha256") != candidate_sha:
        raise ValueError("evaluation_binding_candidate_hash_mismatch")

    evaluation = registration["evaluation"]
    candidate = candidate_binding.get("candidate", {})
    artifact = evaluation["artifact"]
    if (
        candidate != artifact
        or candidate_binding.get("runtime_contract") != evaluation["runtime_contract"]
    ):
        raise ValueError("candidate_binding_identity_mismatch")
    if candidate_binding.get("opponent") != evaluation["opponent_binding"]:
        raise ValueError("candidate_binding_opponent_mismatch")
    if evaluation_binding.get("opponent") != evaluation["opponent_binding"]:
        raise ValueError("evaluation_binding_opponent_mismatch")
    if candidate.get("checkpoint_sha256") != evaluation.get(
        "candidate_checkpoint_sha256"
    ):
        raise ValueError("candidate_checkpoint_identity_mismatch")

    artifact_path = Path(candidate["artifact"])
    checkpoint_path = Path(candidate["checkpoint"])
    if sha256_file(checkpoint_path) != candidate["checkpoint_sha256"]:
        raise ValueError("candidate_checkpoint_hash_mismatch")
    for name, expected in candidate["artifact_sha256"].items():
        if sha256_file(artifact_path / name) != expected:
            raise ValueError(f"candidate_artifact_hash_mismatch:{name}")
    if candidate["artifact_sha256"].get("model.npz") != candidate["checkpoint_sha256"]:
        raise ValueError("candidate_model_checkpoint_mismatch")
    runtime = evaluation["runtime_contract"]
    if candidate.get("runtime_contract") != runtime:
        raise ValueError("candidate_runtime_identity_mismatch")
    validate_frozen_opponent_identity(
        Path(evaluation["opponent_binding"]["artifact"]),
        evaluation["opponent_binding"],
        runtime,
    )
    for suite in evaluation["suites"].values():
        if sha256_file(Path(suite["path"])) != suite["sha256"]:
            raise ValueError("registered_suite_hash_mismatch")


def validate_candidate_source(
    registration_path: Path, registration: dict[str, Any]
) -> None:
    """Ensure candidate binding is derived from the exact registered source bytes."""
    evaluation = registration["evaluation"]
    source_path = Path(evaluation["candidate_binding_source"])
    if sha256_file(source_path) != evaluation["candidate_binding_source_sha256"]:
        raise ValueError("candidate_binding_source_hash_mismatch")
    source = json.loads(source_path.read_text())
    row = source["candidates"][evaluation["candidate"]]
    if row != evaluation["artifact"]:
        raise ValueError("candidate_binding_source_identity_mismatch")
