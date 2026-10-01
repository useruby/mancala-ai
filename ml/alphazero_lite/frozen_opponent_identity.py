"""Preflight identity checks for registered frozen arena opponents."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_frozen_opponent_identity(
    artifact: Path, opponent: dict[str, Any], runtime_contract: dict[str, Any]
) -> None:
    """Reject changed model/runtime artifacts before any arena process launch."""
    if opponent.get("artifact") != str(artifact):
        raise ValueError("opponent_path_mismatch")
    for name, field in (
        ("weights.json", "weights_sha256"),
        ("metadata.json", "metadata_sha256"),
        ("search_policy.json", "sidecar_sha256"),
    ):
        if sha256_file(artifact / name) != opponent.get(field):
            raise ValueError(f"opponent_hash_mismatch:{name}")
    for path_key, hash_key in (
        ("exact_root_native_probe", "native_probe_sha256"),
        ("exact_root_tablebase", "tablebase_sha256"),
    ):
        registered = runtime_contract.get(f"{path_key}_sha256")
        if sha256_file(
            Path(runtime_contract[path_key])
        ) != registered or registered != opponent.get(hash_key):
            raise ValueError(f"runtime_identity_mismatch:{path_key}")
