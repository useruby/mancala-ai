"""Deterministic opponent pool utilities for self-play."""

from __future__ import annotations

import json
from pathlib import Path


def deterministic_pool_index(
    *, pool_size: int, base_seed: int, game_index: int, worker_id: int
) -> int:
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")

    mixed_seed = (
        (int(base_seed) * 1_000_003) + int(game_index) + (int(worker_id) * 9_973)
    )
    return mixed_seed % int(pool_size)


def sample_opponent_checkpoint(
    checkpoints: list[str],
    *,
    base_seed: int,
    game_index: int,
    worker_id: int,
) -> str | None:
    if not checkpoints:
        return None

    index = deterministic_pool_index(
        pool_size=len(checkpoints),
        base_seed=base_seed,
        game_index=game_index,
        worker_id=worker_id,
    )
    return checkpoints[index]


def load_opponent_checkpoints(config_path: str | None) -> list[str]:
    if not config_path:
        return []

    config_file = Path(config_path)
    if not config_file.exists():
        raise ValueError(f"opponent pool config file does not exist: {config_file}")

    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"opponent pool config file is not valid JSON: {config_file}"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"could not read opponent pool config file: {config_file}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError("opponent pool config root must be a JSON object")

    if "checkpoints" not in config:
        raise ValueError("opponent pool config must include a checkpoints field")
    entries = config.get("checkpoints")
    if not isinstance(entries, list):
        raise ValueError("opponent pool config must provide a checkpoints list")
    if not entries:
        raise ValueError("opponent pool config checkpoints list must not be empty")

    resolved: list[str] = []
    for value in entries:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("opponent pool checkpoints must be non-empty strings")
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = config_file.parent / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise ValueError(
                f"opponent pool config file {config_file} references missing checkpoint: {candidate}"
            )
        if not candidate.is_file():
            raise ValueError(
                f"opponent pool config file {config_file} references a non-file checkpoint: {candidate}"
            )
        resolved.append(str(candidate))
    return resolved
