#!/usr/bin/env python3
"""Run the non-promoting full-volume seed48 self-play pooling experiment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite import (  # noqa: E402
    run_seed48_fixed_volume_selfplay_pooling as pooling,
)


def main(argv: list[str] | None = None) -> int:
    overrides = {
        "SCHEMA": "azlite_seed48_full_volume_selfplay_pooling_v1",
        "POOL_SCHEMA": "azlite_full_volume_pool_v1",
        "POOL_SEED": 347,
        "ROWS_PER_SOURCE": 70_691,
        "CLASSIFICATION_PREFIX": "full_volume_pooling",
        "EXPERIMENT_DIRECTORY": "alphazero-lite-seed48-full-volume-selfplay-pooling",
        "POOL_FILENAME": "pooled5_full353455.jsonl",
        "PROVENANCE_UNAVAILABLE_CLASSIFICATION": (
            "full_volume_pool_source_provenance_unavailable"
        ),
    }
    original = {name: getattr(pooling, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(pooling, name, value)
        return pooling.main(argv)
    finally:
        for name, value in original.items():
            setattr(pooling, name, value)


if __name__ == "__main__":
    raise SystemExit(main())
