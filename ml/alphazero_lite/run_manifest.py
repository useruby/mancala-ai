"""Run manifest helpers for AlphaZero-lite pipeline iterations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def build_manifest(*, run_id: str, iteration: int, seed: int, config_path: str, parent_version: str | None, status: str, notes: dict | None = None) -> dict:
    return {
        "schema": "azlite_run_manifest_v1",
        "run_id": run_id,
        "iteration": int(iteration),
        "seed": int(seed),
        "config_path": config_path,
        "parent_version": parent_version,
        "status": status,
        "notes": notes or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
