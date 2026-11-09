"""Deterministic, commit/reveal protocol for a future opening-paired arena.

This module never launches games. Generation is parameterized by a supplied
candidate pool; production use must use the frozen rules-only generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def identities(row: dict[str, Any]) -> tuple[str, str]:
    """Return canonical rule-state and production-float32 identities."""
    state = row["state"]
    canonical = sha256(canonical_bytes(state))
    # The production representation is explicitly little-endian float32.
    import numpy as np

    encoded = np.asarray(row["production_input"], dtype="<f4")
    if encoded.ndim != 1 or not np.isfinite(encoded).all():
        raise ValueError("invalid_production_float32_input")
    return canonical, sha256(encoded.tobytes())


def validate_inventory(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    result = {"canonical": set(), "float32": set()}
    for row in rows:
        can, f32 = identities(row)
        if can in result["canonical"] or f32 in result["float32"]:
            raise ValueError("duplicate_exclusion_identity")
        result["canonical"].add(can)
        result["float32"].add(f32)
    return result


def build_suite(
    pool: list[dict[str, Any]], inventory: list[dict[str, Any]], seed: int, size: int
) -> list[dict[str, Any]]:
    import random

    excluded = validate_inventory(inventory)
    rng = random.Random(seed)
    candidates = list(pool)
    rng.shuffle(candidates)
    chosen: list[dict[str, Any]] = []
    seen = {key: set(values) for key, values in excluded.items()}
    for row in candidates:
        can, f32 = identities(row)
        if can in seen["canonical"] or f32 in seen["float32"]:
            continue
        chosen.append(row)
        seen["canonical"].add(can)
        seen["float32"].add(f32)
        if len(chosen) == size:
            break
    if len(chosen) != size:
        raise ValueError("eligible_pool_shortfall")
    return chosen


def suite_bytes(suite: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in suite)


def commitment(
    suite: bytes, generator: dict[str, Any], inventory: dict[str, Any]
) -> str:
    return sha256(
        canonical_bytes(
            {
                "suite_sha256": sha256(suite),
                "generator": generator,
                "exclusion_inventory": inventory,
            }
        )
    )


def verify(bundle: Path) -> dict[str, Any]:
    """Read-only verifier; paths are relative to bundle, never producer checkout."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    suite = (bundle / "suite.jsonl").read_bytes()
    if sha256(suite) != manifest["suite_sha256"]:
        raise ValueError("suite_bytes_changed")
    rows = [json.loads(line) for line in suite.splitlines() if line]
    if sha256(suite_bytes(rows)) != manifest["suite_sha256"]:
        raise ValueError("suite_order_or_encoding_changed")
    inventory_path = bundle / manifest["inventory_path"]
    inventory_bytes = inventory_path.read_bytes()
    if sha256(inventory_bytes) != manifest["inventory_sha256"]:
        raise ValueError("inventory_bytes_changed")
    inventory_doc = json.loads(inventory_bytes)
    if (
        commitment(suite, manifest["generator"], inventory_doc)
        != manifest["commitment"]
    ):
        raise ValueError("commitment_mismatch")
    excluded = validate_inventory(inventory_doc["rows"])
    seen = {"canonical": set(), "float32": set()}
    for row in rows:
        can, f32 = identities(row)
        if can in excluded["canonical"] or f32 in excluded["float32"]:
            raise ValueError("suite_exclusion_overlap")
        if can in seen["canonical"] or f32 in seen["float32"]:
            raise ValueError("suite_duplicate_identity")
        seen["canonical"].add(can)
        seen["float32"].add(f32)
    freeze = json.loads((bundle / "candidate-freeze.json").read_text())
    if manifest.get("revealed") and not all(
        freeze.get(key)
        for key in (
            "candidate_artifact_sha256",
            "serving_p_artifact_sha256",
            "arena_settings",
            "analysis_rule",
        )
    ):
        raise ValueError("reveal_before_candidate_freeze")
    if (bundle / "consumed.json").exists() and not manifest.get("revealed"):
        raise ValueError("consumed_suite_not_revealed")
    if manifest.get("revealed") and (bundle / "consumed.json").exists():
        consumed = json.loads((bundle / "consumed.json").read_text())
        if consumed.get("commitment") != manifest["commitment"]:
            raise ValueError("consumption_commitment_mismatch")
    return {
        "status": "verified",
        "openings": len(rows),
        "commitment": manifest["commitment"],
        "canonical_identities": len(seen["canonical"]),
        "float32_identities": len(seen["float32"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle), sort_keys=True))


if __name__ == "__main__":
    main()
