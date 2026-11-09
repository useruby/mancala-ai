"""Deterministic, commit/reveal protocol for a future opening-paired arena.

This module never launches games. Generation is parameterized by a supplied
candidate pool; production use must use the frozen rules-only generator.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed482_fresh_confirmation import (
    canonical_identity,
    float32_identity,
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def identities(row: dict[str, Any]) -> tuple[str, str]:
    """Derive both identities from rule state and the registered Kalah encoder."""
    state = row["state"]
    return canonical_identity(state), float32_identity(state)


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


def lifecycle_event(
    kind: str, payload: dict[str, Any], previous: str
) -> dict[str, Any]:
    event = {"kind": kind, "previous": previous, "payload": payload}
    event["event_sha256"] = sha256(canonical_bytes(event))
    return event


def event_log_bytes(events: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(event) + b"\n" for event in events)


def verify(bundle: Path, trusted_anchor_sha256: str) -> dict[str, Any]:
    """Verify a revealed bundle against an independently retained event-log hash.

    The trusted anchor is delivered out-of-band by the commitment custodian; a
    hash stored only inside the bundle would not establish event order.
    """
    manifest = json.loads((bundle / "manifest.json").read_text())
    suite = (bundle / "suite.jsonl").read_bytes()
    if sha256(suite) != manifest["suite_sha256"]:
        raise ValueError("suite_bytes_changed")
    rows = [json.loads(line) for line in suite.splitlines() if line]
    if sha256(suite_bytes(rows)) != manifest["suite_sha256"]:
        raise ValueError("suite_order_or_encoding_changed")
    if len(rows) != manifest["generator"].get("size"):
        raise ValueError("suite_declared_size_mismatch")
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
    generator = manifest["generator"]
    if generator.get("pool_sha256") != manifest["pool_sha256"]:
        raise ValueError("generator_pool_binding_mismatch")
    pool_bytes = (bundle / manifest["pool_path"]).read_bytes()
    if sha256(pool_bytes) != manifest["pool_sha256"]:
        raise ValueError("synthetic_pool_bytes_changed")
    pool = [json.loads(line) for line in pool_bytes.splitlines() if line]
    rebuilt = build_suite(
        pool, inventory_doc["rows"], generator["seed"], generator["size"]
    )
    if suite_bytes(rebuilt) != suite:
        raise ValueError("suite_rebuild_mismatch")
    seen = {"canonical": set(), "float32": set()}
    for row in rows:
        can, f32 = identities(row)
        if can in excluded["canonical"] or f32 in excluded["float32"]:
            raise ValueError("suite_exclusion_overlap")
        if can in seen["canonical"] or f32 in seen["float32"]:
            raise ValueError("suite_duplicate_identity")
        seen["canonical"].add(can)
        seen["float32"].add(f32)
    event_bytes = (bundle / "lifecycle.jsonl").read_bytes()
    anchor_bytes = (bundle / "anchor.json").read_bytes()
    if sha256(anchor_bytes) != trusted_anchor_sha256:
        raise ValueError("untrusted_lifecycle_anchor")
    anchor = json.loads(anchor_bytes)
    if anchor.get("lifecycle_sha256") != sha256(event_bytes):
        raise ValueError("anchor_lifecycle_mismatch")
    events = [json.loads(line) for line in event_bytes.splitlines() if line]
    expected_kinds = ["commitment", "candidate_freeze", "reveal"]
    if len(events) == 4:
        expected_kinds.append("consumption")
    if [event["kind"] for event in events] != expected_kinds:
        raise ValueError("lifecycle_event_order")
    previous = "0" * 64
    for event in events:
        expected = lifecycle_event(event["kind"], event["payload"], previous)
        if event != expected:
            raise ValueError("lifecycle_chain_invalid")
        previous = event["event_sha256"]
    checkpoints = anchor.get("checkpoints", [])
    if len(checkpoints) != len(events) or any(
        point.get("event_sha256") != event["event_sha256"]
        for point, event in zip(checkpoints, events, strict=True)
    ):
        raise ValueError("anchor_checkpoint_mismatch")
    times = [
        dt.datetime.fromisoformat(point["anchored_at_utc"]) for point in checkpoints
    ]
    if any(time.utcoffset() != dt.timedelta(0) for time in times):
        raise ValueError("anchor_timestamp_not_utc")
    if any(left >= right for left, right in zip(times, times[1:])):
        raise ValueError("anchor_checkpoint_order")
    commit_event, freeze_event, reveal_event = [
        event["payload"] for event in events[:3]
    ]
    if commit_event.get("commitment") != manifest["commitment"]:
        raise ValueError("lifecycle_commitment_mismatch")
    freeze = freeze_event.get("freeze", {})
    if freeze_event.get("commitment") != manifest["commitment"] or not all(
        freeze.get(key)
        for key in (
            "candidate_artifact_sha256",
            "serving_p_artifact_sha256",
            "arena_settings",
            "analysis_rule",
        )
    ):
        raise ValueError("candidate_freeze_not_bound")
    if reveal_event.get("commitment") != manifest["commitment"]:
        raise ValueError("reveal_commitment_mismatch")
    if reveal_event.get("suite_sha256") != manifest["suite_sha256"]:
        raise ValueError("reveal_suite_mismatch")
    if (
        reveal_event.get("seed") != generator["seed"]
        or reveal_event.get("pool_sha256") != manifest["pool_sha256"]
    ):
        raise ValueError("reveal_generator_inputs_mismatch")
    consumed = len(events) == 4
    if consumed and events[3]["payload"].get("commitment") != manifest["commitment"]:
        raise ValueError("consumption_commitment_mismatch")
    return {
        "status": "verified",
        "held_back_eligible": not consumed,
        "consumed": consumed,
        "openings": len(rows),
        "commitment": manifest["commitment"],
        "canonical_identities": len(seen["canonical"]),
        "float32_identities": len(seen["float32"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--trusted-anchor-sha256",
        required=True,
        help="SHA-256 of anchor.json obtained independently from its custodian",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle, args.trusted_anchor_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
