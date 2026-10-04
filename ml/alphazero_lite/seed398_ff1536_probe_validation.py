"""Pure checks for the frozen FF1536 public probe bundle."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def derive_ff1536_actions(
    states: list[dict[str, Any]], raw: dict[str, Any]
) -> list[dict[str, Any]]:
    """Validate FF probe identities and return the ordered registered action rows."""
    by_opening: dict[int, dict[str, Any]] = {}
    for probe in raw["probes"]:
        if probe["treatment"] != "FF":
            continue
        opening = probe["opening_index"]
        if opening in by_opening:
            raise ValueError("duplicate_frozen_ff_probe")
        by_opening[opening] = probe

    require(
        len(by_opening) == len(states)
        and set(by_opening) == {state["opening_index"] for state in states},
        "frozen_action_state_coverage_mismatch",
    )
    result = []
    for state in states:
        opening = state["opening_index"]
        probe = by_opening[opening]
        require(
            probe["opening_index"] == opening
            and probe["state_hash"] == state["state_hash"],
            f"frozen_ff_probe_state_identity:{opening}",
        )
        final = [
            snapshot
            for snapshot in probe["snapshots"]
            if snapshot["simulations"] == 1536
        ]
        require(len(final) == 1, f"frozen_action_snapshot_count:{opening}")
        action = final[0]["selected_action"]
        require(
            probe["final_action"] == action,
            f"frozen_action_snapshot_mismatch:{opening}",
        )
        require(action in state["legal_actions"], f"frozen_action_illegal:{opening}")
        result.append(
            {
                "opening_index": opening,
                "state_hash": state["state_hash"],
                "ff1536_action": action,
            }
        )
    return result


def validate_probe_bundle(
    raw_bytes: bytes, expected_probe_sha256: str, registration_bytes: bytes
) -> dict[str, Any]:
    """Require the exact registered probe bytes and their parent-registration binding."""
    require(
        hashlib.sha256(raw_bytes).hexdigest() == expected_probe_sha256,
        "frozen_probe_binding",
    )
    raw = json.loads(raw_bytes)
    require(
        raw["registration_sha256"] == hashlib.sha256(registration_bytes).hexdigest(),
        "probe_registration_binding",
    )
    return raw


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
