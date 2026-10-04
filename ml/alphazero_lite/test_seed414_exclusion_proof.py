from __future__ import annotations

import json

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite.seed414_exclusion_proof import (
    BASE,
    NEW_411,
    OUT,
    build_extension,
    load_jsonl,
)


def test_extension_preserves_historical_union_and_reconciles_counts() -> None:
    base = json.loads(BASE.read_text())
    extension = build_extension()
    assert set(base["excluded_state_identities"]) <= set(
        extension["excluded_state_identities"]
    )
    assert extension["deduplication"]["union_reconciles"]
    assert extension["union_identity_count"] == len(
        extension["excluded_state_identities"]
    )
    assert extension["excluded_state_identities"] == sorted(
        extension["excluded_state_identities"]
    )


def test_preliminary_fallback_and_eight_411_trajectories_are_excluded() -> None:
    proof = build_extension()
    excluded = set(proof["excluded_state_identities"])
    registration = json.loads(
        (OUT.parent / "seed398-paired-first-action" / "registration.json").read_text()
    )
    starts = {row["opening_index"]: row["state"] for row in registration["states"]}
    rows = load_jsonl(NEW_411)
    assert len(rows) == 8
    for wrapped in rows:
        state = starts[wrapped["opening_index"]]
        assert suites.canonical_key(state) in excluded
        for move in wrapped["outcome"]["trajectory"]:
            assert suites.canonical_key(move["state"]) in excluded


def test_known_historical_suite_state_remains_excluded() -> None:
    proof = build_extension()
    known = load_jsonl(
        OUT.parent / "seed398-policy-value-composition" / "seed398-openings-v2.jsonl"
    )[0]
    assert suites.canonical_key(known["state"]) in set(
        proof["excluded_state_identities"]
    )
