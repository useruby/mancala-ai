#!/usr/bin/env python3
"""Freeze an evaluation-only exact descendant expansion of the R61 anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

# ruff: noqa: E402
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_internal_cluster_learning_dynamics import entry

SCHEMA = "azlite_r61_expanded_exact_subcluster_v1"
DEPTH = 4
ORIGINAL = ROOT / "docs/data/alphazero-lite-internal-cluster-frozen-evaluation-set.json"
LABELS = (
    ROOT / "docs/data/alphazero-lite-true-outcome-subtree-attribution-audit-labels.json"
)


def sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def descendants(anchor: dict[str, Any], depth: int) -> list[tuple[int, dict[str, Any]]]:
    """Return unique legal descendants in BFS depth/key order without search."""
    seen = {canonical_state_key(anchor)}
    frontier = [anchor]
    result = [(0, anchor)]
    for current_depth in range(1, depth + 1):
        next_states: dict[str, dict[str, Any]] = {}
        for state in frontier:
            game = KalahGame.from_state(state)
            for action in game.possible_moves():
                child = game.clone()
                child.move(child.pit_index(action))
                value = child.to_state()
                key = canonical_state_key(value)
                if key not in seen:
                    seen.add(key)
                    next_states[key] = value
        frontier = [next_states[key] for key in sorted(next_states)]
        result.extend((current_depth, state) for state in frontier)
    return result


def build(original: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    anchor = next(
        row for row in original["entries"] if row["membership"] == "cluster_anchor"
    )
    records = []
    excluded_all_optimal = 0
    for depth, state in descendants(anchor["state"], DEPTH):
        key = canonical_state_key(state)
        if key not in labels:
            continue
        membership = "cluster_anchor" if depth == 0 else "expanded_exact_descendant"
        item = entry(
            state=state,
            label=labels[key],
            membership=membership,
            provenance=f"R61 deterministic legal descendant depth {depth}; exact inherited label",
            degrading_actions=anchor["degrading_actions"] if depth == 0 else None,
        )
        # Canonical policy margin requires an outcome-suboptimal legal action.
        if len(item["exact_outcome_optimal_actions"]) == len(item["legal_actions"]):
            excluded_all_optimal += 1
            continue
        item["expansion_depth"] = depth
        records.append(item)
    # Preserve the original matched control unchanged for specificity metrics;
    # it is never considered a member of the expanded failure subcluster.
    records.extend(
        row
        for row in original["entries"]
        if row["membership"] == "matched_control"
    )
    records.sort(
        key=lambda row: (
            row["membership"] == "matched_control",
            row.get("expansion_depth", 0),
            row["state_sha256"],
        )
    )
    if records[0]["state_sha256"] != anchor["state_sha256"] or len(records) <= 1:
        raise RuntimeError("expanded_exact_subcluster_invalid")
    return {
        "schema": SCHEMA,
        "source_manifest_sha256": sha256(original),
        "exact_label_artifact_sha256": hashlib.sha256(LABELS.read_bytes()).hexdigest(),
        "depth": DEPTH,
        "entries": records,
        "set_sha256": sha256(records),
        "excluded_all_legal_actions_optimal": excluded_all_optimal,
        "training_injection": False,
        "guardrails": {
            "training": False,
            "replay_mutation": False,
            "search": False,
            "promotion": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    original = json.loads(ORIGINAL.read_text())
    labels = json.loads(LABELS.read_text())["labels"]
    result = build(original, labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
