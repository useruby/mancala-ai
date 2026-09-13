#!/usr/bin/env python3
"""Read-only cluster provenance audit for capture_available-018."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

# ruff: noqa: E402
from ml.alphazero_lite.build_train_only_forensic_suite_from_selfplay import decode_state
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state
from ml.alphazero_lite.run_internal_family_replay_provenance_audit import (
    FAMILY,
    PRIMARY_CASE,
    source_inventory,
)
from ml.alphazero_lite.run_uniform1200_replay_distribution_audit import (
    entropy,
    sha256_file,
)

PR304 = ROOT / "docs/data/alphazero-lite-internal-family-replay-provenance-audit.json"
PR303 = ROOT / "docs/data/alphazero-lite-internal-first-degradation-family-audit.json"
PLAN = ROOT / "ml/alphazero_lite/configs/uniform1200_seed_confirmation.json"


def stable_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def cluster_signature(state: dict[str, Any]) -> str:
    """Rules-only local tactical signature; no outcomes or hand-picked examples."""
    game = KalahGame.from_state(state)
    legal = game.possible_moves()
    consequences = {a: move_consequence_for_state(state, a) for a in legal}
    own_zero_mask = "".join(
        "1" if value == 0 else "0" for value in state["player_pits"]
    )
    return json.dumps(
        {
            "player": state["current_player"],
            "legal_count": len(legal),
            "own_zero_mask": own_zero_mask,
            "extra_turn_actions": [
                a for a in legal if consequences[a]["gives_extra_turn"]
            ],
            "capture_actions": [
                a for a in legal if consequences[a]["produces_capture"]
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def build_clusters(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for event in events:
        if event["case"] == PRIMARY_CASE and event["family"] == FAMILY:
            grouped[cluster_signature(event["state"])][
                event["canonical_pre_action_state"]
            ].append(event)
    rows = []
    for signature, states in grouped.items():
        keys = sorted(states)
        rows.append(
            {
                "definition": signature,
                "states": keys,
                "events": sum(len(values) for values in states.values()),
                "opportunities": sum(
                    values[0]["opportunities"] for values in states.values()
                ),
                "tie_break_sha256": stable_hash("\n".join(keys)),
            }
        )
    return sorted(
        rows, key=lambda row: (-row["events"], row["tie_break_sha256"], row["states"])
    )


def target_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    policy = [statistics.fmean(row["policy"][a] for row in rows) for a in range(6)]
    return {
        "rows": len(rows),
        "policy": policy,
        "entropy": entropy(policy),
        "value": statistics.fmean(float(row["value"]) for row in rows),
    }


def scan_neighbors(
    source: dict[str, Any], signature: str, selected: set[str]
) -> dict[str, Any]:
    if source["availability"] != "verified":
        return {
            "availability": "historical_source_unavailable",
            "exact": "unknown",
            "neighbors": "unknown",
        }
    exact, neighbors = [], []
    with Path(source["artifact_path"]).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            state = decode_state(row["state"])
            key = canonical_state_key(state)
            record = {
                "canonical_state": key,
                "line": line_number,
                "move_index": row.get("move_index"),
                "player": state["current_player"],
                "value": row["value"],
                "policy": [float(value) for value in row["policy"]],
            }
            if key in selected:
                exact.append(record)
            elif cluster_signature(state) == signature:
                neighbors.append(record)
    return {
        "availability": "verified",
        "exact": {
            "rows": len(exact),
            "effective_rows": len(exact) * source["replay_weight"],
            "targets": target_summary(exact),
        },
        "neighbors": {
            "rows": len(neighbors),
            "effective_rows": len(neighbors) * source["replay_weight"],
            "targets": target_summary(neighbors),
        },
    }


def classify(
    state_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]
) -> str:
    exact = sum(row["exact_effective_rows"] for row in source_rows)
    parents = sum(row["parent_effective_rows"] for row in source_rows)
    neighbors = sum(row["neighbor_effective_rows"] for row in source_rows)
    if exact and exact >= parents and exact >= neighbors:
        return "exact_replay_target_provenance_identified"
    if parents and parents >= neighbors:
        return "one_ply_replay_provenance_identified"
    if neighbors:
        return "structural_neighbor_provenance_identified"
    if all(
        row["raw_policy"]["parent"]["top_is_outcome_optimal"] is False
        for row in state_rows
    ):
        return "checkpoint_inherited_prior_failure"
    return "no_replay_local_provenance_identified"


def run() -> dict[str, Any]:
    prior, inherited = json.loads(PR304.read_text()), json.loads(PR303.read_text())
    if (
        prior["inherited"]["classification"]
        != "repeated_internal_prior_failure_identified"
    ):
        raise RuntimeError("unexpected PR #304 lineage")
    clusters = build_clusters(inherited["events"])
    selected = clusters[0]
    previous = {
        row["canonical_state"]: row for row in prior["primary_failure_states"]["states"]
    }
    state_rows = [previous[key] for key in selected["states"]]
    plan = json.loads(PLAN.read_text())
    inventories = prior["source_inventory"]
    source_rows = []
    for lane, sources in inventories.items():
        # Reconstruct through PR #304's manifest semantics and re-verify every byte.
        manifest_path = Path(sources[0]["artifact_path"]).parent / "run_manifest.json"
        verified = source_inventory(json.loads(manifest_path.read_text()), plan)
        for source in verified:
            scan = scan_neighbors(
                source, selected["definition"], set(selected["states"])
            )
            parent_rows = sum(
                coverage["parent_occurrences"]
                for state in state_rows
                for coverage in state["replay_coverage"][lane]
                if coverage["source"] == source["name"]
                and coverage["availability"] == "verified"
            )
            source_rows.append(
                {
                    "lane": lane,
                    "source": source["name"],
                    "sha256": source["sha256"],
                    "availability": source["availability"],
                    "replay_weight": source["replay_weight"],
                    "exact_effective_rows": scan["exact"]
                    if scan["exact"] == "unknown"
                    else scan["exact"]["effective_rows"],
                    "neighbor_effective_rows": scan["neighbors"]
                    if scan["neighbors"] == "unknown"
                    else scan["neighbors"]["effective_rows"],
                    "parent_effective_rows": "unknown"
                    if source["availability"] != "verified"
                    else parent_rows * source["replay_weight"],
                    "targets": scan,
                }
            )
    classification = classify(
        state_rows,
        [row for row in source_rows if isinstance(row["exact_effective_rows"], int)],
    )
    verified = [
        row for row in source_rows if isinstance(row["exact_effective_rows"], int)
    ]
    neighbor_total = sum(row["neighbor_effective_rows"] for row in verified)
    accounting = {
        "exact_state_effective_rows": sum(
            row["exact_effective_rows"] for row in verified
        ),
        "one_ply_effective_rows": sum(row["parent_effective_rows"] for row in verified),
        "structural_neighbor_effective_rows": neighbor_total,
        "dynamic_neighbor_fraction": None
        if not neighbor_total
        else sum(
            row["neighbor_effective_rows"]
            for row in verified
            if row["source"] == "dynamic_current_self_play"
        )
        / neighbor_total,
        "uniform_bad_policy_event_fraction": sum(row["events"] for row in state_rows)
        / sum(cluster["events"] for cluster in clusters),
    }
    # Neighbor membership alone does not establish that its target taught the bad action.
    intervention = classification in {
        "exact_replay_target_provenance_identified",
        "one_ply_replay_provenance_identified",
    }
    return {
        "schema": "azlite_internal_state_cluster_provenance_audit_v1",
        "read_only": {
            "training": False,
            "self_play": False,
            "search": False,
            "promotion": False,
        },
        "inherited": {
            "pr304_sha256": sha256_file(PR304),
            "pr303_sha256": sha256_file(PR303),
        },
        "cluster_ranking": clusters,
        "selected_cluster": selected,
        "states": state_rows,
        "source_accounting": source_rows,
        "cluster_accounting": accounting,
        "provenance_classification": classification,
        "training_data_intervention_justified": intervention,
        "next_experiment": "Exactly one next experiment: construct a frozen cluster evaluation set and measure policy change under normal iterative self-play before changing replay content.",
    }


def markdown(result: dict[str, Any]) -> str:
    cluster = result["selected_cluster"]
    lines = [
        "# Internal State-Cluster Provenance Audit",
        "",
        "## Selected Cluster",
        "",
        f"Definition: `{cluster['definition']}`.",
        f"Deterministic selection: ranked by selected-family degradation events, then SHA-256 of sorted canonical states; top cluster has {cluster['events']} events and {cluster['opportunities']} opportunities.",
        "",
        "## State Evidence",
        "",
        "| State | Events | Parent top | Control top | Uniform top |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| `{row['canonical_state']}` | {row['events']} | {row['raw_policy']['parent']['top_action']} | {row['raw_policy']['control']['top_action']} | {row['raw_policy']['uniform1200']['top_action']} |"
        for row in result["states"]
    ]
    lines += [
        "",
        "## Replay Accounting",
        "",
        "| Lane | Source | Exact effective rows | Parent effective rows | Neighbor effective rows |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {row['lane']} | `{row['source']}` | {row['exact_effective_rows']} | {row['parent_effective_rows']} | {row['neighbor_effective_rows']} |"
        for row in result["source_accounting"]
    ]
    lines += [
        "",
        "## Conclusion",
        "",
        f"Provenance classification: `{result['provenance_classification']}`.",
        f"Training-data intervention justified: `{result['training_data_intervention_justified']}`.",
        "",
        "## Next Experiment",
        "",
        result["next_experiment"],
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run()
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.report.write_text(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
