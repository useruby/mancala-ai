#!/usr/bin/env python3
# ruff: noqa: E402
"""PR #268 diagnostic-only A16 Gumbel root-allocation preflight.

This program creates neither replay nor an arena suite. Its corpus is reachable
from the standard start, explicitly marked ineligible for training, and is used
only for the bounded action-regret comparison described in its output manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, sha256_file
from ml.alphazero_lite.gumbel_root_search import (
    EVALUATION_BUDGET,
    GUMBEL_SCALE,
    RootSearchResult,
    run_gumbel_root_search,
    run_puct_root_search,
)
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state
from ml.alphazero_lite.run_fresh_p1_shadow_target_distillation import A16_WORKDIR
from ml.alphazero_lite.self_play import PUCT, standard_start_state, state_hash

SEED = 267001
SEARCH_SEEDS = (267101, 267102, 267103)
STATE_COUNT = 128
CONTINUATION_SIMULATIONS = 2400
ARTIFACT_SHA = "8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34"
WEIGHTS_SHA = "74e160734554dba9ecf4ecf3ac13b3e44526c4f6e7e5e52d9fe8bde0054eb789"
SNAPSHOT_SHA = "f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff"
BOOTSTRAP_SAMPLES = 10_000


def _phase(ply: int) -> str:
    return "opening" if ply < 8 else "midgame" if ply < 24 else "late"


def _features(game: KalahGame, ply: int) -> dict[str, Any]:
    state = game.to_state()
    legal = game.possible_moves()
    consequences = [move_consequence_for_state(state, move) for move in legal]
    return {
        "state": state,
        "state_hash": state_hash(state),
        "prefix_ply": ply,
        "phase": _phase(ply),
        "player": game.current_player,
        "legal_actions": legal,
        "legal_action_count": len(legal),
        "capture_available": any(row["produces_capture"] for row in consequences),
        "extra_turn_available": any(row["gives_extra_turn"] for row in consequences),
        "diagnostic_only": True,
        "not_training_eligible": True,
    }


def build_corpus(*, count: int = STATE_COUNT, seed: int = SEED) -> list[dict[str, Any]]:
    """Deterministically enumerate prefixes then round-robin sample strata."""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    frontier = [KalahGame.from_state(standard_start_state())]
    for ply in range(32):
        next_frontier: list[KalahGame] = []
        for game in frontier:
            if game.over():
                continue
            row = _features(game, ply)
            if len(row["legal_actions"]) > 1 and row["state_hash"] not in seen:
                candidates.append(row)
                seen.add(row["state_hash"])
            for move in game.possible_moves():
                child = game.clone()
                child.move(child.pit_index(move))
                next_frontier.append(child)
        # Bound enumeration without model-selected moves while retaining all
        # phases; each retained child is a legal standard-start prefix.
        random.Random(seed + ply).shuffle(next_frontier)
        frontier = next_frontier[:512]
    if len(candidates) < count:
        raise RuntimeError(
            "prefix enumeration did not produce enough diagnostic states"
        )
    rng = random.Random(seed)
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        key = (
            row["phase"],
            row["player"],
            row["legal_action_count"],
            row["capture_available"],
            row["extra_turn_available"],
        )
        buckets[key].append(row)
    for rows in buckets.values():
        rng.shuffle(rows)
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
                progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda row: row["state_hash"])


def _forced_reference(
    row: dict[str, Any], evaluator: ArtifactEvaluator, action: int, *, seed: int
) -> dict[str, Any]:
    root = KalahGame.from_state(row["state"])
    root_player = root.current_player
    child = root.clone()
    child.move(child.pit_index(action))
    search = PUCT(
        evaluator,
        CONTINUATION_SIMULATIONS,
        1.25,
        random.Random(seed),
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        root_temperature=0.0,
    )
    search.run(child, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    summary = search.root_summary()
    value = float(summary["root_q_value"])
    if child.current_player != root_player:
        value = -value
    return {
        "action": action,
        "value_root_player": value,
        "seed": seed,
        "child_hash": state_hash(child.to_state()),
        "extra_turn_transition": child.current_player == root_player,
        "search_telemetry": {
            "root_q_value": summary["root_q_value"],
            "terminal_leaf_count": summary["terminal_leaf_count"],
            "nonterminal_leaf_count": summary["nonterminal_leaf_count"],
        },
    }


def action_references(
    row: dict[str, Any], evaluator: ArtifactEvaluator
) -> list[dict[str, Any]]:
    return [
        _forced_reference(
            row,
            evaluator,
            action,
            seed=int(
                hashlib.sha256(
                    f"267001:{row['state_hash']}:{action}".encode()
                ).hexdigest()[:16],
                16,
            ),
        )
        for action in row["legal_actions"]
    ]


def _result_row(
    lane: str,
    result: RootSearchResult,
    references: list[dict[str, Any]],
    runtime: float,
) -> dict[str, Any]:
    values = {
        int(item["action"]): float(item["value_root_player"]) for item in references
    }
    ranked = sorted(values, key=lambda action: (-values[action], action))
    selected = result.selected_move
    return {
        "lane": lane,
        "selected_move": selected,
        "selected_reference_value": values[selected],
        "regret": values[ranked[0]] - values[selected],
        "exact_best_action": selected == ranked[0],
        "top_two_reference_action": selected in ranked[:2],
        # Preregistered: any miss at least 0.25 on the bounded [-1, 1] Q scale.
        "catastrophic_miss": values[ranked[0]] - values[selected] >= 0.25,
        "root_visits": result.visits,
        "fraction_legal_actions_visited": sum(v > 0 for v in result.visits.values())
        / len(values),
        "evaluator_calls": result.evaluator_calls,
        "budget_padding_calls": result.budget_padding_calls,
        "runtime_seconds": runtime,
        "all_legal_actions_initially_visited": result.all_legal_actions_initially_visited,
    }


def hierarchical_bootstrap(
    rows: list[dict[str, Any]], *, seed: int = SEED
) -> dict[str, float]:
    """Bootstrap states, then paired seeds nested inside each sampled state."""
    by_state: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_state[row["state_hash"]].append(
            float(row["gumbel"]["regret"] - row["puct"]["regret"])
        )
    values = list(by_state.values())
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        states = rng.integers(0, len(values), len(values))
        samples.append(float(np.mean([rng.choice(values[index]) for index in states])))
    observed = [value for nested in values for value in nested]
    return {
        "mean": float(np.mean(observed)),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }


def classify(rows: list[dict[str, Any]], budget_ok: bool, invariants_ok: bool) -> str:
    if not budget_ok:
        return "gumbel_root_budget_contract_failed"
    if not invariants_ok:
        return "invariant_failure"
    puct = np.mean([row["puct"]["regret"] for row in rows])
    gumbel = np.mean([row["gumbel"]["regret"] for row in rows])
    ci = hierarchical_bootstrap(rows)
    exact_ok = np.mean([row["gumbel"]["exact_best_action"] for row in rows]) >= np.mean(
        [row["puct"]["exact_best_action"] for row in rows]
    )
    catastrophic_ok = np.mean(
        [row["gumbel"]["catastrophic_miss"] for row in rows]
    ) <= np.mean([row["puct"]["catastrophic_miss"] for row in rows])
    subset_regression = any(
        np.mean([r["gumbel"]["catastrophic_miss"] for r in group])
        - np.mean([r["puct"]["catastrophic_miss"] for r in group])
        > 0.02
        for field in (
            "phase",
            "player",
            "legal_action_count",
            "capture_available",
            "extra_turn_available",
        )
        for group in _groups(rows, field).values()
    )
    if subset_regression:
        return "gumbel_root_regresses_subsets"
    if (
        puct > 0
        and gumbel <= 0.85 * puct
        and ci["upper_95"] < 0
        and exact_ok
        and catastrophic_ok
    ):
        return "gumbel_root_interface_qualified"
    return "gumbel_root_no_allocation_gain"


def _groups(rows: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row[field])].append(row)
    return output


def slice_summary(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Report the preregistered regret and agreement slices for both lanes."""
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for field in (
        "phase",
        "player",
        "legal_action_count",
        "capture_available",
        "extra_turn_available",
    ):
        output[field] = {}
        for label, group in _groups(rows, field).items():
            output[field][label] = {
                "n": len(group),
                "puct_mean_regret": float(
                    np.mean([row["puct"]["regret"] for row in group])
                ),
                "gumbel_mean_regret": float(
                    np.mean([row["gumbel"]["regret"] for row in group])
                ),
                "puct_exact_best_agreement": float(
                    np.mean([row["puct"]["exact_best_action"] for row in group])
                ),
                "gumbel_exact_best_agreement": float(
                    np.mean([row["gumbel"]["exact_best_action"] for row in group])
                ),
                "puct_catastrophic_miss_rate": float(
                    np.mean([row["puct"]["catastrophic_miss"] for row in group])
                ),
                "gumbel_catastrophic_miss_rate": float(
                    np.mean([row["gumbel"]["catastrophic_miss"] for row in group])
                ),
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_pr268_gumbel_root_preflight")
    )
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    artifact = A16_WORKDIR / "artifacts/step_0016/artifact"
    snapshot = A16_WORKDIR / "beta095/snapshots/step_0016.pt"
    protected = {
        "weights": sha256_file(artifact / "weights.json"),
        "snapshot": sha256_file(snapshot),
    }
    metadata = json.loads((artifact / "metadata.json").read_text())
    if (
        protected["weights"] != WEIGHTS_SHA
        or protected["snapshot"] != SNAPSHOT_SHA
        or metadata["artifacts"]["weights_sha256"] != ARTIFACT_SHA
    ):
        raise RuntimeError("frozen A16 identity mismatch")
    corpus = build_corpus()
    evaluator = ArtifactEvaluator(artifact)
    rows = []
    for item in corpus:
        references = action_references(item, evaluator)
        for seed in SEARCH_SEEDS:
            game = KalahGame.from_state(item["state"])
            started = time.perf_counter()
            puct = run_puct_root_search(game, evaluator, seed=seed)
            elapsed = time.perf_counter() - started
            started = time.perf_counter()
            gumbel = run_gumbel_root_search(game, evaluator, seed=seed)
            g_elapsed = time.perf_counter() - started
            rows.append(
                item
                | {
                    "seed": seed,
                    "references": references,
                    "puct": _result_row("puct", puct, references, elapsed),
                    "gumbel": _result_row("gumbel", gumbel, references, g_elapsed),
                }
            )
    budget_ok = all(
        r["puct"]["evaluator_calls"] == EVALUATION_BUDGET
        and r["gumbel"]["evaluator_calls"] == EVALUATION_BUDGET
        for r in rows
    )
    invariants_ok = all(
        r["gumbel"]["all_legal_actions_initially_visited"]
        and r["puct"]["selected_move"] in r["legal_actions"]
        and r["gumbel"]["selected_move"] in r["legal_actions"]
        for r in rows
    )
    repeated = []
    for row in rows[: min(12, len(rows))]:
        game = KalahGame.from_state(row["state"])
        repeated.append(
            {
                "state_hash": row["state_hash"],
                "seed": row["seed"],
                "puct": run_puct_root_search(game, evaluator, seed=row["seed"]),
                "gumbel": run_gumbel_root_search(game, evaluator, seed=row["seed"]),
            }
        )
    determinism_ok = all(
        row["puct"]["selected_move"] == repeat["puct"].selected_move
        and row["puct"]["root_visits"] == repeat["puct"].visits
        and row["gumbel"]["selected_move"] == repeat["gumbel"].selected_move
        and row["gumbel"]["root_visits"] == repeat["gumbel"].visits
        for row, repeat in zip(rows[: len(repeated)], repeated, strict=True)
    )
    invariants_ok = invariants_ok and determinism_ok
    classification = classify(rows, budget_ok, invariants_ok)
    summary = {
        "schema": "azlite_pr268_gumbel_root_preflight_v1",
        "classification": classification,
        "frozen_model": {
            "artifact": ARTIFACT_SHA,
            "weights": WEIGHTS_SHA,
            "snapshot": SNAPSHOT_SHA,
        },
        "preregistration": {
            "candidate_set": "all legal root actions (maximum six)",
            "evaluation_budget": EVALUATION_BUDGET,
            "continuation_simulations": CONTINUATION_SIMULATIONS,
            "seeds": list(SEARCH_SEEDS),
            "gumbel_scale": GUMBEL_SCALE,
            "catastrophic_regret_threshold": 0.25,
        },
        "corpus": corpus,
        "results": rows,
        "paired_hierarchical_bootstrap": hierarchical_bootstrap(rows),
        "results_by_preregistered_slice": slice_summary(rows),
        "invariants": {
            "budget_ok": budget_ok,
            "perspective_and_legal_selection_ok": invariants_ok,
            "repeated_small_subset_deterministic": determinism_ok,
            "a16_unchanged": protected
            == {
                "weights": sha256_file(artifact / "weights.json"),
                "snapshot": sha256_file(snapshot),
            },
        },
        "guardrails": {
            "diagnostic_only": True,
            "not_training_eligible": True,
            "arena_run": False,
            "replay_created": False,
            "consumed_suite_registry_modified": False,
        },
    }
    data = (
        REPO_ROOT / "docs/data/alphazero-lite-pr268-gumbel-root-preflight-summary.json"
    )
    report = REPO_ROOT / "docs/alphazero-lite-pr268-gumbel-root-preflight-results.md"
    data.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report.write_text(
        f"# PR #268 Gumbel Root Preflight\n\n**Classification:** `{classification}`\n\nThis is a diagnostic-only, non-training corpus; no arena or sealed suite was consumed.\n\n```json\n{json.dumps({k: summary[k] for k in ('classification', 'frozen_model', 'preregistration', 'paired_hierarchical_bootstrap', 'invariants', 'guardrails')}, indent=2, sort_keys=True)}\n```\n"
    )
    print(classification)


if __name__ == "__main__":
    main()
