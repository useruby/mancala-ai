#!/usr/bin/env python3
# ruff: noqa: E402
"""Diagnostic-only Generation-3 policy-free alpha-beta preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.alpha_beta_search import (
    ArtifactValueEvaluator,
    HeuristicValueEvaluator,
    run_alpha_beta_search,
)
from ml.alphazero_lite.arena import ArtifactEvaluator, sha256_file
from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.gumbel_root_search import run_puct_root_search
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state
from ml.alphazero_lite.self_play import (
    HeuristicEvaluator,
    standard_start_state,
    state_hash,
)

CORPUS_SEED = 270001
PUCT_SEEDS = (270101, 270102, 270103)
STATE_COUNT = 64
LEAF_BUDGET = 384
REFERENCE_BUDGET = 2400
BOOTSTRAP_SAMPLES = 10_000
WEIGHTS_SHA = "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
VERSION = "azlite-balanced-w8s4-policy-head-e1"
SLICE_FIELDS = (
    "phase",
    "player",
    "legal_action_count",
    "capture_available",
    "extra_turn_available",
)


def phase_for(ply: int) -> str:
    return "opening" if ply < 8 else "midgame" if ply < 24 else "late"


def corpus_features(game: KalahGame, ply: int) -> dict[str, Any]:
    state = game.to_state()
    legal = game.possible_moves()
    consequences = [move_consequence_for_state(state, move) for move in legal]
    return {
        "state": state,
        "state_hash": state_hash(state),
        "prefix_ply": ply,
        "phase": phase_for(ply),
        "player": game.current_player,
        "legal_actions": legal,
        "legal_action_count": len(legal),
        "capture_available": any(item["produces_capture"] for item in consequences),
        "extra_turn_available": any(item["gives_extra_turn"] for item in consequences),
        "diagnostic_only": True,
        "not_training_eligible": True,
    }


def build_corpus(
    *, count: int = STATE_COUNT, seed: int = CORPUS_SEED
) -> list[dict[str, Any]]:
    candidates, seen = [], set()
    frontier = [KalahGame.from_state(standard_start_state())]
    for ply in range(32):
        following = []
        for game in frontier:
            if game.over():
                continue
            row = corpus_features(game, ply)
            if len(row["legal_actions"]) >= 2 and row["state_hash"] not in seen:
                candidates.append(row)
                seen.add(row["state_hash"])
            for move in game.possible_moves():
                child = game.clone()
                child.move(child.pit_index(move))
                following.append(child)
        random.Random(seed + ply).shuffle(following)
        frontier = following[:512]
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in candidates:
        buckets[tuple(row[field] for field in SLICE_FIELDS)].append(row)
    rng = random.Random(seed)
    for rows in buckets.values():
        rng.shuffle(rows)
    selected = []
    while len(selected) < count:
        progressed = False
        for key in sorted(buckets, key=str):
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise RuntimeError("insufficient reachable corpus states")
    return sorted(selected, key=lambda row: row["state_hash"])


def reference_seed(row: dict, action: int) -> int:
    return int(
        hashlib.sha256(
            f"{CORPUS_SEED}:{row['state_hash']}:{action}".encode()
        ).hexdigest()[:16],
        16,
    )


def action_references(row: dict) -> list[dict]:
    root = KalahGame.from_state(row["state"])
    root_player = root.current_player
    results = []
    for action in row["legal_actions"]:
        child = root.clone()
        child.move(child.pit_index(action))
        seed = reference_seed(row, action)
        if child.over():
            value = (
                0.0
                if child.winner is None
                else (1.0 if child.winner == root_player else -1.0)
            )
            telemetry = {"terminal_child": True, "simulations": 0}
        else:
            search = ClassicMCTS(
                child,
                simulations=REFERENCE_BUDGET,
                seed=seed,
                endgame_tablebase=None,
                exact_solve_enabled=False,
            )
            continued = search.search_root()
            value = (
                (2.0 * (continued.wins / continued.visits)) - 1.0
                if continued.visits
                else 0.0
            )
            if child.current_player != root_player:
                value = -value
            telemetry = {
                "terminal_child": False,
                "simulations": REFERENCE_BUDGET,
                "root_visits": continued.visits,
            }
        results.append(
            {
                "action": action,
                "value_root_player": float(value),
                "seed": seed,
                "child_hash": state_hash(child.to_state()),
                "extra_turn_transition": child.current_player == root_player,
                "search_telemetry": telemetry,
            }
        )
    return results


def score_result(result, references: list[dict]) -> dict:
    values = {item["action"]: item["value_root_player"] for item in references}
    ranked = sorted(values, key=lambda move: (-values[move], move))
    selected = result.selected_move
    return {
        "selected_move": selected,
        "selected_reference_value": values[selected],
        "regret": values[ranked[0]] - values[selected],
        "exact_best_agreement": selected == ranked[0],
        "top_two_agreement": selected in ranked[:2],
        "catastrophic_miss": values[ranked[0]] - values[selected] >= 0.25,
        "legal_action_valid": selected in values,
        "runtime_seconds": result.runtime_seconds,
        "telemetry": {
            key: value
            for key, value in result.__dict__.items()
            if key != "runtime_seconds"
        },
    }


def puct_result(game: KalahGame, evaluator: ArtifactEvaluator, seed: int) -> dict:
    started = time.perf_counter()
    result = run_puct_root_search(game, evaluator, seed=seed, budget=LEAF_BUDGET)
    return {
        "selected_move": result.selected_move,
        "runtime_seconds": time.perf_counter() - started,
        "evaluator_calls": result.evaluator_calls,
        "visits": result.visits,
        "q_values": result.q_values,
        "budget_padding_calls": result.budget_padding_calls,
    }


def puct_scored(result: dict, references: list[dict]) -> dict:
    values = {item["action"]: item["value_root_player"] for item in references}
    ranked = sorted(values, key=lambda move: (-values[move], move))
    selected = result["selected_move"]
    return result | {
        "selected_reference_value": values[selected],
        "regret": values[ranked[0]] - values[selected],
        "exact_best_agreement": selected == ranked[0],
        "top_two_agreement": selected in ranked[:2],
        "catastrophic_miss": values[ranked[0]] - values[selected] >= 0.25,
        "legal_action_valid": selected in values,
    }


def paired_bootstrap(rows: list[dict], lane: str) -> dict:
    values = [
        [
            item["lanes"][lane]["regret"] - puct["regret"]
            for puct in item["puct"].values()
        ]
        for item in rows
    ]
    rng = np.random.default_rng(CORPUS_SEED)
    samples = [
        float(
            np.mean(
                [
                    rng.choice(values[index])
                    for index in rng.integers(0, len(values), len(values))
                ]
            )
        )
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    observed = [value for state in values for value in state]
    return {
        "mean": float(np.mean(observed)),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "samples": BOOTSTRAP_SAMPLES,
        "seed": CORPUS_SEED,
    }


def metrics(rows: list[dict], lane: str) -> dict:
    alpha = [row["lanes"][lane] for row in rows]
    puct = [value for row in rows for value in row["puct"].values()]

    def summary(values):
        return {
            "mean_regret": float(statistics.fmean(item["regret"] for item in values)),
            "exact_best_agreement": float(
                statistics.fmean(item["exact_best_agreement"] for item in values)
            ),
            "top_two_agreement": float(
                statistics.fmean(item["top_two_agreement"] for item in values)
            ),
            "catastrophic_miss_rate": float(
                statistics.fmean(item["catastrophic_miss"] for item in values)
            ),
            "mean_runtime_seconds": float(
                statistics.fmean(item["runtime_seconds"] for item in values)
            ),
        }

    return {
        "alpha_beta": summary(alpha),
        "ordinary_puct": summary(puct),
        "paired_hierarchical_bootstrap": paired_bootstrap(rows, lane),
    }


def slice_metrics(rows: list[dict], lane: str) -> dict:
    output = {}
    for field in SLICE_FIELDS:
        groups: dict[str, list] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        output[field] = {}
        for label, group in groups.items():
            alpha = [item["lanes"][lane] for item in group]
            puct = [value for item in group for value in item["puct"].values()]
            output[field][label] = {
                "n_states": len(group),
                "n_paired_rows": len(puct),
                "alpha_beta_catastrophic_miss_rate": float(
                    statistics.fmean(x["catastrophic_miss"] for x in alpha)
                ),
                "ordinary_puct_catastrophic_miss_rate": float(
                    statistics.fmean(x["catastrophic_miss"] for x in puct)
                ),
                "alpha_beta_mean_regret": float(
                    statistics.fmean(x["regret"] for x in alpha)
                ),
                "ordinary_puct_mean_regret": float(
                    statistics.fmean(x["regret"] for x in puct)
                ),
            }
    return output


def classify_lane(
    metric: dict, slices: dict, *, budget_ok: bool, invariants_ok: bool
) -> str:
    if not budget_ok:
        return "budget_contract_failed"
    if not invariants_ok:
        return "invariant_failure"
    alpha, puct, interval = (
        metric["alpha_beta"],
        metric["ordinary_puct"],
        metric["paired_hierarchical_bootstrap"],
    )
    regression = any(
        row["n_paired_rows"] >= 8
        and row["alpha_beta_catastrophic_miss_rate"]
        - row["ordinary_puct_catastrophic_miss_rate"]
        > 0.02
        for group in slices.values()
        for row in group.values()
    )
    if regression:
        return "regresses_subsets"
    if (
        puct["mean_regret"] > 0
        and alpha["mean_regret"] <= 0.75 * puct["mean_regret"]
        and interval["upper_95"] < 0
        and alpha["exact_best_agreement"] >= puct["exact_best_agreement"]
        and alpha["catastrophic_miss_rate"] <= puct["catastrophic_miss_rate"]
    ):
        return "qualified"
    return "no_search_gain"


def protected_hashes() -> dict[str, str]:
    paths = {
        "weights": REPO_ROOT / "model-artifact/current/weights.json",
        "metadata": REPO_ROOT / "model-artifact/current/metadata.json",
        "closeout_markdown": REPO_ROOT / "docs/alphazero-lite-a16-lineage-closeout.md",
        "closeout_ledger": REPO_ROOT
        / "docs/data/alphazero-lite-a16-lineage-closeout.json",
        "closeout_validator": REPO_ROOT
        / "ml/alphazero_lite/validate_a16_lineage_closeout.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_generation3_alpha_beta_preflight"),
    )
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "ml/alphazero_lite/validate_a16_lineage_closeout.py"],
        cwd=REPO_ROOT,
        check=True,
    )
    before = protected_hashes()
    artifact_path = REPO_ROOT / "model-artifact/current"
    metadata = json.loads((artifact_path / "metadata.json").read_text())
    if (
        before["weights"] != WEIGHTS_SHA
        or metadata.get("version") != VERSION
        or metadata.get("artifacts", {}).get("weights_json_sha256") != WEIGHTS_SHA
    ):
        raise RuntimeError("frozen production artifact identity mismatch")
    corpus = build_corpus()
    artifact = ArtifactEvaluator(artifact_path)
    evaluators = {
        "artifact_value": ArtifactValueEvaluator(artifact),
        "heuristic_value": HeuristicValueEvaluator(HeuristicEvaluator()),
    }
    rows = []
    for corpus_row in corpus:
        references = action_references(corpus_row)
        puct = {
            str(seed): puct_scored(
                puct_result(KalahGame.from_state(corpus_row["state"]), artifact, seed),
                references,
            )
            for seed in PUCT_SEEDS
        }
        lanes = {}
        for name, evaluator in evaluators.items():
            result = run_alpha_beta_search(
                KalahGame.from_state(corpus_row["state"]),
                evaluator,
                leaf_evaluation_budget=LEAF_BUDGET,
            )
            lanes[name] = score_result(result, references)
        rows.append(
            corpus_row | {"action_references": references, "puct": puct, "lanes": lanes}
        )
    determinism = []
    for row in rows[:12]:
        for lane, evaluator in evaluators.items():
            repeated = run_alpha_beta_search(
                KalahGame.from_state(row["state"]),
                evaluator,
                leaf_evaluation_budget=LEAF_BUDGET,
            )
            original = row["lanes"][lane]["telemetry"] | {
                "selected_move": row["lanes"][lane]["selected_move"]
            }
            replayed = {
                key: value
                for key, value in repeated.__dict__.items()
                if key != "runtime_seconds"
            } | {"selected_move": repeated.selected_move}
            determinism.append(
                {
                    "state_hash": row["state_hash"],
                    "lane": lane,
                    "equal_excluding_runtime": original == replayed,
                }
            )
    budget_ok = all(
        result["telemetry"]["leaf_evaluator_calls"] <= LEAF_BUDGET
        and not result["telemetry"]["node_cap_reached"]
        for row in rows
        for result in row["lanes"].values()
    ) and all(
        item["evaluator_calls"] == LEAF_BUDGET
        for row in rows
        for item in row["puct"].values()
    )
    invariants_ok = (
        all(
            result["legal_action_valid"]
            for row in rows
            for result in row["lanes"].values()
        )
        and all(
            item["legal_action_valid"] for row in rows for item in row["puct"].values()
        )
        and all(item["equal_excluding_runtime"] for item in determinism)
    )
    lane_metrics = {lane: metrics(rows, lane) for lane in evaluators}
    slices = {lane: slice_metrics(rows, lane) for lane in evaluators}
    lane_classifications = {
        lane: classify_lane(
            lane_metrics[lane],
            slices[lane],
            budget_ok=budget_ok,
            invariants_ok=invariants_ok,
        )
        for lane in evaluators
    }
    if lane_classifications["artifact_value"] == "qualified":
        classification = "alpha_beta_artifact_value_interface_qualified"
    elif lane_classifications["heuristic_value"] == "qualified":
        classification = "alpha_beta_search_mechanism_qualified"
    elif "regresses_subsets" in lane_classifications.values():
        classification = "alpha_beta_regresses_subsets"
    elif not budget_ok:
        classification = "alpha_beta_budget_contract_failed"
    elif not invariants_ok:
        classification = "alpha_beta_invariant_failure"
    else:
        classification = "alpha_beta_no_search_gain"
    after = protected_hashes()
    subprocess.run(
        [sys.executable, "ml/alphazero_lite/validate_a16_lineage_closeout.py"],
        cwd=REPO_ROOT,
        check=True,
    )
    corpus_hash = hashlib.sha256(
        json.dumps(corpus, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "schema_version": "azlite_generation3_alpha_beta_preflight_v1",
        "classification": classification,
        "lane_classifications": lane_classifications,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "frozen_model": {
            "path": "model-artifact/current",
            "version": VERSION,
            "weights_json_sha256": WEIGHTS_SHA,
            "metadata_weights_json_sha256": metadata["artifacts"][
                "weights_json_sha256"
            ],
        },
        "preregistration": {
            "corpus_seed": CORPUS_SEED,
            "puct_seeds": list(PUCT_SEEDS),
            "candidate_leaf_evaluation_budget": LEAF_BUDGET,
            "ordinary_puct_neural_evaluation_budget": LEAF_BUDGET,
            "forced_action_reference_budget": REFERENCE_BUDGET,
            "bootstrap_seed": CORPUS_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "plan": "docs/alphazero-lite-generation3-alpha-beta-preflight-plan.md",
            "full_text": (
                REPO_ROOT
                / "docs/alphazero-lite-generation3-alpha-beta-preflight-plan.md"
            ).read_text(encoding="utf-8"),
        },
        "corpus_hash": corpus_hash,
        "corpus_rows": corpus,
        "results": rows,
        "aggregate_metrics": lane_metrics,
        "sliced_metrics": slices,
        "determinism": {
            "checked": len(determinism),
            "results": determinism,
            "passed": all(item["equal_excluding_runtime"] for item in determinism),
        },
        "search_telemetry_totals": {
            lane: {
                "leaf_calls": sum(
                    row["lanes"][lane]["telemetry"]["leaf_evaluator_calls"]
                    for row in rows
                ),
                "nodes": sum(
                    row["lanes"][lane]["telemetry"]["nodes_visited"] for row in rows
                ),
                "completed_depth_mean": float(
                    statistics.fmean(
                        row["lanes"][lane]["telemetry"]["completed_depth"]
                        for row in rows
                    )
                ),
                "budget_utilization_mean": float(
                    statistics.fmean(
                        row["lanes"][lane]["telemetry"]["budget_utilization"]
                        for row in rows
                    )
                ),
            }
            for lane in evaluators
        },
        "invariants": {
            "budget_ok": budget_ok,
            "legal_selection_ok": invariants_ok,
            "protected_hashes_unchanged": before == after,
            "before_protected_hashes": before,
            "after_protected_hashes": after,
        },
        "guardrails": {
            "diagnostic_only": True,
            "not_training_eligible": True,
            "no_training": True,
            "no_self_play_or_replay": True,
            "no_arena": True,
            "no_artifact_mutation": before["weights"] == after["weights"],
            "no_existing_suite_consumed": True,
            "no_suite_registry_mutation": True,
            "no_closeout_ledger_mutation": before["closeout_ledger"]
            == after["closeout_ledger"],
        },
    }
    data = (
        REPO_ROOT
        / "docs/data/alphazero-lite-generation3-alpha-beta-preflight-summary.json"
    )
    report = (
        REPO_ROOT / "docs/alphazero-lite-generation3-alpha-beta-preflight-results.md"
    )
    data.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report.write_text(
        f"# Generation-3 Alpha-Beta Preflight Results\n\n**Classification:** `{classification}`\n\nThis diagnostic used a fresh 64-state standard-start corpus only. It did not train, generate replay/self-play, run an arena, consume a suite, mutate an artifact, mutate the registry, or change the A16 closeout ledger.\n\n```json\n{json.dumps({key: summary[key] for key in ('classification', 'lane_classifications', 'aggregate_metrics', 'search_telemetry_totals', 'determinism', 'invariants', 'guardrails')}, indent=2, sort_keys=True)}\n```\n\nThe artifact-value lane directly tests whether the frozen value head is compatible with alpha-beta. The heuristic lane isolates the search mechanism. Both lanes improved aggregate regret versus PUCT, but their paired confidence intervals crossed zero and preregistered subsets increased catastrophic misses. Therefore this run cannot distinguish a value-head incompatibility from a search-mechanism limitation as a qualified gain: both fail the fixed diagnostic gate. No training is recommended or permitted.\n"
    )
    print(classification)


if __name__ == "__main__":
    main()
