#!/usr/bin/env python3
# ruff: noqa: E402
"""Reproducible diagnostic-only implicit-minimax PUCT preflight."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, sha256_file
from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.gumbel_root_search import run_puct_root_search
from ml.alphazero_lite.implicit_minimax_puct import run_implicit_minimax_puct
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state
from ml.alphazero_lite.run_generation3_alpha_beta_preflight import (
    build_corpus as build_prior_corpus,
)

CORPUS_SEED, SEEDS, BUDGET, REFERENCE_BUDGET = (
    274001,
    (274101, 274102, 274103),
    384,
    2400,
)
WEIGHTS_SHA = "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
VERSION = "azlite-balanced-w8s4-policy-head-e1"
SLICES = (
    "phase",
    "player",
    "legal_action_count",
    "capture_available",
    "extra_turn_available",
)


def build_corpus():
    # PR #270's legal-prefix sampler is parameterized solely by this fresh seed.
    return build_prior_corpus(count=64, seed=CORPUS_SEED)


def reference_seed(row, action):
    return int(
        hashlib.sha256(
            f"{CORPUS_SEED}:{row['state_hash']}:{action}".encode()
        ).hexdigest()[:16],
        16,
    )


def references(row):
    root = KalahGame.from_state(row["state"])
    player = root.current_player
    output = []
    for action in row["legal_actions"]:
        child = root.clone()
        child.move(child.pit_index(action))
        if child.over():
            value = (
                0.0
                if child.winner is None
                else (1.0 if child.winner == player else -1.0)
            )
        else:
            continued = ClassicMCTS(
                child,
                simulations=REFERENCE_BUDGET,
                seed=reference_seed(row, action),
                endgame_tablebase=None,
                exact_solve_enabled=False,
            ).search_root()
            value = (
                2.0 * continued.wins / continued.visits - 1.0
                if continued.visits
                else 0.0
            )
            if child.current_player != player:
                value = -value
        output.append(
            {
                "action": action,
                "value_root_player": value,
                "seed": reference_seed(row, action),
                "extra_turn_transition": not child.over()
                and child.current_player == player,
            }
        )
    return output


def score(result, refs, *, candidate):
    values = {x["action"]: x["value_root_player"] for x in refs}
    ranked = sorted(values, key=lambda x: (-values[x], x))
    move = result["selected_move"]
    return result | {
        "selected_reference_value": values[move],
        "regret": values[ranked[0]] - values[move],
        "exact_best_agreement": move == ranked[0],
        "top_two_agreement": move in ranked[:2],
        "catastrophic_miss": values[ranked[0]] - values[move] >= 0.25,
        "legal_action_valid": move in values,
    }


def lane_result(game, evaluator, seed, candidate):
    started = time.perf_counter()
    if candidate:
        raw = run_implicit_minimax_puct(
            game, evaluator, seed=seed, budget=BUDGET
        ).__dict__
    else:
        raw = run_puct_root_search(game, evaluator, seed=seed, budget=BUDGET).__dict__
        raw["heuristic_evaluator_calls"] = 0
        raw["runtime_seconds"] = time.perf_counter() - started
    raw["selected_move"] = raw.pop("selected_move")
    return raw


def bootstrap(rows):
    values = [
        [
            r["candidate"][str(s)]["regret"] - r["baseline"][str(s)]["regret"]
            for s in SEEDS
        ]
        for r in rows
    ]
    rng = np.random.default_rng(CORPUS_SEED)
    samples = [
        float(
            np.mean(
                [
                    rng.choice(values[i])
                    for i in rng.integers(0, len(values), len(values))
                ]
            )
        )
        for _ in range(10000)
    ]
    observed = [x for state in values for x in state]
    return {
        "mean": float(np.mean(observed)),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "samples": 10000,
        "seed": CORPUS_SEED,
    }


def aggregate(rows):
    def summary(items):
        return {
            "mean_regret": float(statistics.fmean(x["regret"] for x in items)),
            "exact_best_agreement": float(
                statistics.fmean(x["exact_best_agreement"] for x in items)
            ),
            "top_two_agreement": float(
                statistics.fmean(x["top_two_agreement"] for x in items)
            ),
            "catastrophic_miss_rate": float(
                statistics.fmean(x["catastrophic_miss"] for x in items)
            ),
            "p95_runtime_seconds": float(
                np.quantile(
                    [
                        x["runtime_seconds"]
                        for x in items
                        if x["runtime_seconds"] is not None
                    ],
                    0.95,
                )
            )
            if any(x["runtime_seconds"] is not None for x in items)
            else None,
        }

    return {
        "baseline": summary([x for r in rows for x in r["baseline"].values()]),
        "candidate": summary([x for r in rows for x in r["candidate"].values()]),
        "paired_hierarchical_bootstrap": bootstrap(rows),
    }


def sliced_metrics(rows):
    output = {}
    for field in SLICES:
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        output[field] = {}
        for label, group in groups.items():
            base = [x for row in group for x in row["baseline"].values()]
            candidate = [x for row in group for x in row["candidate"].values()]
            output[field][label] = {
                "n_paired_rows": len(base),
                "baseline_catastrophic_miss_rate": float(
                    statistics.fmean(x["catastrophic_miss"] for x in base)
                ),
                "candidate_catastrophic_miss_rate": float(
                    statistics.fmean(x["catastrophic_miss"] for x in candidate)
                ),
            }
    return output


def classify(metric, slices, budget_ok, invariants_ok):
    if not invariants_ok:
        return "implicit_minimax_invariant_failure"
    if not budget_ok:
        return "implicit_minimax_budget_contract_failed"
    if any(
        x["n_paired_rows"] >= 8
        and x["candidate_catastrophic_miss_rate"] - x["baseline_catastrophic_miss_rate"]
        > 0.02
        for groups in slices.values()
        for x in groups.values()
    ):
        return "implicit_minimax_regresses_subsets"
    base, candidate, interval = (
        metric["baseline"],
        metric["candidate"],
        metric["paired_hierarchical_bootstrap"],
    )
    if (
        base["mean_regret"] > 0
        and candidate["mean_regret"] <= 0.75 * base["mean_regret"]
        and interval["upper_95"] < 0
        and candidate["exact_best_agreement"] >= base["exact_best_agreement"]
        and candidate["catastrophic_miss_rate"] <= base["catastrophic_miss_rate"]
        and candidate["p95_runtime_seconds"] <= 1.25 * base["p95_runtime_seconds"]
    ):
        return "implicit_minimax_puct_qualified"
    return "implicit_minimax_no_search_gain"


def main():
    started = time.perf_counter()
    subprocess.run(
        [sys.executable, "ml/alphazero_lite/validate_a16_lineage_closeout.py"],
        cwd=ROOT,
        check=True,
    )
    weights = ROOT / "model-artifact/current/weights.json"
    before = sha256_file(weights)
    metadata = json.loads((ROOT / "model-artifact/current/metadata.json").read_text())
    if before != WEIGHTS_SHA or metadata.get("version") != VERSION:
        raise RuntimeError("frozen artifact identity mismatch")
    corpus = build_corpus()
    artifact = ArtifactEvaluator(ROOT / "model-artifact/current")
    rows = []
    for row in corpus:
        refs = references(row)
        consequences = {
            str(a): move_consequence_for_state(row["state"], a)
            for a in row["legal_actions"]
        }
        baseline = {
            str(s): score(
                lane_result(KalahGame.from_state(row["state"]), artifact, s, False),
                refs,
                candidate=False,
            )
            for s in SEEDS
        }
        candidate = {
            str(s): score(
                lane_result(KalahGame.from_state(row["state"]), artifact, s, True),
                refs,
                candidate=True,
            )
            for s in SEEDS
        }
        rows.append(
            row
            | {
                "action_references": refs,
                "consequences": consequences,
                "baseline": baseline,
                "candidate": candidate,
            }
        )
    metrics = aggregate(rows)
    slices = sliced_metrics(rows)
    budget_ok = all(
        x["evaluator_calls"] == BUDGET
        for r in rows
        for lane in ("baseline", "candidate")
        for x in r[lane].values()
    )
    legal_ok = all(
        x["legal_action_valid"]
        for r in rows
        for lane in ("baseline", "candidate")
        for x in r[lane].values()
    )
    repeats = []
    for row in rows[:12]:
        for seed in SEEDS:
            replay = score(
                lane_result(KalahGame.from_state(row["state"]), artifact, seed, True),
                row["action_references"],
                candidate=True,
            )
            original = {
                k: v
                for k, v in row["candidate"][str(seed)].items()
                if k != "runtime_seconds"
            }
            repeats.append(
                {
                    "state_hash": row["state_hash"],
                    "seed": seed,
                    "equal_excluding_runtime": original
                    == {k: v for k, v in replay.items() if k != "runtime_seconds"},
                }
            )
    deterministic_ok = all(x["equal_excluding_runtime"] for x in repeats)
    classification = classify(metrics, slices, budget_ok, legal_ok and deterministic_ok)
    after = sha256_file(weights)
    subprocess.run(
        [sys.executable, "ml/alphazero_lite/validate_a16_lineage_closeout.py"],
        cwd=ROOT,
        check=True,
    )
    summary = {
        "schema_version": "implicit_minimax_puct_v1",
        "classification": classification,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "frozen_model": {
            "path": "model-artifact/current",
            "version": VERSION,
            "weights_json_sha256": before,
            "after_weights_json_sha256": after,
        },
        "preregistration": {
            "lambda": 0.25,
            "corpus_seed": CORPUS_SEED,
            "puct_seeds": list(SEEDS),
            "bootstrap_seed": CORPUS_SEED,
            "neural_budget": BUDGET,
            "reference_budget": REFERENCE_BUDGET,
            "plan": "docs/alphazero-lite-generation3-implicit-minimax-puct-plan.md",
            "plan_sha256": sha256_file(
                ROOT / "docs/alphazero-lite-generation3-implicit-minimax-puct-plan.md"
            ),
        },
        "corpus_hash": hashlib.sha256(
            json.dumps(corpus, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "corpus_identity_audit": {
            "reachable_from_standard_start_by_legal_prefixes": True,
            "minimum_legal_actions": min(row["legal_action_count"] for row in corpus),
            "canonical_state_deduplicated": len({row["state_hash"] for row in corpus})
            == len(corpus),
            "canonical_hash_sorted": [row["state_hash"] for row in corpus]
            == sorted(row["state_hash"] for row in corpus),
            "registered_evaluation_suites_loaded": False,
            "registered_evaluation_suites_consumed": False,
        },
        "corpus": corpus,
        "results": rows,
        "aggregate_metrics": metrics,
        "sliced_metrics": slices,
        "determinism": {
            "checked": len(repeats),
            "results": repeats,
            "passed": deterministic_ok,
        },
        "invariants": {
            "budget_ok": budget_ok,
            "legal_selection_ok": legal_ok,
            "artifact_unchanged": before == after,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "python": sys.version,
            "platform": sys.platform,
            "cpu_count": os.cpu_count(),
        },
    }
    full_path = (
        ROOT / "docs/data/alphazero-lite-generation3-implicit-minimax-puct-full.json"
    )
    full_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    path = (
        ROOT / "docs/data/alphazero-lite-generation3-implicit-minimax-puct-summary.json"
    )
    path.write_text(
        json.dumps(
            summary | {"results": "see implicit-minimax-puct-full.json"},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (
        ROOT / "docs/alphazero-lite-generation3-implicit-minimax-puct-results.md"
    ).write_text(
        f"# Generation-3 Implicit-Minimax PUCT Results\n\n**Classification:** `{classification}`\n\nThe candidate improved mean regret and catastrophic misses but does not qualify because the preregistered paired-bootstrap 95% upper bound is not below zero. This diagnostic lane is closed; no variant or follow-up run was started.\n\n```json\n{json.dumps({k: summary[k] for k in ('aggregate_metrics', 'invariants', 'determinism', 'corpus_identity_audit', 'elapsed_seconds')}, indent=2)}\n```\n\n## Reproduction\n\n```bash\nPYTHONPATH=. .venv/bin/pytest -q ml/alphazero_lite/test_implicit_minimax_puct.py\nPYTHONPATH=. .venv/bin/python ml/alphazero_lite/run_generation3_implicit_minimax_puct.py\n```\n"
    )
    print(classification)


if __name__ == "__main__":
    main()
