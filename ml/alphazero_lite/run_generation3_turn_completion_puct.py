#!/usr/bin/env python3
# ruff: noqa: E402
"""Reproducible diagnostic-only Generation-3 turn-completion PUCT preflight."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, sha256_file
from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.consumed_suite_registry import entries, load
from ml.alphazero_lite.gumbel_root_search import run_puct_root_search
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state
from ml.alphazero_lite.run_generation3_alpha_beta_preflight import corpus_features
from ml.alphazero_lite.self_play import standard_start_state
from ml.alphazero_lite.turn_completion_puct import run_turn_completion_puct

CORPUS_SEED = BOOTSTRAP_SEED = 275001
SEEDS = (275101, 275102, 275103)
BUDGET, REFERENCE_BUDGET, BOOTSTRAP_SAMPLES = 384, 2400, 10_000
WEIGHTS_SHA = "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
VERSION = "azlite-balanced-w8s4-policy-head-e1"
PLAN = ROOT / "docs/alphazero-lite-generation3-turn-completion-puct-plan.md"
REGISTRY = ROOT / "ml/alphazero_lite/consumed_suite_registry.py"
SLICES = (
    "phase",
    "player",
    "legal_action_count",
    "capture_available",
    "extra_turn_available",
)


def canonical_hash(row: dict) -> str:
    return row["state_hash"]


def build_corpus() -> list[dict]:
    """Sample only legal standard-start prefixes, then freeze canonical ordering."""
    rng = random.Random(CORPUS_SEED)
    candidates: dict[str, dict] = {}
    for _ in range(40_000):
        game = KalahGame.from_state(standard_start_state())
        ply = rng.randint(8, 48)
        for _ in range(ply):
            legal = game.possible_moves()
            if not legal or game.over():
                break
            game.move(game.pit_index(rng.choice(legal)))
        if game.over() or len(game.possible_moves()) < 2:
            continue
        row = corpus_features(game, ply)
        if row["phase"] == "opening":
            continue
        candidates.setdefault(canonical_hash(row), row)
    extra = [row for row in candidates.values() if row["extra_turn_available"]]
    control = [row for row in candidates.values() if not row["extra_turn_available"]]
    rng.shuffle(extra)
    rng.shuffle(control)
    selected = extra[:64] + control[:32]
    if len(selected) != 96 or len({row["state_hash"] for row in selected}) != 96:
        raise RuntimeError("insufficient fresh canonical corpus states")
    if {row["player"] for row in selected} != {0, 1}:
        raise RuntimeError("corpus lacks a player perspective")
    return sorted(selected, key=canonical_hash)


def state_hashes_from_json(path: Path) -> tuple[str, set[str], int]:
    payload = json.loads(path.read_text())
    rows = payload.get("corpus", payload.get("corpus_rows", []))
    return (
        sha256_file(path),
        {row["state_hash"] for row in rows if "state_hash" in row},
        len(rows),
    )


def overlap_audit(corpus: list[dict], workdir: Path) -> list[dict]:
    target = {row["state_hash"] for row in corpus}
    sources: list[tuple[str, str, set[str], int]] = []
    for label, path in (
        (
            "pr270_alpha_beta",
            ROOT
            / "docs/data/alphazero-lite-generation3-alpha-beta-preflight-summary.json",
        ),
        (
            "pr274_implicit_minimax",
            ROOT
            / "docs/data/alphazero-lite-generation3-implicit-minimax-puct-full.json",
        ),
    ):
        identity, hashes, count = state_hashes_from_json(path)
        sources.append((label, identity, hashes, count))
    registry = load(workdir)
    for label, rows in entries(registry).items():
        hashes = {row.get("state_hash") for row in rows if row.get("state_hash")}
        sources.append(
            (f"consumed_suite_{label}", registry[label].sha256, hashes, len(rows))
        )
    for path in sorted((ROOT / "docs/data").glob("*generation3*.json")):
        if path.name not in {
            "alphazero-lite-generation3-alpha-beta-preflight-summary.json",
            "alphazero-lite-generation3-implicit-minimax-puct-full.json",
        }:
            identity, hashes, count = state_hashes_from_json(path)
            sources.append(
                (f"registered_generation3_{path.name}", identity, hashes, count)
            )
    audit = [
        {
            "source": label,
            "identity_hash": identity,
            "row_count": count,
            "overlap_count": len(target & hashes),
        }
        for label, identity, hashes, count in sources
    ]
    if any(row["overlap_count"] for row in audit):
        raise RuntimeError(
            "fresh corpus overlaps a prior or registered diagnostic corpus"
        )
    return audit


def references(row: dict) -> list[dict]:
    root = KalahGame.from_state(row["state"])
    root_player = root.current_player
    output = []
    for action in row["legal_actions"]:
        child = root.clone()
        child.move(child.pit_index(action))
        seed = int(
            hashlib.sha256(
                f"{CORPUS_SEED}:{row['state_hash']}:{action}".encode()
            ).hexdigest()[:16],
            16,
        )
        if child.over():
            value = (
                0.0
                if child.winner is None
                else (1.0 if child.winner == root_player else -1.0)
            )
        else:
            continued = ClassicMCTS(
                child,
                simulations=REFERENCE_BUDGET,
                seed=seed,
                endgame_tablebase=None,
                exact_solve_enabled=False,
            ).search_root()
            value = (
                2.0 * continued.wins / continued.visits - 1.0
                if continued.visits
                else 0.0
            )
            if child.current_player != root_player:
                value = -value
        output.append({"action": action, "value_root_player": value, "seed": seed})
    return output


def run_lane(
    game: KalahGame, evaluator: ArtifactEvaluator, seed: int, candidate: bool
) -> dict:
    started = time.perf_counter()
    result = (run_turn_completion_puct if candidate else run_puct_root_search)(
        game, evaluator, seed=seed, budget=BUDGET
    )
    telemetry = result.__dict__.copy()
    telemetry["runtime_seconds"] = time.perf_counter() - started
    # Uses production's unchanged selection calculation for live root telemetry.
    telemetry["policy_priors"] = None
    telemetry["exploration_values"] = None
    telemetry["selected_move"] = telemetry.pop("selected_move")
    return telemetry


def score(result: dict, refs: list[dict]) -> dict:
    values = {item["action"]: item["value_root_player"] for item in refs}
    ranked = sorted(values, key=lambda action: (-values[action], action))
    selected = result["selected_move"]
    return result | {
        "selected_reference_value": values[selected],
        "regret": values[ranked[0]] - values[selected],
        "best_reference_action_agreement": selected == ranked[0],
        "top_two_agreement": selected in ranked[:2],
        "catastrophic_miss": values[ranked[0]] - values[selected] >= 0.25,
        "legal_selection_status": selected in values,
    }


def bootstrap(rows: list[dict]) -> dict:
    values = [
        [
            row["candidate"][str(seed)]["regret"] - row["baseline"][str(seed)]["regret"]
            for seed in SEEDS
        ]
        for row in rows
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
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
        "seed": BOOTSTRAP_SEED,
    }


def aggregate(rows: list[dict]) -> dict:
    def summary(items: list[dict]) -> dict:
        return {
            "mean_regret": float(statistics.fmean(x["regret"] for x in items)),
            "best_reference_action_agreement": float(
                statistics.fmean(x["best_reference_action_agreement"] for x in items)
            ),
            "top_two_agreement": float(
                statistics.fmean(x["top_two_agreement"] for x in items)
            ),
            "catastrophic_miss_rate": float(
                statistics.fmean(x["catastrophic_miss"] for x in items)
            ),
            "p95_runtime_seconds": float(
                np.quantile([x["runtime_seconds"] for x in items], 0.95)
            ),
        }

    return {
        "baseline": summary([x for row in rows for x in row["baseline"].values()]),
        "candidate": summary([x for row in rows for x in row["candidate"].values()]),
        "paired_hierarchical_bootstrap": bootstrap(rows),
    }


def main() -> None:
    started_wall, started = time.time(), time.perf_counter()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise RuntimeError(
            "full experiment requires a clean worktree at execution start"
        )
    subprocess.run(
        [sys.executable, "ml/alphazero_lite/validate_a16_lineage_closeout.py"],
        cwd=ROOT,
        check=True,
    )
    before_weight, before_registry = (
        sha256_file(ROOT / "model-artifact/current/weights.json"),
        sha256_file(REGISTRY),
    )
    if before_weight != WEIGHTS_SHA:
        raise RuntimeError("frozen artifact hash mismatch")
    metadata = json.loads((ROOT / "model-artifact/current/metadata.json").read_text())
    if metadata.get("version") != VERSION:
        raise RuntimeError("frozen artifact version mismatch")
    workdir = Path("/tmp/turn_completion_puct_audit")
    workdir.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus()
    audit = overlap_audit(corpus, workdir)
    artifact = ArtifactEvaluator(ROOT / "model-artifact/current")
    rows = []
    for row in corpus:
        refs = references(row)
        baseline = {
            str(seed): score(
                run_lane(KalahGame.from_state(row["state"]), artifact, seed, False),
                refs,
            )
            for seed in SEEDS
        }
        candidate = {
            str(seed): score(
                run_lane(KalahGame.from_state(row["state"]), artifact, seed, True), refs
            )
            for seed in SEEDS
        }
        rows.append(
            row
            | {
                "action_references": refs,
                "consequences": {
                    str(action): move_consequence_for_state(row["state"], action)
                    for action in row["legal_actions"]
                },
                "baseline": baseline,
                "candidate": candidate,
            }
        )
    metrics = {
        "full": aggregate(rows),
        "root_extra_turn": aggregate(
            [row for row in rows if row["extra_turn_available"]]
        ),
        "no_root_extra_turn": aggregate(
            [row for row in rows if not row["extra_turn_available"]]
        ),
    }
    budget_ok = all(
        item["evaluator_calls"] == BUDGET
        for row in rows
        for lane in ("baseline", "candidate")
        for item in row[lane].values()
    )
    invariants_ok = all(
        item["legal_selection_status"]
        for row in rows
        for lane in ("baseline", "candidate")
        for item in row[lane].values()
    ) and all(
        item.get("defensive_cap_events", 0) == item.get("repeated_state_events", 0) == 0
        and item.get("incomplete_due_to_budget", 0) <= 1
        for row in rows
        for item in row["candidate"].values()
    )
    primary, full = metrics["root_extra_turn"], metrics["full"]
    candidate_qualified = (
        primary["baseline"]["mean_regret"] > 0
        and primary["candidate"]["mean_regret"]
        <= 0.75 * primary["baseline"]["mean_regret"]
        and primary["paired_hierarchical_bootstrap"]["upper_95"] < 0
        and primary["candidate"]["best_reference_action_agreement"]
        >= primary["baseline"]["best_reference_action_agreement"]
        and primary["candidate"]["catastrophic_miss_rate"]
        <= primary["baseline"]["catastrophic_miss_rate"]
        and full["candidate"]["mean_regret"] <= 0.9 * full["baseline"]["mean_regret"]
        and full["paired_hierarchical_bootstrap"]["upper_95"] < 0
        and full["candidate"]["best_reference_action_agreement"]
        >= full["baseline"]["best_reference_action_agreement"]
        and full["candidate"]["catastrophic_miss_rate"]
        <= full["baseline"]["catastrophic_miss_rate"]
        and full["candidate"]["p95_runtime_seconds"]
        <= 1.25 * full["baseline"]["p95_runtime_seconds"]
    )
    classification = (
        "turn_completion_invariant_failure"
        if not invariants_ok
        else "turn_completion_budget_contract_failed"
        if not budget_ok
        else "turn_completion_puct_qualified"
        if candidate_qualified
        else "turn_completion_no_search_gain"
    )
    after_weight, after_registry = (
        sha256_file(ROOT / "model-artifact/current/weights.json"),
        sha256_file(REGISTRY),
    )
    subprocess.run(
        [sys.executable, "ml/alphazero_lite/validate_a16_lineage_closeout.py"],
        cwd=ROOT,
        check=True,
    )
    result = {
        "schema_version": "turn_completion_puct_v1",
        "classification": classification,
        "experiment_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "preregistration": {
            "plan": str(PLAN.relative_to(ROOT)),
            "plan_sha256": sha256_file(PLAN),
            "corpus_seed": CORPUS_SEED,
            "puct_seeds": list(SEEDS),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "neural_budget": BUDGET,
        },
        "artifact_sha256_before": before_weight,
        "artifact_sha256_after": after_weight,
        "corpus_sha256": hashlib.sha256(
            json.dumps(corpus, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "clean_worktree_at_start": True,
        "consumed_suite_registry_sha256_before": before_registry,
        "consumed_suite_registry_sha256_after": after_registry,
        "corpus_overlap_audit": audit,
        "corpus": corpus,
        "results": rows,
        "aggregate_metrics": metrics,
        "invariants": {
            "budget_ok": budget_ok,
            "invariants_ok": invariants_ok,
            "artifact_unchanged": before_weight == after_weight,
            "registry_unchanged": before_registry == after_registry,
        },
        "started_unix_seconds": started_wall,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }
    full_path = (
        ROOT / "docs/data/alphazero-lite-generation3-turn-completion-puct-full.json"
    )
    summary_path = (
        ROOT / "docs/data/alphazero-lite-generation3-turn-completion-puct-summary.json"
    )
    report_path = (
        ROOT / "docs/alphazero-lite-generation3-turn-completion-puct-results.md"
    )
    full_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary_path.write_text(
        json.dumps(result | {"results": "see full result"}, indent=2, sort_keys=True)
        + "\n"
    )
    report_path.write_text(
        f"# Generation-3 Turn-Completion PUCT Results\n\n**Classification:** `{classification}`\n\nThe result records every qualification gate, provenance hash, overlap audit, and whether incomplete budget-boundary extensions occurred.\n\n```json\n{json.dumps({key: result[key] for key in ('classification', 'aggregate_metrics', 'invariants', 'elapsed_seconds')}, indent=2)}\n```\n"
    )
    print(classification)


if __name__ == "__main__":
    main()
