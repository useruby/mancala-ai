#!/usr/bin/env python3
# ruff: noqa: E402
"""Audit standard opening-suite evaluation against canonical direct PUCT games.

This diagnostic never trains, generates replay, promotes an artifact, or changes
search tuning.  It intentionally uses a new work directory and refuses a full
claim unless every paired raw record is identical.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from ml.alphazero_lite.arena import ArtifactEvaluator, run_arena_worker
from ml.alphazero_lite.cpuct_schedule import resolve_budget_cpuct
from ml.alphazero_lite.evaluation_seed_contract import derive_search_seed, stable_hash
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import PUCT
from ml.alphazero_lite.run_canonical_policy_interpolation_reconciliation import (
    canonical_game,
)

SCHEMA = "azlite_evaluation_path_equivalence_audit_v1"
BUDGETS = ("384:256", "768:256", "768:768", "1200:1200", "1200:256", "256:768")
CURRENT_SHA = "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
EXPECTED = {
    "D384": "6fe1444d6c82cb4a443d62111c3adb9ccd028d73071de52e1be2d186b6dec779",
    "D1200": "03dc5145a1d2fd471b6eb5a8dfacb4c57b5612594d1a5fd6d07acfc3dd8f02eb",
}
_WORKER_EVALUATORS: dict[str, ArtifactEvaluator] = {}
_WORKER_PATHS: dict[str, str] = {}


def init_worker(paths: dict[str, str]) -> None:
    global _WORKER_EVALUATORS, _WORKER_PATHS
    _WORKER_PATHS = paths
    _WORKER_EVALUATORS = {
        name: ArtifactEvaluator(Path(path)) for name, path in paths.items()
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def opening_state(prefix: list[int]) -> dict[str, Any]:
    game = KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    for move in prefix:
        if not game.move(move):
            raise RuntimeError("illegal opening prefix")
    return game.to_state()


def run_game(
    *,
    path: str,
    opening: dict[str, Any],
    opening_index: int,
    challenger: ArtifactEvaluator,
    current: ArtifactEvaluator,
    challenger_player: int,
    budget: str,
    suite_sha: str,
    base_seed: int,
    initial_ply: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    challenger_sims, current_sims = (int(value) for value in budget.split(":"))
    cpuct = resolve_budget_cpuct(
        schedule={"768:768": 0.90},
        challenger_simulations=challenger_sims,
        current_simulations=current_sims,
        default_c_puct=1.25,
    )
    game, opening_hash, moves, ledger = (
        KalahGame.from_state(opening),
        stable_hash(opening),
        [],
        [],
    )
    for offset in range(200):
        if game.over() or not game.possible_moves():
            break
        role = "challenger" if game.current_player == challenger_player else "current"
        evaluator, simulations = (
            (challenger, challenger_sims)
            if role == "challenger"
            else (current, current_sims)
        )
        state = game.to_state()
        seed_context = {
            "base_seed": base_seed,
            "suite_sha256": suite_sha,
            "opening_index": opening_index,
            "opening_state_hash": opening_hash,
            "challenger_player": challenger_player,
            "game_within_opening": 0,
            "ply": initial_ply + offset,
            "canonical_current_state_hash": stable_hash(state),
            "acting_role": role,
        }
        seed, context_hash = derive_search_seed(**seed_context)
        search = PUCT(
            evaluator=evaluator,
            simulations=simulations,
            c_puct=cpuct,
            rng=random.Random(seed),
            root_policy_mode="deterministic",
            tactical_root_bias=0.0,
            normalize_values=False,
        )
        visits, root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
        move = int(search.select_root_move(root, game.possible_moves()))
        resulting_state = game.clone()
        if not resulting_state.move(resulting_state.pit_index(move)):
            raise RuntimeError("PUCT selected an illegal move")
        ledger.append(
            {
                "evaluator_path": path,
                "opening_index": opening_index,
                "opening_hash": opening_hash,
                "challenger_player": challenger_player,
                "budget": budget,
                "ply": initial_ply + offset,
                "canonical_state_hash": stable_hash(state),
                "acting_role": role,
                "simulations": simulations,
                "c_puct": cpuct,
                "tactical_root_bias": 0.0,
                "seed_context": seed_context,
                "derived_seed": seed,
                "selected_move": move,
                "visits_hash": stable_hash([int(v) for v in visits]),
                "root_value": float(root.q_value),
                "resulting_state_hash": stable_hash(resulting_state.to_state()),
            }
        )
        moves.append(move)
        game = resulting_state
    margin = int(
        game.captured_seeds[challenger_player]
        - game.captured_seeds[1 - challenger_player]
    )
    record = {
        "evaluator_path": path,
        "opening_index": opening_index,
        "challenger_player": challenger_player,
        "budget": budget,
        "score": 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0,
        "margin": margin,
        "trajectory": moves,
        "trajectory_hash": stable_hash(moves),
        "final_score": list(game.captured_seeds),
    }
    for row in ledger:
        row["final_score"], row["final_margin"] = record["final_score"], margin
    return record, ledger


def audit_task(
    task: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    candidate_name = task["candidate"]
    challenger_sims, current_sims = (int(value) for value in task["budget"].split(":"))
    prefix_file = (
        Path(task["task_dir"])
        / f"{candidate_name}_{task['budget'].replace(':', '_')}_{task['opening_index']}_{task['seat']}.jsonl"
    )
    prefix_file.parent.mkdir(parents=True, exist_ok=True)
    prefix_line = json.dumps({"prefix_moves": task["prefix_moves"]}) + "\n"
    prefix_file.write_text(prefix_line * (task["opening_index"] + 1), encoding="utf-8")
    standard_run = run_arena_worker(
        worker_id=0,
        start_index=0,
        games=1,
        challenger_path=_WORKER_PATHS[candidate_name],
        current_path=_WORKER_PATHS["current"],
        challenger_simulations=challenger_sims,
        current_simulations=current_sims,
        seed=42,
        c_puct=0.90 if task["budget"] == "768:768" else 1.25,
        max_moves=200,
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        opening_prefixes_jsonl=str(prefix_file),
        games_per_opening=1,
        challenger_starts=task["seat"],
        seed_contract="azlite_eval_seed_v2",
        suite_sha256_override=task["suite_sha"],
        opening_index_override=task["opening_index"],
        opening_prefix_override=task["prefix_moves"],
        opening_state_override=task["opening"],
    )
    canonical = canonical_game(
        opening=task["opening"],
        opening_index=task["opening_index"],
        challenger=_WORKER_EVALUATORS[candidate_name],
        current=_WORKER_EVALUATORS["current"],
        challenger_player=task["seat"],
        challenger_budget=challenger_sims,
        current_budget=current_sims,
        seed=42,
        suite_hash=task["suite_sha"],
        label=candidate_name,
    )
    standard_game = standard_run["game_entries"][0]
    standard_ledger = [
        {**seed, **outcome}
        for seed, outcome in zip(
            standard_run["seed_identity_ledger"],
            standard_run["search_outcome_ledger"],
            strict=True,
        )
    ]
    canonical_ledger = canonical["trace"]
    fields = (
        "canonical_current_state_hash",
        "acting_role",
        "derived_search_seed",
        "selected_move",
        "visit_hash",
    )
    cause = None
    for left, right in zip(standard_ledger, canonical_ledger, strict=True):
        difference = next(
            (field for field in fields if left.get(field) != right.get(field)), None
        )
        if difference is not None:
            cause = f"{difference}_difference"
            break
    standard = {
        "evaluator_path": "standard",
        "candidate": candidate_name,
        "opening_index": task["opening_index"],
        "challenger_player": task["seat"],
        "budget": task["budget"],
        "score": 1.0
        if standard_game["winner"] == "challenger"
        else 0.5
        if standard_game["winner"] == "draw"
        else 0.0,
        "margin": standard_game["margin"],
        "trajectory_hash": stable_hash(standard_game["trajectory"]),
        "final_score": None,
    }
    canonical_record = {
        "evaluator_path": "canonical",
        "candidate": candidate_name,
        "budget": task["budget"],
        **canonical,
    }
    return [standard, canonical_record], standard_ledger + canonical_ledger, cause


def first_divergence(
    standard: list[dict[str, Any]], canonical: list[dict[str, Any]]
) -> str | None:
    fields = (
        "canonical_state_hash",
        "challenger_player",
        "budget",
        "c_puct",
        "tactical_root_bias",
        "seed_context",
        "derived_seed",
        "selected_move",
        "visits_hash",
        "root_value",
        "resulting_state_hash",
    )
    for left, right in zip(standard, canonical):
        for field in fields:
            if left[field] != right[field]:
                if field in {"seed_context", "derived_seed"}:
                    return "seed_context_difference"
                if field == "c_puct":
                    return "c_puct_schedule_difference"
                if field in {"selected_move", "visits_hash", "root_value"}:
                    return "search_implementation_difference"
                return "opening-state construction difference"
    return None if len(standard) == len(canonical) else "unknown"


def bootstrap(values: list[float], seed: int = 42) -> dict[str, Any]:
    data = np.asarray(values, dtype=float)
    draws = data[
        np.random.default_rng(seed).integers(0, len(data), size=(10_000, len(data)))
    ].mean(axis=1)
    return {
        "mean": float(data.mean()),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "samples": 10_000,
        "n": len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="/tmp/azlite_evaluation_path_equivalence_audit"
    )
    parser.add_argument(
        "--suite", default="/tmp/azlite_opening_suite/medium_eval.jsonl"
    )
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--d384",
        default="/tmp/azlite_stronger_policy_teacher/d384_policy_teacher_e1_run_a/artifact",
    )
    parser.add_argument(
        "--d1200",
        default="/tmp/azlite_stronger_policy_teacher/d1200_policy_teacher_e1_run_a/artifact",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="run all 128 openings; default is the required 32-opening instrumentation slice",
    )
    parser.add_argument(
        "--openings",
        type=int,
        default=None,
        help="test-only limit; prevents a full equivalence classification",
    )
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    workdir, suite = Path(args.workdir), Path(args.suite)
    if workdir.exists():
        raise RuntimeError(
            "workdir must be completely new; refusing cached historical results"
        )
    workdir.mkdir(parents=True)
    paths = {
        "current": Path(args.current),
        "D384": Path(args.d384),
        "D1200": Path(args.d1200),
    }
    hashes = {name: sha256(path / "weights.json") for name, path in paths.items()}
    if hashes["current"] != CURRENT_SHA:
        raise RuntimeError("current artifact SHA256 mismatch")
    # Artifact weight hashes and checkpoint hashes are distinct; record both rather than conflate them.
    for name in ("D384", "D1200"):
        checkpoint = paths[name].parent / "checkpoint.npz"
        if not checkpoint.is_file() or sha256(checkpoint) != EXPECTED[name]:
            raise RuntimeError(f"{name} checkpoint SHA256 mismatch")
    rows, suite_sha = read_jsonl(suite), sha256(suite)
    if len(rows) != 128:
        raise RuntimeError("medium suite must contain exactly 128 openings")
    rows = rows if args.full else rows[:32]
    if args.openings is not None:
        rows = rows[: args.openings]
    all_records, all_ledgers, causes = [], [], Counter()
    tasks = [
        {
            "candidate": candidate,
            "budget": budget,
            "opening": opening_state([int(move) for move in row["prefix_moves"]]),
            "prefix_moves": [int(move) for move in row["prefix_moves"]],
            "opening_index": index,
            "seat": seat,
            "suite_sha": suite_sha,
            "task_dir": str(workdir / "tasks"),
        }
        for candidate in ("current", "D384", "D1200")
        for budget in BUDGETS
        for index, row in enumerate(rows)
        for seat in (0, 1)
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=init_worker,
        initargs=({name: str(path) for name, path in paths.items()},),
    ) as executor:
        for records, ledger, cause in executor.map(audit_task, tasks, chunksize=1):
            causes[cause or "identical"] += 1
            all_records.extend(records)
            all_ledgers.extend(ledger)
    (workdir / "game_ledger.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_ledgers),
        encoding="utf-8",
    )
    paired = [
        r
        for r in all_records
        if r["evaluator_path"] == "canonical" and r["candidate"] != "current"
    ]
    current_controls = {
        (r["budget"], r["opening_index"], r["challenger_player"]): r["score"]
        for r in all_records
        if r["evaluator_path"] == "canonical" and r["candidate"] == "current"
    }
    by_candidate = {}
    for candidate in ("D384", "D1200"):
        by_candidate[candidate] = {}
        for budget in BUDGETS:
            candidate_rows = [
                r
                for r in paired
                if r["candidate"] == candidate and r["budget"] == budget
            ]
            controls = [
                {
                    "opening_index": opening,
                    "challenger_player": seat,
                    "score": score,
                }
                for (control_budget, opening, seat), score in current_controls.items()
                if control_budget == budget
            ]
            paired_metric = paired_opening_candidate_effect(candidate_rows, controls)
            effects = list(paired_metric.pop("per_opening_effect").values())
            by_candidate[candidate][budget] = {
                **paired_metric,
                "mean": paired_metric["paired_candidate_effect"],
                "lower_95": paired_metric["opening_bootstrap_ci"]["lower_95"],
                "upper_95": paired_metric["opening_bootstrap_ci"]["upper_95"],
                "positive_openings": int(sum(x > 0 for x in effects)),
                "zero_openings": int(sum(x == 0 for x in effects)),
                "negative_openings": int(sum(x < 0 for x in effects)),
            }
    complete_scope = args.full and args.openings is None
    summary = {
        "schema": SCHEMA,
        "classification": "evaluation_paths_equivalent"
        if complete_scope and causes.keys() == {"identical"}
        else "evaluation_path_audit_incomplete",
        "scope": "full_128" if complete_scope else f"instrumented_{len(rows)}",
        "inputs": {
            "suite_sha256": suite_sha,
            "artifact_weights_sha256": hashes,
            "checkpoint_sha256": EXPECTED,
        },
        "runtime": {
            "tactical_root_bias": 0.0,
            "default_c_puct": 1.25,
            "768:768_c_puct": 0.90,
            "root_policy": "deterministic",
            "seed_contract": "azlite_eval_seed_v2",
            "base_seed": 42,
        },
        "first_divergence_counts": dict(causes),
        "canonical_paired_opening_statistic": by_candidate,
        "metric_semantics": {
            "candidate_challenger_score": "mean(score(candidate as challenger))",
            "current_current_challenger_score": "mean(score(current as challenger))",
            "candidate_minus_current_control_delta": "candidate_challenger_score - current_current_challenger_score",
            "seat_asymmetry_ds": "P0 challenger score - P1 challenger score; diagnostic only, not candidate strength",
            "paired_opening_candidate_effect": "mean over openings of matched candidate-minus-current-control seat effects",
        },
        "historical_conclusion": "PR #182 DS is a seat difference, while PR #184 reports candidate-minus-current paired score. They are different estimands; source-suite SHA and opening-state ply are now shared.",
    }
    output = (
        REPO_ROOT / "docs/data/alphazero-lite-evaluation-path-equivalence-summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (
        REPO_ROOT / "docs/alphazero-lite-evaluation-path-equivalence-results.md"
    ).write_text(
        "# Evaluation Path Equivalence Audit\n\n```json\n"
        + json.dumps(summary, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"classification": summary["classification"], "games": len(all_records)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
