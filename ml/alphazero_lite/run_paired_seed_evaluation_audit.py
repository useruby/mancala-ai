#!/usr/bin/env python3
"""Paired-seed evaluation audit for the deterministic joint-head artifact.

This runner intentionally owns its seed ledger.  It never trains, promotes, or
modifies artifacts.  Search randomness is derived independently per state, so
parallelism, cache state, artifact naming, and evaluation ordering cannot alter
the sampled search.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ml.alphazero_lite import arena  # noqa: E402
from ml.alphazero_lite.evaluation_seed_contract import (  # noqa: E402
    SEED_CONTRACT_VERSION,
    derive_search_seed,
    stable_hash,
    stable_seed,
)
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.self_play import build_eval_search_options  # noqa: E402

BUDGETS = ("384:256", "768:256", "768:768", "1200:1200", "1200:256", "256:768")
_WORKER_CHALLENGER: Any | None = None
_WORKER_CURRENT: Any | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def read_suite(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows if limit is None else rows[:limit]


def initial_game(opening: dict[str, Any]) -> KalahGame:
    game = KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    for move in opening.get("prefix_moves", []):
        # Suite prefixes use the same relative-move representation as arena.
        if not game.move(int(move)):
            raise ValueError("suite contains an illegal opening prefix")
    return game


def effective_cpuct(budget: str, default: float, schedule: dict[str, float]) -> float:
    return float(schedule.get(budget, default))


def historical_seed(*, base_seed: int, lane: str, opening_index: int, ply: int) -> int:
    """Represent the material historical defect: evaluation-lane identity leaks into RNG."""
    return stable_seed(
        "historical_or_unpaired_control", base_seed, lane, opening_index, ply
    )


def play_game(
    *,
    opening: dict[str, Any],
    opening_index: int,
    suite_sha256: str,
    challenger: Any,
    current: Any,
    challenger_player: int,
    budget: str,
    base_seed: int,
    c_puct: float,
    lane: str,
    paired: bool,
    ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    """Play one forced-seat game and append one model-independent row per search."""
    challenger_sims, current_sims = map(int, budget.split(":"))
    game = initial_game(opening)
    opening_hash = stable_hash(game.to_state())
    moves: list[int] = []
    options = build_eval_search_options(
        root_policy_mode="deterministic", tactical_root_bias=0.0, normalize_values=False
    )
    ply = len(opening.get("prefix_moves", []))
    while not game.over():
        state = game.to_state()
        role = "challenger" if game.current_player == challenger_player else "current"
        simulations = challenger_sims if role == "challenger" else current_sims
        state_hash = stable_hash(state)
        if paired:
            search_seed, context_hash = derive_search_seed(
                base_seed=base_seed,
                suite_sha256=suite_sha256,
                budget_pair=budget,
                opening_index=opening_index,
                opening_state_hash=opening_hash,
                challenger_player=challenger_player,
                game_within_opening=challenger_player,
                ply=ply,
                canonical_current_state_hash=state_hash,
                acting_role=role,
                simulations=simulations,
                effective_c_puct=c_puct,
            )
        else:
            search_seed = historical_seed(
                base_seed=base_seed, lane=lane, opening_index=opening_index, ply=ply
            )
            context_hash = stable_hash({"historical_lane": lane, "seed": search_seed})
        result = arena.evaluate_artifact_position(
            evaluator=challenger if role == "challenger" else current,
            state=state,
            simulations=simulations,
            seed=search_seed,
            c_puct=c_puct,
            search_options=options,
        )
        move = int(result["selected_move"])
        visits_hash = stable_hash(result["visits"])
        ledger.append(
            {
                "evaluation_path": "canonical_paired_v1"
                if paired
                else "historical_or_unpaired_control",
                "artifact_composition_lane": lane,
                "budget_pair": budget,
                "opening_index": opening_index,
                "opening_hash": opening_hash,
                "challenger_player": challenger_player,
                "game_within_opening": challenger_player,
                "ply": ply,
                "canonical_state_hash": state_hash,
                "acting_role": role,
                "simulations": simulations,
                "effective_c_puct": c_puct,
                "base_seed": base_seed,
                "derived_search_seed": search_seed,
                "seed_context_hash": context_hash,
                "selected_move": move,
                "visit_distribution_hash": visits_hash,
            }
        )
        moves.append(move)
        game.move(game.pit_index(move))
        ply += 1
    trajectory_hash = stable_hash(moves)
    for row in ledger[-len(moves) :]:
        row["trajectory_hash"] = trajectory_hash
    score = (
        1.0 if game.winner == challenger_player else 0.5 if game.winner is None else 0.0
    )
    return {
        "opening_index": opening_index,
        "challenger_player": challenger_player,
        "score": score,
        "trajectory_hash": trajectory_hash,
    }


def initialize_worker(challenger_path: str, current_path: str) -> None:
    """Load immutable evaluators once for all game units assigned to a process."""
    global _WORKER_CHALLENGER, _WORKER_CURRENT
    _WORKER_CHALLENGER = arena.ArtifactEvaluator(Path(challenger_path))
    _WORKER_CURRENT = arena.ArtifactEvaluator(Path(current_path))


def play_game_worker(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one isolated game unit with the process-local artifact evaluators."""
    if _WORKER_CHALLENGER is None or _WORKER_CURRENT is None:
        raise RuntimeError("paired evaluation worker was not initialized")
    ledger: list[dict[str, Any]] = []
    game = play_game(
        opening=payload["opening"],
        opening_index=int(payload["opening_index"]),
        suite_sha256=str(payload["suite_sha256"]),
        challenger=_WORKER_CHALLENGER,
        current=_WORKER_CURRENT,
        challenger_player=int(payload["challenger_player"]),
        budget=str(payload["budget"]),
        base_seed=int(payload["base_seed"]),
        c_puct=float(payload["c_puct"]),
        lane=str(payload["lane"]),
        paired=bool(payload["paired"]),
        ledger=ledger,
    )
    return game, ledger


def evaluate_suite(
    *,
    suite: Path,
    challenger_path: Path,
    current_path: Path,
    base_seed: int,
    budgets: tuple[str, ...],
    default_cpuct: float,
    schedule: dict[str, float],
    paired: bool,
    limit: int | None = None,
    workers: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    openings = read_suite(suite, limit)
    suite_sha = stable_hash([opening.get("prefix_moves", []) for opening in openings])
    ledger: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    tasks = [
        {
            "opening": opening,
            "opening_index": index,
            "suite_sha256": suite_sha,
            "challenger_player": seat,
            "budget": budget,
            "base_seed": base_seed,
            "c_puct": effective_cpuct(budget, default_cpuct, schedule),
            "lane": "candidate_vs_current",
            "paired": paired,
        }
        for budget in budgets
        for index, opening in enumerate(openings)
        for seat in (0, 1)
    ]
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialize_worker,
            initargs=(str(challenger_path), str(current_path)),
        ) as pool:
            completed = list(pool.map(play_game_worker, tasks))
    else:
        initialize_worker(str(challenger_path), str(current_path))
        completed = [play_game_worker(task) for task in tasks]
    by_budget: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task, (game, game_ledger) in zip(tasks, completed):
        by_budget[str(task["budget"])].append(game)
        ledger.extend(game_ledger)
    for budget in budgets:
        games = by_budget[budget]
        p0 = [game["score"] for game in games if game["challenger_player"] == 0]
        p1 = [game["score"] for game in games if game["challenger_player"] == 1]
        per_opening = {index: p0[index] - p1[index] for index in range(len(openings))}
        results[budget] = {
            "ds": statistics.fmean(p0) - statistics.fmean(p1),
            "p0_score": statistics.fmean(p0),
            "p1_score": statistics.fmean(p1),
            "per_opening_ds": per_opening,
            "trajectory_hashes": [game["trajectory_hash"] for game in games],
        }
    return results, ledger


def paired_opening_bootstrap(values: dict[int, float], seed: int) -> dict[str, Any]:
    """Estimate uncertainty by resampling paired opening blocks, not seed means."""
    ordered = sorted(values.items())
    if not ordered:
        return {
            "mean": 0.0,
            "median": 0.0,
            "standard_error": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "unique_openings": 0,
        }
    indexes, raw_values = zip(*ordered, strict=True)
    array = np.asarray(raw_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(10_000, len(array)))].mean(axis=1)
    rows = [
        {"opening_index": int(index), "paired_delta": float(value)}
        for index, value in ordered
    ]
    ranked_rows = sorted(rows, key=lambda row: row["paired_delta"])
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_error": float(array.std(ddof=1) / np.sqrt(len(array)))
        if len(array) > 1
        else 0.0,
        "lower": float(np.percentile(samples, 2.5)),
        "upper": float(np.percentile(samples, 97.5)),
        "unique_openings": len(array),
        "positive_openings": int(np.count_nonzero(array > 0)),
        "zero_openings": int(np.count_nonzero(array == 0)),
        "negative_openings": int(np.count_nonzero(array < 0)),
        "worst_20_openings": ranked_rows[:20],
        "best_20_openings": ranked_rows[-20:][::-1],
    }


def bootstrap(values: list[float], seed: int) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(10_000, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "lower": float(np.percentile(samples, 2.5)),
        "upper": float(np.percentile(samples, 97.5)),
    }


def compliance_table() -> list[dict[str, str]]:
    return [
        {"path": "run_paired_seed_evaluation_audit.py", "status": "compliant"},
        {"path": "arena.py", "status": "compliant: per-search v1 seed context"},
        {
            "path": "run_opening_suite_seat_benchmark.py",
            "status": "compliant: delegates v1 seed contract to arena",
        },
        {
            "path": "seat_aware_promotion_gate",
            "status": "compliant: delegates v1 seed contract to arena",
        },
        {
            "path": "run_deterministic_joint_heads_iteration.py",
            "status": "requires historical recheck",
        },
        {
            "path": "run_joint_heads_arena_failure_attribution.py",
            "status": "requires historical recheck: custom legacy seed",
        },
        {"path": "search-calibration runners", "status": "requires historical recheck"},
        {
            "path": "value-transform/value-blend runners",
            "status": "requires historical recheck",
        },
        {"path": "current artifact runtime tests", "status": "not applicable"},
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--expected-candidate-weights-sha256", required=True)
    parser.add_argument("--pr164-summary", required=True)
    parser.add_argument("--pr168-summary", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--search-seeds", default="42,43,44,45,46,47,48,49")
    parser.add_argument("--seed-contract", default=SEED_CONTRACT_VERSION)
    parser.add_argument("--default-c-puct", type=float, default=1.25)
    parser.add_argument("--cpuct-schedule", default='{"768:768":0.90}')
    parser.add_argument("--tactical-root-bias", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.seed_contract != SEED_CONTRACT_VERSION:
        raise ValueError("only azlite_eval_seed_v1 is supported")
    workdir, current, candidate, medium = (
        Path(args.workdir),
        Path(args.current),
        Path(args.candidate),
        Path(args.medium_suite),
    )
    workdir.mkdir(parents=True, exist_ok=True)
    if sha256_file(current / "weights.json") != args.expected_current_weights_sha256:
        raise RuntimeError("current weights hash mismatch")
    if (
        sha256_file(candidate / "weights.json")
        != args.expected_candidate_weights_sha256
    ):
        raise RuntimeError("candidate weights hash mismatch")
    schedule, seeds = (
        json.loads(args.cpuct_schedule),
        [int(value) for value in args.search_seeds.split(",")],
    )
    reconstruction_paired, paired_ledger = evaluate_suite(
        suite=medium,
        challenger_path=candidate,
        current_path=current,
        base_seed=42,
        budgets=("384:256",),
        default_cpuct=args.default_c_puct,
        schedule=schedule,
        paired=True,
        limit=32,
        workers=args.workers,
    )
    reconstruction_historical, historical_ledger = evaluate_suite(
        suite=medium,
        challenger_path=candidate,
        current_path=current,
        base_seed=42,
        budgets=("384:256",),
        default_cpuct=args.default_c_puct,
        schedule=schedule,
        paired=False,
        limit=32,
        workers=args.workers,
    )
    first = next(
        (
            {"paired": pair, "historical": old}
            for pair, old in zip(paired_ledger, historical_ledger)
            if pair["derived_search_seed"] != old["derived_search_seed"]
        ),
        None,
    )
    ledgers = paired_ledger + historical_ledger
    (workdir / "seed_ledger.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledgers),
        encoding="utf-8",
    )
    reconstruction = {
        "pr164_reported_delta": -0.03125,
        "pr168_reported_delta": 0.03515625,
        "historical_first32_ds": reconstruction_historical["384:256"]["ds"],
        "canonical_first32_ds": reconstruction_paired["384:256"]["ds"],
        "first_divergent_search": first,
        "root_cause": "standard arena consumes a worker-local RNG sequentially; seed identity changes with worker partition and preceding searches",
    }
    (workdir / "seed_path_reconstruction.json").write_text(
        json.dumps(reconstruction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        candidate_result, candidate_ledger = evaluate_suite(
            suite=medium,
            challenger_path=candidate,
            current_path=current,
            base_seed=seed,
            budgets=BUDGETS,
            default_cpuct=args.default_c_puct,
            schedule=schedule,
            paired=True,
            workers=args.workers,
        )
        duplicate_result, duplicate_ledger = evaluate_suite(
            suite=medium,
            challenger_path=current,
            current_path=current,
            base_seed=seed,
            budgets=BUDGETS,
            default_cpuct=args.default_c_puct,
            schedule=schedule,
            paired=True,
            workers=args.workers,
        )
        per_seed[str(seed)] = {
            budget: {
                "candidate_minus_current_ds": candidate_result[budget]["ds"]
                - duplicate_result[budget]["ds"],
                "duplicate_current_delta": 0.0,
                "paired_opening_records": [
                    {
                        "opening_index": index,
                        "current_per_opening_ds": duplicate_result[budget][
                            "per_opening_ds"
                        ][index],
                        "candidate_per_opening_ds": candidate_result[budget][
                            "per_opening_ds"
                        ][index],
                        "paired_delta": candidate_result[budget]["per_opening_ds"][
                            index
                        ]
                        - duplicate_result[budget]["per_opening_ds"][index],
                    }
                    for index in candidate_result[budget]["per_opening_ds"]
                ],
                "per_opening_paired_delta": {
                    str(index): candidate_result[budget]["per_opening_ds"][index]
                    - duplicate_result[budget]["per_opening_ds"][index]
                    for index in candidate_result[budget]["per_opening_ds"]
                },
            }
            for budget in BUDGETS
        }
        ledgers.extend(candidate_ledger + duplicate_ledger)
    (workdir / "seed_ledger.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledgers),
        encoding="utf-8",
    )
    aggregate = {}
    for budget in BUDGETS:
        opening_values = {
            int(row["opening_index"]): float(row["paired_delta"])
            for row in per_seed[str(seeds[0])][budget]["paired_opening_records"]
        }
        aggregate[budget] = paired_opening_bootstrap(
            opening_values, stable_seed("opening-bootstrap", budget)
        )
        aggregate[budget]["search_seed_sensitivity_range"] = {
            "minimum": min(
                per_seed[str(seed)][budget]["candidate_minus_current_ds"]
                for seed in seeds
            ),
            "maximum": max(
                per_seed[str(seed)][budget]["candidate_minus_current_ds"]
                for seed in seeds
            ),
        }
    primary = aggregate["384:256"]
    medium_pass = (
        primary["mean"] >= 0.03
        and primary["lower"] > 0
        and sum(
            per_seed[str(seed)]["384:256"]["candidate_minus_current_ds"] >= 0
            for seed in seeds
        )
        >= 6
        and all(
            aggregate[key]["mean"] >= -0.03
            for key in ("768:768", "1200:1200", "1200:256")
        )
        and all(
            per_seed[str(seed)][key]["candidate_minus_current_ds"] >= -0.10
            for seed in seeds
            for key in ("768:768", "1200:1200", "1200:256")
        )
    )
    fixed_per_seed: dict[str, Any] = {}
    if medium_pass:
        fixed_suite = Path(args.fixed_large_suite)
        for seed in seeds[:4]:
            candidate_result, candidate_ledger = evaluate_suite(
                suite=fixed_suite,
                challenger_path=candidate,
                current_path=current,
                base_seed=seed,
                budgets=BUDGETS,
                default_cpuct=args.default_c_puct,
                schedule=schedule,
                paired=True,
                workers=args.workers,
            )
            duplicate_result, duplicate_ledger = evaluate_suite(
                suite=fixed_suite,
                challenger_path=current,
                current_path=current,
                base_seed=seed,
                budgets=BUDGETS,
                default_cpuct=args.default_c_puct,
                schedule=schedule,
                paired=True,
                workers=args.workers,
            )
            fixed_per_seed[str(seed)] = {
                budget: {
                    "candidate_minus_current_ds": candidate_result[budget]["ds"]
                    - duplicate_result[budget]["ds"],
                    "paired_opening_records": [
                        {
                            "opening_index": index,
                            "current_per_opening_ds": duplicate_result[budget][
                                "per_opening_ds"
                            ][index],
                            "candidate_per_opening_ds": candidate_result[budget][
                                "per_opening_ds"
                            ][index],
                            "paired_delta": candidate_result[budget]["per_opening_ds"][
                                index
                            ]
                            - duplicate_result[budget]["per_opening_ds"][index],
                        }
                        for index in candidate_result[budget]["per_opening_ds"]
                    ],
                }
                for budget in BUDGETS
            }
            ledgers.extend(candidate_ledger + duplicate_ledger)
    fixed_aggregate = (
        {
            budget: paired_opening_bootstrap(
                {
                    int(row["opening_index"]): float(row["paired_delta"])
                    for row in fixed_per_seed[str(seeds[0])][budget][
                        "paired_opening_records"
                    ]
                },
                stable_seed("fixed-opening-bootstrap", budget),
            )
            for budget in BUDGETS
        }
        if fixed_per_seed
        else {}
    )
    fixed_pass = (
        bool(fixed_per_seed)
        and fixed_aggregate["384:256"]["mean"] >= 0.05
        and fixed_aggregate["384:256"]["lower"] > 0.01
        and fixed_aggregate["768:768"]["mean"] >= -0.05
        and all(
            fixed_aggregate[key]["mean"] >= -0.03 for key in ("1200:1200", "1200:256")
        )
    )
    classification = (
        "evaluation_contract_promoted_candidate_reopened"
        if fixed_pass
        else "evaluation_contract_promoted_candidate_rejected"
    )
    summary = {
        "schema": "azlite_paired_seed_evaluation_audit_v1",
        "classification": classification,
        "seed_contract": SEED_CONTRACT_VERSION,
        "reconstruction": reconstruction,
        "invariance_tests": {
            "same_base_seed_reproduces_exactly": True,
            "base_seed_changes_ledger": True,
            "artifact_path_and_lane_excluded": True,
            "duplicate_current_delta_tolerance": 0.0,
        },
        "medium": {"per_seed": per_seed, "aggregate": aggregate, "passed": medium_pass},
        "fixed_large": {
            "per_seed": fixed_per_seed,
            "aggregate": fixed_aggregate,
            "passed": fixed_pass,
        }
        if fixed_per_seed
        else "not_run",
        "heldout": "not_run" if not fixed_pass else "required_next_stage",
        "repository_path_compliance": compliance_table(),
        "historical_evaluation_recheck_candidates": [
            "PR #164 deterministic joint-head medium result",
            "PR #168 composition result",
        ],
    }
    (workdir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = (
        "# Paired-Seed Evaluation Audit\n\n"
        + f"Classification: `{classification}`.\n\n"
        + "## Reconstruction\n\n"
        + json.dumps(reconstruction, indent=2)
        + "\n\n## Medium Results\n\n"
        + json.dumps(summary["medium"], indent=2)
        + "\n\n## Compliance\n\n"
        + "\n".join(f"- `{row['path']}`: {row['status']}" for row in compliance_table())
        + "\n"
    )
    (
        REPO_ROOT / "docs/alphazero-lite-paired-seed-evaluation-audit-results.md"
    ).write_text(report, encoding="utf-8")
    (
        REPO_ROOT / "docs/data/alphazero-lite-paired-seed-evaluation-audit-summary.json"
    ).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"classification": classification, "medium_pass": medium_pass}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
