#!/usr/bin/env python3
"""Benchmark contract runner skeleton for AlphaZero-lite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.report_validation import ArenaReportValidationError, validate_arena_report
from ml.alphazero_lite.self_play import add_search_option_args, build_eval_search_options, search_options_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sanity", "promotion"], required=True)
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument("--challenger-path", default=None)
    parser.add_argument("--current-path", default="storage/ai/alphazero_lite/current")
    parser.add_argument("--arena-report", default=None)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--mcts-report", default=None)
    parser.add_argument("--current-baseline-mcts-report", default=None)
    parser.add_argument("--min-mcts-score", type=float, default=0.45)
    parser.add_argument("--dry-run", action="store_true")
    add_search_option_args(parser)
    parser.set_defaults(root_policy_mode="deterministic", tactical_root_bias=0.1)
    return parser.parse_args()


def checks_for_mode(mode: str) -> list[dict[str, str]]:
    if mode == "sanity":
        return [
            {"id": "az_identity", "description": "AlphaZero identity check"},
            {"id": "mcts_identity", "description": "MCTS1200 identity check"},
            {"id": "mcts_monotonic", "description": "MCTS1800 vs MCTS1200"},
            {"id": "runtime_parity", "description": "Chooser vs preloaded parity"},
        ]

    return [
        {"id": "promotion_arena", "description": "Candidate versus current arena gate"},
    ]


def promotion_checks(args: argparse.Namespace) -> list[dict]:
    if args.dry_run:
        return checks_for_mode("promotion")

    if not args.arena_report:
        raise SystemExit("--arena-report is required for promotion mode when not using --dry-run")
    if not args.mcts_report:
        raise SystemExit("--mcts-report is required for promotion mode when not using --dry-run")

    arena_path = Path(args.arena_report)
    if not arena_path.exists():
        raise SystemExit(f"arena report not found: {arena_path}")

    arena = json.loads(arena_path.read_text(encoding="utf-8"))
    try:
        arena_result = validate_arena_report(report=arena, min_score=float(args.min_score))
    except ArenaReportValidationError as error:
        raise SystemExit(str(error)) from error

    games_played = int(arena_result["games_played"])
    wins = int(arena_result["wins"])
    losses = int(arena_result["losses"])
    draws = int(arena_result["draws"])
    score = float(arena_result["score"])
    declared = bool(arena.get("promotion_decision", {}).get("passed", False))
    check_passed = bool(arena_result["passed"])

    mcts_path = Path(args.mcts_report)
    if not mcts_path.exists():
        raise SystemExit(f"mcts report not found: {mcts_path}")

    mcts = json.loads(mcts_path.read_text(encoding="utf-8"))
    mcts_games = int(mcts.get("games", 0))
    mcts_wins = int(mcts.get("az_wins", 0))
    mcts_losses = int(mcts.get("mcts_wins", 0))
    mcts_draws = int(mcts.get("draws", 0))
    if mcts_games <= 0:
        raise SystemExit("mcts report games must be > 0")

    mcts_score = (mcts_wins + (0.5 * mcts_draws)) / mcts_games
    mcts_min_score = float(args.min_mcts_score)
    mcts_passed = mcts_score >= mcts_min_score
    mcts_check: dict[str, int | float | bool | str] = {
        "id": "mcts1200_gate",
        "description": "Candidate versus MCTS1200 minimum strength",
        "passed": bool(mcts_passed),
        "score": round(mcts_score, 4),
        "games_played": mcts_games,
        "wins": mcts_wins,
        "losses": mcts_losses,
        "draws": mcts_draws,
        "min_score": mcts_min_score,
    }

    if args.current_baseline_mcts_report:
        baseline_mcts_path = Path(args.current_baseline_mcts_report)
        if not baseline_mcts_path.exists():
            raise SystemExit(f"current baseline mcts report not found: {baseline_mcts_path}")

        baseline_mcts = json.loads(baseline_mcts_path.read_text(encoding="utf-8"))
        baseline_games = int(baseline_mcts.get("games", 0))
        baseline_wins = int(baseline_mcts.get("az_wins", 0))
        baseline_draws = int(baseline_mcts.get("draws", 0))
        if baseline_games <= 0:
            raise SystemExit("current baseline mcts report games must be > 0")

        baseline_score = (baseline_wins + (0.5 * baseline_draws)) / baseline_games
        mcts_check.update(
            {
                "description": "Candidate versus current baseline MCTS1200 strength",
                "passed": bool(mcts_score >= baseline_score),
                "baseline_score": round(baseline_score, 4),
                "comparison": "current_baseline",
            }
        )
        del mcts_check["min_score"]

    return [
        {
            "id": "promotion_arena",
            "description": "Candidate versus current arena gate",
            "passed": check_passed,
            "score": round(score, 4),
            "games_played": games_played,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "declared_passed": declared,
            "min_score": float(args.min_score),
        },
        mcts_check,
    ]


def build_report(args: argparse.Namespace) -> dict:
    checks = checks_for_mode(args.mode)
    if args.mode == "promotion":
        checks = promotion_checks(args)

    return {
        "schema": "azlite_benchmark_v1",
        "mode": args.mode,
        "games": args.games,
        "seed": args.seed,
        "challenger_path": args.challenger_path,
        "current_path": args.current_path,
        "arena_report": args.arena_report,
        "mcts_report": args.mcts_report,
        "current_baseline_mcts_report": args.current_baseline_mcts_report,
        "min_score": float(args.min_score),
        "min_mcts_score": float(args.min_mcts_score),
        "dry_run": bool(args.dry_run),
        "search_options": build_eval_search_options(**search_options_from_args(args)),
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = build_report(args)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"wrote benchmark report to {out_path}")


if __name__ == "__main__":
    main()
