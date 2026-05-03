#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arena-inputs", required=True)
    parser.add_argument("--candidate-mcts-inputs", required=True)
    parser.add_argument("--current-mcts-inputs", required=True)
    parser.add_argument("--min-arena-score", type=float, default=0.55)
    parser.add_argument("--out-arena", required=True)
    parser.add_argument("--out-candidate-mcts", required=True)
    parser.add_argument("--out-current-mcts", required=True)
    return parser.parse_args(argv)


def split_paths(text: str) -> list[Path]:
    return [Path(part.strip()) for part in text.split(",") if part.strip()]


def read_reports(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def arena_latency_notes(report: dict[str, Any]) -> tuple[float, float]:
    notes = report.get("notes")
    if isinstance(notes, dict):
        mean = notes.get("move_time_mean_ms")
        p95 = notes.get("move_time_p95_ms")
        if mean is not None and p95 is not None:
            return float(mean), float(p95)

    return float(report["move_time_mean_ms"]), float(report["move_time_p95_ms"])


def aggregate_arena(reports, min_arena_score=0.55):
    if not reports:
        raise ValueError("at least one arena report is required")

    games = 0
    wins = 0
    losses = 0
    draws = 0
    weighted_latency_total = 0.0
    p95_values = []

    for report in reports:
        report_games = report["games_played"]
        report_wins = report["wins"]
        report_losses = report["losses"]
        report_draws = report["draws"]
        mean_latency, p95_latency = arena_latency_notes(report)
        if report_wins + report_losses + report_draws != report_games:
            raise ValueError("arena report counts do not match games_played")

        games += report_games
        wins += report_wins
        losses += report_losses
        draws += report_draws
        weighted_latency_total += mean_latency * report_games
        p95_values.append(p95_latency)

    if wins + losses + draws != games:
        raise ValueError("aggregated arena counts do not match games_played")
    if games == 0:
        raise ValueError("aggregated arena games_played cannot be zero")

    score = (wins + (0.5 * draws)) / games

    return {
        "schema": "arena_v1",
        "games_played": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "promotion_decision": {
            "passed": score >= min_arena_score,
        },
        "notes": {
            "move_time_mean_ms": weighted_latency_total / games,
            "move_time_p95_ms": max(p95_values),
        },
    }


def aggregate_mcts(reports):
    if not reports:
        raise ValueError("at least one mcts report is required")

    games = 0
    az_wins = 0
    mcts_wins = 0
    draws = 0

    for report in reports:
        report_games = report["games"]
        report_az_wins = report["az_wins"]
        report_mcts_wins = report["mcts_wins"]
        report_draws = report["draws"]
        if report_az_wins + report_mcts_wins + report_draws != report_games:
            raise ValueError("mcts report counts do not match games")

        games += report_games
        az_wins += report_az_wins
        mcts_wins += report_mcts_wins
        draws += report_draws

    if az_wins + mcts_wins + draws != games:
        raise ValueError("aggregated mcts counts do not match games")

    return {
        "schema": "azlite_vs_mcts_v1",
        "games": games,
        "az_wins": az_wins,
        "mcts_wins": mcts_wins,
        "draws": draws,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        write_json(
            Path(args.out_arena),
            aggregate_arena(
                read_reports(split_paths(args.arena_inputs)),
                min_arena_score=args.min_arena_score,
            ),
        )
        write_json(
            Path(args.out_candidate_mcts),
            aggregate_mcts(read_reports(split_paths(args.candidate_mcts_inputs))),
        )
        write_json(
            Path(args.out_current_mcts),
            aggregate_mcts(read_reports(split_paths(args.current_mcts_inputs))),
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
