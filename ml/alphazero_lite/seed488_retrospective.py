"""Read-only descriptive accounting for the completed seed488 evaluation."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .kalah_rules import KalahGame


def _root_action(opening: dict[str, Any], row: dict[str, Any]) -> int:
    """Return the relative pit chosen at the shared opening root."""
    moves = row.get("trajectory")
    if not isinstance(moves, str) or not moves:
        raise ValueError("first_trajectory_action_missing")
    absolute = [int(value) for value in moves.split(",")]
    if not absolute:
        raise ValueError("first_trajectory_action_missing")
    game = KalahGame.from_state(opening["state"])
    root_player = game.current_player
    first = absolute[0]
    if not 0 <= first < 12 or first // 6 != root_player:
        raise ValueError("root_action_player_mismatch")
    seat = int(row["challenger_player"])
    logged_key = (
        "first_move_challenger" if seat == root_player else "first_move_current"
    )
    logged = row.get(logged_key)
    if logged is None or int(logged) != first % 6:
        raise ValueError("first_trajectory_action_missing_or_mismatch")
    return first % 6


def _validate_ledger(ledger: list[dict[str, Any]], games: list[dict[str, Any]]) -> None:
    ledger_keys = [(int(row["opening_index"]), int(row["F_player"])) for row in ledger]
    game_keys = [
        (int(row["parent_opening_index"]), int(row["challenger_player"]))
        for row in games
    ]
    if len(set(ledger_keys)) != len(ledger_keys) or len(set(game_keys)) != len(
        game_keys
    ):
        raise ValueError("duplicate_ledger_or_game_identity")
    if Counter(ledger_keys) != Counter(game_keys):
        raise ValueError("ledger_game_identity_mismatch")
    if len(ledger) != len(games):
        raise ValueError("ledger_game_identity_mismatch")
    game_by_key = dict(zip(game_keys, games, strict=True))
    winner_names = {"challenger": "F", "current": "initializer"}
    for key, ledger_row in zip(ledger_keys, ledger, strict=True):
        ledger_winner = ledger_row.get("winner")
        game_winner = game_by_key[key].get("winner")
        expected_winner = (
            winner_names.get(game_winner, game_winner)
            if isinstance(game_winner, str)
            else game_winner
        )
        if ledger_winner is not None and ledger_winner != expected_winner:
            raise ValueError("ledger_game_result_mismatch")


def analyze(root: Path) -> dict[str, Any]:
    data = root / "docs/data/seed488-frozen-strength-evaluation"
    openings = [
        json.loads(line) for line in (data / "openings.jsonl").read_text().splitlines()
    ]
    ledger = [
        json.loads(line) for line in (data / "outcomes.jsonl").read_text().splitlines()
    ]
    games: list[dict[str, Any]] = []
    for opening in range(512):
        for line in (
            (data / "resume-corrected" / f"games-{opening:04d}.jsonl")
            .read_text()
            .splitlines()
        ):
            row = json.loads(line)
            row["parent_opening_index"] = opening
            games.append(row)
    if len(openings) != 512 or len(ledger) != 1024 or len(games) != 1024:
        raise ValueError("opening_or_game_count")
    # Outcome ledger identity is (parent opening, F seat); log row order is
    # deliberately irrelevant.
    _validate_ledger(ledger, games)
    points = {
        "challenger": 1.0,
        "F": 1.0,
        "draw": 0.5,
        "current": 0.0,
        "initializer": 0.0,
    }
    seat: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    by_opening: dict[int, list[dict[str, Any]]] = {i: [] for i in range(512)}
    for row in games:
        seat[int(row["challenger_player"])].append(row)
        by_opening[int(row["parent_opening_index"])].append(row)
    seat_scores = {
        str(k): {
            "score": sum(points[r["winner"]] for r in rows) / len(rows),
            "wins": sum(r["winner"] == "challenger" for r in rows),
            "draws": sum(r["winner"] == "draw" for r in rows),
            "losses": sum(r["winner"] == "current" for r in rows),
        }
        for k, rows in seat.items()
    }
    pair_scores = [
        sum(points[r["winner"]] for r in by_opening[i]) / 2 for i in range(512)
    ]
    outcomes = Counter(r["winner"] for r in games)
    lengths = [int(r["game_length"]) for r in games]
    handoffs = [int(r["exact_root_handoff_count"]) for r in games]
    first_handoff = [
        int(r["first_exact_root_handoff_ply"])
        for r in games
        if r["first_exact_root_handoff_ply"] is not None
    ]
    normal_ms = [float(r["normal_puct_time_ms"]) for r in games]
    exact_ms = [float(r["exact_root_solve_time_ms"]) for r in games]
    paired_first = []
    for opening_index, pair in by_opening.items():
        if len(pair) != 2 or {int(r["challenger_player"]) for r in pair} != {0, 1}:
            raise ValueError("opening_pair_coverage")
        opening = openings[opening_index]
        actions: list[tuple[int, int]] = []
        for row in pair:
            prefix = opening.get("prefix_moves")
            if not isinstance(prefix, list) or len(prefix) != int(
                row.get("opening_applied_prefix_length", -1)
            ):
                raise ValueError("opening_prefix_accounting")
            actions.append((int(row["challenger_player"]), _root_action(opening, row)))
        paired_first.append(
            {
                "agree": actions[0][1] == actions[1][1],
                "score": sum(points[r["winner"]] for r in pair) / 2,
            }
        )
    agreement = sum(p["agree"] for p in paired_first)
    return {
        "game_count": len(games),
        "opening_count": len(openings),
        "seat_scores": seat_scores,
        "outcomes": dict(outcomes),
        "score": sum(points[r["winner"]] for r in games) / len(games),
        "pair_score_distribution": {
            str(x): sum(s == x for s in pair_scores)
            for x in (0.0, 0.25, 0.5, 0.75, 1.0)
        },
        "game_length": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "mean": statistics.mean(lengths),
            "max": max(lengths),
            "at_200": sum(x == 200 for x in lengths),
        },
        "handoff": {
            "games_with_any": sum(x > 0 for x in handoffs),
            "total": sum(handoffs),
            "per_game_mean": statistics.mean(handoffs),
            "first_ply_games": len(first_handoff),
            "first_ply_median": statistics.median(first_handoff),
            "first_ply_mean": statistics.mean(first_handoff),
        },
        "timing_ms": {
            "normal_puct_sum": sum(normal_ms),
            "normal_puct_per_game_mean": statistics.mean(normal_ms),
            "exact_root_sum": sum(exact_ms),
            "exact_root_per_game_mean": statistics.mean(exact_ms),
            "exact_root_per_handoff_mean": sum(exact_ms) / sum(handoffs),
        },
        "root_action_comparison": {
            "compared_openings": len(paired_first),
            "agreements": agreement,
            "disagreements": 512 - agreement,
            "paired_score_distribution_by_agreement": {
                str(agree): {
                    str(score): sum(
                        p["agree"] == agree and p["score"] == score
                        for p in paired_first
                    )
                    for score in (0.0, 0.25, 0.5, 0.75, 1.0)
                }
                for agree in (True, False)
            },
        },
        "ledger_game_count": len(ledger),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
