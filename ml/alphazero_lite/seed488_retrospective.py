"""Read-only descriptive accounting for the completed seed488 evaluation."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


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
    assert len(openings) == 512 and len(ledger) == len(games) == 1024
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
    for pair in by_opening.values():
        if len(pair) != 2 or {int(r["challenger_player"]) for r in pair} != {0, 1}:
            raise ValueError("opening_pair_coverage")
        paired_first.append(
            {
                "challenger_agree": pair[0].get("first_move_challenger")
                == pair[1].get("first_move_challenger"),
                "current_agree": pair[0].get("first_move_current")
                == pair[1].get("first_move_current"),
            }
        )
    agreement = {
        arm: sum(p[f"{arm}_agree"] for p in paired_first)
        for arm in ("challenger", "current")
    }
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
        "paired_first_action": {
            "compared_openings": len(paired_first),
            "challenger_agreements": agreement["challenger"],
            "challenger_disagreements": 512 - agreement["challenger"],
            "current_agreements": agreement["current"],
            "current_disagreements": 512 - agreement["current"],
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
