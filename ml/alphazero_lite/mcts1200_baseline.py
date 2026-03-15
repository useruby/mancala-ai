#!/usr/bin/env python3
"""Parallel MCTS1200 baseline driver with deterministic per-game MCTS seeding."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.classic_mcts import MCTS
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import PUCT, add_search_option_args, build_eval_search_options, search_options_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenger-path", required=True)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--az-base-simulations", type=int, default=640)
    parser.add_argument("--mcts-simulations", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", required=True)
    add_search_option_args(parser)
    return parser.parse_args()


def partition_counts(total: int, workers: int) -> list[int]:
    workers = max(1, workers)
    base = total // workers
    remainder = total % workers
    return [base + (1 if index < remainder else 0) for index in range(workers)]


def partition_starts(counts: list[int]) -> list[int]:
    starts: list[int] = []
    cursor = 0
    for count in counts:
        starts.append(cursor)
        cursor += count
    return starts


def simulation_budget_for(game: KalahGame, base_simulations: int) -> int:
    stones_in_pits = sum(game.pits)
    if stones_in_pits >= 36:
        multiplier = 1.25
    elif stones_in_pits <= 12:
        multiplier = 1.15
    else:
        multiplier = 1.0

    scaled = round(int(base_simulations) * multiplier)
    return max(96, min(1024, scaled))


def initial_game() -> KalahGame:
    return KalahGame.from_state(
        {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )


def run_stub_worker(*, start_index: int, games: int) -> dict:
    az_wins = 0
    mcts_wins = 0
    draws = 0

    for local_index in range(games):
        game_index = start_index + local_index
        outcome = game_index % 5
        if outcome in (0, 4):
            az_wins += 1
        elif outcome == 1:
            draws += 1
        else:
            mcts_wins += 1

    return {
        "start_index": start_index,
        "games": games,
        "az_wins": az_wins,
        "mcts_wins": mcts_wins,
        "draws": draws,
    }


def run_worker(
    *,
    challenger_path: str,
    games: int,
    start_index: int,
    seed: int,
    az_base_simulations: int,
    mcts_simulations: int,
    search_options: dict,
) -> dict:
    if os.environ.get("AZLITE_MCTS1200_BASELINE_STUB") == "1":
        return run_stub_worker(start_index=start_index, games=games)

    evaluator = ArtifactEvaluator(Path(challenger_path))
    az_wins = 0
    mcts_wins = 0
    draws = 0

    for local_index in range(games):
        game_index = start_index + local_index
        game = initial_game()
        az_player = 0 if (game_index % 2) == 0 else 1
        az_reusable_root = None

        while not game.over():
            search = None
            root = None
            acting_player = game.current_player
            if acting_player == az_player:
                simulations = simulation_budget_for(game, az_base_simulations)
                search = PUCT(
                    evaluator=evaluator,
                    simulations=simulations,
                    c_puct=1.25,
                    rng=random_for_game(seed, game_index, acting_player),
                    root=az_reusable_root,
                    fpu_mode=str(search_options["fpu_mode"]),
                    reuse_subtree=bool(search_options["reuse_subtree"]),
                    normalize_values=bool(search_options["normalize_values"]),
                    root_policy_mode=str(search_options["root_policy_mode"]),
                    tactical_root_bias=float(search_options["tactical_root_bias"]),
                )
                legal_moves = game.possible_moves()
                if not legal_moves:
                    break
                _visits, root = search.run(game)
                relative_move = search.select_root_move(root, legal_moves)
            else:
                relative_move = MCTS(game, simulations=mcts_simulations, seed=seed + game_index).choose_move()

            if relative_move is None:
                break
            if not game.move(game.pit_index(relative_move)):
                break

            if acting_player == az_player and bool(search_options["reuse_subtree"]) and root is not None and game.current_player == az_player:
                az_reusable_root = root.child_for_action(relative_move)
            else:
                az_reusable_root = None

        az_score = game.captured_seeds[az_player]
        baseline_score = game.captured_seeds[1 - az_player]
        if az_score > baseline_score:
            az_wins += 1
        elif az_score < baseline_score:
            mcts_wins += 1
        else:
            draws += 1

    return {
        "start_index": start_index,
        "games": games,
        "az_wins": az_wins,
        "mcts_wins": mcts_wins,
        "draws": draws,
    }


def random_for_game(seed: int, game_index: int, acting_player: int) -> random.Random:
    return random.Random(seed + (game_index * 37) + acting_player)


def build_report(*, games: int, az_base_simulations: int, mcts_simulations: int, results: list[dict]) -> dict:
    az_wins = sum(int(result["az_wins"]) for result in results)
    mcts_wins = sum(int(result["mcts_wins"]) for result in results)
    draws = sum(int(result["draws"]) for result in results)
    score = (az_wins + (0.5 * draws)) / float(games)
    return {
        "schema": "azlite_vs_mcts_v1",
        "games": games,
        "az_base_simulations": az_base_simulations,
        "mcts_simulations": mcts_simulations,
        "az_wins": az_wins,
        "mcts_wins": mcts_wins,
        "draws": draws,
        "score": round(score, 4),
    }


def main() -> None:
    args = parse_args()
    if args.games <= 0:
        raise SystemExit("--games must be > 0")

    challenger_path = Path(args.challenger_path).resolve()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    search_options = build_eval_search_options(**search_options_from_args(args))

    counts = partition_counts(args.games, args.workers)
    starts = partition_starts(counts)

    results: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(
                run_worker,
                challenger_path=str(challenger_path),
                games=count,
                start_index=start_index,
                seed=args.seed,
                az_base_simulations=args.az_base_simulations,
                mcts_simulations=args.mcts_simulations,
                search_options=search_options,
            )
            for start_index, count in zip(starts, counts, strict=True)
            if count > 0
        ]
        results = [future.result() for future in futures]

    results.sort(key=lambda item: int(item["start_index"]))
    report = build_report(
        games=args.games,
        az_base_simulations=args.az_base_simulations,
        mcts_simulations=args.mcts_simulations,
        results=results,
    )
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote mcts1200 baseline report to {out_path}")


if __name__ == "__main__":
    main()
