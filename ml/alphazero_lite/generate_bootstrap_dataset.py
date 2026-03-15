#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.kalah_rules import KalahGame, NUMBER_OF_PLAYERS, PITS_PER_PLAYER
from ml.alphazero_lite.self_play import (
    DEFAULT_POLICY_TARGET_MODE,
    DEFAULT_VALUE_TARGET_MODE,
    HYBRID_VALUE_TARGET_MODE,
    PUCT,
    HeuristicEvaluator,
    derive_self_play_value_target,
    encode_state,
    merge_worker_shards,
    normalize_policy_target_mode,
    normalize_value_target_mode,
    partition_counts,
    policy_from_visits,
    sample_move,
    value_from_classic_mcts_root,
    visits_from_classic_mcts_root,
)

DISAGREEMENT_DEEPER_SIMULATION_FACTOR = 2
DISAGREEMENT_POLICY_THRESHOLD = 0.2
DEFAULT_DIRICHLET_EPSILON = 0.25
MAX_MOVES = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--simulations", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--input-encoding", default="kalah_v1")
    parser.add_argument("--max-positions-per-game", type=int, default=24)
    parser.add_argument("--tree-reuse-enabled", action="store_true")
    parser.add_argument("--position-selection-mode", default="generic")
    parser.add_argument("--policy-target-mode", default=DEFAULT_POLICY_TARGET_MODE)
    parser.add_argument("--value-target-mode", default=DEFAULT_VALUE_TARGET_MODE)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--dirichlet-alpha", type=float)
    parser.add_argument("--dirichlet-epsilon", type=float, default=DEFAULT_DIRICHLET_EPSILON)
    parser.add_argument("--dirichlet-opening-moves", type=int, default=0)
    parser.add_argument("--teacher-mode", default="puct", choices=["puct", "classic_mcts"])
    return parser.parse_args()


def search_seed(base_seed: int, game_index: int, move_index: int) -> int:
    return (base_seed * 1_000_003) + (game_index * 10_007) + move_index


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


def shape_policy(
    *,
    visits: list[float],
    legal_moves: list[int],
    move_index: int,
    rng: random.Random,
    apply_dirichlet: bool,
    top_k: int | None,
    tau: float,
    dirichlet_alpha: float | None,
    dirichlet_epsilon: float,
    dirichlet_opening_moves: int,
) -> list[float]:
    if not legal_moves:
        return [0.0] * PITS_PER_PLAYER

    weighted = [0.0] * PITS_PER_PLAYER
    for move in legal_moves:
        base = float(visits[move])
        weighted[move] = 0.0 if base == 0.0 else base ** (1.0 / tau)

    if apply_dirichlet and dirichlet_alpha and dirichlet_alpha > 0 and move_index < dirichlet_opening_moves:
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).dirichlet([dirichlet_alpha] * len(legal_moves))
        for index, move in enumerate(legal_moves):
            weighted[move] = ((1.0 - dirichlet_epsilon) * weighted[move]) + (dirichlet_epsilon * float(noise[index]))

    if top_k and 0 < top_k < len(legal_moves):
        keep = sorted(legal_moves, key=lambda move: (-weighted[move], move))[:top_k]
        filtered = [0.0] * PITS_PER_PLAYER
        for move in keep:
            filtered[move] = weighted[move]
        weighted = filtered

    total = sum(weighted[move] for move in legal_moves)
    if total <= 0.0:
        probability = 1.0 / len(legal_moves)
        return [probability if move in legal_moves else 0.0 for move in range(PITS_PER_PLAYER)]

    return [weighted[move] / total if move in legal_moves else 0.0 for move in range(PITS_PER_PLAYER)]
def build_policy_target(
    *,
    visits: list[float],
    legal_moves: list[int],
    move_index: int,
    rng: random.Random,
    mode: str,
    tau: float,
    top_k: int | None,
) -> list[float]:
    default_target = shape_policy(
        visits=visits,
        legal_moves=legal_moves,
        move_index=move_index,
        rng=rng,
        apply_dirichlet=False,
        top_k=top_k,
        tau=tau,
        dirichlet_alpha=None,
        dirichlet_epsilon=DEFAULT_DIRICHLET_EPSILON,
        dirichlet_opening_moves=0,
    )
    if mode != "sharpened":
        return default_target

    sharpened = [value**2 for value in default_target]
    total = sum(sharpened)
    if total <= 0.0:
        return default_target
    return [value / total for value in sharpened]


def policy_divergence(shallow_policy: list[float], deeper_policy: list[float]) -> float:
    return sum(abs(float(a) - float(b)) for a, b in zip(shallow_policy, deeper_policy, strict=True)) / 2.0


def best_move_for(policy: list[float]) -> int | None:
    indexed = list(enumerate(policy))
    if not indexed:
        return None
    return max(indexed, key=lambda item: (float(item[1]), -item[0]))[0]




def disagreement_position(position: dict[str, Any]) -> bool:
    deeper_policy = position.get("deeper_policy")
    if not deeper_policy:
        return False
    shallow_policy = position["policy"]
    return best_move_for(shallow_policy) != best_move_for(deeper_policy) or policy_divergence(shallow_policy, deeper_policy) >= DISAGREEMENT_POLICY_THRESHOLD


def own_store_passes(move: int, seeds: int) -> int:
    distance_to_store = PITS_PER_PLAYER - move
    if seeds < distance_to_store:
        return 0
    return 1 + ((seeds - distance_to_store) // ((PITS_PER_PLAYER * NUMBER_OF_PLAYERS) + 1))


def game_after_move(game: KalahGame, move: int) -> KalahGame:
    simulated = game.clone()
    simulated.move(simulated.pit_index(move))
    return simulated


def extra_turn_move(game: KalahGame, move: int) -> bool:
    return game_after_move(game, move).current_player == game.current_player


def capture_move(game: KalahGame, move: int) -> bool:
    player = game.current_player
    absolute_index = game.pit_index(move)
    seeds = game.pits[absolute_index]
    store_before = game.captured_seeds[player]
    simulated = game_after_move(game, move)
    return (simulated.captured_seeds[player] - store_before) > own_store_passes(move, seeds)


def immediate_capture_available(game: KalahGame) -> bool:
    return any(capture_move(game, move) for move in game.possible_moves())


def prevents_immediate_opponent_capture(game: KalahGame, move: int) -> bool:
    player = game.current_player
    simulated = game_after_move(game, move)
    if simulated.current_player == player or immediate_capture_available(simulated):
        return False
    return any(
        game_after_move(game, alternative).current_player != player and immediate_capture_available(game_after_move(game, alternative))
        for alternative in game.possible_moves()
        if alternative != move
    )


def tactical_position(position: dict[str, Any]) -> bool:
    game = KalahGame.from_state(position["game_state"])
    return any(extra_turn_move(game, move) or capture_move(game, move) or prevents_immediate_opponent_capture(game, move) for move in game.possible_moves())


def sample_bucket(bucket: list[dict[str, Any]], target: int, tactical_mode: bool) -> list[dict[str, Any]]:
    if not bucket or target <= 0:
        return []
    ranked = sorted(
        bucket,
        key=lambda position: (
            0 if tactical_mode and tactical_position(position) else 1,
            -sum(float(value) ** 2 for value in position["policy"]),
            position["move_index"],
        ),
    )
    return ranked[: min(target, len(ranked))]


def apply_curriculum(positions: list[dict[str, Any]], max_positions_per_game: int, tactical_mode: bool) -> list[dict[str, Any]]:
    if len(positions) <= max_positions_per_game:
        return positions
    early = [position for position in positions if position["move_index"] <= 8]
    mid = [position for position in positions if 9 <= position["move_index"] <= 24]
    late = [position for position in positions if position["move_index"] >= 25]
    early_target = max(round(max_positions_per_game * 0.34), 1)
    mid_target = max(round(max_positions_per_game * 0.33), 1)
    late_target = max(max_positions_per_game - early_target - mid_target, 1)
    selected = sample_bucket(early, early_target, tactical_mode)
    selected.extend(sample_bucket(mid, mid_target, tactical_mode))
    selected.extend(sample_bucket(late, late_target, tactical_mode))
    if len(selected) < max_positions_per_game:
        leftovers = [position for position in positions if position not in selected]
        selected.extend(sample_bucket(leftovers, max_positions_per_game - len(selected), tactical_mode))
    return selected[:max_positions_per_game]


def selected_teacher_positions(positions: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "tactical":
        return [position for position in positions if tactical_position(position)]
    if mode == "hybrid_teacher":
        selected = []
        for position in positions:
            tactical = tactical_position(position)
            disagreement = disagreement_position(position)
            if tactical and disagreement:
                position = {**position, "teacher_bucket": "both"}
            elif tactical:
                position = {**position, "teacher_bucket": "tactical"}
            elif disagreement:
                position = {**position, "teacher_bucket": "disagreement"}
            else:
                continue
            selected.append(position)
        return selected
    return positions


def annotate_rows(positions: list[dict[str, Any]], *, winner: int | None, value_target_mode: str, position_selection_mode: str) -> list[dict[str, Any]]:
    rows = []
    for position in positions:
        player = int(position["player"])
        outcome_value = 0.0 if winner is None else (1.0 if winner == player else -1.0)
        row = {
            "move_index": int(position["move_index"]),
            "player": player,
            "state": position["state"],
            "policy": position["policy"],
            "policy_target_mode": position["policy_target_mode"],
            "value_target_mode": value_target_mode,
            "value": derive_self_play_value_target(
                outcome_value=outcome_value,
                search_value=float(position.get("root_search_value", 0.0)),
                move_index=int(position["move_index"]),
                mode=value_target_mode,
            ),
        }
        if position_selection_mode == "tactical":
            row["position_selection_mode"] = "tactical"
        elif position_selection_mode == "hybrid_teacher":
            row["position_selection_mode"] = "hybrid_teacher"
            row["teacher_bucket"] = position["teacher_bucket"]
        rows.append(row)
    return rows


def run_worker(
    *,
    worker_id: int,
    start_index: int,
    games: int,
    seed: int,
    simulations: int,
    input_encoding: str,
    max_positions_per_game: int,
    tree_reuse_enabled: bool,
    position_selection_mode: str,
    policy_target_mode: str,
    value_target_mode: str,
    tau: float,
    top_k: int | None,
    dirichlet_alpha: float | None,
    dirichlet_epsilon: float,
    dirichlet_opening_moves: int,
    shard_path: str,
) -> dict[str, Any]:
    rng = random.Random(seed + worker_id)
    evaluator = HeuristicEvaluator()
    rows_written = 0
    positions_visited = 0
    simulations_run = 0
    path = Path(shard_path)

    with path.open("w", encoding="utf-8") as handle:
        for local_index in range(games):
            game_index = start_index + local_index
            game = initial_game()
            positions: list[dict[str, Any]] = []
            reusable_root = None

            for move_index in range(MAX_MOVES):
                if game.over():
                    break
                legal_moves = game.possible_moves()
                if not legal_moves:
                    break

                search_rng = random.Random(search_seed(seed, game_index, move_index))
                search = PUCT(evaluator=evaluator, simulations=simulations, c_puct=1.25, rng=search_rng, root=reusable_root, reuse_subtree=tree_reuse_enabled)
                visits, root = search.run(
                    game,
                    dirichlet_alpha=dirichlet_alpha if move_index < dirichlet_opening_moves else None,
                    dirichlet_epsilon=dirichlet_epsilon,
                )
                visit_list = visits.tolist()
                positions_visited += 1
                simulations_run += simulations

                deeper_policy = None
                if position_selection_mode == "hybrid_teacher":
                    deeper_search = PUCT(
                        evaluator=evaluator,
                        simulations=simulations * DISAGREEMENT_DEEPER_SIMULATION_FACTOR,
                        c_puct=1.25,
                        rng=random.Random(search_seed(seed + 17, game_index, move_index)),
                    )
                    deeper_visits, _ = deeper_search.run(game.clone())
                    deeper_policy = shape_policy(
                        visits=deeper_visits.tolist(),
                        legal_moves=legal_moves,
                        move_index=move_index,
                        rng=rng,
                        apply_dirichlet=False,
                        top_k=top_k,
                        tau=tau,
                        dirichlet_alpha=None,
                        dirichlet_epsilon=dirichlet_epsilon,
                        dirichlet_opening_moves=dirichlet_opening_moves,
                    )
                    simulations_run += simulations * DISAGREEMENT_DEEPER_SIMULATION_FACTOR

                policy = build_policy_target(
                    visits=visit_list,
                    legal_moves=legal_moves,
                    move_index=move_index,
                    rng=rng,
                    mode=policy_target_mode,
                    tau=tau,
                    top_k=top_k,
                )
                sampling_policy = shape_policy(
                    visits=visit_list,
                    legal_moves=legal_moves,
                    move_index=move_index,
                    rng=rng,
                    apply_dirichlet=True,
                    top_k=top_k,
                    tau=tau,
                    dirichlet_alpha=dirichlet_alpha,
                    dirichlet_epsilon=dirichlet_epsilon,
                    dirichlet_opening_moves=dirichlet_opening_moves,
                )
                positions.append(
                    {
                        "state": encode_state(game.to_state(), input_encoding=input_encoding),
                        "game_state": game.to_state(),
                        "player": game.current_player,
                        "move_index": move_index,
                        "policy": policy,
                        "deeper_policy": deeper_policy,
                        "root_search_value": float(root.q_value if root is not None else 0.0),
                        "policy_target_mode": policy_target_mode,
                    }
                )
                move = sample_move(sampling_policy, legal_moves=legal_moves, rng=rng)
                reusable_root = root.child_for_action(move) if tree_reuse_enabled and root is not None else None
                if not game.move(game.pit_index(move)):
                    break

            selected = selected_teacher_positions(
                apply_curriculum(positions, max_positions_per_game=max_positions_per_game, tactical_mode=position_selection_mode == "tactical"),
                mode=position_selection_mode,
            )
            rows = annotate_rows(
                selected,
                winner=game.winner,
                value_target_mode=value_target_mode,
                position_selection_mode=position_selection_mode,
            )
            for row in rows:
                handle.write(json.dumps(row) + "\n")
            rows_written += len(rows)

    return {
        "worker_id": worker_id,
        "rows_written": rows_written,
        "games_processed": games,
        "positions_visited": positions_visited,
        "positions_searched": positions_visited,
        "simulations_run": simulations_run,
        "shard_path": str(path),
    }


def run_worker_classic_mcts(
    *,
    worker_id: int,
    start_index: int,
    games: int,
    seed: int,
    simulations: int,
    input_encoding: str,
    max_positions_per_game: int,
    position_selection_mode: str,
    policy_target_mode: str,
    value_target_mode: str,
    tau: float,
    top_k: int | None,
    shard_path: str,
) -> dict[str, Any]:
    rng = random.Random(seed + worker_id)
    rows_written = 0
    positions_visited = 0
    simulations_run = 0
    path = Path(shard_path)

    with path.open("w", encoding="utf-8") as handle:
        for local_index in range(games):
            game_index = start_index + local_index
            game = initial_game()
            positions: list[dict[str, Any]] = []

            for move_index in range(MAX_MOVES):
                if game.over():
                    break
                legal_moves = game.possible_moves()
                if not legal_moves:
                    break

                mcts = ClassicMCTS(
                    game.clone(),
                    simulations=simulations,
                    seed=search_seed(seed, game_index, move_index),
                )
                root = mcts.search_root()
                visit_list = visits_from_classic_mcts_root(root)
                root_value = value_from_classic_mcts_root(root)

                positions_visited += 1
                simulations_run += simulations

                policy = build_policy_target(
                    visits=visit_list,
                    legal_moves=legal_moves,
                    move_index=move_index,
                    rng=rng,
                    mode=policy_target_mode,
                    tau=tau,
                    top_k=top_k,
                )
                sampling_policy = shape_policy(
                    visits=visit_list,
                    legal_moves=legal_moves,
                    move_index=move_index,
                    rng=rng,
                    apply_dirichlet=False,
                    top_k=top_k,
                    tau=tau,
                    dirichlet_alpha=None,
                    dirichlet_epsilon=DEFAULT_DIRICHLET_EPSILON,
                    dirichlet_opening_moves=0,
                )
                positions.append(
                    {
                        "state": encode_state(game.to_state(), input_encoding=input_encoding),
                        "game_state": game.to_state(),
                        "player": game.current_player,
                        "move_index": move_index,
                        "policy": policy,
                        "deeper_policy": None,
                        "root_search_value": root_value,
                        "policy_target_mode": policy_target_mode,
                    }
                )
                move = sample_move(sampling_policy, legal_moves=legal_moves, rng=rng)
                if not game.move(game.pit_index(move)):
                    break

            selected = selected_teacher_positions(
                apply_curriculum(positions, max_positions_per_game=max_positions_per_game, tactical_mode=position_selection_mode == "tactical"),
                mode=position_selection_mode,
            )
            rows = annotate_rows(
                selected,
                winner=game.winner,
                value_target_mode=value_target_mode,
                position_selection_mode=position_selection_mode,
            )
            for row in rows:
                handle.write(json.dumps(row) + "\n")
            rows_written += len(rows)

    return {
        "worker_id": worker_id,
        "rows_written": rows_written,
        "games_processed": games,
        "positions_visited": positions_visited,
        "positions_searched": positions_visited,
        "simulations_run": simulations_run,
        "shard_path": str(path),
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    workers = max(1, args.workers)
    policy_target_mode = normalize_policy_target_mode(args.policy_target_mode)
    value_target_mode = normalize_value_target_mode(args.value_target_mode)
    position_selection_mode = str(args.position_selection_mode)
    teacher_mode = str(args.teacher_mode)

    game_counts = partition_counts(args.games, workers)
    starts: list[int] = []
    cursor = 0
    for count in game_counts:
        starts.append(cursor)
        cursor += count

    with tempfile.TemporaryDirectory(prefix="azlite-bootstrap-shards-", dir=out_path.parent) as shard_dir:
        shard_paths = [str(Path(shard_dir) / f"worker_{worker_id}.jsonl") for worker_id in range(workers)]
        futures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for worker_id, (start_index, game_count) in enumerate(zip(starts, game_counts, strict=True)):
                if game_count <= 0:
                    Path(shard_paths[worker_id]).write_text("", encoding="utf-8")
                    continue
                if teacher_mode == "classic_mcts":
                    futures.append(
                        pool.submit(
                            run_worker_classic_mcts,
                            worker_id=worker_id,
                            start_index=start_index,
                            games=game_count,
                            seed=args.seed,
                            simulations=args.simulations,
                            input_encoding=args.input_encoding,
                            max_positions_per_game=args.max_positions_per_game,
                            position_selection_mode=position_selection_mode,
                            policy_target_mode=policy_target_mode,
                            value_target_mode=value_target_mode,
                            tau=max(float(args.tau), 0.05),
                            top_k=args.top_k,
                            shard_path=shard_paths[worker_id],
                        )
                    )
                else:
                    futures.append(
                        pool.submit(
                            run_worker,
                            worker_id=worker_id,
                            start_index=start_index,
                            games=game_count,
                            seed=args.seed,
                            simulations=args.simulations,
                            input_encoding=args.input_encoding,
                            max_positions_per_game=args.max_positions_per_game,
                            tree_reuse_enabled=bool(args.tree_reuse_enabled),
                            position_selection_mode=position_selection_mode,
                            policy_target_mode=policy_target_mode,
                            value_target_mode=value_target_mode,
                            tau=max(float(args.tau), 0.05),
                            top_k=args.top_k,
                            dirichlet_alpha=args.dirichlet_alpha,
                            dirichlet_epsilon=max(float(args.dirichlet_epsilon), 0.0),
                            dirichlet_opening_moves=max(int(args.dirichlet_opening_moves), 0),
                            shard_path=shard_paths[worker_id],
                        )
                    )

            results = [future.result() for future in futures]
            results.sort(key=lambda item: int(item["worker_id"]))

        rows_written = merge_worker_shards(results, out_path)

    games_processed = sum(int(result["games_processed"]) for result in results)
    positions_visited = sum(int(result["positions_visited"]) for result in results)
    positions_searched = sum(int(result["positions_searched"]) for result in results)
    simulations_run = sum(int(result["simulations_run"]) for result in results)
    average_rows = 0.0 if games_processed == 0 else rows_written / float(games_processed)

    print(f"wrote {rows_written} rows to {out_path}")
    print(
        "dataset_stats "
        f"games_processed={games_processed} rows_written={rows_written} positions_visited={positions_visited} "
        f"positions_searched={positions_searched} simulations_run={simulations_run} average_rows_retained_per_game={average_rows}"
    )


if __name__ == "__main__":
    main()
