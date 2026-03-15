#!/usr/bin/env python3
"""Arena evaluator for candidate-vs-current AlphaZero-lite checkpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.input_encodings import DEFAULT_INPUT_ENCODING
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import (
    PUCT,
    DEFAULT_EVAL_SEARCH_OPTIONS,
    DEFAULT_SEARCH_OPTIONS,
    add_search_option_args,
    build_eval_search_options,
    build_search_options,
    encode_state,
    search_options_from_args,
)


PITS_PER_PLAYER = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--challenger-simulations", type=int, default=384)
    parser.add_argument("--current-simulations", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", required=True)
    add_search_option_args(parser)
    parser.set_defaults(
        root_policy_mode=DEFAULT_EVAL_SEARCH_OPTIONS["root_policy_mode"],
        tactical_root_bias=DEFAULT_EVAL_SEARCH_OPTIONS["tactical_root_bias"],
    )
    return parser.parse_args()


def partition_counts(total: int, workers: int) -> list[int]:
    workers = max(1, workers)
    base = total // workers
    remainder = total % workers
    return [base + (1 if i < remainder else 0) for i in range(workers)]


class ArtifactEvaluator:
    def __init__(self, artifact_dir: Path):
        metadata_path = artifact_dir / "metadata.json"
        weights_path = artifact_dir / "weights.json"
        if not weights_path.exists():
            raise FileNotFoundError(f"missing weights.json in artifact directory: {artifact_dir}")

        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        weights = json.loads(weights_path.read_text(encoding="utf-8"))
        architecture = metadata.get("architecture", {})
        self.model_type = architecture.get("model_type", "mlp_v1")
        self.input_encoding = metadata.get("input_encoding", DEFAULT_INPUT_ENCODING)
        self.residual_blocks = self._extract_residual_blocks(weights)
        self.hidden_layers = [] if self.residual_blocks else self._extract_hidden_layers(weights)
        self.w_input = np.asarray(weights["w_input"], dtype=np.float32) if "w_input" in weights else None
        self.b_input = np.asarray(weights["b_input"], dtype=np.float32) if "b_input" in weights else None
        self.w_policy = np.asarray(weights["w_policy"], dtype=np.float32)
        self.b_policy = np.asarray(weights["b_policy"], dtype=np.float32)
        self.w_value = np.asarray(weights["w_value"], dtype=np.float32)
        self.b_value = np.asarray(weights["b_value"], dtype=np.float32)
        self.w_policy_hidden = None
        self.b_policy_hidden = None
        self.w_value_hidden = None
        self.b_value_hidden = None
        if self.model_type == "residual_v3":
            required_keys = ["w_policy_hidden", "b_policy_hidden", "w_value_hidden", "b_value_hidden"]
            missing_keys = [key for key in required_keys if key not in weights]
            if missing_keys:
                missing = ", ".join(missing_keys)
                raise ValueError(f"residual_v3 artifact is missing specialized head weights: {missing}")
            self.w_policy_hidden = np.asarray(weights["w_policy_hidden"], dtype=np.float32)
            self.b_policy_hidden = np.asarray(weights["b_policy_hidden"], dtype=np.float32)
            self.w_value_hidden = np.asarray(weights["w_value_hidden"], dtype=np.float32)
            self.b_value_hidden = np.asarray(weights["b_value_hidden"], dtype=np.float32)

    def _extract_residual_blocks(self, weights: dict) -> list[tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]:
        if "w_input" not in weights or "b_input" not in weights:
            return []

        blocks = []
        index = 1
        while f"w_residual_{index}_1" in weights and f"b_residual_{index}_1" in weights:
            blocks.append(
                (
                    (
                        np.asarray(weights[f"w_residual_{index}_1"], dtype=np.float32),
                        np.asarray(weights[f"b_residual_{index}_1"], dtype=np.float32),
                    ),
                    (
                        np.asarray(weights[f"w_residual_{index}_2"], dtype=np.float32),
                        np.asarray(weights[f"b_residual_{index}_2"], dtype=np.float32),
                    ),
                )
            )
            index += 1
        return blocks

    def _extract_hidden_layers(self, weights: dict) -> list[tuple[np.ndarray, np.ndarray]]:
        indexed_layers: list[tuple[np.ndarray, np.ndarray]] = []
        index = 1
        while f"w_hidden_{index}" in weights and f"b_hidden_{index}" in weights:
            w = np.asarray(weights[f"w_hidden_{index}"], dtype=np.float32)
            b = np.asarray(weights[f"b_hidden_{index}"], dtype=np.float32)
            indexed_layers.append((w, b))
            index += 1

        if indexed_layers:
            return indexed_layers

        return [
            (np.asarray(weights["w1"], dtype=np.float32), np.asarray(weights["b1"], dtype=np.float32)),
            (np.asarray(weights["w2"], dtype=np.float32), np.asarray(weights["b2"], dtype=np.float32)),
        ]

    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        x = np.asarray(encode_state(game.to_state(), input_encoding=self.input_encoding), dtype=np.float32)

        if self.residual_blocks:
            assert self.w_input is not None
            assert self.b_input is not None
            hidden = np.maximum(0.0, (x @ self.w_input) + self.b_input)
            for (w1, b1), (w2, b2) in self.residual_blocks:
                residual = hidden
                hidden = np.maximum(0.0, (hidden @ w1) + b1)
                hidden = np.maximum(0.0, (hidden @ w2) + b2 + residual)
        else:
            hidden = x
            for w, b in self.hidden_layers:
                hidden = np.maximum(0.0, (hidden @ w) + b)

        policy_hidden = hidden
        value_hidden = hidden
        if self.model_type == "residual_v3":
            assert self.w_policy_hidden is not None
            assert self.b_policy_hidden is not None
            assert self.w_value_hidden is not None
            assert self.b_value_hidden is not None
            policy_hidden = np.maximum(0.0, (hidden @ self.w_policy_hidden) + self.b_policy_hidden)
            value_hidden = np.maximum(0.0, (hidden @ self.w_value_hidden) + self.b_value_hidden)

        logits = (policy_hidden @ self.w_policy) + self.b_policy
        logits = logits - np.max(logits)
        exp_values = np.exp(logits)
        priors = exp_values / np.sum(exp_values)

        legal_moves = game.possible_moves()
        masked = np.zeros(PITS_PER_PLAYER, dtype=np.float32)
        if legal_moves:
            masked[legal_moves] = priors[legal_moves]
            total = float(np.sum(masked))
            if total <= 0:
                masked[legal_moves] = 1.0 / len(legal_moves)
            else:
                masked /= total

        value = float(np.tanh((value_hidden @ self.w_value + self.b_value).reshape(-1)[0]))
        return masked, value


def choose_best_move(visits: np.ndarray, legal_moves: list[int]) -> int:
    legal_visits = {move: visits[move] for move in legal_moves}
    best_visit = max(legal_visits.values())
    best_moves = [move for move, visit in legal_visits.items() if visit == best_visit]
    return min(best_moves)


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile_value))


def run_arena_worker(
    *,
    worker_id: int,
    start_index: int,
    games: int,
    challenger_path: str,
    current_path: str,
    challenger_simulations: int,
    current_simulations: int,
    seed: int,
    c_puct: float,
    max_moves: int,
    fpu_mode: str = DEFAULT_SEARCH_OPTIONS["fpu_mode"],
    reuse_subtree: bool = DEFAULT_SEARCH_OPTIONS["reuse_subtree"],
    normalize_values: bool = DEFAULT_SEARCH_OPTIONS["normalize_values"],
    root_policy_mode: str = DEFAULT_EVAL_SEARCH_OPTIONS["root_policy_mode"],
    tactical_root_bias: float = DEFAULT_EVAL_SEARCH_OPTIONS["tactical_root_bias"],
) -> dict:
    rng = np.random.default_rng(seed + worker_id)
    challenger = ArtifactEvaluator(Path(challenger_path))
    current = ArtifactEvaluator(Path(current_path))

    wins = 0
    losses = 0
    draws = 0
    move_durations_ms: list[float] = []

    for local_index in range(games):
        game_index = start_index + local_index
        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        challenger_player = 0 if game_index % 2 == 0 else 1
        reusable_roots = {
            0: None,
            1: None,
        }

        for _ in range(max_moves):
            if game.over():
                break

            legal_moves = game.possible_moves()
            if not legal_moves:
                break

            if game.current_player == challenger_player:
                evaluator = challenger
                sims = challenger_simulations
            else:
                evaluator = current
                sims = current_simulations
            acting_player = game.current_player

            search = PUCT(
                evaluator=evaluator,
                simulations=sims,
                c_puct=c_puct,
                rng=random.Random(int(rng.integers(0, 2**31 - 1))),
                root=reusable_roots[acting_player],
                fpu_mode=fpu_mode,
                reuse_subtree=reuse_subtree,
                normalize_values=normalize_values,
                root_policy_mode=root_policy_mode,
                tactical_root_bias=tactical_root_bias,
            )
            started = time.perf_counter()
            visits, root = search.run(game)
            move_durations_ms.append((time.perf_counter() - started) * 1000.0)

            if hasattr(search, "select_root_move") and root is not None:
                move = search.select_root_move(root, legal_moves)
            else:
                move = choose_best_move(visits, legal_moves)
            if not game.move(game.pit_index(move)):
                break
            if reuse_subtree and root is not None and game.current_player == acting_player:
                reusable_roots[acting_player] = root.child_for_action(move)
            else:
                reusable_roots[acting_player] = None
            if game.current_player != acting_player:
                reusable_roots[game.current_player] = None

        challenger_store = game.captured_seeds[challenger_player]
        current_store = game.captured_seeds[1 - challenger_player]
        if challenger_store > current_store:
            wins += 1
        elif challenger_store < current_store:
            losses += 1
        else:
            draws += 1

    return {
        "worker_id": worker_id,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "move_durations_ms": move_durations_ms,
        "search_options": build_search_options(
            fpu_mode=fpu_mode,
            reuse_subtree=reuse_subtree,
            normalize_values=normalize_values,
            root_policy_mode=root_policy_mode,
            tactical_root_bias=tactical_root_bias,
        ),
    }


def main() -> None:
    args = parse_args()

    challenger_path = Path(args.challenger)
    current_path = Path(args.current)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workers = max(1, args.workers)
    search_options = build_eval_search_options(**search_options_from_args(args))
    counts = partition_counts(args.games, workers)
    starts: list[int] = []
    cursor = 0
    for count in counts:
        starts.append(cursor)
        cursor += count

    futures = []
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for worker_id, (start_index, count) in enumerate(zip(starts, counts, strict=True)):
            if count <= 0:
                continue
            futures.append(
                pool.submit(
                    run_arena_worker,
                    worker_id=worker_id,
                    start_index=start_index,
                    games=count,
                    challenger_path=str(challenger_path),
                    current_path=str(current_path),
                    challenger_simulations=args.challenger_simulations,
                    current_simulations=args.current_simulations,
                    seed=args.seed,
                    c_puct=args.c_puct,
                    max_moves=args.max_moves,
                    fpu_mode=str(search_options["fpu_mode"]),
                    reuse_subtree=bool(search_options["reuse_subtree"]),
                    normalize_values=bool(search_options["normalize_values"]),
                    root_policy_mode=str(search_options["root_policy_mode"]),
                    tactical_root_bias=float(search_options["tactical_root_bias"]),
                )
            )
        results = [future.result() for future in futures]

    wins = sum(int(result["wins"]) for result in results)
    losses = sum(int(result["losses"]) for result in results)
    draws = sum(int(result["draws"]) for result in results)
    move_durations_ms = [
        duration
        for result in results
        for duration in result["move_durations_ms"]
    ]

    score = (wins + (0.5 * draws)) / float(args.games)
    report = {
        "schema": "arena_v1",
        "games_played": int(args.games),
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "promotion_decision": {
            "passed": bool(score >= args.min_score),
        },
        "notes": {
            "challenger_path": str(challenger_path),
            "current_path": str(current_path),
            "challenger_simulations": int(args.challenger_simulations),
            "current_simulations": int(args.current_simulations),
            "seed": int(args.seed),
            "workers_requested": int(args.workers),
            "workers_used": len(results),
            "worker_game_counts": [int(result["wins"] + result["losses"] + result["draws"]) for result in results],
            "search_options": search_options,
            "move_time_mean_ms": round(statistics.fmean(move_durations_ms), 2) if move_durations_ms else 0.0,
            "move_time_p95_ms": round(percentile(move_durations_ms, 95), 2),
        },
    }

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote arena report to {out_path}")
    print(f"score={score:.4f} passed={report['promotion_decision']['passed']}")


if __name__ == "__main__":
    main()
