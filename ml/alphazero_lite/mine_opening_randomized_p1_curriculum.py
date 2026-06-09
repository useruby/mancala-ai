#!/usr/bin/env python3
"""Mine opening-randomized P1 curriculum states from iter0_reference.

Generates randomized opening prefixes across multiple ply counts.
Plays paired games: iter0_reference (challenger) vs current.
Collects every position where the challenger is P1.
Relabels each position with high-search classic MCTS at 1200 simulations.

Classifies opening prefixes into:
  A. positive P1: prefixes where challenger wins or draws as P1 at practical budget
  B. hard P1:    prefixes where challenger loses as P1

Outputs train and holdout JSONL files for curriculum training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import (
    DEFAULT_SEARCH_OPTIONS,
    SUPPORTED_POLICY_TARGET_MODES,
    SUPPORTED_VALUE_TARGET_MODES,
    build_policy_target,
    canonical_value_target,
    encode_state,
    top_policy_move_for_legal_moves,
    visits_from_classic_mcts_root,
    value_from_classic_mcts_root,
)

PITS_PER_PLAYER = 6
MAX_MOVES = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine opening-randomized P1 curriculum states from iter0_reference."
    )
    parser.add_argument("--out-train", required=True, help="Training JSONL output path")
    parser.add_argument(
        "--out-holdout", required=True, help="Holdout JSONL output path"
    )
    parser.add_argument(
        "--out-summary",
        default=None,
        help="Summary JSON output path",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to iter0_reference artifact directory",
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to current model artifact directory",
    )
    parser.add_argument(
        "--opening-plies",
        default="2,4,6",
        help="Comma-separated opening ply counts (default: 2,4,6)",
    )
    parser.add_argument(
        "--opening-samples",
        type=int,
        default=128,
        help="Distinct opening prefixes per ply count",
    )
    parser.add_argument(
        "--games-per-opening",
        type=int,
        default=4,
        help="Games per opening prefix",
    )
    parser.add_argument(
        "--opening-seed",
        type=int,
        default=48,
        help="Seed for opening prefix generation",
    )
    parser.add_argument(
        "--budget-pairs",
        default="384:256,768:256",
        help="Comma-separated budget pairs for classification",
    )
    parser.add_argument(
        "--teacher-mode",
        default="classic_mcts",
        choices=["classic_mcts"],
    )
    parser.add_argument(
        "--teacher-simulations",
        type=int,
        default=1200,
        help="Classic MCTS simulations for relabeling each state",
    )
    parser.add_argument(
        "--target-train-rows",
        type=int,
        default=1536,
        help="Target train row count",
    )
    parser.add_argument(
        "--target-holdout-rows",
        type=int,
        default=384,
        help="Target holdout row count",
    )
    parser.add_argument(
        "--positive-bucket-share",
        type=float,
        default=0.70,
        help="Fraction of rows from positive P1 prefixes",
    )
    parser.add_argument(
        "--input-encoding",
        default="kalah_v3",
        choices=["kalah_v1", "kalah_v2", "kalah_v3"],
    )
    parser.add_argument(
        "--policy-target-mode",
        default="sharpened",
        choices=sorted(SUPPORTED_POLICY_TARGET_MODES),
    )
    parser.add_argument(
        "--value-target-mode",
        default="sharpened",
        choices=sorted(SUPPORTED_VALUE_TARGET_MODES),
    )
    parser.add_argument(
        "--policy-temperature",
        type=float,
        default=1.0,
        help="Temperature for converting visits to policy target",
    )
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument(
        "--max-positions-per-game",
        type=int,
        default=12,
        help="Maximum positions to keep per game after filtering",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.25,
        help="PUCT exploration constant",
    )
    parser.add_argument(
        "--tactical-root-bias",
        type=float,
        default=DEFAULT_SEARCH_OPTIONS["tactical_root_bias"],
    )
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


def generate_opening_moves(
    *,
    opening_plies: int,
    seed: int,
) -> list[int]:
    game = initial_game()
    rng = np.random.default_rng(seed)
    moves: list[int] = []
    for _ in range(opening_plies):
        if game.over():
            break
        legal = game.possible_moves()
        if not legal:
            break
        move = int(rng.choice(legal))
        relative = game.pit_index(move)
        if not game.move(relative):
            break
        moves.append(relative)
    return moves


def apply_opening_moves(game: KalahGame, moves: list[int]) -> int:
    applied = 0
    for move in moves:
        if game.over():
            break
        legal = game.possible_moves()
        if not legal:
            break
        if move not in legal:
            break
        if not game.move(move):
            break
        applied += 1
    return applied


def game_from_opening_moves(moves: list[int]) -> KalahGame:
    game = initial_game()
    apply_opening_moves(game, moves)
    return game


def _phase_label(move_index: int) -> str:
    if move_index <= 8:
        return "early"
    if move_index <= 24:
        return "mid"
    return "late"


def _state_fingerprint(encoded_state: list[float]) -> str:
    raw = json.dumps([round(v, 6) for v in encoded_state], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _top_move(visits: list[float], legal_moves: list[int]) -> int | None:
    if not legal_moves:
        return None
    best_move = legal_moves[0]
    best_visits = visits[best_move]
    for move in legal_moves[1:]:
        if visits[move] > best_visits:
            best_move = move
            best_visits = visits[move]
    return best_move


def _top_visit_share(visits: list[float], legal_moves: list[int]) -> float:
    total = sum(float(visits[m]) for m in legal_moves)
    if total <= 0:
        return 1.0 / max(len(legal_moves), 1)
    top = max(float(visits[m]) for m in legal_moves)
    return top / total


def _prob_for_move(policy: np.ndarray, move: int) -> float:
    return float(policy[move])


def run_curriculum_mining(
    *,
    candidate_path: str,
    current_path: str,
    opening_plies_list: list[int],
    opening_samples: int,
    games_per_opening: int,
    opening_seed: int,
    budget_pairs: list[tuple[int, int]],
    teacher_simulations: int,
    seed: int,
    max_positions_per_game: int,
    input_encoding: str,
    policy_target_mode: str,
    value_target_mode: str,
    policy_temperature: float,
    target_train_rows: int,
    target_holdout_rows: int,
    positive_bucket_share: float,
    c_puct: float,
    tactical_root_bias: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    candidate_evaluator = ArtifactEvaluator(Path(candidate_path))
    current_evaluator = ArtifactEvaluator(Path(current_path))

    all_mined_rows: list[dict[str, Any]] = []
    fingerprint_counts: dict[str, int] = {}
    max_fingerprint_copies = max(200, max_positions_per_game * 4)

    # Statistics
    total_positions_visited = 0
    total_positions_kept = 0
    p1_positions_visited = 0
    phase_counts: dict[str, int] = {"early": 0, "mid": 0, "late": 0}
    filter_reasons: dict[str, int] = {}
    ply_counts_counter: Counter = Counter()
    positive_prefix_count = 0
    hard_prefix_count = 0
    positive_rows_count = 0
    hard_rows_count = 0
    source_prefix_count = 0
    p1_win_games = 0
    p1_draw_games = 0
    p1_loss_games = 0
    first_divergence_plies: list[int] = []
    unique_boards: set[str] = set()
    low_budget_vs_teacher_disagreements = 0

    # Use the first budget pair as the game-play budget for classification
    game_budget_challenger, game_budget_current = budget_pairs[0]

    global_game_index = 0

    for ply_count in opening_plies_list:
        # Generate opening prefixes for this ply count
        opening_prefixes: list[list[int]] = []
        for sample_idx in range(opening_samples):
            prefix_seed = opening_seed + ply_count * 100_003 + sample_idx * 10007
            moves = generate_opening_moves(
                opening_plies=ply_count,
                seed=prefix_seed,
            )
            if len(moves) == ply_count:
                opening_prefixes.append(moves)

        for prefix_moves in opening_prefixes:
            prefix_p1_outcomes: list[str] = []  # "win", "draw", "loss"
            prefix_rows: list[dict[str, Any]] = []

            for game_in_prefix in range(games_per_opening):
                game = game_from_opening_moves(prefix_moves)
                candidate_seat = game_in_prefix % 2
                game_positions: list[dict[str, Any]] = []
                candidate_wins_game = None
                first_divergence_ply: int | None = None

                for move_index in range(MAX_MOVES):
                    if game.over():
                        candidate_wins_game = (
                            None
                            if game.winner is None
                            else (game.winner == candidate_seat)
                        )
                        break
                    legal_moves = game.possible_moves()
                    if not legal_moves:
                        break

                    is_p1 = game.current_player == 1
                    is_candidate_p1 = game.current_player == candidate_seat and is_p1

                    raw_state = game.to_state()
                    encoded = encode_state(raw_state, input_encoding=input_encoding)

                    candidate_policy, _candidate_value = candidate_evaluator.evaluate(
                        game
                    )

                    low_budget_top = top_policy_move_for_legal_moves(
                        list(candidate_policy), legal_moves
                    )

                    if is_candidate_p1:
                        p1_positions_visited += 1

                        mcts = ClassicMCTS(
                            game.clone(),
                            simulations=teacher_simulations,
                            seed=search_seed(seed, global_game_index, move_index),
                        )
                        root = mcts.search_root()
                        teacher_visits = visits_from_classic_mcts_root(root)
                        teacher_value = value_from_classic_mcts_root(root)

                        teacher_policy = build_policy_target(
                            visits=np.asarray(teacher_visits, dtype=np.float64),
                            legal_moves=legal_moves,
                            temperature=policy_temperature,
                            mode=policy_target_mode,
                        )

                        teacher_top = _top_move(teacher_visits, legal_moves)
                        teacher_top_share = _top_visit_share(
                            teacher_visits, legal_moves
                        )
                        low_prob_on_teacher_top = (
                            _prob_for_move(candidate_policy, teacher_top)
                            if teacher_top is not None
                            else 0.0
                        )

                        filters_passed: list[str] = []

                        if low_budget_top is not None and teacher_top is not None:
                            if low_budget_top != teacher_top:
                                filters_passed.append("top_move_disagreement")

                        if teacher_top_share >= 0.55:
                            filters_passed.append("strong_high_search_preference")

                        if teacher_top is not None and low_prob_on_teacher_top <= 0.30:
                            filters_passed.append("low_prob_on_high_search_top_move")

                        if not filters_passed:
                            filter_reasons["no_divergence"] = (
                                filter_reasons.get("no_divergence", 0) + 1
                            )

                        if first_divergence_ply is None and filters_passed:
                            first_divergence_ply = move_index

                        policy_entropy = float(
                            -np.sum(
                                np.asarray(candidate_policy)[legal_moves]
                                * np.log(
                                    np.maximum(
                                        np.asarray(candidate_policy)[legal_moves],
                                        1e-12,
                                    )
                                )
                            )
                        )

                        total_positions_visited += 1
                        game_positions.append(
                            {
                                "encoded_state": encoded,
                                "player": int(game.current_player),
                                "move_index": move_index,
                                "legal_moves": legal_moves,
                                "low_budget_policy": [
                                    float(v) for v in candidate_policy
                                ],
                                "low_budget_top_move": low_budget_top,
                                "teacher_visits": teacher_visits,
                                "teacher_policy": teacher_policy,
                                "teacher_value": teacher_value,
                                "teacher_top_move": teacher_top,
                                "teacher_top_visit_share": teacher_top_share,
                                "low_budget_prob_on_teacher_top": low_prob_on_teacher_top,
                                "filters_passed": list(filters_passed),
                                "game_index": global_game_index,
                                "candidate_seat": candidate_seat,
                                "policy_entropy": policy_entropy,
                                "ply_count": ply_count,
                            }
                        )

                    if game.current_player == candidate_seat:
                        chosen_move = top_policy_move_for_legal_moves(
                            list(candidate_policy), legal_moves
                        )
                    else:
                        current_policy, _ = current_evaluator.evaluate(game)
                        chosen_move = top_policy_move_for_legal_moves(
                            list(current_policy), legal_moves
                        )

                    if chosen_move is None or chosen_move not in legal_moves:
                        chosen_move = rng.choice(legal_moves)
                    if not game.move(game.pit_index(chosen_move)):
                        break

                if candidate_wins_game is None and not game.over():
                    candidate_wins_game = None

                if first_divergence_ply is not None:
                    first_divergence_plies.append(first_divergence_ply)

                # Record outcome for prefix classification (only when candidate is P1)
                if candidate_seat == 1:
                    if candidate_wins_game is True:
                        prefix_p1_outcomes.append("win")
                        p1_win_games += 1
                    elif candidate_wins_game is False:
                        prefix_p1_outcomes.append("loss")
                        p1_loss_games += 1
                    elif candidate_wins_game is None:
                        prefix_p1_outcomes.append("draw")
                        p1_draw_games += 1

                # Filter and keep rows from this game
                keep: list[dict[str, Any]] = []
                for pos in game_positions:
                    if not pos["filters_passed"]:
                        continue
                    fp = _state_fingerprint(pos["encoded_state"])
                    fp_count = fingerprint_counts.get(fp, 0)
                    if fp_count >= max_fingerprint_copies:
                        continue
                    fingerprint_counts[fp] = fp_count + 1
                    keep.append(pos)

                if len(keep) > max_positions_per_game:
                    keep.sort(
                        key=lambda p: (
                            -p["teacher_top_visit_share"],
                            p["move_index"],
                        )
                    )
                    keep = keep[:max_positions_per_game]

                if keep:
                    for pos in keep:
                        for fname in pos["filters_passed"]:
                            filter_reasons[fname] = filter_reasons.get(fname, 0) + 1

                        player = pos["player"]
                        outcome_value = (
                            0.0
                            if candidate_wins_game is None
                            else (1.0 if candidate_wins_game else -1.0)
                        )
                        value = canonical_value_target(
                            outcome_value=outcome_value,
                            search_value=pos["teacher_value"],
                            move_index=pos["move_index"],
                            mode=value_target_mode,
                        )
                        board_fp = _state_fingerprint(pos["encoded_state"])
                        unique_boards.add(board_fp)

                        row = {
                            "move_index": pos["move_index"],
                            "player": player,
                            "state": pos["encoded_state"],
                            "policy": pos["teacher_policy"],
                            "policy_target_mode": policy_target_mode,
                            "value_target_mode": value_target_mode,
                            "value": value,
                            "source_game_id": pos["game_index"],
                            "source_ply": pos["move_index"],
                            "candidate_seat": pos["candidate_seat"],
                            "low_budget_top_move": pos["low_budget_top_move"],
                            "teacher_top_move": pos["teacher_top_move"],
                            "teacher_top_visit_share": pos["teacher_top_visit_share"],
                            "low_budget_prob_on_teacher_top": pos[
                                "low_budget_prob_on_teacher_top"
                            ],
                            "filters_passed": pos["filters_passed"],
                            "teacher_value_at_state": pos["teacher_value"],
                            "policy_entropy": pos["policy_entropy"],
                            "opening_ply_count": ply_count,
                        }
                        prefix_rows.append(row)
                        phase_counts[_phase_label(pos["move_index"])] += 1
                        total_positions_kept += 1

                global_game_index += 1

            # Classify prefix
            p1_wins = prefix_p1_outcomes.count("win")
            p1_draws_prefix = prefix_p1_outcomes.count("draw")
            is_positive = (p1_wins + p1_draws_prefix) > 0

            if prefix_rows:
                source_prefix_count += 1
                if is_positive:
                    positive_prefix_count += 1
                else:
                    hard_prefix_count += 1

            for row in prefix_rows:
                row["prefix_bucket"] = "positive" if is_positive else "hard"
                row["opening_prefix_moves"] = prefix_moves
                if row["low_budget_top_move"] != row["teacher_top_move"]:
                    low_budget_vs_teacher_disagreements += 1

            all_mined_rows.extend(prefix_rows)

            ply_counts_counter[ply_count] += len(prefix_rows)

    # Shuffle and apply bucket constraints
    rng.shuffle(all_mined_rows)
    positive_rows = [r for r in all_mined_rows if r["prefix_bucket"] == "positive"]
    hard_rows = [r for r in all_mined_rows if r["prefix_bucket"] == "hard"]

    total_positive = len(positive_rows)
    total_hard = len(hard_rows)
    positive_share = total_positive / max(total_positive + total_hard, 1)

    # Target counts with bucket share
    positive_target = max(1, round(target_train_rows * positive_bucket_share))
    hard_target = max(1, target_train_rows - positive_target)
    holdout_positive_target = max(1, round(target_holdout_rows * positive_bucket_share))
    holdout_hard_target = max(1, target_holdout_rows - holdout_positive_target)

    # Sample from each bucket
    ds_rng = random.Random(seed + 100)
    ds_rng.shuffle(positive_rows)
    ds_rng.shuffle(hard_rows)

    train_positive = positive_rows[:positive_target]
    train_hard = hard_rows[:hard_target]
    holdout_positive = positive_rows[
        positive_target : positive_target + holdout_positive_target
    ]
    holdout_hard = hard_rows[hard_target : hard_target + holdout_hard_target]

    train_rows = train_positive + train_hard
    holdout_rows = holdout_positive + holdout_hard
    ds_rng.shuffle(train_rows)
    ds_rng.shuffle(holdout_rows)
    train_rows = train_rows[:target_train_rows]
    holdout_rows = holdout_rows[:target_holdout_rows]

    positive_rows_count = len(
        [r for r in train_rows if r["prefix_bucket"] == "positive"]
    )
    hard_rows_count = len([r for r in train_rows if r["prefix_bucket"] == "hard"])

    duplicate_count = sum(
        max(0, count - 1) for count in fingerprint_counts.values() if count > 1
    )
    capped_count = sum(
        max(0, count - max_fingerprint_copies)
        for count in fingerprint_counts.values()
        if count > max_fingerprint_copies
    )

    total_kept = len(all_mined_rows)
    teacher_top_shares = [r["teacher_top_visit_share"] for r in all_mined_rows]
    policy_entropies = [r.get("policy_entropy", 0.0) for r in all_mined_rows]

    summary = {
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "total_positions_kept": total_positions_kept,
        "total_positions_visited": total_positions_visited,
        "p1_positions_visited": p1_positions_visited,
        "source_prefix_count": source_prefix_count,
        "positive_prefix_count": positive_prefix_count,
        "hard_prefix_count": hard_prefix_count,
        "positive_rows_in_train": positive_rows_count,
        "hard_rows_in_train": hard_rows_count,
        "positive_bucket_share_actual": round(positive_share, 4),
        "opening_ply_distribution": dict(ply_counts_counter),
        "rows_per_phase": {
            "early": phase_counts.get("early", 0),
            "mid": phase_counts.get("mid", 0),
            "late": phase_counts.get("late", 0),
        },
        "p1_win_games": p1_win_games,
        "p1_draw_games": p1_draw_games,
        "p1_loss_games": p1_loss_games,
        "filter_reasons": filter_reasons,
        "low_budget_vs_teacher_disagreement_rate": round(
            low_budget_vs_teacher_disagreements / max(total_kept, 1), 4
        ),
        "mean_teacher_top1_visit_share": round(
            float(np.mean(teacher_top_shares)) if teacher_top_shares else 0.0, 4
        ),
        "mean_policy_entropy": round(
            float(np.mean(policy_entropies)) if policy_entropies else 0.0, 4
        ),
        "first_divergence_ply_mean": round(
            float(np.mean(first_divergence_plies)) if first_divergence_plies else 0.0,
            2,
        ),
        "first_divergence_ply_median": round(
            float(np.median(first_divergence_plies)) if first_divergence_plies else 0.0,
            2,
        ),
        "duplicate_state_count": duplicate_count,
        "capped_state_count": capped_count,
        "unique_board_count": len(unique_boards),
        "teacher_simulations": teacher_simulations,
        "input_encoding": input_encoding,
        "policy_target_mode": policy_target_mode,
        "value_target_mode": value_target_mode,
        "max_positions_per_game": max_positions_per_game,
        "positive_bucket_share_target": positive_bucket_share,
        "seed": seed,
        "opening_seed": opening_seed,
        "opening_plies": opening_plies_list,
        "opening_samples": opening_samples,
        "games_per_opening": games_per_opening,
        "budget_pairs": [[int(a), int(b)] for a, b in budget_pairs],
    }

    return train_rows, holdout_rows, summary


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return sha


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    opening_plies_list = [
        int(p.strip()) for p in args.opening_plies.split(",") if p.strip()
    ]
    if not opening_plies_list:
        raise SystemExit("--opening-plies must provide at least one ply count")

    budget_pairs: list[tuple[int, int]] = []
    for bp in args.budget_pairs.split(","):
        bp = bp.strip()
        if ":" in bp:
            c_str, cur_str = bp.split(":", 1)
            budget_pairs.append((int(c_str), int(cur_str)))
    if not budget_pairs:
        raise SystemExit("--budget-pairs must provide at least one budget pair")

    train_rows, holdout_rows, summary = run_curriculum_mining(
        candidate_path=args.candidate,
        current_path=args.current,
        opening_plies_list=opening_plies_list,
        opening_samples=args.opening_samples,
        games_per_opening=args.games_per_opening,
        opening_seed=args.opening_seed,
        budget_pairs=budget_pairs,
        teacher_simulations=args.teacher_simulations,
        seed=args.seed,
        max_positions_per_game=args.max_positions_per_game,
        input_encoding=args.input_encoding,
        policy_target_mode=args.policy_target_mode,
        value_target_mode=args.value_target_mode,
        policy_temperature=args.policy_temperature,
        target_train_rows=args.target_train_rows,
        target_holdout_rows=args.target_holdout_rows,
        positive_bucket_share=args.positive_bucket_share,
        c_puct=args.c_puct,
        tactical_root_bias=args.tactical_root_bias,
    )

    out_train = Path(args.out_train)
    out_holdout = Path(args.out_holdout)

    train_sha = write_jsonl(train_rows, out_train)
    holdout_sha = write_jsonl(holdout_rows, out_holdout)

    elapsed = time.perf_counter() - started

    summary["train_sha256"] = train_sha
    summary["holdout_sha256"] = holdout_sha
    summary["train_path"] = str(out_train)
    summary["holdout_path"] = str(out_holdout)
    summary["elapsed_seconds"] = round(elapsed, 1)

    summary_path = (
        Path(args.out_summary)
        if args.out_summary
        else out_train.with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"curriculum_train_rows={len(train_rows)}")
    print(f"curriculum_holdout_rows={len(holdout_rows)}")
    print(f"curriculum_train_sha256={train_sha}")
    print(f"curriculum_holdout_sha256={holdout_sha}")
    print(f"curriculum_prefixes={summary['source_prefix_count']}")
    print(f"curriculum_positive_prefixes={summary['positive_prefix_count']}")
    print(f"curriculum_hard_prefixes={summary['hard_prefix_count']}")
    print(f"curriculum_positions_kept={summary['total_positions_kept']}")
    print(f"curriculum_elapsed_seconds={elapsed:.1f}")
    print(f"summary_written={summary_path}")


if __name__ == "__main__":
    main()
