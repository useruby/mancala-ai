#!/usr/bin/env python3
"""Mine disadvantaged-seat distillation states from iter0_reference.

Plays paired games: iter0_reference (challenger) vs current.
Collects every position where the challenger is player 1.
Relabels each position with high-search classic MCTS at 1200 simulations.
Filters to states where low-budget policy diverges from high-search policy.

Outputs train and holdout JSONL files for use in multi-file AlphaZero-lite
training as the disadvantaged-seat distillation curriculum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
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
        description="Mine disadvantaged-seat distillation states from iter0_reference."
    )
    parser.add_argument("--out-train", required=True, help="Training JSONL output path")
    parser.add_argument(
        "--out-holdout", required=True, help="Holdout JSONL output path"
    )
    parser.add_argument(
        "--out-summary",
        default=None,
        help="Summary JSON output path (default: <out-train>.summary.json)",
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
        "--low-budget",
        default="384:256",
        help="Low-budget challenger:current sims pair (default: 384:256)",
    )
    parser.add_argument(
        "--teacher-budget",
        default="1200:1200",
        help="Teacher-budget challenger:current sims pair (default: 1200:1200)",
    )
    parser.add_argument(
        "--teacher-simulations",
        type=int,
        default=1200,
        help="Classic MCTS simulations for relabeling each state",
    )
    parser.add_argument(
        "--games", type=int, default=240, help="Number of games to generate"
    )
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument(
        "--max-positions-per-game",
        type=int,
        default=12,
        help="Maximum positions to keep per game after filtering",
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
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of mined rows for training (remainder is holdout)",
    )
    parser.add_argument(
        "--target-train-rows",
        type=int,
        default=1024,
        help="Optional: downsample train rows to this count",
    )
    parser.add_argument(
        "--target-holdout-rows",
        type=int,
        default=256,
        help="Optional: downsample holdout rows to this count",
    )
    parser.add_argument(
        "--top-visit-share-threshold",
        type=float,
        default=0.55,
        help="Minimum top-1 visit share for high-search strong preference filter",
    )
    parser.add_argument(
        "--low-prob-threshold",
        type=float,
        default=0.30,
        help="Maximum low-budget probability on high-search top move for divergence",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.25,
        help="PUCT exploration constant",
    )
    parser.add_argument(
        "--random-opening-plies",
        type=int,
        default=4,
        help="Number of random opening plies before recording positions (diversifies state space)",
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


def run_distillation_mining(
    *,
    candidate_path: str,
    current_path: str,
    games: int,
    low_challenger_sims: int,
    low_current_sims: int,
    teacher_simulations: int,
    seed: int,
    max_positions_per_game: int,
    input_encoding: str,
    policy_target_mode: str,
    value_target_mode: str,
    policy_temperature: float,
    train_split: float,
    target_train_rows: int,
    target_holdout_rows: int,
    top_visit_share_threshold: float,
    low_prob_threshold: float,
    c_puct: float,
    tactical_root_bias: float,
    random_opening_plies: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    candidate_evaluator = ArtifactEvaluator(Path(candidate_path))
    current_evaluator = ArtifactEvaluator(Path(current_path))

    all_mined_rows: list[dict[str, Any]] = []
    mined_games_count = 0
    total_positions_visited = 0
    total_positions_kept = 0
    p1_positions_visited = 0
    phase_counts: dict[str, int] = {"early": 0, "mid": 0, "late": 0}
    filter_reasons: dict[str, int] = {
        "top_move_disagreement": 0,
        "strong_high_search_preference": 0,
        "low_prob_on_high_search_top_move": 0,
        "no_divergence": 0,
    }
    fingerprint_counts: dict[str, int] = {}
    max_fingerprint_copies = max(200, max_positions_per_game * 4)
    first_divergence_plies: list[int] = []

    for game_index in range(games):
        game = initial_game()
        game_positions: list[dict[str, Any]] = []
        candidate_seat = game_index % 2
        candidate_wins_game = None
        first_divergence_ply: int | None = None

        opening_plies_applied = 0
        if random_opening_plies > 0:
            opening_rng = np.random.default_rng(seed + game_index * 10007)
            for _ in range(random_opening_plies):
                if game.over():
                    break
                legal = game.possible_moves()
                if not legal:
                    break
                chosen = int(opening_rng.choice(legal))
                if not game.move(game.pit_index(chosen)):
                    break
                opening_plies_applied += 1

        for move_index in range(MAX_MOVES):
            if game.over():
                candidate_wins_game = (
                    None if game.winner is None else (game.winner == candidate_seat)
                )
                break
            legal_moves = game.possible_moves()
            if not legal_moves:
                break

            is_p1 = game.current_player == 1
            is_candidate_p1 = game.current_player == candidate_seat and is_p1

            raw_state = game.to_state()
            encoded = encode_state(raw_state, input_encoding=input_encoding)

            candidate_policy, _candidate_value = candidate_evaluator.evaluate(game)

            low_budget_top = top_policy_move_for_legal_moves(
                list(candidate_policy), legal_moves
            )

            if is_candidate_p1:
                p1_positions_visited += 1
                mcts = ClassicMCTS(
                    game.clone(),
                    simulations=teacher_simulations,
                    seed=search_seed(seed, game_index, move_index),
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
                teacher_top_share = _top_visit_share(teacher_visits, legal_moves)
                low_prob_on_teacher_top = (
                    _prob_for_move(candidate_policy, teacher_top)
                    if teacher_top is not None
                    else 0.0
                )

                filters_passed: list[str] = []

                if low_budget_top is not None and teacher_top is not None:
                    if low_budget_top != teacher_top:
                        filters_passed.append("top_move_disagreement")

                if teacher_top_share >= top_visit_share_threshold:
                    filters_passed.append("strong_high_search_preference")

                if (
                    teacher_top is not None
                    and low_prob_on_teacher_top <= low_prob_threshold
                ):
                    filters_passed.append("low_prob_on_high_search_top_move")

                if not filters_passed:
                    filter_reasons["no_divergence"] += 1

                if first_divergence_ply is None and filters_passed:
                    first_divergence_ply = move_index

                policy_entropy = float(
                    -np.sum(
                        np.asarray(candidate_policy)[legal_moves]
                        * np.log(
                            np.maximum(np.asarray(candidate_policy)[legal_moves], 1e-12)
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
                        "low_budget_policy": [float(v) for v in candidate_policy],
                        "low_budget_top_move": low_budget_top,
                        "teacher_visits": teacher_visits,
                        "teacher_policy": teacher_policy,
                        "teacher_value": teacher_value,
                        "teacher_top_move": teacher_top,
                        "teacher_top_visit_share": teacher_top_share,
                        "low_budget_prob_on_teacher_top": low_prob_on_teacher_top,
                        "filters_passed": list(filters_passed),
                        "game_index": game_index,
                        "candidate_seat": candidate_seat,
                        "policy_entropy": policy_entropy,
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
        elif not game.over():
            pass

        if first_divergence_ply is not None:
            first_divergence_plies.append(first_divergence_ply)

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
            early = [p for p in keep if _phase_label(p["move_index"]) == "early"]
            mid = [p for p in keep if _phase_label(p["move_index"]) == "mid"]
            late = [p for p in keep if _phase_label(p["move_index"]) == "late"]
            early_target = max(round(max_positions_per_game * 0.34), 1)
            mid_target = max(round(max_positions_per_game * 0.33), 1)
            late_target = max(max_positions_per_game - early_target - mid_target, 1)

            def _rank_key(pos: dict[str, Any]) -> tuple:
                return (-pos["teacher_top_visit_share"], pos["move_index"])

            early.sort(key=_rank_key)
            mid.sort(key=_rank_key)
            late.sort(key=_rank_key)
            keep_sorted = early[:early_target] + mid[:mid_target] + late[:late_target]
            keep = keep_sorted[:max_positions_per_game]

        if keep:
            mined_games_count += 1
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
                }
                all_mined_rows.append(row)
                phase_counts[_phase_label(pos["move_index"])] += 1
                total_positions_kept += 1

    rng.shuffle(all_mined_rows)
    split_index = max(1, round(len(all_mined_rows) * train_split))
    train_rows = all_mined_rows[:split_index]
    holdout_rows = all_mined_rows[split_index:]

    if target_train_rows is not None and len(train_rows) > target_train_rows:
        ds_rng = random.Random(seed + 1)
        ds_rng.shuffle(train_rows)
        train_rows = train_rows[:target_train_rows]
    if target_holdout_rows is not None and len(holdout_rows) > target_holdout_rows:
        ds_rng = random.Random(seed + 2)
        ds_rng.shuffle(holdout_rows)
        holdout_rows = holdout_rows[:target_holdout_rows]

    duplicate_count = sum(
        max(0, count - 1) for count in fingerprint_counts.values() if count > 1
    )
    capped_count = sum(
        max(0, count - max_fingerprint_copies)
        for count in fingerprint_counts.values()
        if count > max_fingerprint_copies
    )

    total_kept = len(all_mined_rows)
    disagreement_rows = sum(
        1 for r in all_mined_rows if "top_move_disagreement" in r["filters_passed"]
    )
    teacher_top_shares = [r["teacher_top_visit_share"] for r in all_mined_rows]
    low_probs = [r["low_budget_prob_on_teacher_top"] for r in all_mined_rows]
    policy_entropies = [r.get("policy_entropy", 0.0) for r in all_mined_rows]

    summary = {
        "mined_games": mined_games_count,
        "total_games_requested": games,
        "p1_positions_visited": p1_positions_visited,
        "total_positions_visited": total_positions_visited,
        "total_positions_kept": total_positions_kept,
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "rows_per_phase": {
            "early": phase_counts.get("early", 0),
            "mid": phase_counts.get("mid", 0),
            "late": phase_counts.get("late", 0),
        },
        "filter_reasons": filter_reasons,
        "disagreement_rate": round(disagreement_rows / max(total_kept, 1), 4),
        "mean_teacher_top1_visit_share": round(
            float(np.mean(teacher_top_shares)) if teacher_top_shares else 0.0, 4
        ),
        "mean_low_budget_prob_on_teacher_top": round(
            float(np.mean(low_probs)) if low_probs else 0.0, 4
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
        "teacher_simulations": teacher_simulations,
        "input_encoding": input_encoding,
        "policy_target_mode": policy_target_mode,
        "value_target_mode": value_target_mode,
        "max_positions_per_game": max_positions_per_game,
        "train_split": train_split,
        "seed": seed,
        "top_visit_share_threshold": top_visit_share_threshold,
        "low_prob_threshold": low_prob_threshold,
        "low_challenger_sims": low_challenger_sims,
        "low_current_sims": low_current_sims,
        "random_opening_plies": random_opening_plies,
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

    low_parts = args.low_budget.split(":")
    low_challenger_sims = int(low_parts[0])
    low_current_sims = int(low_parts[1])

    train_rows, holdout_rows, summary = run_distillation_mining(
        candidate_path=args.candidate,
        current_path=args.current,
        games=args.games,
        low_challenger_sims=low_challenger_sims,
        low_current_sims=low_current_sims,
        teacher_simulations=args.teacher_simulations,
        seed=args.seed,
        max_positions_per_game=args.max_positions_per_game,
        input_encoding=args.input_encoding,
        policy_target_mode=args.policy_target_mode,
        value_target_mode=args.value_target_mode,
        policy_temperature=args.policy_temperature,
        train_split=args.train_split,
        target_train_rows=args.target_train_rows,
        target_holdout_rows=args.target_holdout_rows,
        top_visit_share_threshold=args.top_visit_share_threshold,
        low_prob_threshold=args.low_prob_threshold,
        c_puct=args.c_puct,
        tactical_root_bias=args.tactical_root_bias,
        random_opening_plies=args.random_opening_plies,
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

    print(f"distill_train_rows={len(train_rows)}")
    print(f"distill_holdout_rows={len(holdout_rows)}")
    print(f"distill_train_sha256={train_sha}")
    print(f"distill_holdout_sha256={holdout_sha}")
    print(f"distill_games={summary['mined_games']}")
    print(f"distill_p1_positions_visited={summary['p1_positions_visited']}")
    print(f"distill_positions_kept={summary['total_positions_kept']}")
    print(f"distill_disagreement_rate={summary['disagreement_rate']}")
    print(f"distill_elapsed_seconds={elapsed:.1f}")
    print(f"summary_written={summary_path}")


if __name__ == "__main__":
    main()
