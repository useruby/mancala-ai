#!/usr/bin/env python3
"""Mine failure/disagreement states from current model games and relabel with classic MCTS.

Generates a targeted replay dataset: plays games where the current production model
faces classic MCTS, collects every position, relabels each with high-simulation classic
MCTS, then filters to states where the current model likely needs the most help.

Outputs train and holdout JSONL files suitable for use as additional replay data
in multi-file AlphaZero-lite training.
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
    DEFAULT_POLICY_TARGET_MODE,
    SUPPORTED_POLICY_TARGET_MODES,
    SUPPORTED_VALUE_TARGET_MODES,
    build_policy_target,
    canonical_value_target,
    encode_state,
    top_policy_move_for_legal_moves,
    visits_from_classic_mcts_root,
    value_from_classic_mcts_root,
)

PITS_DIVISOR = 48.0
MAX_MOVES = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine failure/disagreement states and relabel with classic MCTS."
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
        "--current",
        default="storage/ai/alphazero_lite/current",
        help="Path to the current model artifact directory",
    )
    parser.add_argument(
        "--games", type=int, default=400, help="Number of games to generate"
    )
    parser.add_argument(
        "--teacher-mode",
        default="classic_mcts",
        choices=["classic_mcts"],
        help="Teacher mode for relabeling",
    )
    parser.add_argument(
        "--teacher-simulations",
        type=int,
        default=1200,
        help="Classic MCTS simulations for relabeling each state",
    )
    parser.add_argument("--seed", type=int, default=42)
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
        default=DEFAULT_POLICY_TARGET_MODE,
        choices=sorted(SUPPORTED_POLICY_TARGET_MODES),
    )
    parser.add_argument(
        "--value-target-mode",
        default="default",
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
        "--disagreement-prob-threshold",
        type=float,
        default=0.15,
        help="If current model assigns less than this probability to the classic-MCTS"
        " top move, flag as disagreement",
    )
    parser.add_argument(
        "--top-visit-margin-threshold",
        type=float,
        default=0.6,
        help="Minimum top-1 visit share for a 'strong preference' filter",
    )
    parser.add_argument(
        "--tactical-mode",
        action="store_true",
        help="Prefer tactical positions (capture, extra-turn) in mining",
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


def _state_fingerprint(state: list[float]) -> str:
    raw = json.dumps([round(v, 6) for v in state], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _game_state_for_encoding(raw_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "player_pits": list(raw_state["player_pits"]),
        "opponent_pits": list(raw_state["opponent_pits"]),
        "player_store": int(raw_state["player_store"]),
        "opponent_store": int(raw_state["opponent_store"]),
        "current_player": int(raw_state["current_player"]),
    }


def _tactical_features(game: KalahGame) -> dict[str, bool]:
    legal = game.possible_moves()
    has_extra_turn = False
    has_capture = False
    for move in legal:
        sim = game.clone()
        player = sim.current_player
        if sim.move(sim.pit_index(move)):
            if sim.current_player == player and not sim.over():
                has_extra_turn = True
            store_gain = sim.captured_seeds[player] - game.captured_seeds[player]
            if store_gain > 0:
                has_capture = True
    return {"extra_turn": has_extra_turn, "capture": has_capture}


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


def _current_prob_for_move(current_policy: np.ndarray, move: int) -> float:
    return float(current_policy[move])


def run_failure_mining(
    *,
    current_path: str,
    games: int,
    teacher_simulations: int,
    seed: int,
    max_positions_per_game: int,
    input_encoding: str,
    policy_target_mode: str,
    value_target_mode: str,
    policy_temperature: float,
    train_split: float,
    disagreement_prob_threshold: float,
    top_visit_margin_threshold: float,
    tactical_mode: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    evaluator = ArtifactEvaluator(Path(current_path))

    all_mined_rows: list[dict[str, Any]] = []
    mined_games_count = 0
    total_positions_visited = 0
    total_positions_kept = 0
    phase_counts: dict[str, int] = {"early": 0, "mid": 0, "late": 0}
    filter_counts: dict[str, int] = {
        "top_move_disagreement": 0,
        "low_current_prob": 0,
        "strong_mcts_preference": 0,
        "game_lost_by_current": 0,
        "tactical": 0,
    }
    fingerprint_counts: dict[str, int] = {}
    max_fingerprint_copies = max(3, max_positions_per_game // 3)

    for game_index in range(games):
        game = initial_game()
        game_positions: list[dict[str, Any]] = []
        current_seat = game_index % 2  # strict seat alternation
        current_wins_game = None

        for move_index in range(MAX_MOVES):
            if game.over():
                current_wins_game = (
                    None if game.winner is None else (game.winner == current_seat)
                )
                break
            legal_moves = game.possible_moves()
            if not legal_moves:
                break

            raw_state = game.to_state()
            encoded = encode_state(raw_state, input_encoding=input_encoding)

            current_policy, _current_value = evaluator.evaluate(game)

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

            current_top = top_policy_move_for_legal_moves(
                list(current_policy), legal_moves
            )
            teacher_top = _top_move(teacher_visits, legal_moves)
            teacher_top_share = _top_visit_share(teacher_visits, legal_moves)
            current_prob_on_teacher_top = (
                _current_prob_for_move(current_policy, teacher_top)
                if teacher_top is not None
                else 0.0
            )

            tactical = _tactical_features(game)

            filters_passed: list[str] = []

            if current_top is not None and teacher_top is not None:
                if current_top != teacher_top:
                    filters_passed.append("top_move_disagreement")

            if (
                teacher_top is not None
                and current_prob_on_teacher_top < disagreement_prob_threshold
            ):
                filters_passed.append("low_current_prob")

            if teacher_top_share >= top_visit_margin_threshold:
                filters_passed.append("strong_mcts_preference")

            if tactical["extra_turn"] or tactical["capture"]:
                if tactical_mode:
                    filters_passed.append("tactical")

            position_meta = {
                "encoded_state": encoded,
                "game_state": _game_state_for_encoding(raw_state),
                "player": int(game.current_player),
                "move_index": move_index,
                "legal_moves": legal_moves,
                "current_policy": [float(v) for v in current_policy],
                "current_top_move": current_top,
                "teacher_visits": teacher_visits,
                "teacher_policy": teacher_policy,
                "teacher_value": teacher_value,
                "teacher_top_move": teacher_top,
                "teacher_top_visit_share": teacher_top_share,
                "current_prob_on_teacher_top": current_prob_on_teacher_top,
                "tactical": tactical,
                "filters_passed": list(filters_passed),
                "game_index": game_index,
                "current_seat": current_seat,
                "is_current_to_move": game.current_player == current_seat,
            }
            game_positions.append(position_meta)
            total_positions_visited += 1

            if game.current_player == current_seat:
                chosen_move = current_top
            else:
                chosen_move = teacher_top
            if chosen_move is None or chosen_move not in legal_moves:
                chosen_move = rng.choice(legal_moves)
            if not game.move(game.pit_index(chosen_move)):
                break

        if not game.over():
            current_wins_game = None
        elif current_wins_game is None:
            pass

        keep: list[dict[str, Any]] = []
        for pos in game_positions:
            keep_flag = False
            for filter_name in pos["filters_passed"]:
                keep_flag = True
                filter_counts[filter_name] = filter_counts.get(filter_name, 0) + 1

            if current_wins_game is False and pos["is_current_to_move"]:
                keep_flag = True
                filter_counts["game_lost_by_current"] = (
                    filter_counts.get("game_lost_by_current", 0) + 1
                )

            if not keep_flag:
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

            def _rank_key(pos: dict[str, Any]) -> tuple[int, float, int]:
                return (
                    0
                    if tactical_mode
                    and (pos["tactical"]["extra_turn"] or pos["tactical"]["capture"])
                    else 1,
                    -pos["teacher_top_visit_share"],
                    pos["move_index"],
                )

            early.sort(key=_rank_key)
            mid.sort(key=_rank_key)
            late.sort(key=_rank_key)
            keep = early[:early_target] + mid[:mid_target] + late[:late_target]
            keep = keep[:max_positions_per_game]

        if keep:
            mined_games_count += 1
            for pos in keep:
                player = pos["player"]
                outcome_value = (
                    0.0
                    if current_wins_game is None
                    else (1.0 if current_wins_game else -1.0)
                )
                move_index = pos["move_index"]
                value = canonical_value_target(
                    outcome_value=outcome_value,
                    search_value=pos["teacher_value"],
                    move_index=move_index,
                    mode=value_target_mode,
                )
                row = {
                    "move_index": move_index,
                    "player": player,
                    "state": pos["encoded_state"],
                    "policy": pos["teacher_policy"],
                    "policy_target_mode": policy_target_mode,
                    "value_target_mode": value_target_mode,
                    "value": value,
                    "source_game_id": pos["game_index"],
                    "source_ply": move_index,
                    "current_seat": pos["current_seat"],
                    "current_top_move": pos["current_top_move"],
                    "teacher_top_move": pos["teacher_top_move"],
                    "teacher_top_visit_share": pos["teacher_top_visit_share"],
                    "current_prob_on_teacher_top": pos["current_prob_on_teacher_top"],
                    "filters_passed": pos["filters_passed"],
                    "tactical_extra_turn": pos["tactical"]["extra_turn"],
                    "tactical_capture": pos["tactical"]["capture"],
                    "teacher_value_at_state": pos["teacher_value"],
                }
                all_mined_rows.append(row)
                phase_counts[_phase_label(move_index)] += 1
                total_positions_kept += 1

    rng.shuffle(all_mined_rows)
    split_index = max(1, round(len(all_mined_rows) * train_split))
    train_rows = all_mined_rows[:split_index]
    holdout_rows = all_mined_rows[split_index:]

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
    current_probs = [r["current_prob_on_teacher_top"] for r in all_mined_rows]

    summary = {
        "mined_games": mined_games_count,
        "total_positions_visited": total_positions_visited,
        "total_positions_kept": total_positions_kept,
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "rows_per_phase": {
            "early": phase_counts.get("early", 0),
            "mid": phase_counts.get("mid", 0),
            "late": phase_counts.get("late", 0),
        },
        "filter_counts": filter_counts,
        "disagreement_rate": round(disagreement_rows / max(total_kept, 1), 4),
        "mean_classic_mcts_top1_visit_share": round(
            float(np.mean(teacher_top_shares)) if teacher_top_shares else 0.0, 4
        ),
        "mean_current_prob_on_teacher_top": round(
            float(np.mean(current_probs)) if current_probs else 0.0, 4
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

    train_rows, holdout_rows, summary = run_failure_mining(
        current_path=args.current,
        games=args.games,
        teacher_simulations=args.teacher_simulations,
        seed=args.seed,
        max_positions_per_game=args.max_positions_per_game,
        input_encoding=args.input_encoding,
        policy_target_mode=args.policy_target_mode,
        value_target_mode=args.value_target_mode,
        policy_temperature=args.policy_temperature,
        train_split=args.train_split,
        disagreement_prob_threshold=args.disagreement_prob_threshold,
        top_visit_margin_threshold=args.top_visit_margin_threshold,
        tactical_mode=args.tactical_mode,
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

    print(f"failure_mining_train_rows={len(train_rows)}")
    print(f"failure_mining_holdout_rows={len(holdout_rows)}")
    print(f"failure_mining_train_sha256={train_sha}")
    print(f"failure_mining_holdout_sha256={holdout_sha}")
    print(f"failure_mining_games={summary['mined_games']}")
    print(f"failure_mining_positions_visited={summary['total_positions_visited']}")
    print(f"failure_mining_disagreement_rate={summary['disagreement_rate']}")
    print(f"failure_mining_elapsed_seconds={elapsed:.1f}")
    print(f"summary_written={summary_path}")


if __name__ == "__main__":
    main()
