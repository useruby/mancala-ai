#!/usr/bin/env python3
"""Mine opening-suite bucket-balanced curriculum from iter0_reference.

Loads the deduplicated opening suite, classifies openings into weakness
buckets by playing fast deterministic games, then re-plays selected
openings to mine training states with high-search teacher relabeling.

Buckets:
  weak_p1: iter0_reference loses as P1
  weak_p0: iter0_reference loses as P0
  high_search_rescue: loses at policy-network play, but teacher (1200 sim)
    suggests a better path
  preservation: iter0_reference already beats current
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
MAX_FINGERPRINT_COPIES = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine opening-suite bucket-balanced curriculum."
    )
    parser.add_argument("--suite", required=True, help="Opening suite JSONL path.")
    parser.add_argument(
        "--out-train", required=True, help="Training JSONL output path."
    )
    parser.add_argument(
        "--out-holdout", required=True, help="Holdout JSONL output path."
    )
    parser.add_argument(
        "--out-summary", required=True, help="Summary JSON output path."
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to iter0_reference artifact directory.",
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to current model artifact directory.",
    )
    parser.add_argument(
        "--budget-pairs",
        default="384:256,768:256,1200:1200",
        help="Comma-separated budget pairs for classification labels.",
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
        help="Classic MCTS simulations for relabeling each state.",
    )
    parser.add_argument(
        "--target-train-rows",
        type=int,
        default=2048,
        help="Target train row count.",
    )
    parser.add_argument(
        "--target-holdout-rows",
        type=int,
        default=512,
        help="Target holdout row count.",
    )
    parser.add_argument(
        "--min-unique-boards",
        type=int,
        default=1000,
        help="Minimum unique boards target.",
    )
    parser.add_argument(
        "--max-rows-per-opening",
        type=int,
        default=8,
        help="Maximum rows to keep per opening prefix.",
    )
    parser.add_argument(
        "--max-rows-per-game",
        type=int,
        default=16,
        help="Maximum rows to keep per game trajectory.",
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
        help="Temperature for converting visits to policy target.",
    )
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument(
        "--games-per-opening-classify",
        type=int,
        default=4,
        help="Games per opening for classification (evenly split by seat).",
    )
    parser.add_argument(
        "--bucket-mix-weak-p1",
        type=float,
        default=0.35,
        help="Fraction of rows from weak P1 bucket.",
    )
    parser.add_argument(
        "--bucket-mix-weak-p0",
        type=float,
        default=0.25,
        help="Fraction of rows from weak P0 bucket.",
    )
    parser.add_argument(
        "--bucket-mix-rescue",
        type=float,
        default=0.25,
        help="Fraction of rows from high-search rescue bucket.",
    )
    parser.add_argument(
        "--bucket-mix-preservation",
        type=float,
        default=0.15,
        help="Fraction of rows from preservation bucket.",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.25,
    )
    return parser.parse_args()


def _search_seed(base_seed: int, game_index: int, move_index: int) -> int:
    return (base_seed * 1_000_003) + (game_index * 10_007) + move_index


def _initial_game() -> KalahGame:
    return KalahGame.from_state(
        {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )


def _apply_opening_moves(game: KalahGame, moves: list[int]) -> int:
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


def _game_from_opening_moves(moves: list[int]) -> KalahGame:
    game = _initial_game()
    _apply_opening_moves(game, moves)
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


def _load_suite(path: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _classify_outcomes(
    *,
    opening: dict[str, Any],
    candidate_evaluator: ArtifactEvaluator,
    current_evaluator: ArtifactEvaluator,
    games_per_opening: int,
    seed: int,
    global_game_index: int,
) -> tuple[list[dict[str, Any]], int]:
    """Play classification games for one opening.

    Returns (per_game_outcomes, next_global_game_index).
    Each outcome dict has: seat, result (win/loss/draw), game_index.
    """
    prefix_moves: list[int] = opening["prefix_moves"]
    outcomes: list[dict[str, Any]] = []
    rng = random.Random(seed + global_game_index)

    for game_in_opening in range(games_per_opening):
        game = _game_from_opening_moves(prefix_moves)
        candidate_seat = game_in_opening % 2  # alternate seats
        candidate_wins_game = None

        for _move_index in range(MAX_MOVES):
            if game.over():
                winner = game.winner
                if winner is None:
                    candidate_wins_game = None
                else:
                    candidate_wins_game = winner == candidate_seat
                break

            legal_moves = game.possible_moves()
            if not legal_moves:
                break

            if game.current_player == candidate_seat:
                candidate_policy, _ = candidate_evaluator.evaluate(game)
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

        result = (
            "draw"
            if candidate_wins_game is None
            else ("win" if candidate_wins_game else "loss")
        )
        outcomes.append(
            {
                "seat": candidate_seat,
                "result": result,
                "game_index": global_game_index,
            }
        )
        global_game_index += 1

    return outcomes, global_game_index


def _classify_all_openings(
    *,
    suite: list[dict[str, Any]],
    candidate_path: str,
    current_path: str,
    games_per_opening: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Classify every opening into a weakness bucket.

    Phase 1: policy-network play (fast) for initial weak_p1/weak_p0 split.
    Phase 2: quick teacher MCTS check on post-opening positions for rescue detection.
    """
    candidate_evaluator = ArtifactEvaluator(Path(candidate_path))
    current_evaluator = ArtifactEvaluator(Path(current_path))
    global_game_index = 0
    classified: list[dict[str, Any]] = []

    for entry in suite:
        prefix_moves: list[int] = entry["prefix_moves"]

        # --- Phase 1: policy-network classification ---
        outcomes, global_game_index = _classify_outcomes(
            opening=entry,
            candidate_evaluator=candidate_evaluator,
            current_evaluator=current_evaluator,
            games_per_opening=games_per_opening,
            seed=seed,
            global_game_index=global_game_index,
        )

        p0_outcomes = [o for o in outcomes if o["seat"] == 0]
        p1_outcomes = [o for o in outcomes if o["seat"] == 1]

        p0_wins = sum(1 for o in p0_outcomes if o["result"] == "win")
        p0_draws = sum(1 for o in p0_outcomes if o["result"] == "draw")
        p0_losses = sum(1 for o in p0_outcomes if o["result"] == "loss")
        p0_total = len(p0_outcomes)

        p1_wins = sum(1 for o in p1_outcomes if o["result"] == "win")
        p1_draws = sum(1 for o in p1_outcomes if o["result"] == "draw")
        p1_losses = sum(1 for o in p1_outcomes if o["result"] == "loss")
        p1_total = len(p1_outcomes)

        # --- Phase 2: quick teacher check for rescue detection ---
        # Run classic MCTS at teacher sims on the post-opening position
        # to detect if high search would change the outcome
        game = _game_from_opening_moves(prefix_moves)
        teacher_value_p0 = 0.0
        teacher_value_p1 = 0.0
        if not game.over() and game.possible_moves():
            try:
                mcts = ClassicMCTS(
                    game.clone(),
                    simulations=1200,
                    seed=_search_seed(seed, global_game_index, 0),
                )
                root = mcts.search_root()
                raw_value = value_from_classic_mcts_root(root)
                # teacher_value is from current player's perspective
                # For P0 classification: if current player is 0, raw_value is P0 value
                # For P1 classification: if current player is 1, raw_value is P1 value
                current_player = game.current_player
                if current_player == 0:
                    teacher_value_p0 = raw_value
                else:
                    teacher_value_p1 = raw_value
            except Exception:
                pass

        # Determine bucket
        bucket = _determine_bucket(
            p0_wins=p0_wins,
            p0_draws=p0_draws,
            p0_losses=p0_losses,
            p1_wins=p1_wins,
            p1_draws=p1_draws,
            p1_losses=p1_losses,
            teacher_value_p0=teacher_value_p0,
            teacher_value_p1=teacher_value_p1,
        )

        classified.append(
            {
                **entry,
                "bucket": bucket,
                "p0_wins": p0_wins,
                "p0_draws": p0_draws,
                "p0_losses": p0_losses,
                "p1_wins": p1_wins,
                "p1_draws": p1_draws,
                "p1_losses": p1_losses,
                "p0_total": p0_total,
                "p1_total": p1_total,
                "teacher_value_p0": round(teacher_value_p0, 4),
                "teacher_value_p1": round(teacher_value_p1, 4),
            }
        )

    return classified


def _determine_bucket(
    *,
    p0_wins: int,
    p0_draws: int,
    p0_losses: int,
    p1_wins: int,
    p1_draws: int,
    p1_losses: int,
    teacher_value_p0: float = 0.0,
    teacher_value_p1: float = 0.0,
) -> str:
    """Classify opening into: weak_p1, weak_p0, high_search_rescue, preservation.

    Uses policy-network outcome for base classification, teacher MCTS value
    for rescue detection.
    """
    p0_lost = p0_losses > (p0_wins + p0_draws)
    p1_lost = p1_losses > (p1_wins + p1_draws)
    p0_good = (p0_wins + p0_draws) >= p0_losses and (p0_wins + p0_draws) > 0
    p1_good = (p1_wins + p1_draws) >= p1_losses and (p1_wins + p1_draws) > 0

    # Preservation: openings where iter0_reference beats current
    if p0_good or p1_good:
        return "preservation"

    if p1_lost and p0_lost:
        # Check if teacher sees rescue potential for P1
        if teacher_value_p1 > 0.05:
            return "high_search_rescue"
        if teacher_value_p0 > 0.05:
            return "high_search_rescue"
        return "weak_p1"  # prioritize P1 weakness when both lost

    if p1_lost:
        if teacher_value_p1 > 0.05:
            return "high_search_rescue"
        return "weak_p1"

    if p0_lost:
        if teacher_value_p0 > 0.05:
            return "high_search_rescue"
        return "weak_p0"

    # Default: treat as weak P1
    return "weak_p1"


def _select_openings(
    classified: list[dict[str, Any]],
    *,
    bucket_mix: dict[str, float],
    target_train_rows: int,
    max_rows_per_opening: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select openings from each bucket proportional to target mix."""
    rng = random.Random(seed + 200)
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for entry in classified:
        bucket = entry["bucket"]
        by_bucket.setdefault(bucket, []).append(entry)

    # Shuffle within each bucket
    for entries in by_bucket.values():
        rng.shuffle(entries)

    # Target rows per bucket
    bucket_row_targets: dict[str, int] = {}
    for bucket, frac in bucket_mix.items():
        bucket_row_targets[bucket] = max(1, round(target_train_rows * frac))

    # Normalize to match target_train_rows approximately
    total_target = sum(bucket_row_targets.values())
    if total_target > 0:
        scale = target_train_rows / total_target
        for bucket in bucket_row_targets:
            bucket_row_targets[bucket] = max(
                1, round(bucket_row_targets[bucket] * scale)
            )

    # Select openings from each bucket (enough to fill row targets)
    selected: list[dict[str, Any]] = []
    bucket_openings_selected: dict[str, int] = {}
    for bucket, target_rows in bucket_row_targets.items():
        needed_openings = max(
            1, (target_rows + max_rows_per_opening - 1) // max_rows_per_opening
        )
        available = by_bucket.get(bucket, [])
        taken = min(needed_openings, len(available))
        selected.extend(available[:taken])
        bucket_openings_selected[bucket] = taken

    rng.shuffle(selected)
    return selected, bucket_openings_selected


def _mine_states_from_selected(
    *,
    selected_openings: list[dict[str, Any]],
    candidate_path: str,
    current_path: str,
    teacher_simulations: int,
    seed: int,
    max_rows_per_opening: int,
    max_rows_per_game: int,
    input_encoding: str,
    policy_target_mode: str,
    value_target_mode: str,
    policy_temperature: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Mine training states from selected openings."""
    candidate_evaluator = ArtifactEvaluator(Path(candidate_path))
    current_evaluator = ArtifactEvaluator(Path(current_path))

    all_mined_rows: list[dict[str, Any]] = []
    fingerprint_counts: dict[str, int] = {}
    unique_boards: set[str] = set()
    rng = random.Random(seed + 300)

    global_game_index = 0
    positions_visited = 0
    positions_kept = 0
    dups_capped = 0
    phase_counts: Counter = Counter()
    low_budget_vs_teacher_disagreements = 0
    rows_by_bucket: dict[str, int] = {}
    rows_by_seat: Counter = Counter()
    rows_by_ply: Counter = Counter()
    teacher_top_shares: list[float] = []
    policy_entropies: list[float] = []
    first_divergence_moves: list[int] = []

    for opening in selected_openings:
        prefix_moves: list[int] = opening["prefix_moves"]
        bucket = opening["bucket"]
        opening_rows: list[dict[str, Any]] = []

        for game_in_opening in range(4):  # 4 games per opening for state diversity
            game = _game_from_opening_moves(prefix_moves)
            candidate_seat = game_in_opening % 2
            candidate_wins_game = None
            first_divergence_move: int | None = None
            game_positions: list[dict[str, Any]] = []

            for move_index in range(MAX_MOVES):
                if game.over():
                    winner = game.winner
                    if winner is None:
                        candidate_wins_game = None
                    else:
                        candidate_wins_game = winner == candidate_seat
                    break

                legal_moves = game.possible_moves()
                if not legal_moves:
                    break

                raw_state = game.to_state()
                encoded = encode_state(raw_state, input_encoding=input_encoding)

                candidate_policy, _ = candidate_evaluator.evaluate(game)

                low_budget_top = top_policy_move_for_legal_moves(
                    list(candidate_policy), legal_moves
                )

                # Mine every position where candidate would move
                is_candidate_turn = game.current_player == candidate_seat
                if is_candidate_turn:
                    positions_visited += 1

                    mcts = ClassicMCTS(
                        game.clone(),
                        simulations=teacher_simulations,
                        seed=_search_seed(seed, global_game_index, move_index),
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

                    if teacher_top_share >= 0.55:
                        filters_passed.append("strong_high_search_preference")

                    if teacher_top is not None and low_prob_on_teacher_top <= 0.30:
                        filters_passed.append("low_prob_on_high_search_top_move")

                    if first_divergence_move is None and filters_passed:
                        first_divergence_move = move_index

                    policy_entropy = float(
                        -np.sum(
                            np.asarray(candidate_policy)[legal_moves]
                            * np.log(
                                np.maximum(
                                    np.asarray(candidate_policy)[legal_moves], 1e-12
                                )
                            )
                        )
                    )

                    game_positions.append(
                        {
                            "encoded_state": encoded,
                            "player": int(game.current_player),
                            "move_index": move_index,
                            "legal_moves": list(legal_moves),
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
                            "opening_ply": len(prefix_moves),
                            "seat": int(game.current_player),
                        }
                    )

                # Advance the game
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

            if first_divergence_move is not None:
                first_divergence_moves.append(first_divergence_move)

            # Filter and deduplicate rows from this game
            keep: list[dict[str, Any]] = []
            for pos in game_positions:
                if not pos["filters_passed"]:
                    continue
                fp = _state_fingerprint(pos["encoded_state"])
                fp_count = fingerprint_counts.get(fp, 0)
                if fp_count >= MAX_FINGERPRINT_COPIES:
                    dups_capped += 1
                    continue
                fingerprint_counts[fp] = fp_count + 1
                keep.append(pos)

            if len(keep) > max_rows_per_game:
                keep.sort(
                    key=lambda p: (
                        -p["teacher_top_visit_share"],
                        p["move_index"],
                    )
                )
                keep = keep[:max_rows_per_game]

            for pos in keep:
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
                    "seat": pos["seat"],
                    "low_budget_top_move": pos["low_budget_top_move"],
                    "teacher_top_move": pos["teacher_top_move"],
                    "teacher_top_visit_share": pos["teacher_top_visit_share"],
                    "low_budget_prob_on_teacher_top": pos[
                        "low_budget_prob_on_teacher_top"
                    ],
                    "filters_passed": pos["filters_passed"],
                    "teacher_value_at_state": pos["teacher_value"],
                    "policy_entropy": pos["policy_entropy"],
                    "opening_ply": pos["opening_ply"],
                    "opening_prefix_moves": prefix_moves,
                    "bucket": bucket,
                }
                opening_rows.append(row)
                phase_counts[_phase_label(pos["move_index"])] += 1
                teacher_top_shares.append(pos["teacher_top_visit_share"])
                policy_entropies.append(pos["policy_entropy"])
                rows_by_seat[pos["seat"]] += 1
                rows_by_ply[pos["opening_ply"]] += 1
                positions_kept += 1

                if pos["low_budget_top_move"] != pos["teacher_top_move"]:
                    low_budget_vs_teacher_disagreements += 1

            global_game_index += 1

        # Cap rows per opening
        if len(opening_rows) > max_rows_per_opening:
            rng.shuffle(opening_rows)
            opening_rows = opening_rows[:max_rows_per_opening]

        if bucket not in rows_by_bucket:
            rows_by_bucket[bucket] = 0
        rows_by_bucket[bucket] += len(opening_rows)
        all_mined_rows.extend(opening_rows)

    mine_stats = {
        "positions_visited": positions_visited,
        "positions_kept": positions_kept,
        "duplicate_capped": dups_capped,
        "unique_boards": len(unique_boards),
        "low_budget_vs_teacher_disagreement_rate": (
            low_budget_vs_teacher_disagreements / max(positions_kept, 1)
        ),
        "mean_teacher_top1_visit_share": (
            float(np.mean(teacher_top_shares)) if teacher_top_shares else 0.0
        ),
        "mean_policy_entropy": (
            float(np.mean(policy_entropies)) if policy_entropies else 0.0
        ),
        "first_divergence_move_mean": (
            float(np.mean(first_divergence_moves)) if first_divergence_moves else 0.0
        ),
        "first_divergence_move_median": (
            float(np.median(first_divergence_moves)) if first_divergence_moves else 0.0
        ),
        "rows_per_bucket": rows_by_bucket,
        "rows_per_seat": dict(rows_by_seat),
        "rows_per_ply": dict(rows_by_ply),
        "train_rows_raw": len(all_mined_rows),
    }

    return all_mined_rows, mine_stats


def _split_train_holdout(
    rows: list[dict[str, Any]],
    *,
    target_train: int,
    target_holdout: int,
    bucket_mix: dict[str, float],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split into train/holdout preserving bucket proportions.

    When some target buckets have no data, redistributes proportionally
    across buckets that do have data."""
    rng = random.Random(seed + 400)

    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        bucket = row["bucket"]
        by_bucket.setdefault(bucket, []).append(row)

    for entries in by_bucket.values():
        rng.shuffle(entries)

    # Build actual mix from available buckets
    available_buckets = sorted(by_bucket.keys())
    actual_mix: dict[str, float] = {}
    if available_buckets:
        raw_total = sum(bucket_mix.get(b, 0.0) for b in available_buckets)
        if raw_total > 0:
            for b in available_buckets:
                actual_mix[b] = bucket_mix.get(b, 0.0) / raw_total
        else:
            # All target fractions are zero — equal split
            for b in available_buckets:
                actual_mix[b] = 1.0 / len(available_buckets)

    # Allocate rows proportionally
    total_available = len(rows)
    holdout_share = target_holdout / max(target_train + target_holdout, 1)
    actual_holdout = min(target_holdout, max(1, round(total_available * holdout_share)))
    actual_train = min(target_train, total_available - actual_holdout)

    train_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    for bucket, frac in actual_mix.items():
        pool = by_bucket.get(bucket, [])
        if not pool:
            continue
        train_quota = max(1, round(actual_train * frac))
        train_quota = min(train_quota, len(pool))
        remain = max(0, len(pool) - train_quota)
        holdout_quota = min(max(1, round(actual_holdout * frac)), remain)

        train_rows.extend(pool[:train_quota])
        if holdout_quota > 0:
            holdout_rows.extend(pool[train_quota : train_quota + holdout_quota])

    # Trim to targets
    rng.shuffle(train_rows)
    train_rows = train_rows[:target_train]
    rng.shuffle(holdout_rows)
    holdout_rows = holdout_rows[:target_holdout]

    rng.shuffle(train_rows)
    rng.shuffle(holdout_rows)
    return train_rows, holdout_rows


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return sha


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    # --- Phase 1: Load suite and classify ---
    print("Loading opening suite...", flush=True)
    suite = _load_suite(args.suite)
    print(f"Loaded {len(suite)} openings from {args.suite}")

    print("Classifying openings by weakness bucket...", flush=True)
    classified = _classify_all_openings(
        suite=suite,
        candidate_path=args.candidate,
        current_path=args.current,
        games_per_opening=args.games_per_opening_classify,
        seed=args.seed,
    )

    bucket_counts: dict[str, int] = {}
    for entry in classified:
        bucket = entry["bucket"]
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    print(f"Bucket distribution: {bucket_counts}")

    # --- Phase 2: Mine from all classified openings ---
    bucket_mix = {
        "weak_p1": args.bucket_mix_weak_p1,
        "weak_p0": args.bucket_mix_weak_p0,
        "high_search_rescue": args.bucket_mix_rescue,
        "preservation": args.bucket_mix_preservation,
    }

    # Use all classified openings for maximum board diversity
    print(
        f"Mining training states from all {len(classified)} classified openings...",
        flush=True,
    )
    all_rows, mine_stats = _mine_states_from_selected(
        selected_openings=classified,
        candidate_path=args.candidate,
        current_path=args.current,
        teacher_simulations=args.teacher_simulations,
        seed=args.seed,
        max_rows_per_opening=args.max_rows_per_opening,
        max_rows_per_game=args.max_rows_per_game,
        input_encoding=args.input_encoding,
        policy_target_mode=args.policy_target_mode,
        value_target_mode=args.value_target_mode,
        policy_temperature=args.policy_temperature,
    )

    print(
        f"Mined {len(all_rows)} raw rows "
        f"({mine_stats['positions_visited']} positions visited, "
        f"{mine_stats['positions_kept']} kept, "
        f"{mine_stats['unique_boards']} unique boards)"
    )

    # --- Phase 4: Split into train/holdout ---
    print("Splitting into train/holdout...", flush=True)
    train_rows, holdout_rows = _split_train_holdout(
        all_rows,
        target_train=args.target_train_rows,
        target_holdout=args.target_holdout_rows,
        bucket_mix=bucket_mix,
        seed=args.seed,
    )

    # --- Write outputs ---
    out_train = Path(args.out_train)
    out_holdout = Path(args.out_holdout)
    train_sha = _write_jsonl(train_rows, out_train)
    holdout_sha = _write_jsonl(holdout_rows, out_holdout)

    elapsed = time.perf_counter() - started

    # Compute duplicate count
    fp_counts: dict[str, int] = {}
    for row in all_rows:
        fp = _state_fingerprint(row["state"])
        fp_counts[fp] = fp_counts.get(fp, 0) + 1
    duplicate_count = sum(max(0, c - 1) for c in fp_counts.values())

    train_buckets: Counter = Counter(r["bucket"] for r in train_rows)
    holdout_buckets: Counter = Counter(r["bucket"] for r in holdout_rows)

    summary = {
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "train_sha256": train_sha,
        "holdout_sha256": holdout_sha,
        "unique_board_count": mine_stats["unique_boards"],
        "duplicate_row_count": duplicate_count,
        "capped_row_count": mine_stats["duplicate_capped"],
        "rows_per_bucket_train": dict(train_buckets),
        "rows_per_bucket_holdout": dict(holdout_buckets),
        "rows_per_opening_ply": mine_stats["rows_per_ply"],
        "rows_per_seat": mine_stats["rows_per_seat"],
        "low_budget_vs_teacher_disagreement_rate": round(
            mine_stats["low_budget_vs_teacher_disagreement_rate"], 4
        ),
        "mean_teacher_top1_visit_share": round(
            mine_stats["mean_teacher_top1_visit_share"], 4
        ),
        "mean_policy_entropy": round(mine_stats["mean_policy_entropy"], 4),
        "first_divergence_move_mean": round(
            mine_stats["first_divergence_move_mean"], 2
        ),
        "first_divergence_move_median": round(
            mine_stats["first_divergence_move_median"], 2
        ),
        "elapsed_seconds": round(elapsed, 1),
        "teacher_simulations": args.teacher_simulations,
        "input_encoding": args.input_encoding,
        "policy_target_mode": args.policy_target_mode,
        "value_target_mode": args.value_target_mode,
        "max_rows_per_opening": args.max_rows_per_opening,
        "target_train_rows": args.target_train_rows,
        "target_holdout_rows": args.target_holdout_rows,
        "min_unique_boards": args.min_unique_boards,
        "bucket_mix_target": bucket_mix,
        "bucket_distribution": bucket_counts,
        "openings_used": len(classified),
        "openings_classified": len(classified),
        "train_path": str(out_train),
        "holdout_path": str(out_holdout),
        "seed": args.seed,
    }

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\ntrain_rows={len(train_rows)}")
    print(f"holdout_rows={len(holdout_rows)}")
    print(f"unique_boards={mine_stats['unique_boards']}")
    print(f"train_sha256={train_sha}")
    print(f"holdout_sha256={holdout_sha}")
    print(f"elapsed={elapsed:.1f}s")
    print(f"summary={out_summary}")


if __name__ == "__main__":
    main()
