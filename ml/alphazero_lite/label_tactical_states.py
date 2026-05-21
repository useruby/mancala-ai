#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite import classic_mcts, self_play, train
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.input_encodings import SUPPORTED_INPUT_ENCODINGS
from ml.alphazero_lite.kalah_rules import KalahGame


TACTICAL_BUCKETS = frozenset(
    {
        "capture_available",
        "high_imbalance",
        "high_value_swing",
        "early_extra_turn",
    }
)
PRESERVATION_BUCKETS = frozenset(
    {
        "opening_plies_1_8",
        "sparse_endgame",
        "starvation_pressure",
    }
)


class UnclassifiableMinedRowError(ValueError):
    pass


def _side_view(
    state: dict[str, Any], player: int | None = None
) -> tuple[list[int], list[int], int, int]:
    current_player = int(state["current_player"] if player is None else player)
    if current_player == 0:
        return (
            list(state["player_pits"]),
            list(state["opponent_pits"]),
            int(state["player_store"]),
            int(state["opponent_store"]),
        )
    return (
        list(state["opponent_pits"]),
        list(state["player_pits"]),
        int(state["opponent_store"]),
        int(state["player_store"]),
    )


def _move_features(game: KalahGame, move: int) -> dict[str, bool | int]:
    simulated = game.clone()
    original_player = simulated.current_player
    absolute_move = simulated.pit_index(move)
    if not simulated.move(absolute_move):
        raise ValueError(f"illegal move {move} for state {game.to_state()}")
    state_after = simulated.to_state()
    side_pits_after, _, side_store_after, _ = _side_view(
        state_after, player=original_player
    )
    player_store_gain = side_store_after - game.captured_seeds[original_player]
    return {
        "extra_turn": simulated.current_player == original_player
        and not simulated.over(),
        "capture": player_store_gain > 1,
        "swing": player_store_gain >= 4,
        "post_move_empty_pits": sum(1 for value in side_pits_after if value == 0),
    }


def choose_bucket(raw_state: dict[str, Any], *, ply: int) -> str | None:
    game = KalahGame.from_state(raw_state)
    legal_moves = game.possible_moves()
    if not legal_moves:
        raise ValueError("tactical state labeling requires at least one legal move")

    side_pits, opponent_pits, side_store, opponent_store = _side_view(raw_state)
    total_side = sum(side_pits)
    total_opponent = sum(opponent_pits)
    empty_player = sum(1 for value in side_pits if value == 0)
    empty_opponent = sum(1 for value in opponent_pits if value == 0)
    features = [_move_features(game, move) for move in legal_moves]

    if total_side + total_opponent <= 16:
        return "sparse_endgame"
    if any(bool(feature["swing"]) for feature in features):
        return "high_value_swing"
    if (
        total_side <= 6
        or total_opponent <= 6
        or empty_player >= 4
        or empty_opponent >= 4
    ):
        return "starvation_pressure"
    if abs(side_store - opponent_store) >= 8:
        return "high_imbalance"
    if any(bool(feature["capture"]) for feature in features):
        return "capture_available"
    if ply <= 8:
        return "opening_plies_1_8"
    if any(bool(feature["extra_turn"]) for feature in features):
        return "early_extra_turn"
    return None


def bucket_group(bucket: str) -> str:
    if bucket in TACTICAL_BUCKETS:
        return "tactical"
    if bucket in PRESERVATION_BUCKETS:
        return "preservation"
    raise ValueError(f"unsupported bucket for train split: {bucket}")


def _row_ply(row: dict[str, Any]) -> int:
    if "move_index" in row:
        return int(row["move_index"])
    if "ply" in row:
        return int(row["ply"])
    raise ValueError("row must include move_index or ply")


def _policy_for_root(root: Any, *, legal_moves: list[int], mode: str) -> list[float]:
    visits = np.asarray(self_play.visits_from_classic_mcts_root(root), dtype=np.float32)
    missing_moves = [move for move in legal_moves if visits[move] <= 0.0]
    if missing_moves:
        raise ValueError(f"teacher child_stats missing legal moves: {missing_moves}")
    policy = self_play.build_policy_target(
        visits, legal_moves=legal_moves, temperature=1.0, mode=mode
    )
    if not np.isclose(sum(policy), 1.0, atol=1e-6):
        raise ValueError("teacher policy must sum to 1")
    for move, probability in enumerate(policy):
        if move not in legal_moves and probability != 0.0:
            raise ValueError(
                "teacher policy must assign zero probability to illegal moves"
            )
    return policy


def _value_for_root(
    root: Any, *, raw_state: dict[str, Any], mode: str, move_index: int
) -> float:
    search_value = self_play.value_from_classic_mcts_root(root)
    return self_play.build_value_target(
        search_value,
        move_index=move_index,
        mode=mode,
    )


def _copy_list_of_strings(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    copied: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{field}[{index}] must be a non-empty string")
        copied.append(item)
    return copied


def _copy_list_of_dicts(value: Any, *, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    copied: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{index}] must be a dictionary")
        copied.append(dict(item))
    return copied


def _validate_mined_row(row: dict[str, Any]) -> dict[str, Any]:
    raw_state = dict(row["state"])
    canonical_state = canonical_state_key(raw_state)
    if str(row.get("canonical_state")) != canonical_state:
        raise ValueError("canonical_state does not match state")

    side_to_move = row.get("side_to_move")
    if side_to_move != raw_state["current_player"]:
        raise ValueError("side_to_move must match state.current_player")

    actual_legal_moves = KalahGame.from_state(raw_state).possible_moves()
    row_legal_moves = [int(move) for move in row["legal_moves"]]
    if row_legal_moves != actual_legal_moves:
        raise ValueError("legal_moves must match state legal moves")

    priority_score = float(row["priority_score"])
    if not np.isfinite(priority_score):
        raise ValueError("priority_score must be finite")

    return {
        "canonical_state": canonical_state,
        "raw_state": raw_state,
        "side_to_move": int(side_to_move),
        "legal_moves": row_legal_moves,
        "selection_reasons": _copy_list_of_strings(
            row["selection_reasons"], field="selection_reasons"
        ),
        "source_artifacts": _copy_list_of_strings(
            row["source_artifacts"], field="source_artifacts"
        ),
        "source_runs": _copy_list_of_dicts(row["source_runs"], field="source_runs"),
        "priority_score": priority_score,
        "ply": _row_ply(row),
    }


def label_row(
    row: dict[str, Any],
    *,
    policy_simulations: int,
    value_simulations: int,
    seed: int,
    policy_target_mode: str,
    value_target_mode: str,
    input_encoding: str,
) -> dict[str, Any]:
    normalized_row = _validate_mined_row(row)
    raw_state = normalized_row["raw_state"]
    ply = normalized_row["ply"]
    bucket = choose_bucket(raw_state, ply=ply)
    if bucket is None:
        raise UnclassifiableMinedRowError(
            f"unable to classify row at ply={ply}: {raw_state}"
        )

    game = KalahGame.from_state(raw_state)
    legal_moves = normalized_row["legal_moves"]
    teacher_seed = int(seed)
    teacher_policy_seed = teacher_seed
    teacher_value_seed = teacher_seed + 10_000
    policy_search = classic_mcts.MCTS(
        game.clone(), simulations=int(policy_simulations), seed=teacher_policy_seed
    )
    value_search = classic_mcts.MCTS(
        game.clone(), simulations=int(value_simulations), seed=teacher_value_seed
    )
    policy_root = policy_search.search_root()
    value_root = value_search.search_root()
    policy = _policy_for_root(
        policy_root, legal_moves=legal_moves, mode=policy_target_mode
    )
    teacher_selected_move = min(legal_moves, key=lambda move: (-policy[move], move))

    return {
        "canonical_state": normalized_row["canonical_state"],
        "state": self_play.encode_state(raw_state, input_encoding=input_encoding),
        "raw_state": raw_state,
        "side_to_move": normalized_row["side_to_move"],
        "legal_moves": legal_moves,
        "policy": policy,
        "value": _value_for_root(
            value_root, raw_state=raw_state, mode=value_target_mode, move_index=ply
        ),
        "bucket": bucket,
        "bucket_group": bucket_group(bucket),
        "input_encoding": input_encoding,
        "policy_target_mode": policy_target_mode,
        "value_target_mode": value_target_mode,
        "selection_reasons": normalized_row["selection_reasons"],
        "source_artifacts": normalized_row["source_artifacts"],
        "source_runs": normalized_row["source_runs"],
        "priority_score": normalized_row["priority_score"],
        "teacher_policy_simulations": int(policy_simulations),
        "teacher_value_simulations": int(value_simulations),
        "teacher_seed": teacher_seed,
        "teacher_policy_seed": teacher_policy_seed,
        "teacher_value_seed": teacher_value_seed,
        "teacher_selected_move": teacher_selected_move,
        "teacher_child_stats": [
            {
                "move": int(move),
                "visits": int(child.visits),
                "win_rate": 0.0
                if child.visits == 0
                else float(child.wins) / float(child.visits),
            }
            for move, child in sorted(policy_root.children.items())
        ],
    }


def label_rows(
    rows: list[dict[str, Any]],
    *,
    policy_simulations: int,
    value_simulations: int,
    seed: int,
    policy_target_mode: str,
    value_target_mode: str,
    input_encoding: str,
) -> list[dict[str, Any]]:
    normalized_policy_target_mode = train.normalize_policy_target_mode(
        policy_target_mode
    )
    normalized_value_target_mode = train.normalize_value_target_mode(value_target_mode)
    if input_encoding not in SUPPORTED_INPUT_ENCODINGS:
        raise ValueError(f"unsupported input_encoding: {input_encoding}")

    labeled_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            labeled_rows.append(
                label_row(
                    row,
                    policy_simulations=policy_simulations,
                    value_simulations=value_simulations,
                    seed=seed + index,
                    policy_target_mode=normalized_policy_target_mode,
                    value_target_mode=normalized_value_target_mode,
                    input_encoding=input_encoding,
                )
            )
        except UnclassifiableMinedRowError:
            continue
    return labeled_rows


def _train_row_from_labeled_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "raw_state"}


def split_train_rows(
    labeled_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tactical_rows: list[dict[str, Any]] = []
    preservation_rows: list[dict[str, Any]] = []
    for row in labeled_rows:
        train_row = _train_row_from_labeled_row(row)
        if row["bucket"] in TACTICAL_BUCKETS:
            tactical_rows.append(train_row)
        elif row["bucket"] in PRESERVATION_BUCKETS:
            preservation_rows.append(train_row)
        else:
            raise ValueError(f"unsupported bucket for train split: {row['bucket']}")
    return tactical_rows, preservation_rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--policy-simulations", required=True, type=int)
    parser.add_argument("--value-simulations", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--policy-target-mode", required=True)
    parser.add_argument("--value-target-mode", required=True)
    parser.add_argument(
        "--input-encoding", choices=SUPPORTED_INPUT_ENCODINGS, required=True
    )
    parser.add_argument("--out-labeled", required=True)
    parser.add_argument("--out-tactical-train", required=True)
    parser.add_argument("--out-preservation-train", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        labeled_rows = label_rows(
            read_jsonl(Path(args.input)),
            policy_simulations=args.policy_simulations,
            value_simulations=args.value_simulations,
            seed=args.seed,
            policy_target_mode=args.policy_target_mode,
            value_target_mode=args.value_target_mode,
            input_encoding=args.input_encoding,
        )
        tactical_rows, preservation_rows = split_train_rows(labeled_rows)

        write_jsonl(Path(args.out_labeled), labeled_rows)
        write_jsonl(Path(args.out_tactical_train), tactical_rows)
        write_jsonl(Path(args.out_preservation_train), preservation_rows)
    except (OSError, json.JSONDecodeError, TypeError, KeyError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
