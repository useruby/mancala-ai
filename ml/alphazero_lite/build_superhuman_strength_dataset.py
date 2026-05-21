#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import encode_state

DEFAULT_SELECTIONS = (
    {"game_id": 11, "move_number": 18, "label": "g11_m18"},
    {"game_id": 11, "move_number": 29, "label": "g11_m29"},
    {"game_id": 11, "move_number": 32, "label": "g11_m32"},
    {"game_id": 11, "move_number": 38, "label": "g11_m38"},
    {"game_id": 11, "move_number": 40, "label": "g11_m40"},
    {"game_id": 12, "move_number": 19, "label": "g12_m19"},
    {"game_id": 12, "move_number": 31, "label": "g12_m31"},
    {"game_id": 12, "move_number": 34, "label": "g12_m34"},
    {"game_id": 12, "move_number": 37, "label": "g12_m37"},
    {"game_id": 12, "move_number": 39, "label": "g12_m39"},
    {"game_id": 12, "move_number": 44, "label": "g12_m44"},
    {"game_id": 12, "move_number": 45, "label": "g12_m45"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a curated replay dataset for the superhuman strength experiment."
    )
    parser.add_argument("--games-json", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def normalize_winner(winner: int | None) -> int:
    return -1 if winner is None else int(winner)


def sharpened_policy(legal_moves: list[int], chosen_move: int) -> list[float]:
    policy = [0.0] * 6
    if chosen_move not in legal_moves:
        raise ValueError(f"chosen move {chosen_move} is not legal in {legal_moves}")

    if len(legal_moves) == 1:
        policy[chosen_move] = 1.0
        return policy

    off_target_total = 0.03
    off_target_share = off_target_total / (len(legal_moves) - 1)
    for move in legal_moves:
        policy[move] = off_target_share
    policy[chosen_move] = 1.0 - off_target_total
    return policy


def build_dataset_rows(
    games: list[dict], *, selections=DEFAULT_SELECTIONS
) -> list[dict]:
    games_by_id = {int(game["id"]): game for game in games}
    rows: list[dict] = []

    for selection in selections:
        game_id = int(selection["game_id"])
        if game_id not in games_by_id:
            available_ids = sorted(games_by_id)
            raise ValueError(
                f"selection references missing game_id {game_id} "
                f"(label={selection.get('label')!r}); available ids: {available_ids}"
            )
        game = games_by_id[game_id]
        engine = KalahGame(
            pits=[4] * 12,
            captured_seeds=[0, 0],
            current_player=0,
        )

        move_history = game["move_history"]
        move_number = int(selection["move_number"])
        if move_number < 1 or move_number > len(move_history):
            raise ValueError(
                f"move_number out of range for selection {selection}: "
                f"move_history length is {len(move_history)}"
            )
        for prior_move in move_history[: move_number - 1]:
            relative_move = int(prior_move["pit"])
            if relative_move not in engine.possible_moves():
                raise ValueError(
                    f"illegal historical move {relative_move} before selection {selection}"
                )
            engine.move(engine.pit_index(relative_move))

        selected_move = move_history[move_number - 1]
        chosen_move = int(selected_move["pit"])
        legal_moves = engine.possible_moves()
        current_state = engine.to_state()
        current_player = int(engine.current_player)
        winner = normalize_winner(game.get("winner"))
        value = 0.0 if winner == -1 else (1.0 if winner == current_player else -1.0)

        rows.append(
            {
                "state": encode_state(current_state, input_encoding="kalah_v3"),
                "policy": sharpened_policy(legal_moves, chosen_move),
                "value": value,
                "player": current_player,
                "move_index": chosen_move,
                "winner": winner,
                "policy_target_mode": "sharpened",
                "value_target_mode": "sharpened",
                "source": f"superhuman_strength_{selection['label']}",
            }
        )

    return rows


def main() -> int:
    args = parse_args()
    games = json.loads(args.games_json.read_text(encoding="utf-8"))
    rows = build_dataset_rows(games)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(json.dumps({"out": str(args.out), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
