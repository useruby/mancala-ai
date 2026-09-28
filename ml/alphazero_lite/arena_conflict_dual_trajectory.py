"""Offline dual-trajectory traversal for frozen replay-attribution arena games."""

from __future__ import annotations

import argparse
import json

from ml.alphazero_lite.arena_conflict_state_localization import (
    ARMS,
    ROOT,
    WORKDIR,
    evaluator,
    probe,
    search_seed,
)
from ml.alphazero_lite.kalah_rules import KalahGame


def initial_game(prefix: list[int]) -> KalahGame:
    game = KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    for move in prefix:
        game.move(game.pit_index(move))
    return game


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS))
    args = parser.parse_args()
    treatments = (args.arm,) if args.arm else ARMS
    evaluators = {name: evaluator(name) for name in ("F", *treatments)}
    rows = []
    for treatment in treatments:
        games = [
            json.loads(line)
            for line in (WORKDIR / f"{treatment}-vs-F.games.jsonl")
            .read_text()
            .splitlines()
        ]
        for game_record in games:
            trajectories = {
                "F": initial_game(game_record["opening_prefix_moves"]),
                treatment: initial_game(game_record["opening_prefix_moves"]),
            }
            for trajectory_name, game in trajectories.items():
                for ply in range(200):
                    if game.over():
                        break
                    state = game.to_state()
                    challenger_role = (
                        "challenger"
                        if game.current_player == game_record["challenger_player"]
                        else "current"
                    )
                    treatment_role = (
                        challenger_role
                        if trajectory_name == treatment
                        else (
                            "current"
                            if challenger_role == "challenger"
                            else "challenger"
                        )
                    )
                    f = probe(
                        evaluators["F"],
                        state,
                        search_seed(game_record, state, ply, challenger_role),
                    )
                    tx = probe(
                        evaluators[treatment],
                        state,
                        search_seed(game_record, state, ply, treatment_role),
                    )
                    if f["selected_move"] != tx["selected_move"]:
                        rows.append(
                            {
                                "arm": treatment,
                                "opening_id": game_record["opening_index"],
                                "seat": game_record["challenger_player"],
                                "trajectory": trajectory_name,
                                "ply": ply,
                                "state": state,
                                "F_move": f["selected_move"],
                                "treatment_move": tx["selected_move"],
                                "exact_root_handoff": bool(
                                    f.get("exact_root_decision")
                                    or tx.get("exact_root_decision")
                                ),
                            }
                        )
                    selected = (
                        f["selected_move"]
                        if trajectory_name == "F"
                        else tx["selected_move"]
                    )
                    game.move(game.pit_index(selected))
    suffix = "" if args.arm is None else f"-{args.arm}"
    out = (
        ROOT
        / f"docs/data/alphazero-lite-replay-source-attribution/dual-trajectory-divergences{suffix}.json"
    )
    out.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
