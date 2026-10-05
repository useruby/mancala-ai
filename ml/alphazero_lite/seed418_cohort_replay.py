"""Pure Kalah replay and deterministic cohort selection for seed418."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ml.alphazero_lite.kalah_rules import KalahGame


def canonical_hash(state: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def replay_cohort(
    suite: list[dict[str, Any]], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    openings = dict(enumerate(suite))
    candidates: list[dict[str, Any]] = []
    for record in records:
        game_row = record["game"]
        if record["lane"] != "A" or int(game_row["challenger_player"]) != 0:
            continue
        opening_index = int(record["opening_id"])
        opening = openings[opening_index]
        game = KalahGame.from_state(opening["state"])
        trajectory = [int(x) for x in game_row["trajectory"].split(",") if x]
        for ply, absolute in enumerate(trajectory):
            if game.over():
                raise ValueError("source_moves_after_terminal")
            state = game.to_state()
            active = sum(game.pits)
            if 17 <= active <= 21:
                candidates.append(
                    {
                        "opening_index": opening_index,
                        "source_opening_state_hash": opening["state_hash"],
                        "decision_ply": ply,
                        "state": state,
                        "state_hash": canonical_hash(state),
                        "active_pit_stones": active,
                        "source_record_sha256": hashlib.sha256(
                            json.dumps(
                                record, sort_keys=True, separators=(",", ":")
                            ).encode()
                        ).hexdigest(),
                        "source_game_index": int(game_row["game_index"]),
                        "source_game_within_opening": int(
                            game_row["game_within_opening"]
                        ),
                        "opening_prefix_moves": list(opening["prefix_moves"]),
                        "trajectory_prefix_absolute": trajectory[:ply],
                        "decision_absolute_action": absolute,
                    }
                )
                break
            if absolute // 6 != game.current_player or not game.move(absolute):
                raise ValueError("source_trajectory_illegal")
        else:
            if not game.over():
                raise ValueError("source_trajectory_not_terminal")
    unique: dict[str, dict[str, Any]] = {}
    for row in candidates:
        unique.setdefault(row["state_hash"], row)
    ordered = sorted(
        unique.values(),
        key=lambda row: (
            hashlib.sha256(f"418:{row['state_hash']}".encode()).hexdigest(),
            row["state_hash"],
        ),
    )
    chosen: list[dict[str, Any]] = []
    used: set[int] = set()
    for row in ordered:
        if row["opening_index"] in used:
            continue
        chosen.append(row)
        used.add(row["opening_index"])
        if len(chosen) == 128:
            break
    if len(chosen) != 128:
        raise ValueError(f"eligible_cohort_too_small:{len(chosen)}")
    return chosen
