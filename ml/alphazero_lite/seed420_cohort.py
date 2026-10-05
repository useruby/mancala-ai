"""Pure replay and deterministic retrospective root selection for seed420."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ml.alphazero_lite.kalah_rules import KalahGame


PHASES = ((">32", lambda n: n > 32), ("17-32", lambda n: 17 <= n <= 32))


def canonical_hash(state: dict[str, Any]) -> str:
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def derive_cohort(
    suite: list[dict[str, Any]], game_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Pick 16 roots per phase; >32 phase has first claim on each opening."""
    openings = {index: row for index, row in enumerate(suite)}
    candidates: dict[str, dict[str, list[dict[str, Any]]]] = {
        phase: {"rows": []} for phase, _predicate in PHASES
    }
    for record in game_rows:
        if record.get("lane", "A") != "A":
            continue
        game_row = record.get("game", record)
        if int(game_row["challenger_player"]) != 0:
            continue
        opening_index = int(record.get("opening_id", game_row["opening_index"]))
        opening = openings[opening_index]
        game = KalahGame.from_state(opening["state"])
        trajectory = [
            int(action) for action in game_row["trajectory"].split(",") if action
        ]
        for ply, absolute_action in enumerate(trajectory):
            if game.over():
                raise ValueError("source_moves_after_terminal")
            state = game.to_state()
            active = sum(game.pits)
            legal = game.possible_moves()
            for phase, predicate in PHASES:
                state_hash = canonical_hash(state)
                if predicate(active) and len(legal) >= 2:
                    candidates[phase]["rows"].append(
                        {
                            "phase": phase,
                            "opening_index": opening_index,
                            "opening_state_hash": opening["state_hash"],
                            "decision_ply": ply,
                            "state": state,
                            "state_hash": state_hash,
                            "active_pit_stones": active,
                            "legal_moves": legal,
                            "source_game_index": int(game_row["game_index"]),
                            "source_record_sha256": hashlib.sha256(
                                json.dumps(
                                    record, sort_keys=True, separators=(",", ":")
                                ).encode()
                            ).hexdigest(),
                            "trajectory_prefix_absolute": trajectory[:ply],
                        }
                    )
            if absolute_action // 6 != game.current_player or not game.move(
                absolute_action
            ):
                raise ValueError("source_trajectory_illegal")
        if not game.over():
            raise ValueError("source_trajectory_not_terminal")

    used_openings: set[int] = set()
    selected: list[dict[str, Any]] = []
    for phase, _predicate in PHASES:
        unique: dict[str, dict[str, Any]] = {}
        for row in candidates[phase]["rows"]:
            unique.setdefault(row["state_hash"], row)
        ordered = sorted(
            unique.values(),
            key=lambda row: (
                hashlib.sha256(f"420:{row['state_hash']}".encode()).hexdigest(),
                row["state_hash"],
            ),
        )
        phase_rows = []
        for row in ordered:
            if row["opening_index"] in used_openings:
                continue
            phase_rows.append(row)
            used_openings.add(row["opening_index"])
            if len(phase_rows) == 16:
                break
        if len(phase_rows) != 16:
            raise ValueError(f"phase_cohort_incomplete:{phase}:{len(phase_rows)}")
        selected.extend(phase_rows)
    return selected
