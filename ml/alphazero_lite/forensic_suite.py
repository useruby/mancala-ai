from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_BUCKETS = frozenset(
    {
        "opening_plies_1_8",
        "early_extra_turn",
        "capture_available",
        "high_imbalance",
        "starvation_pressure",
        "sparse_endgame",
        "high_value_swing",
        "incumbent_proxy_disagreement",
    }
)

PITS_PER_PLAYER = 6


@dataclass(frozen=True)
class ForensicPosition:
    id: str
    state: dict
    side_to_move: int
    legal_moves: tuple[int, ...]
    phase: str
    bucket: str
    tags: tuple[str, ...]
    source: str

    @property
    def canonical_key(self) -> str:
        return canonical_state_key(self.state)


def canonical_state_key(state: dict[str, Any]) -> str:
    payload = {
        "player_pits": [int(value) for value in state["player_pits"]],
        "opponent_pits": [int(value) for value in state["opponent_pits"]],
        "player_store": int(state["player_store"]),
        "opponent_store": int(state["opponent_store"]),
        "current_player": int(state["current_player"]),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _require_keys(row: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key not in row:
            raise ValueError(f"suite row is missing required field: {key}")


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"suite row {field_name} must be a non-empty string")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"suite row {field_name} must be an integer")
    return value


def _validate_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise ValueError("suite row state must be a dictionary")

    for key in (
        "player_pits",
        "opponent_pits",
        "player_store",
        "opponent_store",
        "current_player",
    ):
        if key not in state:
            raise ValueError(f"suite row state is missing required field: {key}")

    player_pits = state["player_pits"]
    opponent_pits = state["opponent_pits"]
    if not isinstance(player_pits, list) or len(player_pits) != PITS_PER_PLAYER:
        raise ValueError("suite row state player_pits must be a list of 6 integers")
    if not isinstance(opponent_pits, list) or len(opponent_pits) != PITS_PER_PLAYER:
        raise ValueError("suite row state opponent_pits must be a list of 6 integers")

    canonical_player_pits = [
        _require_int(value, "state.player_pits") for value in player_pits
    ]
    canonical_opponent_pits = [
        _require_int(value, "state.opponent_pits") for value in opponent_pits
    ]
    player_store = _require_int(state["player_store"], "state.player_store")
    opponent_store = _require_int(state["opponent_store"], "state.opponent_store")
    current_player = _require_int(state["current_player"], "state.current_player")
    if current_player not in (0, 1):
        raise ValueError("suite row state.current_player must be 0 or 1")

    return {
        "player_pits": canonical_player_pits,
        "opponent_pits": canonical_opponent_pits,
        "player_store": player_store,
        "opponent_store": opponent_store,
        "current_player": current_player,
    }


def _validate_legal_moves(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("suite row legal_moves must be a non-empty list")
    legal_moves = tuple(_require_int(move, "legal_moves") for move in value)
    if any(move < 0 or move >= PITS_PER_PLAYER for move in legal_moves):
        raise ValueError(
            "suite row legal_moves must contain unique moves in range 0..5"
        )
    if len(set(legal_moves)) != len(legal_moves):
        raise ValueError(
            "suite row legal_moves must contain unique moves in range 0..5"
        )
    return legal_moves


def _validate_tags(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("suite row tags must be a non-empty list")
    tags: list[str] = []
    for tag in value:
        tags.append(_require_non_empty_string(tag, "tags"))
    return tuple(tags)


def _validate_row(row: dict[str, Any]) -> ForensicPosition:
    if not isinstance(row, dict):
        raise ValueError("suite row must be a dictionary")

    _require_keys(
        row,
        (
            "id",
            "state",
            "side_to_move",
            "legal_moves",
            "phase",
            "bucket",
            "tags",
            "source",
        ),
    )

    bucket = _require_non_empty_string(row["bucket"], "bucket")
    if bucket not in REQUIRED_BUCKETS:
        raise ValueError(f"unsupported forensic bucket: {bucket}")

    state = _validate_state(row["state"])
    side_to_move = _require_int(row["side_to_move"], "side_to_move")
    if side_to_move not in (0, 1):
        raise ValueError("suite row side_to_move must be 0 or 1")
    if side_to_move != int(state["current_player"]):
        raise ValueError("suite row side_to_move must match state.current_player")

    return ForensicPosition(
        id=_require_non_empty_string(row["id"], "id"),
        state=state,
        side_to_move=side_to_move,
        legal_moves=_validate_legal_moves(row["legal_moves"]),
        phase=_require_non_empty_string(row["phase"], "phase"),
        bucket=bucket,
        tags=_validate_tags(row["tags"]),
        source=_require_non_empty_string(row["source"], "source"),
    )


def load_suite(path: str | Path) -> list[ForensicPosition]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("forensic suite must be a JSON list")

    suite: list[ForensicPosition] = []
    seen_keys: set[str] = set()
    for row in rows:
        position = _validate_row(row)
        if position.canonical_key in seen_keys:
            raise ValueError(f"duplicate canonical state detected: {position.id}")
        seen_keys.add(position.canonical_key)
        suite.append(position)
    return suite


def _round_metric(value: float) -> float:
    return round(float(value), 4)


def centered_value_from_probability(probability: float) -> float:
    return _round_metric((2.0 * float(probability)) - 1.0)


def summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positions = len(rows)
    top1 = sum(1 for row in rows if bool(row.get("agrees_top1")))
    regrets = [float(row["regret"]) for row in rows if row.get("regret") is not None]
    blunders = sum(1 for regret in regrets if math.isfinite(regret) and regret > 0.0)
    value_errors = [
        float(row["value_error"]) for row in rows if row.get("value_error") is not None
    ]
    return {
        "positions": positions,
        "top1_agreement": 0.0 if positions == 0 else _round_metric(top1 / positions),
        "average_regret": 0.0
        if not regrets
        else _round_metric(sum(regrets) / len(regrets)),
        "blunder_rate": 0.0 if positions == 0 else _round_metric(blunders / positions),
        "value_calibration_mae": None
        if not value_errors
        else _round_metric(sum(value_errors) / len(value_errors)),
    }


def summarize_system(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = sorted({str(row["bucket"]) for row in rows})
    return {
        "overall": summarize_bucket(rows),
        "buckets": {
            bucket: summarize_bucket([row for row in rows if row["bucket"] == bucket])
            for bucket in buckets
        },
        "rows": rows,
    }


def summarize_bucket_matrix(
    system_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    bucket_names = sorted(
        {str(row["bucket"]) for rows in system_rows.values() for row in rows}
    )
    matrix: dict[str, dict[str, Any]] = {}
    for bucket in bucket_names:
        rows_for_bucket = {
            system_name: [row for row in rows if row["bucket"] == bucket]
            for system_name, rows in system_rows.items()
        }
        matrix[bucket] = {
            "positions": len(next(iter(rows_for_bucket.values()), [])),
            "systems": {
                system_name: summarize_bucket(rows)
                for system_name, rows in rows_for_bucket.items()
            },
        }
    return matrix
