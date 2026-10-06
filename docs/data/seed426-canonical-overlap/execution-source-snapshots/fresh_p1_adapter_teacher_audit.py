"""Pure helpers for the PR #214 teacher-quality audit."""

from __future__ import annotations

from typing import Any

from ml.alphazero_lite.self_play import encode_state


def decode_kalah_v3_base_state(features: list[Any]) -> dict[str, Any]:
    """Recover the invertible 15-feature Kalah state prefix from `kalah_v3`."""
    if not isinstance(features, list) or len(features) < 15:
        raise ValueError("kalah_v3 state must contain its 15-feature base prefix")

    def stone(index: int) -> int:
        value = float(features[index]) * 48.0
        rounded = round(value)
        if abs(value - rounded) > 1e-5 or rounded < 0:
            raise ValueError(f"kalah_v3 base feature {index} is not an exact /48 value")
        return int(rounded)

    player = int(round(float(features[14])))
    if player not in (0, 1) or abs(float(features[14]) - player) > 1e-5:
        raise ValueError("kalah_v3 current_player must be exactly 0 or 1")
    return {
        "player_pits": [stone(index) for index in range(6)],
        "opponent_pits": [stone(index) for index in range(6, 12)],
        "player_store": stone(12),
        "opponent_store": stone(13),
        "current_player": player,
    }


def state_round_trips_kalah_v3(features: list[Any]) -> bool:
    """Check that decoded base fields reproduce the complete encoded state."""
    encoded = encode_state(
        decode_kalah_v3_base_state(features), input_encoding="kalah_v3"
    )
    return len(encoded) == len(features) and all(
        abs(float(left) - float(right)) <= 1e-6
        for left, right in zip(encoded, features, strict=True)
    )
