import json

from ml.alphazero_lite.replay_policy_refresh import (
    POLICY_FIELDS,
    game_from_encoded_state,
)
from ml.alphazero_lite.self_play import encode_state


def test_game_from_encoded_state_round_trips_rules_state() -> None:
    state = {
        "player_pits": [4, 0, 1, 2, 3, 4],
        "opponent_pits": [0, 1, 2, 3, 4, 5],
        "player_store": 7,
        "opponent_store": 12,
        "current_player": 1,
    }
    assert (
        game_from_encoded_state(
            encode_state(state, input_encoding="kalah_v3")
        ).to_state()
        == state
    )


def test_policy_field_scope_excludes_trajectory_and_value_labels() -> None:
    row = {
        "state": [0.0],
        "value": 0.5,
        "winner": 1,
        "bucket": "x",
        "policy": [1.0],
        "teacher_source": "old",
    }
    immutable = {key: value for key, value in row.items() if key not in POLICY_FIELDS}
    changed = json.loads(json.dumps(row))
    changed["policy"] = [0.0]
    changed["teacher_source"] = "new"
    assert immutable == {
        key: value for key, value in changed.items() if key not in POLICY_FIELDS
    }
