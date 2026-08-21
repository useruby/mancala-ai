from ml.alphazero_lite.fresh_p1_adapter_teacher_audit import (
    decode_kalah_v3_base_state,
    state_round_trips_kalah_v3,
)
from ml.alphazero_lite.self_play import encode_state


def test_decode_kalah_v3_base_state_round_trips() -> None:
    state = {
        "player_pits": [0, 1, 2, 3, 4, 5],
        "opponent_pits": [6, 7, 8, 9, 0, 1],
        "player_store": 15,
        "opponent_store": 17,
        "current_player": 1,
    }
    features = encode_state(state, input_encoding="kalah_v3")

    assert decode_kalah_v3_base_state(features) == state
    assert state_round_trips_kalah_v3(features)


def test_decode_kalah_v3_base_state_rejects_non_integral_base_values() -> None:
    features = [0.0] * 27
    features[0] = 0.001

    try:
        decode_kalah_v3_base_state(features)
    except ValueError as error:
        assert "exact /48" in str(error)
    else:
        raise AssertionError("non-integral base features must be rejected")
