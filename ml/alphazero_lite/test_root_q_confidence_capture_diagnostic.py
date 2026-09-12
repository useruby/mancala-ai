import random

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_root_q_confidence_capture_diagnostic import (
    lane_name,
    state_hash,
)
from ml.alphazero_lite.self_play import PUCT


def test_confidence_lane_names_and_root_hash_match_puct():
    game = KalahGame.from_state(
        {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    probe = PUCT(None, 1, 1.25, random.Random(42))

    assert lane_name(0.75) == "root_q_alpha_075"
    assert state_hash(game) == probe._state_hash(game)
