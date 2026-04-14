import unittest

from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.kalah_rules import KalahGame


class EndgameTablebaseTest(unittest.TestCase):
    def test_lookup_returns_none_when_position_not_present(self):
        tablebase = EndgameTablebase()
        game = KalahGame.from_state(
            {
                "player_pits": [1, 0, 0, 0, 0, 0],
                "opponent_pits": [0, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 19,
                "current_player": 0,
            }
        )

        self.assertIsNone(tablebase.lookup(game, perspective_player=0))

    def test_record_and_lookup_round_trip(self):
        tablebase = EndgameTablebase()
        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 0],
                "opponent_pits": [0, 0, 0, 0, 0, 0],
                "player_store": 22,
                "opponent_store": 15,
                "current_player": 0,
            }
        )

        tablebase.record(game, perspective_player=0, value=1.0)

        self.assertEqual(1.0, tablebase.lookup(game, perspective_player=0))

    def test_lookup_is_perspective_specific(self):
        tablebase = EndgameTablebase()
        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 1],
                "opponent_pits": [0, 0, 0, 0, 0, 0],
                "player_store": 21,
                "opponent_store": 20,
                "current_player": 0,
            }
        )

        tablebase.record(game, perspective_player=0, value=1.0)

        self.assertEqual(1.0, tablebase.lookup(game, perspective_player=0))
        self.assertIsNone(tablebase.lookup(game, perspective_player=1))
