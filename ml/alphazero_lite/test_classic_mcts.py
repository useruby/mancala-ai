import unittest

from ml.alphazero_lite.kalah_rules import KalahGame


class ClassicMCTSTest(unittest.TestCase):
    def test_choose_move_returns_none_without_legal_moves(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 0],
                "opponent_pits": [1, 1, 1, 1, 1, 1],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )

        self.assertIsNone(MCTS(game, simulations=32, seed=42).choose_move())

    def test_choose_move_is_deterministic_for_same_seed(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }

        first = MCTS(KalahGame.from_state(state), simulations=64, seed=42).choose_move()
        second = MCTS(KalahGame.from_state(state), simulations=64, seed=42).choose_move()

        self.assertEqual(first, second)

    def test_choose_playout_move_prefers_extra_turn(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 0],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )

        self.assertEqual(4, MCTS(game, simulations=8, seed=42).choose_playout_move(game))
