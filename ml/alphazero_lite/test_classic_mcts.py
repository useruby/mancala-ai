import unittest

from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.kalah_rules import KalahGame


class ClassicMCTSTest(unittest.TestCase):
    def test_root_summary_selected_move_stays_aligned_with_choose_move_on_same_instance(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(game, simulations=8, seed=11)
        call_count = 0

        def fake_run_search(node, simulations_to_run, *, allow_early_stop):
            nonlocal call_count
            del simulations_to_run, allow_early_stop
            call_count += 1
            for action, child in node.children.items():
                child.visits = 1
                child.wins = 0.0
            if call_count == 1:
                node.children[0].wins = 1.0
            else:
                node.children[5].wins = 1.0
            return 1

        search.run_search = fake_run_search

        selected_move = search.choose_move()
        summary = search.root_summary()

        self.assertEqual(selected_move, summary["selected_move"])
        self.assertEqual(1, call_count)

    def test_root_summary_exposes_child_visit_and_win_rate(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(game, simulations=32, seed=11)

        summary = search.root_summary()

        self.assertIn("selected_move", summary)
        self.assertTrue(summary["child_stats"])
        self.assertIn("visits", summary["child_stats"][0])
        self.assertIn("win_rate", summary["child_stats"][0])

    def test_search_root_rebuilds_when_underlying_game_state_changes(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(game, simulations=8, seed=11)
        observed_states = []

        def fake_run_search(node, simulations_to_run, *, allow_early_stop):
            del simulations_to_run, allow_early_stop
            observed_states.append(node.game.to_state())
            for child in node.children.values():
                child.visits = 1
                child.wins = 0.0
            return 1

        search.run_search = fake_run_search

        search.choose_move()
        game.move(game.pit_index(0))
        search.root_summary()

        self.assertEqual(2, len(observed_states))
        self.assertNotEqual(observed_states[0], observed_states[1])

    def test_search_root_refreshes_player_when_underlying_game_turn_changes(self):
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
        search = MCTS(game, simulations=1, seed=11)

        self.assertEqual(0, search.player)
        search.search_root()

        game.move(game.pit_index(4))
        self.assertEqual(1, game.current_player)

        search.search_root()

        self.assertEqual(1, search.player)

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

    def test_simulate_playout_uses_tablebase_value_when_available(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 0],
                "opponent_pits": [0, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 20,
                "current_player": 0,
            }
        )
        tablebase = EndgameTablebase()
        tablebase.record(game, perspective_player=0, value=0.75)

        mcts = MCTS(game, simulations=4, seed=42, endgame_tablebase=tablebase)

        self.assertEqual(0.75, mcts.simulate_playout(game))

    def test_simulate_playout_falls_back_without_tablebase_hit(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 0],
                "opponent_pits": [0, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 20,
                "current_player": 0,
            }
        )

        tablebase = EndgameTablebase()
        mcts_with_tb = MCTS(game, simulations=4, seed=42, endgame_tablebase=tablebase)
        mcts_without_tb = MCTS(game, simulations=4, seed=42)

        self.assertEqual(mcts_without_tb.simulate_playout(game), mcts_with_tb.simulate_playout(game))
