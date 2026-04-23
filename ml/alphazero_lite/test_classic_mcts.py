import unittest

from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.kalah_rules import KalahGame


class SpyTablebase:
    def __init__(self, *, cached_value=None, lookup_value=None):
        self.cached_value = cached_value
        self.lookup_value = lookup_value
        self.lookup_calls = 0
        self.lookup_cached_calls = 0

    def lookup_cached(self, game, perspective_player):
        del game, perspective_player
        self.lookup_cached_calls += 1
        return self.cached_value

    def lookup(self, game, perspective_player):
        del game, perspective_player
        self.lookup_calls += 1
        return self.lookup_value


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

    def test_simulate_playout_runtime_mode_skips_exact_solver_for_sparse_nonterminal_state(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 1],
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 20,
                "current_player": 0,
            }
        )
        tablebase = SpyTablebase(lookup_value=0.75)

        mcts = MCTS(game, simulations=4, seed=42, endgame_tablebase=tablebase)

        self.assertEqual(0.5, mcts.simulate_playout(game))
        self.assertEqual(0, tablebase.lookup_calls)

    def test_simulate_playout_uses_exact_solver_when_evaluation_mode_threshold_applies(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 1],
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 20,
                "current_player": 0,
            }
        )
        tablebase = SpyTablebase(lookup_value=0.75)

        mcts = MCTS(
            game,
            simulations=4,
            seed=42,
            endgame_tablebase=tablebase,
            exact_solve_enabled=True,
            exact_solve_stone_threshold=2,
        )

        self.assertEqual(0.75, mcts.simulate_playout(game))
        self.assertEqual(1, tablebase.lookup_calls)

    def test_simulate_playout_skips_cached_probe_when_exact_solver_applies(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 1],
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 20,
                "current_player": 0,
            }
        )
        tablebase = SpyTablebase(lookup_value=0.75)

        mcts = MCTS(
            game,
            simulations=4,
            seed=42,
            endgame_tablebase=tablebase,
            exact_solve_enabled=True,
            exact_solve_stone_threshold=2,
        )

        self.assertEqual(0.75, mcts.simulate_playout(game))
        self.assertEqual(1, tablebase.lookup_calls)
        self.assertEqual(0, tablebase.lookup_cached_calls)

    def test_simulate_playout_falls_back_to_rollout_when_exact_solver_returns_none(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 1],
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 20,
                "current_player": 0,
            }
        )
        tablebase = SpyTablebase(lookup_value=None)

        mcts = MCTS(
            game,
            simulations=4,
            seed=42,
            endgame_tablebase=tablebase,
            exact_solve_enabled=True,
            exact_solve_stone_threshold=2,
        )
        playout_move_calls = 0

        def fake_choose_playout_move(playout_game):
            nonlocal playout_move_calls
            self.assertIsNot(playout_game, game)
            playout_move_calls += 1
            return 5

        mcts.choose_playout_move = fake_choose_playout_move

        self.assertTrue(mcts.exact_solve_applies(game))
        self.assertEqual(0.5, mcts.simulate_playout(game))
        self.assertEqual(1, tablebase.lookup_calls)
        self.assertEqual(1, playout_move_calls)

    def test_simulate_playout_skips_exact_solver_above_threshold(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 1],
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 20,
                "current_player": 0,
            }
        )
        tablebase = SpyTablebase(lookup_value=0.75)

        mcts = MCTS(
            game,
            simulations=4,
            seed=42,
            endgame_tablebase=tablebase,
            exact_solve_enabled=True,
            exact_solve_stone_threshold=1,
        )

        self.assertEqual(0.5, mcts.simulate_playout(game))
        self.assertEqual(0, tablebase.lookup_calls)

    def test_simulate_playout_returns_terminal_tablebase_value_for_no_legal_move_state(self):
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

        tablebase = EndgameTablebase()
        mcts = MCTS(game, simulations=4, seed=42, endgame_tablebase=tablebase)

        self.assertEqual(0.0, mcts.simulate_playout(game))

    def test_simulate_playout_uses_recorded_tablebase_hit_when_exact_solver_disabled(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 1],
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 20,
                "current_player": 0,
            }
        )
        tablebase = EndgameTablebase()
        tablebase.record(game, perspective_player=0, value=0.75)

        mcts = MCTS(game, simulations=4, seed=42, endgame_tablebase=tablebase)

        self.assertEqual(0.75, mcts.simulate_playout(game))
