import unittest

from ml.alphazero_lite.endgame_tablebase import (
    EndgameTablebase,
    EndgameTablebaseContract,
)
from ml.alphazero_lite.kalah_rules import KalahGame


class EndgameTablebaseTest(unittest.TestCase):
    def test_protocol_methods_do_not_silently_return_none(self):
        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 0],
                "opponent_pits": [0, 0, 0, 0, 0, 0],
                "player_store": 22,
                "opponent_store": 15,
                "current_player": 0,
            }
        )

        with self.assertRaises(NotImplementedError):
            EndgameTablebaseContract.lookup(object(), game, 0)
        with self.assertRaises(NotImplementedError):
            EndgameTablebaseContract.lookup_cached(object(), game, 0)

    def test_lookup_scores_unswept_tied_terminal_position_as_draw_from_both_perspectives(
        self,
    ):
        tablebase = EndgameTablebase()
        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 0],
                "opponent_pits": [1, 1, 0, 0, 0, 0],
                "player_store": 22,
                "opponent_store": 20,
                "current_player": 0,
            }
        )

        self.assertTrue(game.over())
        self.assertEqual(0.5, tablebase.lookup(game, perspective_player=0))
        self.assertEqual(0.5, tablebase.lookup(game, perspective_player=1))

    def test_lookup_scores_unswept_terminal_position_from_settled_totals(self):
        tablebase = EndgameTablebase()
        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 0],
                "opponent_pits": [1, 1, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 19,
                "current_player": 0,
            }
        )

        self.assertTrue(game.over())
        self.assertEqual(0.0, tablebase.lookup(game, perspective_player=0))
        self.assertEqual(1.0, tablebase.lookup(game, perspective_player=1))

    def test_lookup_scores_terminal_position_above_sparse_threshold(self):
        tablebase = EndgameTablebase()
        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 0],
                "opponent_pits": [3, 3, 3, 3, 3, 3],
                "player_store": 20,
                "opponent_store": 19,
                "current_player": 0,
            }
        )

        self.assertTrue(game.over())
        self.assertGreater(sum(game.pits), tablebase.MAX_SOLVED_SEEDS)
        self.assertEqual(0.0, tablebase.lookup(game, perspective_player=0))
        self.assertEqual(1.0, tablebase.lookup(game, perspective_player=1))

    def test_lookup_recurses_when_opponent_side_is_empty_but_current_player_can_still_move(
        self,
    ):
        tablebase = EndgameTablebase()
        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 3],
                "opponent_pits": [0, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 19,
                "current_player": 0,
            }
        )

        self.assertFalse(game.over())
        self.assertIsNone(tablebase.lookup_cached(game, perspective_player=0))
        self.assertIsNone(tablebase.lookup_cached(game, perspective_player=1))

        child = game.clone()
        child.move(child.pit_index(5))

        self.assertEqual(0.5, tablebase.lookup(game, perspective_player=0))
        self.assertEqual(0.5, tablebase.lookup(game, perspective_player=1))
        self.assertIn(tablebase._key(child, 0), tablebase._values)
        self.assertIn(tablebase._key(child, 1), tablebase._values)

    def test_lookup_solves_sparse_position_on_demand(self):
        tablebase = EndgameTablebase()
        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 0],
                "opponent_pits": [0, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 19,
                "current_player": 0,
            }
        )

        self.assertEqual(1.0, tablebase.lookup(game, perspective_player=0))
        self.assertEqual(0.0, tablebase.lookup(game, perspective_player=1))

    def test_lookup_caches_recursive_sparse_subpositions(self):
        tablebase = EndgameTablebase()
        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 1],
                "opponent_pits": [0, 0, 0, 0, 0, 1],
                "player_store": 20,
                "opponent_store": 19,
                "current_player": 0,
            }
        )
        child = game.clone()
        child.move(child.pit_index(5))

        solved = tablebase.lookup(game, perspective_player=0)

        self.assertEqual(solved, tablebase.lookup(game, perspective_player=0))
        self.assertIn(tablebase._key(child, 0), tablebase._values)
        self.assertGreaterEqual(len(tablebase._values), 2)

    def test_lookup_prefers_extra_turn_branch_over_losing_handoff_branch(self):
        tablebase = EndgameTablebase()
        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 1],
                "opponent_pits": [0, 0, 0, 0, 1, 0],
                "player_store": 22,
                "opponent_store": 23,
                "current_player": 0,
            }
        )
        handoff_child = game.clone()
        extra_turn_child = game.clone()
        handoff_child.move(handoff_child.pit_index(4))
        extra_turn_child.move(extra_turn_child.pit_index(5))

        self.assertEqual([4, 5], game.possible_moves())
        self.assertEqual(1, handoff_child.current_player)
        self.assertEqual(0, extra_turn_child.current_player)
        self.assertEqual(0.0, tablebase.lookup(handoff_child, perspective_player=0))
        self.assertEqual(0.5, tablebase.lookup(extra_turn_child, perspective_player=0))
        self.assertEqual(1.0, tablebase.lookup(handoff_child, perspective_player=1))
        self.assertEqual(0.5, tablebase.lookup(extra_turn_child, perspective_player=1))
        self.assertEqual(0.5, tablebase.lookup(game, perspective_player=0))
        self.assertEqual(0.5, tablebase.lookup(game, perspective_player=1))

    def test_lookup_returns_none_when_position_not_present(self):
        tablebase = EndgameTablebase()
        game = KalahGame.from_state(
            {
                "player_pits": [3, 0, 0, 0, 0, 0],
                "opponent_pits": [3, 3, 3, 3, 3, 3],
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
                "player_pits": [3, 0, 0, 0, 0, 0],
                "opponent_pits": [3, 3, 3, 3, 3, 3],
                "player_store": 21,
                "opponent_store": 20,
                "current_player": 0,
            }
        )

        tablebase.record(game, perspective_player=0, value=1.0)

        self.assertEqual(1.0, tablebase.lookup(game, perspective_player=0))
        self.assertIsNone(tablebase.lookup(game, perspective_player=1))
