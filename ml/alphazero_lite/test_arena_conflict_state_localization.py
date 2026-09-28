import unittest

from ml.alphazero_lite.arena_conflict_state_localization import (
    active_bucket,
    state_rows,
)


class ArenaConflictStateLocalizationTest(unittest.TestCase):
    def test_active_stone_buckets_are_fixed(self):
        self.assertEqual(">40", active_bucket(41))
        self.assertEqual("33-40", active_bucket(33))
        self.assertEqual("<=16", active_bucket(16))

    def test_trajectory_uses_absolute_arena_pit_indices(self):
        game = {
            "opening_prefix_moves": [],
            # Player 0 selects absolute pit 0, then player 1 selects absolute pit 8.
            "trajectory": "0,8,0",
        }
        rows = state_rows(game)
        self.assertEqual(3, len(rows))
        self.assertEqual(0, rows[0][1]["current_player"])
        self.assertEqual(1, rows[1][1]["current_player"])
        self.assertEqual(0, rows[2][1]["opponent_pits"][2])


if __name__ == "__main__":
    unittest.main()
