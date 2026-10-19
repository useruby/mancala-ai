"""Focused accounting tests for seed488 retrospective analysis."""

import unittest

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed488_retrospective import (
    _root_action,
    _validate_ledger,
    analyze,
)


class Seed488RetrospectiveTests(unittest.TestCase):
    @staticmethod
    def _opening(root_player=0):
        game = KalahGame.from_state(
            {
                "player_pits": [4] * 6,
                "opponent_pits": [4] * 6,
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        prefix = []
        if root_player == 1:
            game.move(game.pit_index(0))
            prefix.append(0)
        return {"state": game.to_state(), "prefix_moves": prefix}

    def test_complete_accounting(self):
        from pathlib import Path

        result = analyze(Path.cwd())
        self.assertEqual(
            (
                result["opening_count"],
                result["game_count"],
                result["ledger_game_count"],
            ),
            (512, 1024, 1024),
        )
        self.assertEqual(sum(result["outcomes"].values()), 1024)
        self.assertAlmostEqual(result["score"], 0.4892578125)
        self.assertEqual(sum(result["pair_score_distribution"].values()), 512)

    def test_telemetry_and_first_action_accounting(self):
        from pathlib import Path

        result = analyze(Path.cwd())
        self.assertEqual(result["handoff"]["games_with_any"], 953)
        self.assertEqual(result["root_action_comparison"]["compared_openings"], 512)
        self.assertEqual(
            result["root_action_comparison"]["agreements"]
            + result["root_action_comparison"]["disagreements"],
            512,
        )
        self.assertEqual(result["root_action_comparison"]["agreements"], 480)

    def test_root_action_uses_absolute_trajectory_and_root_player(self):
        opening = self._opening(root_player=1)
        F_as_root = {
            "challenger_player": 1,
            "trajectory": "8,7,2",
            "first_move_challenger": 2,
            "first_move_current": 1,
        }
        F_after_root_move = {
            "challenger_player": 0,
            "trajectory": "8,7,2",
            "first_move_current": 2,
            "first_move_challenger": 4,
        }
        self.assertEqual(_root_action(opening, F_as_root), 2)
        self.assertEqual(_root_action(opening, F_after_root_move), 2)
        # F's own first actions differ because its first turn is at another
        # state in the swapped-seat game; root actions agree nonetheless.
        self.assertNotEqual(
            F_as_root["first_move_challenger"],
            F_after_root_move["first_move_challenger"],
        )

    def test_player_zero_root_uses_absolute_pit_indices(self):
        opening = self._opening()
        row = {
            "challenger_player": 1,
            "trajectory": "3,9,2",
            "first_move_current": 3,
        }
        self.assertEqual(_root_action(opening, row), 3)

    def test_extra_turn_does_not_change_opening_root_action(self):
        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 1],
                "opponent_pits": [4] * 6,
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        opening = {"state": game.to_state(), "prefix_moves": []}
        self.assertTrue(game.move(5))
        self.assertEqual(game.current_player, 0)
        # Pit 5 grants an extra turn; the next action is still not the
        # action being compared across swapped seats.
        row = {
            "challenger_player": 0,
            "trajectory": "5,4",
            "first_move_challenger": 5,
        }
        self.assertEqual(_root_action(opening, row), 5)

    def test_missing_first_action_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "first_trajectory_action_missing"):
            _root_action(
                self._opening(),
                {
                    "challenger_player": 0,
                    "trajectory": "",
                    "first_move_challenger": None,
                },
            )
        with self.assertRaisesRegex(
            ValueError, "first_trajectory_action_missing_or_mismatch"
        ):
            _root_action(
                self._opening(),
                {
                    "challenger_player": 0,
                    "trajectory": "3,9",
                    "first_move_challenger": None,
                },
            )

    def test_ledger_must_match_archived_opening_and_seat(self):
        ledger = [{"opening_index": 3, "F_player": 0, "winner": "F"}]
        games = [
            {"parent_opening_index": 3, "challenger_player": 0, "winner": "challenger"}
        ]
        _validate_ledger(ledger, games)
        with self.assertRaisesRegex(ValueError, "ledger_game_identity_mismatch"):
            _validate_ledger(
                ledger,
                [{"parent_opening_index": 4, "challenger_player": 0}],
            )
        with self.assertRaisesRegex(ValueError, "ledger_game_result_mismatch"):
            _validate_ledger(
                [{"opening_index": 3, "F_player": 0, "winner": "initializer"}],
                games,
            )
        with self.assertRaisesRegex(ValueError, "duplicate_ledger_or_game_identity"):
            _validate_ledger(ledger * 2, games * 2)


if __name__ == "__main__":
    unittest.main()
