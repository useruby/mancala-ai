import json
import unittest
from pathlib import Path

from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state


class KalahRulesParityTest(unittest.TestCase):
    def test_python_engine_matches_golden_vectors(self):
        fixture_path = Path("test/fixtures/ai/kalah_rule_vectors.json")
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))

        self.assertEqual("kalah_v1", payload["rules_version"])

        for vector in payload["vectors"]:
            game = KalahGame.from_state(vector["initial_state"])

            for step in vector["steps"]:
                absolute_move = game.pit_index(step["relative_move"])
                self.assertEqual(step["absolute_move"], absolute_move, vector["id"])

                ok = game.move(absolute_move)
                self.assertEqual(step["ok"], ok, vector["id"])
                self.assertEqual(step["state"], game.to_state(), vector["id"])
                self.assertEqual(step["winner"], game.winner, vector["id"])
                self.assertEqual(step["over"], game.over(), vector["id"])
                self.assertEqual(
                    step["possible_moves"], game.possible_moves(), vector["id"]
                )

    def test_move_consequence_reports_illegal_move_without_state_change(self):
        state = {
            "current_player": 0,
            "player_pits": [1, 0, 0, 0, 0, 0],
            "opponent_pits": [1, 1, 1, 1, 1, 1],
            "player_store": 0,
            "opponent_store": 0,
        }

        consequence = move_consequence_for_state(state, 1)
        game = KalahGame.from_state(state)

        self.assertFalse(game.move(game.pit_index(1)))
        self.assertEqual(game.to_state(), state)
        self.assertFalse(consequence["legal"])
        self.assertEqual(consequence["seed_count"], 0)
        self.assertFalse(consequence["gives_extra_turn"])
        self.assertFalse(consequence["produces_capture"])
        self.assertEqual(consequence["store_delta_immediate"], 0)
        self.assertEqual(consequence["opponent_store_delta_immediate"], 0)
        self.assertEqual(consequence["immediate_score_delta"], 0)
        self.assertIsNone(consequence["resulting_side_to_move"])
        self.assertFalse(consequence["game_over_after_move"])

    def test_move_consequence_matches_extra_turn_move(self):
        state = {
            "current_player": 0,
            "player_pits": [1, 0, 0, 0, 0, 1],
            "opponent_pits": [1, 1, 1, 1, 1, 1],
            "player_store": 0,
            "opponent_store": 0,
        }

        self._assert_move_consequence_matches_state_transition(
            state,
            5,
            gives_extra_turn=True,
            produces_capture=False,
            capture_count=0,
            lands_on_own_empty_pit=False,
            game_over_after_move=False,
        )

    def test_move_consequence_matches_capture_move(self):
        state = {
            "current_player": 0,
            "player_pits": [1, 0, 1, 0, 0, 0],
            "opponent_pits": [0, 0, 5, 1, 0, 0],
            "player_store": 0,
            "opponent_store": 0,
        }

        self._assert_move_consequence_matches_state_transition(
            state,
            2,
            gives_extra_turn=False,
            produces_capture=True,
            capture_count=6,
            lands_on_own_empty_pit=True,
            game_over_after_move=False,
        )

    def test_move_consequence_matches_game_over_move(self):
        state = {
            "current_player": 0,
            "player_pits": [1, 0, 0, 0, 0, 0],
            "opponent_pits": [0, 0, 0, 0, 0, 0],
            "player_store": 0,
            "opponent_store": 0,
        }

        self._assert_move_consequence_matches_state_transition(
            state,
            0,
            gives_extra_turn=False,
            produces_capture=False,
            capture_count=0,
            lands_on_own_empty_pit=True,
            game_over_after_move=True,
        )

    def _assert_move_consequence_matches_state_transition(
        self,
        state,
        move,
        *,
        gives_extra_turn,
        produces_capture,
        capture_count,
        lands_on_own_empty_pit,
        game_over_after_move,
    ):
        consequence = move_consequence_for_state(state, move)
        game = KalahGame.from_state(state)
        current_player = game.current_player
        opponent = 1 - current_player
        own_store_before = game.captured_seeds[current_player]
        opponent_store_before = game.captured_seeds[opponent]
        score_before = own_store_before - opponent_store_before

        ok = game.move(game.pit_index(move))
        own_store_after = game.captured_seeds[current_player]
        opponent_store_after = game.captured_seeds[opponent]
        score_after = own_store_after - opponent_store_after

        self.assertTrue(ok)
        self.assertTrue(consequence["legal"])
        self.assertEqual(consequence["seed_count"], state["player_pits"][move])
        self.assertEqual(consequence["gives_extra_turn"], gives_extra_turn)
        self.assertEqual(consequence["produces_capture"], produces_capture)
        self.assertEqual(consequence["capture_count"], capture_count)
        self.assertEqual(consequence["lands_on_own_empty_pit"], lands_on_own_empty_pit)
        self.assertEqual(
            consequence["store_delta_immediate"], own_store_after - own_store_before
        )
        self.assertEqual(
            consequence["opponent_store_delta_immediate"],
            opponent_store_after - opponent_store_before,
        )
        self.assertEqual(consequence["resulting_side_to_move"], game.current_player)
        self.assertEqual(consequence["game_over_after_move"], game_over_after_move)
        self.assertEqual(consequence["game_over_after_move"], game.over())
        self.assertEqual(
            consequence["immediate_score_delta"], score_after - score_before
        )


if __name__ == "__main__":
    unittest.main()
