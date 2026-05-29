import json
import unittest
from pathlib import Path

from ml.alphazero_lite import self_play
from ml.alphazero_lite.kalah_rules import KalahGame


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

if __name__ == "__main__":
    unittest.main()
