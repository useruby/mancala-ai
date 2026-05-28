import unittest
from pathlib import Path


class TournamentExperimentTest(unittest.TestCase):
    def test_mcts_baseline_command_uses_python_baseline_script(self):
        from ml.alphazero_lite.tournament_experiment import mcts_baseline_command

        command = mcts_baseline_command(
            challenger_path=Path("/tmp/challenger"),
            games=10,
            az_base_simulations=640,
            mcts_simulations=1200,
            out_path=Path("/tmp/report.json"),
        )

        self.assertEqual(".venv/bin/python", command[0])
        self.assertEqual("ml/alphazero_lite/mcts1200_baseline.py", command[1])
        self.assertNotIn("bin/rails", command)
        self.assertIn("--challenger-path", command)
        self.assertIn("/tmp/challenger", command)
        self.assertIn("--games", command)
        self.assertIn("10", command)
        self.assertIn("--out", command)
        self.assertIn("/tmp/report.json", command)
        self.assertIn("--workers", command)
        self.assertEqual("24", command[command.index("--workers") + 1])
        self.assertIn("--root-policy-mode", command)
        self.assertIn("visit_count", command)
        self.assertIn("--tactical-root-bias", command)
        self.assertIn("0.0", command)

    def test_parse_mcts_score_accepts_pretty_printed_report_json(self):
        from ml.alphazero_lite.tournament_experiment import parse_mcts_score

        payload = '{\n  "schema": "azlite_vs_mcts_v1",\n  "score": 0.375\n}\n'

        self.assertEqual(0.375, parse_mcts_score(payload))


if __name__ == "__main__":
    unittest.main()
