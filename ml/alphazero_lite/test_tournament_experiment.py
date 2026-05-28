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

    def test_arena_confirm_command_uses_shared_worker_default(self):
        from ml.alphazero_lite.tournament_experiment import arena_confirm_command

        command = arena_confirm_command(
            winner_dir="/tmp/winner",
            confirm_games=30,
            challenger_sims=640,
            current_sims=384,
            arena_path=Path("/tmp/arena.json"),
            min_arena_score=0.55,
        )

        self.assertEqual(".venv/bin/python", command[0])
        self.assertEqual("ml/alphazero_lite/arena.py", command[1])
        self.assertIn("--challenger", command)
        self.assertEqual("/tmp/winner", command[command.index("--challenger") + 1])
        self.assertIn("--games", command)
        self.assertEqual("30", command[command.index("--games") + 1])
        self.assertIn("--workers", command)
        self.assertEqual("24", command[command.index("--workers") + 1])
        self.assertIn("--out", command)
        self.assertEqual("/tmp/arena.json", command[command.index("--out") + 1])

    def test_parse_mcts_score_accepts_pretty_printed_report_json(self):
        from ml.alphazero_lite.tournament_experiment import parse_mcts_score

        payload = '{\n  "schema": "azlite_vs_mcts_v1",\n  "score": 0.375\n}\n'

        self.assertEqual(0.375, parse_mcts_score(payload))


if __name__ == "__main__":
    unittest.main()
