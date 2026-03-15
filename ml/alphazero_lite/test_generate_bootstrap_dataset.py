import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ml.alphazero_lite import generate_bootstrap_dataset


class GenerateBootstrapDatasetTest(unittest.TestCase):
    def test_cli_writes_jsonl_rows(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-bootstrap-") as tmp:
            out_path = Path(tmp) / "bootstrap.jsonl"

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/generate_bootstrap_dataset.py",
                    "--out",
                    str(out_path),
                    "--games",
                    "2",
                    "--simulations",
                    "8",
                    "--seed",
                    "42",
                    "--max-positions-per-game",
                    "4",
                    "--workers",
                    "1",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertTrue(out_path.exists())
            rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(rows)
            self.assertIn("state", rows[0])
            self.assertIn("policy", rows[0])
            self.assertIn("value", rows[0])

    def test_shape_policy_uses_true_dirichlet_sampler(self):
        fake_rng = mock.Mock()
        fake_rng.randint.return_value = 7
        fake_generator = mock.Mock()
        fake_generator.dirichlet.return_value = [0.2, 0.8]

        with mock.patch.object(generate_bootstrap_dataset.np.random, "default_rng", return_value=fake_generator):
            policy = generate_bootstrap_dataset.shape_policy(
                visits=[4.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                legal_moves=[0, 1],
                move_index=0,
                rng=fake_rng,
                apply_dirichlet=True,
                top_k=None,
                tau=1.0,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.25,
                dirichlet_opening_moves=8,
            )

        fake_rng.randint.assert_called_once_with(0, 2**31 - 1)
        fake_generator.dirichlet.assert_called_once_with([0.3, 0.3])
        self.assertAlmostEqual(1.0, sum(policy), places=6)

    def test_cli_classic_mcts_teacher_writes_valid_jsonl(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-bootstrap-classic-") as tmp:
            out_path = Path(tmp) / "bootstrap.jsonl"

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/generate_bootstrap_dataset.py",
                    "--out",
                    str(out_path),
                    "--games",
                    "2",
                    "--simulations",
                    "8",
                    "--seed",
                    "42",
                    "--max-positions-per-game",
                    "4",
                    "--workers",
                    "1",
                    "--teacher-mode",
                    "classic_mcts",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertTrue(out_path.exists())
            rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(rows)
            for row in rows:
                self.assertIn("state", row)
                self.assertIn("policy", row)
                self.assertIn("value", row)
                self.assertAlmostEqual(1.0, sum(row["policy"]), places=5)
                self.assertGreaterEqual(row["value"], -1.0)
                self.assertLessEqual(row["value"], 1.0)

    def test_visits_from_classic_mcts_root_maps_children_to_array(self):
        from ml.alphazero_lite.classic_mcts import Node
        from ml.alphazero_lite.kalah_rules import KalahGame

        game = KalahGame.from_state({
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        })
        root = Node(game)
        child_a = Node(game, parent=root)
        child_a.visits = 5
        child_b = Node(game, parent=root)
        child_b.visits = 3
        root.children = {0: child_a, 2: child_b}

        result = generate_bootstrap_dataset.visits_from_classic_mcts_root(root)

        self.assertEqual(6, len(result))
        self.assertAlmostEqual(5.0, result[0])
        self.assertAlmostEqual(0.0, result[1])
        self.assertAlmostEqual(3.0, result[2])
        self.assertAlmostEqual(0.0, result[3])

    def test_value_from_classic_mcts_root_converts_win_rate(self):
        from ml.alphazero_lite.classic_mcts import Node
        from ml.alphazero_lite.kalah_rules import KalahGame

        game = KalahGame.from_state({
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        })
        root = Node(game)
        root.visits = 10
        root.wins = 7.0

        value = generate_bootstrap_dataset.value_from_classic_mcts_root(root)

        # 2*(7/10) - 1 = 0.4
        self.assertAlmostEqual(0.4, value, places=6)

    def test_value_from_classic_mcts_root_zero_visits(self):
        from ml.alphazero_lite.classic_mcts import Node
        from ml.alphazero_lite.kalah_rules import KalahGame

        game = KalahGame.from_state({
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        })
        root = Node(game)

        value = generate_bootstrap_dataset.value_from_classic_mcts_root(root)
        self.assertAlmostEqual(0.0, value, places=6)
