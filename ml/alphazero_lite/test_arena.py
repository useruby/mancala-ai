import json
import random
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ml.alphazero_lite import arena
from ml.alphazero_lite import self_play
from ml.alphazero_lite.kalah_rules import KalahGame


class ArenaScriptTest(unittest.TestCase):
    def test_deterministic_root_policy_breaks_tied_root_visits_using_search_signal(self):
        root = arena.PUCT(
            evaluator=arena.ArtifactEvaluator,
            simulations=1,
            c_puct=1.25,
            rng=random.Random(7),
        )._root_for(
            KalahGame.from_state(
                {
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                }
            )
        )
        root.children[0] = self_play.Node(game=self_play.KalahGame.from_state(root.game.to_state()), prior=0.2, visit_count=5, value_sum=1.0)
        root.children[1] = self_play.Node(game=self_play.KalahGame.from_state(root.game.to_state()), prior=0.6, visit_count=5, value_sum=3.0)

        visit_count_search = self_play.PUCT(
            evaluator=self_play.HeuristicEvaluator(),
            simulations=1,
            c_puct=1.25,
            rng=random.Random(7),
            root_policy_mode="visit_count",
        )
        deterministic_search = self_play.PUCT(
            evaluator=self_play.HeuristicEvaluator(),
            simulations=1,
            c_puct=1.25,
            rng=random.Random(7),
            root_policy_mode="deterministic",
        )

        self.assertEqual(0, visit_count_search.select_root_move(root, [0, 1]))
        self.assertEqual(1, deterministic_search.select_root_move(root, [0, 1]))

    def test_tactical_root_bias_only_adjusts_immediate_tactical_moves(self):
        game = KalahGame.from_state(
            {
                "player_pits": [1, 0, 0, 0, 0, 1],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = arena.PUCT(
            evaluator=arena.ArtifactEvaluator,
            simulations=1,
            c_puct=1.25,
            rng=random.Random(7),
            tactical_root_bias=0.1,
        )
        base_priors = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.45], dtype=np.float32)

        biased_priors = search.apply_tactical_root_bias(game, base_priors)

        self.assertGreater(biased_priors[5], base_priors[5])
        self.assertLess(biased_priors[0], base_priors[0])
        self.assertEqual(0.0, biased_priors[1])
        self.assertAlmostEqual(1.0, float(np.sum(biased_priors)), places=5)

    def test_run_arena_worker_uses_eval_search_defaults(self):
        captured_search_options = []

        class FakeArtifactEvaluator:
            def __init__(self, _artifact_dir):
                pass

        class FakePUCT:
            def __init__(self, *, evaluator, simulations, c_puct, rng, root=None, **search_options):
                del evaluator, simulations, c_puct, rng, root
                captured_search_options.append(search_options)

            def run(self, game):
                del game
                visits = np.zeros(6, dtype=np.float32)
                visits[0] = 1.0
                return visits, None

        with mock.patch("ml.alphazero_lite.arena.ArtifactEvaluator", FakeArtifactEvaluator), mock.patch(
            "ml.alphazero_lite.arena.PUCT", FakePUCT
        ):
            result = arena.run_arena_worker(
                worker_id=0,
                start_index=0,
                games=1,
                challenger_path="challenger",
                current_path="current",
                challenger_simulations=32,
                current_simulations=16,
                seed=42,
                c_puct=1.25,
                max_moves=1,
            )

        self.assertEqual("deterministic", result["search_options"]["root_policy_mode"])
        self.assertEqual(0.1, result["search_options"]["tactical_root_bias"])
        self.assertEqual([result["search_options"]], captured_search_options)

    def test_artifact_evaluator_loads_residual_v2_weights_and_selected_input_encoding(self):
        with tempfile.TemporaryDirectory(prefix="azlite-arena-unit-") as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "input_encoding": "kalah_v2",
                        "architecture": {
                            "model_type": "residual_v2",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (artifact_dir / "weights.json").write_text(
                json.dumps(
                    {
                        "w_input": [[0.5, 0.0], [0.0, 0.5]] + [[0.0, 0.0]] * 13,
                        "b_input": [0.0, 0.0],
                        "w_residual_1_1": [[1.0, 0.0], [0.0, 1.0]],
                        "b_residual_1_1": [0.0, 0.0],
                        "w_residual_1_2": [[1.0, 0.0], [0.0, 1.0]],
                        "b_residual_1_2": [0.0, 0.0],
                        "w_policy": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
                        "b_policy": [0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
                        "w_value": [[0.25], [0.25]],
                        "b_value": [0.0],
                    }
                ),
                encoding="utf-8",
            )

            evaluator = arena.ArtifactEvaluator(artifact_dir)
            game = KalahGame.from_state(
                {
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                }
            )
            encoded_state = [0.2] * 15

            with mock.patch("ml.alphazero_lite.arena.encode_state", return_value=encoded_state) as encode_state:
                priors, value = evaluator.evaluate(game)

            self.assertEqual("kalah_v2", evaluator.input_encoding)
            self.assertEqual("residual_v2", evaluator.model_type)
            self.assertEqual((6,), priors.shape)
            self.assertAlmostEqual(1.0, float(np.sum(priors)), places=5)
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)
            encode_state.assert_called_once_with(game.to_state(), input_encoding="kalah_v2")

    def test_artifact_evaluator_uses_specialized_residual_v3_heads(self):
        with tempfile.TemporaryDirectory(prefix="azlite-arena-unit-") as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "input_encoding": "kalah_v3",
                        "architecture": {
                            "model_type": "residual_v3",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (artifact_dir / "weights.json").write_text(
                json.dumps(
                    {
                        "w_input": [[1.0, 0.0], [0.0, 1.0]],
                        "b_input": [0.0, 0.0],
                        "w_residual_1_1": [[0.0, 0.0], [0.0, 0.0]],
                        "b_residual_1_1": [0.0, 0.0],
                        "w_residual_1_2": [[0.0, 0.0], [0.0, 0.0]],
                        "b_residual_1_2": [0.0, 0.0],
                        "w_policy_hidden": [[2.0, 0.0, 1.0], [0.0, 3.0, 0.0]],
                        "b_policy_hidden": [0.0, 0.0, 0.0],
                        "w_policy": [
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        ],
                        "b_policy": [0.0, 0.0, 0.0, -10.0, -10.0, -10.0],
                        "w_value_hidden": [[1.0], [1.0]],
                        "b_value_hidden": [0.0],
                        "w_value": [[0.5]],
                        "b_value": [0.0],
                    }
                ),
                encoding="utf-8",
            )

            evaluator = arena.ArtifactEvaluator(artifact_dir)
            game = KalahGame.from_state(
                {
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                }
            )

            with mock.patch("ml.alphazero_lite.arena.encode_state", return_value=[1.0, 2.0]):
                priors, value = evaluator.evaluate(game)

            expected_logits = np.array([2.0, 6.0, 1.0, -10.0, -10.0, -10.0], dtype=np.float32)
            expected_logits = expected_logits - np.max(expected_logits)
            expected_priors = np.exp(expected_logits)
            expected_priors /= np.sum(expected_priors)

            np.testing.assert_allclose(priors, expected_priors, rtol=1e-6, atol=1e-6)
            self.assertAlmostEqual(float(np.tanh(1.5)), value, places=6)

    def test_run_arena_worker_preserves_search_options(self):
        captured_search_options = []

        class FakeArtifactEvaluator:
            def __init__(self, _artifact_dir):
                pass

        class FakePUCT:
            def __init__(self, *, evaluator, simulations, c_puct, rng, root=None, **search_options):
                del evaluator, simulations, c_puct, rng, root
                captured_search_options.append(search_options)

            def run(self, game):
                del game
                visits = np.zeros(6, dtype=np.float32)
                visits[0] = 1.0
                return visits, None

        with mock.patch("ml.alphazero_lite.arena.ArtifactEvaluator", FakeArtifactEvaluator), mock.patch(
            "ml.alphazero_lite.arena.PUCT", FakePUCT
        ):
            result = arena.run_arena_worker(
                worker_id=0,
                start_index=0,
                games=1,
                challenger_path="challenger",
                current_path="current",
                challenger_simulations=32,
                current_simulations=16,
                seed=42,
                c_puct=1.25,
                max_moves=1,
                fpu_mode="parent_q",
                reuse_subtree=True,
                normalize_values=True,
                root_policy_mode="deterministic",
                tactical_root_bias=0.2,
            )

        self.assertEqual(
            {
                "fpu_mode": "parent_q",
                "reuse_subtree": True,
                "normalize_values": True,
                "root_policy_mode": "deterministic",
                "tactical_root_bias": 0.2,
            },
            result["search_options"],
        )
        self.assertEqual([result["search_options"]], captured_search_options)

    def test_run_arena_worker_reuses_selected_child_between_moves_when_enabled(self):
        created_roots = []

        class FakeArtifactEvaluator:
            def __init__(self, _artifact_dir):
                pass

        class FakeGame:
            def __init__(self):
                self.moves_played = 0
                self.current_player = 0
                self.captured_seeds = [0, 0]

            def over(self):
                return self.moves_played >= 2

            def possible_moves(self):
                return [0] if not self.over() else []

            def pit_index(self, move):
                return move

            def move(self, absolute_move):
                self.moves_played += 1
                self.current_player = 0
                return absolute_move == 0

        class FakeRoot:
            def __init__(self, child=None):
                self.child = child

            def child_for_action(self, action):
                if action != 0:
                    raise AssertionError(f"unexpected action {action}")
                return self.child

        second_root = FakeRoot()
        first_root = FakeRoot(child=second_root)
        roots_to_return = [first_root, second_root]

        class FakePUCT:
            def __init__(self, *, evaluator, simulations, c_puct, rng, root=None, **search_options):
                del evaluator, simulations, c_puct, rng, search_options
                created_roots.append(root)

            def run(self, game):
                del game
                root = roots_to_return[len(created_roots) - 1]
                visits = np.zeros(6, dtype=np.float32)
                visits[0] = 1.0
                return visits, root

        with mock.patch("ml.alphazero_lite.arena.ArtifactEvaluator", FakeArtifactEvaluator), mock.patch(
            "ml.alphazero_lite.arena.PUCT", FakePUCT
        ), mock.patch("ml.alphazero_lite.arena.KalahGame.from_state", return_value=FakeGame()):
            arena.run_arena_worker(
                worker_id=0,
                start_index=0,
                games=1,
                challenger_path="challenger",
                current_path="current",
                challenger_simulations=32,
                current_simulations=16,
                seed=42,
                c_puct=1.25,
                max_moves=4,
                reuse_subtree=True,
            )

        self.assertEqual([None, second_root], created_roots)

    def test_run_arena_worker_does_not_reuse_subtree_across_evaluator_switches(self):
        created = []

        class FakeArtifactEvaluator:
            def __init__(self, artifact_dir):
                self.name = str(artifact_dir)

        class FakeGame:
            def __init__(self):
                self.moves_played = 0
                self.current_player = 0
                self.captured_seeds = [0, 0]

            def over(self):
                return self.moves_played >= 2

            def possible_moves(self):
                return [0] if not self.over() else []

            def pit_index(self, move):
                return move

            def move(self, absolute_move):
                self.moves_played += 1
                self.current_player = self.moves_played % 2
                return absolute_move == 0

        class FakeRoot:
            def __init__(self, child=None):
                self.child = child

            def child_for_action(self, action):
                if action != 0:
                    raise AssertionError(f"unexpected action {action}")
                return self.child

        first_root = FakeRoot(child=FakeRoot())
        second_root = FakeRoot()
        roots_to_return = [first_root, second_root]

        class FakePUCT:
            def __init__(self, *, evaluator, simulations, c_puct, rng, root=None, **search_options):
                del simulations, c_puct, rng, search_options
                created.append((evaluator.name, root))

            def run(self, game):
                del game
                root = roots_to_return[len(created) - 1]
                visits = np.zeros(6, dtype=np.float32)
                visits[0] = 1.0
                return visits, root

        with mock.patch("ml.alphazero_lite.arena.ArtifactEvaluator", FakeArtifactEvaluator), mock.patch(
            "ml.alphazero_lite.arena.PUCT", FakePUCT
        ), mock.patch("ml.alphazero_lite.arena.KalahGame.from_state", return_value=FakeGame()):
            arena.run_arena_worker(
                worker_id=0,
                start_index=0,
                games=1,
                challenger_path="challenger",
                current_path="current",
                challenger_simulations=32,
                current_simulations=16,
                seed=42,
                c_puct=1.25,
                max_moves=4,
                reuse_subtree=True,
            )

        self.assertEqual([("challenger", None), ("current", None)], created)

    def test_cli_generates_validator_compatible_arena_report(self):
        with tempfile.TemporaryDirectory(prefix="azlite-arena-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.jsonl"
            checkpoint_path = tmp_path / "checkpoint.npz"
            challenger_dir = tmp_path / "challenger"
            current_dir = tmp_path / "current"
            out_path = tmp_path / "arena_report.json"

            row = {
                "state": [0.1] * 15,
                "policy": [0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
                "value": 1.0,
            }
            data_path.write_text("\n".join([json.dumps(row) for _ in range(64)]) + "\n", encoding="utf-8")

            train = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/train.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(checkpoint_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "32",
                    "--device",
                    "cpu",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, train.returncode, msg=train.stderr)

            for out_dir, version in ((challenger_dir, "azlite-challenger"), (current_dir, "azlite-current")):
                export = subprocess.run(
                    [
                        ".venv/bin/python",
                        "ml/alphazero_lite/export_artifact.py",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--out-dir",
                        str(out_dir),
                        "--version",
                        version,
                    ],
                    cwd=Path(__file__).resolve().parents[2],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(0, export.returncode, msg=export.stderr)

            arena = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/arena.py",
                    "--challenger",
                    str(challenger_dir),
                    "--current",
                    str(current_dir),
                    "--games",
                    "6",
                    "--challenger-simulations",
                    "32",
                    "--current-simulations",
                    "32",
                    "--seed",
                    "42",
                    "--out",
                    str(out_path),
                    "--min-score",
                    "0.55",
                    "--workers",
                    "2",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, arena.returncode, msg=arena.stderr)
            self.assertTrue(out_path.exists())

            report = json.loads(out_path.read_text(encoding="utf-8"))
            validate = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/validate_arena_report.py",
                    "--report",
                    str(out_path),
                    "--min-score",
                    "0.55",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, validate.returncode, msg=validate.stderr)
            self.assertIn("arena_report_valid", validate.stdout)

    def test_cli_records_multi_worker_distribution_when_workers_requested(self):
        with tempfile.TemporaryDirectory(prefix="azlite-arena-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.jsonl"
            checkpoint_path = tmp_path / "checkpoint.npz"
            challenger_dir = tmp_path / "challenger"
            current_dir = tmp_path / "current"
            out_path = tmp_path / "arena_report.json"

            row = {
                "state": [0.1] * 15,
                "policy": [0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
                "value": 1.0,
            }
            data_path.write_text("\n".join([json.dumps(row) for _ in range(64)]) + "\n", encoding="utf-8")

            train = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/train.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(checkpoint_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "32",
                    "--device",
                    "cpu",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, train.returncode, msg=train.stderr)

            for out_dir, version in ((challenger_dir, "azlite-challenger"), (current_dir, "azlite-current")):
                export = subprocess.run(
                    [
                        ".venv/bin/python",
                        "ml/alphazero_lite/export_artifact.py",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--out-dir",
                        str(out_dir),
                        "--version",
                        version,
                    ],
                    cwd=Path(__file__).resolve().parents[2],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(0, export.returncode, msg=export.stderr)

            arena = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/arena.py",
                    "--challenger",
                    str(challenger_dir),
                    "--current",
                    str(current_dir),
                    "--games",
                    "5",
                    "--challenger-simulations",
                    "16",
                    "--current-simulations",
                    "16",
                    "--seed",
                    "42",
                    "--out",
                    str(out_path),
                    "--min-score",
                    "0.55",
                    "--workers",
                    "2",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, arena.returncode, msg=arena.stderr)

            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(2, report["notes"]["workers_requested"])
            self.assertEqual(2, report["notes"]["workers_used"])
            self.assertEqual([3, 2], report["notes"]["worker_game_counts"])


if __name__ == "__main__":
    unittest.main()
