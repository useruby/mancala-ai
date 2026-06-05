import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ml.alphazero_lite import generate_bootstrap_dataset


class GenerateBootstrapDatasetTest(unittest.TestCase):
    def executable_python(self) -> str:
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / ".venv/bin/python",
            repo_root.parents[1] / ".venv/bin/python",
        ]
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        return sys.executable

    def test_cli_writes_jsonl_rows(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-bootstrap-") as tmp:
            out_path = Path(tmp) / "bootstrap.jsonl"

            result = subprocess.run(
                [
                    self.executable_python(),
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
            rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(rows)
            self.assertIn("state", rows[0])
            self.assertIn("policy", rows[0])
            self.assertIn("value", rows[0])

    def test_cli_emits_machine_readable_teacher_reuse_metrics(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-bootstrap-metrics-") as tmp:
            out_path = Path(tmp) / "bootstrap.jsonl"

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/generate_bootstrap_dataset.py",
                    "--out",
                    str(out_path),
                    "--games",
                    "1",
                    "--simulations",
                    "8",
                    "--seed",
                    "42",
                    "--max-positions-per-game",
                    "4",
                    "--workers",
                    "1",
                    "--position-selection-mode",
                    "hybrid_teacher",
                    "--teacher-search-reuse",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertRegex(result.stdout, r"teacher_search_reuse=true")
            self.assertRegex(
                result.stdout, r"dataset_metrics .*teacher_search_reuse=true"
            )

    def test_cli_reports_effective_teacher_search_reuse_false_for_classic_mcts_teacher(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(
            prefix="azlite-bootstrap-metrics-classic-"
        ) as tmp:
            out_path = Path(tmp) / "bootstrap.jsonl"

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/generate_bootstrap_dataset.py",
                    "--out",
                    str(out_path),
                    "--games",
                    "1",
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
                    "--teacher-search-reuse",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertRegex(result.stdout, r"teacher_search_reuse=false")
            self.assertRegex(
                result.stdout, r"dataset_metrics .*teacher_search_reuse=false"
            )

    def test_shape_policy_uses_true_dirichlet_sampler(self):
        fake_rng = mock.Mock()
        fake_rng.randint.return_value = 7
        fake_generator = mock.Mock()
        fake_generator.dirichlet.return_value = [0.2, 0.8]

        with mock.patch.object(
            generate_bootstrap_dataset.np.random,
            "default_rng",
            return_value=fake_generator,
        ):
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
                    self.executable_python(),
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
            rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
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

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
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

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        root = Node(game)
        root.visits = 10
        root.wins = 7.0

        value = generate_bootstrap_dataset.value_from_classic_mcts_root(root)

        # 2*(7/10) - 1 = 0.4
        self.assertAlmostEqual(0.4, value, places=6)

    def test_value_from_classic_mcts_root_zero_visits(self):
        from ml.alphazero_lite.classic_mcts import Node
        from ml.alphazero_lite.kalah_rules import KalahGame

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        root = Node(game)

        value = generate_bootstrap_dataset.value_from_classic_mcts_root(root)
        self.assertAlmostEqual(0.0, value, places=6)

    def test_run_worker_hybrid_teacher_reuses_shallow_root_when_enabled(self):
        created_searches = []
        shallow_root = mock.Mock(q_value=0.25)

        class TrackingPUCT:
            def __init__(
                self,
                *,
                evaluator,
                simulations,
                c_puct,
                rng,
                root=None,
                reuse_subtree=False,
                **kwargs,
            ):
                del evaluator, c_puct, rng, kwargs
                created_searches.append(
                    {
                        "simulations": simulations,
                        "root": root,
                        "reuse_subtree": reuse_subtree,
                    }
                )
                self._simulations = simulations
                self._root = root

            def run(self, game, *, dirichlet_alpha=None, dirichlet_epsilon=0.25):
                del game, dirichlet_alpha, dirichlet_epsilon
                if self._simulations == 6:
                    visits = np.array([6.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    return visits, shallow_root
                visits = np.array([0.0, 12.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                return visits, self._root

        with tempfile.TemporaryDirectory(prefix="azlite-bootstrap-worker-") as tmp:
            shard_path = Path(tmp) / "worker.jsonl"

            with (
                mock.patch.object(generate_bootstrap_dataset, "PUCT", TrackingPUCT),
                mock.patch.object(generate_bootstrap_dataset, "MAX_MOVES", 1),
            ):
                result_disabled = generate_bootstrap_dataset.run_worker(
                    worker_id=0,
                    start_index=0,
                    games=1,
                    seed=42,
                    simulations=6,
                    input_encoding="kalah_v1",
                    max_positions_per_game=1,
                    tree_reuse_enabled=False,
                    teacher_search_reuse=False,
                    position_selection_mode="hybrid_teacher",
                    policy_target_mode="visit_distribution",
                    value_target_mode="default",
                    tau=1.0,
                    top_k=None,
                    dirichlet_alpha=None,
                    dirichlet_epsilon=0.25,
                    dirichlet_opening_moves=0,
                    shard_path=str(shard_path),
                )

                result_enabled = generate_bootstrap_dataset.run_worker(
                    worker_id=0,
                    start_index=0,
                    games=1,
                    seed=42,
                    simulations=6,
                    input_encoding="kalah_v1",
                    max_positions_per_game=1,
                    tree_reuse_enabled=False,
                    teacher_search_reuse=True,
                    position_selection_mode="hybrid_teacher",
                    policy_target_mode="visit_distribution",
                    value_target_mode="default",
                    tau=1.0,
                    top_k=None,
                    dirichlet_alpha=None,
                    dirichlet_epsilon=0.25,
                    dirichlet_opening_moves=0,
                    shard_path=str(shard_path),
                )

        self.assertEqual(1, result_disabled["rows_written"])
        self.assertEqual(1, result_enabled["rows_written"])
        self.assertEqual(4, len(created_searches))
        self.assertIsNone(created_searches[1]["root"])
        self.assertFalse(created_searches[1]["reuse_subtree"])
        self.assertIsNotNone(created_searches[3]["root"])
        self.assertIsNot(shallow_root, created_searches[3]["root"])
        self.assertTrue(created_searches[3]["reuse_subtree"])

    def test_run_worker_teacher_search_reuse_treats_deeper_budget_as_total_target(self):
        created_searches = []
        shallow_root = mock.Mock(q_value=0.25)

        class TrackingPUCT:
            def __init__(
                self,
                *,
                evaluator,
                simulations,
                c_puct,
                rng,
                root=None,
                reuse_subtree=False,
                **kwargs,
            ):
                del evaluator, c_puct, rng, kwargs
                created_searches.append(
                    {
                        "simulations": simulations,
                        "root": root,
                        "reuse_subtree": reuse_subtree,
                    }
                )
                self._simulations = simulations
                self._root = root

            def run(self, game, *, dirichlet_alpha=None, dirichlet_epsilon=0.25):
                del game, dirichlet_alpha, dirichlet_epsilon
                if self._simulations == 6:
                    visits = np.array([6.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    return visits, shallow_root
                visits = np.array([0.0, 6.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                return visits, self._root

        with tempfile.TemporaryDirectory(prefix="azlite-bootstrap-worker-") as tmp:
            shard_path = Path(tmp) / "worker.jsonl"

            with (
                mock.patch.object(generate_bootstrap_dataset, "PUCT", TrackingPUCT),
                mock.patch.object(generate_bootstrap_dataset, "MAX_MOVES", 1),
            ):
                result_disabled = generate_bootstrap_dataset.run_worker(
                    worker_id=0,
                    start_index=0,
                    games=1,
                    seed=42,
                    simulations=6,
                    input_encoding="kalah_v1",
                    max_positions_per_game=1,
                    tree_reuse_enabled=False,
                    teacher_search_reuse=False,
                    position_selection_mode="hybrid_teacher",
                    policy_target_mode="visit_distribution",
                    value_target_mode="default",
                    tau=1.0,
                    top_k=None,
                    dirichlet_alpha=None,
                    dirichlet_epsilon=0.25,
                    dirichlet_opening_moves=0,
                    shard_path=str(shard_path),
                )

                result_enabled = generate_bootstrap_dataset.run_worker(
                    worker_id=0,
                    start_index=0,
                    games=1,
                    seed=42,
                    simulations=6,
                    input_encoding="kalah_v1",
                    max_positions_per_game=1,
                    tree_reuse_enabled=False,
                    teacher_search_reuse=True,
                    position_selection_mode="hybrid_teacher",
                    policy_target_mode="visit_distribution",
                    value_target_mode="default",
                    tau=1.0,
                    top_k=None,
                    dirichlet_alpha=None,
                    dirichlet_epsilon=0.25,
                    dirichlet_opening_moves=0,
                    shard_path=str(shard_path),
                )

        self.assertEqual(1, result_disabled["rows_written"])
        self.assertEqual(1, result_enabled["rows_written"])
        self.assertEqual(4, len(created_searches))
        self.assertEqual(12, created_searches[1]["simulations"])
        self.assertEqual(6, created_searches[3]["simulations"])
        self.assertIsNone(created_searches[1]["root"])
        self.assertIsNotNone(created_searches[3]["root"])
        self.assertIsNot(shallow_root, created_searches[3]["root"])

    def test_run_worker_teacher_search_reuse_preserves_shallow_root_value(self):
        captured_positions = []

        class MutableRoot:
            def __init__(self, q_value):
                self.q_value = q_value

        shallow_values_seen = []
        shallow_root = MutableRoot(0.25)

        class MutatingPUCT:
            def __init__(
                self,
                *,
                evaluator,
                simulations,
                c_puct,
                rng,
                root=None,
                reuse_subtree=False,
                **kwargs,
            ):
                del evaluator, c_puct, rng, reuse_subtree, kwargs
                self._simulations = simulations
                self._root = root

            def run(self, game, *, dirichlet_alpha=None, dirichlet_epsilon=0.25):
                del game, dirichlet_alpha, dirichlet_epsilon
                if self._root is None:
                    shallow_values_seen.append(shallow_root.q_value)
                    return np.array(
                        [6.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
                    ), shallow_root

                self._root.q_value = 0.75
                return np.array(
                    [0.0, 6.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
                ), self._root

        real_annotate_rows = generate_bootstrap_dataset.annotate_rows

        def capture_annotate_rows(positions, **kwargs):
            captured_positions.extend(positions)
            return real_annotate_rows(positions, **kwargs)

        with tempfile.TemporaryDirectory(prefix="azlite-bootstrap-worker-") as tmp:
            shard_path = Path(tmp) / "worker.jsonl"

            with (
                mock.patch.object(generate_bootstrap_dataset, "PUCT", MutatingPUCT),
                mock.patch.object(generate_bootstrap_dataset, "MAX_MOVES", 1),
                mock.patch.object(
                    generate_bootstrap_dataset,
                    "annotate_rows",
                    side_effect=capture_annotate_rows,
                ),
            ):
                result = generate_bootstrap_dataset.run_worker(
                    worker_id=0,
                    start_index=0,
                    games=1,
                    seed=42,
                    simulations=6,
                    input_encoding="kalah_v1",
                    max_positions_per_game=1,
                    tree_reuse_enabled=False,
                    teacher_search_reuse=True,
                    position_selection_mode="hybrid_teacher",
                    policy_target_mode="visit_distribution",
                    value_target_mode="hybrid",
                    tau=1.0,
                    top_k=None,
                    dirichlet_alpha=None,
                    dirichlet_epsilon=0.25,
                    dirichlet_opening_moves=0,
                    shard_path=str(shard_path),
                )

            rows = [
                json.loads(line)
                for line in shard_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(1, result["rows_written"])
        self.assertEqual(1, len(rows))
        self.assertEqual(1, len(captured_positions))
        self.assertEqual(1, len(shallow_values_seen))
        self.assertAlmostEqual(0.25, shallow_values_seen[0], places=6)
        self.assertAlmostEqual(
            0.25, captured_positions[0]["root_search_value"], places=6
        )

    def test_run_worker_teacher_search_reuse_does_not_leak_into_next_real_reusable_root(
        self,
    ):
        created_searches = []
        chosen_child_marks = []

        class FakeRoot:
            def __init__(
                self,
                name=None,
                *,
                game=None,
                prior=0.0,
                visit_count=0,
                value_sum=0.0,
                expanded=True,
            ):
                self.name = name or "cloned"
                self.q_value = 0.0
                self.children = {}
                self.teacher_marks = []
                self.prior = prior
                self.visit_count = visit_count
                self.value_sum = value_sum
                self.expanded = expanded
                self.game = game or mock.Mock()
                if not hasattr(self.game, "clone"):
                    cloned_game = mock.Mock()
                    self.game.clone = mock.Mock(return_value=cloned_game)

            def child_for_action(self, move):
                return self.children.get(move)

        shallow_root = FakeRoot("shallow_root")
        shallow_child = FakeRoot("shallow_child")
        teacher_child = FakeRoot("teacher_child")
        shallow_root.children[0] = shallow_child
        shallow_child.children[0] = teacher_child

        class TrackingPUCT:
            def __init__(
                self,
                *,
                evaluator,
                simulations,
                c_puct,
                rng,
                root=None,
                reuse_subtree=False,
                **kwargs,
            ):
                del evaluator, c_puct, rng, kwargs
                created_searches.append(
                    {
                        "simulations": simulations,
                        "root": root,
                        "reuse_subtree": reuse_subtree,
                    }
                )
                self._root = root

            def run(self, game, *, dirichlet_alpha=None, dirichlet_epsilon=0.25):
                del game, dirichlet_alpha, dirichlet_epsilon
                if self._root is None:
                    return np.array(
                        [6.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
                    ), shallow_root

                chosen_child = self._root.child_for_action(0)
                chosen_child.teacher_marks.append("teacher")
                return np.array(
                    [6.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
                ), self._root

        real_annotate_rows = generate_bootstrap_dataset.annotate_rows

        def capture_annotate_rows(positions, **kwargs):
            for position in positions:
                chosen_child_marks.append(len(shallow_child.teacher_marks))
            return real_annotate_rows(positions, **kwargs)

        with tempfile.TemporaryDirectory(prefix="azlite-bootstrap-worker-") as tmp:
            shard_path = Path(tmp) / "worker.jsonl"

            with (
                mock.patch.object(generate_bootstrap_dataset, "PUCT", TrackingPUCT),
                mock.patch.object(generate_bootstrap_dataset, "MAX_MOVES", 2),
                mock.patch.object(
                    generate_bootstrap_dataset,
                    "annotate_rows",
                    side_effect=capture_annotate_rows,
                ),
            ):
                result = generate_bootstrap_dataset.run_worker(
                    worker_id=0,
                    start_index=0,
                    games=1,
                    seed=42,
                    simulations=6,
                    input_encoding="kalah_v1",
                    max_positions_per_game=2,
                    tree_reuse_enabled=True,
                    teacher_search_reuse=True,
                    position_selection_mode="hybrid_teacher",
                    policy_target_mode="visit_distribution",
                    value_target_mode="default",
                    tau=1.0,
                    top_k=None,
                    dirichlet_alpha=None,
                    dirichlet_epsilon=0.25,
                    dirichlet_opening_moves=0,
                    shard_path=str(shard_path),
                )

        self.assertEqual(2, result["rows_written"])
        self.assertEqual(4, len(created_searches))
        self.assertIsNotNone(created_searches[1]["root"])
        self.assertIsNot(shallow_root, created_searches[1]["root"])
        self.assertIs(shallow_child, created_searches[2]["root"])
        self.assertEqual([0, 0], chosen_child_marks)
        self.assertEqual([], shallow_child.teacher_marks)

    def test_cli_teacher_search_reuse_keeps_hybrid_teacher_output_deterministic(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(
            prefix="azlite-bootstrap-hybrid-reuse-"
        ) as tmp:
            first_out = Path(tmp) / "bootstrap-first.jsonl"
            second_out = Path(tmp) / "bootstrap-second.jsonl"
            command = [
                self.executable_python(),
                "ml/alphazero_lite/generate_bootstrap_dataset.py",
                "--out",
                str(first_out),
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
                "--position-selection-mode",
                "hybrid_teacher",
                "--teacher-search-reuse",
            ]

            first = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, first.returncode, msg=first.stderr)

            second_command = command.copy()
            second_command[3] = str(second_out)
            second = subprocess.run(
                second_command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, second.returncode, msg=second.stderr)

            first_rows = [
                json.loads(line)
                for line in first_out.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            second_rows = [
                json.loads(line)
                for line in second_out.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertTrue(first_rows)
        self.assertEqual(first_rows, second_rows)
        for row in first_rows:
            self.assertEqual("hybrid_teacher", row["position_selection_mode"])
            self.assertIn(row["teacher_bucket"], {"tactical", "disagreement", "both"})
            self.assertAlmostEqual(1.0, sum(row["policy"]), places=5)
            self.assertGreaterEqual(row["value"], -1.0)
            self.assertLessEqual(row["value"], 1.0)

    def _sample_position(self):
        return {
            "state": [0.1] * 15,
            "game_state": {
                "player_pits": [0, 0, 0, 0, 0, 1],
                "opponent_pits": [0, 0, 0, 0, 0, 0],
                "player_store": 22,
                "opponent_store": 19,
                "current_player": 0,
            },
            "player": 0,
            "move_index": 4,
            "policy": [0.2, 0.2, 0.2, 0.1, 0.1, 0.2],
            "root_search_value": 0.5,
            "policy_target_mode": "sharpened",
        }

    def test_tablebase_overlay_off_produces_no_metadata(self):
        from ml.alphazero_lite.endgame_tablebase import EndgameTablebase

        position = self._sample_position()
        tablebase = EndgameTablebase()
        rows, tb_stats = generate_bootstrap_dataset.annotate_rows(
            [position],
            winner=0,
            value_target_mode="default",
            position_selection_mode="generic",
            tablebase=tablebase,
            tablebase_value_overlay="off",
        )
        self.assertEqual(1, len(rows))
        self.assertEqual(0, tb_stats["tb_rows"])
        self.assertNotIn("tablebase_value_applied", rows[0])
        self.assertNotIn("tablebase_value", rows[0])
        self.assertNotIn("original_value", rows[0])
        self.assertNotIn("tablebase_value_overlay", rows[0])

    def test_tablebase_overlay_overwrite_changes_value_when_solved(self):
        from ml.alphazero_lite.endgame_tablebase import EndgameTablebase

        position = self._sample_position()
        tablebase = EndgameTablebase()
        rows_off, _ = generate_bootstrap_dataset.annotate_rows(
            [position],
            winner=0,
            value_target_mode="default",
            position_selection_mode="generic",
            tablebase=tablebase,
            tablebase_value_overlay="off",
        )
        original_value = rows_off[0]["value"]

        rows_over, tb_stats = generate_bootstrap_dataset.annotate_rows(
            [position],
            winner=0,
            value_target_mode="default",
            position_selection_mode="generic",
            tablebase=tablebase,
            tablebase_value_overlay="overwrite",
        )
        self.assertEqual(1, tb_stats["tb_rows"])
        self.assertTrue(rows_over[0]["tablebase_value_applied"])
        self.assertIsNotNone(rows_over[0]["tablebase_value"])
        self.assertEqual(original_value, rows_over[0]["original_value"])
        self.assertEqual("overwrite", rows_over[0]["tablebase_value_overlay"])
        self.assertAlmostEqual(rows_over[0]["tablebase_value"], rows_over[0]["value"])
        self.assertGreaterEqual(rows_over[0]["value"], -1.0)
        self.assertLessEqual(rows_over[0]["value"], 1.0)

    def test_tablebase_overlay_skips_unsolved_position(self):
        from ml.alphazero_lite.endgame_tablebase import EndgameTablebase

        position = {
            "state": [0.1] * 15,
            "game_state": {
                "player_pits": [3, 3, 3, 3, 3, 3],
                "opponent_pits": [3, 3, 3, 3, 3, 3],
                "player_store": 10,
                "opponent_store": 10,
                "current_player": 0,
            },
            "player": 0,
            "move_index": 4,
            "policy": [0.2, 0.2, 0.2, 0.1, 0.1, 0.2],
            "root_search_value": 0.0,
            "policy_target_mode": "sharpened",
        }
        tablebase = EndgameTablebase()
        rows, tb_stats = generate_bootstrap_dataset.annotate_rows(
            [position],
            winner=None,
            value_target_mode="default",
            position_selection_mode="generic",
            tablebase=tablebase,
            tablebase_value_overlay="overwrite",
        )
        self.assertEqual(1, len(rows))
        self.assertEqual(0, tb_stats["tb_rows"])
        self.assertNotIn("tablebase_value_applied", rows[0])

    def test_tablebase_overlay_blend_correct_convex_combination(self):
        from ml.alphazero_lite.endgame_tablebase import EndgameTablebase

        position = self._sample_position()
        tablebase = EndgameTablebase()
        rows_off, _ = generate_bootstrap_dataset.annotate_rows(
            [position],
            winner=0,
            value_target_mode="default",
            position_selection_mode="generic",
            tablebase=tablebase,
            tablebase_value_overlay="off",
        )
        original_value = rows_off[0]["value"]

        rows_blend, tb_stats = generate_bootstrap_dataset.annotate_rows(
            [position],
            winner=0,
            value_target_mode="default",
            position_selection_mode="generic",
            tablebase=tablebase,
            tablebase_value_overlay="blend",
            tablebase_blend_alpha=0.5,
        )
        self.assertEqual(1, tb_stats["tb_rows"])
        self.assertTrue(rows_blend[0]["tablebase_value_applied"])
        exact_value = rows_blend[0]["tablebase_value"]
        expected = 0.5 * exact_value + 0.5 * original_value
        self.assertAlmostEqual(expected, rows_blend[0]["value"], places=6)
        self.assertEqual("blend", rows_blend[0]["tablebase_value_overlay"])

    def test_tablebase_overlay_preserves_policy(self):
        from ml.alphazero_lite.endgame_tablebase import EndgameTablebase

        position = self._sample_position()
        tablebase = EndgameTablebase()
        rows_off, _ = generate_bootstrap_dataset.annotate_rows(
            [position],
            winner=0,
            value_target_mode="default",
            position_selection_mode="generic",
            tablebase=tablebase,
            tablebase_value_overlay="off",
        )
        rows_over, _ = generate_bootstrap_dataset.annotate_rows(
            [position],
            winner=0,
            value_target_mode="default",
            position_selection_mode="generic",
            tablebase=tablebase,
            tablebase_value_overlay="overwrite",
        )
        rows_blend, _ = generate_bootstrap_dataset.annotate_rows(
            [position],
            winner=0,
            value_target_mode="default",
            position_selection_mode="generic",
            tablebase=tablebase,
            tablebase_value_overlay="blend",
            tablebase_blend_alpha=0.5,
        )
        self.assertAlmostEqual(
            sum(rows_off[0]["policy"]), sum(rows_over[0]["policy"]), places=6
        )
        self.assertEqual(rows_off[0]["policy"], rows_over[0]["policy"])
        self.assertEqual(rows_off[0]["policy"], rows_blend[0]["policy"])
        self.assertEqual(rows_off[0]["move_index"], rows_over[0]["move_index"])
        self.assertEqual(rows_off[0]["player"], rows_over[0]["player"])

    def test_train_loads_rows_with_tablebase_metadata(self):
        from ml.alphazero_lite.train import load_jsonl

        row = {
            "move_index": 4,
            "player": 0,
            "state": [0.1] * 15,
            "policy": [0.2, 0.2, 0.2, 0.1, 0.1, 0.2],
            "policy_target_mode": "default",
            "value_target_mode": "default",
            "value": 0.75,
            "tablebase_value_applied": True,
            "tablebase_value": 0.75,
            "original_value": 0.5,
            "tablebase_value_overlay": "overwrite",
        }
        with tempfile.TemporaryDirectory(prefix="azlite-tb-train-") as tmp:
            path = Path(tmp) / "test.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            x, p, v = load_jsonl(
                path, policy_target_mode="default", value_target_mode="default"
            )
            self.assertEqual(1, x.shape[0])
            self.assertEqual(1, p.shape[0])
            self.assertAlmostEqual(0.75, float(v[0][0]), places=6)

    def test_cli_tablebase_overlay_off_no_summary_printed(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-tb-off-") as tmp:
            out_path = Path(tmp) / "bootstrap.jsonl"
            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/generate_bootstrap_dataset.py",
                    "--out",
                    str(out_path),
                    "--games",
                    "1",
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
                    "--tablebase-value-overlay",
                    "off",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertNotIn("tablebase_value_overlay_summary", result.stdout)
            self.assertTrue(out_path.exists())
            rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for row in rows:
                self.assertNotIn("tablebase_value_applied", row)
                self.assertNotIn("tablebase_value", row)

    def test_cli_tablebase_overlay_overwrite_prints_summary(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-tb-overwrite-") as tmp:
            out_path = Path(tmp) / "bootstrap.jsonl"
            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/generate_bootstrap_dataset.py",
                    "--out",
                    str(out_path),
                    "--games",
                    "2",
                    "--simulations",
                    "16",
                    "--seed",
                    "42",
                    "--max-positions-per-game",
                    "4",
                    "--workers",
                    "1",
                    "--teacher-mode",
                    "classic_mcts",
                    "--tablebase-value-overlay",
                    "overwrite",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertIn("tablebase_value_overlay_summary", result.stdout)
            self.assertIn("coverage_rate=", result.stdout)
            self.assertIn("coverage_early=", result.stdout)
            self.assertIn("coverage_mid=", result.stdout)
            self.assertIn("coverage_late=", result.stdout)

    def test_cli_tablebase_overlay_blend_prints_summary(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-tb-blend-") as tmp:
            out_path = Path(tmp) / "bootstrap.jsonl"
            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/generate_bootstrap_dataset.py",
                    "--out",
                    str(out_path),
                    "--games",
                    "2",
                    "--simulations",
                    "16",
                    "--seed",
                    "42",
                    "--max-positions-per-game",
                    "4",
                    "--workers",
                    "1",
                    "--teacher-mode",
                    "classic_mcts",
                    "--tablebase-value-overlay",
                    "blend",
                    "--tablebase-blend-alpha",
                    "0.5",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertIn("tablebase_value_overlay_summary", result.stdout)
            self.assertIn("overlay=blend", result.stdout)

    def test_cli_tablebase_overlay_overwrite_produces_metadata_on_solved_rows(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-tb-metadata-") as tmp:
            out_path = Path(tmp) / "bootstrap.jsonl"
            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/generate_bootstrap_dataset.py",
                    "--out",
                    str(out_path),
                    "--games",
                    "4",
                    "--simulations",
                    "32",
                    "--seed",
                    "42",
                    "--max-positions-per-game",
                    "8",
                    "--workers",
                    "1",
                    "--teacher-mode",
                    "classic_mcts",
                    "--tablebase-value-overlay",
                    "overwrite",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(rows)
            tb_rows = [row for row in rows if row.get("tablebase_value_applied")]
            for row in tb_rows:
                self.assertIn("tablebase_value", row)
                self.assertIn("original_value", row)
                self.assertEqual("overwrite", row["tablebase_value_overlay"])
                self.assertAlmostEqual(row["value"], row["tablebase_value"])
            self.assertTrue(tb_rows, "expected at least some tb-covered rows")
