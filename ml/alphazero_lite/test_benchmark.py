import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ml.alphazero_lite import self_play


class BenchmarkScriptTest(unittest.TestCase):
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

    def test_executable_python_skips_non_executable_candidates(self):
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / ".venv/bin/python",
            repo_root.parents[1] / ".venv/bin/python",
        ]
        executable_fallback = candidates[1]

        def fake_is_file(self):
            return self in candidates

        def fake_access(path, mode):
            return path == executable_fallback and mode == os.X_OK

        with mock.patch.object(Path, "is_file", fake_is_file), mock.patch("os.access", side_effect=fake_access):
            self.assertEqual(str(executable_fallback), self.executable_python())

    def test_executable_python_falls_back_to_sys_executable(self):
        expected = __import__("sys").executable

        with mock.patch.object(Path, "is_file", return_value=False):
            self.assertEqual(expected, self.executable_python())

    def seeded_game(self):
        return self_play.KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )

    def test_parent_value_fpu_uses_parent_q_for_unvisited_children(self):
        parent = self_play.Node(game=self.seeded_game(), visit_count=1, value_sum=0.6)
        child = self_play.Node(game=self.seeded_game(), prior=0.2)

        search = self_play.PUCT(
            evaluator=self_play.HeuristicEvaluator(),
            simulations=1,
            c_puct=1.25,
            rng=random.Random(7),
            fpu_mode="parent_value",
        )

        self.assertAlmostEqual(0.6, search._child_q_value(parent, child))

    def test_parent_value_fpu_selection_prefers_unvisited_child_using_parent_q(self):
        root = self_play.Node(game=self.seeded_game(), visit_count=4, value_sum=2.4)
        visited_child = self_play.Node(game=self.seeded_game(), prior=0.1, visit_count=3, value_sum=0.3)
        unvisited_child = self_play.Node(game=self.seeded_game(), prior=0.7)
        root.children[0] = visited_child
        root.children[1] = unvisited_child

        search = self_play.PUCT(
            evaluator=self_play.HeuristicEvaluator(),
            simulations=1,
            c_puct=0.5,
            rng=random.Random(11),
            fpu_mode="parent_value",
        )

        self.assertIs(unvisited_child, search._select_child(root))

    def test_parent_value_fpu_changes_live_root_selection_during_run(self):
        class FakeGame:
            def __init__(self, state="root", current_player=0):
                self.state = state
                self.current_player = current_player
                self.winner = None

            def clone(self):
                return FakeGame(self.state, self.current_player)

            def possible_moves(self):
                return [0, 1] if self.state == "root" else []

            def pit_index(self, move):
                return move

            def move(self, absolute_move):
                self.state = f"child_{absolute_move}"
                self.current_player = 1
                return True

            def over(self):
                return False

        class ScriptedEvaluator(self_play.Evaluator):
            def evaluate(self, game):
                if game.state == "root":
                    return np.array([0.6, 0.4, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 0.0
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), -0.3

        zero_search = self_play.PUCT(
            evaluator=ScriptedEvaluator(),
            simulations=2,
            c_puct=1.0,
            rng=random.Random(5),
            fpu_mode="zero",
        )
        zero_visits, _zero_root = zero_search.run(FakeGame())

        parent_value_search = self_play.PUCT(
            evaluator=ScriptedEvaluator(),
            simulations=2,
            c_puct=1.0,
            rng=random.Random(5),
            fpu_mode="parent_value",
        )
        parent_value_visits, parent_value_root = parent_value_search.run(FakeGame())

        self.assertEqual([2.0, 0.0], zero_visits[:2].tolist())
        self.assertEqual([1.0, 1.0], parent_value_visits[:2].tolist())
        self.assertAlmostEqual(0.3, parent_value_root.q_value)

    def test_normalize_child_values_preserves_ordering_and_flat_ranges(self):
        search = self_play.PUCT(
            evaluator=self_play.HeuristicEvaluator(),
            simulations=1,
            c_puct=1.25,
            rng=random.Random(17),
            normalize_values=True,
        )

        normalized = search._normalize_child_values([0.12, 0.13, 0.14])

        self.assertLess(normalized[0], normalized[1])
        self.assertLess(normalized[1], normalized[2])
        self.assertEqual([0.25, 0.25], search._normalize_child_values([0.25, 0.25]))

    def test_normalized_values_are_used_in_child_selection_when_enabled(self):
        root = self_play.Node(game=self.seeded_game(), visit_count=6)
        high_prior_child = self_play.Node(game=self.seeded_game(), prior=0.8, visit_count=1, value_sum=0.13)
        high_value_child = self_play.Node(game=self.seeded_game(), prior=0.2, visit_count=5, value_sum=0.7)
        root.children[0] = high_prior_child
        root.children[1] = high_value_child

        raw_search = self_play.PUCT(
            evaluator=self_play.HeuristicEvaluator(),
            simulations=1,
            c_puct=0.03,
            rng=random.Random(19),
            normalize_values=False,
        )
        normalized_search = self_play.PUCT(
            evaluator=self_play.HeuristicEvaluator(),
            simulations=1,
            c_puct=0.03,
            rng=random.Random(19),
            normalize_values=True,
        )

        self.assertIs(high_prior_child, raw_search._select_child(root))
        self.assertIs(high_value_child, normalized_search._select_child(root))

    def test_reuse_subtree_only_reuses_matching_child_root(self):
        evaluator = self_play.HeuristicEvaluator()
        first_search = self_play.PUCT(
            evaluator=evaluator,
            simulations=8,
            c_puct=1.25,
            rng=random.Random(13),
            reuse_subtree=True,
        )
        game = self.seeded_game()

        first_visits, first_root = first_search.run(game)
        first_move = int(np.argmax(first_visits))
        next_game = game.clone()
        self.assertTrue(next_game.move(next_game.pit_index(first_move)))
        next_root = first_root.child_for_action(first_move)

        reused_search = self_play.PUCT(
            evaluator=evaluator,
            simulations=4,
            c_puct=1.25,
            rng=random.Random(13),
            root=next_root,
            reuse_subtree=True,
        )
        _reused_visits, reused_root = reused_search.run(next_game)

        cold_search = self_play.PUCT(
            evaluator=evaluator,
            simulations=4,
            c_puct=1.25,
            rng=random.Random(13),
            root=next_root,
            reuse_subtree=False,
        )
        _cold_visits, cold_root = cold_search.run(next_game)

        mismatched_game = self.seeded_game()
        mismatched_search = self_play.PUCT(
            evaluator=evaluator,
            simulations=4,
            c_puct=1.25,
            rng=random.Random(13),
            root=next_root,
            reuse_subtree=True,
        )
        _mismatched_visits, mismatched_root = mismatched_search.run(mismatched_game)

        self.assertIs(reused_root, next_root)
        self.assertIsNot(cold_root, next_root)
        self.assertIsNot(mismatched_root, next_root)

    def test_cli_writes_skeleton_report_with_required_fields(self):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/benchmark.py",
                    "--mode",
                    "sanity",
                    "--games",
                    "60",
                    "--seed",
                    "42",
                    "--out",
                    str(out_path),
                    "--dry-run",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertTrue(out_path.exists())

            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual("azlite_benchmark_v1", report["schema"])
            self.assertEqual("sanity", report["mode"])
            self.assertEqual(60, report["games"])
            self.assertEqual(42, report["seed"])
            self.assertIn("checks", report)

    def test_cli_records_search_options_in_report(self):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/benchmark.py",
                    "--mode",
                    "sanity",
                    "--games",
                    "12",
                    "--seed",
                    "7",
                    "--out",
                    str(out_path),
                    "--dry-run",
                    "--fpu-mode",
                    "parent_q",
                    "--reuse-subtree",
                    "--normalize-values",
                    "--root-policy-mode",
                    "deterministic",
                    "--tactical-root-bias",
                    "0.2",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {
                    "fpu_mode": "parent_q",
                    "reuse_subtree": True,
                    "normalize_values": True,
                    "root_policy_mode": "deterministic",
                    "tactical_root_bias": 0.2,
                },
                report["search_options"],
            )

    def test_cli_uses_eval_search_defaults_in_report(self):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/benchmark.py",
                    "--mode",
                    "sanity",
                    "--games",
                    "12",
                    "--seed",
                    "7",
                    "--out",
                    str(out_path),
                    "--dry-run",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual("deterministic", report["search_options"]["root_policy_mode"])
            self.assertEqual(0.1, report["search_options"]["tactical_root_bias"])

    def test_promotion_mode_reads_arena_report_and_sets_check_pass_status(self):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            arena_report_path = tmp_path / "arena_report.json"
            mcts_report_path = tmp_path / "mcts1200_report.json"
            arena_report_path.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 60,
                        "wins": 36,
                        "losses": 24,
                        "draws": 0,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            mcts_report_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 30,
                        "az_wins": 15,
                        "mcts_wins": 15,
                        "draws": 0,
                        "score": 0.5,
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/benchmark.py",
                    "--mode",
                    "promotion",
                    "--games",
                    "60",
                    "--seed",
                    "42",
                    "--arena-report",
                    str(arena_report_path),
                    "--mcts-report",
                    str(mcts_report_path),
                    "--min-mcts-score",
                    "0.45",
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual("promotion", report["mode"])
            self.assertEqual(2, len(report["checks"]))
            arena_check = report["checks"][0]
            self.assertEqual("promotion_arena", arena_check["id"])
            self.assertTrue(arena_check["passed"])
            self.assertEqual(0.6, arena_check["score"])

            mcts_check = report["checks"][1]
            self.assertEqual("mcts1200_gate", mcts_check["id"])
            self.assertTrue(mcts_check["passed"])
            self.assertEqual(0.5, mcts_check["score"])

    def test_promotion_mode_includes_confidence_fields_when_confidence_gate_is_set(self):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            arena_report_path = tmp_path / "arena_report.json"
            mcts_report_path = tmp_path / "mcts1200_report.json"
            arena_report_path.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 60,
                        "wins": 36,
                        "losses": 24,
                        "draws": 0,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            mcts_report_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 30,
                        "az_wins": 15,
                        "mcts_wins": 15,
                        "draws": 0,
                        "score": 0.5,
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/benchmark.py",
                    "--mode",
                    "promotion",
                    "--games",
                    "60",
                    "--seed",
                    "42",
                    "--arena-report",
                    str(arena_report_path),
                    "--mcts-report",
                    str(mcts_report_path),
                    "--min-confidence-lower-bound",
                    "0.45",
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            arena_check = report["checks"][0]
            self.assertEqual("promotion_arena", arena_check["id"])
            self.assertIn("confidence_lower_bound", arena_check)
            self.assertIn("confidence_passed", arena_check)
            self.assertIn("min_confidence_lower_bound", arena_check)
            self.assertEqual(0.45, arena_check["min_confidence_lower_bound"])

    def test_promotion_mode_accepts_parallel_mcts1200_report_shape(self):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            arena_report_path = tmp_path / "arena_report.json"
            mcts_report_path = tmp_path / "mcts1200_report.json"
            arena_report_path.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 60,
                        "wins": 36,
                        "losses": 24,
                        "draws": 0,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )

            mcts_result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/mcts1200_baseline.py",
                    "--challenger-path",
                    "storage/ai/alphazero_lite/current",
                    "--games",
                    "30",
                    "--seed",
                    "42",
                    "--az-base-simulations",
                    "640",
                    "--mcts-simulations",
                    "1200",
                    "--workers",
                    "2",
                    "--out",
                    str(mcts_report_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "AZLITE_MCTS1200_BASELINE_STUB": "1"},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, mcts_result.returncode, msg=mcts_result.stderr)
            mcts_source_report = json.loads(mcts_report_path.read_text(encoding="utf-8"))

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/benchmark.py",
                    "--mode",
                    "promotion",
                    "--games",
                    "60",
                    "--seed",
                    "42",
                    "--arena-report",
                    str(arena_report_path),
                    "--mcts-report",
                    str(mcts_report_path),
                    "--min-mcts-score",
                    "0.45",
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            mcts_check = report["checks"][1]
            self.assertEqual("mcts1200_gate", mcts_check["id"])
            self.assertTrue(mcts_check["passed"])
            self.assertEqual(mcts_source_report["score"], mcts_check["score"])
            self.assertEqual(mcts_source_report["games"], mcts_check["games_played"])
            self.assertEqual(mcts_source_report["az_wins"], mcts_check["wins"])
            self.assertEqual(mcts_source_report["mcts_wins"], mcts_check["losses"])
            self.assertEqual(mcts_source_report["draws"], mcts_check["draws"])
            self.assertEqual(0.45, mcts_check["min_score"])

    def test_promotion_mode_preserves_parallel_mcts1200_gate_failure_shape(self):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            arena_report_path = tmp_path / "arena_report.json"
            mcts_report_path = tmp_path / "mcts1200_report.json"
            arena_report_path.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 60,
                        "wins": 36,
                        "losses": 24,
                        "draws": 0,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )

            mcts_result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/mcts1200_baseline.py",
                    "--challenger-path",
                    "storage/ai/alphazero_lite/current",
                    "--games",
                    "30",
                    "--seed",
                    "42",
                    "--az-base-simulations",
                    "640",
                    "--mcts-simulations",
                    "1200",
                    "--workers",
                    "2",
                    "--out",
                    str(mcts_report_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "AZLITE_MCTS1200_BASELINE_STUB": "1"},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, mcts_result.returncode, msg=mcts_result.stderr)
            mcts_source_report = json.loads(mcts_report_path.read_text(encoding="utf-8"))

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/benchmark.py",
                    "--mode",
                    "promotion",
                    "--games",
                    "60",
                    "--seed",
                    "42",
                    "--arena-report",
                    str(arena_report_path),
                    "--mcts-report",
                    str(mcts_report_path),
                    "--min-mcts-score",
                    "0.51",
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            mcts_check = report["checks"][1]
            self.assertEqual("mcts1200_gate", mcts_check["id"])
            self.assertFalse(mcts_check["passed"])
            self.assertEqual(mcts_source_report["score"], mcts_check["score"])
            self.assertEqual(mcts_source_report["games"], mcts_check["games_played"])
            self.assertEqual(mcts_source_report["az_wins"], mcts_check["wins"])
            self.assertEqual(mcts_source_report["mcts_wins"], mcts_check["losses"])
            self.assertEqual(mcts_source_report["draws"], mcts_check["draws"])
            self.assertEqual(0.51, mcts_check["min_score"])

    def test_promotion_mode_passes_when_mcts_score_matches_current_baseline(self):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            arena_report_path = tmp_path / "arena_report.json"
            challenger_mcts_report_path = tmp_path / "challenger_mcts1200_report.json"
            current_baseline_mcts_report_path = tmp_path / "current_baseline_mcts1200_report.json"

            arena_report_path.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 60,
                        "wins": 36,
                        "losses": 24,
                        "draws": 0,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            challenger_mcts_report_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 30,
                        "az_wins": 15,
                        "mcts_wins": 15,
                        "draws": 0,
                        "score": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            current_baseline_mcts_report_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 30,
                        "az_wins": 15,
                        "mcts_wins": 15,
                        "draws": 0,
                        "score": 0.5,
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/benchmark.py",
                    "--mode",
                    "promotion",
                    "--games",
                    "60",
                    "--seed",
                    "42",
                    "--arena-report",
                    str(arena_report_path),
                    "--mcts-report",
                    str(challenger_mcts_report_path),
                    "--current-baseline-mcts-report",
                    str(current_baseline_mcts_report_path),
                    "--min-mcts-score",
                    "0.9",
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            mcts_check = report["checks"][1]
            self.assertEqual("mcts1200_gate", mcts_check["id"])
            self.assertTrue(mcts_check["passed"])
            self.assertEqual(0.5, mcts_check["score"])

    def test_promotion_mode_fails_when_mcts_score_is_below_current_baseline(self):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            arena_report_path = tmp_path / "arena_report.json"
            challenger_mcts_report_path = tmp_path / "challenger_mcts1200_report.json"
            current_baseline_mcts_report_path = tmp_path / "current_baseline_mcts1200_report.json"

            arena_report_path.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 60,
                        "wins": 36,
                        "losses": 24,
                        "draws": 0,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            challenger_mcts_report_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 30,
                        "az_wins": 12,
                        "mcts_wins": 18,
                        "draws": 0,
                        "score": 0.4,
                    }
                ),
                encoding="utf-8",
            )
            current_baseline_mcts_report_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 30,
                        "az_wins": 15,
                        "mcts_wins": 15,
                        "draws": 0,
                        "score": 0.5,
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/benchmark.py",
                    "--mode",
                    "promotion",
                    "--games",
                    "60",
                    "--seed",
                    "42",
                    "--arena-report",
                    str(arena_report_path),
                    "--mcts-report",
                    str(challenger_mcts_report_path),
                    "--current-baseline-mcts-report",
                    str(current_baseline_mcts_report_path),
                    "--min-mcts-score",
                    "0.1",
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            mcts_check = report["checks"][1]
            self.assertEqual("mcts1200_gate", mcts_check["id"])
            self.assertFalse(mcts_check["passed"])
            self.assertEqual(0.4, mcts_check["score"])

    def test_promotion_mode_reports_baseline_relative_mcts_gate_details(self):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            arena_report_path = tmp_path / "arena_report.json"
            challenger_mcts_report_path = tmp_path / "challenger_mcts1200_report.json"
            current_baseline_mcts_report_path = tmp_path / "current_baseline_mcts1200_report.json"

            arena_report_path.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 60,
                        "wins": 36,
                        "losses": 24,
                        "draws": 0,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            challenger_mcts_report_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 30,
                        "az_wins": 12,
                        "mcts_wins": 18,
                        "draws": 0,
                        "score": 0.4,
                    }
                ),
                encoding="utf-8",
            )
            current_baseline_mcts_report_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 30,
                        "az_wins": 15,
                        "mcts_wins": 15,
                        "draws": 0,
                        "score": 0.5,
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/benchmark.py",
                    "--mode",
                    "promotion",
                    "--games",
                    "60",
                    "--seed",
                    "42",
                    "--arena-report",
                    str(arena_report_path),
                    "--mcts-report",
                    str(challenger_mcts_report_path),
                    "--current-baseline-mcts-report",
                    str(current_baseline_mcts_report_path),
                    "--min-mcts-score",
                    "0.1",
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            mcts_check = report["checks"][1]
            self.assertEqual("mcts1200_gate", mcts_check["id"])
            self.assertEqual("current_baseline", mcts_check["comparison"])
            self.assertEqual(0.4, mcts_check["score"])
            self.assertEqual(0.5, mcts_check["baseline_score"])
            self.assertFalse(mcts_check["passed"])
            self.assertNotIn("min_score", mcts_check)
            self.assertNotIn("challenger_score", mcts_check)
            self.assertNotIn("current_baseline_score", mcts_check)

    def test_promotion_mode_rejects_mcts_report_missing_required_fields(self):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            arena_report_path = tmp_path / "arena_report.json"
            mcts_report_path = tmp_path / "mcts1200_report.json"
            arena_report_path.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 60,
                        "wins": 36,
                        "losses": 24,
                        "draws": 0,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            mcts_report_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 30,
                        "az_wins": 15,
                        "draws": 0,
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/benchmark.py",
                    "--mode",
                    "promotion",
                    "--games",
                    "60",
                    "--seed",
                    "42",
                    "--arena-report",
                    str(arena_report_path),
                    "--mcts-report",
                    str(mcts_report_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("missing required field", result.stderr)

    def test_promotion_mode_rejects_mcts_report_with_negative_counts(self):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            arena_report_path = tmp_path / "arena_report.json"
            mcts_report_path = tmp_path / "mcts1200_report.json"
            arena_report_path.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 60,
                        "wins": 36,
                        "losses": 24,
                        "draws": 0,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            mcts_report_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 30,
                        "az_wins": -1,
                        "mcts_wins": 31,
                        "draws": 0,
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/benchmark.py",
                    "--mode",
                    "promotion",
                    "--games",
                    "60",
                    "--seed",
                    "42",
                    "--arena-report",
                    str(arena_report_path),
                    "--mcts-report",
                    str(mcts_report_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("must be non-negative", result.stderr)

    def test_promotion_mode_rejects_mcts_report_with_boolean_integer_field(self):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            arena_report_path = tmp_path / "arena_report.json"
            mcts_report_path = tmp_path / "mcts1200_report.json"
            arena_report_path.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 60,
                        "wins": 36,
                        "losses": 24,
                        "draws": 0,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            mcts_report_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 30,
                        "az_wins": 15,
                        "mcts_wins": True,
                        "draws": 0,
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/benchmark.py",
                    "--mode",
                    "promotion",
                    "--games",
                    "60",
                    "--seed",
                    "42",
                    "--arena-report",
                    str(arena_report_path),
                    "--mcts-report",
                    str(mcts_report_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("field mcts_wins must be an integer", result.stderr)

    def test_promotion_mode_rejects_mcts_report_with_string_integer_field(self):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            arena_report_path = tmp_path / "arena_report.json"
            mcts_report_path = tmp_path / "mcts1200_report.json"
            arena_report_path.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 60,
                        "wins": 36,
                        "losses": 24,
                        "draws": 0,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            mcts_report_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": "30",
                        "az_wins": 15,
                        "mcts_wins": 15,
                        "draws": 0,
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/benchmark.py",
                    "--mode",
                    "promotion",
                    "--games",
                    "60",
                    "--seed",
                    "42",
                    "--arena-report",
                    str(arena_report_path),
                    "--mcts-report",
                    str(mcts_report_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("field games must be an integer", result.stderr)

    def test_promotion_mode_rejects_mcts_report_with_inconsistent_totals(self):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            arena_report_path = tmp_path / "arena_report.json"
            mcts_report_path = tmp_path / "mcts1200_report.json"
            arena_report_path.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 60,
                        "wins": 36,
                        "losses": 24,
                        "draws": 0,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            mcts_report_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 30,
                        "az_wins": 10,
                        "mcts_wins": 10,
                        "draws": 5,
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/benchmark.py",
                    "--mode",
                    "promotion",
                    "--games",
                    "60",
                    "--seed",
                    "42",
                    "--arena-report",
                    str(arena_report_path),
                    "--mcts-report",
                    str(mcts_report_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("must equal games", result.stderr)


if __name__ == "__main__":
    unittest.main()
