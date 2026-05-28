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

        with (
            mock.patch.object(Path, "is_file", fake_is_file),
            mock.patch("os.access", side_effect=fake_access),
        ):
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

    def classic_mcts_dynamic_search_profile(
        self, *, exact_solve_enabled=False, exact_solve_stone_threshold=None
    ):
        return {
            "kind": "mcts1200_baseline_eval",
            "player_mode": "classic_mcts",
            "classic_mcts_simulations": 1200,
            "az_base_simulations": 640,
            "mcts_simulations": 1200,
            "exact_solve_enabled": exact_solve_enabled,
            "exact_solve_stone_threshold": exact_solve_stone_threshold,
            "simulation_budget_policy": "fixed_vs_dynamic_classic_mcts",
        }

    def classic_mcts_fixed_search_profile(
        self, *, exact_solve_enabled=False, exact_solve_stone_threshold=None
    ):
        return {
            "kind": "mcts1200_baseline_eval",
            "player_mode": "classic_mcts",
            "classic_mcts_simulations": 1200,
            "az_base_simulations": 640,
            "mcts_simulations": 1200,
            "exact_solve_enabled": exact_solve_enabled,
            "exact_solve_stone_threshold": exact_solve_stone_threshold,
            "simulation_budget_policy": "fixed_classic_mcts",
        }

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
        visited_child = self_play.Node(
            game=self.seeded_game(), prior=0.1, visit_count=3, value_sum=0.3
        )
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
                    return np.array(
                        [0.6, 0.4, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
                    ), 0.0
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
        high_prior_child = self_play.Node(
            game=self.seeded_game(), prior=0.8, visit_count=1, value_sum=0.13
        )
        high_value_child = self_play.Node(
            game=self.seeded_game(), prior=0.2, visit_count=5, value_sum=0.7
        )
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
            self.assertEqual(
                "deterministic", report["search_options"]["root_policy_mode"]
            )
            self.assertEqual(0.1, report["search_options"]["tactical_root_bias"])

    def test_cli_records_value_trust_schedule_in_report(self):
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
                    "--value-trust-enabled",
                    "--value-trust-opening",
                    "0.8",
                    "--value-trust-midgame",
                    "1.0",
                    "--value-trust-late",
                    "1.15",
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
                    "enabled": True,
                    "opening": 0.8,
                    "midgame": 1.0,
                    "late": 1.15,
                },
                report["search_options"]["value_trust_schedule"],
            )

    def test_promotion_build_report_preserves_arena_value_trust_summary(self):
        from ml.alphazero_lite import benchmark

        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            arena_report = tmp_path / "arena.json"
            mcts_report = tmp_path / "mcts.json"
            out_path = tmp_path / "out.json"

            arena_report.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 10,
                        "wins": 6,
                        "losses": 4,
                        "draws": 0,
                        "score": 0.6,
                        "promotion_decision": {"passed": True},
                        "value_trust_summary": {
                            "enabled": True,
                            "phase_bucket": "opening",
                            "effective_multiplier": 0.8,
                            "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
                        },
                    }
                ),
                encoding="utf-8",
            )
            mcts_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 10,
                        "az_wins": 5,
                        "mcts_wins": 5,
                        "draws": 0,
                        "score": 0.5,
                    }
                ),
                encoding="utf-8",
            )

            args = benchmark.parse_args(
                [
                    "--mode",
                    "promotion",
                    "--out",
                    str(out_path),
                    "--arena-report",
                    str(arena_report),
                    "--mcts-report",
                    str(mcts_report),
                ]
            )
            report = benchmark.build_report(args)

        self.assertEqual(
            {
                "enabled": True,
                "phase_bucket": "opening",
                "effective_multiplier": 0.8,
                "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
            },
            report["value_trust_summary"],
        )

    def test_promotion_build_report_marks_scheduled_value_trust_as_experimental_when_checks_pass(
        self,
    ):
        from ml.alphazero_lite import benchmark

        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            arena_report = tmp_path / "arena.json"
            mcts_report = tmp_path / "mcts.json"
            out_path = tmp_path / "out.json"

            arena_report.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 10,
                        "wins": 6,
                        "losses": 4,
                        "draws": 0,
                        "score": 0.6,
                        "promotion_decision": {"passed": True},
                        "value_trust_summary": {
                            "enabled": True,
                            "phase_bucket": "opening",
                            "effective_multiplier": 0.8,
                            "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
                        },
                    }
                ),
                encoding="utf-8",
            )
            mcts_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 10,
                        "az_wins": 5,
                        "mcts_wins": 5,
                        "draws": 0,
                        "score": 0.5,
                    }
                ),
                encoding="utf-8",
            )

            args = benchmark.parse_args(
                [
                    "--mode",
                    "promotion",
                    "--out",
                    str(out_path),
                    "--arena-report",
                    str(arena_report),
                    "--mcts-report",
                    str(mcts_report),
                ]
            )
            report = benchmark.build_report(args)

        self.assertEqual(
            "stay_experimental", report["value_trust_recommendation"]["decision"]
        )
        self.assertTrue(report["value_trust_recommendation"]["scheduled_trust_active"])
        self.assertIn(
            "remain experimental", report["value_trust_recommendation"]["summary"]
        )
        self.assertEqual(0.6, report["value_trust_recommendation"]["arena_score"])
        self.assertEqual(0.5, report["value_trust_recommendation"]["mcts_score"])

    def test_promotion_build_report_recommends_ship_for_uniform_value_trust_when_checks_pass(
        self,
    ):
        from ml.alphazero_lite import benchmark

        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            arena_report = tmp_path / "arena.json"
            mcts_report = tmp_path / "mcts.json"
            out_path = tmp_path / "out.json"

            arena_report.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 10,
                        "wins": 6,
                        "losses": 4,
                        "draws": 0,
                        "score": 0.6,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            mcts_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 10,
                        "az_wins": 5,
                        "mcts_wins": 5,
                        "draws": 0,
                        "score": 0.5,
                    }
                ),
                encoding="utf-8",
            )

            args = benchmark.parse_args(
                [
                    "--mode",
                    "promotion",
                    "--out",
                    str(out_path),
                    "--arena-report",
                    str(arena_report),
                    "--mcts-report",
                    str(mcts_report),
                ]
            )
            report = benchmark.build_report(args)

        self.assertEqual("ship", report["value_trust_recommendation"]["decision"])
        self.assertFalse(report["value_trust_recommendation"]["scheduled_trust_active"])
        self.assertIn("ship", report["value_trust_recommendation"]["summary"])

    def test_promotion_build_report_recommends_drop_when_existing_gate_fails(self):
        from ml.alphazero_lite import benchmark

        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            arena_report = tmp_path / "arena.json"
            mcts_report = tmp_path / "mcts.json"
            out_path = tmp_path / "out.json"

            arena_report.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 10,
                        "wins": 6,
                        "losses": 4,
                        "draws": 0,
                        "score": 0.6,
                        "promotion_decision": {"passed": True},
                        "value_trust_summary": {
                            "enabled": True,
                            "phase_bucket": "opening",
                            "effective_multiplier": 0.8,
                            "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
                        },
                    }
                ),
                encoding="utf-8",
            )
            mcts_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 10,
                        "az_wins": 4,
                        "mcts_wins": 6,
                        "draws": 0,
                        "score": 0.4,
                    }
                ),
                encoding="utf-8",
            )

            args = benchmark.parse_args(
                [
                    "--mode",
                    "promotion",
                    "--out",
                    str(out_path),
                    "--arena-report",
                    str(arena_report),
                    "--mcts-report",
                    str(mcts_report),
                ]
            )
            report = benchmark.build_report(args)

        self.assertEqual("drop", report["value_trust_recommendation"]["decision"])
        self.assertIn("did not clear", report["value_trust_recommendation"]["summary"])
        self.assertFalse(report["value_trust_recommendation"]["mcts_passed"])

    def test_promotion_build_report_recommends_drop_when_confidence_gate_fails(self):
        from ml.alphazero_lite import benchmark

        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            arena_report = tmp_path / "arena.json"
            mcts_report = tmp_path / "mcts.json"
            out_path = tmp_path / "out.json"

            arena_report.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 20,
                        "wins": 11,
                        "losses": 9,
                        "draws": 0,
                        "score": 0.55,
                        "promotion_decision": {"passed": True},
                        "value_trust_summary": {
                            "enabled": True,
                            "phase_bucket": "opening",
                            "effective_multiplier": 0.8,
                            "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
                        },
                    }
                ),
                encoding="utf-8",
            )
            mcts_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 10,
                        "az_wins": 5,
                        "mcts_wins": 5,
                        "draws": 0,
                        "score": 0.5,
                    }
                ),
                encoding="utf-8",
            )

            args = benchmark.parse_args(
                [
                    "--mode",
                    "promotion",
                    "--out",
                    str(out_path),
                    "--arena-report",
                    str(arena_report),
                    "--mcts-report",
                    str(mcts_report),
                    "--min-confidence-lower-bound",
                    "0.6",
                ]
            )
            report = benchmark.build_report(args)

        self.assertEqual("drop", report["value_trust_recommendation"]["decision"])
        self.assertFalse(report["value_trust_recommendation"]["arena_passed"])
        self.assertFalse(
            report["value_trust_recommendation"]["arena_confidence_passed"]
        )
        self.assertTrue(report["value_trust_recommendation"]["mcts_passed"])

    def test_promotion_build_report_omits_value_trust_summary_when_arena_report_lacks_it(
        self,
    ):
        from ml.alphazero_lite import benchmark

        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            arena_report = tmp_path / "arena.json"
            mcts_report = tmp_path / "mcts.json"
            out_path = tmp_path / "out.json"

            arena_report.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 10,
                        "wins": 6,
                        "losses": 4,
                        "draws": 0,
                        "score": 0.6,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            mcts_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 10,
                        "az_wins": 5,
                        "mcts_wins": 5,
                        "draws": 0,
                        "score": 0.5,
                    }
                ),
                encoding="utf-8",
            )

            args = benchmark.parse_args(
                [
                    "--mode",
                    "promotion",
                    "--out",
                    str(out_path),
                    "--arena-report",
                    str(arena_report),
                    "--mcts-report",
                    str(mcts_report),
                ]
            )
            report = benchmark.build_report(args)

        self.assertNotIn("value_trust_summary", report)

    def test_build_report_from_inputs_preserves_real_arena_value_trust_summary(self):
        from ml.alphazero_lite import benchmark
        from ml.alphazero_lite import arena

        root_value_trust = {
            "enabled": True,
            "phase_bucket": "late",
            "effective_multiplier": 1.15,
            "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
        }

        arena_report = arena.aggregate_worker_reports(
            games=10,
            min_score=0.55,
            challenger_path=Path("challenger"),
            current_path=Path("current"),
            challenger_simulations=96,
            current_simulations=96,
            seed=42,
            workers=1,
            search_options=arena.build_eval_search_options(
                value_trust_schedule={
                    "enabled": True,
                    "opening": 0.8,
                    "midgame": 1.0,
                    "late": 1.15,
                }
            ),
            results=[
                {
                    "wins": 6,
                    "losses": 4,
                    "draws": 0,
                    "move_durations_ms": [5.0],
                    "hard_suite_buckets": arena.empty_hard_suite_buckets(),
                    "search_profile": arena.build_search_profile(
                        kind="arena_eval",
                        player_mode="puct",
                        simulations=96,
                        c_puct=1.25,
                        search_options=arena.build_eval_search_options(
                            value_trust_schedule={
                                "enabled": True,
                                "opening": 0.8,
                                "midgame": 1.0,
                                "late": 1.15,
                            }
                        ),
                    ),
                    "search_profile_hash": "placeholder",
                    "value_trust_summary": root_value_trust,
                }
            ],
        )
        mcts_report = {
            "schema": "azlite_vs_mcts_v1",
            "games": 10,
            "az_wins": 5,
            "mcts_wins": 5,
            "draws": 0,
            "score": 0.5,
        }

        report = benchmark.build_report_from_inputs(
            arena_report=arena_report, mcts_report=mcts_report
        )

        self.assertEqual(
            root_value_trust, report["arena_report"]["value_trust_summary"]
        )
        self.assertNotEqual(
            {"enabled": True, "opening": 0.8, "midgame": 1.0, "late": 1.15},
            report["arena_report"]["value_trust_summary"],
        )

    def test_build_report_from_inputs_omits_value_trust_summary_when_arena_report_lacks_it(
        self,
    ):
        from ml.alphazero_lite import benchmark

        arena_report = {
            "schema": "arena_v1",
            "games_played": 10,
            "wins": 6,
            "losses": 4,
            "draws": 0,
            "score": 0.6,
            "promotion_decision": {"passed": True},
        }
        mcts_report = {
            "schema": "azlite_vs_mcts_v1",
            "games": 10,
            "az_wins": 5,
            "mcts_wins": 5,
            "draws": 0,
            "score": 0.5,
        }

        report = benchmark.build_report_from_inputs(
            arena_report=arena_report, mcts_report=mcts_report
        )

        self.assertNotIn("value_trust_summary", report)

    def test_build_report_from_inputs_uses_configured_schedule_to_keep_scheduled_recommendation_without_runtime_summary(
        self,
    ):
        from ml.alphazero_lite import benchmark

        arena_report = {
            "schema": "arena_v1",
            "games_played": 10,
            "wins": 6,
            "losses": 4,
            "draws": 0,
            "score": 0.6,
            "promotion_decision": {"passed": True},
            "notes": {
                "search_options": {
                    "value_trust_schedule": {
                        "enabled": True,
                        "opening": 0.8,
                        "midgame": 1.0,
                        "late": 1.15,
                    }
                }
            },
        }
        mcts_report = {
            "schema": "azlite_vs_mcts_v1",
            "games": 10,
            "az_wins": 5,
            "mcts_wins": 5,
            "draws": 0,
            "score": 0.5,
        }

        report = benchmark.build_report_from_inputs(
            arena_report=arena_report, mcts_report=mcts_report
        )

        self.assertEqual(
            "stay_experimental", report["value_trust_recommendation"]["decision"]
        )
        self.assertTrue(report["value_trust_recommendation"]["scheduled_trust_active"])

    def test_build_report_from_inputs_marks_scheduled_value_trust_as_experimental_when_checks_pass(
        self,
    ):
        from ml.alphazero_lite import benchmark

        arena_report = {
            "schema": "arena_v1",
            "games_played": 10,
            "wins": 6,
            "losses": 4,
            "draws": 0,
            "score": 0.6,
            "promotion_decision": {"passed": True},
            "value_trust_summary": {
                "enabled": True,
                "phase_bucket": "opening",
                "effective_multiplier": 0.8,
                "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
            },
        }
        mcts_report = {
            "schema": "azlite_vs_mcts_v1",
            "games": 10,
            "az_wins": 5,
            "mcts_wins": 5,
            "draws": 0,
            "score": 0.5,
        }

        report = benchmark.build_report_from_inputs(
            arena_report=arena_report, mcts_report=mcts_report
        )

        self.assertEqual(
            "stay_experimental", report["value_trust_recommendation"]["decision"]
        )
        self.assertTrue(report["value_trust_recommendation"]["scheduled_trust_active"])
        self.assertTrue(report["value_trust_recommendation"]["arena_passed"])
        self.assertTrue(report["value_trust_recommendation"]["mcts_passed"])

    def test_build_report_from_inputs_recommends_ship_for_uniform_value_trust_when_checks_pass(
        self,
    ):
        from ml.alphazero_lite import benchmark

        arena_report = {
            "schema": "arena_v1",
            "games_played": 10,
            "wins": 6,
            "losses": 4,
            "draws": 0,
            "score": 0.6,
            "promotion_decision": {"passed": True},
        }
        mcts_report = {
            "schema": "azlite_vs_mcts_v1",
            "games": 10,
            "az_wins": 5,
            "mcts_wins": 5,
            "draws": 0,
            "score": 0.5,
        }

        report = benchmark.build_report_from_inputs(
            arena_report=arena_report, mcts_report=mcts_report
        )

        self.assertEqual("ship", report["value_trust_recommendation"]["decision"])
        self.assertFalse(report["value_trust_recommendation"]["scheduled_trust_active"])
        self.assertTrue(report["value_trust_recommendation"]["arena_passed"])
        self.assertTrue(report["value_trust_recommendation"]["mcts_passed"])

    def test_build_report_from_inputs_recommends_drop_when_gate_fails(self):
        from ml.alphazero_lite import benchmark

        arena_report = {
            "schema": "arena_v1",
            "games_played": 20,
            "wins": 11,
            "losses": 9,
            "draws": 0,
            "score": 0.55,
            "promotion_decision": {"passed": True},
            "value_trust_summary": {
                "enabled": True,
                "phase_bucket": "opening",
                "effective_multiplier": 0.8,
                "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
            },
        }
        mcts_report = {
            "schema": "azlite_vs_mcts_v1",
            "games": 10,
            "az_wins": 4,
            "mcts_wins": 6,
            "draws": 0,
            "score": 0.4,
        }

        report = benchmark.build_report_from_inputs(
            arena_report=arena_report,
            mcts_report=mcts_report,
            min_score=0.55,
            min_confidence_lower_bound=0.6,
            min_mcts_score=0.45,
        )

        self.assertEqual("drop", report["value_trust_recommendation"]["decision"])
        self.assertFalse(report["value_trust_recommendation"]["arena_passed"])
        self.assertFalse(
            report["value_trust_recommendation"]["arena_confidence_passed"]
        )
        self.assertFalse(report["value_trust_recommendation"]["mcts_passed"])

    def test_build_report_from_inputs_recommends_drop_when_current_baseline_outscores_candidate(
        self,
    ):
        from ml.alphazero_lite import benchmark

        arena_report = {
            "schema": "arena_v1",
            "games_played": 10,
            "wins": 6,
            "losses": 4,
            "draws": 0,
            "score": 0.6,
            "promotion_decision": {"passed": True},
            "value_trust_summary": {
                "enabled": True,
                "phase_bucket": "opening",
                "effective_multiplier": 0.8,
                "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
            },
        }
        mcts_report = {
            "schema": "azlite_vs_mcts_v1",
            "games": 10,
            "az_wins": 5,
            "mcts_wins": 5,
            "draws": 0,
            "score": 0.5,
        }
        current_baseline_mcts_report = {
            "schema": "azlite_vs_mcts_v1",
            "games": 10,
            "az_wins": 6,
            "mcts_wins": 4,
            "draws": 0,
            "score": 0.6,
        }

        report = benchmark.build_report_from_inputs(
            arena_report=arena_report,
            mcts_report=mcts_report,
            current_baseline_mcts_report=current_baseline_mcts_report,
        )

        self.assertEqual("drop", report["value_trust_recommendation"]["decision"])
        self.assertTrue(report["value_trust_recommendation"]["arena_passed"])
        self.assertFalse(report["value_trust_recommendation"]["mcts_passed"])
        self.assertEqual(
            "current_baseline", report["value_trust_recommendation"]["mcts_comparison"]
        )
        self.assertEqual(
            0.6, report["value_trust_recommendation"]["mcts_baseline_score"]
        )
        self.assertIn(
            "current baseline", report["value_trust_recommendation"]["summary"]
        )
        mcts_check = next(
            check for check in report["checks"] if check["id"] == "mcts1200_gate"
        )
        self.assertEqual("current_baseline", mcts_check["comparison"])
        self.assertEqual(0.6, mcts_check["baseline_score"])
        self.assertNotIn("min_score", mcts_check)

    def test_build_report_and_helper_match_for_baseline_and_confidence_context(self):
        from ml.alphazero_lite import benchmark

        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            arena_report_path = tmp_path / "arena.json"
            mcts_report_path = tmp_path / "mcts.json"
            baseline_report_path = tmp_path / "baseline.json"
            out_path = tmp_path / "out.json"

            arena_report = {
                "schema": "arena_v1",
                "games_played": 20,
                "wins": 11,
                "losses": 9,
                "draws": 0,
                "score": 0.55,
                "promotion_decision": {"passed": True},
                "value_trust_summary": {
                    "enabled": False,
                    "phase_bucket": "midgame",
                    "effective_multiplier": 1.0,
                    "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
                },
            }
            mcts_report = {
                "schema": "azlite_vs_mcts_v1",
                "games": 10,
                "az_wins": 5,
                "mcts_wins": 5,
                "draws": 0,
                "score": 0.5,
            }
            baseline_report = {
                "schema": "azlite_vs_mcts_v1",
                "games": 10,
                "az_wins": 6,
                "mcts_wins": 4,
                "draws": 0,
                "score": 0.6,
            }

            arena_report_path.write_text(json.dumps(arena_report), encoding="utf-8")
            mcts_report_path.write_text(json.dumps(mcts_report), encoding="utf-8")
            baseline_report_path.write_text(
                json.dumps(baseline_report), encoding="utf-8"
            )

            args = benchmark.parse_args(
                [
                    "--mode",
                    "promotion",
                    "--out",
                    str(out_path),
                    "--arena-report",
                    str(arena_report_path),
                    "--mcts-report",
                    str(mcts_report_path),
                    "--current-baseline-mcts-report",
                    str(baseline_report_path),
                    "--min-confidence-lower-bound",
                    "0.6",
                ]
            )

            production_report = benchmark.build_report(args)
            helper_report = benchmark.build_report_from_inputs(
                arena_report=arena_report,
                mcts_report=mcts_report,
                current_baseline_mcts_report=baseline_report,
                min_confidence_lower_bound=0.6,
            )

        self.assertEqual(production_report["checks"], helper_report["checks"])
        self.assertEqual(
            production_report["value_trust_recommendation"],
            helper_report["value_trust_recommendation"],
        )

    def test_build_report_from_inputs_keeps_baseline_tie_when_raw_scores_match_before_rounding(
        self,
    ):
        from ml.alphazero_lite import benchmark

        arena_report = {
            "schema": "arena_v1",
            "games_played": 10,
            "wins": 6,
            "losses": 4,
            "draws": 0,
            "score": 0.6,
            "promotion_decision": {"passed": True},
            "value_trust_summary": {
                "enabled": True,
                "phase_bucket": "opening",
                "effective_multiplier": 0.8,
                "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
            },
        }
        mcts_report = {
            "schema": "azlite_vs_mcts_v1",
            "games": 80000,
            "az_wins": 39997,
            "mcts_wins": 40003,
            "draws": 0,
            "score": 0.5,
        }
        current_baseline_mcts_report = {
            "schema": "azlite_vs_mcts_v1",
            "games": 80000,
            "az_wins": 39997,
            "mcts_wins": 40003,
            "draws": 0,
            "score": 0.5,
        }

        report = benchmark.build_report_from_inputs(
            arena_report=arena_report,
            mcts_report=mcts_report,
            current_baseline_mcts_report=current_baseline_mcts_report,
        )

        self.assertEqual(
            "stay_experimental", report["value_trust_recommendation"]["decision"]
        )
        self.assertTrue(report["value_trust_recommendation"]["mcts_passed"])
        mcts_check = next(
            check for check in report["checks"] if check["id"] == "mcts1200_gate"
        )
        self.assertTrue(mcts_check["passed"])
        self.assertEqual(0.5, mcts_check["score"])
        self.assertEqual(0.5, mcts_check["baseline_score"])

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

    def test_promotion_mode_rejects_missing_current_baseline_report_with_explicit_error(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            arena_report_path = tmp_path / "arena_report.json"
            mcts_report_path = tmp_path / "mcts1200_report.json"
            missing_baseline_path = tmp_path / "missing_baseline.json"

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
                    "--current-baseline-mcts-report",
                    str(missing_baseline_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn(
                f"current baseline mcts report not found: {missing_baseline_path}",
                result.stderr,
            )

    def test_promotion_mode_includes_confidence_fields_when_confidence_gate_is_set(
        self,
    ):
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
            self.assertEqual(60, arena_check["games"])
            self.assertIn("confidence_lower_bound", arena_check)
            self.assertIn("confidence_passed", arena_check)
            self.assertIn("confidence_interval_95", arena_check)
            self.assertIn("threshold", arena_check)
            self.assertIn("threshold_margin", arena_check)
            self.assertIn("unstable_decision", arena_check)
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
            mcts_source_report = json.loads(
                mcts_report_path.read_text(encoding="utf-8")
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
            mcts_source_report = json.loads(
                mcts_report_path.read_text(encoding="utf-8")
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
            current_baseline_mcts_report_path = (
                tmp_path / "current_baseline_mcts1200_report.json"
            )

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
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "classic_mcts_mode": "dynamic",
                        "search_profile": self.classic_mcts_dynamic_search_profile(),
                        "search_profile_hash": "dynamic-profile-hash",
                        "games": 30,
                        "az_wins": 15,
                        "mcts_wins": 15,
                        "draws": 0,
                        "score": 0.5,
                        "budget_summary": {
                            "source": "classic_mcts_dynamic_runtime",
                            "mean_final_simulations": 128,
                            "mean_root_latency_ms": 6.5,
                        },
                        "classic_mcts_dynamic_budget_config": {
                            "enabled": True,
                            "probe_simulations": 12,
                            "min_simulations": 96,
                            "max_simulations": 128,
                            "entropy_weight": 0.8,
                            "low_margin_threshold": 0.2,
                            "low_margin_weight": 1.5,
                            "variance_weight": 1.5,
                        },
                        "dynamic_budget_comparison": {
                            "comparison_mode": "classic_dynamic_vs_fixed",
                            "runtime_target_ms": 6.3,
                            "runtime_target_matched": True,
                            "seat_bias_neutralized": True,
                            "dynamic_mean_final_simulations": 128.0,
                            "dynamic_mean_root_latency_ms": 6.5,
                            "fixed_mean_final_simulations": 96.0,
                            "fixed_mean_root_latency_ms": 6.3,
                            "dynamic_score": 0.5,
                            "fixed_score": 0.5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            current_baseline_mcts_report_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "classic_mcts_mode": "fixed",
                        "search_profile": self.classic_mcts_fixed_search_profile(),
                        "search_profile_hash": "fixed-profile-hash",
                        "games": 30,
                        "az_wins": 15,
                        "mcts_wins": 15,
                        "draws": 0,
                        "score": 0.5,
                        "budget_summary": {
                            "source": "classic_mcts_fixed_runtime",
                            "mean_final_simulations": 96,
                            "mean_root_latency_ms": 6.3,
                        },
                        "dynamic_budget_comparison": {
                            "comparison_mode": "classic_dynamic_vs_fixed",
                            "runtime_target_ms": 6.3,
                            "runtime_target_matched": True,
                            "seat_bias_neutralized": True,
                            "dynamic_mean_final_simulations": 128.0,
                            "dynamic_mean_root_latency_ms": 6.5,
                            "fixed_mean_final_simulations": 96.0,
                            "fixed_mean_root_latency_ms": 6.3,
                            "dynamic_score": 0.5,
                            "fixed_score": 0.5,
                        },
                        "classic_mcts_dynamic_budget_config": {"enabled": False},
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
            current_baseline_mcts_report_path = (
                tmp_path / "current_baseline_mcts1200_report.json"
            )

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
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "classic_mcts_mode": "dynamic",
                        "search_profile": self.classic_mcts_dynamic_search_profile(),
                        "search_profile_hash": "dynamic-profile-hash",
                        "games": 30,
                        "az_wins": 12,
                        "mcts_wins": 18,
                        "draws": 0,
                        "score": 0.4,
                        "budget_summary": {
                            "source": "classic_mcts_dynamic_runtime",
                            "mean_final_simulations": 128,
                            "mean_root_latency_ms": 6.5,
                        },
                        "classic_mcts_dynamic_budget_config": {
                            "enabled": True,
                            "probe_simulations": 12,
                            "min_simulations": 96,
                            "max_simulations": 128,
                            "entropy_weight": 0.8,
                            "low_margin_threshold": 0.2,
                            "low_margin_weight": 1.5,
                            "variance_weight": 1.5,
                        },
                        "dynamic_budget_comparison": {
                            "comparison_mode": "classic_dynamic_vs_fixed",
                            "runtime_target_ms": 6.3,
                            "runtime_target_matched": True,
                            "seat_bias_neutralized": True,
                            "dynamic_mean_final_simulations": 128.0,
                            "dynamic_mean_root_latency_ms": 6.5,
                            "fixed_mean_final_simulations": 96.0,
                            "fixed_mean_root_latency_ms": 6.3,
                            "dynamic_score": 0.4,
                            "fixed_score": 0.5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            current_baseline_mcts_report_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "classic_mcts_mode": "fixed",
                        "search_profile": self.classic_mcts_fixed_search_profile(),
                        "search_profile_hash": "fixed-profile-hash",
                        "games": 30,
                        "az_wins": 15,
                        "mcts_wins": 15,
                        "draws": 0,
                        "score": 0.5,
                        "budget_summary": {
                            "source": "classic_mcts_fixed_runtime",
                            "mean_final_simulations": 96,
                            "mean_root_latency_ms": 6.3,
                        },
                        "dynamic_budget_comparison": {
                            "comparison_mode": "classic_dynamic_vs_fixed",
                            "runtime_target_ms": 6.3,
                            "runtime_target_matched": True,
                            "seat_bias_neutralized": True,
                            "dynamic_mean_final_simulations": 128.0,
                            "dynamic_mean_root_latency_ms": 6.5,
                            "fixed_mean_final_simulations": 96.0,
                            "fixed_mean_root_latency_ms": 6.3,
                            "dynamic_score": 0.4,
                            "fixed_score": 0.5,
                        },
                        "classic_mcts_dynamic_budget_config": {"enabled": False},
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
            current_baseline_mcts_report_path = (
                tmp_path / "current_baseline_mcts1200_report.json"
            )

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
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "classic_mcts_mode": "dynamic",
                        "search_profile": self.classic_mcts_dynamic_search_profile(),
                        "search_profile_hash": "dynamic-profile-hash",
                        "games": 30,
                        "az_wins": 12,
                        "mcts_wins": 18,
                        "draws": 0,
                        "score": 0.4,
                        "budget_summary": {
                            "source": "classic_mcts_dynamic_runtime",
                            "mean_final_simulations": 128,
                            "mean_root_latency_ms": 6.5,
                        },
                        "classic_mcts_dynamic_budget_config": {
                            "enabled": True,
                            "probe_simulations": 12,
                            "min_simulations": 96,
                            "max_simulations": 128,
                            "entropy_weight": 0.8,
                            "low_margin_threshold": 0.2,
                            "low_margin_weight": 1.5,
                            "variance_weight": 1.5,
                        },
                        "dynamic_budget_comparison": {
                            "comparison_mode": "classic_dynamic_vs_fixed",
                            "runtime_target_ms": 6.3,
                            "runtime_target_matched": True,
                            "seat_bias_neutralized": True,
                            "dynamic_mean_final_simulations": 128.0,
                            "dynamic_mean_root_latency_ms": 6.5,
                            "fixed_mean_final_simulations": 96.0,
                            "fixed_mean_root_latency_ms": 6.3,
                            "dynamic_score": 0.4,
                            "fixed_score": 0.5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            current_baseline_mcts_report_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "classic_mcts_mode": "fixed",
                        "search_profile": self.classic_mcts_fixed_search_profile(),
                        "search_profile_hash": "fixed-profile-hash",
                        "games": 30,
                        "az_wins": 15,
                        "mcts_wins": 15,
                        "draws": 0,
                        "score": 0.5,
                        "budget_summary": {
                            "source": "classic_mcts_fixed_runtime",
                            "mean_final_simulations": 96,
                            "mean_root_latency_ms": 6.3,
                        },
                        "dynamic_budget_comparison": {
                            "comparison_mode": "classic_dynamic_vs_fixed",
                            "runtime_target_ms": 6.3,
                            "runtime_target_matched": True,
                            "seat_bias_neutralized": True,
                            "dynamic_mean_final_simulations": 128.0,
                            "dynamic_mean_root_latency_ms": 6.5,
                            "fixed_mean_final_simulations": 96.0,
                            "fixed_mean_root_latency_ms": 6.3,
                            "dynamic_score": 0.4,
                            "fixed_score": 0.5,
                        },
                        "classic_mcts_dynamic_budget_config": {"enabled": False},
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

    def test_promotion_report_includes_forensic_quality_section(self):
        from ml.alphazero_lite import benchmark

        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            arena_report = tmp_path / "arena.json"
            mcts_report = tmp_path / "mcts.json"
            baseline_report = tmp_path / "baseline.json"
            forensic_report = tmp_path / "forensic.json"

            arena_report.write_text(
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
            mcts_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "classic_mcts_mode": "dynamic",
                        "search_profile": self.classic_mcts_dynamic_search_profile(),
                        "search_profile_hash": "dynamic-profile-hash",
                        "games": 30,
                        "az_wins": 15,
                        "mcts_wins": 15,
                        "draws": 0,
                        "score": 0.5,
                        "budget_summary": {
                            "source": "classic_mcts_dynamic_runtime",
                            "mean_final_simulations": 128,
                            "mean_root_latency_ms": 6.5,
                        },
                        "classic_mcts_dynamic_budget_config": {
                            "enabled": True,
                            "probe_simulations": 12,
                            "min_simulations": 96,
                            "max_simulations": 128,
                            "entropy_weight": 0.8,
                            "low_margin_threshold": 0.2,
                            "low_margin_weight": 1.5,
                            "variance_weight": 1.5,
                        },
                        "dynamic_budget_comparison": {
                            "comparison_mode": "classic_dynamic_vs_fixed",
                            "runtime_target_ms": 6.3,
                            "runtime_target_matched": True,
                            "seat_bias_neutralized": True,
                            "dynamic_mean_final_simulations": 128.0,
                            "dynamic_mean_root_latency_ms": 6.5,
                            "fixed_mean_final_simulations": 96.0,
                            "fixed_mean_root_latency_ms": 6.3,
                            "dynamic_score": 0.5,
                            "fixed_score": 0.5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            baseline_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "classic_mcts_mode": "fixed",
                        "search_profile": self.classic_mcts_fixed_search_profile(),
                        "search_profile_hash": "fixed-profile-hash",
                        "games": 30,
                        "az_wins": 15,
                        "mcts_wins": 15,
                        "draws": 0,
                        "score": 0.5,
                        "budget_summary": {
                            "source": "classic_mcts_fixed_runtime",
                            "mean_final_simulations": 96,
                            "mean_root_latency_ms": 6.3,
                        },
                        "classic_mcts_dynamic_budget_config": {
                            "enabled": False,
                            "probe_simulations": 0,
                            "min_simulations": 1200,
                            "max_simulations": 1200,
                            "entropy_weight": 0.0,
                            "low_margin_threshold": 0.0,
                            "low_margin_weight": 0.0,
                            "variance_weight": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            forensic_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_forensic_suite_v1",
                        "systems": {
                            "current": {
                                "overall": {
                                    "top1_agreement": 0.61,
                                    "average_regret": 0.11,
                                    "blunder_rate": 0.03,
                                }
                            },
                            "challenger": {
                                "overall": {
                                    "top1_agreement": 0.64,
                                    "average_regret": 0.09,
                                    "blunder_rate": 0.02,
                                }
                            },
                        },
                        "buckets": {
                            "sparse_endgame": {
                                "systems": {
                                    "current": {
                                        "top1_agreement": 0.58,
                                        "average_regret": 0.14,
                                        "blunder_rate": 0.05,
                                    },
                                    "challenger": {
                                        "top1_agreement": 0.62,
                                        "average_regret": 0.10,
                                        "blunder_rate": 0.03,
                                    },
                                }
                            },
                            "capture_available": {
                                "systems": {
                                    "current": {
                                        "top1_agreement": 0.55,
                                        "average_regret": 0.09,
                                        "blunder_rate": 0.02,
                                    },
                                    "challenger": {
                                        "top1_agreement": 0.56,
                                        "average_regret": 0.08,
                                        "blunder_rate": 0.02,
                                    },
                                }
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            args = benchmark.parse_args(
                [
                    "--mode",
                    "promotion",
                    "--out",
                    str(tmp_path / "out.json"),
                    "--arena-report",
                    str(arena_report),
                    "--mcts-report",
                    str(mcts_report),
                    "--current-baseline-mcts-report",
                    str(baseline_report),
                    "--forensic-report",
                    str(forensic_report),
                ]
            )

            report = benchmark.build_report(args)

            self.assertIn("forensic_quality", report)
            self.assertTrue(report["forensic_quality"]["passed"])
            forensic_check = next(
                check
                for check in report["checks"]
                if check["id"] == "forensic_quality_gate"
            )
            self.assertTrue(forensic_check["passed"])

    def test_promotion_report_adds_failing_forensic_quality_check(self):
        from ml.alphazero_lite import benchmark

        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            arena_report = tmp_path / "arena.json"
            mcts_report = tmp_path / "mcts.json"
            baseline_report = tmp_path / "baseline.json"
            forensic_report = tmp_path / "forensic.json"

            arena_report.write_text(
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
            mcts_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "classic_mcts_mode": "dynamic",
                        "search_profile": self.classic_mcts_dynamic_search_profile(),
                        "search_profile_hash": "dynamic-profile-hash",
                        "games": 30,
                        "az_wins": 15,
                        "mcts_wins": 15,
                        "draws": 0,
                        "score": 0.5,
                        "budget_summary": {
                            "source": "classic_mcts_dynamic_runtime",
                            "mean_final_simulations": 128,
                            "mean_root_latency_ms": 6.5,
                        },
                        "classic_mcts_dynamic_budget_config": {
                            "enabled": True,
                            "probe_simulations": 12,
                            "min_simulations": 96,
                            "max_simulations": 128,
                            "entropy_weight": 0.8,
                            "low_margin_threshold": 0.2,
                            "low_margin_weight": 1.5,
                            "variance_weight": 1.5,
                        },
                        "dynamic_budget_comparison": {
                            "comparison_mode": "classic_dynamic_vs_fixed",
                            "runtime_target_ms": 6.3,
                            "runtime_target_matched": True,
                            "seat_bias_neutralized": True,
                            "dynamic_mean_final_simulations": 128.0,
                            "dynamic_mean_root_latency_ms": 6.5,
                            "fixed_mean_final_simulations": 96.0,
                            "fixed_mean_root_latency_ms": 6.3,
                            "dynamic_score": 0.5,
                            "fixed_score": 0.5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            baseline_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "classic_mcts_mode": "fixed",
                        "search_profile": self.classic_mcts_fixed_search_profile(),
                        "search_profile_hash": "fixed-profile-hash",
                        "games": 30,
                        "az_wins": 15,
                        "mcts_wins": 15,
                        "draws": 0,
                        "score": 0.5,
                        "budget_summary": {
                            "source": "classic_mcts_fixed_runtime",
                            "mean_final_simulations": 96,
                            "mean_root_latency_ms": 6.3,
                        },
                        "classic_mcts_dynamic_budget_config": {
                            "enabled": False,
                            "probe_simulations": 0,
                            "min_simulations": 1200,
                            "max_simulations": 1200,
                            "entropy_weight": 0.0,
                            "low_margin_threshold": 0.0,
                            "low_margin_weight": 0.0,
                            "variance_weight": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            forensic_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_forensic_suite_v1",
                        "systems": {
                            "current": {
                                "overall": {
                                    "top1_agreement": 0.61,
                                    "average_regret": 0.11,
                                    "blunder_rate": 0.03,
                                }
                            },
                            "challenger": {
                                "overall": {
                                    "top1_agreement": 0.64,
                                    "average_regret": 0.09,
                                    "blunder_rate": 0.02,
                                }
                            },
                        },
                        "buckets": {
                            "sparse_endgame": {
                                "systems": {
                                    "current": {
                                        "top1_agreement": 0.58,
                                        "average_regret": 0.14,
                                        "blunder_rate": 0.05,
                                    },
                                    "challenger": {
                                        "top1_agreement": 0.58,
                                        "average_regret": 0.18,
                                        "blunder_rate": 0.05,
                                    },
                                }
                            },
                            "capture_available": {
                                "systems": {
                                    "current": {
                                        "top1_agreement": 0.55,
                                        "average_regret": 0.09,
                                        "blunder_rate": 0.02,
                                    },
                                    "challenger": {
                                        "top1_agreement": 0.56,
                                        "average_regret": 0.08,
                                        "blunder_rate": 0.02,
                                    },
                                }
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            args = benchmark.parse_args(
                [
                    "--mode",
                    "promotion",
                    "--out",
                    str(tmp_path / "out.json"),
                    "--arena-report",
                    str(arena_report),
                    "--mcts-report",
                    str(mcts_report),
                    "--current-baseline-mcts-report",
                    str(baseline_report),
                    "--forensic-report",
                    str(forensic_report),
                ]
            )

            report = benchmark.build_report(args)

            forensic_check = next(
                check
                for check in report["checks"]
                if check["id"] == "forensic_quality_gate"
            )
            self.assertFalse(forensic_check["passed"])

    def test_promotion_dry_run_allows_future_forensic_report_path(self):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            future_forensic_report_path = tmp_path / "future_forensic_report.json"

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
                    "--dry-run",
                    "--forensic-report",
                    str(future_forensic_report_path),
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
            self.assertEqual(
                str(future_forensic_report_path), report["forensic_report"]
            )
            self.assertIsNone(report["forensic_quality"])
            self.assertFalse(
                any(
                    check["id"] == "forensic_quality_gate" for check in report["checks"]
                )
            )

    def test_promotion_report_rejects_missing_forensic_report_when_requested(self):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            arena_report_path = tmp_path / "arena_report.json"
            mcts_report_path = tmp_path / "mcts1200_report.json"
            missing_forensic_report_path = tmp_path / "missing_forensic_report.json"

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
                    "--forensic-report",
                    str(missing_forensic_report_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("forensic report not found", result.stderr)

    def test_promotion_report_rejects_malformed_forensic_report(self):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            arena_report_path = tmp_path / "arena_report.json"
            mcts_report_path = tmp_path / "mcts1200_report.json"
            forensic_report_path = tmp_path / "forensic_report.json"

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
            forensic_report_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_forensic_suite_v1",
                        "systems": {
                            "current": {
                                "overall": {
                                    "top1_agreement": 0.61,
                                    "average_regret": 0.11,
                                    "blunder_rate": 0.03,
                                }
                            },
                            "challenger": {
                                "overall": {
                                    "top1_agreement": 0.64,
                                    "average_regret": 0.09,
                                    "blunder_rate": 0.02,
                                }
                            },
                        },
                        "buckets": {
                            "capture_available": {
                                "systems": {
                                    "current": {
                                        "top1_agreement": 0.55,
                                        "average_regret": 0.09,
                                        "blunder_rate": 0.02,
                                    },
                                    "challenger": {
                                        "top1_agreement": 0.56,
                                        "average_regret": 0.08,
                                        "blunder_rate": 0.02,
                                    },
                                }
                            }
                        },
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
                    "--forensic-report",
                    str(forensic_report_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("missing required forensic bucket", result.stderr)

    def test_promotion_report_rejects_non_classic_dynamic_budget_comparison_input(self):
        from ml.alphazero_lite import benchmark

        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            arena_report = tmp_path / "arena.json"
            mcts_report = tmp_path / "mcts.json"
            baseline_report = tmp_path / "baseline.json"

            arena_report.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "wins": 36,
                        "losses": 12,
                        "draws": 12,
                        "games_played": 60,
                        "promotion_decision": {"passed": True},
                        "budget_summary": {
                            "mean_final_simulations": 128,
                            "p95_root_latency_ms": 8.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            mcts_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 40,
                        "az_wins": 22,
                        "mcts_wins": 10,
                        "draws": 8,
                        "budget_summary": {
                            "source": "challenger_puct_runtime",
                            "mean_final_simulations": 128,
                            "mean_root_latency_ms": 6.5,
                        },
                        "classic_mcts_dynamic_budget_config": {
                            "enabled": True,
                            "probe_simulations": 12,
                            "min_simulations": 24,
                            "max_simulations": 96,
                            "entropy_weight": 0.75,
                            "low_margin_threshold": 0.18,
                            "low_margin_weight": 1.25,
                            "variance_weight": 1.1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            baseline_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 40,
                        "az_wins": 20,
                        "mcts_wins": 12,
                        "draws": 8,
                        "budget_summary": {
                            "source": "challenger_puct_runtime",
                            "mean_final_simulations": 96,
                            "mean_root_latency_ms": 6.3,
                        },
                        "classic_mcts_dynamic_budget_config": {
                            "enabled": False,
                            "probe_simulations": 0,
                            "min_simulations": None,
                            "max_simulations": None,
                            "entropy_weight": 0.8,
                            "low_margin_threshold": 0.2,
                            "low_margin_weight": 1.5,
                            "variance_weight": 1.5,
                        },
                    }
                ),
                encoding="utf-8",
            )

            args = benchmark.parse_args(
                [
                    "--mode",
                    "promotion",
                    "--out",
                    str(tmp_path / "out.json"),
                    "--arena-report",
                    str(arena_report),
                    "--mcts-report",
                    str(mcts_report),
                    "--current-baseline-mcts-report",
                    str(baseline_report),
                ]
            )
            with self.assertRaisesRegex(SystemExit, "candidate comparison_mode"):
                benchmark.build_report(args)

    def test_promotion_report_skips_dynamic_budget_comparison_for_fixed_vs_fixed_baseline_inputs(
        self,
    ):
        from ml.alphazero_lite import benchmark

        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            arena_report = tmp_path / "arena.json"
            mcts_report = tmp_path / "mcts.json"
            baseline_report = tmp_path / "baseline.json"

            arena_report.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "wins": 36,
                        "losses": 12,
                        "draws": 12,
                        "games_played": 60,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            mcts_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "classic_mcts_mode": "fixed",
                        "games": 30,
                        "az_wins": 18,
                        "mcts_wins": 11,
                        "draws": 1,
                        "score": 0.6167,
                        "budget_summary": {
                            "source": "classic_mcts_fixed_runtime",
                            "mean_final_simulations": 1148.94,
                            "mean_root_latency_ms": 133.77,
                        },
                        "classic_mcts_dynamic_budget_config": {
                            "enabled": False,
                            "probe_simulations": 0,
                            "min_simulations": 1200,
                            "max_simulations": 1200,
                            "entropy_weight": 0.0,
                            "low_margin_threshold": 0.0,
                            "low_margin_weight": 0.0,
                            "variance_weight": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            baseline_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "classic_mcts_mode": "fixed",
                        "games": 30,
                        "az_wins": 18,
                        "mcts_wins": 11,
                        "draws": 1,
                        "score": 0.6167,
                        "budget_summary": {
                            "source": "classic_mcts_fixed_runtime",
                            "mean_final_simulations": 1148.94,
                            "mean_root_latency_ms": 132.13,
                        },
                        "classic_mcts_dynamic_budget_config": {
                            "enabled": False,
                            "probe_simulations": 0,
                            "min_simulations": 1200,
                            "max_simulations": 1200,
                            "entropy_weight": 0.0,
                            "low_margin_threshold": 0.0,
                            "low_margin_weight": 0.0,
                            "variance_weight": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            args = benchmark.parse_args(
                [
                    "--mode",
                    "promotion",
                    "--out",
                    str(tmp_path / "out.json"),
                    "--arena-report",
                    str(arena_report),
                    "--mcts-report",
                    str(mcts_report),
                    "--current-baseline-mcts-report",
                    str(baseline_report),
                ]
            )
            report = benchmark.build_report(args)

        mcts_check = report["checks"][1]
        self.assertEqual("current_baseline", mcts_check["comparison"])
        self.assertEqual(0.6167, mcts_check["score"])
        self.assertEqual(0.6167, mcts_check["baseline_score"])
        self.assertTrue(mcts_check["passed"])
        self.assertIsNone(report["dynamic_budget_comparison"])
        self.assertEqual(
            "classic_mcts_fixed_runtime",
            report["dynamic_budget_metric_source"]["candidate"],
        )
        self.assertEqual(
            "classic_mcts_fixed_runtime",
            report["dynamic_budget_metric_source"]["baseline"],
        )

    def test_dynamic_budget_comparison_requires_explicit_fixed_vs_dynamic_classic_mcts_reports(
        self,
    ):
        from ml.alphazero_lite import benchmark

        with self.assertRaisesRegex(SystemExit, "comparison_mode"):
            benchmark.dynamic_budget_comparison(
                {
                    "budget_summary": {
                        "mean_final_simulations": 128,
                        "mean_root_latency_ms": 6.5,
                    },
                },
                {
                    "budget_summary": {
                        "mean_final_simulations": 96,
                        "mean_root_latency_ms": 6.3,
                    },
                },
            )

    def test_dynamic_budget_comparison_rejects_unexpected_mode_when_strict(self):
        from ml.alphazero_lite import benchmark

        with self.assertRaisesRegex(SystemExit, "candidate comparison_mode"):
            benchmark.dynamic_budget_comparison(
                {
                    "comparison_mode": "unexpected_mode",
                    "classic_mcts_mode": "dynamic",
                    "search_profile": self.classic_mcts_dynamic_search_profile(),
                    "games": 30,
                    "az_wins": 15,
                    "mcts_wins": 15,
                    "draws": 0,
                    "score": 0.5,
                    "budget_summary": {
                        "source": "classic_mcts_dynamic_runtime",
                        "mean_final_simulations": 128,
                        "mean_root_latency_ms": 6.5,
                    },
                    "dynamic_budget_comparison": {
                        "comparison_mode": "classic_dynamic_vs_fixed"
                    },
                    "classic_mcts_dynamic_budget_config": {"enabled": True},
                },
                {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "classic_mcts_mode": "fixed",
                    "search_profile": self.classic_mcts_fixed_search_profile(),
                    "games": 30,
                    "az_wins": 15,
                    "mcts_wins": 15,
                    "draws": 0,
                    "score": 0.5,
                    "budget_summary": {
                        "source": "classic_mcts_fixed_runtime",
                        "mean_final_simulations": 96,
                        "mean_root_latency_ms": 6.3,
                    },
                    "classic_mcts_dynamic_budget_config": {"enabled": False},
                },
                strict=True,
            )

    def test_dynamic_budget_comparison_reports_runtime_target_match_for_fixed_vs_dynamic_classic_mcts(
        self,
    ):
        from ml.alphazero_lite import benchmark

        report = benchmark.dynamic_budget_comparison(
            {
                "comparison_mode": "classic_dynamic_vs_fixed",
                "classic_mcts_mode": "dynamic",
                "search_profile": self.classic_mcts_dynamic_search_profile(),
                "search_profile_hash": "dynamic-profile-hash",
                "games": 25,
                "az_wins": 13,
                "mcts_wins": 12,
                "draws": 0,
                "score": 0.52,
                "budget_summary": {
                    "source": "classic_mcts_dynamic_runtime",
                    "mean_final_simulations": 128,
                    "mean_root_latency_ms": 6.5,
                },
                "dynamic_budget_comparison": {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "runtime_target_ms": 6.3,
                    "runtime_target_matched": True,
                    "seat_bias_neutralized": True,
                    "dynamic_mean_final_simulations": 128.0,
                    "dynamic_mean_root_latency_ms": 6.5,
                    "fixed_mean_final_simulations": 96.0,
                    "fixed_mean_root_latency_ms": 6.3,
                    "dynamic_score": 0.52,
                    "fixed_score": 0.49,
                },
                "classic_mcts_dynamic_budget_config": {"enabled": True},
            },
            {
                "comparison_mode": "classic_dynamic_vs_fixed",
                "classic_mcts_mode": "fixed",
                "search_profile": self.classic_mcts_fixed_search_profile(),
                "search_profile_hash": "fixed-profile-hash",
                "games": 100,
                "az_wins": 49,
                "mcts_wins": 51,
                "draws": 0,
                "score": 0.49,
                "budget_summary": {
                    "source": "classic_mcts_fixed_runtime",
                    "mean_final_simulations": 96,
                    "mean_root_latency_ms": 6.3,
                },
                "dynamic_budget_comparison": {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "runtime_target_ms": 6.3,
                    "runtime_target_matched": True,
                    "seat_bias_neutralized": True,
                    "dynamic_mean_final_simulations": 128.0,
                    "dynamic_mean_root_latency_ms": 6.5,
                    "fixed_mean_final_simulations": 96.0,
                    "fixed_mean_root_latency_ms": 6.3,
                    "dynamic_score": 0.52,
                    "fixed_score": 0.49,
                },
                "classic_mcts_dynamic_budget_config": {"enabled": False},
            },
        )

        self.assertEqual(
            {
                "comparison_mode": "classic_dynamic_vs_fixed",
                "runtime_target_ms": 6.3,
                "runtime_target_matched": True,
                "seat_bias_neutralized": True,
                "dynamic_mean_final_simulations": 128.0,
                "dynamic_mean_root_latency_ms": 6.5,
                "fixed_mean_final_simulations": 96.0,
                "fixed_mean_root_latency_ms": 6.3,
                "dynamic_score": 0.52,
                "fixed_score": 0.49,
            },
            report,
        )

    def test_dynamic_budget_comparison_accepts_real_fixed_baseline_without_embedded_comparison_payload(
        self,
    ):
        from ml.alphazero_lite import benchmark

        report = benchmark.dynamic_budget_comparison(
            {
                "comparison_mode": "classic_dynamic_vs_fixed",
                "classic_mcts_mode": "dynamic",
                "search_profile": self.classic_mcts_dynamic_search_profile(),
                "search_profile_hash": "dynamic-profile-hash",
                "games": 25,
                "az_wins": 13,
                "mcts_wins": 12,
                "draws": 0,
                "score": 0.52,
                "budget_summary": {
                    "source": "classic_mcts_dynamic_runtime",
                    "mean_final_simulations": 128,
                    "mean_root_latency_ms": 6.5,
                },
                "dynamic_budget_comparison": {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "runtime_target_ms": 6.3,
                    "runtime_target_matched": True,
                    "seat_bias_neutralized": True,
                    "dynamic_mean_final_simulations": 128.0,
                    "dynamic_mean_root_latency_ms": 6.5,
                    "fixed_mean_final_simulations": 96.0,
                    "fixed_mean_root_latency_ms": 6.3,
                    "dynamic_score": 0.52,
                    "fixed_score": 0.49,
                },
                "classic_mcts_dynamic_budget_config": {"enabled": True},
            },
            {
                "comparison_mode": "classic_dynamic_vs_fixed",
                "classic_mcts_mode": "fixed",
                "search_profile": self.classic_mcts_fixed_search_profile(),
                "search_profile_hash": "fixed-profile-hash",
                "games": 100,
                "az_wins": 49,
                "mcts_wins": 51,
                "draws": 0,
                "score": 0.49,
                "budget_summary": {
                    "source": "classic_mcts_fixed_runtime",
                    "mean_final_simulations": 96,
                    "mean_root_latency_ms": 6.3,
                },
                "classic_mcts_dynamic_budget_config": {
                    "enabled": False,
                    "probe_simulations": 0,
                    "min_simulations": 1200,
                    "max_simulations": 1200,
                    "entropy_weight": 0.0,
                    "low_margin_threshold": 0.0,
                    "low_margin_weight": 0.0,
                    "variance_weight": 0.0,
                },
            },
        )

        self.assertEqual(6.3, report["runtime_target_ms"])
        self.assertEqual(96.0, report["fixed_mean_final_simulations"])
        self.assertEqual(0.49, report["fixed_score"])

    def test_dynamic_budget_comparison_accepts_matching_producer_reports_with_different_profile_hashes(
        self,
    ):
        from ml.alphazero_lite import benchmark

        report = benchmark.dynamic_budget_comparison(
            {
                "comparison_mode": "classic_dynamic_vs_fixed",
                "classic_mcts_mode": "dynamic",
                "search_profile": {
                    "kind": "mcts1200_baseline_eval",
                    "player_mode": "classic_mcts",
                    "simulation_budget_policy": "fixed_vs_dynamic_classic_mcts",
                    "classic_mcts_simulations": 1200,
                    "az_base_simulations": 640,
                    "mcts_simulations": 1200,
                },
                "search_profile_hash": "dynamic-profile-hash",
                "games": 25,
                "az_wins": 13,
                "mcts_wins": 12,
                "draws": 0,
                "score": 0.52,
                "budget_summary": {
                    "source": "classic_mcts_dynamic_runtime",
                    "mean_final_simulations": 128,
                    "mean_root_latency_ms": 6.5,
                },
                "dynamic_budget_comparison": {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "runtime_target_ms": 6.3,
                    "runtime_target_matched": True,
                    "seat_bias_neutralized": True,
                    "dynamic_mean_final_simulations": 128.0,
                    "dynamic_mean_root_latency_ms": 6.5,
                    "fixed_mean_final_simulations": 96.0,
                    "fixed_mean_root_latency_ms": 6.3,
                    "dynamic_score": 0.52,
                    "fixed_score": 0.49,
                },
                "classic_mcts_dynamic_budget_config": {
                    "enabled": True,
                    "probe_simulations": 12,
                    "min_simulations": 24,
                    "max_simulations": 96,
                    "entropy_weight": 0.75,
                    "low_margin_threshold": 0.18,
                    "low_margin_weight": 1.25,
                    "variance_weight": 1.1,
                },
            },
            {
                "classic_mcts_mode": "fixed",
                "search_profile": {
                    "kind": "mcts1200_baseline_eval",
                    "player_mode": "classic_mcts",
                    "simulation_budget_policy": "fixed_classic_mcts",
                    "classic_mcts_simulations": 1200,
                    "az_base_simulations": 640,
                    "mcts_simulations": 1200,
                },
                "search_profile_hash": "fixed-profile-hash",
                "games": 100,
                "az_wins": 49,
                "mcts_wins": 51,
                "draws": 0,
                "score": 0.49,
                "budget_summary": {
                    "source": "classic_mcts_fixed_runtime",
                    "mean_final_simulations": 96,
                    "mean_root_latency_ms": 6.3,
                },
                "classic_mcts_dynamic_budget_config": {
                    "enabled": False,
                    "probe_simulations": 0,
                    "min_simulations": 1200,
                    "max_simulations": 1200,
                    "entropy_weight": 0.0,
                    "low_margin_threshold": 0.0,
                    "low_margin_weight": 0.0,
                    "variance_weight": 0.0,
                },
            },
        )

        self.assertEqual(0.49, report["fixed_score"])

    def test_dynamic_budget_comparison_rejects_missing_producer_provenance_metadata(
        self,
    ):
        from ml.alphazero_lite import benchmark

        with self.assertRaisesRegex(SystemExit, "producer provenance metadata"):
            benchmark.dynamic_budget_comparison(
                {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "classic_mcts_mode": "dynamic",
                    "score": 0.52,
                    "budget_summary": {
                        "source": "classic_mcts_dynamic_runtime",
                        "mean_final_simulations": 128,
                        "mean_root_latency_ms": 6.5,
                    },
                    "dynamic_budget_comparison": {
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "runtime_target_ms": 6.3,
                        "runtime_target_matched": True,
                        "seat_bias_neutralized": True,
                        "dynamic_mean_final_simulations": 128.0,
                        "dynamic_mean_root_latency_ms": 6.5,
                        "fixed_mean_final_simulations": 96.0,
                        "fixed_mean_root_latency_ms": 6.3,
                        "dynamic_score": 0.52,
                        "fixed_score": 0.49,
                    },
                    "classic_mcts_dynamic_budget_config": {
                        "enabled": True,
                        "probe_simulations": 12,
                        "min_simulations": 24,
                        "max_simulations": 96,
                        "entropy_weight": 0.75,
                        "low_margin_threshold": 0.18,
                        "low_margin_weight": 1.25,
                        "variance_weight": 1.1,
                    },
                },
                {
                    "classic_mcts_mode": "fixed",
                    "score": 0.49,
                    "budget_summary": {
                        "source": "classic_mcts_fixed_runtime",
                        "mean_final_simulations": 96,
                        "mean_root_latency_ms": 6.3,
                    },
                    "classic_mcts_dynamic_budget_config": {
                        "enabled": False,
                        "probe_simulations": 0,
                        "min_simulations": 1200,
                        "max_simulations": 1200,
                        "entropy_weight": 0.0,
                        "low_margin_threshold": 0.0,
                        "low_margin_weight": 0.0,
                        "variance_weight": 0.0,
                    },
                },
            )

    def test_dynamic_budget_comparison_rejects_baseline_from_different_fixed_arm_configuration(
        self,
    ):
        from ml.alphazero_lite import benchmark

        with self.assertRaisesRegex(SystemExit, "matching fixed arm configuration"):
            benchmark.dynamic_budget_comparison(
                {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "classic_mcts_mode": "dynamic",
                    "search_profile": {
                        "kind": "mcts1200_baseline_eval",
                        "player_mode": "classic_mcts",
                        "classic_mcts_simulations": 1200,
                        "az_base_simulations": 640,
                        "mcts_simulations": 1200,
                        "exact_solve_enabled": False,
                        "exact_solve_stone_threshold": None,
                        "simulation_budget_policy": "fixed_vs_dynamic_classic_mcts",
                    },
                    "search_profile_hash": "shared-profile-hash",
                    "score": 0.52,
                    "budget_summary": {
                        "source": "classic_mcts_dynamic_runtime",
                        "mean_final_simulations": 128,
                        "mean_root_latency_ms": 6.5,
                    },
                    "dynamic_budget_comparison": {
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "runtime_target_ms": 6.3,
                        "runtime_target_matched": True,
                        "seat_bias_neutralized": True,
                        "dynamic_mean_final_simulations": 128.0,
                        "dynamic_mean_root_latency_ms": 6.5,
                        "fixed_mean_final_simulations": 96.0,
                        "fixed_mean_root_latency_ms": 6.3,
                        "dynamic_score": 0.52,
                        "fixed_score": 0.49,
                    },
                    "classic_mcts_dynamic_budget_config": {
                        "enabled": True,
                        "probe_simulations": 12,
                        "min_simulations": 24,
                        "max_simulations": 96,
                        "entropy_weight": 0.75,
                        "low_margin_threshold": 0.18,
                        "low_margin_weight": 1.25,
                        "variance_weight": 1.1,
                    },
                },
                {
                    "classic_mcts_mode": "fixed",
                    "search_profile": {
                        "kind": "mcts1200_baseline_eval",
                        "player_mode": "classic_mcts",
                        "classic_mcts_simulations": 1200,
                        "az_base_simulations": 640,
                        "mcts_simulations": 999,
                        "exact_solve_enabled": False,
                        "exact_solve_stone_threshold": None,
                        "simulation_budget_policy": "fixed_classic_mcts",
                    },
                    "search_profile_hash": "different-profile-hash",
                    "score": 0.49,
                    "budget_summary": {
                        "source": "classic_mcts_fixed_runtime",
                        "mean_final_simulations": 96,
                        "mean_root_latency_ms": 6.3,
                    },
                    "classic_mcts_dynamic_budget_config": {
                        "enabled": False,
                        "probe_simulations": 0,
                        "min_simulations": 24,
                        "max_simulations": 96,
                        "entropy_weight": 0.0,
                        "low_margin_threshold": 0.0,
                        "low_margin_weight": 0.0,
                        "variance_weight": 0.0,
                    },
                },
            )

    def test_dynamic_budget_comparison_rejects_baseline_with_mismatched_exact_solve_invariant(
        self,
    ):
        from ml.alphazero_lite import benchmark

        with self.assertRaisesRegex(SystemExit, "matching fixed arm configuration"):
            benchmark.dynamic_budget_comparison(
                {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "classic_mcts_mode": "dynamic",
                    "search_profile": {
                        "kind": "mcts1200_baseline_eval",
                        "player_mode": "classic_mcts",
                        "classic_mcts_simulations": 1200,
                        "az_base_simulations": 640,
                        "mcts_simulations": 1200,
                        "exact_solve_enabled": True,
                        "exact_solve_stone_threshold": 10,
                        "simulation_budget_policy": "fixed_vs_dynamic_classic_mcts",
                    },
                    "search_profile_hash": "dynamic-profile-hash",
                    "score": 0.52,
                    "budget_summary": {
                        "source": "classic_mcts_dynamic_runtime",
                        "mean_final_simulations": 128,
                        "mean_root_latency_ms": 6.5,
                    },
                    "dynamic_budget_comparison": {
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "runtime_target_ms": 6.3,
                        "runtime_target_matched": True,
                        "seat_bias_neutralized": True,
                        "dynamic_mean_final_simulations": 128.0,
                        "dynamic_mean_root_latency_ms": 6.5,
                        "fixed_mean_final_simulations": 96.0,
                        "fixed_mean_root_latency_ms": 6.3,
                        "dynamic_score": 0.52,
                        "fixed_score": 0.49,
                    },
                    "classic_mcts_dynamic_budget_config": {
                        "enabled": True,
                        "probe_simulations": 12,
                        "min_simulations": 24,
                        "max_simulations": 96,
                        "entropy_weight": 0.75,
                        "low_margin_threshold": 0.18,
                        "low_margin_weight": 1.25,
                        "variance_weight": 1.1,
                    },
                },
                {
                    "classic_mcts_mode": "fixed",
                    "search_profile": {
                        "kind": "mcts1200_baseline_eval",
                        "player_mode": "classic_mcts",
                        "classic_mcts_simulations": 1200,
                        "az_base_simulations": 640,
                        "mcts_simulations": 1200,
                        "exact_solve_enabled": False,
                        "exact_solve_stone_threshold": None,
                        "simulation_budget_policy": "fixed_classic_mcts",
                    },
                    "search_profile_hash": "fixed-profile-hash",
                    "score": 0.49,
                    "budget_summary": {
                        "source": "classic_mcts_fixed_runtime",
                        "mean_final_simulations": 96,
                        "mean_root_latency_ms": 6.3,
                    },
                    "classic_mcts_dynamic_budget_config": {
                        "enabled": False,
                        "probe_simulations": 0,
                        "min_simulations": 1200,
                        "max_simulations": 1200,
                        "entropy_weight": 0.0,
                        "low_margin_threshold": 0.0,
                        "low_margin_weight": 0.0,
                        "variance_weight": 0.0,
                    },
                },
            )

    def test_dynamic_budget_comparison_preserves_embedded_seat_bias_neutralization_value(
        self,
    ):
        from ml.alphazero_lite import benchmark

        report = benchmark.dynamic_budget_comparison(
            {
                "comparison_mode": "classic_dynamic_vs_fixed",
                "classic_mcts_mode": "dynamic",
                "search_profile": self.classic_mcts_dynamic_search_profile(),
                "search_profile_hash": "dynamic-profile-hash",
                "games": 25,
                "az_wins": 13,
                "mcts_wins": 12,
                "draws": 0,
                "score": 0.52,
                "budget_summary": {
                    "source": "classic_mcts_dynamic_runtime",
                    "mean_final_simulations": 128,
                    "mean_root_latency_ms": 6.5,
                },
                "dynamic_budget_comparison": {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "runtime_target_ms": 6.3,
                    "runtime_target_matched": True,
                    "seat_bias_neutralized": True,
                    "dynamic_mean_final_simulations": 128.0,
                    "dynamic_mean_root_latency_ms": 6.5,
                    "fixed_mean_final_simulations": 96.0,
                    "fixed_mean_root_latency_ms": 6.3,
                    "dynamic_score": 0.52,
                    "fixed_score": 0.49,
                },
                "classic_mcts_dynamic_budget_config": {
                    "enabled": True,
                    "probe_simulations": 12,
                    "min_simulations": 24,
                    "max_simulations": 96,
                    "entropy_weight": 0.75,
                    "low_margin_threshold": 0.18,
                    "low_margin_weight": 1.25,
                    "variance_weight": 1.1,
                },
            },
            {
                "comparison_mode": "classic_dynamic_vs_fixed",
                "classic_mcts_mode": "fixed",
                "search_profile": self.classic_mcts_fixed_search_profile(),
                "search_profile_hash": "fixed-profile-hash",
                "games": 100,
                "az_wins": 49,
                "mcts_wins": 51,
                "draws": 0,
                "score": 0.49,
                "budget_summary": {
                    "source": "classic_mcts_fixed_runtime",
                    "mean_final_simulations": 96,
                    "mean_root_latency_ms": 6.3,
                },
                "dynamic_budget_comparison": {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "runtime_target_ms": 6.3,
                    "runtime_target_matched": True,
                    "seat_bias_neutralized": True,
                    "dynamic_mean_final_simulations": 128.0,
                    "dynamic_mean_root_latency_ms": 6.5,
                    "fixed_mean_final_simulations": 96.0,
                    "fixed_mean_root_latency_ms": 6.3,
                    "dynamic_score": 0.52,
                    "fixed_score": 0.49,
                },
                "classic_mcts_dynamic_budget_config": {
                    "enabled": False,
                    "probe_simulations": 0,
                    "min_simulations": 1200,
                    "max_simulations": 1200,
                    "entropy_weight": 0.0,
                    "low_margin_threshold": 0.0,
                    "low_margin_weight": 0.0,
                    "variance_weight": 0.0,
                },
            },
        )

        self.assertIs(True, report["seat_bias_neutralized"])

    def test_dynamic_budget_comparison_accepts_derived_non_neutralized_seat_bias(self):
        from ml.alphazero_lite import benchmark

        report = benchmark.dynamic_budget_comparison(
            {
                "comparison_mode": "classic_dynamic_vs_fixed",
                "classic_mcts_mode": "dynamic",
                "search_profile": self.classic_mcts_dynamic_search_profile(),
                "search_profile_hash": "dynamic-profile-hash",
                "games": 25,
                "az_wins": 13,
                "mcts_wins": 12,
                "draws": 0,
                "score": 0.52,
                "budget_summary": {
                    "source": "classic_mcts_dynamic_runtime",
                    "mean_final_simulations": 128,
                    "mean_root_latency_ms": 6.5,
                },
                "dynamic_budget_comparison": {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "runtime_target_ms": 6.3,
                    "runtime_target_matched": True,
                    "seat_bias_neutralized": False,
                    "dynamic_mean_final_simulations": 128.0,
                    "dynamic_mean_root_latency_ms": 6.5,
                    "fixed_mean_final_simulations": 96.0,
                    "fixed_mean_root_latency_ms": 6.3,
                    "dynamic_score": 0.52,
                    "fixed_score": 0.49,
                },
                "classic_mcts_dynamic_budget_config": {"enabled": True},
            },
            {
                "comparison_mode": "classic_dynamic_vs_fixed",
                "classic_mcts_mode": "fixed",
                "search_profile": self.classic_mcts_fixed_search_profile(),
                "search_profile_hash": "fixed-profile-hash",
                "games": 100,
                "az_wins": 49,
                "mcts_wins": 51,
                "draws": 0,
                "score": 0.49,
                "budget_summary": {
                    "source": "classic_mcts_fixed_runtime",
                    "mean_final_simulations": 96,
                    "mean_root_latency_ms": 6.3,
                },
                "dynamic_budget_comparison": {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "runtime_target_ms": 6.3,
                    "runtime_target_matched": True,
                    "seat_bias_neutralized": False,
                    "dynamic_mean_final_simulations": 128.0,
                    "dynamic_mean_root_latency_ms": 6.5,
                    "fixed_mean_final_simulations": 96.0,
                    "fixed_mean_root_latency_ms": 6.3,
                    "dynamic_score": 0.52,
                    "fixed_score": 0.49,
                },
                "classic_mcts_dynamic_budget_config": {"enabled": False},
            },
        )

        self.assertIs(False, report["seat_bias_neutralized"])

    def test_dynamic_budget_comparison_rejects_untruthful_candidate_comparison_payload(
        self,
    ):
        from ml.alphazero_lite import benchmark

        with self.assertRaisesRegex(
            SystemExit, "truthful fixed-vs-dynamic ClassicMCTS comparison data"
        ):
            benchmark.dynamic_budget_comparison(
                {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "classic_mcts_mode": "dynamic",
                    "search_profile": self.classic_mcts_dynamic_search_profile(),
                    "search_profile_hash": "dynamic-profile-hash",
                    "games": 25,
                    "az_wins": 13,
                    "mcts_wins": 12,
                    "draws": 0,
                    "score": 0.52,
                    "budget_summary": {
                        "source": "classic_mcts_dynamic_runtime",
                        "mean_final_simulations": 128,
                        "mean_root_latency_ms": 6.5,
                    },
                    "dynamic_budget_comparison": {
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "runtime_target_ms": 6.3,
                        "runtime_target_matched": True,
                        "dynamic_mean_final_simulations": 128.0,
                        "dynamic_mean_root_latency_ms": 6.5,
                        "fixed_mean_final_simulations": 95.0,
                        "fixed_mean_root_latency_ms": 6.3,
                        "dynamic_score": 0.52,
                        "fixed_score": 0.49,
                        "seat_bias_neutralized": False,
                    },
                    "classic_mcts_dynamic_budget_config": {"enabled": True},
                },
                {
                    "classic_mcts_mode": "fixed",
                    "search_profile": self.classic_mcts_fixed_search_profile(),
                    "search_profile_hash": "fixed-profile-hash",
                    "games": 100,
                    "az_wins": 49,
                    "mcts_wins": 51,
                    "draws": 0,
                    "score": 0.49,
                    "budget_summary": {
                        "source": "classic_mcts_fixed_runtime",
                        "mean_final_simulations": 96,
                        "mean_root_latency_ms": 6.3,
                    },
                    "classic_mcts_dynamic_budget_config": {"enabled": False},
                },
            )

    def test_dynamic_budget_comparison_rejects_missing_score_fields_with_controlled_error(
        self,
    ):
        from ml.alphazero_lite import benchmark

        with self.assertRaisesRegex(SystemExit, "score fields"):
            benchmark.dynamic_budget_comparison(
                {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "classic_mcts_mode": "dynamic",
                    "search_profile": {
                        "kind": "mcts1200_baseline_eval",
                        "simulation_budget_policy": "fixed_vs_dynamic_classic_mcts",
                    },
                    "search_profile_hash": "shared-profile-hash",
                    "budget_summary": {
                        "source": "classic_mcts_dynamic_runtime",
                        "mean_final_simulations": 128,
                        "mean_root_latency_ms": 6.5,
                    },
                    "dynamic_budget_comparison": {
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "runtime_target_ms": 6.3,
                        "runtime_target_matched": True,
                        "seat_bias_neutralized": True,
                        "dynamic_mean_final_simulations": 128.0,
                        "dynamic_mean_root_latency_ms": 6.5,
                        "fixed_mean_final_simulations": 96.0,
                        "fixed_mean_root_latency_ms": 6.3,
                        "dynamic_score": 0.52,
                        "fixed_score": 0.49,
                    },
                    "classic_mcts_dynamic_budget_config": {
                        "enabled": True,
                        "probe_simulations": 12,
                        "min_simulations": 24,
                        "max_simulations": 96,
                        "entropy_weight": 0.75,
                        "low_margin_threshold": 0.18,
                        "low_margin_weight": 1.25,
                        "variance_weight": 1.1,
                    },
                },
                {
                    "classic_mcts_mode": "fixed",
                    "search_profile": {
                        "kind": "mcts1200_baseline_eval",
                        "simulation_budget_policy": "fixed_classic_mcts",
                    },
                    "search_profile_hash": "shared-profile-hash",
                    "budget_summary": {
                        "source": "classic_mcts_fixed_runtime",
                        "mean_final_simulations": 96,
                        "mean_root_latency_ms": 6.3,
                    },
                    "classic_mcts_dynamic_budget_config": {
                        "enabled": False,
                        "probe_simulations": 0,
                        "min_simulations": 24,
                        "max_simulations": 96,
                        "entropy_weight": 0.0,
                        "low_margin_threshold": 0.0,
                        "low_margin_weight": 0.0,
                        "variance_weight": 0.0,
                    },
                },
            )

    def test_dynamic_budget_comparison_rejects_score_fields_that_do_not_match_raw_counts(
        self,
    ):
        from ml.alphazero_lite import benchmark

        with self.assertRaisesRegex(SystemExit, "score fields"):
            benchmark.dynamic_budget_comparison(
                {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "classic_mcts_mode": "dynamic",
                    "search_profile": self.classic_mcts_dynamic_search_profile(),
                    "search_profile_hash": "dynamic-profile-hash",
                    "games": 40,
                    "az_wins": 22,
                    "mcts_wins": 10,
                    "draws": 8,
                    "score": 0.52,
                    "budget_summary": {
                        "source": "classic_mcts_dynamic_runtime",
                        "mean_final_simulations": 128,
                        "mean_root_latency_ms": 6.5,
                    },
                    "dynamic_budget_comparison": {
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "runtime_target_ms": 6.3,
                        "runtime_target_matched": True,
                        "seat_bias_neutralized": True,
                        "dynamic_mean_final_simulations": 128.0,
                        "dynamic_mean_root_latency_ms": 6.5,
                        "fixed_mean_final_simulations": 96.0,
                        "fixed_mean_root_latency_ms": 6.3,
                        "dynamic_score": 0.65,
                        "fixed_score": 0.49,
                    },
                    "classic_mcts_dynamic_budget_config": {
                        "enabled": True,
                        "probe_simulations": 12,
                        "min_simulations": 24,
                        "max_simulations": 96,
                        "entropy_weight": 0.75,
                        "low_margin_threshold": 0.18,
                        "low_margin_weight": 1.25,
                        "variance_weight": 1.1,
                    },
                },
                {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "classic_mcts_mode": "fixed",
                    "search_profile": self.classic_mcts_fixed_search_profile(),
                    "search_profile_hash": "fixed-profile-hash",
                    "games": 40,
                    "az_wins": 20,
                    "mcts_wins": 12,
                    "draws": 8,
                    "score": 0.6,
                    "budget_summary": {
                        "source": "classic_mcts_fixed_runtime",
                        "mean_final_simulations": 96,
                        "mean_root_latency_ms": 6.3,
                    },
                    "classic_mcts_dynamic_budget_config": {
                        "enabled": False,
                        "probe_simulations": 0,
                        "min_simulations": 1200,
                        "max_simulations": 1200,
                        "entropy_weight": 0.0,
                        "low_margin_threshold": 0.0,
                        "low_margin_weight": 0.0,
                        "variance_weight": 0.0,
                    },
                },
            )

    def test_dynamic_budget_comparison_accepts_truthful_producer_rounded_scores(self):
        from ml.alphazero_lite import benchmark

        report = benchmark.dynamic_budget_comparison(
            {
                "comparison_mode": "classic_dynamic_vs_fixed",
                "classic_mcts_mode": "dynamic",
                "search_profile": self.classic_mcts_dynamic_search_profile(),
                "search_profile_hash": "dynamic-profile-hash",
                "games": 30,
                "az_wins": 11,
                "mcts_wins": 17,
                "draws": 2,
                "score": 0.4,
                "budget_summary": {
                    "source": "classic_mcts_dynamic_runtime",
                    "mean_final_simulations": 128,
                    "mean_root_latency_ms": 6.5,
                },
                "dynamic_budget_comparison": {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "runtime_target_ms": 6.3,
                    "runtime_target_matched": True,
                    "seat_bias_neutralized": True,
                    "dynamic_mean_final_simulations": 128.0,
                    "dynamic_mean_root_latency_ms": 6.5,
                    "fixed_mean_final_simulations": 96.0,
                    "fixed_mean_root_latency_ms": 6.3,
                    "dynamic_score": 0.4,
                    "fixed_score": 0.4333,
                },
                "classic_mcts_dynamic_budget_config": {
                    "enabled": True,
                    "probe_simulations": 12,
                    "min_simulations": 24,
                    "max_simulations": 96,
                    "entropy_weight": 0.75,
                    "low_margin_threshold": 0.18,
                    "low_margin_weight": 1.25,
                    "variance_weight": 1.1,
                },
            },
            {
                "comparison_mode": "classic_dynamic_vs_fixed",
                "classic_mcts_mode": "fixed",
                "search_profile": self.classic_mcts_fixed_search_profile(),
                "search_profile_hash": "fixed-profile-hash",
                "games": 30,
                "az_wins": 12,
                "mcts_wins": 16,
                "draws": 2,
                "score": 0.4333,
                "budget_summary": {
                    "source": "classic_mcts_fixed_runtime",
                    "mean_final_simulations": 96,
                    "mean_root_latency_ms": 6.3,
                },
                "classic_mcts_dynamic_budget_config": {
                    "enabled": False,
                    "probe_simulations": 0,
                    "min_simulations": 1200,
                    "max_simulations": 1200,
                    "entropy_weight": 0.0,
                    "low_margin_threshold": 0.0,
                    "low_margin_weight": 0.0,
                    "variance_weight": 0.0,
                },
            },
        )

        self.assertEqual(0.4333, report["fixed_score"])

    def test_dynamic_budget_comparison_rejects_missing_candidate_latency_metric(self):
        from ml.alphazero_lite import benchmark

        with self.assertRaisesRegex(SystemExit, "latency metrics"):
            benchmark.dynamic_budget_comparison(
                {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "classic_mcts_mode": "dynamic",
                    "search_profile": self.classic_mcts_dynamic_search_profile(),
                    "search_profile_hash": "dynamic-profile-hash",
                    "games": 40,
                    "az_wins": 22,
                    "mcts_wins": 10,
                    "draws": 8,
                    "score": 0.65,
                    "budget_summary": {
                        "source": "classic_mcts_dynamic_runtime",
                        "mean_final_simulations": 128,
                    },
                    "dynamic_budget_comparison": {
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "runtime_target_ms": 6.3,
                        "runtime_target_matched": True,
                        "seat_bias_neutralized": True,
                        "dynamic_mean_final_simulations": 128.0,
                        "dynamic_mean_root_latency_ms": None,
                        "fixed_mean_final_simulations": 96.0,
                        "fixed_mean_root_latency_ms": 6.3,
                        "dynamic_score": 0.65,
                        "fixed_score": 0.6,
                    },
                    "classic_mcts_dynamic_budget_config": {
                        "enabled": True,
                        "probe_simulations": 12,
                        "min_simulations": 24,
                        "max_simulations": 96,
                        "entropy_weight": 0.75,
                        "low_margin_threshold": 0.18,
                        "low_margin_weight": 1.25,
                        "variance_weight": 1.1,
                    },
                },
                {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "classic_mcts_mode": "fixed",
                    "search_profile": self.classic_mcts_fixed_search_profile(),
                    "search_profile_hash": "fixed-profile-hash",
                    "games": 40,
                    "az_wins": 20,
                    "mcts_wins": 12,
                    "draws": 8,
                    "score": 0.6,
                    "budget_summary": {
                        "source": "classic_mcts_fixed_runtime",
                        "mean_final_simulations": 96,
                        "mean_root_latency_ms": 6.3,
                    },
                    "classic_mcts_dynamic_budget_config": {
                        "enabled": False,
                        "probe_simulations": 0,
                        "min_simulations": 1200,
                        "max_simulations": 1200,
                        "entropy_weight": 0.0,
                        "low_margin_threshold": 0.0,
                        "low_margin_weight": 0.0,
                        "variance_weight": 0.0,
                    },
                },
            )

    def test_dynamic_budget_comparison_rejects_missing_baseline_latency_metric(self):
        from ml.alphazero_lite import benchmark

        with self.assertRaisesRegex(SystemExit, "latency metrics"):
            benchmark.dynamic_budget_comparison(
                {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "classic_mcts_mode": "dynamic",
                    "search_profile": self.classic_mcts_dynamic_search_profile(),
                    "search_profile_hash": "dynamic-profile-hash",
                    "games": 40,
                    "az_wins": 22,
                    "mcts_wins": 10,
                    "draws": 8,
                    "score": 0.65,
                    "budget_summary": {
                        "source": "classic_mcts_dynamic_runtime",
                        "mean_final_simulations": 128,
                        "mean_root_latency_ms": 6.5,
                    },
                    "dynamic_budget_comparison": {
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "runtime_target_ms": None,
                        "runtime_target_matched": False,
                        "seat_bias_neutralized": True,
                        "dynamic_mean_final_simulations": 128.0,
                        "dynamic_mean_root_latency_ms": 6.5,
                        "fixed_mean_final_simulations": 96.0,
                        "fixed_mean_root_latency_ms": None,
                        "dynamic_score": 0.65,
                        "fixed_score": 0.6,
                    },
                    "classic_mcts_dynamic_budget_config": {
                        "enabled": True,
                        "probe_simulations": 12,
                        "min_simulations": 24,
                        "max_simulations": 96,
                        "entropy_weight": 0.75,
                        "low_margin_threshold": 0.18,
                        "low_margin_weight": 1.25,
                        "variance_weight": 1.1,
                    },
                },
                {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "classic_mcts_mode": "fixed",
                    "search_profile": self.classic_mcts_fixed_search_profile(),
                    "search_profile_hash": "fixed-profile-hash",
                    "games": 40,
                    "az_wins": 20,
                    "mcts_wins": 12,
                    "draws": 8,
                    "score": 0.6,
                    "budget_summary": {
                        "source": "classic_mcts_fixed_runtime",
                        "mean_final_simulations": 96,
                    },
                    "classic_mcts_dynamic_budget_config": {
                        "enabled": False,
                        "probe_simulations": 0,
                        "min_simulations": 1200,
                        "max_simulations": 1200,
                        "entropy_weight": 0.0,
                        "low_margin_threshold": 0.0,
                        "low_margin_weight": 0.0,
                        "variance_weight": 0.0,
                    },
                },
            )

    def test_promotion_report_uses_embedded_fixed_vs_dynamic_classic_mcts_comparison(
        self,
    ):
        from ml.alphazero_lite import benchmark

        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            arena_report = tmp_path / "arena.json"
            mcts_report = tmp_path / "mcts.json"
            baseline_report = tmp_path / "baseline.json"

            arena_report.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "wins": 36,
                        "losses": 12,
                        "draws": 12,
                        "games_played": 60,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            mcts_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "classic_mcts_mode": "dynamic",
                        "search_profile": self.classic_mcts_dynamic_search_profile(),
                        "search_profile_hash": "dynamic-profile-hash",
                        "games": 40,
                        "az_wins": 22,
                        "mcts_wins": 10,
                        "draws": 8,
                        "score": 0.65,
                        "budget_summary": {
                            "source": "classic_mcts_dynamic_runtime",
                            "mean_final_simulations": 128,
                            "mean_root_latency_ms": 6.5,
                        },
                        "dynamic_budget_comparison": {
                            "comparison_mode": "classic_dynamic_vs_fixed",
                            "runtime_target_ms": 6.3,
                            "runtime_target_matched": True,
                            "seat_bias_neutralized": True,
                            "dynamic_mean_final_simulations": 128.0,
                            "dynamic_mean_root_latency_ms": 6.5,
                            "fixed_mean_final_simulations": 96.0,
                            "fixed_mean_root_latency_ms": 6.3,
                            "dynamic_score": 0.65,
                            "fixed_score": 0.6,
                        },
                        "classic_mcts_dynamic_budget_config": {
                            "enabled": True,
                            "probe_simulations": 12,
                            "min_simulations": 24,
                            "max_simulations": 96,
                            "entropy_weight": 0.75,
                            "low_margin_threshold": 0.18,
                            "low_margin_weight": 1.25,
                            "variance_weight": 1.1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            baseline_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "classic_mcts_mode": "fixed",
                        "search_profile": self.classic_mcts_fixed_search_profile(),
                        "search_profile_hash": "fixed-profile-hash",
                        "games": 40,
                        "az_wins": 20,
                        "mcts_wins": 12,
                        "draws": 8,
                        "score": 0.6,
                        "budget_summary": {
                            "source": "classic_mcts_fixed_runtime",
                            "mean_final_simulations": 96,
                            "mean_root_latency_ms": 6.3,
                        },
                        "dynamic_budget_comparison": {
                            "comparison_mode": "classic_dynamic_vs_fixed",
                            "runtime_target_ms": 6.3,
                            "runtime_target_matched": True,
                            "seat_bias_neutralized": True,
                            "dynamic_mean_final_simulations": 128.0,
                            "dynamic_mean_root_latency_ms": 6.5,
                            "fixed_mean_final_simulations": 96.0,
                            "fixed_mean_root_latency_ms": 6.3,
                            "dynamic_score": 0.65,
                            "fixed_score": 0.6,
                        },
                        "classic_mcts_dynamic_budget_config": {
                            "enabled": False,
                            "probe_simulations": 0,
                            "min_simulations": 96,
                            "max_simulations": 96,
                            "entropy_weight": 0.8,
                            "low_margin_threshold": 0.2,
                            "low_margin_weight": 1.5,
                            "variance_weight": 1.5,
                        },
                    }
                ),
                encoding="utf-8",
            )

            args = benchmark.parse_args(
                [
                    "--mode",
                    "promotion",
                    "--out",
                    str(tmp_path / "out.json"),
                    "--arena-report",
                    str(arena_report),
                    "--mcts-report",
                    str(mcts_report),
                    "--current-baseline-mcts-report",
                    str(baseline_report),
                ]
            )
            report = benchmark.build_report(args)

        self.assertEqual(0.65, report["dynamic_budget_comparison"]["dynamic_score"])
        self.assertEqual(0.6, report["dynamic_budget_comparison"]["fixed_score"])
        self.assertEqual(
            "classic_mcts_dynamic_runtime",
            report["dynamic_budget_metric_source"]["candidate"],
        )
        self.assertEqual(
            "classic_mcts_fixed_runtime",
            report["dynamic_budget_metric_source"]["baseline"],
        )

    def test_promotion_report_accepts_real_fixed_baseline_without_comparison_payload(
        self,
    ):
        from ml.alphazero_lite import benchmark

        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            arena_report = tmp_path / "arena.json"
            mcts_report = tmp_path / "mcts.json"
            baseline_report = tmp_path / "baseline.json"

            arena_report.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "wins": 36,
                        "losses": 12,
                        "draws": 12,
                        "games_played": 60,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            mcts_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "classic_mcts_mode": "dynamic",
                        "search_profile": self.classic_mcts_dynamic_search_profile(),
                        "search_profile_hash": "dynamic-profile-hash",
                        "games": 40,
                        "az_wins": 22,
                        "mcts_wins": 10,
                        "draws": 8,
                        "score": 0.65,
                        "budget_summary": {
                            "source": "classic_mcts_dynamic_runtime",
                            "mean_final_simulations": 128,
                            "mean_root_latency_ms": 6.5,
                        },
                        "dynamic_budget_comparison": {
                            "comparison_mode": "classic_dynamic_vs_fixed",
                            "runtime_target_ms": 6.3,
                            "runtime_target_matched": True,
                            "seat_bias_neutralized": True,
                            "dynamic_mean_final_simulations": 128.0,
                            "dynamic_mean_root_latency_ms": 6.5,
                            "fixed_mean_final_simulations": 96.0,
                            "fixed_mean_root_latency_ms": 6.3,
                            "dynamic_score": 0.65,
                            "fixed_score": 0.6,
                        },
                        "classic_mcts_dynamic_budget_config": {
                            "enabled": True,
                            "probe_simulations": 12,
                            "min_simulations": 24,
                            "max_simulations": 96,
                            "entropy_weight": 0.75,
                            "low_margin_threshold": 0.18,
                            "low_margin_weight": 1.25,
                            "variance_weight": 1.1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            baseline_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "classic_mcts_mode": "fixed",
                        "search_profile": self.classic_mcts_fixed_search_profile(),
                        "search_profile_hash": "fixed-profile-hash",
                        "games": 40,
                        "az_wins": 20,
                        "mcts_wins": 12,
                        "draws": 8,
                        "score": 0.6,
                        "budget_summary": {
                            "source": "classic_mcts_fixed_runtime",
                            "mean_final_simulations": 96,
                            "mean_root_latency_ms": 6.3,
                        },
                        "classic_mcts_dynamic_budget_config": {
                            "enabled": False,
                            "probe_simulations": 0,
                            "min_simulations": 96,
                            "max_simulations": 96,
                            "entropy_weight": 0.8,
                            "low_margin_threshold": 0.2,
                            "low_margin_weight": 1.5,
                            "variance_weight": 1.5,
                        },
                    }
                ),
                encoding="utf-8",
            )

            args = benchmark.parse_args(
                [
                    "--mode",
                    "promotion",
                    "--out",
                    str(tmp_path / "out.json"),
                    "--arena-report",
                    str(arena_report),
                    "--mcts-report",
                    str(mcts_report),
                    "--current-baseline-mcts-report",
                    str(baseline_report),
                ]
            )
            report = benchmark.build_report(args)

        self.assertEqual(0.65, report["dynamic_budget_comparison"]["dynamic_score"])
        self.assertEqual(0.6, report["dynamic_budget_comparison"]["fixed_score"])

    def test_promotion_dry_run_allows_baseline_flag_without_mcts_report(self):
        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/benchmark.py",
                    "--mode",
                    "promotion",
                    "--games",
                    "12",
                    "--seed",
                    "7",
                    "--out",
                    str(out_path),
                    "--dry-run",
                    "--current-baseline-mcts-report",
                    str(tmp_path / "baseline.json"),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIsNone(report["dynamic_budget_comparison"])

    def test_promotion_report_preserves_candidate_dynamic_budget_metadata_without_baseline(
        self,
    ):
        from ml.alphazero_lite import benchmark

        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            arena_report = tmp_path / "arena.json"
            mcts_report = tmp_path / "mcts.json"

            arena_report.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "wins": 36,
                        "losses": 12,
                        "draws": 12,
                        "games_played": 60,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            mcts_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 40,
                        "az_wins": 22,
                        "mcts_wins": 10,
                        "draws": 8,
                        "budget_summary": {
                            "source": "challenger_puct_runtime",
                            "mean_final_simulations": 128,
                            "mean_root_latency_ms": 6.5,
                        },
                        "classic_mcts_dynamic_budget_config": {
                            "enabled": True,
                            "probe_simulations": 12,
                            "min_simulations": 24,
                            "max_simulations": 96,
                            "entropy_weight": 0.75,
                            "low_margin_threshold": 0.18,
                            "low_margin_weight": 1.25,
                            "variance_weight": 1.1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            args = benchmark.parse_args(
                [
                    "--mode",
                    "promotion",
                    "--out",
                    str(tmp_path / "out.json"),
                    "--arena-report",
                    str(arena_report),
                    "--mcts-report",
                    str(mcts_report),
                ]
            )
            report = benchmark.build_report(args)

        self.assertEqual(
            "challenger_puct_runtime",
            report["dynamic_budget_metric_source"]["candidate"],
        )
        self.assertIsNone(report["dynamic_budget_metric_source"].get("baseline"))
        self.assertEqual(
            0.75,
            report["classic_mcts_dynamic_budget_config"]["candidate"]["entropy_weight"],
        )
        self.assertIsNone(report["classic_mcts_dynamic_budget_config"].get("baseline"))
        self.assertNotIn("dynamic_budget_config", report)
        self.assertIsNone(report["dynamic_budget_comparison"])

    def test_build_report_preserves_opening_cache_summary_fields_from_arena_report(
        self,
    ):
        from ml.alphazero_lite import benchmark

        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            tmp_path = Path(tmp)
            arena_report = tmp_path / "arena.json"
            mcts_report = tmp_path / "mcts.json"

            arena_report.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 60,
                        "wins": 36,
                        "losses": 24,
                        "draws": 0,
                        "promotion_decision": {"passed": True},
                        "opening_cache_summary": {
                            "runtime_hit_rate": 0.25,
                            "training_hit_rate": 0.4,
                            "opening_bucket_quality_delta": 0.03,
                            "latency_delta_ms": -5.2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            mcts_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 40,
                        "az_wins": 22,
                        "mcts_wins": 10,
                        "draws": 8,
                        "score": 0.65,
                    }
                ),
                encoding="utf-8",
            )

            args = benchmark.parse_args(
                [
                    "--mode",
                    "promotion",
                    "--out",
                    str(tmp_path / "out.json"),
                    "--arena-report",
                    str(arena_report),
                    "--mcts-report",
                    str(mcts_report),
                ]
            )
            report = benchmark.build_report(args)

        self.assertEqual(0.25, report["opening_cache_summary"]["runtime_hit_rate"])
        self.assertEqual(0.4, report["opening_cache_summary"]["training_hit_rate"])
        self.assertEqual(
            0.03, report["opening_cache_summary"]["opening_bucket_quality_delta"]
        )
        self.assertEqual(-5.2, report["opening_cache_summary"]["latency_delta_ms"])

    def test_promotion_report_preserves_opening_cache_summary_fields(self):
        from ml.alphazero_lite import benchmark

        report = benchmark.build_report_from_inputs(
            arena_report={
                "schema": "arena_v1",
                "games_played": 60,
                "wins": 36,
                "losses": 24,
                "draws": 0,
                "promotion_decision": {"passed": True},
            },
            mcts_report={
                "schema": "azlite_vs_mcts_v1",
                "games": 40,
                "az_wins": 22,
                "mcts_wins": 10,
                "draws": 8,
                "score": 0.65,
            },
            opening_cache_summary={
                "runtime_hit_rate": 0.25,
                "training_hit_rate": 0.4,
                "opening_bucket_quality_delta": 0.03,
                "latency_delta_ms": -5.2,
            },
        )

        self.assertEqual(0.25, report["opening_cache_summary"]["runtime_hit_rate"])
        self.assertEqual(0.4, report["opening_cache_summary"]["training_hit_rate"])
        self.assertEqual(
            0.03, report["opening_cache_summary"]["opening_bucket_quality_delta"]
        )
        self.assertEqual(-5.2, report["opening_cache_summary"]["latency_delta_ms"])

    def test_promotion_report_uses_arena_embedded_opening_cache_summary_when_input_not_overridden(
        self,
    ):
        from ml.alphazero_lite import benchmark

        report = benchmark.build_report_from_inputs(
            arena_report={
                "schema": "arena_v1",
                "games_played": 60,
                "wins": 36,
                "losses": 24,
                "draws": 0,
                "promotion_decision": {"passed": True},
                "opening_cache_summary": {
                    "runtime_hit_rate": 0.25,
                    "training_hit_rate": 0.4,
                    "opening_bucket_quality_delta": 0.03,
                    "latency_delta_ms": -5.2,
                },
            },
            mcts_report={
                "schema": "azlite_vs_mcts_v1",
                "games": 40,
                "az_wins": 22,
                "mcts_wins": 10,
                "draws": 8,
                "score": 0.65,
            },
        )

        self.assertEqual(0.25, report["opening_cache_summary"]["runtime_hit_rate"])
        self.assertEqual(0.4, report["opening_cache_summary"]["training_hit_rate"])
        self.assertEqual(
            0.03, report["opening_cache_summary"]["opening_bucket_quality_delta"]
        )
        self.assertEqual(-5.2, report["opening_cache_summary"]["latency_delta_ms"])

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

    def test_dynamic_budget_comparison_rejects_missing_candidate_dynamic_budget_config(
        self,
    ):
        from ml.alphazero_lite import benchmark

        with self.assertRaisesRegex(
            SystemExit, "candidate dynamic budget config metadata"
        ):
            benchmark.dynamic_budget_comparison(
                {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "classic_mcts_mode": "dynamic",
                    "search_profile": self.classic_mcts_dynamic_search_profile(),
                    "games": 30,
                    "az_wins": 15,
                    "mcts_wins": 15,
                    "draws": 0,
                    "score": 0.5,
                    "budget_summary": {
                        "source": "classic_mcts_dynamic_runtime",
                        "mean_final_simulations": 128,
                        "mean_root_latency_ms": 6.5,
                    },
                    "dynamic_budget_comparison": {
                        "comparison_mode": "classic_dynamic_vs_fixed",
                        "runtime_target_ms": 6.3,
                        "runtime_target_matched": True,
                        "seat_bias_neutralized": True,
                        "dynamic_mean_final_simulations": 128.0,
                        "dynamic_mean_root_latency_ms": 6.5,
                        "fixed_mean_final_simulations": 96.0,
                        "fixed_mean_root_latency_ms": 6.3,
                        "dynamic_score": 0.5,
                        "fixed_score": 0.5,
                    },
                },
                {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "classic_mcts_mode": "fixed",
                    "search_profile": self.classic_mcts_fixed_search_profile(),
                    "games": 30,
                    "az_wins": 15,
                    "mcts_wins": 15,
                    "draws": 0,
                    "score": 0.5,
                    "budget_summary": {
                        "source": "classic_mcts_fixed_runtime",
                        "mean_final_simulations": 96,
                        "mean_root_latency_ms": 6.3,
                    },
                    "classic_mcts_dynamic_budget_config": {"enabled": False},
                },
            )

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
