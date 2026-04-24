import json
import importlib.machinery
import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class LocalPromotionGateTest(unittest.TestCase):
    def load_gate_module(self):
        script_path = Path(__file__).resolve().parents[2] / "script/ai/local_promotion_gate"
        spec = importlib.util.spec_from_file_location(
            "local_promotion_gate",
            script_path,
            loader=importlib.machinery.SourceFileLoader("local_promotion_gate", str(script_path)),
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def run_gate(self, *args: str) -> subprocess.CompletedProcess[str]:
        repo_root = Path(__file__).resolve().parents[2]
        return subprocess.run(
            ["script/ai/local_promotion_gate", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

    def write_report(self, path: Path, *, games_played: int | None = None, games: int | None = None, wins: int, losses: int, draws: int, move_time_mean_ms=None, move_time_p95_ms=None, az_wins=None) -> None:
        report = {
            "schema": "arena_v1",
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "promotion_decision": {"passed": True},
        }
        if games_played is not None:
            report["games_played"] = games_played
        if games is not None:
            report["games"] = games
        if move_time_mean_ms is not None:
            report["notes"] = report.get("notes", {})
            report["notes"]["move_time_mean_ms"] = move_time_mean_ms
        if move_time_p95_ms is not None:
            report["notes"] = report.get("notes", {})
            report["notes"]["move_time_p95_ms"] = move_time_p95_ms
        if az_wins is not None:
            report["az_wins"] = az_wins
        path.write_text(json.dumps(report), encoding="utf-8")

    def write_regression_report(self, path: Path, *, passed: bool) -> None:
        path.write_text(
            json.dumps(
                {
                    "passed": passed,
                    "artifact_path": "candidate",
                    "positions_path": "test/fixtures/ai/superhuman_regression_positions.json",
                    "results": [],
                }
            ),
            encoding="utf-8",
        )

    def test_rejects_candidate_when_regression_report_is_missing_passed_field(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)
            (tmp / "regression.json").write_text(json.dumps({"results": []}), encoding="utf-8")

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-regression-report",
                str(tmp / "regression.json"),
                "--out",
                str(out),
            )

            self.assertNotEqual(0, result.returncode)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertTrue(any(reason["code"] == "regression_check_failed" for reason in report["failure_reasons"]))

    def test_rejects_invalid_search_config_flag_shape(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            config_path = tmp / "search_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "steps": [
                            {
                                "name": "arena_confirm_report",
                                "command": ["python", "ml/alphazero_lite/arena.py", "--fpu-mode"],
                            },
                            {
                                "name": "mcts1200_baseline_report",
                                "command": ["python", "ml/alphazero_lite/mcts1200_baseline.py", "--fpu-mode", "parent_q"],
                            },
                            {
                                "name": "current_mcts1200_baseline_report",
                                "command": ["python", "ml/alphazero_lite/mcts1200_baseline.py", "--fpu-mode", "parent_q"],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--config-path",
                str(config_path),
                "--dry-run",
                "--out",
                str(out),
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("missing value for --fpu-mode", result.stderr)

    def test_rejects_candidate_that_does_not_beat_current_superhuman(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=20, losses=0, draws=40, move_time_mean_ms=120, move_time_p95_ms=180)
            self.write_report(tmp / "hard.json", games_played=120, wins=72, losses=0, draws=48, move_time_mean_ms=110, move_time_p95_ms=170)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=25, losses=5, draws=10, az_wins=25)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=6, draws=10, az_wins=24)
            self.write_regression_report(tmp / "regression.json", passed=True)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--hard-path",
                str(tmp / "hard_model"),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-hard-report",
                str(tmp / "hard.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-regression-report",
                str(tmp / "regression.json"),
                "--out",
                str(out),
                "--max-arena-move-time-mean-ms",
                "200",
                "--max-arena-move-time-p95-ms",
                "250",
            )

            self.assertNotEqual(0, result.returncode)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertTrue(any(reason["code"] == "arena_score_below_threshold" for reason in report["failure_reasons"]))

    def test_rejects_latency_regression(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=80, losses=0, draws=40, move_time_mean_ms=260, move_time_p95_ms=310)
            self.write_report(tmp / "hard.json", games_played=120, wins=70, losses=10, draws=40, move_time_mean_ms=100, move_time_p95_ms=150)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=5, draws=5, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=10, draws=6, az_wins=24)
            self.write_regression_report(tmp / "regression.json", passed=True)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--hard-path",
                str(tmp / "hard_model"),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-hard-report",
                str(tmp / "hard.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-regression-report",
                str(tmp / "regression.json"),
                "--out",
                str(out),
                "--max-arena-move-time-mean-ms",
                "200",
                "--max-arena-move-time-p95-ms",
                "250",
            )

            self.assertNotEqual(0, result.returncode)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertTrue(any(reason["code"] == "arena_move_time_mean_above_threshold" for reason in report["failure_reasons"]))
            self.assertTrue(any(reason["code"] == "arena_move_time_p95_above_threshold" for reason in report["failure_reasons"]))

    def test_accepts_candidate_that_beats_current_superhuman_and_hard(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28, move_time_mean_ms=120, move_time_p95_ms=160)
            self.write_report(tmp / "hard.json", games_played=120, wins=72, losses=0, draws=48, move_time_mean_ms=110, move_time_p95_ms=150)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)
            self.write_regression_report(tmp / "regression.json", passed=True)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--hard-path",
                str(tmp / "hard_model"),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-hard-report",
                str(tmp / "hard.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-regression-report",
                str(tmp / "regression.json"),
                "--out",
                str(out),
                "--max-arena-move-time-mean-ms",
                "200",
                "--max-arena-move-time-p95-ms",
                "250",
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])

    def test_reports_endgame_threshold_sweep_results(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28, move_time_mean_ms=120, move_time_p95_ms=160)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)
            self.write_regression_report(tmp / "regression.json", passed=True)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-regression-report",
                str(tmp / "regression.json"),
                "--out",
                str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual([8, 10, 12], report["endgame_exact_solve"]["thresholds"])
            self.assertTrue(report["endgame_exact_solve"]["scaffold_only"])
            self.assertEqual("planned_evaluation_metadata_only", report["endgame_exact_solve"]["results_source"])
            self.assertEqual([], report["endgame_exact_solve"]["results"])
            self.assertEqual(
                [8, 10, 12],
                [evaluation["threshold"] for evaluation in report["endgame_exact_solve"]["planned_evaluations"]],
            )
            self.assertIn("recommendation", report["endgame_exact_solve"])

    def test_dry_run_includes_endgame_exact_solve_scaffold_metadata(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--dry-run",
                "--out",
                str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual([8, 10, 12], report["endgame_exact_solve"]["thresholds"])
            self.assertTrue(report["endgame_exact_solve"]["scaffold_only"])
            self.assertEqual("planned_only", report["endgame_exact_solve"]["status"])
            self.assertEqual([], report["endgame_exact_solve"]["results"])
            self.assertEqual(
                [8, 10, 12],
                [evaluation["threshold"] for evaluation in report["endgame_exact_solve"]["planned_evaluations"]],
            )
            self.assertTrue(
                all(evaluation["mode"] == "planned" for evaluation in report["endgame_exact_solve"]["planned_evaluations"])
            )

    def test_endgame_threshold_results_are_marked_as_placeholder_scaffold(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28, move_time_mean_ms=120, move_time_p95_ms=160)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)
            self.write_regression_report(tmp / "regression.json", passed=True)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-regression-report",
                str(tmp / "regression.json"),
                "--out",
                str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(report["endgame_exact_solve"]["scaffold_only"])
            self.assertEqual("planned_evaluation_metadata_only", report["endgame_exact_solve"]["results_source"])
            self.assertEqual([], report["endgame_exact_solve"]["results"])
            self.assertEqual(3, len(report["endgame_exact_solve"]["planned_evaluations"]))
            for planned_evaluation in report["endgame_exact_solve"]["planned_evaluations"]:
                self.assertTrue(planned_evaluation["scaffold_only"])
                self.assertIn("report_path", planned_evaluation)

    def test_stub_decision_mode_anchors_endgame_planned_report_paths_to_output_directory(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            stub_dir = tmp / "stub-reports"
            stub_dir.mkdir()
            out_dir = tmp / "gate-output"
            out_dir.mkdir()
            out = out_dir / "report.json"
            self.write_report(stub_dir / "arena.json", games_played=120, wins=92, losses=0, draws=28, move_time_mean_ms=120, move_time_p95_ms=160)
            self.write_report(stub_dir / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(stub_dir / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)
            self.write_regression_report(stub_dir / "regression.json", passed=True)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--stub-arena-report",
                str(stub_dir / "arena.json"),
                "--stub-candidate-mcts-report",
                str(stub_dir / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(stub_dir / "cur_mcts.json"),
                "--stub-regression-report",
                str(stub_dir / "regression.json"),
                "--out",
                str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(
                [
                    str(out_dir / "endgame_exact_solve_threshold_8.json"),
                    str(out_dir / "endgame_exact_solve_threshold_10.json"),
                    str(out_dir / "endgame_exact_solve_threshold_12.json"),
                ],
                [evaluation["report_path"] for evaluation in report["endgame_exact_solve"]["planned_evaluations"]],
            )

    def test_missing_mcts_config_steps_fall_back_to_defaults(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            config_path = tmp / "search_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "steps": [
                            {
                                "name": "arena_confirm_report",
                                "command": ["python", "ml/alphazero_lite/arena.py", "--fpu-mode", "parent_q"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--config-path",
                str(config_path),
                "--dry-run",
                "--out",
                str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual([8, 10, 12], [row["threshold"] for row in report["endgame_exact_solve"]["planned_evaluations"]])

    def test_non_dry_run_requires_configured_search_steps(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            config_path = tmp / "search_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "steps": [
                            {
                                "name": "arena_confirm_report",
                                "command": ["python", "ml/alphazero_lite/arena.py", "--fpu-mode", "parent_q"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)
            self.write_regression_report(tmp / "regression.json", passed=True)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--config-path",
                str(config_path),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-regression-report",
                str(tmp / "regression.json"),
                "--out",
                str(out),
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("config missing required step", result.stderr)

    def test_dry_run_plans_threshold_specific_mcts_commands(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--dry-run",
                "--out",
                str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            planned = report["endgame_exact_solve"]["planned_evaluations"]
            self.assertEqual([8, 10, 12], [row["threshold"] for row in planned])
            for row, threshold in zip(planned, [8, 10, 12], strict=True):
                self.assertIn("candidate_mcts_command", row)
                self.assertIn("current_mcts_command", row)
                self.assertIn("--exact-solve-enabled", row["candidate_mcts_command"])
                self.assertIn("--exact-solve-enabled", row["current_mcts_command"])
                self.assertEqual(
                    str(threshold),
                    row["candidate_mcts_command"][row["candidate_mcts_command"].index("--exact-solve-stone-threshold") + 1],
                )
                self.assertEqual(
                    str(threshold),
                    row["current_mcts_command"][row["current_mcts_command"].index("--exact-solve-stone-threshold") + 1],
                )

    def test_executed_threshold_runs_record_threshold_specific_results(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            current = tmp / "current"
            candidate.mkdir()
            current.mkdir()
            out = tmp / "report.json"

            env = {
                **dict(os.environ),
                "AZLITE_ARENA_STUB": "1",
                "AZLITE_MCTS1200_BASELINE_STUB": "1",
                "AZLITE_CHECK_SUPERHUMAN_REGRESSIONS_STUB": "1",
            }
            repo_root = Path(__file__).resolve().parents[2]
            result = subprocess.run(
                [
                    "script/ai/local_promotion_gate",
                    "--candidate-path",
                    str(candidate),
                    "--current-path",
                    str(current),
                    "--out",
                    str(out),
                ],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual("executed", report["endgame_exact_solve"]["status"])
            self.assertFalse(report["endgame_exact_solve"]["scaffold_only"])
            self.assertEqual("threshold_specific_reports", report["endgame_exact_solve"]["results_source"])
            self.assertEqual([8, 10, 12], [row["threshold"] for row in report["endgame_exact_solve"]["results"]])
            self.assertFalse(report["endgame_exact_solve"]["recommendation"]["enabled"])
            self.assertIsNone(report["endgame_exact_solve"]["recommendation"]["threshold"])
            self.assertTrue(
                all(evaluation["mode"] == "executed" for evaluation in report["endgame_exact_solve"]["planned_evaluations"])
            )
            self.assertTrue(
                all(not evaluation["scaffold_only"] for evaluation in report["endgame_exact_solve"]["planned_evaluations"])
            )
            for row in report["endgame_exact_solve"]["results"]:
                self.assertIn("candidate_mcts_score", row)
                self.assertIn("current_mcts_score", row)
                self.assertTrue(Path(row["candidate_mcts_report_path"]).exists())
                self.assertTrue(Path(row["current_mcts_report_path"]).exists())

    def test_endgame_exact_solve_recommendation_prefers_best_threshold_gain(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28, move_time_mean_ms=120, move_time_p95_ms=160)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)
            self.write_regression_report(tmp / "regression.json", passed=True)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-regression-report",
                str(tmp / "regression.json"),
                "--out",
                str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            report["endgame_exact_solve"]["results"] = [
                {"threshold": 8, "candidate_mcts_score": 0.55, "current_mcts_score": 0.50},
                {"threshold": 10, "candidate_mcts_score": 0.65, "current_mcts_score": 0.50},
                {"threshold": 12, "candidate_mcts_score": 0.60, "current_mcts_score": 0.50},
            ]

            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "local_promotion_gate",
                Path(__file__).resolve().parents[2] / "script/ai/local_promotion_gate",
                loader=importlib.machinery.SourceFileLoader(
                    "local_promotion_gate",
                    str(Path(__file__).resolve().parents[2] / "script/ai/local_promotion_gate"),
                ),
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            recommendation = module.build_endgame_exact_solve_recommendation(report["endgame_exact_solve"]["results"])
            self.assertTrue(recommendation["enabled"])
            self.assertEqual(10, recommendation["threshold"])

    def test_omitting_hard_path_uses_default_current_path_and_records_regression_report(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)
            self.write_regression_report(tmp / "regression.json", passed=True)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-regression-report",
                str(tmp / "regression.json"),
                "--out",
                str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual("model-artifact/current", report["hard_path"])
            self.assertIsNone(report["hard_score"])
            self.assertIsNone(report["hard_report_path"])
            self.assertEqual(str(tmp / "regression.json"), report["regression_report_path"])
            self.assertTrue(any(e["id"] == "candidate_vs_hard_arena" for e in report["evaluations"]))

    def test_rejects_candidate_without_hard_path_when_arena_score_low(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=20, losses=0, draws=40)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=25, losses=5, draws=10, az_wins=25)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=6, draws=10, az_wins=24)
            self.write_regression_report(tmp / "regression.json", passed=True)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-regression-report",
                str(tmp / "regression.json"),
                "--out",
                str(out),
            )

            self.assertNotEqual(0, result.returncode)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertTrue(any(reason["code"] == "arena_score_below_threshold" for reason in report["failure_reasons"]))
            self.assertFalse(any(reason["code"] == "candidate_not_stronger_than_hard" for reason in report["failure_reasons"]))

    def test_rejects_candidate_when_regression_check_fails(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)
            self.write_regression_report(tmp / "regression.json", passed=False)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-regression-report",
                str(tmp / "regression.json"),
                "--out",
                str(out),
            )

            self.assertNotEqual(0, result.returncode)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertEqual("model-artifact/current", report["hard_path"])
            self.assertEqual(str(tmp / "regression.json"), report["regression_report_path"])
            self.assertTrue(any(reason["code"] == "regression_check_failed" for reason in report["failure_reasons"]))

    def test_stub_decision_report_carries_dynamic_budget_summary(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"

            (tmp / "arena.json").write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "wins": 92,
                        "losses": 0,
                        "draws": 28,
                        "games_played": 120,
                        "promotion_decision": {"passed": True},
                        "budget_summary": {"mean_final_simulations": 128, "trigger_counts": {"late_high_entropy": 6}},
                    }
                ),
                encoding="utf-8",
            )
            (tmp / "cand_mcts.json").write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 40,
                        "az_wins": 30,
                        "mcts_wins": 2,
                        "draws": 8,
                        "budget_summary": {"mean_final_simulations": 128, "mean_root_latency_ms": 6.5},
                    }
                ),
                encoding="utf-8",
            )
            (tmp / "cur_mcts.json").write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 40,
                        "az_wins": 24,
                        "mcts_wins": 8,
                        "draws": 8,
                        "budget_summary": {"mean_final_simulations": 96, "mean_root_latency_ms": 6.2},
                    }
                ),
                encoding="utf-8",
            )
            self.write_regression_report(tmp / "regression.json", passed=True)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-regression-report",
                str(tmp / "regression.json"),
                "--out",
                str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("dynamic_budget", report)
            self.assertEqual(128, report["dynamic_budget"]["candidate_mean_final_simulations"])
            self.assertEqual(96, report["dynamic_budget"]["current_mean_final_simulations"])

    def test_stub_decision_report_includes_hard_suite_bucket_breakdown(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"

            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28)
            (tmp / "hard.json").write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "wins": 72,
                        "losses": 0,
                        "draws": 48,
                        "games_played": 120,
                        "promotion_decision": {"passed": True},
                        "hard_suite_buckets": {
                            "opening": {"games": 20, "score": 0.58},
                            "midgame": {"games": 20, "score": 0.61},
                            "late": {"games": 20, "score": 0.55},
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)
            self.write_regression_report(tmp / "regression.json", passed=True)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--hard-path",
                str(tmp / "hard_model"),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-hard-report",
                str(tmp / "hard.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-regression-report",
                str(tmp / "regression.json"),
                "--out",
                str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("hard_suite_buckets", report)
            self.assertEqual({"games": 20, "score": 0.58}, report["hard_suite_buckets"]["opening"])

    def test_stub_decision_report_preserves_real_producer_backed_zero_bucket_keys(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"

            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28)
            (tmp / "hard.json").write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "wins": 72,
                        "losses": 0,
                        "draws": 48,
                        "games_played": 120,
                        "promotion_decision": {"passed": True},
                        "hard_suite_buckets": {
                            "opening": {"games": 0, "score": None},
                            "midgame": {"games": 3, "score": None},
                            "late": {"games": 0, "score": None},
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)
            self.write_regression_report(tmp / "regression.json", passed=True)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--hard-path",
                str(tmp / "hard_model"),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-hard-report",
                str(tmp / "hard.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-regression-report",
                str(tmp / "regression.json"),
                "--out",
                str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual({"games": 0, "score": None}, report["hard_suite_buckets"]["opening"])
            self.assertEqual({"games": 3, "score": None}, report["hard_suite_buckets"]["midgame"])
            self.assertEqual({"games": 0, "score": None}, report["hard_suite_buckets"]["late"])

    def test_stub_decision_report_ignores_generic_buckets_without_explicit_hard_suite_schema(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"

            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28)
            (tmp / "hard.json").write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "wins": 72,
                        "losses": 0,
                        "draws": 48,
                        "games_played": 120,
                        "promotion_decision": {"passed": True},
                        "buckets": {
                            "opening": {"games": 20, "score": 0.58},
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)
            self.write_regression_report(tmp / "regression.json", passed=True)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--hard-path",
                str(tmp / "hard_model"),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-hard-report",
                str(tmp / "hard.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-regression-report",
                str(tmp / "regression.json"),
                "--out",
                str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual({}, report["hard_suite_buckets"])

    def test_stub_decision_report_marks_dynamic_budget_unavailable_without_budget_summaries(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"

            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)
            self.write_regression_report(tmp / "regression.json", passed=True)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-regression-report",
                str(tmp / "regression.json"),
                "--out",
                str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertIsNone(report["dynamic_budget"])

    def test_stub_decision_report_emits_dynamic_budget_recommendation(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"

            (tmp / "arena.json").write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "wins": 92,
                        "losses": 0,
                        "draws": 28,
                        "games_played": 120,
                        "promotion_decision": {"passed": True},
                        "budget_summary": {"mean_final_simulations": 128, "trigger_counts": {"late_high_entropy": 6}},
                    }
                ),
                encoding="utf-8",
            )
            (tmp / "hard.json").write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "wins": 72,
                        "losses": 0,
                        "draws": 48,
                        "games_played": 120,
                        "promotion_decision": {"passed": True},
                        "hard_suite_buckets": {
                            "opening": {"games": 20, "score": None},
                            "midgame": {"games": 20, "score": None},
                            "late": {"games": 20, "score": None},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (tmp / "cand_mcts.json").write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 40,
                        "az_wins": 30,
                        "mcts_wins": 2,
                        "draws": 8,
                        "budget_summary": {
                            "mean_final_simulations": 128,
                            "mean_root_latency_ms": 6.5,
                            "dynamic_budget_comparison": {
                                "comparison_mode": "classic_dynamic_vs_fixed",
                                "runtime_target_matched": True,
                                "dynamic_score": 0.52,
                                "fixed_score": 0.49,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (tmp / "cur_mcts.json").write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 40,
                        "az_wins": 24,
                        "mcts_wins": 8,
                        "draws": 8,
                        "budget_summary": {"mean_final_simulations": 96, "mean_root_latency_ms": 6.2},
                    }
                ),
                encoding="utf-8",
            )
            self.write_regression_report(tmp / "regression.json", passed=True)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--hard-path",
                str(tmp / "hard_model"),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-hard-report",
                str(tmp / "hard.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-regression-report",
                str(tmp / "regression.json"),
                "--out",
                str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual("keep", report["dynamic_budget_recommendation"]["decision"])
            self.assertIn("promotion safety", report["dynamic_budget_recommendation"]["reason"])
            self.assertEqual(0.8833, report["dynamic_budget_recommendation"]["evidence"]["arena_score"])

    def test_stub_decision_report_emits_drop_recommendation_when_promotion_safety_fails(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"

            (tmp / "arena.json").write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "wins": 20,
                        "losses": 0,
                        "draws": 40,
                        "games_played": 120,
                        "promotion_decision": {"passed": False},
                        "budget_summary": {"mean_final_simulations": 128, "trigger_counts": {"late_high_entropy": 6}},
                    }
                ),
                encoding="utf-8",
            )
            (tmp / "cand_mcts.json").write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 40,
                        "az_wins": 30,
                        "mcts_wins": 2,
                        "draws": 8,
                        "budget_summary": {
                            "mean_final_simulations": 128,
                            "mean_root_latency_ms": 6.5,
                            "dynamic_budget_comparison": {
                                "comparison_mode": "classic_dynamic_vs_fixed",
                                "runtime_target_matched": True,
                                "dynamic_score": 0.52,
                                "fixed_score": 0.49,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (tmp / "cur_mcts.json").write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 40,
                        "az_wins": 24,
                        "mcts_wins": 8,
                        "draws": 8,
                        "budget_summary": {"mean_final_simulations": 96, "mean_root_latency_ms": 6.2},
                    }
                ),
                encoding="utf-8",
            )
            self.write_regression_report(tmp / "regression.json", passed=True)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-regression-report",
                str(tmp / "regression.json"),
                "--out",
                str(out),
            )

            self.assertNotEqual(0, result.returncode)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual("drop", report["dynamic_budget_recommendation"]["decision"])
            self.assertIn("promotion safety", report["dynamic_budget_recommendation"]["reason"])

    def test_stub_decision_report_emits_tune_recommendation_when_runtime_matched_evidence_is_missing(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"

            (tmp / "arena.json").write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "wins": 92,
                        "losses": 0,
                        "draws": 28,
                        "games_played": 120,
                        "promotion_decision": {"passed": True},
                        "budget_summary": {"mean_final_simulations": 128, "trigger_counts": {"late_high_entropy": 6}},
                    }
                ),
                encoding="utf-8",
            )
            (tmp / "cand_mcts.json").write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 40,
                        "az_wins": 30,
                        "mcts_wins": 2,
                        "draws": 8,
                        "budget_summary": {
                            "mean_final_simulations": 128,
                            "mean_root_latency_ms": 6.5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (tmp / "cur_mcts.json").write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 40,
                        "az_wins": 24,
                        "mcts_wins": 8,
                        "draws": 8,
                        "budget_summary": {"mean_final_simulations": 96, "mean_root_latency_ms": 6.2},
                    }
                ),
                encoding="utf-8",
            )
            self.write_regression_report(tmp / "regression.json", passed=True)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-regression-report",
                str(tmp / "regression.json"),
                "--out",
                str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual("tune", report["dynamic_budget_recommendation"]["decision"])
            self.assertIn("mixed", report["dynamic_budget_recommendation"]["reason"])

    def test_dynamic_budget_recommendation_keeps_when_promotion_safety_passes_and_runtime_match_is_favorable(self):
        module = self.load_gate_module()

        recommendation = module.build_dynamic_budget_recommendation(
            passed=True,
            arena_score_value=0.88,
            candidate_mcts_score_value=0.48,
            current_mcts_score_value=0.55,
            dynamic_budget={
                "dynamic_budget_comparison": {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "runtime_target_matched": True,
                    "dynamic_score": 0.52,
                    "fixed_score": 0.49,
                }
            },
            failure_reasons=[],
        )

        self.assertEqual("keep", recommendation["decision"])

    def test_dynamic_budget_summary_reads_top_level_dynamic_budget_comparison(self):
        module = self.load_gate_module()

        summary = module.dynamic_budget_summary(
            {
                "budget_summary": {
                    "mean_final_simulations": 128,
                    "mean_root_latency_ms": 6.5,
                },
                "dynamic_budget_comparison": {
                    "comparison_mode": "classic_dynamic_vs_fixed",
                    "runtime_target_matched": True,
                },
            },
            {
                "budget_summary": {
                    "mean_final_simulations": 96,
                    "mean_root_latency_ms": 6.2,
                }
            },
            {"budget_summary": {"trigger_counts": {"late_high_entropy": 6}}},
        )

        self.assertEqual(
            {
                "comparison_mode": "classic_dynamic_vs_fixed",
                "runtime_target_matched": True,
            },
            summary["dynamic_budget_comparison"],
        )


if __name__ == "__main__":
    unittest.main()
