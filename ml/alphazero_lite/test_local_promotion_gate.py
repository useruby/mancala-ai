import argparse
import json
import importlib.machinery
import importlib.util
import os
import sys
import types
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


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

    def test_run_regression_check_allows_failed_regression_report_exit_code(self):
        module = self.load_gate_module()

        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp_path = Path(tmp)
            report_path = tmp_path / "candidate_regression_suite.json"
            payload = {
                "passed": False,
                "artifact_path": "candidate",
                "positions_path": "test/fixtures/ai/superhuman_regression_positions.json",
                "results": [],
            }

            completed = subprocess.CompletedProcess(
                args=["script/ai/check_superhuman_regressions"],
                returncode=1,
                stdout=json.dumps(payload),
                stderr="",
            )

            with mock.patch.object(module.subprocess, "run", return_value=completed):
                report = module.run_regression_check(
                    ["script/ai/check_superhuman_regressions", "--artifact", "candidate"],
                    report_path,
                )

        self.assertFalse(report["passed"])

    def test_python_executable_falls_back_to_workspace_venv(self):
        module = self.load_gate_module()

        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp_path = Path(tmp)
            workspace_root = tmp_path / "workspace"
            workspace_python = workspace_root / ".venv/bin/python"
            worktree_root = workspace_root / "nested/worktree"
            workspace_python.parent.mkdir(parents=True)
            worktree_root.mkdir(parents=True)
            workspace_python.symlink_to(Path(sys.executable))

            original_file = module.__file__
            module.__file__ = str(worktree_root / "script/ai/local_promotion_gate")
            try:
                resolved = module.python_executable()
            finally:
                module.__file__ = original_file

        self.assertEqual(str(workspace_python), resolved)

    def write_forensic_report(
        self,
        path: Path,
        *,
        overall_current: dict,
        overall_challenger: dict,
        sparse_endgame_current: dict,
        sparse_endgame_challenger: dict,
        capture_available_current: dict,
        capture_available_challenger: dict,
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema": "azlite_forensic_suite_v1",
                    "systems": {
                        "current": {"overall": overall_current},
                        "challenger": {"overall": overall_challenger},
                    },
                    "buckets": {
                        "sparse_endgame": {
                            "systems": {
                                "current": sparse_endgame_current,
                                "challenger": sparse_endgame_challenger,
                            }
                        },
                        "capture_available": {
                            "systems": {
                                "current": capture_available_current,
                                "challenger": capture_available_challenger,
                            }
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

    def write_passing_forensic_report(self, path: Path) -> None:
        self.write_forensic_report(
            path,
            overall_current={"top1_agreement": 0.70, "average_regret": 0.10, "blunder_rate": 0.03},
            overall_challenger={"top1_agreement": 0.70, "average_regret": 0.10, "blunder_rate": 0.03},
            sparse_endgame_current={"top1_agreement": 0.60, "average_regret": 0.10, "blunder_rate": 0.04},
            sparse_endgame_challenger={"top1_agreement": 0.58, "average_regret": 0.13, "blunder_rate": 0.06},
            capture_available_current={"top1_agreement": 0.65, "average_regret": 0.12, "blunder_rate": 0.04},
            capture_available_challenger={"top1_agreement": 0.63, "average_regret": 0.15, "blunder_rate": 0.06},
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
            self.write_passing_forensic_report(tmp / "forensic.json")

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
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

    def test_rejects_stub_mode_without_forensic_report(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)
            self.write_regression_report(tmp / "regression.json", passed=True)
            self.write_passing_forensic_report(tmp / "forensic.json")

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
            self.assertIn("provide arena, candidate/current mcts, and forensic stub report paths together or none", result.stderr)

    def test_stub_mode_allows_omitting_regression_report_when_forensic_stub_is_present(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)
            self.write_passing_forensic_report(tmp / "forensic.json")

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
                "--out",
                str(out),
            )

            self.assertNotIn("config missing required step", result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("forensic_quality", report)
            self.assertTrue(report["regression_report_path"].endswith("candidate_regression_suite.json"))
            self.assertTrue(report["forensic_report_path"].endswith("forensic.json"))

    def test_rejects_only_forensic_stub_report_in_dry_run(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_forensic_report(
                tmp / "forensic.json",
                overall_current={"top1_agreement": 0.70, "average_regret": 0.10, "blunder_rate": 0.03},
                overall_challenger={"top1_agreement": 0.70, "average_regret": 0.10, "blunder_rate": 0.03},
                sparse_endgame_current={"top1_agreement": 0.60, "average_regret": 0.10, "blunder_rate": 0.04},
                sparse_endgame_challenger={"top1_agreement": 0.58, "average_regret": 0.13, "blunder_rate": 0.06},
                capture_available_current={"top1_agreement": 0.65, "average_regret": 0.12, "blunder_rate": 0.04},
                capture_available_challenger={"top1_agreement": 0.63, "average_regret": 0.15, "blunder_rate": 0.06},
            )

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
                "--dry-run",
                "--out",
                str(out),
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("--stub-forensic-report requires full stub bundle", result.stderr)

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
            self.write_passing_forensic_report(tmp / "forensic.json")

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
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
            self.write_passing_forensic_report(tmp / "forensic.json")

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
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
            self.write_passing_forensic_report(tmp / "forensic.json")

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
                "--out",
                str(out),
                "--max-arena-move-time-mean-ms",
                "200",
                "--max-arena-move-time-p95-ms",
                "250",
            )

            self.assertNotIn("config missing required step", result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("forensic_quality", report)
            self.assertTrue(report["forensic_quality"]["passed"])
            self.assertTrue(report["forensic_report_path"].endswith("forensic.json"))

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
            self.write_passing_forensic_report(tmp / "forensic.json")

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
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
            self.assertTrue(report["forensic_quality"]["passed"])
            self.assertTrue(report["forensic_report_path"].endswith("forensic.json"))

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

    def test_dry_run_report_includes_forensic_report_path(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"

            result = self.run_gate(
                "--candidate-path", str(candidate),
                "--dry-run",
                "--out", str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(report["forensic_report_path"].endswith("candidate_forensic_suite.json"))

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
            self.write_passing_forensic_report(tmp / "forensic.json")

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
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
            self.write_passing_forensic_report(stub_dir / "forensic.json")

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
                "--stub-forensic-report",
                str(stub_dir / "forensic.json"),
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
            self.write_passing_forensic_report(tmp / "forensic.json")

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
                "--out",
                str(out),
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("config missing required step", result.stderr)

    def test_dry_run_with_phase1_config_still_plans_threshold_evaluations(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--config-path",
                str(repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json"),
                "--dry-run",
                "--out",
                str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual([8, 10, 12], [row["threshold"] for row in report["endgame_exact_solve"]["planned_evaluations"]])

    def test_load_search_flag_map_can_scope_missing_phase2_steps_to_threshold_planning_only(self):
        module = self.load_gate_module()

        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            config_path = tmp / "search_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "steps": [
                            {
                                "name": "arena_confirm_report",
                                "command": [
                                    "python",
                                    "ml/alphazero_lite/arena.py",
                                    "--fpu-mode",
                                    "parent_q",
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit) as error:
                module.load_search_flag_map(str(config_path))

            self.assertIn("config missing required step", str(error.exception))
            scoped_flags = module.load_search_flag_map(str(config_path), require_steps=False)
            self.assertEqual(["--fpu-mode", "parent_q"], scoped_flags["arena"])
            self.assertEqual([], scoped_flags["candidate_mcts"])
            self.assertEqual([], scoped_flags["current_mcts"])

    def test_non_dry_run_allows_phase1_config_without_tracked_evaluation_steps(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            config_path = tmp / "phase1_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "steps": [
                            {
                                "name": "self_play",
                                "command": ["python", "ml/alphazero_lite/self_play.py", "--simulations", "256"],
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
            self.write_forensic_report(
                tmp / "forensic.json",
                overall_current={"top1_agreement": 0.61, "average_regret": 0.11, "blunder_rate": 0.03},
                overall_challenger={"top1_agreement": 0.64, "average_regret": 0.09, "blunder_rate": 0.02},
                sparse_endgame_current={"top1_agreement": 0.58, "average_regret": 0.14, "blunder_rate": 0.05},
                sparse_endgame_challenger={"top1_agreement": 0.62, "average_regret": 0.10, "blunder_rate": 0.03},
                capture_available_current={"top1_agreement": 0.55, "average_regret": 0.09, "blunder_rate": 0.02},
                capture_available_challenger={"top1_agreement": 0.52, "average_regret": 0.13, "blunder_rate": 0.05},
            )

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
                "--out",
                str(out),
            )

            self.assertNotIn("config missing required step", result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("forensic_quality", report)

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
                "AZLITE_FORENSIC_SUITE_STUB": "1",
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

            self.assertIn(result.returncode, (0, 1), msg=result.stderr)
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
            self.assertIn("forensic_quality", report)

    def test_build_real_decision_report_runs_and_loads_forensic_report(self):
        module = self.load_gate_module()

        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            current = tmp / "current"
            candidate.mkdir()
            current.mkdir()

            args = argparse.Namespace(
                candidate_path=candidate,
                config_path=None,
                current_path=str(current),
                hard_path=None,
                regression_positions_path="test/fixtures/ai/superhuman_regression_positions.json",
                arena_games=120,
                mcts_games=40,
                min_arena_score=0.55,
                min_arena_games=120,
                min_mcts_games=40,
                hard_arena_games=120,
                hard_min_score=0.55,
                max_arena_move_time_mean_ms=None,
                max_arena_move_time_p95_ms=None,
                require_lossless=False,
                max_losses=0,
                skip_mcts_relative_check=False,
            )

            report = {
                "schema": "azlite_local_promotion_gate_v1",
                "candidate_path": str(candidate),
                "current_path": str(current),
                "hard_path": None,
                "report_path": str(tmp / "report.json"),
                "regression_report_path": str(tmp / "candidate_regression_suite.json"),
                "forensic_report_path": str(tmp / "candidate_forensic_suite.json"),
                "evaluations": [
                    module.build_evaluation(
                        evaluation_id="candidate_vs_current_arena",
                        subject=str(candidate),
                        opponent=str(current),
                        games=120,
                        report_dir=tmp,
                        mode="planned",
                    ),
                    module.build_evaluation(
                        evaluation_id="candidate_vs_mcts1200",
                        subject=str(candidate),
                        opponent="mcts1200",
                        games=40,
                        report_dir=tmp,
                        mode="planned",
                    ),
                    module.build_evaluation(
                        evaluation_id="current_vs_mcts1200",
                        subject=str(current),
                        opponent="mcts1200",
                        games=40,
                        report_dir=tmp,
                        mode="planned",
                    ),
                    module.forensic_evaluation(tmp, candidate_path=candidate, current_path=str(current)),
                ],
                "endgame_exact_solve": module.build_endgame_exact_solve_section(
                    status="planned_only",
                    results_source="evaluation_plan",
                    results=[],
                ),
            }

            def fake_run_command(command):
                out_path = Path(command[command.index("--out") + 1])
                if command[1] == "ml/alphazero_lite/arena.py":
                    payload = {
                        "schema": "arena_v1",
                        "wins": 92,
                        "losses": 0,
                        "draws": 28,
                        "games_played": 120,
                        "promotion_decision": {"passed": True},
                    }
                elif command[1] == "ml/alphazero_lite/mcts1200_baseline.py":
                    payload = {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 40,
                        "az_wins": 30 if str(candidate) in command else 24,
                        "mcts_wins": 2 if str(candidate) in command else 8,
                        "draws": 8,
                    }
                elif command[1] == "ml/alphazero_lite/run_forensic_suite.py":
                    payload = {
                        "schema": "azlite_forensic_suite_v1",
                        "systems": {
                            "current": {"overall": {"top1_agreement": 0.70, "average_regret": 0.10, "blunder_rate": 0.03}},
                            "challenger": {"overall": {"top1_agreement": 0.70, "average_regret": 0.10, "blunder_rate": 0.03}},
                        },
                        "buckets": {
                            "sparse_endgame": {
                                "systems": {
                                    "current": {"top1_agreement": 0.60, "average_regret": 0.10, "blunder_rate": 0.04},
                                    "challenger": {"top1_agreement": 0.58, "average_regret": 0.13, "blunder_rate": 0.06},
                                }
                            },
                            "capture_available": {
                                "systems": {
                                    "current": {"top1_agreement": 0.65, "average_regret": 0.12, "blunder_rate": 0.04},
                                    "challenger": {"top1_agreement": 0.63, "average_regret": 0.15, "blunder_rate": 0.06},
                                }
                            },
                        },
                    }
                else:
                    raise AssertionError(f"unexpected command: {command}")
                out_path.write_text(json.dumps(payload), encoding="utf-8")
                recorded_commands.append(command)

            def fake_run_regression_check(command, report_path):
                report = {
                    "passed": True,
                    "artifact_path": str(candidate),
                    "positions_path": "test/fixtures/ai/superhuman_regression_positions.json",
                    "results": [],
                }
                report_path.write_text(json.dumps(report), encoding="utf-8")
                return report

            original_run_command = module.run_command
            original_run_regression_check = module.run_regression_check
            recorded_commands = []
            try:
                module.run_command = fake_run_command
                module.run_regression_check = fake_run_regression_check

                result = module.build_real_decision_report(args, report)
            finally:
                module.run_command = original_run_command
                module.run_regression_check = original_run_regression_check

            self.assertEqual(str(tmp / "candidate_forensic_suite.json"), result["forensic_report_path"])
            self.assertTrue(result["forensic_quality"]["passed"])
            forensic_command = next(command for command in recorded_commands if command[1] == "ml/alphazero_lite/run_forensic_suite.py")
            self.assertEqual("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json", forensic_command[forensic_command.index("--suite") + 1])
            self.assertEqual(str(tmp / "candidate_forensic_suite.json"), forensic_command[forensic_command.index("--out") + 1])

    def test_build_real_decision_report_fails_when_forensic_report_regresses(self):
        module = self.load_gate_module()

        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            current = tmp / "current"
            candidate.mkdir()
            current.mkdir()

            args = argparse.Namespace(
                candidate_path=candidate,
                config_path=None,
                current_path=str(current),
                hard_path=None,
                regression_positions_path="test/fixtures/ai/superhuman_regression_positions.json",
                arena_games=120,
                mcts_games=40,
                min_arena_score=0.55,
                min_arena_games=120,
                min_mcts_games=40,
                hard_arena_games=120,
                hard_min_score=0.55,
                max_arena_move_time_mean_ms=None,
                max_arena_move_time_p95_ms=None,
                require_lossless=False,
                max_losses=0,
                skip_mcts_relative_check=False,
            )

            report = {
                "schema": "azlite_local_promotion_gate_v1",
                "candidate_path": str(candidate),
                "current_path": str(current),
                "hard_path": None,
                "report_path": str(tmp / "report.json"),
                "regression_report_path": str(tmp / "candidate_regression_suite.json"),
                "forensic_report_path": str(tmp / "candidate_forensic_suite.json"),
                "evaluations": [
                    module.build_evaluation(
                        evaluation_id="candidate_vs_current_arena",
                        subject=str(candidate),
                        opponent=str(current),
                        games=120,
                        report_dir=tmp,
                        mode="planned",
                    ),
                    module.build_evaluation(
                        evaluation_id="candidate_vs_mcts1200",
                        subject=str(candidate),
                        opponent="mcts1200",
                        games=40,
                        report_dir=tmp,
                        mode="planned",
                    ),
                    module.build_evaluation(
                        evaluation_id="current_vs_mcts1200",
                        subject=str(current),
                        opponent="mcts1200",
                        games=40,
                        report_dir=tmp,
                        mode="planned",
                    ),
                    module.forensic_evaluation(tmp, candidate_path=candidate, current_path=str(current)),
                ],
                "endgame_exact_solve": module.build_endgame_exact_solve_section(
                    status="planned_only",
                    results_source="evaluation_plan",
                    results=[],
                ),
            }

            recorded_commands = []

            def fake_run_command(command):
                out_path = Path(command[command.index("--out") + 1])
                if command[1] == "ml/alphazero_lite/arena.py":
                    payload = {
                        "schema": "arena_v1",
                        "wins": 92,
                        "losses": 0,
                        "draws": 28,
                        "games_played": 120,
                        "promotion_decision": {"passed": True},
                    }
                elif command[1] == "ml/alphazero_lite/mcts1200_baseline.py":
                    payload = {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 40,
                        "az_wins": 30 if str(candidate) in command else 24,
                        "mcts_wins": 2 if str(candidate) in command else 8,
                        "draws": 8,
                    }
                elif command[1] == "ml/alphazero_lite/run_forensic_suite.py":
                    payload = {
                        "schema": "azlite_forensic_suite_v1",
                        "systems": {
                            "current": {"overall": {"top1_agreement": 0.70, "average_regret": 0.10, "blunder_rate": 0.03}},
                            "challenger": {"overall": {"top1_agreement": 0.70, "average_regret": 0.10, "blunder_rate": 0.03}},
                        },
                        "buckets": {
                            "sparse_endgame": {
                                "systems": {
                                    "current": {"top1_agreement": 0.60, "average_regret": 0.10, "blunder_rate": 0.04},
                                    "challenger": {"top1_agreement": 0.55, "average_regret": 0.17, "blunder_rate": 0.09},
                                }
                            },
                            "capture_available": {
                                "systems": {
                                    "current": {"top1_agreement": 0.65, "average_regret": 0.12, "blunder_rate": 0.04},
                                    "challenger": {"top1_agreement": 0.63, "average_regret": 0.15, "blunder_rate": 0.06},
                                }
                            },
                        },
                    }
                else:
                    raise AssertionError(f"unexpected command: {command}")
                out_path.write_text(json.dumps(payload), encoding="utf-8")
                recorded_commands.append(command)

            def fake_run_regression_check(command, report_path):
                regression_report = {
                    "passed": True,
                    "artifact_path": str(candidate),
                    "positions_path": "test/fixtures/ai/superhuman_regression_positions.json",
                    "results": [],
                }
                report_path.write_text(json.dumps(regression_report), encoding="utf-8")
                return regression_report

            original_run_command = module.run_command
            original_run_regression_check = module.run_regression_check
            try:
                module.run_command = fake_run_command
                module.run_regression_check = fake_run_regression_check

                result = module.build_real_decision_report(args, report)
            finally:
                module.run_command = original_run_command
                module.run_regression_check = original_run_regression_check

            self.assertEqual(str(tmp / "candidate_forensic_suite.json"), result["forensic_report_path"])
            self.assertFalse(result["passed"])
            self.assertFalse(result["forensic_quality"]["passed"])
            self.assertTrue(any(reason["code"] == "forensic_bucket_sparse_endgame_regressed" for reason in result["failure_reasons"]))
            forensic_command = next(command for command in recorded_commands if command[1] == "ml/alphazero_lite/run_forensic_suite.py")
            self.assertEqual(str(tmp / "candidate_forensic_suite.json"), forensic_command[forensic_command.index("--out") + 1])

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
            self.write_passing_forensic_report(tmp / "forensic.json")

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
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
            self.write_passing_forensic_report(tmp / "forensic.json")

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
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
            self.write_passing_forensic_report(tmp / "forensic.json")

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
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
            self.write_passing_forensic_report(tmp / "forensic.json")

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
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
            self.write_passing_forensic_report(tmp / "forensic.json")

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
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
            self.write_passing_forensic_report(tmp / "forensic.json")

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
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
            self.write_passing_forensic_report(tmp / "forensic.json")

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
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
            self.write_passing_forensic_report(tmp / "forensic.json")

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
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
            self.write_passing_forensic_report(tmp / "forensic.json")

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
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
            self.write_passing_forensic_report(tmp / "forensic.json")

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
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
            self.write_passing_forensic_report(tmp / "forensic.json")

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
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
            self.write_passing_forensic_report(tmp / "forensic.json")

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
                "--out",
                str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual("tune", report["dynamic_budget_recommendation"]["decision"])
            self.assertIn("mixed", report["dynamic_budget_recommendation"]["reason"])

    def test_stub_decision_report_fails_when_critical_forensic_bucket_regresses(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)
            self.write_regression_report(tmp / "regression.json", passed=True)
            self.write_forensic_report(
                tmp / "forensic.json",
                overall_current={"top1_agreement": 0.70, "average_regret": 0.10, "blunder_rate": 0.03},
                overall_challenger={"top1_agreement": 0.70, "average_regret": 0.10, "blunder_rate": 0.03},
                sparse_endgame_current={"top1_agreement": 0.60, "average_regret": 0.10, "blunder_rate": 0.04},
                sparse_endgame_challenger={"top1_agreement": 0.55, "average_regret": 0.14, "blunder_rate": 0.07},
                capture_available_current={"top1_agreement": 0.65, "average_regret": 0.12, "blunder_rate": 0.04},
                capture_available_challenger={"top1_agreement": 0.65, "average_regret": 0.12, "blunder_rate": 0.04},
            )

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
                "--out",
                str(out),
            )

            self.assertNotEqual(0, result.returncode)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertTrue(any(reason["code"] == "forensic_bucket_sparse_endgame_regressed" for reason in report["failure_reasons"]))

    def test_stub_decision_report_passes_with_clean_forensic_quality(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)
            self.write_regression_report(tmp / "regression.json", passed=True)
            self.write_forensic_report(
                tmp / "forensic.json",
                overall_current={"top1_agreement": 0.70, "average_regret": 0.10, "blunder_rate": 0.03},
                overall_challenger={"top1_agreement": 0.70, "average_regret": 0.10, "blunder_rate": 0.03},
                sparse_endgame_current={"top1_agreement": 0.60, "average_regret": 0.10, "blunder_rate": 0.04},
                sparse_endgame_challenger={"top1_agreement": 0.58, "average_regret": 0.13, "blunder_rate": 0.06},
                capture_available_current={"top1_agreement": 0.65, "average_regret": 0.12, "blunder_rate": 0.04},
                capture_available_challenger={"top1_agreement": 0.63, "average_regret": 0.15, "blunder_rate": 0.06},
            )

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
                "--out",
                str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("forensic_quality", report)
            self.assertTrue(report["forensic_quality"]["passed"])

    def test_stub_decision_report_fails_when_overall_top1_agreement_delta_exceeds_threshold_by_rounding_margin(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)
            self.write_regression_report(tmp / "regression.json", passed=True)
            self.write_forensic_report(
                tmp / "forensic.json",
                overall_current={"top1_agreement": 0.70, "average_regret": 0.10, "blunder_rate": 0.03},
                overall_challenger={"top1_agreement": 0.67996, "average_regret": 0.10, "blunder_rate": 0.03},
                sparse_endgame_current={"top1_agreement": 0.60, "average_regret": 0.10, "blunder_rate": 0.04},
                sparse_endgame_challenger={"top1_agreement": 0.60, "average_regret": 0.10, "blunder_rate": 0.04},
                capture_available_current={"top1_agreement": 0.65, "average_regret": 0.12, "blunder_rate": 0.04},
                capture_available_challenger={"top1_agreement": 0.65, "average_regret": 0.12, "blunder_rate": 0.04},
            )

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
                "--out",
                str(out),
            )

            self.assertNotEqual(0, result.returncode)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertFalse(report["forensic_quality"]["overall"]["passed"])
            self.assertEqual(-0.02, report["forensic_quality"]["overall"]["top1_agreement"]["delta"])
            self.assertFalse(report["forensic_quality"]["overall"]["top1_agreement"]["passed"])
            self.assertTrue(any(reason["code"] == "forensic_overall_regressed" for reason in report["failure_reasons"]))

    def test_stub_decision_report_fails_when_bucket_average_regret_delta_exceeds_threshold_by_rounding_margin(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)
            self.write_regression_report(tmp / "regression.json", passed=True)
            self.write_forensic_report(
                tmp / "forensic.json",
                overall_current={"top1_agreement": 0.70, "average_regret": 0.10, "blunder_rate": 0.03},
                overall_challenger={"top1_agreement": 0.70, "average_regret": 0.10, "blunder_rate": 0.03},
                sparse_endgame_current={"top1_agreement": 0.60, "average_regret": 0.10, "blunder_rate": 0.04},
                sparse_endgame_challenger={"top1_agreement": 0.60, "average_regret": 0.13004, "blunder_rate": 0.04},
                capture_available_current={"top1_agreement": 0.65, "average_regret": 0.12, "blunder_rate": 0.04},
                capture_available_challenger={"top1_agreement": 0.65, "average_regret": 0.12, "blunder_rate": 0.04},
            )

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
                "--out",
                str(out),
            )

            self.assertNotEqual(0, result.returncode)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertFalse(report["forensic_quality"]["critical_buckets"]["sparse_endgame"]["passed"])
            self.assertEqual(0.03, report["forensic_quality"]["critical_buckets"]["sparse_endgame"]["average_regret"]["delta"])
            self.assertFalse(report["forensic_quality"]["critical_buckets"]["sparse_endgame"]["average_regret"]["passed"])
            self.assertTrue(any(reason["code"] == "forensic_bucket_sparse_endgame_regressed" for reason in report["failure_reasons"]))

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

    def test_dynamic_budget_summary_includes_opening_cache_keep_signal(self):
        module = self.load_gate_module()

        summary = module.dynamic_budget_summary(
            candidate_mcts_report={"budget_summary": {}},
            current_mcts_report={"budget_summary": {}},
            arena_report={
                "budget_summary": {},
                "opening_cache_summary": {
                    "runtime_hit_rate": 0.25,
                    "training_hit_rate": 0.4,
                    "opening_bucket_quality_delta": 0.03,
                    "latency_delta_ms": -5.2,
                },
            },
        )

        self.assertEqual(0.25, summary["opening_cache_summary"]["runtime_hit_rate"])
        self.assertEqual(0.4, summary["opening_cache_summary"]["training_hit_rate"])
        self.assertEqual(0.03, summary["opening_cache_summary"]["opening_bucket_quality_delta"])
        self.assertEqual(-5.2, summary["opening_cache_summary"]["latency_delta_ms"])

    def test_opening_cache_recommendation_keeps_when_cache_evidence_is_favorable(self):
        module = self.load_gate_module()

        recommendation = module.build_opening_cache_recommendation(
            opening_cache_summary={
                "runtime_hit_rate": 0.25,
                "training_hit_rate": 0.4,
                "opening_bucket_quality_delta": 0.03,
                "latency_delta_ms": -5.2,
            }
        )

        self.assertEqual("keep", recommendation["decision"])

    def test_opening_cache_recommendation_tunes_when_runtime_cache_evidence_is_unavailable(self):
        module = self.load_gate_module()

        recommendation = module.build_opening_cache_recommendation(
            opening_cache_summary={
                "runtime_hit_rate": None,
                "training_hit_rate": 0.4,
                "opening_bucket_quality_delta": None,
                "latency_delta_ms": None,
            }
        )

        self.assertEqual("tune", recommendation["decision"])

    def test_opening_cache_recommendation_drop_reason_describes_unfavorable_evidence(self):
        module = self.load_gate_module()

        recommendation = module.build_opening_cache_recommendation(
            opening_cache_summary={
                "runtime_hit_rate": 0.2,
                "training_hit_rate": 0.3,
                "opening_bucket_quality_delta": -0.1,
                "latency_delta_ms": 4.0,
            }
        )

        self.assertEqual("drop", recommendation["decision"])
        self.assertIn("unfavorable", recommendation["reason"])
        self.assertNotIn("missing", recommendation["reason"])

    def test_stub_decision_report_tunes_when_opening_cache_runtime_evidence_is_unavailable(self):
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
                        "opening_cache_summary": {
                            "runtime_hit_rate": None,
                            "training_hit_rate": 0.4,
                            "opening_bucket_quality_delta": None,
                            "latency_delta_ms": None,
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
                    }
                ),
                encoding="utf-8",
            )
            self.write_regression_report(tmp / "regression.json", passed=True)
            self.write_forensic_report(
                tmp / "forensic.json",
                overall_current={"top1_agreement": 0.61, "average_regret": 0.11, "blunder_rate": 0.03},
                overall_challenger={"top1_agreement": 0.64, "average_regret": 0.09, "blunder_rate": 0.02},
                sparse_endgame_current={"top1_agreement": 0.58, "average_regret": 0.14, "blunder_rate": 0.05},
                sparse_endgame_challenger={"top1_agreement": 0.62, "average_regret": 0.10, "blunder_rate": 0.03},
                capture_available_current={"top1_agreement": 0.55, "average_regret": 0.09, "blunder_rate": 0.02},
                capture_available_challenger={"top1_agreement": 0.52, "average_regret": 0.13, "blunder_rate": 0.05},
            )

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
                "--out",
                str(out),
            )

            self.assertIn(result.returncode, (0, 1), msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual("tune", report["opening_cache_recommendation"]["decision"])

    def test_stub_decision_report_emits_opening_cache_recommendation(self):
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
            (tmp / "cand_mcts.json").write_text(
                json.dumps(
                    {
                        "schema": "azlite_vs_mcts_v1",
                        "games": 40,
                        "az_wins": 30,
                        "mcts_wins": 2,
                        "draws": 8,
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
                    }
                ),
                encoding="utf-8",
            )
            self.write_regression_report(tmp / "regression.json", passed=True)
            self.write_forensic_report(
                tmp / "forensic.json",
                overall_current={"top1_agreement": 0.61, "average_regret": 0.11, "blunder_rate": 0.03},
                overall_challenger={"top1_agreement": 0.64, "average_regret": 0.09, "blunder_rate": 0.02},
                sparse_endgame_current={"top1_agreement": 0.58, "average_regret": 0.14, "blunder_rate": 0.05},
                sparse_endgame_challenger={"top1_agreement": 0.62, "average_regret": 0.10, "blunder_rate": 0.03},
                capture_available_current={"top1_agreement": 0.55, "average_regret": 0.09, "blunder_rate": 0.02},
                capture_available_challenger={"top1_agreement": 0.52, "average_regret": 0.13, "blunder_rate": 0.05},
            )

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
                "--stub-forensic-report",
                str(tmp / "forensic.json"),
                "--out",
                str(out),
            )

            self.assertIn(result.returncode, (0, 1), msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual("keep", report["opening_cache_recommendation"]["decision"])

    def test_run_regression_check_accepts_exit_code_one_with_valid_report(self):
        module = self.load_gate_module()

        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            report_path = Path(tmp) / "regression.json"
            payload = json.dumps(
                {
                    "passed": False,
                    "artifact_path": "candidate",
                    "positions_path": "positions.json",
                    "results": [{"passed": False}],
                }
            )

            original_run = module.subprocess.run
            module.subprocess.run = lambda *args, **kwargs: types.SimpleNamespace(returncode=1, stdout=payload, stderr="")
            try:
                report = module.run_regression_check(["script/ai/check_superhuman_regressions"], report_path)
            finally:
                module.subprocess.run = original_run

        self.assertFalse(report["passed"])

    def test_python_executable_prefers_shared_workspace_venv_when_worktree_venv_is_missing(self):
        module = self.load_gate_module()
        repo_root = module.repo_root()
        worktree_python = repo_root / ".venv/bin/python"
        shared_python = repo_root.parents[1] / ".venv/bin/python"

        original_is_file = Path.is_file
        original_access = os.access

        def fake_is_file(path_self):
            return path_self == shared_python

        def fake_access(path_value, mode):
            return path_value == shared_python and mode == os.X_OK

        Path.is_file = fake_is_file
        os.access = fake_access
        try:
            self.assertEqual(str(shared_python), module.python_executable())
        finally:
            Path.is_file = original_is_file
            os.access = original_access

    def test_python_executable_falls_back_to_sys_executable_outside_worktrees(self):
        module = self.load_gate_module()
        original_repo_root = module.repo_root
        module.repo_root = lambda: Path("/tmp/project")
        try:
            self.assertEqual(sys.executable, module.python_executable())
        finally:
            module.repo_root = original_repo_root

    def test_forensic_quality_summary_extracts_overall_and_bucket_metrics(self):
        module = self.load_gate_module()
        report = {
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
                        "current": {"top1_agreement": 0.58, "average_regret": 0.14, "blunder_rate": 0.05},
                        "challenger": {"top1_agreement": 0.62, "average_regret": 0.10, "blunder_rate": 0.03},
                    }
                },
                "capture_available": {
                    "systems": {
                        "current": {"top1_agreement": 0.55, "average_regret": 0.09, "blunder_rate": 0.02},
                        "challenger": {"top1_agreement": 0.52, "average_regret": 0.13, "blunder_rate": 0.05},
                    }
                },
            },
        }

        summary = module.forensic_quality_summary(report)

        self.assertEqual(0.64, summary["overall"]["challenger"]["top1_agreement"])
        self.assertEqual(0.09, summary["overall"]["challenger"]["average_regret"])
        self.assertEqual(0.02, summary["overall"]["challenger"]["blunder_rate"])
        self.assertEqual(0.61, summary["overall"]["current"]["top1_agreement"])
        self.assertEqual(0.11, summary["overall"]["current"]["average_regret"])
        self.assertEqual(0.03, summary["overall"]["current"]["blunder_rate"])
        self.assertIn("sparse_endgame", summary["critical_buckets"])
        self.assertIn("capture_available", summary["critical_buckets"])
        self.assertEqual(0.13, summary["critical_buckets"]["capture_available"]["challenger"]["average_regret"])
        self.assertEqual(0.05, summary["critical_buckets"]["capture_available"]["challenger"]["blunder_rate"])

    def test_forensic_quality_summary_rejects_missing_required_bucket(self):
        module = self.load_gate_module()
        report = {
            "schema": "azlite_forensic_suite_v1",
            "systems": {
                "current": {"overall": {"top1_agreement": 0.5, "average_regret": 0.1, "blunder_rate": 0.0}},
                "challenger": {"overall": {"top1_agreement": 0.5, "average_regret": 0.1, "blunder_rate": 0.0}},
            },
            "buckets": {},
        }

        with self.assertRaisesRegex(SystemExit, "missing required forensic bucket"):
            module.forensic_quality_summary(report)

    def test_forensic_quality_summary_rejects_non_finite_metric(self):
        module = self.load_gate_module()
        report = {
            "schema": "azlite_forensic_suite_v1",
            "systems": {
                "current": {
                    "overall": {
                        "top1_agreement": 0.61,
                        "average_regret": float("nan"),
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
                        "current": {"top1_agreement": 0.58, "average_regret": 0.14, "blunder_rate": 0.05},
                        "challenger": {"top1_agreement": 0.62, "average_regret": 0.10, "blunder_rate": 0.03},
                    }
                },
                "capture_available": {
                    "systems": {
                        "current": {"top1_agreement": 0.55, "average_regret": 0.09, "blunder_rate": 0.02},
                        "challenger": {"top1_agreement": 0.52, "average_regret": 0.13, "blunder_rate": 0.05},
                    }
                },
            },
        }

        with self.assertRaisesRegex(SystemExit, "non-finite"):
            module.forensic_quality_summary(report)

    def test_numeric_metric_rejects_infinite_value(self):
        module = self.load_gate_module()

        with self.assertRaisesRegex(SystemExit, "non-finite"):
            module.numeric_metric({"top1_agreement": float("inf")}, "top1_agreement", context="systems.current.overall")

    def test_numeric_metric_rejects_blunder_rate_above_one(self):
        module = self.load_gate_module()

        with self.assertRaisesRegex(SystemExit, "blunder_rate: expected value between 0.0 and 1.0"):
            module.numeric_metric({"blunder_rate": 1.2}, "blunder_rate", context="systems.current.overall")

    def test_numeric_metric_rejects_negative_blunder_rate(self):
        module = self.load_gate_module()

        with self.assertRaisesRegex(SystemExit, "blunder_rate: expected value between 0.0 and 1.0"):
            module.numeric_metric({"blunder_rate": -0.01}, "blunder_rate", context="systems.current.overall")

        with self.assertRaisesRegex(SystemExit, "top1_agreement: expected value between 0.0 and 1.0"):
            module.numeric_metric({"top1_agreement": -0.01}, "top1_agreement", context="systems.current.overall")

    def test_numeric_metric_rejects_negative_average_regret(self):
        module = self.load_gate_module()

        with self.assertRaisesRegex(SystemExit, "average_regret: expected value greater than or equal to 0.0"):
            module.numeric_metric({"average_regret": -0.01}, "average_regret", context="systems.current.overall")


if __name__ == "__main__":
    unittest.main()
