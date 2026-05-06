import importlib.machinery
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path


class ModelRobustnessConfirmationTest(unittest.TestCase):
    def load_module(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "script/ai/model_robustness_confirmation"
        loader = importlib.machinery.SourceFileLoader("model_robustness_confirmation", str(script_path))
        module = types.ModuleType(loader.name)
        module.__file__ = str(script_path)
        loader.exec_module(module)
        return module

    def test_dry_run_includes_aggregate_summary_location(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-robustness-confirmation-") as tmp:
            output_root = Path(tmp)
            result = subprocess.run(
                [
                    "script/ai/model_robustness_confirmation",
                    "--dry-run",
                    "--output-root",
                    str(output_root),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        plan = json.loads(result.stdout)

        self.assertEqual(str(output_root / "aggregate_summary.json"), plan["aggregate_summary_path"])
        self.assertEqual(5, len(plan["lanes"]))

    def test_dry_run_includes_required_qualifying_seed_count(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-robustness-confirmation-") as tmp:
            output_root = Path(tmp)
            result = subprocess.run(
                [
                    "script/ai/model_robustness_confirmation",
                    "--dry-run",
                    "--output-root",
                    str(output_root),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        plan = json.loads(result.stdout)
        self.assertIn("required_qualifying_seed_count", plan)

    def test_aggregate_summary_includes_qualifying_seed_count(self):
        module = self.load_module()

        lane_summaries = [
            {"seed": 41, "benchmark_passed": True, "local_promotion_gate_passed": True},
            {"seed": 42, "benchmark_passed": True, "local_promotion_gate_passed": True},
            {"seed": 43, "benchmark_passed": True, "local_promotion_gate_passed": False},
        ]

        aggregate_summary = module.build_aggregate_summary(
            parent_artifact="parent",
            current_path="current",
            lane_summaries=lane_summaries,
            required_qualifying_seed_count=2,
        )

        self.assertEqual(2, aggregate_summary["qualifying_seed_count"])
        self.assertEqual(2, aggregate_summary["required_qualifying_seed_count"])
        self.assertTrue(aggregate_summary["passed"])
        self.assertFalse(aggregate_summary["all_lanes_passed"])

    def test_aggregate_summary_passes_when_required_qualifying_seed_count_is_met(self):
        module = self.load_module()

        lane_summaries = [
            {"seed": 41, "benchmark_passed": True, "local_promotion_gate_passed": True},
            {"seed": 42, "benchmark_passed": True, "local_promotion_gate_passed": True},
            {"seed": 43, "benchmark_passed": False, "local_promotion_gate_passed": False},
        ]

        aggregate_summary = module.build_aggregate_summary(
            parent_artifact="parent",
            current_path="current",
            lane_summaries=lane_summaries,
            required_qualifying_seed_count=2,
        )

        self.assertTrue(aggregate_summary["passed"])

    def test_build_lane_config_overrides_runtime_current_path_and_records_parent_artifact_path(self):
        module = self.load_module()

        base_config = {
            "run_id": "custom-run",
            "seed": 42,
            "current_path": "base/current",
            "steps": [
                {
                    "name": "benchmark_contract",
                    "command": [
                        "python",
                        "benchmark.py",
                        "--current-path",
                        "{current_path}",
                    ],
                }
            ],
        }
        lane = module.build_lane(41, base_run_id="custom-run", iteration=2, output_root=Path("/tmp/robustness"))

        lane_config = module.build_lane_config(
            base_config,
            lane,
            parent_artifact="parent/artifact",
            current_path="runtime/current",
        )

        self.assertEqual("runtime/current", lane_config["current_path"])
        self.assertEqual("parent/artifact", lane_config["parent_artifact_path"])
        self.assertEqual(
            ["python", "benchmark.py", "--current-path", "{current_path}", "--min-confidence-lower-bound", "0.55"],
            lane_config["steps"][0]["command"],
        )

    def test_build_lane_config_injects_replay_dataset_for_tactical_replay_train_step(self):
        module = self.load_module()

        base_config = {
            "run_id": "tactical-replay",
            "seed": 42,
            "current_path": "base/current",
            "fixed_replay_sources": [
                {"path": "ml/alphazero_lite/tactical_capture_protection.jsonl", "weight": 8}
            ],
            "steps": [
                {
                    "name": "train",
                    "command": [
                        "python",
                        "train.py",
                        "--data-files",
                        "{replay_data}",
                        "--replay-weights",
                        "{replay_weights}",
                    ],
                }
            ],
        }
        lane = module.build_lane(41, base_run_id="tactical-replay", iteration=1, output_root=Path("/tmp/robustness"))

        lane_config = module.build_lane_config(
            base_config,
            lane,
            parent_artifact="parent/artifact",
            current_path="runtime/current",
        )

        self.assertEqual(
            [
                {
                    "path": str(Path(lane["lane_root"]) / "replay" / "labeled_tactical_states.jsonl"),
                    "weight": 1,
                },
                {"path": "ml/alphazero_lite/tactical_capture_protection.jsonl", "weight": 8},
            ],
            lane_config["fixed_replay_sources"],
        )
        self.assertEqual("{replay_data}", lane_config["steps"][0]["command"][3])
        self.assertEqual("{replay_weights}", lane_config["steps"][0]["command"][5])

    def test_build_lane_config_allows_legacy_lane_without_replay_metadata(self):
        module = self.load_module()

        base_config = {
            "run_id": "custom-run",
            "seed": 42,
            "current_path": "base/current",
            "steps": [
                {
                    "name": "train",
                    "command": [
                        "python",
                        "train.py",
                        "--data-files",
                        "existing-dataset.jsonl",
                    ],
                }
            ],
        }
        lane = {
            "seed": 41,
            "seed_sweep": "41,141,241",
            "run_id": "custom-run-robust-s41",
            "results_dir": "/tmp/robustness/custom-run-robust-s41/versions",
        }

        lane_config = module.build_lane_config(
            base_config,
            lane,
            parent_artifact="parent/artifact",
            current_path="runtime/current",
        )

        self.assertEqual(
            ["python", "train.py", "--data-files", "existing-dataset.jsonl"],
            lane_config["steps"][0]["command"],
        )

    def test_build_lane_config_rejects_partial_replay_metadata(self):
        module = self.load_module()

        base_config = {
            "run_id": "custom-run",
            "seed": 42,
            "current_path": "base/current",
            "steps": [],
        }
        lane = {
            "seed": 41,
            "run_id": "custom-run-robust-s41",
            "results_dir": "/tmp/robustness/custom-run-robust-s41/versions",
            "replay_dir": "/tmp/robustness/custom-run-robust-s41/replay",
        }

        with self.assertRaisesRegex(SystemExit, "replay_dir and replay_source_path"):
            module.build_lane_config(
                base_config,
                lane,
                parent_artifact="parent/artifact",
                current_path="runtime/current",
            )

    def test_main_rejects_missing_default_replay_source_before_running_lanes(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory(prefix="azlite-robustness-plan-") as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "run"
            base_config_path = tmp_path / "base_config.json"
            base_config_path.write_text(
                json.dumps(
                    {
                        "run_id": "custom-run",
                        "start_iteration": 1,
                        "iterations": 1,
                        "steps": [],
                    }
                ),
                encoding="utf-8",
            )

            namespace = type("Args", (), {})()
            namespace.base_config = str(base_config_path)
            namespace.parent_artifact = "parent/artifact"
            namespace.current_path = "model-artifact/current"
            namespace.output_root = str(output_root)
            namespace.dry_run = False

            original_parse_args = module.parse_args
            original_require_existing_runtime_path = module.require_existing_runtime_path
            original_validate_output_root_safety = module.validate_output_root_safety
            module.parse_args = lambda: namespace
            module.require_existing_runtime_path = lambda path_value, **kwargs: Path(tmp_path / path_value.replace("/", "_"))
            module.validate_output_root_safety = lambda **kwargs: None
            try:
                with self.assertRaisesRegex(SystemExit, "replay source does not exist"):
                    module.main()
            finally:
                module.parse_args = original_parse_args
                module.require_existing_runtime_path = original_require_existing_runtime_path
                module.validate_output_root_safety = original_validate_output_root_safety

    def test_python_executable_falls_back_to_workspace_venv(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory(prefix="azlite-robustness-python-") as tmp:
            tmp_path = Path(tmp)
            workspace_root = tmp_path / "workspace"
            workspace_python = workspace_root / ".venv/bin/python"
            repo_root = workspace_root / "nested/worktree"
            workspace_python.parent.mkdir(parents=True)
            repo_root.mkdir(parents=True)
            workspace_python.symlink_to(Path(sys.executable))

            original_repo_root = module.REPO_ROOT
            module.REPO_ROOT = repo_root
            try:
                resolved = module.python_executable()
            finally:
                module.REPO_ROOT = original_repo_root

        self.assertEqual(str(workspace_python), resolved)

    def test_execute_lane_includes_hard_state_summary_when_present(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "versions" / "custom-run-iter4"
            candidate_path.mkdir(parents=True)
            benchmark_report_path = candidate_path / "benchmark_report.json"
            hard_state_report_path = candidate_path / "hard_state_validation.json"
            gate_report_path = tmp_path / "lane" / "local_promotion_gate.json"
            summary_path = tmp_path / "lane" / "seed_summary.json"
            config_path = tmp_path / "custom-run.json"

            benchmark_report_path.write_text(
                json.dumps(
                    {
                        "checks": [{"id": "runtime_parity", "passed": True}],
                        "arena_confidence_lower_bound": 0.56,
                    }
                ),
                encoding="utf-8",
            )
            hard_state_report_path.write_text(
                json.dumps(
                    {
                        "policy_top1_agreement": 0.82,
                        "average_regret": 0.07,
                        "value_calibration_mae": 0.11,
                    }
                ),
                encoding="utf-8",
            )
            gate_report_path.parent.mkdir(parents=True)
            gate_report_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "arena_score": 0.61,
                        "candidate_mcts_score": 0.57,
                        "current_mcts_score": 0.53,
                        "failure_reasons": [],
                    }
                ),
                encoding="utf-8",
            )

            lane = {
                "seed": 41,
                "candidate_path": str(candidate_path),
                "lane_root": str(gate_report_path.parent),
                "results_dir": str(tmp_path / "versions"),
                "replay_dir": str(tmp_path / "lane" / "replay"),
                "replay_source_path": str(tmp_path / "source" / "labeled_tactical_states.jsonl"),
                "config_path": str(config_path),
                "gate_report_path": str(gate_report_path),
                "summary_path": str(summary_path),
            }
            Path(lane["replay_source_path"]).parent.mkdir(parents=True)
            Path(lane["replay_source_path"]).write_text('{"row": 1}\n', encoding="utf-8")

            original_run_command = module.run_command
            module.run_command = lambda command, stage: None
            try:
                summary = module.execute_lane(lane, {"steps": []}, current_path=str(tmp_path / "current"))
            finally:
                module.run_command = original_run_command

        self.assertEqual(str(hard_state_report_path), summary["hard_state_report_path"])
        self.assertEqual(
            {
                "policy_top1_agreement": 0.82,
                "average_regret": 0.07,
                "value_calibration_mae": 0.11,
            },
            summary["hard_state_summary"],
        )

    def test_execute_lane_runs_local_promotion_gate_with_repo_python(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "versions" / "custom-run-iter4"
            candidate_path.mkdir(parents=True)
            (candidate_path / "benchmark_report.json").write_text(
                json.dumps(
                    {
                        "checks": [{"id": "runtime_parity", "passed": True}],
                        "arena_confidence_lower_bound": 0.56,
                    }
                ),
                encoding="utf-8",
            )
            gate_report_path = tmp_path / "lane" / "local_promotion_gate.json"
            gate_report_path.parent.mkdir(parents=True)
            gate_report_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "arena_score": 0.61,
                        "candidate_mcts_score": 0.57,
                        "current_mcts_score": 0.53,
                        "failure_reasons": [],
                    }
                ),
                encoding="utf-8",
            )
            lane = {
                "seed": 41,
                "candidate_path": str(candidate_path),
                "lane_root": str(gate_report_path.parent),
                "results_dir": str(tmp_path / "versions"),
                "replay_dir": str(tmp_path / "lane" / "replay"),
                "replay_source_path": str(tmp_path / "source" / "labeled_tactical_states.jsonl"),
                "config_path": str(tmp_path / "custom-run.json"),
                "gate_report_path": str(gate_report_path),
                "summary_path": str(tmp_path / "lane" / "seed_summary.json"),
            }
            Path(lane["replay_source_path"]).parent.mkdir(parents=True)
            Path(lane["replay_source_path"]).write_text('{"row": 1}\n', encoding="utf-8")

            commands: list[list[str]] = []

            def capture(command, *, stage):
                commands.append(command)

            original_run_command = module.run_command
            original_python_executable = module.python_executable
            module.run_command = capture
            module.python_executable = lambda: "/tmp/fake-repo-python"
            try:
                module.execute_lane(lane, {"steps": []}, current_path=str(tmp_path / "current"))
            finally:
                module.run_command = original_run_command
                module.python_executable = original_python_executable

        local_gate_command = next(command for command in commands if "script/ai/local_promotion_gate" in command)
        self.assertEqual("/tmp/fake-repo-python", local_gate_command[0])

    def test_execute_lane_stages_replay_dataset_before_pipeline(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "versions" / "custom-run-iter4"
            candidate_path.mkdir(parents=True)
            replay_source = tmp_path / "source" / "labeled_tactical_states.jsonl"
            replay_source.parent.mkdir(parents=True)
            replay_source.write_text('{"row": 1}\n', encoding="utf-8")
            (candidate_path / "benchmark_report.json").write_text(
                json.dumps(
                    {
                        "checks": [{"id": "runtime_parity", "passed": True}],
                        "arena_confidence_lower_bound": 0.56,
                    }
                ),
                encoding="utf-8",
            )
            gate_report_path = tmp_path / "lane" / "local_promotion_gate.json"
            gate_report_path.parent.mkdir(parents=True)
            gate_report_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "arena_score": 0.61,
                        "candidate_mcts_score": 0.57,
                        "current_mcts_score": 0.53,
                        "failure_reasons": [],
                    }
                ),
                encoding="utf-8",
            )
            replay_dir = tmp_path / "lane" / "replay"
            lane = {
                "seed": 41,
                "candidate_path": str(candidate_path),
                "lane_root": str(gate_report_path.parent),
                "results_dir": str(tmp_path / "versions"),
                "replay_dir": str(replay_dir),
                "replay_source_path": str(replay_source),
                "config_path": str(tmp_path / "custom-run.json"),
                "gate_report_path": str(gate_report_path),
                "summary_path": str(tmp_path / "lane" / "seed_summary.json"),
            }

            original_run_command = module.run_command
            module.run_command = lambda command, stage: None
            try:
                module.execute_lane(lane, {"steps": []}, current_path=str(tmp_path / "current"))
            finally:
                module.run_command = original_run_command

            staged_replay = replay_dir / "labeled_tactical_states.jsonl"
            self.assertTrue(staged_replay.exists())
            self.assertEqual(replay_source.read_text(encoding="utf-8"), staged_replay.read_text(encoding="utf-8"))

    def test_execute_lane_reads_benchmark_report_without_benchmark_contract_step(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "versions" / "custom-run-iter4"
            candidate_path.mkdir(parents=True)
            (candidate_path / "benchmark_report.json").write_text(
                json.dumps(
                    {
                        "checks": [{"id": "runtime_parity", "passed": True}],
                        "arena_confidence_lower_bound": 0.56,
                    }
                ),
                encoding="utf-8",
            )
            gate_report_path = tmp_path / "lane" / "local_promotion_gate.json"
            gate_report_path.parent.mkdir(parents=True)
            gate_report_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "arena_score": 0.61,
                        "candidate_mcts_score": 0.57,
                        "current_mcts_score": 0.53,
                        "failure_reasons": [],
                    }
                ),
                encoding="utf-8",
            )
            lane = {
                "seed": 41,
                "candidate_path": str(candidate_path),
                "lane_root": str(gate_report_path.parent),
                "results_dir": str(tmp_path / "versions"),
                "config_path": str(tmp_path / "custom-run.json"),
                "gate_report_path": str(gate_report_path),
                "summary_path": str(tmp_path / "lane" / "seed_summary.json"),
            }

            original_run_command = module.run_command
            module.run_command = lambda command, stage: None
            try:
                summary = module.execute_lane(lane, {"steps": []}, current_path=str(tmp_path / "current"))
            finally:
                module.run_command = original_run_command

        self.assertTrue(summary["benchmark_passed"])
        self.assertEqual(0.56, summary["arena_confidence_lower_bound"])

    def test_execute_lane_passes_runtime_current_path_as_hard_path(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "versions" / "custom-run-iter4"
            candidate_path.mkdir(parents=True)
            (candidate_path / "benchmark_report.json").write_text(
                json.dumps(
                    {
                        "checks": [{"id": "runtime_parity", "passed": True}],
                        "arena_confidence_lower_bound": 0.56,
                    }
                ),
                encoding="utf-8",
            )
            gate_report_path = tmp_path / "lane" / "local_promotion_gate.json"
            gate_report_path.parent.mkdir(parents=True)
            gate_report_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "arena_score": 0.61,
                        "candidate_mcts_score": 0.57,
                        "current_mcts_score": 0.53,
                        "failure_reasons": [],
                    }
                ),
                encoding="utf-8",
            )
            lane = {
                "seed": 41,
                "candidate_path": str(candidate_path),
                "lane_root": str(gate_report_path.parent),
                "results_dir": str(tmp_path / "versions"),
                "replay_dir": str(tmp_path / "lane" / "replay"),
                "replay_source_path": str(tmp_path / "source" / "labeled_tactical_states.jsonl"),
                "config_path": str(tmp_path / "custom-run.json"),
                "gate_report_path": str(gate_report_path),
                "summary_path": str(tmp_path / "lane" / "seed_summary.json"),
            }
            Path(lane["replay_source_path"]).parent.mkdir(parents=True)
            Path(lane["replay_source_path"]).write_text('{"row": 1}\n', encoding="utf-8")

            commands: list[list[str]] = []

            def capture(command, *, stage):
                commands.append(command)

            original_run_command = module.run_command
            original_python_executable = module.python_executable
            module.run_command = capture
            module.python_executable = lambda: "/tmp/fake-repo-python"
            try:
                module.execute_lane(
                    lane,
                    {"steps": []},
                    current_path=str(tmp_path / "runtime-current"),
                )
            finally:
                module.run_command = original_run_command
                module.python_executable = original_python_executable

        local_gate_command = next(command for command in commands if "script/ai/local_promotion_gate" in command)
        self.assertIn("--hard-path", local_gate_command)
        self.assertEqual(
            str(tmp_path / "runtime-current"),
            local_gate_command[local_gate_command.index("--hard-path") + 1],
        )

    def test_execute_lane_allows_tactical_lane_without_benchmark_report(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "versions" / "custom-run-iter4"
            candidate_path.mkdir(parents=True)
            (candidate_path / "arena_report.json").write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 120,
                        "wins": 72,
                        "losses": 24,
                        "draws": 24,
                        "score": 0.7,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            gate_report_path = tmp_path / "lane" / "local_promotion_gate.json"
            gate_report_path.parent.mkdir(parents=True)
            gate_report_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "arena_score": 0.61,
                        "candidate_mcts_score": 0.57,
                        "current_mcts_score": 0.53,
                        "failure_reasons": [],
                    }
                ),
                encoding="utf-8",
            )
            replay_source = tmp_path / "source" / "labeled_tactical_states.jsonl"
            replay_source.parent.mkdir(parents=True)
            replay_source.write_text('{"row": 1}\n', encoding="utf-8")
            lane = {
                "seed": 41,
                "candidate_path": str(candidate_path),
                "lane_root": str(gate_report_path.parent),
                "results_dir": str(tmp_path / "versions"),
                "replay_dir": str(tmp_path / "lane" / "replay"),
                "replay_source_path": str(replay_source),
                "config_path": str(tmp_path / "custom-run.json"),
                "gate_report_path": str(gate_report_path),
                "summary_path": str(tmp_path / "lane" / "seed_summary.json"),
            }

            original_run_command = module.run_command
            module.run_command = lambda command, stage: None
            try:
                summary = module.execute_lane(
                    lane,
                    {"steps": [{"name": "arena_confirm_report", "command": ["python", "arena.py"]}]},
                    current_path=str(tmp_path / "current"),
                )
            finally:
                module.run_command = original_run_command

        self.assertTrue(summary["benchmark_passed"])
        self.assertIsNone(summary["arena_confidence_lower_bound"])


if __name__ == "__main__":
    unittest.main()
