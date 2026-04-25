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
                "config_path": str(config_path),
                "gate_report_path": str(gate_report_path),
                "summary_path": str(summary_path),
            }

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
                "config_path": str(tmp_path / "custom-run.json"),
                "gate_report_path": str(gate_report_path),
                "summary_path": str(tmp_path / "lane" / "seed_summary.json"),
            }

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
                "config_path": str(tmp_path / "custom-run.json"),
                "gate_report_path": str(gate_report_path),
                "summary_path": str(tmp_path / "lane" / "seed_summary.json"),
            }

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


if __name__ == "__main__":
    unittest.main()
