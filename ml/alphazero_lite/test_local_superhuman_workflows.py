import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class PromoteSuperhumanCandidateUsageTest(unittest.TestCase):
    def test_requires_checkpoint_arg(self):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            ["script/ai/promote_superhuman_candidate"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            "Usage: script/ai/promote_superhuman_candidate CANDIDATE_DIR",
            result.stderr,
        )


class RunLocalSuperhumanRecoveryTest(unittest.TestCase):
    def test_dry_run_without_workers_flag_uses_shared_default_runtime_workers(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-local-recovery-") as tmp:
            output_root = Path(tmp) / "recovery"

            result = subprocess.run(
                [
                    "script/ai/run_local_superhuman_recovery",
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
            payload = json.loads(result.stdout)
            runtime_config = json.loads(
                Path(payload["runtime_config_path"]).read_text(encoding="utf-8")
            )
            self_play_command = next(
                step["command"]
                for step in runtime_config["steps"]
                if step.get("name") == "self_play"
            )
            worker_values = []
            for step in runtime_config["steps"]:
                command = step.get("command")
                if not isinstance(command, list) or "--workers" not in command:
                    continue
                worker_values.append(command[command.index("--workers") + 1])

            self.assertEqual(
                "high_memory_local", runtime_config.get("memory_speed_profile")
            )
            self.assertEqual(["24", "24", "24", "24", "24", "24", "24"], worker_values)
            self.assertEqual(
                "200000",
                self_play_command[
                    self_play_command.index("--evaluator-cache-size") + 1
                ],
            )

    def test_dry_run_workers_override_rewrites_runtime_config(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-local-recovery-") as tmp:
            output_root = Path(tmp) / "recovery"

            result = subprocess.run(
                [
                    "script/ai/run_local_superhuman_recovery",
                    "--dry-run",
                    "--output-root",
                    str(output_root),
                    "--workers",
                    "11",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            payload = json.loads(result.stdout)
            runtime_config = json.loads(
                Path(payload["runtime_config_path"]).read_text(encoding="utf-8")
            )
            self_play_command = next(
                step["command"]
                for step in runtime_config["steps"]
                if step.get("name") == "self_play"
            )
            worker_values = []
            for step in runtime_config["steps"]:
                command = step.get("command")
                if not isinstance(command, list) or "--workers" not in command:
                    continue
                worker_values.append(command[command.index("--workers") + 1])

            self.assertEqual(["11", "11", "11", "11", "11", "11", "11"], worker_values)
            self.assertEqual(
                "200000",
                self_play_command[
                    self_play_command.index("--evaluator-cache-size") + 1
                ],
            )


class LocalSuperhumanRecoveryProgressTest(unittest.TestCase):
    def run_progress_script(
        self, output_root: Path
    ) -> subprocess.CompletedProcess[str]:
        repo_root = Path(__file__).resolve().parents[2]
        return subprocess.run(
            [
                "script/ai/local_superhuman_recovery_progress",
                "--output-root",
                str(output_root),
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

    def write_runtime_config(self, output_root: Path) -> None:
        output_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": "aggressive-v3-superhuman",
            "start_iteration": 2,
            "iterations": 1,
            "steps": [],
        }
        (output_root / "pipeline_config.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_reports_self_play_when_only_runtime_config_exists(self):
        with tempfile.TemporaryDirectory(prefix="azlite-recovery-progress-") as tmp:
            output_root = Path(tmp) / "recovery"
            self.write_runtime_config(output_root)

            result = self.run_progress_script(output_root)

            self.assertEqual(0, result.returncode, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("self_play", payload["status"])

    def test_reports_train_when_self_play_and_bootstrap_outputs_exist(self):
        with tempfile.TemporaryDirectory(prefix="azlite-recovery-progress-") as tmp:
            output_root = Path(tmp) / "recovery"
            self.write_runtime_config(output_root)
            versions = output_root / "versions"
            (versions / "aggressive-v3-superhuman-iter2").mkdir(
                parents=True, exist_ok=True
            )
            (versions / "aggressive-v3-superhuman-iter1").mkdir(
                parents=True, exist_ok=True
            )
            (
                versions / "aggressive-v3-superhuman-iter2" / "self_play.jsonl"
            ).write_text("selfplay", encoding="utf-8")
            (
                versions / "aggressive-v3-superhuman-iter1" / "mcts_bootstrap.jsonl"
            ).write_text("bootstrap", encoding="utf-8")

            result = self.run_progress_script(output_root)

            self.assertEqual(0, result.returncode, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("train", payload["status"])

    def test_reports_completed_when_gate_report_exists(self):
        with tempfile.TemporaryDirectory(prefix="azlite-recovery-progress-") as tmp:
            output_root = Path(tmp) / "recovery"
            self.write_runtime_config(output_root)
            (output_root / "local_promotion_gate.json").write_text(
                json.dumps({"passed": False}), encoding="utf-8"
            )

            result = self.run_progress_script(output_root)

            self.assertEqual(0, result.returncode, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("completed", payload["status"])
