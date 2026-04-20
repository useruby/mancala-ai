import json
import importlib.util
from importlib.machinery import SourceFileLoader
import subprocess
import tempfile
import unittest
from pathlib import Path


class RunpodTrainingExperimentContractTest(unittest.TestCase):
    def test_superhuman_dry_run_bundles_regression_gate_dependencies(self):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                "script/ai/runpod_superhuman_experiment",
                "--config-path",
                "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json",
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        payload = json.loads(result.stdout)
        include_paths = payload["include_paths"]

        self.assertIn("app/models", include_paths)
        self.assertIn("config", include_paths)
        self.assertIn("Gemfile", include_paths)
        self.assertIn("Gemfile.lock", include_paths)
        self.assertIn("bin/rails", include_paths)
        self.assertIn("test/fixtures/ai/superhuman_regression_positions.json", include_paths)


class PromoteRunpodCandidateScriptTest(unittest.TestCase):
    @staticmethod
    def load_promote_module():
        repo_root = Path(__file__).resolve().parents[2]
        module_path = repo_root / "script/ai/promote_runpod_candidate"
        loader = SourceFileLoader("promote_runpod_candidate", str(module_path))
        spec = importlib.util.spec_from_loader("promote_runpod_candidate", loader)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        repo_root = Path(__file__).resolve().parents[2]
        return subprocess.run(
            ["script/ai/promote_runpod_candidate", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

    def write_candidate_artifact(self, candidate_dir: Path, *, run_id: str = "candidate-run", iteration: int = 2) -> dict:
        candidate_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "iteration": iteration,
        }
        weights = {"w_input": [[0.125]]}
        arena_report = {
            "schema": "arena_v1",
            "games_played": 400,
            "wins": 220,
            "losses": 0,
            "draws": 180,
            "promotion_decision": {"passed": True},
        }
        run_manifest = {
            "run_id": run_id,
            "iteration": iteration,
            "status": "completed",
        }
        (candidate_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (candidate_dir / "weights.json").write_text(json.dumps(weights), encoding="utf-8")
        (candidate_dir / "arena_report.json").write_text(json.dumps(arena_report), encoding="utf-8")
        (candidate_dir / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
        return {
            "metadata": metadata,
            "weights": weights,
            "arena_report": arena_report,
            "run_manifest": run_manifest,
        }

    def write_root_reports(self, downloaded_root: Path) -> None:
        downloaded_root.mkdir(parents=True, exist_ok=True)
        for name in (
            "candidate_vs_current_arena.json",
            "candidate_vs_mcts1200.json",
            "current_vs_mcts1200.json",
            "candidate_regression_suite.json",
        ):
            (downloaded_root / name).write_text("{}", encoding="utf-8")

    def write_gate_report(self, path: Path, *, candidate_path: str, passed: bool = True) -> dict:
        payload = {
            "schema": "azlite_local_promotion_gate_v1",
            "candidate_path": candidate_path,
            "passed": passed,
            "failure_reasons": [],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def test_rejects_missing_required_root_level_report(self):
        with tempfile.TemporaryDirectory(prefix="azlite-runpod-promote-") as tmp:
            tmp_path = Path(tmp)
            downloaded_root = tmp_path / "downloaded"
            candidate_dir = downloaded_root / "candidate-run-iter2"
            target_dir = tmp_path / "current"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "metadata.json").write_text("live-metadata", encoding="utf-8")

            self.write_candidate_artifact(candidate_dir)
            self.write_root_reports(downloaded_root)

            result = self.run_script(
                str(downloaded_root),
                "--target",
                str(target_dir),
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("Missing required file", result.stderr)
            self.assertIn("local_promotion_gate.json", result.stderr)
            self.assertEqual("live-metadata", (target_dir / "metadata.json").read_text(encoding="utf-8"))

    def test_rejects_required_root_level_report_directory(self):
        with tempfile.TemporaryDirectory(prefix="azlite-runpod-promote-") as tmp:
            tmp_path = Path(tmp)
            downloaded_root = tmp_path / "downloaded"
            candidate_dir = downloaded_root / "candidate-run-iter2"

            self.write_candidate_artifact(candidate_dir)
            self.write_root_reports(downloaded_root)
            (downloaded_root / "local_promotion_gate.json").mkdir(parents=True, exist_ok=True)

            result = self.run_script(str(downloaded_root))

            self.assertNotEqual(0, result.returncode)
            self.assertIn("Required path is not a file", result.stderr)
            self.assertIn("local_promotion_gate.json", result.stderr)

    def test_rejects_failed_gate_report(self):
        with tempfile.TemporaryDirectory(prefix="azlite-runpod-promote-") as tmp:
            tmp_path = Path(tmp)
            downloaded_root = tmp_path / "downloaded"
            candidate_dir = downloaded_root / "candidate-run-iter2"

            self.write_candidate_artifact(candidate_dir)
            self.write_root_reports(downloaded_root)
            self.write_gate_report(
                downloaded_root / "local_promotion_gate.json",
                candidate_path=str(candidate_dir),
                passed=False,
            )

            result = self.run_script(str(downloaded_root))

            self.assertNotEqual(0, result.returncode)
            self.assertIn("Gate report did not pass", result.stderr)

    def test_rejects_gate_report_candidate_path_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="azlite-runpod-promote-") as tmp:
            tmp_path = Path(tmp)
            downloaded_root = tmp_path / "downloaded"
            candidate_dir = downloaded_root / "candidate-run-iter2"

            self.write_candidate_artifact(candidate_dir)
            self.write_root_reports(downloaded_root)
            self.write_gate_report(
                downloaded_root / "local_promotion_gate.json",
                candidate_path=str(downloaded_root / "other-candidate"),
                passed=True,
            )

            result = self.run_script(str(downloaded_root))

            self.assertNotEqual(0, result.returncode)
            self.assertIn("candidate-path consistency", result.stderr)

    def test_rejects_multiple_candidate_artifact_directories(self):
        with tempfile.TemporaryDirectory(prefix="azlite-runpod-promote-") as tmp:
            tmp_path = Path(tmp)
            downloaded_root = tmp_path / "downloaded"
            self.write_candidate_artifact(downloaded_root / "candidate-run-iter1")
            self.write_candidate_artifact(downloaded_root / "candidate-run-iter2")
            self.write_root_reports(downloaded_root)
            self.write_gate_report(
                downloaded_root / "local_promotion_gate.json",
                candidate_path=str(downloaded_root / "candidate-run-iter2"),
                passed=True,
            )

            result = self.run_script(str(downloaded_root))

            self.assertNotEqual(0, result.returncode)
            self.assertIn("exactly one candidate artifact directory", result.stderr)

    def test_rejects_invalid_gate_report_json(self):
        with tempfile.TemporaryDirectory(prefix="azlite-runpod-promote-") as tmp:
            tmp_path = Path(tmp)
            downloaded_root = tmp_path / "downloaded"
            candidate_dir = downloaded_root / "candidate-run-iter2"

            self.write_candidate_artifact(candidate_dir)
            self.write_root_reports(downloaded_root)
            gate_path = downloaded_root / "local_promotion_gate.json"
            gate_path.parent.mkdir(parents=True, exist_ok=True)
            gate_path.write_text("[]", encoding="utf-8")

            result = self.run_script(str(downloaded_root))

            self.assertNotEqual(0, result.returncode)
            self.assertIn("JSON payload must be an object", result.stderr)

    def test_accepts_gate_report_with_matching_remote_candidate_dir_name(self):
        with tempfile.TemporaryDirectory(prefix="azlite-runpod-promote-") as tmp:
            tmp_path = Path(tmp)
            downloaded_root = tmp_path / "downloaded"
            candidate_dir = downloaded_root / "candidate-run-iter2"
            target_dir = tmp_path / "current"

            payload = self.write_candidate_artifact(candidate_dir)
            self.write_root_reports(downloaded_root)
            self.write_gate_report(
                downloaded_root / "local_promotion_gate.json",
                candidate_path="/tmp/azlite_v3_superhuman_versions/candidate-run-iter2",
                passed=True,
            )

            result = self.run_script(
                str(downloaded_root),
                "--target",
                str(target_dir),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertEqual(payload["metadata"], json.loads((target_dir / "metadata.json").read_text(encoding="utf-8")))

    def test_promotes_candidate_artifact_and_root_reports(self):
        with tempfile.TemporaryDirectory(prefix="azlite-runpod-promote-") as tmp:
            tmp_path = Path(tmp)
            downloaded_root = tmp_path / "downloaded"
            candidate_dir = downloaded_root / "candidate-run-iter2"
            target_dir = tmp_path / "current"

            payload = self.write_candidate_artifact(candidate_dir)
            self.write_root_reports(downloaded_root)
            gate_report = self.write_gate_report(
                downloaded_root / "local_promotion_gate.json",
                candidate_path=str(candidate_dir),
                passed=True,
            )

            result = self.run_script(
                str(downloaded_root),
                "--target",
                str(target_dir),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertEqual(payload["metadata"], json.loads((target_dir / "metadata.json").read_text(encoding="utf-8")))
            self.assertEqual(payload["weights"], json.loads((target_dir / "weights.json").read_text(encoding="utf-8")))
            self.assertEqual(payload["arena_report"], json.loads((target_dir / "arena_report.json").read_text(encoding="utf-8")))

            stdout_payload = json.loads(result.stdout)
            self.assertEqual(str(candidate_dir), stdout_payload["candidate_path"])
            self.assertEqual(str(downloaded_root / "local_promotion_gate.json"), stdout_payload["gate_report_path"])
            self.assertEqual(gate_report["candidate_path"], stdout_payload["candidate_path"])

    def test_rejects_target_when_existing_path_is_file(self):
        with tempfile.TemporaryDirectory(prefix="azlite-runpod-promote-") as tmp:
            tmp_path = Path(tmp)
            downloaded_root = tmp_path / "downloaded"
            candidate_dir = downloaded_root / "candidate-run-iter2"
            target_path = tmp_path / "current"

            self.write_candidate_artifact(candidate_dir)
            self.write_root_reports(downloaded_root)
            self.write_gate_report(
                downloaded_root / "local_promotion_gate.json",
                candidate_path=str(candidate_dir),
                passed=True,
            )
            target_path.write_text("not-a-directory", encoding="utf-8")

            result = self.run_script(
                str(downloaded_root),
                "--target",
                str(target_path),
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("target path is not a directory", result.stderr)

    def test_preserves_existing_target_when_promotion_swap_fails(self):
        with tempfile.TemporaryDirectory(prefix="azlite-runpod-promote-") as tmp:
            tmp_path = Path(tmp)
            downloaded_root = tmp_path / "downloaded"
            candidate_dir = downloaded_root / "candidate-run-iter2"
            target_dir = tmp_path / "current"

            self.write_candidate_artifact(candidate_dir)
            self.write_root_reports(downloaded_root)
            self.write_gate_report(
                downloaded_root / "local_promotion_gate.json",
                candidate_path=str(candidate_dir),
                passed=True,
            )
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "metadata.json").write_text("live-metadata", encoding="utf-8")
            (target_dir / "weights.json").write_text("live-weights", encoding="utf-8")
            (target_dir / "arena_report.json").write_text("live-arena", encoding="utf-8")

            promote_module = self.load_promote_module()
            original_rename = Path.rename

            def failing_rename(path_obj, destination):
                if Path(path_obj).name.endswith(".staging"):
                    raise OSError("simulated rename failure")
                return original_rename(path_obj, destination)

            from unittest import mock

            with mock.patch("pathlib.Path.rename", new=failing_rename):
                with self.assertRaises(OSError):
                    promote_module.promote(candidate_dir, target_dir)

            self.assertEqual("live-metadata", (target_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual("live-weights", (target_dir / "weights.json").read_text(encoding="utf-8"))
            self.assertEqual("live-arena", (target_dir / "arena_report.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
