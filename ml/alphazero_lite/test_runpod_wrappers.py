import importlib.machinery
import io
import json
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from ml.alphazero_lite import runpod_experiment


class RunpodWrappersTest(unittest.TestCase):
    def test_ultra_wrapper_scripts_are_removed(self):
        repo_root = Path(__file__).resolve().parents[2]

        self.assertFalse((repo_root / "script/ai/runpod_ultra_experiment").exists())
        self.assertFalse((repo_root / "script/ai/promote_ultra_candidate").exists())

    def test_runpod_superhuman_experiment_uses_lossless_superhuman_gate(self):
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
        plan = json.loads(result.stdout)
        command = plan["command"]

        self.assertIn("--current-path model-artifact/current", command)
        self.assertIn("--arena-games 400", command)
        self.assertIn("--min-arena-games 400", command)
        self.assertIn("--min-arena-score 0.55", command)
        self.assertIn("--require-lossless", command)
        self.assertIn("--max-losses 0", command)
        self.assertIn("--skip-mcts-relative-check", command)

    def test_promote_superhuman_candidate_requires_checkpoint_arg(self):
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
            "Usage: script/ai/promote_superhuman_candidate CANDIDATE_DIR", result.stderr
        )


class RunpodTrainingExperimentValidationTest(unittest.TestCase):
    def write_downloaded_aggregate_summary(
        self, *, local_results_path, results_path, passed, lanes=None
    ):
        aggregate_summary_path = (
            Path(local_results_path)
            / Path(results_path).name
            / "aggregate_summary.json"
        )
        aggregate_summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"passed": passed}
        if lanes is not None:
            payload["lanes"] = lanes
        aggregate_summary_path.write_text(json.dumps(payload), encoding="utf-8")
        return aggregate_summary_path

    def run_model_robustness_wrapper(self, *, orchestrate_impl, cli_args):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "script/ai/runpod_model_robustness_confirmation"
        fake_runpod_experiment = types.ModuleType("ml.alphazero_lite.runpod_experiment")
        fake_runpod_experiment.build_dry_run_plan = lambda **kwargs: kwargs
        fake_runpod_experiment.orchestrate = orchestrate_impl
        stdout = io.StringIO()
        stderr = io.StringIO()
        normalized_cli_args = list(cli_args)

        with tempfile.TemporaryDirectory(
            prefix="runpod-default-parent-", dir="/tmp"
        ) as default_parent_artifact:
            if "--parent-artifact" not in normalized_cli_args:
                normalized_cli_args = [
                    "--parent-artifact",
                    default_parent_artifact,
                    *normalized_cli_args,
                ]

            with mock.patch.dict(
                sys.modules,
                {"ml.alphazero_lite.runpod_experiment": fake_runpod_experiment},
            ):
                with (
                    mock.patch.object(
                        sys, "argv", [str(script_path), *normalized_cli_args]
                    ),
                    mock.patch("sys.stdout", stdout),
                    mock.patch("sys.stderr", stderr),
                ):
                    try:
                        runpy.run_path(str(script_path), run_name="__main__")
                    except SystemExit as exc:
                        if isinstance(exc.code, int):
                            code = exc.code
                        else:
                            if exc.code not in (None, ""):
                                print(exc.code, file=sys.stderr)
                            code = 1
                    else:
                        code = 0

        return code, stdout.getvalue(), stderr.getvalue()

    def test_runpod_model_robustness_confirmation_dry_run_builds_remote_confirmation_plan(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                "script/ai/runpod_model_robustness_confirmation",
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        plan = json.loads(result.stdout)
        command = plan["command"]

        self.assertIn("script/ai/model_robustness_confirmation", command)
        self.assertIn(
            "--parent-artifact /tmp/azlite_v3_superhuman_versions/aggressive-v3-superhuman-iter1",
            command,
        )
        self.assertIn("--current-path model-artifact/current", command)
        self.assertIn("--output-root /tmp/azlite_v3_robustness_confirmation", command)
        self.assertIn(
            "cp -R /tmp/azlite_v3_robustness_confirmation/. storage/ai/alphazero_lite/versions/runpod-robustness-confirmation/",
            command,
        )
        self.assertEqual(
            [
                "script/ai/model_robustness_confirmation",
                "script/ai/local_promotion_gate",
                "ml/alphazero_lite",
                "model-artifact/current",
                "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json",
                "tmp/azlite_v3_superhuman_versions/aggressive-v3-superhuman-iter1",
            ],
            plan["include_paths"],
        )

    def test_runpod_model_robustness_confirmation_dry_run_wraps_robustness_run_and_copy_in_shell_block(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                "script/ai/runpod_model_robustness_confirmation",
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        plan = json.loads(result.stdout)
        command = plan["command"]

        self.assertIn(" && (", command)
        self.assertIn("if script/ai/model_robustness_confirmation", command)
        self.assertIn("then robustness_status=0;", command)
        self.assertIn("else robustness_status=$?;", command)
        self.assertIn(
            "fi; if cp -R /tmp/azlite_v3_robustness_confirmation/. storage/ai/alphazero_lite/versions/runpod-robustness-confirmation/;",
            command,
        )
        self.assertIn("then copy_status=0;", command)
        self.assertIn("else copy_status=$?;", command)
        self.assertIn('if [ "$copy_status" -ne 0 ]; then', command)
        self.assertIn('exit "$robustness_status"', command)
        self.assertTrue(command.rstrip().endswith(")"), command)

    def test_runpod_model_robustness_confirmation_dry_run_passes_base_config_override(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as temp_dir:
            base_config = Path(temp_dir) / "hard-state.json"
            base_config.write_text("{}", encoding="utf-8")

            result = subprocess.run(
                [
                    "script/ai/runpod_model_robustness_confirmation",
                    "--dry-run",
                    "--base-config",
                    str(base_config),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        plan = json.loads(result.stdout)
        command = plan["command"]

        self.assertIn(f"--base-config {base_config}", command)

    def test_runpod_model_robustness_confirmation_exits_zero_when_downloaded_aggregate_passed(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            local_results_path = tmp_path / "downloaded-results"
            results_path = (
                "storage/ai/alphazero_lite/versions/runpod-robustness-confirmation"
            )

            def fake_orchestrate(**kwargs):
                aggregate_summary_path = self.write_downloaded_aggregate_summary(
                    local_results_path=kwargs["local_results_path"],
                    results_path=kwargs["results_path"],
                    passed=True,
                )
                return {
                    "pod_id": "pod-123",
                    "bundle_path": "/tmp/bundle.tar.gz",
                    "shell_plan": {"delete_command": "runpodctl pod delete pod-123"},
                    "experiment_report_path": None,
                    "experiment_passed": None,
                    "manifest_path": None,
                    "manifest_status": None,
                    "_aggregate_summary_path": str(aggregate_summary_path),
                }

            code, stdout, stderr = self.run_model_robustness_wrapper(
                orchestrate_impl=fake_orchestrate,
                cli_args=[
                    "--local-results-path",
                    str(local_results_path),
                    "--results-path",
                    results_path,
                ],
            )

        self.assertEqual("", stderr)
        self.assertEqual(0, code)
        payload = json.loads(stdout)
        self.assertTrue(payload["robustness_passed"])
        self.assertTrue(
            payload["robustness_summary_path"].endswith("aggregate_summary.json")
        )

    def test_runpod_model_robustness_confirmation_preserves_downloaded_lane_summaries(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            local_results_path = tmp_path / "downloaded-results"
            results_path = (
                "storage/ai/alphazero_lite/versions/runpod-robustness-confirmation"
            )
            parent_artifact = tmp_path / "parent-artifact"
            parent_artifact.mkdir()
            lanes = [
                {
                    "seed": 41,
                    "hard_state_report_path": "/tmp/lane-41/hard_state_validation.json",
                    "hard_state_summary": {
                        "policy_top1_agreement": 0.82,
                        "average_regret": 0.07,
                        "value_calibration_mae": 0.11,
                    },
                }
            ]

            def fake_orchestrate(**kwargs):
                aggregate_summary_path = self.write_downloaded_aggregate_summary(
                    local_results_path=kwargs["local_results_path"],
                    results_path=kwargs["results_path"],
                    passed=True,
                    lanes=lanes,
                )
                return {
                    "pod_id": "pod-123",
                    "bundle_path": "/tmp/bundle.tar.gz",
                    "shell_plan": {"delete_command": "runpodctl pod delete pod-123"},
                    "experiment_report_path": None,
                    "experiment_passed": None,
                    "manifest_path": None,
                    "manifest_status": None,
                    "_aggregate_summary_path": str(aggregate_summary_path),
                }

            code, stdout, stderr = self.run_model_robustness_wrapper(
                orchestrate_impl=fake_orchestrate,
                cli_args=[
                    "--parent-artifact",
                    str(parent_artifact),
                    "--local-results-path",
                    str(local_results_path),
                    "--results-path",
                    results_path,
                ],
            )

        self.assertEqual("", stderr)
        self.assertEqual(0, code)
        payload = json.loads(stdout)
        self.assertEqual(lanes, payload["lanes"])

    def test_runpod_model_robustness_confirmation_exits_non_zero_when_downloaded_aggregate_failed(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            local_results_path = tmp_path / "downloaded-results"
            results_path = (
                "storage/ai/alphazero_lite/versions/runpod-robustness-confirmation"
            )

            def fake_orchestrate(**kwargs):
                aggregate_summary_path = self.write_downloaded_aggregate_summary(
                    local_results_path=kwargs["local_results_path"],
                    results_path=kwargs["results_path"],
                    passed=False,
                )
                return {
                    "pod_id": "pod-123",
                    "bundle_path": "/tmp/bundle.tar.gz",
                    "shell_plan": {"delete_command": "runpodctl pod delete pod-123"},
                    "experiment_report_path": None,
                    "experiment_passed": None,
                    "manifest_path": None,
                    "manifest_status": None,
                    "_aggregate_summary_path": str(aggregate_summary_path),
                }

            code, stdout, stderr = self.run_model_robustness_wrapper(
                orchestrate_impl=fake_orchestrate,
                cli_args=[
                    "--local-results-path",
                    str(local_results_path),
                    "--results-path",
                    results_path,
                ],
            )

        self.assertEqual("", stderr)
        self.assertNotEqual(0, code)
        payload = json.loads(stdout)
        self.assertFalse(payload["robustness_passed"])
        self.assertTrue(
            payload["robustness_summary_path"].endswith("aggregate_summary.json")
        )

    def test_runpod_model_robustness_confirmation_errors_when_downloaded_aggregate_is_missing(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            local_results_path = tmp_path / "downloaded-results"
            results_path = (
                "storage/ai/alphazero_lite/versions/runpod-robustness-confirmation"
            )

            code, stdout, stderr = self.run_model_robustness_wrapper(
                orchestrate_impl=lambda **kwargs: {
                    "pod_id": "pod-123",
                    "bundle_path": "/tmp/bundle.tar.gz",
                    "shell_plan": {"delete_command": "runpodctl pod delete pod-123"},
                    "experiment_report_path": None,
                    "experiment_passed": None,
                    "manifest_path": None,
                    "manifest_status": None,
                },
                cli_args=[
                    "--local-results-path",
                    str(local_results_path),
                    "--results-path",
                    results_path,
                ],
            )

        self.assertEqual("", stdout)
        self.assertNotEqual(0, code)
        self.assertIn("Missing downloaded robustness aggregate summary", stderr)

    def test_runpod_model_robustness_confirmation_rejects_non_bool_downloaded_passed_value(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            local_results_path = tmp_path / "downloaded-results"
            results_path = (
                "storage/ai/alphazero_lite/versions/runpod-robustness-confirmation"
            )

            def fake_orchestrate(**kwargs):
                aggregate_summary_path = (
                    Path(kwargs["local_results_path"])
                    / Path(kwargs["results_path"]).name
                    / "aggregate_summary.json"
                )
                aggregate_summary_path.parent.mkdir(parents=True, exist_ok=True)
                aggregate_summary_path.write_text(
                    json.dumps({"passed": "false"}), encoding="utf-8"
                )
                return {
                    "pod_id": "pod-123",
                    "bundle_path": "/tmp/bundle.tar.gz",
                    "shell_plan": {"delete_command": "runpodctl pod delete pod-123"},
                    "experiment_report_path": None,
                    "experiment_passed": None,
                    "manifest_path": None,
                    "manifest_status": None,
                }

            code, stdout, stderr = self.run_model_robustness_wrapper(
                orchestrate_impl=fake_orchestrate,
                cli_args=[
                    "--local-results-path",
                    str(local_results_path),
                    "--results-path",
                    results_path,
                ],
            )

        self.assertEqual("", stdout)
        self.assertNotEqual(0, code)
        self.assertIn("must contain boolean 'passed'", stderr)

    def test_runpod_model_robustness_confirmation_succeeds_when_passed_summary_exists_after_orchestrate_raises(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            local_results_path = tmp_path / "downloaded-results"
            results_path = (
                "storage/ai/alphazero_lite/versions/runpod-robustness-confirmation"
            )

            def fake_orchestrate(**kwargs):
                self.write_downloaded_aggregate_summary(
                    local_results_path=kwargs["local_results_path"],
                    results_path=kwargs["results_path"],
                    passed=True,
                )
                raise RuntimeError("remote robustness command failed after download")

            code, stdout, stderr = self.run_model_robustness_wrapper(
                orchestrate_impl=fake_orchestrate,
                cli_args=[
                    "--local-results-path",
                    str(local_results_path),
                    "--results-path",
                    results_path,
                ],
            )

        self.assertEqual("", stderr)
        self.assertEqual(0, code)
        payload = json.loads(stdout)
        self.assertTrue(payload["robustness_passed"])
        self.assertTrue(
            payload["robustness_summary_path"].endswith("aggregate_summary.json")
        )
        self.assertIsNone(payload["pod_id"])

    def test_runpod_model_robustness_confirmation_uses_downloaded_aggregate_when_orchestrate_raises(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            local_results_path = tmp_path / "downloaded-results"
            results_path = (
                "storage/ai/alphazero_lite/versions/runpod-robustness-confirmation"
            )

            def fake_orchestrate(**kwargs):
                self.write_downloaded_aggregate_summary(
                    local_results_path=kwargs["local_results_path"],
                    results_path=kwargs["results_path"],
                    passed=False,
                )
                raise RuntimeError("remote robustness command failed after download")

            code, stdout, stderr = self.run_model_robustness_wrapper(
                orchestrate_impl=fake_orchestrate,
                cli_args=[
                    "--local-results-path",
                    str(local_results_path),
                    "--results-path",
                    results_path,
                ],
            )

        self.assertEqual("", stderr)
        self.assertEqual(1, code)
        payload = json.loads(stdout)
        self.assertFalse(payload["robustness_passed"])
        self.assertTrue(
            payload["robustness_summary_path"].endswith("aggregate_summary.json")
        )
        self.assertIsNone(payload["pod_id"])

    def test_runpod_model_robustness_confirmation_surfaces_orchestrate_error_when_no_summary_exists(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            local_results_path = tmp_path / "downloaded-results"
            results_path = (
                "storage/ai/alphazero_lite/versions/runpod-robustness-confirmation"
            )

            def fake_orchestrate(**kwargs):
                raise RuntimeError("remote robustness command failed before download")

            with self.assertRaisesRegex(
                RuntimeError, "remote robustness command failed before download"
            ):
                self.run_model_robustness_wrapper(
                    orchestrate_impl=fake_orchestrate,
                    cli_args=[
                        "--local-results-path",
                        str(local_results_path),
                        "--results-path",
                        results_path,
                    ],
                )

    def test_runpod_model_robustness_confirmation_clears_stale_local_summary_before_orchestrate(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            local_results_path = tmp_path / "downloaded-results"
            results_path = (
                "storage/ai/alphazero_lite/versions/runpod-robustness-confirmation"
            )
            stale_summary_path = (
                local_results_path / Path(results_path).name / "aggregate_summary.json"
            )
            stale_summary_path.parent.mkdir(parents=True)
            stale_summary_path.write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )

            def fake_orchestrate(**kwargs):
                raise RuntimeError("remote robustness command failed before download")

            with self.assertRaisesRegex(
                RuntimeError, "remote robustness command failed before download"
            ):
                self.run_model_robustness_wrapper(
                    orchestrate_impl=fake_orchestrate,
                    cli_args=[
                        "--local-results-path",
                        str(local_results_path),
                        "--results-path",
                        results_path,
                    ],
                )

        self.assertFalse(stale_summary_path.exists())

    def test_runpod_model_robustness_confirmation_dry_run_syncs_absolute_tmp_current_path(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                "script/ai/runpod_model_robustness_confirmation",
                "--current-path",
                "/tmp/runpod-current-model",
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        plan = json.loads(result.stdout)
        command = plan["command"]

        self.assertIn("mkdir -p /tmp", command)
        self.assertIn("rm -rf /tmp/runpod-current-model", command)
        self.assertIn(
            "cp -R tmp/runpod-current-model /tmp/runpod-current-model", command
        )
        self.assertIn("--current-path /tmp/runpod-current-model", command)
        self.assertEqual(
            [
                "script/ai/model_robustness_confirmation",
                "script/ai/local_promotion_gate",
                "ml/alphazero_lite",
                "model-artifact/current",
                "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json",
                "tmp/azlite_v3_superhuman_versions/aggressive-v3-superhuman-iter1",
                "tmp/runpod-current-model",
            ],
            plan["include_paths"],
        )

    def test_runpod_model_robustness_confirmation_dry_run_syncs_absolute_tmp_parent_artifact(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                "script/ai/runpod_model_robustness_confirmation",
                "--parent-artifact",
                "/tmp/runpod-parent-artifact",
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        plan = json.loads(result.stdout)
        command = plan["command"]

        self.assertIn("mkdir -p /tmp", command)
        self.assertIn("rm -rf /tmp/runpod-parent-artifact", command)
        self.assertIn(
            "cp -R tmp/runpod-parent-artifact /tmp/runpod-parent-artifact", command
        )
        self.assertIn("--parent-artifact /tmp/runpod-parent-artifact", command)
        self.assertEqual(
            [
                "script/ai/model_robustness_confirmation",
                "script/ai/local_promotion_gate",
                "ml/alphazero_lite",
                "model-artifact/current",
                "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json",
                "tmp/runpod-parent-artifact",
            ],
            plan["include_paths"],
        )

    def test_runpod_model_robustness_confirmation_dry_run_accepts_repo_absolute_parent_and_current_paths(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[2]
        temp_root = repo_root / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix="runpod-repo-absolute-artifact-", dir=temp_root
        ) as tmp:
            repo_artifact = Path(tmp)
            staged_artifact = str(
                repo_root / "tmp/runpod-staged" / repo_artifact.relative_to(Path("/"))
            )
            staged_include_path = (
                Path("tmp/runpod-staged") / repo_artifact.relative_to(Path("/"))
            ).as_posix()

            result = subprocess.run(
                [
                    "script/ai/runpod_model_robustness_confirmation",
                    "--dry-run",
                    "--parent-artifact",
                    str(repo_artifact),
                    "--current-path",
                    str(repo_artifact),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            plan = json.loads(result.stdout)
            command = plan["command"]

            self.assertIn(f"--parent-artifact {staged_artifact}", command)
            self.assertIn(f"--current-path {staged_artifact}", command)
            self.assertIn(staged_include_path, plan["include_paths"])

    def test_runpod_model_robustness_confirmation_stages_absolute_tmp_parent_artifact_before_orchestrate(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(
            prefix="runpod-parent-artifact-", dir="/tmp"
        ) as tmp:
            parent_artifact = Path(tmp) / "artifact"
            parent_artifact.mkdir()
            (parent_artifact / "model.npz").write_text("stub", encoding="utf-8")
            local_results_path = Path(tmp) / "downloaded-results"
            results_path = (
                "storage/ai/alphazero_lite/versions/runpod-robustness-confirmation"
            )

            def fake_orchestrate(**kwargs):
                include_paths = kwargs["include_paths"]
                self.assertIn("tmp/runpod-parent-artifact-", " ".join(include_paths))
                for include_path in include_paths:
                    self.assertTrue((repo_root / include_path).exists(), include_path)
                self.write_downloaded_aggregate_summary(
                    local_results_path=kwargs["local_results_path"],
                    results_path=kwargs["results_path"],
                    passed=True,
                )
                return {
                    "pod_id": "pod-123",
                    "bundle_path": "/tmp/bundle.tar.gz",
                    "shell_plan": {"delete_command": "runpodctl pod delete pod-123"},
                    "experiment_report_path": None,
                    "experiment_passed": None,
                    "manifest_path": None,
                    "manifest_status": None,
                }

            code, stdout, stderr = self.run_model_robustness_wrapper(
                orchestrate_impl=fake_orchestrate,
                cli_args=[
                    "--parent-artifact",
                    str(parent_artifact),
                    "--local-results-path",
                    str(local_results_path),
                    "--results-path",
                    results_path,
                ],
            )

        self.assertEqual("", stderr)
        self.assertEqual(0, code)
        payload = json.loads(stdout)
        self.assertTrue(payload["robustness_passed"])

    def test_model_robustness_confirmation_dry_run_lists_five_seed_lanes(self):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                "script/ai/model_robustness_confirmation",
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        plan = json.loads(result.stdout)

        self.assertEqual(
            "/tmp/azlite_v3_superhuman_versions/aggressive-v3-superhuman-iter1",
            plan["parent_artifact"],
        )
        self.assertEqual(5, len(plan["lanes"]))
        self.assertEqual([41, 42, 43, 44, 45], [lane["seed"] for lane in plan["lanes"]])
        self.assertTrue(
            plan["aggregate_summary_path"].endswith("aggregate_summary.json")
        )
        self.assertTrue(
            all(lane["config_path"].endswith(".json") for lane in plan["lanes"])
        )
        self.assertTrue(
            all(lane["results_dir"].endswith("/versions") for lane in plan["lanes"])
        )

    def test_model_robustness_confirmation_dry_run_uses_seed_specific_run_ids_and_sweeps(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                "script/ai/model_robustness_confirmation",
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        plan = json.loads(result.stdout)

        self.assertEqual(
            [
                "aggressive-v3-superhuman-robust-s41",
                "aggressive-v3-superhuman-robust-s42",
                "aggressive-v3-superhuman-robust-s43",
                "aggressive-v3-superhuman-robust-s44",
                "aggressive-v3-superhuman-robust-s45",
            ],
            [lane["run_id"] for lane in plan["lanes"]],
        )
        self.assertEqual(
            [
                "41,141,241",
                "42,142,242",
                "43,143,243",
                "44,144,244",
                "45,145,245",
            ],
            [lane["seed_sweep"] for lane in plan["lanes"]],
        )

    def test_model_robustness_confirmation_writes_decision_ready_seed_summary_fields(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "robustness"
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

    def test_model_robustness_confirmation_dry_run_uses_base_config_override(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "override.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_id": "custom-run",
                        "seed": 42,
                        "iterations": 1,
                        "start_iteration": 2,
                        "versions_dir": "/tmp/custom_versions",
                        "current_path": "/tmp/custom_current",
                        "steps": [],
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "script/ai/model_robustness_confirmation",
                    "--dry-run",
                    "--base-config",
                    str(config_path),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        plan = json.loads(result.stdout)

        self.assertEqual(str(config_path), plan["base_config"])
        self.assertEqual("custom-run-robust-s41", plan["lanes"][0]["run_id"])

    def test_model_robustness_confirmation_rejects_missing_parent_artifact_directory(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as current_dir:
            result = subprocess.run(
                [
                    "script/ai/model_robustness_confirmation",
                    "--parent-artifact",
                    "/definitely/missing-parent-artifact",
                    "--current-path",
                    current_dir,
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("parent artifact does not exist", result.stderr)

    def test_model_robustness_confirmation_rejects_current_path_that_is_not_directory(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as parent_dir:
            with tempfile.NamedTemporaryFile() as current_file:
                result = subprocess.run(
                    [
                        "script/ai/model_robustness_confirmation",
                        "--parent-artifact",
                        parent_dir,
                        "--current-path",
                        current_file.name,
                    ],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("current path must be a directory", result.stderr)

    def test_model_robustness_confirmation_rejects_output_root_that_overlaps_parent_artifact(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as shared_dir:
            with tempfile.TemporaryDirectory() as current_dir:
                result = subprocess.run(
                    [
                        "script/ai/model_robustness_confirmation",
                        "--parent-artifact",
                        shared_dir,
                        "--current-path",
                        current_dir,
                        "--output-root",
                        shared_dir,
                    ],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("output root must not overlap parent artifact", result.stderr)

    def test_model_robustness_confirmation_rejects_symlink_output_root_into_current_path(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as parent_dir:
            with tempfile.TemporaryDirectory() as current_dir:
                with tempfile.TemporaryDirectory() as temp_dir:
                    symlink_path = Path(temp_dir) / "linked-output"
                    symlink_path.symlink_to(current_dir, target_is_directory=True)

                    result = subprocess.run(
                        [
                            "script/ai/model_robustness_confirmation",
                            "--parent-artifact",
                            parent_dir,
                            "--current-path",
                            current_dir,
                            "--output-root",
                            str(symlink_path),
                        ],
                        cwd=repo_root,
                        capture_output=True,
                        text=True,
                        check=False,
                    )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("output root must not overlap current path", result.stderr)

    def test_model_robustness_confirmation_execute_lane_reads_benchmark_report_from_candidate_path(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "script/ai/model_robustness_confirmation"
        loader = importlib.machinery.SourceFileLoader(
            "model_robustness_confirmation", str(script_path)
        )
        module = types.ModuleType(loader.name)
        module.__file__ = str(script_path)
        loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "versions" / "custom-run-iter4"
            candidate_path.mkdir(parents=True)
            benchmark_report_path = candidate_path / "benchmark_report.json"
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
                summary = module.execute_lane(
                    lane, {"steps": []}, current_path=str(tmp_path / "current")
                )
            finally:
                module.run_command = original_run_command

        self.assertTrue(summary["benchmark_passed"])
        self.assertTrue(summary["local_promotion_gate_passed"])
        self.assertEqual(0.56, summary["arena_confidence_lower_bound"])
        self.assertEqual(str(candidate_path), summary["candidate_path"])

    def test_model_robustness_confirmation_execute_lane_keeps_gate_failure_report(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "script/ai/model_robustness_confirmation"
        loader = importlib.machinery.SourceFileLoader(
            "model_robustness_confirmation", str(script_path)
        )
        module = types.ModuleType(loader.name)
        module.__file__ = str(script_path)
        loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "versions" / "custom-run-iter2"
            candidate_path.mkdir(parents=True)
            benchmark_report_path = candidate_path / "benchmark_report.json"
            gate_report_path = tmp_path / "lane" / "local_promotion_gate.json"
            summary_path = tmp_path / "lane" / "seed_summary.json"
            config_path = tmp_path / "custom-run.json"

            benchmark_report_path.write_text(
                json.dumps(
                    {
                        "checks": [{"id": "runtime_parity", "passed": True}],
                        "arena_confidence_lower_bound": 0.51,
                    }
                ),
                encoding="utf-8",
            )
            gate_report_path.parent.mkdir(parents=True)

            lane = {
                "seed": 41,
                "candidate_path": str(candidate_path),
                "lane_root": str(gate_report_path.parent),
                "results_dir": str(tmp_path / "versions"),
                "config_path": str(config_path),
                "gate_report_path": str(gate_report_path),
                "summary_path": str(summary_path),
            }

            def fake_run_command(command, stage):
                if "local promotion gate" in stage:
                    gate_report_path.write_text(
                        json.dumps(
                            {
                                "passed": False,
                                "arena_score": 0.49,
                                "candidate_mcts_score": 0.47,
                                "current_mcts_score": 0.53,
                                "failure_reasons": [
                                    {"code": "arena_score_below_threshold"}
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
                    raise SystemExit("lane 41 local promotion gate failed")

            original_run_command = module.run_command
            module.run_command = fake_run_command
            try:
                summary = module.execute_lane(
                    lane, {"steps": []}, current_path=str(tmp_path / "current")
                )
                written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            finally:
                module.run_command = original_run_command

        self.assertTrue(summary["benchmark_passed"])
        self.assertFalse(summary["local_promotion_gate_passed"])
        self.assertEqual(
            [{"code": "arena_score_below_threshold"}], summary["failure_reasons"]
        )
        self.assertEqual(summary, written_summary)

    def test_model_robustness_confirmation_continues_after_manifest_backed_pipeline_failure(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "script/ai/model_robustness_confirmation"
        loader = importlib.machinery.SourceFileLoader(
            "model_robustness_confirmation", str(script_path)
        )
        module = types.ModuleType(loader.name)
        module.__file__ = str(script_path)
        loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            parent_artifact = tmp_path / "parent"
            current_path = tmp_path / "current"
            output_root = tmp_path / "output"
            parent_artifact.mkdir()
            current_path.mkdir()

            lane_41 = {
                "seed": 41,
                "run_id": "custom-run-robust-s41",
                "seed_sweep": "41,141,241",
                "lane_root": str(output_root / "custom-run-robust-s41"),
                "results_dir": str(output_root / "custom-run-robust-s41" / "versions"),
                "config_path": str(output_root / "custom-run-robust-s41.json"),
                "candidate_path": str(
                    output_root
                    / "custom-run-robust-s41"
                    / "versions"
                    / "custom-run-robust-s41-iter2"
                ),
                "gate_report_path": str(
                    output_root / "custom-run-robust-s41" / "local_promotion_gate.json"
                ),
                "summary_path": str(
                    output_root / "custom-run-robust-s41" / "seed_summary.json"
                ),
            }
            lane_42 = {
                "seed": 42,
                "run_id": "custom-run-robust-s42",
                "seed_sweep": "42,142,242",
                "lane_root": str(output_root / "custom-run-robust-s42"),
                "results_dir": str(output_root / "custom-run-robust-s42" / "versions"),
                "config_path": str(output_root / "custom-run-robust-s42.json"),
                "candidate_path": str(
                    output_root
                    / "custom-run-robust-s42"
                    / "versions"
                    / "custom-run-robust-s42-iter2"
                ),
                "gate_report_path": str(
                    output_root / "custom-run-robust-s42" / "local_promotion_gate.json"
                ),
                "summary_path": str(
                    output_root / "custom-run-robust-s42" / "seed_summary.json"
                ),
            }
            aggregate_summary_path = output_root / "aggregate_summary.json"
            plan = {
                "base_config": str(tmp_path / "base.json"),
                "parent_artifact": str(parent_artifact),
                "current_path": str(current_path),
                "aggregate_summary_path": str(aggregate_summary_path),
                "lanes": [lane_41, lane_42],
            }
            base_config = {
                "run_id": "custom-run",
                "seed": 41,
                "iterations": 1,
                "start_iteration": 2,
                "versions_dir": str(tmp_path / "unused-versions"),
                "current_path": str(parent_artifact),
                "steps": [],
            }
            calls = []

            def fake_run_command(command, stage):
                calls.append(stage)
                if stage == "lane 41 pipeline":
                    candidate_path = Path(lane_41["candidate_path"])
                    candidate_path.mkdir(parents=True)
                    (candidate_path / "run_manifest.json").write_text(
                        json.dumps(
                            {
                                "schema": "azlite_run_manifest_v1",
                                "run_id": lane_41["run_id"],
                                "iteration": 2,
                                "seed": 41,
                                "config_path": lane_41["config_path"],
                                "status": "failed",
                                "steps": [
                                    {
                                        "name": "arena_prefilter_validate",
                                        "status": "failed",
                                    }
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
                    raise SystemExit("lane 41 pipeline failed")
                if stage == "lane 42 pipeline":
                    candidate_path = Path(lane_42["candidate_path"])
                    candidate_path.mkdir(parents=True)
                    (candidate_path / "benchmark_report.json").write_text(
                        json.dumps(
                            {
                                "checks": [{"id": "runtime_parity", "passed": True}],
                                "arena_confidence_lower_bound": 0.58,
                            }
                        ),
                        encoding="utf-8",
                    )
                    return
                if stage == "lane 42 local promotion gate":
                    Path(lane_42["gate_report_path"]).write_text(
                        json.dumps(
                            {
                                "passed": True,
                                "arena_score": 0.62,
                                "candidate_mcts_score": 0.58,
                                "current_mcts_score": 0.52,
                                "failure_reasons": [],
                            }
                        ),
                        encoding="utf-8",
                    )
                    return
                self.fail(f"unexpected stage: {stage}")

            stdout = io.StringIO()
            args = Namespace(
                base_config=plan["base_config"],
                parent_artifact=str(parent_artifact),
                current_path=str(current_path),
                output_root=str(output_root),
                dry_run=False,
            )
            plan["required_qualifying_seed_count"] = 2

            with (
                mock.patch.object(module, "parse_args", return_value=args),
                mock.patch.object(
                    module, "build_plan", return_value=(plan, base_config)
                ),
                mock.patch.object(module, "run_command", side_effect=fake_run_command),
                mock.patch("sys.stdout", stdout),
            ):
                exit_code = module.main()

            lane_41_summary = json.loads(
                Path(lane_41["summary_path"]).read_text(encoding="utf-8")
            )
            lane_42_summary = json.loads(
                Path(lane_42["summary_path"]).read_text(encoding="utf-8")
            )
            aggregate_summary = json.loads(
                aggregate_summary_path.read_text(encoding="utf-8")
            )

        self.assertEqual(1, exit_code)
        self.assertEqual(
            ["lane 41 pipeline", "lane 42 pipeline", "lane 42 local promotion gate"],
            calls,
        )
        self.assertFalse(lane_41_summary["benchmark_passed"])
        self.assertIsNone(lane_41_summary["local_promotion_gate_passed"])
        self.assertEqual("pipeline", lane_41_summary["failure_source"])
        self.assertEqual(
            [{"name": "arena_prefilter_validate", "status": "failed"}],
            lane_41_summary["failure_reasons"],
        )
        self.assertTrue(lane_42_summary["benchmark_passed"])
        self.assertTrue(lane_42_summary["local_promotion_gate_passed"])
        self.assertFalse(aggregate_summary["passed"])
        self.assertEqual(
            [41, 42], [summary["seed"] for summary in aggregate_summary["lanes"]]
        )
        self.assertEqual(aggregate_summary, json.loads(stdout.getvalue()))

    def test_model_robustness_confirmation_rejects_stale_manifest_on_pipeline_failure(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "script/ai/model_robustness_confirmation"
        loader = importlib.machinery.SourceFileLoader(
            "model_robustness_confirmation", str(script_path)
        )
        module = types.ModuleType(loader.name)
        module.__file__ = str(script_path)
        loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "versions" / "custom-run-robust-s41-iter2"
            gate_report_path = tmp_path / "lane" / "local_promotion_gate.json"
            summary_path = tmp_path / "lane" / "seed_summary.json"
            config_path = tmp_path / "custom-run-robust-s41.json"
            lane = {
                "seed": 41,
                "run_id": "custom-run-robust-s41",
                "candidate_path": str(candidate_path),
                "lane_root": str(gate_report_path.parent),
                "results_dir": str(tmp_path / "versions"),
                "config_path": str(config_path),
                "gate_report_path": str(gate_report_path),
                "summary_path": str(summary_path),
            }

            def fake_run_command(command, stage):
                if stage == "lane 41 pipeline":
                    candidate_path.mkdir(parents=True)
                    (candidate_path / "run_manifest.json").write_text(
                        json.dumps(
                            {
                                "schema": "azlite_run_manifest_v1",
                                "run_id": "other-run",
                                "iteration": 2,
                                "seed": 41,
                                "config_path": lane["config_path"],
                                "status": "failed",
                                "gate_failures": [
                                    {"code": "arena_score_below_threshold"}
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
                    raise SystemExit("lane 41 pipeline failed")

            original_run_command = module.run_command
            module.run_command = fake_run_command
            try:
                with self.assertRaises(SystemExit):
                    module.execute_lane(
                        lane, {"steps": []}, current_path=str(tmp_path / "current")
                    )
            finally:
                module.run_command = original_run_command

        self.assertFalse(summary_path.exists())

    def test_rejects_negative_promotion_max_losses(self):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                "script/ai/runpod_training_experiment",
                "--config-path",
                "ml/alphazero_lite/configs/aggressive_v3_clone_extend_phase1.json",
                "--promotion-max-losses",
                "-1",
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("must be a non-negative integer", result.stderr)

    def test_runpod_training_experiment_dry_run_bundles_full_app_models_for_regression_gate(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                "script/ai/runpod_training_experiment",
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
        plan = json.loads(result.stdout)

        self.assertIn("app/models", plan["include_paths"])


class RunpodStrongerBootstrapMoreDataWrapperTest(unittest.TestCase):
    def write_downloaded_result_tree(
        self,
        *,
        local_results_path,
        results_path,
        candidate_dir_name="aggressive-v3-stronger-bootstrap-more-data-local-iter1",
        gate_report=None,
        arena_report=None,
        candidate_mcts_report=None,
        current_mcts_report=None,
        include_regression=True,
        include_forensic=True,
    ):
        downloaded_root = Path(local_results_path) / Path(results_path).name
        candidate_dir = downloaded_root / candidate_dir_name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "weights.json").write_text("{}", encoding="utf-8")
        (candidate_dir / "metadata.json").write_text("{}", encoding="utf-8")
        (candidate_dir / "arena_report.json").write_text("{}", encoding="utf-8")
        (candidate_dir / "run_manifest.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")

        default_gate_report = {
            "passed": True,
            "candidate_path": f"/tmp/versions/{candidate_dir_name}",
            "arena_score": 0.61,
            "candidate_mcts_score": 0.56,
            "current_mcts_score": 0.53,
            "failure_reasons": [],
        }
        default_arena_report = {"wins": 70, "draws": 10, "losses": 40, "games_played": 120}
        default_candidate_mcts_report = {"az_wins": 24, "draws": 4, "losses": 12, "games": 40}
        default_current_mcts_report = {"az_wins": 20, "draws": 4, "losses": 16, "games": 40}

        (downloaded_root / "local_promotion_gate.json").write_text(
            json.dumps(gate_report or default_gate_report),
            encoding="utf-8",
        )
        (downloaded_root / "candidate_vs_current_arena.json").write_text(
            json.dumps(arena_report or default_arena_report),
            encoding="utf-8",
        )
        (downloaded_root / "candidate_vs_mcts1200.json").write_text(
            json.dumps(candidate_mcts_report or default_candidate_mcts_report),
            encoding="utf-8",
        )
        (downloaded_root / "current_vs_mcts1200.json").write_text(
            json.dumps(current_mcts_report or default_current_mcts_report),
            encoding="utf-8",
        )
        if include_regression:
            (downloaded_root / "candidate_regression_suite.json").write_text(
                json.dumps({"passed": True}),
                encoding="utf-8",
            )
        if include_forensic:
            (downloaded_root / "candidate_forensic_suite.json").write_text(
                json.dumps({"schema": "azlite_forensic_suite_v1"}),
                encoding="utf-8",
            )
        return downloaded_root

    def run_wrapper(self, *, orchestrate_impl, cli_args):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "script/ai/runpod_stronger_bootstrap_more_data_experiment"
        fake_runpod_experiment = types.ModuleType("ml.alphazero_lite.runpod_experiment")
        fake_runpod_experiment.build_dry_run_plan = lambda **kwargs: kwargs
        fake_runpod_experiment.orchestrate = orchestrate_impl
        stdout = io.StringIO()
        stderr = io.StringIO()

        with mock.patch.dict(sys.modules, {"ml.alphazero_lite.runpod_experiment": fake_runpod_experiment}):
            with mock.patch.object(sys, "argv", [str(script_path), *cli_args]), mock.patch(
                "sys.stdout", stdout
            ), mock.patch("sys.stderr", stderr):
                try:
                    runpy.run_path(str(script_path), run_name="__main__")
                except SystemExit as exc:
                    if isinstance(exc.code, int):
                        code = exc.code
                    else:
                        if exc.code not in (None, ""):
                            print(exc.code, file=sys.stderr)
                        code = 1
                else:
                    code = 0

        return code, stdout.getvalue(), stderr.getvalue()

    def test_dry_run_uses_lane_defaults(self):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                "script/ai/runpod_stronger_bootstrap_more_data_experiment",
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(
            "storage/ai/alphazero_lite/versions/runpod-stronger-bootstrap-more-data",
            plan["results_path"],
        )
        self.assertEqual(
            "ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_more_data_local.json",
            plan["config_path"],
        )

    def test_dry_run_excludes_missing_optional_bundle_paths(self):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                "script/ai/runpod_stronger_bootstrap_more_data_experiment",
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        plan = json.loads(result.stdout)
        self.assertNotIn("Gemfile", plan["include_paths"])
        self.assertNotIn("Gemfile.lock", plan["include_paths"])
        self.assertNotIn("bin/rails", plan["include_paths"])
        self.assertNotIn("app/models", plan["include_paths"])
        self.assertNotIn("config", plan["include_paths"])

    def test_dry_run_bundles_config_current_path_when_distinct_from_gate_current_path(self):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                "script/ai/runpod_stronger_bootstrap_more_data_experiment",
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        plan = json.loads(result.stdout)
        self.assertIn("tmp/runpod-staged/storage/ai/alphazero_lite/current", plan["include_paths"])

    def test_dry_run_syncs_model_artifact_into_config_current_path_before_pipeline(self):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                "script/ai/runpod_stronger_bootstrap_more_data_experiment",
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        plan = json.loads(result.stdout)
        self.assertIn(
            "cp -R model-artifact/current/. storage/ai/alphazero_lite/current/",
            plan["command"],
        )

    def test_rejects_missing_config_path_before_launch(self):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                "script/ai/runpod_stronger_bootstrap_more_data_experiment",
                "--config-path",
                "ml/alphazero_lite/configs/does_not_exist.json",
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("Missing config path", result.stderr)

    def test_completed_run_recommends_confirm_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            local_results_path = tmp_path / "downloaded-results"
            results_path = "storage/ai/alphazero_lite/versions/runpod-stronger-bootstrap-more-data"

            def fake_orchestrate(**kwargs):
                self.write_downloaded_result_tree(
                    local_results_path=kwargs["local_results_path"],
                    results_path=kwargs["results_path"],
                )
                return {
                    "pod_id": "pod-123",
                    "bundle_path": "/tmp/bundle.tar.gz",
                    "shell_plan": {"delete_command": "runpodctl pod delete pod-123"},
                    "experiment_report_path": str(local_results_path / Path(results_path).name / "local_promotion_gate.json"),
                    "experiment_passed": True,
                    "manifest_path": None,
                    "manifest_status": None,
                }

            code, stdout, stderr = self.run_wrapper(
                orchestrate_impl=fake_orchestrate,
                cli_args=["--local-results-path", str(local_results_path), "--results-path", results_path],
            )

            summary = json.loads(stdout)
            written_summary = json.loads((local_results_path / Path(results_path).name / "issue1_summary.json").read_text(encoding="utf-8"))

        self.assertEqual("", stderr)
        self.assertEqual(0, code)
        self.assertEqual("confirm", summary["recommendation"])
        self.assertEqual(summary, written_summary)
        self.assertTrue(summary["local_promotion_gate_path"].endswith("local_promotion_gate.json"))
        self.assertTrue(summary["candidate_path"].endswith("aggressive-v3-stronger-bootstrap-more-data-local-iter1"))

    def test_missing_regression_report_keeps_summary_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            local_results_path = tmp_path / "downloaded-results"
            results_path = "storage/ai/alphazero_lite/versions/runpod-stronger-bootstrap-more-data"

            def fake_orchestrate(**kwargs):
                self.write_downloaded_result_tree(
                    local_results_path=kwargs["local_results_path"],
                    results_path=kwargs["results_path"],
                    include_regression=False,
                )
                return {
                    "pod_id": "pod-123",
                    "bundle_path": "/tmp/bundle.tar.gz",
                    "shell_plan": {"delete_command": "runpodctl pod delete pod-123"},
                    "experiment_report_path": str(local_results_path / Path(results_path).name / "local_promotion_gate.json"),
                    "experiment_passed": True,
                    "manifest_path": None,
                    "manifest_status": None,
                }

            code, stdout, stderr = self.run_wrapper(
                orchestrate_impl=fake_orchestrate,
                cli_args=["--local-results-path", str(local_results_path), "--results-path", results_path],
            )

            summary = json.loads(stdout)

        self.assertEqual("", stderr)
        self.assertEqual(1, code)
        self.assertFalse(summary["completed"])
        self.assertIsNone(summary["recommendation"])
        self.assertIsNone(summary["candidate_regression_suite_path"])

    def test_completed_run_recommends_pivot_when_mcts1200_regresses(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            local_results_path = tmp_path / "downloaded-results"
            results_path = "storage/ai/alphazero_lite/versions/runpod-stronger-bootstrap-more-data"

            def fake_orchestrate(**kwargs):
                self.write_downloaded_result_tree(
                    local_results_path=kwargs["local_results_path"],
                    results_path=kwargs["results_path"],
                    gate_report={
                        "passed": False,
                        "candidate_path": "/tmp/versions/aggressive-v3-stronger-bootstrap-more-data-local-iter1",
                        "arena_score": 0.58,
                        "candidate_mcts_score": 0.47,
                        "current_mcts_score": 0.53,
                        "failure_reasons": [{"code": "candidate_mcts_below_current"}],
                    },
                    arena_report={"wins": 68, "draws": 6, "losses": 46, "games_played": 120},
                    candidate_mcts_report={"az_wins": 16, "draws": 6, "losses": 18, "games": 40},
                    current_mcts_report={"az_wins": 20, "draws": 4, "losses": 16, "games": 40},
                )
                return {
                    "pod_id": "pod-123",
                    "bundle_path": "/tmp/bundle.tar.gz",
                    "shell_plan": {"delete_command": "runpodctl pod delete pod-123"},
                    "experiment_report_path": str(local_results_path / Path(results_path).name / "local_promotion_gate.json"),
                    "experiment_passed": False,
                    "manifest_path": None,
                    "manifest_status": None,
                }

            code, stdout, stderr = self.run_wrapper(
                orchestrate_impl=fake_orchestrate,
                cli_args=["--local-results-path", str(local_results_path), "--results-path", results_path],
            )

            summary = json.loads(stdout)

        self.assertEqual("", stderr)
        self.assertEqual(1, code)
        self.assertEqual("pivot", summary["recommendation"])

    def test_completed_run_recommends_reject_when_candidate_loses_to_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            local_results_path = tmp_path / "downloaded-results"
            results_path = "storage/ai/alphazero_lite/versions/runpod-stronger-bootstrap-more-data"

            def fake_orchestrate(**kwargs):
                self.write_downloaded_result_tree(
                    local_results_path=kwargs["local_results_path"],
                    results_path=kwargs["results_path"],
                    gate_report={
                        "passed": False,
                        "candidate_path": "/tmp/versions/aggressive-v3-stronger-bootstrap-more-data-local-iter1",
                        "arena_score": 0.42,
                        "candidate_mcts_score": 0.46,
                        "current_mcts_score": 0.53,
                        "failure_reasons": [{"code": "arena_score_below_threshold"}],
                    },
                    arena_report={"wins": 40, "draws": 20, "losses": 60, "games_played": 120},
                    candidate_mcts_report={"az_wins": 16, "draws": 5, "losses": 19, "games": 40},
                    current_mcts_report={"az_wins": 20, "draws": 4, "losses": 16, "games": 40},
                    include_forensic=False,
                )
                return {
                    "pod_id": "pod-123",
                    "bundle_path": "/tmp/bundle.tar.gz",
                    "shell_plan": {"delete_command": "runpodctl pod delete pod-123"},
                    "experiment_report_path": str(local_results_path / Path(results_path).name / "local_promotion_gate.json"),
                    "experiment_passed": False,
                    "manifest_path": None,
                    "manifest_status": None,
                }

            code, stdout, stderr = self.run_wrapper(
                orchestrate_impl=fake_orchestrate,
                cli_args=["--local-results-path", str(local_results_path), "--results-path", results_path],
            )

            summary = json.loads(stdout)

        self.assertEqual("", stderr)
        self.assertEqual(1, code)
        self.assertEqual("reject", summary["recommendation"])
        self.assertTrue(summary["candidate_regression_suite_path"].endswith("candidate_regression_suite.json"))
        self.assertIsNone(summary["candidate_forensic_suite_path"])


class RunpodExperimentCleanupTest(unittest.TestCase):
    def test_inspect_downloaded_results_marks_partial_artifacts_present_when_only_remote_log_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            local_results_path = tmp_path / "downloaded-results"
            results_path = "storage/ai/alphazero_lite/versions/runpod-stronger-bootstrap-more-data"
            downloaded_results_dir = local_results_path / Path(results_path).name
            remote_log_path = downloaded_results_dir / "remote_run.log"

            remote_log_path.parent.mkdir(parents=True, exist_ok=True)
            remote_log_path.write_text("exit_status=1\n", encoding="utf-8")

            result = runpod_experiment._inspect_downloaded_results(str(local_results_path), results_path)

        self.assertFalse(result["execution_completed"])
        self.assertTrue(result["downloaded_results_present"])
        self.assertIsNone(result["experiment_report_path"])

    def test_build_pod_request_supports_cpu3c_2_4_profile(self):
        pod_request = runpod_experiment.build_pod_request(
            name="azlite-runpod-stronger-bootstrap-more-data",
            include_paths=["ml/alphazero_lite"],
            pod_profile="cpu3c-2-4",
        )

        self.assertEqual(["cpu3c"], pod_request["cpuFlavorIds"])
        self.assertEqual(2, pod_request["vcpuCount"])

    def test_build_bundle_does_not_require_ruby_version_when_ruby_paths_are_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = str(Path(tmp) / "bundle.tar.gz")

            result_path = runpod_experiment.build_bundle(
                bundle_path=bundle_path,
                command="script/ai/runpod_stronger_bootstrap_more_data_experiment",
                results_path="storage/ai/alphazero_lite/versions/runpod-stronger-bootstrap-more-data",
                include_paths=[
                    "ml/alphazero_lite",
                    "script/ai",
                    "ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_more_data_local.json",
                    "model-artifact/current",
                    "test/fixtures/ai/superhuman_regression_positions.json",
                ],
            )

            self.assertEqual(bundle_path, result_path)
            self.assertTrue(Path(bundle_path).exists())

    def test_inspect_downloaded_results_falls_back_to_manifest_when_aggregate_summary_is_invalid(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            local_results_path = tmp_path / "downloaded-results"
            results_path = (
                "storage/ai/alphazero_lite/versions/runpod-robustness-confirmation"
            )
            downloaded_results_dir = local_results_path / Path(results_path).name
            aggregate_summary_path = downloaded_results_dir / "aggregate_summary.json"
            manifest_path = downloaded_results_dir / "lane-41" / "run_manifest.json"

            aggregate_summary_path.parent.mkdir(parents=True, exist_ok=True)
            aggregate_summary_path.write_text("{\n", encoding="utf-8")
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps({"status": "failed"}), encoding="utf-8")

            result = runpod_experiment._inspect_downloaded_results(
                str(local_results_path), results_path
            )

        self.assertFalse(result["execution_completed"])
        self.assertIsNone(result["experiment_report_path"])
        self.assertIsNone(result["experiment_passed"])
        self.assertEqual(str(manifest_path), result["manifest_path"])
        self.assertEqual("failed", result["manifest_status"])

    def test_orchestrate_deletes_pod_when_downloaded_robustness_summary_exists_after_remote_failure(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle_path = tmp_path / "bundle.tar.gz"
            local_results_path = tmp_path / "downloaded-results"
            results_path = (
                "storage/ai/alphazero_lite/versions/runpod-robustness-confirmation"
            )
            downloaded_results_dir = local_results_path / Path(results_path).name
            aggregate_summary_path = downloaded_results_dir / "aggregate_summary.json"

            def fake_bundle_builder(**kwargs):
                bundle_path.write_text("bundle", encoding="utf-8")
                return str(bundle_path)

            def fake_timeout_runner(seconds, fn):
                return fn()

            executed_commands = []

            def fake_command_runner(command):
                executed_commands.append(command)
                if "https://rest.runpod.io/v1/pods" in command:
                    return json.dumps({"id": "pod-123"})
                if command == "runpodctl ssh info pod-123":
                    return json.dumps(
                        {
                            "host": "example.com",
                            "port": 22,
                            "user": "root",
                            "keyPath": "/tmp/fake-key",
                        }
                    )
                if "runpod_remote_run.sh" in command:
                    raise RuntimeError("remote robustness command failed")
                if command.startswith("scp ") and results_path in command:
                    aggregate_summary_path.parent.mkdir(parents=True, exist_ok=True)
                    aggregate_summary_path.write_text(
                        json.dumps({"passed": False}), encoding="utf-8"
                    )
                    return ""
                if command == "runpodctl pod delete pod-123":
                    return ""
                return ""

            result = runpod_experiment.orchestrate(
                name="robustness-confirmation",
                command="script/ai/model_robustness_confirmation",
                bundle_path=str(bundle_path),
                results_path=results_path,
                local_results_path=str(local_results_path),
                include_paths=["ml/alphazero_lite"],
                api_key_env="RUNPOD_API_KEY",
                keep_pod_on_failure=True,
                bundle_builder=fake_bundle_builder,
                command_runner=fake_command_runner,
                timeout_runner=fake_timeout_runner,
                sleeper=lambda _: None,
            )

        self.assertEqual("pod-123", result["pod_id"])
        self.assertIsNone(result["experiment_report_path"])
        self.assertIs(False, result["experiment_passed"])
        self.assertIsNone(result["manifest_path"])
        self.assertIn("runpodctl pod delete pod-123", executed_commands)

    def test_orchestrate_preserves_pod_when_only_debug_log_exists_after_remote_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle_path = tmp_path / "bundle.tar.gz"
            local_results_path = tmp_path / "downloaded-results"
            results_path = "storage/ai/alphazero_lite/versions/runpod-stronger-bootstrap-more-data"
            downloaded_results_dir = local_results_path / Path(results_path).name
            remote_log_path = downloaded_results_dir / "remote_run.log"

            def fake_bundle_builder(**kwargs):
                bundle_path.write_text("bundle", encoding="utf-8")
                return str(bundle_path)

            def fake_timeout_runner(seconds, fn):
                return fn()

            executed_commands = []

            def fake_command_runner(command):
                executed_commands.append(command)
                if "https://rest.runpod.io/v1/pods" in command:
                    return json.dumps({"id": "pod-123"})
                if command == "runpodctl ssh info pod-123":
                    return json.dumps({
                        "host": "example.com",
                        "port": 22,
                        "user": "root",
                        "keyPath": "/tmp/fake-key",
                    })
                if "runpod_remote_run.sh" in command:
                    raise RuntimeError("remote command failed")
                if command.startswith("scp ") and results_path in command:
                    remote_log_path.parent.mkdir(parents=True, exist_ok=True)
                    remote_log_path.write_text("exit_status=1\n", encoding="utf-8")
                    return ""
                if command == "runpodctl pod delete pod-123":
                    return ""
                return ""

            with self.assertRaises(RuntimeError):
                runpod_experiment.orchestrate(
                    name="stronger-bootstrap",
                    command="script/ai/runpod_stronger_bootstrap_more_data_experiment",
                    bundle_path=str(bundle_path),
                    results_path=results_path,
                    local_results_path=str(local_results_path),
                    include_paths=["ml/alphazero_lite"],
                    api_key_env="RUNPOD_API_KEY",
                    keep_pod_on_failure=True,
                    bundle_builder=fake_bundle_builder,
                    command_runner=fake_command_runner,
                    timeout_runner=fake_timeout_runner,
                    sleeper=lambda _: None,
                )

        self.assertNotIn("runpodctl pod delete pod-123", executed_commands)

    def test_orchestrate_deletes_pod_when_manifest_exists_after_remote_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle_path = tmp_path / "bundle.tar.gz"
            local_results_path = tmp_path / "downloaded-results"
            results_path = "storage/ai/alphazero_lite/versions/runpod-stronger-bootstrap-more-data"
            downloaded_results_dir = local_results_path / Path(results_path).name / "candidate-dir"
            manifest_path = downloaded_results_dir / "run_manifest.json"

            def fake_bundle_builder(**kwargs):
                bundle_path.write_text("bundle", encoding="utf-8")
                return str(bundle_path)

            def fake_timeout_runner(seconds, fn):
                return fn()

            executed_commands = []

            def fake_command_runner(command):
                executed_commands.append(command)
                if "https://rest.runpod.io/v1/pods" in command:
                    return json.dumps({"id": "pod-123"})
                if command == "runpodctl ssh info pod-123":
                    return json.dumps({
                        "host": "example.com",
                        "port": 22,
                        "user": "root",
                        "keyPath": "/tmp/fake-key",
                    })
                if "runpod_remote_run.sh" in command:
                    raise RuntimeError("remote command failed")
                if command.startswith("scp ") and results_path in command:
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    manifest_path.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
                    return ""
                if command == "runpodctl pod delete pod-123":
                    return ""
                return ""

            with self.assertRaises(RuntimeError):
                runpod_experiment.orchestrate(
                    name="stronger-bootstrap",
                    command="script/ai/runpod_stronger_bootstrap_more_data_experiment",
                    bundle_path=str(bundle_path),
                    results_path=results_path,
                    local_results_path=str(local_results_path),
                    include_paths=["ml/alphazero_lite"],
                    api_key_env="RUNPOD_API_KEY",
                    keep_pod_on_failure=True,
                    bundle_builder=fake_bundle_builder,
                    command_runner=fake_command_runner,
                    timeout_runner=fake_timeout_runner,
                    sleeper=lambda _: None,
                )

        self.assertIn("runpodctl pod delete pod-123", executed_commands)


if __name__ == "__main__":
    unittest.main()
