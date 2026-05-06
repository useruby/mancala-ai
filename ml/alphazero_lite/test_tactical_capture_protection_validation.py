import importlib.machinery
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


class TacticalCaptureProtectionValidationScriptTest(unittest.TestCase):
    def load_module(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "script/ai/run_tactical_capture_protection_validation"
        loader = importlib.machinery.SourceFileLoader(
            "run_tactical_capture_protection_validation",
            str(script_path),
        )
        module = types.ModuleType(loader.name)
        module.__file__ = str(script_path)
        loader.exec_module(module)
        return module

    def test_dry_run_pins_run_id_and_baseline_sources(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-tactical-capture-protection-validation-") as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "capture-protection-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                    "--dry-run",
                ]
            )

            payload = module.build_dry_run_payload(args, repo_root)

        run_paths = module.build_run_paths(output_root, "capture-protection-demo")
        pinned_sources = payload["pinned_sources"]
        self.assertEqual("capture-protection-demo", payload["run_id"])
        self.assertEqual(["select", "validation_pack", "write_verdict"], payload["stages"])
        self.assertEqual(
            str(run_paths["final_dir"] / "baseline_candidate_forensics.json"),
            pinned_sources["baseline-forensics"],
        )
        self.assertEqual(
            str(run_paths["final_dir"] / "arena_seed_1041.json"),
            pinned_sources["baseline-arena"],
        )
        self.assertEqual(
            str(run_paths["selection_artifact"]),
            pinned_sources["selected-artifact"],
        )
        self.assertEqual(
            str(repo_root / "ml/alphazero_lite/configs/aggressive_v3_tactical_replay_local.json"),
            pinned_sources["base-config"],
        )
        self.assertEqual(
            str(repo_root / "ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"),
            pinned_sources["forensic-suite"],
        )
        self.assertEqual(
            str(run_paths["wrapper_summary_path"]),
            payload["summary_path"],
        )

        select_command = payload["commands"]["select"]
        validation_command = payload["commands"]["validation_pack"]
        verdict_command = payload["commands"]["write_verdict"]

        self.assertTrue(select_command[1].endswith("script/ai/run_local_tactical_replay_experiment"))
        self.assertNotIn("--start-stage", select_command)
        self.assertNotIn("--selected-artifact", select_command)
        self.assertIn("--start-stage", validation_command)
        self.assertEqual("validation_pack", validation_command[validation_command.index("--start-stage") + 1])
        self.assertIn("--selected-artifact", validation_command)
        self.assertEqual(
            str(run_paths["selection_artifact"]),
            validation_command[validation_command.index("--selected-artifact") + 1],
        )
        self.assertEqual(["-m", "ml.alphazero_lite.write_tactical_validation_verdict"], verdict_command[1:3])
        self.assertEqual(
            str(run_paths["final_dir"] / "baseline_candidate_forensics.json"),
            verdict_command[verdict_command.index("--baseline-forensics") + 1],
        )
        self.assertEqual(
            str(run_paths["final_dir"] / "arena_seed_1041.json"),
            verdict_command[verdict_command.index("--arena-report") + 1],
        )

    def test_run_executes_launcher_then_writes_verdict(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-tactical-capture-protection-validation-") as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "capture-protection-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                ]
            )
            run_paths = module.build_run_paths(output_root, args.run_id)
            commands_run: list[tuple[list[str], tuple[int, ...]]] = []

            def fake_run_command(command, cwd, allowed_returncodes=(0,)):
                commands_run.append((list(command), tuple(allowed_returncodes)))

                if command[1].endswith("script/ai/run_local_tactical_replay_experiment"):
                    if "--start-stage" not in command:
                        run_paths["selection_artifact"].mkdir(parents=True, exist_ok=True)
                        (run_paths["selection_artifact"] / "model.npz").write_text("stub", encoding="utf-8")
                    else:
                        run_paths["final_dir"].mkdir(parents=True, exist_ok=True)
                        for artifact_name in (
                            "baseline_candidate_forensics.json",
                            "selected_candidate_forensics.json",
                            "bucket_gate.json",
                            "candidate_regression_suite.json",
                            "arena_seed_1041.json",
                        ):
                            (run_paths["final_dir"] / artifact_name).write_text("{}", encoding="utf-8")

                if command[1:3] == ["-m", "ml.alphazero_lite.write_tactical_validation_verdict"]:
                    verdict_path = Path(command[command.index("--out") + 1])
                    verdict_path.parent.mkdir(parents=True, exist_ok=True)
                    verdict_path.write_text(
                        json.dumps(
                            {
                                "schema": "azlite_tactical_validation_verdict_v1",
                                "passed": True,
                                "verdict": "pass",
                                "failure_reasons": [],
                            }
                        ),
                        encoding="utf-8",
                    )

                return {"command": list(command), "cwd": str(cwd), "returncode": 0}

            with mock.patch.object(module, "run_command", side_effect=fake_run_command):
                summary = module.run(args, repo_root)

            self.assertTrue(run_paths["wrapper_summary_path"].exists())
            self.assertTrue(run_paths["verdict_path"].exists())
            self.assertEqual(summary, json.loads(run_paths["wrapper_summary_path"].read_text(encoding="utf-8")))

        self.assertEqual(["select", "validation_pack", "write_verdict"], summary["executed_stages"])
        self.assertEqual(3, len(commands_run))
        self.assertTrue(commands_run[0][0][1].endswith("script/ai/run_local_tactical_replay_experiment"))
        self.assertNotIn("--start-stage", commands_run[0][0])
        self.assertTrue(commands_run[1][0][1].endswith("script/ai/run_local_tactical_replay_experiment"))
        self.assertEqual((0,), commands_run[1][1])
        self.assertEqual("validation_pack", commands_run[1][0][commands_run[1][0].index("--start-stage") + 1])
        self.assertEqual(
            ["-m", "ml.alphazero_lite.write_tactical_validation_verdict"],
            commands_run[2][0][1:3],
        )
        self.assertEqual((0, 1), commands_run[2][1])
        self.assertEqual(str(run_paths["selection_artifact"]), summary["selected_artifact"])
        self.assertTrue(summary["verdict"]["passed"])

    def test_run_rejects_nonzero_verdict_exit_without_fresh_verdict_artifact(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-tactical-capture-protection-validation-") as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "capture-protection-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                ]
            )
            run_paths = module.build_run_paths(output_root, args.run_id)
            run_paths["verdict_path"].parent.mkdir(parents=True, exist_ok=True)
            run_paths["verdict_path"].write_text(
                json.dumps(
                    {
                        "schema": "azlite_tactical_validation_verdict_v1",
                        "passed": True,
                        "verdict": "pass",
                        "failure_reasons": [],
                    }
                ),
                encoding="utf-8",
            )

            def fake_run_command(command, cwd, allowed_returncodes=(0,)):
                if command[1].endswith("script/ai/run_local_tactical_replay_experiment"):
                    if "--start-stage" not in command:
                        run_paths["selection_artifact"].mkdir(parents=True, exist_ok=True)
                        (run_paths["selection_artifact"] / "model.npz").write_text("stub", encoding="utf-8")
                    else:
                        run_paths["final_dir"].mkdir(parents=True, exist_ok=True)
                        for artifact_name in (
                            "baseline_candidate_forensics.json",
                            "selected_candidate_forensics.json",
                            "bucket_gate.json",
                            "candidate_regression_suite.json",
                            "arena_seed_1041.json",
                        ):
                            (run_paths["final_dir"] / artifact_name).write_text("{}", encoding="utf-8")

                if command[1:3] == ["-m", "ml.alphazero_lite.write_tactical_validation_verdict"]:
                    if run_paths["verdict_path"].exists():
                        run_paths["verdict_path"].unlink()
                    return {"command": list(command), "cwd": str(cwd), "returncode": 1}

                return {"command": list(command), "cwd": str(cwd), "returncode": 0}

            with mock.patch.object(module, "run_command", side_effect=fake_run_command):
                with self.assertRaisesRegex(SystemExit, "fresh verdict"):
                    module.run(args, repo_root)

    def test_run_preserves_prior_verdict_if_earlier_stage_fails(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-tactical-capture-protection-validation-") as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "capture-protection-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                ]
            )
            run_paths = module.build_run_paths(output_root, args.run_id)
            prior_verdict = {
                "schema": "azlite_tactical_validation_verdict_v1",
                "passed": False,
                "verdict": "fail",
                "failure_reasons": ["prior_verdict"],
            }
            run_paths["verdict_path"].parent.mkdir(parents=True, exist_ok=True)
            run_paths["verdict_path"].write_text(json.dumps(prior_verdict), encoding="utf-8")

            def fake_run_command(command, cwd, allowed_returncodes=(0,)):
                if command[1].endswith("script/ai/run_local_tactical_replay_experiment") and "--start-stage" not in command:
                    raise SystemExit("select failed")
                return {"command": list(command), "cwd": str(cwd), "returncode": 0}

            with mock.patch.object(module, "run_command", side_effect=fake_run_command):
                with self.assertRaisesRegex(SystemExit, "select failed"):
                    module.run(args, repo_root)

            self.assertEqual(prior_verdict, json.loads(run_paths["verdict_path"].read_text(encoding="utf-8")))

    def test_run_accepts_fresh_failing_verdict_when_writer_exits_one(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-tactical-capture-protection-validation-") as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "capture-protection-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                ]
            )
            run_paths = module.build_run_paths(output_root, args.run_id)
            stale_verdict = {
                "schema": "azlite_tactical_validation_verdict_v1",
                "passed": True,
                "verdict": "pass",
                "failure_reasons": [],
            }
            run_paths["verdict_path"].parent.mkdir(parents=True, exist_ok=True)
            run_paths["verdict_path"].write_text(json.dumps(stale_verdict), encoding="utf-8")

            def fake_run_command(command, cwd, allowed_returncodes=(0,)):
                if command[1].endswith("script/ai/run_local_tactical_replay_experiment"):
                    if "--start-stage" not in command:
                        run_paths["selection_artifact"].mkdir(parents=True, exist_ok=True)
                        (run_paths["selection_artifact"] / "model.npz").write_text("stub", encoding="utf-8")
                    else:
                        run_paths["final_dir"].mkdir(parents=True, exist_ok=True)
                        for artifact_name in (
                            "baseline_candidate_forensics.json",
                            "selected_candidate_forensics.json",
                            "bucket_gate.json",
                            "candidate_regression_suite.json",
                            "arena_seed_1041.json",
                        ):
                            (run_paths["final_dir"] / artifact_name).write_text("{}", encoding="utf-8")

                if command[1:3] == ["-m", "ml.alphazero_lite.write_tactical_validation_verdict"]:
                    run_paths["verdict_path"].write_text(
                        json.dumps(
                            {
                                "schema": "azlite_tactical_validation_verdict_v1",
                                "passed": False,
                                "verdict": "fail",
                                "failure_reasons": ["bucket_gate_failed"],
                            }
                        ),
                        encoding="utf-8",
                    )
                    return {"command": list(command), "cwd": str(cwd), "returncode": 1}

                return {"command": list(command), "cwd": str(cwd), "returncode": 0}

            with mock.patch.object(module, "run_command", side_effect=fake_run_command):
                summary = module.run(args, repo_root)

            self.assertFalse(summary["verdict"]["passed"])
            self.assertEqual("fail", summary["verdict"]["verdict"])
            self.assertEqual(["bucket_gate_failed"], summary["verdict"]["failure_reasons"])
            self.assertEqual(summary, json.loads(run_paths["wrapper_summary_path"].read_text(encoding="utf-8")))

    def test_run_removes_old_verdict_after_select_when_validation_pack_fails(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-tactical-capture-protection-validation-") as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "capture-protection-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                ]
            )
            run_paths = module.build_run_paths(output_root, args.run_id)
            old_verdict = {
                "schema": "azlite_tactical_validation_verdict_v1",
                "passed": True,
                "verdict": "pass",
                "failure_reasons": [],
            }
            run_paths["verdict_path"].parent.mkdir(parents=True, exist_ok=True)
            run_paths["verdict_path"].write_text(json.dumps(old_verdict), encoding="utf-8")

            def fake_run_command(command, cwd, allowed_returncodes=(0,)):
                if command[1].endswith("script/ai/run_local_tactical_replay_experiment"):
                    if "--start-stage" not in command:
                        run_paths["selection_artifact"].mkdir(parents=True, exist_ok=True)
                        (run_paths["selection_artifact"] / "model.npz").write_text("stub", encoding="utf-8")
                        return {"command": list(command), "cwd": str(cwd), "returncode": 0}
                    raise SystemExit("validation pack failed")
                return {"command": list(command), "cwd": str(cwd), "returncode": 0}

            with mock.patch.object(module, "run_command", side_effect=fake_run_command):
                with self.assertRaisesRegex(SystemExit, "validation pack failed"):
                    module.run(args, repo_root)

            self.assertFalse(run_paths["verdict_path"].exists())

    def test_run_removes_old_summary_after_select_when_validation_pack_fails(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-tactical-capture-protection-validation-") as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "capture-protection-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                ]
            )
            run_paths = module.build_run_paths(output_root, args.run_id)
            old_summary = {
                "schema": "azlite_tactical_capture_protection_validation_summary_v1",
                "run_id": "capture-protection-demo",
                "executed_stages": ["select", "validation_pack", "write_verdict"],
                "selected_artifact": "stale-selection",
            }
            run_paths["wrapper_summary_path"].parent.mkdir(parents=True, exist_ok=True)
            run_paths["wrapper_summary_path"].write_text(json.dumps(old_summary), encoding="utf-8")

            def fake_run_command(command, cwd, allowed_returncodes=(0,)):
                if command[1].endswith("script/ai/run_local_tactical_replay_experiment"):
                    if "--start-stage" not in command:
                        run_paths["selection_artifact"].mkdir(parents=True, exist_ok=True)
                        (run_paths["selection_artifact"] / "model.npz").write_text("stub", encoding="utf-8")
                        return {"command": list(command), "cwd": str(cwd), "returncode": 0}
                    raise SystemExit("validation pack failed")
                return {"command": list(command), "cwd": str(cwd), "returncode": 0}

            with mock.patch.object(module, "run_command", side_effect=fake_run_command):
                with self.assertRaisesRegex(SystemExit, "validation pack failed"):
                    module.run(args, repo_root)

            self.assertFalse(run_paths["wrapper_summary_path"].exists())

    def test_wrapper_python_executable_honors_env_override_first(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]
        original_value = os.environ.get("AZLITE_EXPERIMENT_PYTHON")

        try:
            os.environ["AZLITE_EXPERIMENT_PYTHON"] = "/tmp/custom-python"
            self.assertEqual("/tmp/custom-python", module.python_executable(repo_root))
        finally:
            if original_value is None:
                os.environ.pop("AZLITE_EXPERIMENT_PYTHON", None)
            else:
                os.environ["AZLITE_EXPERIMENT_PYTHON"] = original_value
