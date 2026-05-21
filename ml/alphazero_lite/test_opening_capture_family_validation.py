import importlib.machinery
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


class OpeningCaptureFamilyValidationScriptTest(unittest.TestCase):
    def load_module(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = (
            repo_root / "script/ai/run_tactical_opening_capture_family_validation"
        )
        loader = importlib.machinery.SourceFileLoader(
            "run_tactical_opening_capture_family_validation",
            str(script_path),
        )
        module = types.ModuleType(loader.name)
        module.__file__ = str(script_path)
        loader.exec_module(module)
        return module

    def family_report_payload(self):
        return {"schema": "opening_capture_family_report_v1", "rows": []}

    def test_dry_run_pins_stages_sources_and_report_command(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(
            prefix="azlite-opening-capture-family-validation-"
        ) as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "opening-capture-family-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                    "--dry-run",
                ]
            )

            payload = module.build_dry_run_payload(args, repo_root)

        run_paths = module.build_run_paths(output_root, "opening-capture-family-demo")
        pinned_sources = payload["pinned_sources"]
        self.assertEqual(
            ["select", "validation_pack", "write_family_report", "write_verdict"],
            payload["stages"],
        )
        self.assertEqual(
            str(
                repo_root
                / "ml/alphazero_lite/configs/aggressive_v3_tactical_opening_capture_family_local.json"
            ),
            pinned_sources["base-config"],
        )
        self.assertEqual(
            str(
                repo_root
                / "ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"
            ),
            pinned_sources["forensic-suite"],
        )
        self.assertEqual(
            str(run_paths["selection_artifact"]), pinned_sources["selected-artifact"]
        )
        self.assertEqual(
            str(run_paths["base_dir"] / "inputs" / "reference_moves.json"),
            pinned_sources["reference-artifact"],
        )
        self.assertEqual(
            str(run_paths["final_dir"] / "baseline_candidate_forensics.json"),
            pinned_sources["baseline-forensics"],
        )
        self.assertEqual(
            str(run_paths["final_dir"] / "selected_candidate_forensics.json"),
            pinned_sources["candidate-forensics"],
        )
        self.assertEqual(
            str(run_paths["final_dir"] / "bucket_gate.json"),
            pinned_sources["bucket-gate"],
        )
        self.assertEqual(
            str(run_paths["final_dir"] / "candidate_regression_suite.json"),
            pinned_sources["regression-report"],
        )
        self.assertEqual(
            str(run_paths["final_dir"] / "arena_seed_1041.json"),
            pinned_sources["arena-report"],
        )
        self.assertEqual(
            str(run_paths["family_report_path"]),
            pinned_sources["family-report"],
        )
        self.assertEqual(
            str(run_paths["base_dir"] / "inputs" / "reference_moves.json"),
            pinned_sources["reference-forensics"],
        )

        select_command = payload["commands"]["select"]
        validation_command = payload["commands"]["validation_pack"]
        family_report_command = payload["commands"]["write_family_report"]
        verdict_command = payload["commands"]["write_verdict"]

        self.assertTrue(
            select_command[1].endswith("script/ai/run_local_tactical_replay_experiment")
        )
        self.assertNotIn("--start-stage", select_command)
        self.assertNotIn("--selected-artifact", select_command)
        self.assertTrue(
            validation_command[1].endswith(
                "script/ai/run_local_tactical_replay_experiment"
            )
        )
        self.assertEqual(
            "validation_pack",
            validation_command[validation_command.index("--start-stage") + 1],
        )
        skip_stage_values = [
            validation_command[index + 1]
            for index, value in enumerate(validation_command)
            if value == "--skip-stage"
        ]
        self.assertEqual(["final_holdout", "decision"], skip_stage_values)
        self.assertEqual(
            str(run_paths["selection_artifact"]),
            validation_command[validation_command.index("--selected-artifact") + 1],
        )
        self.assertEqual(
            pinned_sources["reference-artifact"],
            validation_command[validation_command.index("--reference-artifact") + 1],
        )
        self.assertEqual(
            ["-m", "ml.alphazero_lite.write_opening_capture_family_report"],
            family_report_command[1:3],
        )
        self.assertEqual(
            pinned_sources["forensic-suite"],
            family_report_command[family_report_command.index("--suite") + 1],
        )
        self.assertEqual(
            args.current_path,
            family_report_command[
                family_report_command.index("--current-artifact") + 1
            ],
        )
        self.assertEqual(
            str(run_paths["selection_artifact"]),
            family_report_command[
                family_report_command.index("--candidate-artifact") + 1
            ],
        )
        self.assertEqual(
            str(run_paths["family_report_path"]),
            family_report_command[family_report_command.index("--out") + 1],
        )
        self.assertEqual(
            pinned_sources["reference-forensics"],
            family_report_command[family_report_command.index("--reference") + 1],
        )
        self.assertEqual(
            ["-m", "ml.alphazero_lite.write_tactical_validation_verdict"],
            verdict_command[1:3],
        )
        self.assertNotIn("--family-report", verdict_command)

    def test_dry_run_pins_shared_reference_artifact_for_select_and_validation(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(
            prefix="azlite-opening-capture-family-validation-"
        ) as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "opening-capture-family-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                    "--dry-run",
                ]
            )

            payload = module.build_dry_run_payload(args, repo_root)

        run_paths = module.build_run_paths(output_root, "opening-capture-family-demo")
        select_command = payload["commands"]["select"]
        validation_command = payload["commands"]["validation_pack"]
        reference_artifact = str(
            run_paths["base_dir"] / "inputs" / "reference_moves.json"
        )

        self.assertEqual(
            reference_artifact,
            select_command[select_command.index("--reference-artifact") + 1],
        )
        self.assertEqual(
            reference_artifact,
            validation_command[validation_command.index("--reference-artifact") + 1],
        )

    def test_run_executes_report_stage_before_verdict_and_writes_summary(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(
            prefix="azlite-opening-capture-family-validation-"
        ) as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "opening-capture-family-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                ]
            )
            run_paths = module.build_run_paths(output_root, args.run_id)
            commands_run = []

            def fake_run_command(command, cwd, allowed_returncodes=(0,)):
                commands_run.append((list(command), tuple(allowed_returncodes)))

                if command[1].endswith(
                    "script/ai/run_local_tactical_replay_experiment"
                ):
                    if "--start-stage" not in command:
                        run_paths["selection_artifact"].mkdir(
                            parents=True, exist_ok=True
                        )
                        (run_paths["selection_artifact"] / "model.npz").write_text(
                            "stub", encoding="utf-8"
                        )
                    else:
                        run_paths["final_dir"].mkdir(parents=True, exist_ok=True)
                        for artifact_name in (
                            "baseline_candidate_forensics.json",
                            "selected_candidate_forensics.json",
                            "bucket_gate.json",
                            "candidate_regression_suite.json",
                            "arena_seed_1041.json",
                        ):
                            (run_paths["final_dir"] / artifact_name).write_text(
                                "{}", encoding="utf-8"
                            )

                if command[1:3] == [
                    "-m",
                    "ml.alphazero_lite.write_opening_capture_family_report",
                ]:
                    report_path = Path(command[command.index("--out") + 1])
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(
                        json.dumps(self.family_report_payload()), encoding="utf-8"
                    )

                if command[1:3] == [
                    "-m",
                    "ml.alphazero_lite.write_tactical_validation_verdict",
                ]:
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
            self.assertTrue(run_paths["family_report_path"].exists())
            self.assertTrue(run_paths["verdict_path"].exists())
            self.assertEqual(
                summary,
                json.loads(
                    run_paths["wrapper_summary_path"].read_text(encoding="utf-8")
                ),
            )

        self.assertEqual(
            ["select", "validation_pack", "write_family_report", "write_verdict"],
            summary["executed_stages"],
        )
        self.assertEqual(4, len(commands_run))
        self.assertTrue(
            commands_run[2][0][1:3]
            == ["-m", "ml.alphazero_lite.write_opening_capture_family_report"]
        )
        self.assertTrue(
            commands_run[3][0][1:3]
            == ["-m", "ml.alphazero_lite.write_tactical_validation_verdict"]
        )
        self.assertEqual((0, 1), commands_run[3][1])
        self.assertEqual(self.family_report_payload(), summary["family_report"])
        self.assertEqual(
            str(run_paths["family_report_path"]), summary["family_report_path"]
        )
        self.assertTrue(summary["verdict"]["passed"])

    def test_run_requires_fresh_family_report_artifact(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(
            prefix="azlite-opening-capture-family-validation-"
        ) as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "opening-capture-family-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                ]
            )
            run_paths = module.build_run_paths(output_root, args.run_id)
            run_paths["family_report_path"].parent.mkdir(parents=True, exist_ok=True)
            run_paths["family_report_path"].write_text(
                json.dumps(self.family_report_payload()), encoding="utf-8"
            )

            def fake_run_command(command, cwd, allowed_returncodes=(0,)):
                if command[1].endswith(
                    "script/ai/run_local_tactical_replay_experiment"
                ):
                    if "--start-stage" not in command:
                        run_paths["selection_artifact"].mkdir(
                            parents=True, exist_ok=True
                        )
                        (run_paths["selection_artifact"] / "model.npz").write_text(
                            "stub", encoding="utf-8"
                        )
                    else:
                        run_paths["final_dir"].mkdir(parents=True, exist_ok=True)
                        for artifact_name in (
                            "baseline_candidate_forensics.json",
                            "selected_candidate_forensics.json",
                            "bucket_gate.json",
                            "candidate_regression_suite.json",
                            "arena_seed_1041.json",
                        ):
                            (run_paths["final_dir"] / artifact_name).write_text(
                                "{}", encoding="utf-8"
                            )

                if command[1:3] == [
                    "-m",
                    "ml.alphazero_lite.write_opening_capture_family_report",
                ]:
                    if run_paths["family_report_path"].exists():
                        run_paths["family_report_path"].unlink()
                    return {"command": list(command), "cwd": str(cwd), "returncode": 0}

                return {"command": list(command), "cwd": str(cwd), "returncode": 0}

            with mock.patch.object(module, "run_command", side_effect=fake_run_command):
                with self.assertRaisesRegex(SystemExit, "fresh family report"):
                    module.run(args, repo_root)

    def test_run_validates_family_report_schema(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(
            prefix="azlite-opening-capture-family-validation-"
        ) as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "opening-capture-family-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                ]
            )
            run_paths = module.build_run_paths(output_root, args.run_id)

            def fake_run_command(command, cwd, allowed_returncodes=(0,)):
                if command[1].endswith(
                    "script/ai/run_local_tactical_replay_experiment"
                ):
                    if "--start-stage" not in command:
                        run_paths["selection_artifact"].mkdir(
                            parents=True, exist_ok=True
                        )
                        (run_paths["selection_artifact"] / "model.npz").write_text(
                            "stub", encoding="utf-8"
                        )
                    else:
                        run_paths["final_dir"].mkdir(parents=True, exist_ok=True)
                        for artifact_name in (
                            "baseline_candidate_forensics.json",
                            "selected_candidate_forensics.json",
                            "bucket_gate.json",
                            "candidate_regression_suite.json",
                            "arena_seed_1041.json",
                        ):
                            (run_paths["final_dir"] / artifact_name).write_text(
                                "{}", encoding="utf-8"
                            )

                if command[1:3] == [
                    "-m",
                    "ml.alphazero_lite.write_opening_capture_family_report",
                ]:
                    run_paths["family_report_path"].parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    run_paths["family_report_path"].write_text(
                        json.dumps({"schema": "wrong_schema", "rows": []}),
                        encoding="utf-8",
                    )

                return {"command": list(command), "cwd": str(cwd), "returncode": 0}

            with mock.patch.object(module, "run_command", side_effect=fake_run_command):
                with self.assertRaisesRegex(
                    SystemExit, "opening_capture_family_report_v1"
                ):
                    module.run(args, repo_root)

    def test_run_preserves_failing_verdict_exit_path_with_fresh_artifacts(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(
            prefix="azlite-opening-capture-family-validation-"
        ) as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "opening-capture-family-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                ]
            )
            run_paths = module.build_run_paths(output_root, args.run_id)
            commands_run = []

            def fake_run_command(command, cwd, allowed_returncodes=(0,)):
                commands_run.append((list(command), tuple(allowed_returncodes)))
                if command[1].endswith(
                    "script/ai/run_local_tactical_replay_experiment"
                ):
                    if "--start-stage" not in command:
                        run_paths["selection_artifact"].mkdir(
                            parents=True, exist_ok=True
                        )
                        (run_paths["selection_artifact"] / "model.npz").write_text(
                            "stub", encoding="utf-8"
                        )
                    else:
                        run_paths["final_dir"].mkdir(parents=True, exist_ok=True)
                        for artifact_name in (
                            "baseline_candidate_forensics.json",
                            "selected_candidate_forensics.json",
                            "bucket_gate.json",
                            "candidate_regression_suite.json",
                            "arena_seed_1041.json",
                        ):
                            (run_paths["final_dir"] / artifact_name).write_text(
                                "{}", encoding="utf-8"
                            )
                elif command[1:3] == [
                    "-m",
                    "ml.alphazero_lite.write_opening_capture_family_report",
                ]:
                    run_paths["family_report_path"].parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    run_paths["family_report_path"].write_text(
                        json.dumps(self.family_report_payload()), encoding="utf-8"
                    )
                elif command[1:3] == [
                    "-m",
                    "ml.alphazero_lite.write_tactical_validation_verdict",
                ]:
                    run_paths["verdict_path"].parent.mkdir(parents=True, exist_ok=True)
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
        self.assertEqual(["bucket_gate_failed"], summary["verdict"]["failure_reasons"])
        self.assertEqual((0, 1), commands_run[-1][1])

    def test_run_continues_after_failing_validation_pack_to_emit_report_and_verdict(
        self,
    ):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(
            prefix="azlite-opening-capture-family-validation-"
        ) as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "opening-capture-family-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                ]
            )
            run_paths = module.build_run_paths(output_root, args.run_id)
            commands_run = []

            def fake_run_command(command, cwd, allowed_returncodes=(0,)):
                commands_run.append((list(command), tuple(allowed_returncodes)))
                if command[1].endswith(
                    "script/ai/run_local_tactical_replay_experiment"
                ):
                    if "--start-stage" not in command:
                        run_paths["selection_artifact"].mkdir(
                            parents=True, exist_ok=True
                        )
                        (run_paths["selection_artifact"] / "model.npz").write_text(
                            "stub", encoding="utf-8"
                        )
                        return {
                            "command": list(command),
                            "cwd": str(cwd),
                            "returncode": 0,
                        }

                    run_paths["final_dir"].mkdir(parents=True, exist_ok=True)
                    for artifact_name in (
                        "baseline_candidate_forensics.json",
                        "selected_candidate_forensics.json",
                        "bucket_gate.json",
                        "candidate_regression_suite.json",
                        "arena_seed_1041.json",
                    ):
                        (run_paths["final_dir"] / artifact_name).write_text(
                            "{}", encoding="utf-8"
                        )
                    return {"command": list(command), "cwd": str(cwd), "returncode": 1}

                if command[1:3] == [
                    "-m",
                    "ml.alphazero_lite.write_opening_capture_family_report",
                ]:
                    run_paths["family_report_path"].parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    run_paths["family_report_path"].write_text(
                        json.dumps(self.family_report_payload()), encoding="utf-8"
                    )
                    return {"command": list(command), "cwd": str(cwd), "returncode": 0}

                if command[1:3] == [
                    "-m",
                    "ml.alphazero_lite.write_tactical_validation_verdict",
                ]:
                    run_paths["verdict_path"].parent.mkdir(parents=True, exist_ok=True)
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

            self.assertTrue(run_paths["family_report_path"].exists())
            self.assertTrue(run_paths["verdict_path"].exists())

        self.assertEqual((0, 1), commands_run[1][1])
        self.assertTrue(
            commands_run[2][0][1:3]
            == ["-m", "ml.alphazero_lite.write_opening_capture_family_report"]
        )
        self.assertTrue(
            commands_run[3][0][1:3]
            == ["-m", "ml.alphazero_lite.write_tactical_validation_verdict"]
        )
        self.assertFalse(summary["verdict"]["passed"])


if __name__ == "__main__":
    unittest.main()
