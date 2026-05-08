import importlib.machinery
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


class StableFailureFamilyValidationScriptTest(unittest.TestCase):
    def load_module(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "script/ai/run_tactical_stable_failure_family_validation"
        loader = importlib.machinery.SourceFileLoader(
            "run_tactical_stable_failure_family_validation",
            str(script_path),
        )
        module = types.ModuleType(loader.name)
        module.__file__ = str(script_path)
        loader.exec_module(module)
        return module

    def stable_family_summary(self):
        return {
            "schema": "azlite_stable_failure_family_summary_v1",
            "capture_available": {
                "tracked_rows": 1,
                "average_regret": 0.0882,
                "blunder_rate_0_20": 0.0,
                "blunder_ids": [],
                "search_flipped_rows": 0,
                "search_flipped_ids": [],
                "missing_opening_family_rows": [],
            },
            "high_imbalance": {
                "stable_rows": 1,
                "average_regret": 0.2112,
                "blunder_rate_0_20": 1.0,
                "blunder_ids": ["high_imbalance-001"],
            },
        }

    def replay_summary(self):
        return {
            "schema": "azlite_tactical_stable_failure_family_replay_summary_v1",
            "summary_artifact_path": "/tmp/tactical_stable_failure_family_replay_summary.json",
            "replay_artifact_path": "/tmp/tactical_stable_failure_family_replay.jsonl",
            "target_counts": {
                "capture_protection": 1,
                "capture_preservation": 4,
                "opening_capture_family": 7,
                "high_imbalance_stable": 3,
                "nearby_preservation": 8,
            },
            "role_counts": {
                "capture_protection": 1,
                "capture_preservation": 4,
                "opening_capture_family": 7,
                "high_imbalance_stable": 3,
                "nearby_preservation": 8,
            },
            "selected_ids_by_role": {
                "capture_protection": ["capture_protection_regression"],
                "capture_preservation": [
                    "capture-preservation-001",
                    "capture-preservation-002",
                    "capture-preservation-003",
                    "capture-preservation-004",
                ],
                "opening_capture_family": [
                    "opening-capture-001",
                    "opening-capture-002",
                    "opening-capture-003",
                    "opening-capture-004",
                    "opening-capture-005",
                    "opening-capture-006",
                    "opening-capture-007",
                ],
                "high_imbalance_stable": ["high_imbalance-001", "high_imbalance-002", "high_imbalance-003"],
                "nearby_preservation": [
                    "nearby-preservation-001",
                    "nearby-preservation-002",
                    "nearby-preservation-003",
                    "nearby-preservation-004",
                    "nearby-preservation-005",
                    "nearby-preservation-006",
                    "nearby-preservation-007",
                    "nearby-preservation-008",
                ],
            },
            "shortfalls_by_role": {
                "capture_protection": 0,
                "capture_preservation": 0,
                "opening_capture_family": 0,
                "high_imbalance_stable": 0,
                "nearby_preservation": 0,
            },
            "invalid_reasons": [],
            "total_rows": 23,
        }

    def test_dry_run_pins_reference_replay_and_summary_artifacts(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-stable-failure-family-validation-") as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "stable-failure-family-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                    "--dry-run",
                ]
            )
            payload = module.build_dry_run_payload(args, repo_root)

        run_paths = module.build_run_paths(output_root, "stable-failure-family-demo")
        pins = payload["pinned_sources"]
        self.assertEqual(
            ["select", "validation_pack", "write_family_report", "write_stable_family_summary", "write_verdict"],
            payload["stages"],
        )
        self.assertEqual(str(run_paths["base_dir"] / "inputs" / "reference_moves.json"), pins["reference-artifact"])
        self.assertEqual(
            str(run_paths["base_dir"] / "inputs" / "tactical_stable_failure_family_replay_summary.json"),
            pins["replay-summary"],
        )
        self.assertEqual(str(run_paths["family_report_path"]), pins["family-report"])
        self.assertEqual(str(run_paths["stable_family_summary_path"]), pins["stable-family-summary"])

        stable_summary_command = payload["commands"]["write_stable_family_summary"]
        self.assertEqual(
            ["-m", "ml.alphazero_lite.write_stable_failure_family_summary"],
            stable_summary_command[1:3],
        )
        self.assertEqual(
            pins["candidate-forensics"],
            stable_summary_command[stable_summary_command.index("--candidate-forensics") + 1],
        )
        self.assertEqual(
            pins["family-report"],
            stable_summary_command[stable_summary_command.index("--opening-family-report") + 1],
        )
        self.assertEqual(
            pins["stable-family-summary"],
            stable_summary_command[stable_summary_command.index("--out") + 1],
        )

    def test_run_executes_stable_summary_stage_before_verdict_and_writes_summary(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-stable-failure-family-validation-") as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "stable-failure-family-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                ]
            )
            run_paths = module.build_run_paths(output_root, args.run_id)
            commands_run = []
            stale_replay_summary = {"stale": True}
            fresh_replay_summary = self.replay_summary()
            replay_summary_seen_by_validation = []

            run_paths["replay_summary_path"].parent.mkdir(parents=True, exist_ok=True)
            run_paths["replay_summary_path"].write_text(json.dumps(stale_replay_summary), encoding="utf-8")

            def fake_run_command(command, cwd, allowed_returncodes=(0,)):
                commands_run.append((list(command), tuple(allowed_returncodes)))
                if command[1].endswith("script/ai/run_local_tactical_replay_experiment"):
                    if "--start-stage" not in command:
                        run_paths["selection_artifact"].mkdir(parents=True, exist_ok=True)
                        (run_paths["selection_artifact"] / "model.npz").write_text("stub", encoding="utf-8")
                        replay_summary_seen_by_validation.append(run_paths["replay_summary_path"].exists())
                        run_paths["replay_summary_path"].write_text(json.dumps(fresh_replay_summary), encoding="utf-8")
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
                elif command[1:3] == ["-m", "ml.alphazero_lite.write_opening_capture_family_report"]:
                    run_paths["family_report_path"].parent.mkdir(parents=True, exist_ok=True)
                    run_paths["family_report_path"].write_text(
                        json.dumps({"schema": "opening_capture_family_report_v1", "rows": [], "missing_references": []}),
                        encoding="utf-8",
                    )
                elif command[1:3] == ["-m", "ml.alphazero_lite.write_stable_failure_family_summary"]:
                    run_paths["stable_family_summary_path"].parent.mkdir(parents=True, exist_ok=True)
                    run_paths["stable_family_summary_path"].write_text(json.dumps(self.stable_family_summary()), encoding="utf-8")
                elif command[1:3] == ["-m", "ml.alphazero_lite.write_tactical_validation_verdict"]:
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

            self.assertTrue(run_paths["wrapper_summary_path"].exists())
            self.assertEqual(summary, json.loads(run_paths["wrapper_summary_path"].read_text(encoding="utf-8")))

        stage_names = [
            command[0][2] if command[0][1:3] and command[0][1] == "-m" else command[0][1]
            for command in commands_run
        ]
        self.assertTrue(
            stage_names.index("ml.alphazero_lite.write_stable_failure_family_summary")
            < stage_names.index("ml.alphazero_lite.write_tactical_validation_verdict")
        )
        self.assertEqual(
            ["select", "validation_pack", "write_family_report", "write_stable_family_summary", "write_verdict"],
            summary["executed_stages"],
        )
        self.assertEqual([False], replay_summary_seen_by_validation)
        self.assertEqual(fresh_replay_summary, summary["replay_summary"])
        self.assertEqual(str(run_paths["replay_summary_path"]), summary["replay_summary_path"])
        self.assertEqual("azlite_stable_failure_family_summary_v1", summary["stable_family_summary"]["schema"])
        self.assertEqual(str(run_paths["stable_family_summary_path"]), summary["stable_family_summary_path"])

    def test_run_command_persists_log_paths_instead_of_full_stdout_stderr(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory(prefix="azlite-stable-failure-family-validation-") as tmp:
            cwd = Path(tmp)
            result = module.run_command(
                [
                    "/bin/sh",
                    "-c",
                    "python3 - <<'PY'\nprint('x' * 5000)\nimport sys\nsys.stderr.write('y' * 5000)\nPY",
                ],
                cwd,
                log_path=cwd / "command.log",
            )

            log_text = (cwd / "command.log").read_text(encoding="utf-8")

        self.assertEqual(0, result["returncode"])
        self.assertNotIn("stdout", result)
        self.assertNotIn("stderr", result)
        self.assertEqual(str(cwd / "command.log"), result["log_path"])
        self.assertIn("x" * 100, log_text)
        self.assertIn("y" * 100, log_text)

    def test_run_uses_original_selected_target_for_validation_pack_when_selection_artifact_is_materialized_copy(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-stable-failure-family-validation-") as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "stable-failure-family-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                ]
            )
            run_paths = module.build_run_paths(output_root, args.run_id)
            original_selected_target = Path(tmp) / "provided-artifact"
            original_selected_target.mkdir(parents=True, exist_ok=True)
            (original_selected_target / "model.npz").write_text("stub", encoding="utf-8")
            validation_selected_artifacts = []

            def fake_run_command(command, cwd, allowed_returncodes=(0,)):
                if command[1].endswith("script/ai/run_local_tactical_replay_experiment"):
                    if "--start-stage" not in command:
                        run_paths["selection_artifact"].mkdir(parents=True, exist_ok=True)
                        (run_paths["selection_artifact"] / "model.npz").write_text("stub", encoding="utf-8")
                        selection_manifest_path = run_paths["base_dir"] / "selection" / "selection_manifest.json"
                        selection_manifest_path.parent.mkdir(parents=True, exist_ok=True)
                        selection_manifest_path.write_text(
                            json.dumps(
                                {
                                    "selected_artifact": str(run_paths["selection_artifact"]),
                                    "selected_target": str(original_selected_target),
                                    "selection_rule": "explicit_artifact",
                                    "candidate_count": 1,
                                }
                            ),
                            encoding="utf-8",
                        )
                        run_paths["replay_summary_path"].parent.mkdir(parents=True, exist_ok=True)
                        run_paths["replay_summary_path"].write_text(json.dumps(self.replay_summary()), encoding="utf-8")
                    else:
                        validation_selected_artifacts.append(command[command.index("--selected-artifact") + 1])
                        run_paths["final_dir"].mkdir(parents=True, exist_ok=True)
                        for artifact_name in (
                            "baseline_candidate_forensics.json",
                            "selected_candidate_forensics.json",
                            "bucket_gate.json",
                            "candidate_regression_suite.json",
                            "arena_seed_1041.json",
                        ):
                            (run_paths["final_dir"] / artifact_name).write_text("{}", encoding="utf-8")
                elif command[1:3] == ["-m", "ml.alphazero_lite.write_opening_capture_family_report"]:
                    run_paths["family_report_path"].parent.mkdir(parents=True, exist_ok=True)
                    run_paths["family_report_path"].write_text(
                        json.dumps({"schema": "opening_capture_family_report_v1", "rows": [], "missing_references": []}),
                        encoding="utf-8",
                    )
                elif command[1:3] == ["-m", "ml.alphazero_lite.write_stable_failure_family_summary"]:
                    run_paths["stable_family_summary_path"].parent.mkdir(parents=True, exist_ok=True)
                    run_paths["stable_family_summary_path"].write_text(json.dumps(self.stable_family_summary()), encoding="utf-8")
                elif command[1:3] == ["-m", "ml.alphazero_lite.write_tactical_validation_verdict"]:
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
                module.run(args, repo_root)

        self.assertEqual([str(original_selected_target)], validation_selected_artifacts)

    def test_run_rejects_invalid_verdict_artifact(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-stable-failure-family-validation-") as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "stable-failure-family-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                ]
            )
            run_paths = module.build_run_paths(output_root, args.run_id)

            def fake_run_command(command, cwd, allowed_returncodes=(0,)):
                if command[1].endswith("script/ai/run_local_tactical_replay_experiment"):
                    if "--start-stage" not in command:
                        run_paths["selection_artifact"].mkdir(parents=True, exist_ok=True)
                        (run_paths["selection_artifact"] / "model.npz").write_text("stub", encoding="utf-8")
                        run_paths["replay_summary_path"].parent.mkdir(parents=True, exist_ok=True)
                        run_paths["replay_summary_path"].write_text(json.dumps(self.replay_summary()), encoding="utf-8")
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
                elif command[1:3] == ["-m", "ml.alphazero_lite.write_opening_capture_family_report"]:
                    run_paths["family_report_path"].parent.mkdir(parents=True, exist_ok=True)
                    run_paths["family_report_path"].write_text(
                        json.dumps({"schema": "opening_capture_family_report_v1", "rows": [], "missing_references": []}),
                        encoding="utf-8",
                    )
                elif command[1:3] == ["-m", "ml.alphazero_lite.write_stable_failure_family_summary"]:
                    run_paths["stable_family_summary_path"].parent.mkdir(parents=True, exist_ok=True)
                    run_paths["stable_family_summary_path"].write_text(json.dumps(self.stable_family_summary()), encoding="utf-8")
                elif command[1:3] == ["-m", "ml.alphazero_lite.write_tactical_validation_verdict"]:
                    run_paths["verdict_path"].parent.mkdir(parents=True, exist_ok=True)
                    run_paths["verdict_path"].write_text(
                        json.dumps(
                            {
                                "schema": "wrong_schema",
                                "passed": "no",
                                "verdict": "fail",
                                "failure_reasons": ["bucket_gate_failed"],
                            }
                        ),
                        encoding="utf-8",
                    )

                return {"command": list(command), "cwd": str(cwd), "returncode": 0}

            with mock.patch.object(module, "run_command", side_effect=fake_run_command):
                with self.assertRaisesRegex(SystemExit, "azlite_tactical_validation_verdict_v1|invalid verdict artifact|boolean passed"):
                    module.run(args, repo_root)

    def test_run_rejects_malformed_stable_family_summary(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-stable-failure-family-validation-") as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "stable-failure-family-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                ]
            )
            run_paths = module.build_run_paths(output_root, args.run_id)

            def fake_run_command(command, cwd, allowed_returncodes=(0,)):
                if command[1].endswith("script/ai/run_local_tactical_replay_experiment"):
                    if "--start-stage" not in command:
                        run_paths["selection_artifact"].mkdir(parents=True, exist_ok=True)
                        (run_paths["selection_artifact"] / "model.npz").write_text("stub", encoding="utf-8")
                        run_paths["replay_summary_path"].parent.mkdir(parents=True, exist_ok=True)
                        run_paths["replay_summary_path"].write_text(json.dumps(self.replay_summary()), encoding="utf-8")
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
                elif command[1:3] == ["-m", "ml.alphazero_lite.write_opening_capture_family_report"]:
                    run_paths["family_report_path"].parent.mkdir(parents=True, exist_ok=True)
                    run_paths["family_report_path"].write_text(
                        json.dumps({"schema": "opening_capture_family_report_v1", "rows": [], "missing_references": []}),
                        encoding="utf-8",
                    )
                elif command[1:3] == ["-m", "ml.alphazero_lite.write_stable_failure_family_summary"]:
                    run_paths["stable_family_summary_path"].parent.mkdir(parents=True, exist_ok=True)
                    run_paths["stable_family_summary_path"].write_text(
                        json.dumps({"schema": "azlite_stable_failure_family_summary_v1", "capture_available": {}, "high_imbalance": {}}),
                        encoding="utf-8",
                    )

                return {"command": list(command), "cwd": str(cwd), "returncode": 0}

            with mock.patch.object(module, "run_command", side_effect=fake_run_command):
                with self.assertRaisesRegex(SystemExit, "stable family summary.*capture_available.tracked_rows"):
                    module.run(args, repo_root)

    def test_load_fresh_stable_family_summary_rejects_non_finite_metrics(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory(prefix="azlite-stable-family-summary-") as tmp:
            path = Path(tmp) / "stable_failure_family_summary.json"
            summary = self.stable_family_summary()
            summary["capture_available"]["average_regret"] = float("nan")
            path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "stable family summary.*capture_available.average_regret"):
                module.load_fresh_stable_family_summary(path)

    def test_load_fresh_replay_summary_rejects_wrong_schema(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory(prefix="azlite-replay-summary-") as tmp:
            path = Path(tmp) / "tactical_stable_failure_family_replay_summary.json"
            summary = self.replay_summary()
            summary["schema"] = "wrong_schema"
            path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(
                SystemExit,
                "replay builder produced schema.*azlite_tactical_stable_failure_family_replay_summary_v1",
            ):
                module.load_fresh_replay_summary(path)

    def test_run_aborts_when_validation_pack_fails_with_stale_final_artifacts_present(self):
        module = self.load_module()
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-stable-failure-family-validation-") as tmp:
            output_root = Path(tmp) / "runs"
            args = module.parse_args(
                [
                    "--run-id",
                    "stable-failure-family-demo",
                    "--output-root",
                    str(output_root),
                    "--current-path",
                    "model-artifact/current",
                ]
            )
            run_paths = module.build_run_paths(output_root, args.run_id)

            run_paths["final_dir"].mkdir(parents=True, exist_ok=True)
            for artifact_name in (
                "baseline_candidate_forensics.json",
                "selected_candidate_forensics.json",
                "bucket_gate.json",
                "candidate_regression_suite.json",
                "arena_seed_1041.json",
            ):
                (run_paths["final_dir"] / artifact_name).write_text('{"stale": true}', encoding="utf-8")

            run_paths["family_report_path"].write_text(
                json.dumps({"schema": "opening_capture_family_report_v1", "stale": True}),
                encoding="utf-8",
            )
            run_paths["stable_family_summary_path"].write_text(
                json.dumps(self.stable_family_summary()),
                encoding="utf-8",
            )
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
                        run_paths["replay_summary_path"].parent.mkdir(parents=True, exist_ok=True)
                        run_paths["replay_summary_path"].write_text(json.dumps(self.replay_summary()), encoding="utf-8")
                        return {"command": list(command), "cwd": str(cwd), "returncode": 0}

                    return {"command": list(command), "cwd": str(cwd), "returncode": 1}

                raise AssertionError(f"wrapper should not continue after validation_pack failure: {command}")

            with mock.patch.object(module, "run_command", side_effect=fake_run_command):
                with self.assertRaisesRegex(SystemExit, "validation_pack|exit code 1|failed"):
                    module.run(args, repo_root)

            self.assertFalse(run_paths["wrapper_summary_path"].exists())


if __name__ == "__main__":
    unittest.main()
