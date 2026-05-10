import importlib.machinery
import json
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import mock_open, patch

from ml.alphazero_lite import run_targeted_source_coverage_redesign as module


class RunTargetedSourceCoverageRedesignTest(unittest.TestCase):
    def load_wrapper_module(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "script/ai/run_targeted_source_coverage_redesign"
        loader = importlib.machinery.SourceFileLoader(
            "run_targeted_source_coverage_redesign_wrapper",
            str(script_path),
        )
        module = types.ModuleType(loader.name)
        module.__file__ = str(script_path)
        loader.exec_module(module)
        return module

    def test_builder_phase_must_pass_for_both_variants_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "runs"

            with patch.object(module, "build_variant_artifact_summaries") as build_summaries, patch.object(
                module, "run_variant_training"
            ) as run_training:
                build_summaries.return_value = {
                    "capped_11": {"pass_flags": {"structurally_valid": True}},
                    "expanded_12_guard_reinforced": {"pass_flags": {"structurally_valid": False}},
                }

                result = module.run_redesign(
                    output_root=output_root,
                    base_config_path=Path(
                        "ml/alphazero_lite/configs/aggressive_v3_tactical_balanced_replay_local.json"
                    ),
                    current_path="model-artifact/current",
                    forensic_suite_path=Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"),
                )

            self.assertFalse(result["training_started"])
            run_training.assert_not_called()

    def test_training_starts_when_both_variant_summaries_are_structurally_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "runs"

            with patch.object(module, "build_variant_artifact_summaries") as build_summaries, patch.object(
                module, "run_variant_training"
            ) as run_training:
                build_summaries.return_value = {
                    "capped_11": {"pass_flags": {"structurally_valid": True}},
                    "expanded_12_guard_reinforced": {"pass_flags": {"structurally_valid": True}},
                }
                run_training.return_value = {"status": "ok"}

                result = module.run_redesign(
                    output_root=output_root,
                    base_config_path=Path(
                        "ml/alphazero_lite/configs/aggressive_v3_tactical_balanced_replay_local.json"
                    ),
                    current_path="model-artifact/current",
                    forensic_suite_path=Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"),
                )

            self.assertTrue(result["training_started"])
            self.assertEqual(set(module.VARIANT_RUN_IDS), set(result["training_results"]))
            self.assertEqual(len(module.VARIANT_RUN_IDS), run_training.call_count)

    def test_builder_phase_requires_all_expected_variant_summaries_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "runs"

            with patch.object(module, "build_variant_artifact_summaries") as build_summaries, patch.object(
                module, "run_variant_training"
            ) as run_training:
                build_summaries.return_value = {
                    "capped_11": {"pass_flags": {"structurally_valid": True}},
                }

                result = module.run_redesign(
                    output_root=output_root,
                    base_config_path=Path(
                        "ml/alphazero_lite/configs/aggressive_v3_tactical_balanced_replay_local.json"
                    ),
                    current_path="model-artifact/current",
                    forensic_suite_path=Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"),
                )

            self.assertFalse(result["training_started"])
            run_training.assert_not_called()

    def test_builder_phase_fails_closed_for_malformed_variant_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "runs"

            with patch.object(module, "build_variant_artifact_summaries") as build_summaries, patch.object(
                module, "run_variant_training"
            ) as run_training:
                build_summaries.return_value = {
                    "capped_11": {"pass_flags": {"structurally_valid": True}},
                    "expanded_12_guard_reinforced": {},
                }

                result = module.run_redesign(
                    output_root=output_root,
                    base_config_path=Path(
                        "ml/alphazero_lite/configs/aggressive_v3_tactical_balanced_replay_local.json"
                    ),
                    current_path="model-artifact/current",
                    forensic_suite_path=Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"),
                )

            self.assertFalse(result["training_started"])
            run_training.assert_not_called()

    def test_build_variant_artifact_summaries_builds_both_expected_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "runs"
            regression_positions_path = Path("test/fixtures/ai/superhuman_regression_positions.json")
            tactical_replay_path = Path("ml/alphazero_lite/tactical_balanced_replay_source.jsonl")

            def fake_build_balanced_replay_dataset(*, regression_positions_path, tactical_replay_path, out_path, variant):
                return [], {
                    "variant": variant,
                    "replay_artifact_path": str(out_path),
                    "pass_flags": {"structurally_valid": True},
                }

            with patch.object(
                module,
                "build_balanced_replay_dataset",
                side_effect=fake_build_balanced_replay_dataset,
                create=True,
            ) as builder:
                summaries = module.build_variant_artifact_summaries(
                    output_root=output_root,
                    regression_positions_path=regression_positions_path,
                    tactical_replay_path=tactical_replay_path,
                )

            self.assertEqual(set(module.VARIANT_RUN_IDS), set(summaries))
            self.assertEqual([call.kwargs["variant"] for call in builder.call_args_list], list(module.VARIANT_RUN_IDS))
            for variant, run_id in module.VARIANT_RUN_IDS.items():
                expected_replay_path = output_root / run_id / "final" / "tactical_balanced_replay.jsonl"
                builder_call = next(call for call in builder.call_args_list if call.kwargs["variant"] == variant)
                self.assertEqual(regression_positions_path, builder_call.kwargs["regression_positions_path"])
                self.assertEqual(tactical_replay_path, builder_call.kwargs["tactical_replay_path"])
                self.assertEqual(expected_replay_path, builder_call.kwargs["out_path"])
                self.assertEqual(str(expected_replay_path), summaries[variant]["replay_artifact_path"])

    def test_run_variant_training_invokes_local_experiment_wrapper_and_returns_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "runs"
            base_config_path = Path("ml/alphazero_lite/configs/aggressive_v3_tactical_balanced_replay_local.json")
            forensic_suite_path = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")

            completed = SimpleNamespace(
                stdout='{"status": "completed", "run_id": "targeted-source-coverage-033-capped-11"}\n',
                returncode=0,
            )
            base_config = {
                "fixed_replay_sources": [{"path": "ml/alphazero_lite/tactical_balanced_replay.jsonl", "weight": 8}]
            }

            fake_subprocess = SimpleNamespace(run=None)

            with patch.object(module, "subprocess", fake_subprocess, create=True), patch.object(
                fake_subprocess, "run", return_value=completed
            ) as run_subprocess, patch.object(module, "_load_json", return_value=base_config), patch.object(
                module, "_write_json"
            ) as write_json:
                summary = module.run_variant_training(
                    variant="capped_11",
                    run_id="targeted-source-coverage-033-capped-11",
                    output_root=output_root,
                    base_config_path=base_config_path,
                    current_path="model-artifact/current",
                    forensic_suite_path=forensic_suite_path,
                )

            self.assertEqual("completed", summary["status"])
            self.assertEqual("targeted-source-coverage-033-capped-11", summary["run_id"])
            command = run_subprocess.call_args.kwargs["args"]
            self.assertEqual([str(module._python_bin()), str(module.LOCAL_EXPERIMENT_WRAPPER)], command[:2])
            self.assertIn("--base-config", command)
            runtime_config_path = Path(command[command.index("--base-config") + 1])
            self.assertEqual(
                output_root / "targeted-source-coverage-033-capped-11" / "inputs" / "runtime_config.json",
                runtime_config_path,
            )
            self.assertEqual(runtime_config_path, write_json.call_args.kwargs["path"])
            self.assertEqual(
                [{
                    "path": str(
                        output_root
                        / "targeted-source-coverage-033-capped-11"
                        / "final"
                        / "tactical_balanced_replay.jsonl"
                    ),
                    "weight": 8,
                }],
                write_json.call_args.kwargs["payload"]["fixed_replay_sources"],
            )
            self.assertIn("--run-id", command)
            self.assertIn("targeted-source-coverage-033-capped-11", command)
            self.assertIn("--output-root", command)
            self.assertIn(str(output_root), command)
            self.assertIn("--current-path", command)
            self.assertIn("model-artifact/current", command)
            self.assertIn("--forensic-suite", command)
            self.assertIn(str(forensic_suite_path), command)

    def test_run_variant_training_parses_final_json_line_from_noisy_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "runs"
            base_config_path = Path("ml/alphazero_lite/configs/aggressive_v3_tactical_balanced_replay_local.json")
            forensic_suite_path = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
            base_config = {
                "fixed_replay_sources": [{"path": "ml/alphazero_lite/tactical_balanced_replay.jsonl", "weight": 8}]
            }
            completed = SimpleNamespace(
                stdout='wrote forensic report\npipeline_scaffold_complete\n{"status": "completed", "run_id": "targeted-source-coverage-033-capped-11"}\n',
                returncode=0,
            )
            fake_subprocess = SimpleNamespace(run=None)

            with patch.object(module, "subprocess", fake_subprocess, create=True), patch.object(
                fake_subprocess, "run", return_value=completed
            ), patch.object(module, "_load_json", return_value=base_config), patch.object(module, "_write_json"):
                summary = module.run_variant_training(
                    variant="capped_11",
                    run_id="targeted-source-coverage-033-capped-11",
                    output_root=output_root,
                    base_config_path=base_config_path,
                    current_path="model-artifact/current",
                    forensic_suite_path=forensic_suite_path,
                )

        self.assertEqual("completed", summary["status"])
        self.assertEqual("targeted-source-coverage-033-capped-11", summary["run_id"])

    def test_main_prints_json_payload(self):
        expected = {"training_started": False}

        with patch.object(
            module,
            "parse_args",
            return_value=SimpleNamespace(
                output_root="/tmp/redesign",
                base_config="base.json",
                current_path="model-artifact/current",
                forensic_suite="suite.json",
            ),
        ), patch.object(module, "run_redesign", return_value=expected), patch("builtins.print") as print_mock:
            module.main()

        print_mock.assert_called_once_with(json.dumps(expected))

    def test_wrapper_bootstraps_repo_root_before_importing_ml_module(self):
        repo_root = Path(__file__).resolve().parents[2]
        removed_modules = {}
        for name in [
            "ml",
            "ml.alphazero_lite",
            "ml.alphazero_lite.run_targeted_source_coverage_redesign",
        ]:
            if name in sys.modules:
                removed_modules[name] = sys.modules.pop(name)

        try:
            isolated_path = [entry for entry in sys.path if Path(entry or ".").resolve() != repo_root.resolve()]
            with patch.object(sys, "path", isolated_path):
                wrapper = self.load_wrapper_module()
        finally:
            sys.modules.update(removed_modules)

        self.assertTrue(callable(wrapper.main))

    def test_wrapper_main_runs_from_repo_root(self):
        repo_root = Path(__file__).resolve().parents[2]
        wrapper = self.load_wrapper_module()

        with patch.object(wrapper.os, "chdir") as chdir, patch.object(wrapper, "run_redesign_main") as runner_main:
            wrapper.main()

        chdir.assert_called_once_with(repo_root)
        runner_main.assert_called_once_with()
