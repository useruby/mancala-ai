import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class SearchQualityOptionAblationTest(unittest.TestCase):
    def test_derive_candidate_path_uses_final_iteration_directory(self):
        from ml.alphazero_lite import run_search_quality_option_ablation as module

        candidate_path = module.derive_candidate_path(
            {
                "run_id": "aggressive-v2-search-quality-local",
                "start_iteration": 2,
                "iterations": 3,
                "versions_dir": "/tmp/azlite_v2_search_quality_local_versions",
            }
        )

        self.assertEqual(
            "/tmp/azlite_v2_search_quality_local_versions/aggressive-v2-search-quality-local-iter4",
            candidate_path,
        )

    def test_resolve_step_command_falls_back_to_sys_executable_without_repo_venv(self):
        from ml.alphazero_lite import run_search_quality_option_ablation as module

        command = [
            ".venv/bin/python",
            "ml/alphazero_lite/arena.py",
            "--out",
            "/tmp/report.json",
        ]

        with tempfile.TemporaryDirectory(
            prefix="azlite-search-quality-option-root-"
        ) as tmp:
            resolved_repo_root = Path(tmp) / "repo"
            resolved_repo_root.mkdir()

            resolved = module.resolve_step_command(
                command, resolved_repo_root=resolved_repo_root
            )

        self.assertEqual(sys.executable, resolved[0])
        self.assertEqual(command[1:], resolved[1:])

    def test_build_variant_plan_keeps_baseline_eval_budgets_and_paths(self):
        from ml.alphazero_lite import run_search_quality_option_ablation as module

        config = module.load_json(module.DEFAULT_CONFIG_PATH)
        with tempfile.TemporaryDirectory(
            prefix="azlite-search-quality-option-plan-"
        ) as tmp:
            with mock.patch.object(
                module,
                "resolve_step_command",
                side_effect=lambda command, resolved_repo_root: command,
            ):
                plans = module.build_variant_plan(
                    config,
                    candidate_path="/tmp/candidate-artifact",
                    current_path="storage/ai/alphazero_lite/current",
                    report_dir=Path(tmp),
                )

        for spec in module.VARIANT_SPECS:
            plan = plans[spec["name"]]
            self.assertEqual("/tmp/candidate-artifact", plan["candidate_path"])
            self.assertEqual("storage/ai/alphazero_lite/current", plan["current_path"])
            self.assertEqual(
                "/tmp/candidate-artifact",
                module.command_flag_value(plan["arena_command"], "--challenger"),
            )
            self.assertEqual(
                "storage/ai/alphazero_lite/current",
                module.command_flag_value(plan["arena_command"], "--current"),
            )
            self.assertEqual(
                "120",
                module.command_flag_value(plan["arena_command"], "--games"),
            )
            self.assertEqual(
                "640",
                module.command_flag_value(
                    plan["arena_command"], "--challenger-simulations"
                ),
            )
            self.assertEqual(
                "256",
                module.command_flag_value(
                    plan["arena_command"], "--current-simulations"
                ),
            )
            self.assertEqual(
                "30",
                module.command_flag_value(plan["candidate_mcts_command"], "--games"),
            )
            self.assertEqual(
                "42",
                module.command_flag_value(plan["candidate_mcts_command"], "--seed"),
            )
            self.assertEqual(
                "640",
                module.command_flag_value(
                    plan["candidate_mcts_command"], "--az-base-simulations"
                ),
            )
            self.assertEqual(
                "1200",
                module.command_flag_value(
                    plan["candidate_mcts_command"], "--mcts-simulations"
                ),
            )
            self.assertEqual(
                "/tmp/candidate-artifact",
                module.command_flag_value(
                    plan["candidate_mcts_command"], "--challenger-path"
                ),
            )
            self.assertEqual(
                "storage/ai/alphazero_lite/current",
                module.command_flag_value(
                    plan["current_mcts_command"], "--challenger-path"
                ),
            )

    def test_variant_commands_only_change_requested_search_flags(self):
        from ml.alphazero_lite import run_search_quality_option_ablation as module

        config = module.load_json(module.DEFAULT_CONFIG_PATH)
        with tempfile.TemporaryDirectory(
            prefix="azlite-search-quality-option-flags-"
        ) as tmp:
            with mock.patch.object(
                module,
                "resolve_step_command",
                side_effect=lambda command, resolved_repo_root: command,
            ):
                plans = module.build_variant_plan(
                    config,
                    candidate_path="/tmp/candidate-artifact",
                    current_path="storage/ai/alphazero_lite/current",
                    report_dir=Path(tmp),
                )

        parent_q_command = plans["parent_q_only"]["arena_command"]
        self.assertEqual(
            "parent_q", module.command_flag_value(parent_q_command, "--fpu-mode")
        )
        self.assertFalse(module.command_has_flag(parent_q_command, "--reuse-subtree"))
        self.assertFalse(
            module.command_has_flag(parent_q_command, "--normalize-values")
        )
        self.assertEqual(
            "deterministic",
            module.command_flag_value(parent_q_command, "--root-policy-mode"),
        )
        self.assertEqual(
            "0.1",
            module.command_flag_value(parent_q_command, "--tactical-root-bias"),
        )

        normalize_command = plans["normalize_values_only"]["arena_command"]
        self.assertEqual(
            "zero", module.command_flag_value(normalize_command, "--fpu-mode")
        )
        self.assertTrue(
            module.command_has_flag(normalize_command, "--normalize-values")
        )
        self.assertFalse(module.command_has_flag(normalize_command, "--reuse-subtree"))

        reuse_command = plans["reuse_subtree_only"]["arena_command"]
        self.assertEqual("zero", module.command_flag_value(reuse_command, "--fpu-mode"))
        self.assertTrue(module.command_has_flag(reuse_command, "--reuse-subtree"))
        self.assertFalse(module.command_has_flag(reuse_command, "--normalize-values"))

        tactical_command = plans["tactical_root_bias_only"]["arena_command"]
        self.assertEqual(
            "zero", module.command_flag_value(tactical_command, "--fpu-mode")
        )
        self.assertFalse(module.command_has_flag(tactical_command, "--reuse-subtree"))
        self.assertFalse(
            module.command_has_flag(tactical_command, "--normalize-values")
        )
        self.assertEqual(
            "0.1",
            module.command_flag_value(tactical_command, "--tactical-root-bias"),
        )

        all_flags_command = plans["all_search_quality_flags"]["arena_command"]
        self.assertEqual(
            "parent_q", module.command_flag_value(all_flags_command, "--fpu-mode")
        )
        self.assertTrue(module.command_has_flag(all_flags_command, "--reuse-subtree"))
        self.assertTrue(
            module.command_has_flag(all_flags_command, "--normalize-values")
        )

    def test_render_markdown_summary_lists_all_requested_columns(self):
        from ml.alphazero_lite import run_search_quality_option_ablation as module

        entries = [
            {
                "variant": "parent_q_only",
                "label": "parent_q only",
                "flags_enabled": ["--fpu-mode parent_q"],
                "arena_score": None,
                "mcts1200_score": None,
                "runtime_seconds": None,
                "pass": None,
                "notes": "",
            }
        ]

        markdown = module.render_markdown_summary(
            entries, config_path=module.DEFAULT_CONFIG_PATH
        )

        self.assertIn("# Search-Quality Option Ablation Matrix", markdown)
        self.assertIn(
            "| Variant | Flags enabled | Arena score | MCTS1200 score | Runtime | Pass/fail | Notes |",
            markdown,
        )
        self.assertIn(
            "| parent_q only | --fpu-mode parent_q | TBD | TBD | TBD | TBD |  |",
            markdown,
        )

    def test_main_runs_all_variants_against_stub_evaluators(self):
        runner = Path(__file__).with_name("run_search_quality_option_ablation.py")

        with tempfile.TemporaryDirectory(
            prefix="azlite-search-quality-option-live-"
        ) as tmp:
            out_path = Path(tmp) / "summary.json"
            report_dir = Path(tmp) / "reports"
            env = os.environ.copy()
            env["AZLITE_ARENA_STUB"] = "1"
            env["AZLITE_MCTS1200_BASELINE_STUB"] = "1"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    "--report-dir",
                    str(report_dir),
                    "--out",
                    str(out_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            payload = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual("search_quality_option_ablation_v1", payload["schema"])
        self.assertEqual(5, len(payload["variants"]))
        for entry in payload["variants"]:
            self.assertEqual(0.8, entry["arena_score"])
            self.assertEqual(0.5, entry["mcts1200_score"])
            self.assertTrue(entry["pass"])
            self.assertIn("current_mcts1200_score=0.5000", entry["notes"])

    def test_checked_in_markdown_template_lists_all_variants(self):
        doc = (
            Path(__file__).resolve().parents[2]
            / "docs/alphazero-lite-search-quality-option-ablation.md"
        ).read_text(encoding="utf-8")

        self.assertIn("# Search-Quality Option Ablation Matrix", doc)
        self.assertIn("parent_q only", doc)
        self.assertIn("normalize-values only", doc)
        self.assertIn("reuse-subtree only", doc)
        self.assertIn("tactical-root-bias only", doc)
        self.assertIn("all search-quality flags", doc)


if __name__ == "__main__":
    unittest.main()
