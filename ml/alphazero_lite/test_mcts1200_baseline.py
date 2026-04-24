import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class MCTS1200BaselineScriptTest(unittest.TestCase):
    def executable_python(self) -> str:
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / ".venv/bin/python",
            repo_root.parents[1] / ".venv/bin/python",
        ]
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        return sys.executable

    def test_parse_args_uses_search_defaults_without_flags(self):
        from ml.alphazero_lite.mcts1200_baseline import parse_args

        original_argv = list(sys.argv)
        try:
            sys.argv = [
                "mcts1200_baseline.py",
                "--challenger-path",
                "/tmp/challenger",
                "--out",
                "/tmp/report.json",
            ]
            args = parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual("visit_count", args.root_policy_mode)
        self.assertEqual(0.0, args.tactical_root_bias)

    def test_parse_args_clamps_exact_solve_threshold_to_tablebase_limit(self):
        from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
        from ml.alphazero_lite.mcts1200_baseline import parse_args

        original_argv = list(sys.argv)
        try:
            sys.argv = [
                "mcts1200_baseline.py",
                "--challenger-path",
                "/tmp/challenger",
                "--out",
                "/tmp/report.json",
                "--exact-solve-enabled",
                "--exact-solve-stone-threshold",
                "99",
            ]
            args = parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual(EndgameTablebase.MAX_SOLVED_SEEDS, args.exact_solve_stone_threshold)

    def test_parse_args_requires_threshold_when_exact_solve_enabled(self):
        from ml.alphazero_lite.mcts1200_baseline import parse_args

        original_argv = list(sys.argv)
        try:
            sys.argv = [
                "mcts1200_baseline.py",
                "--challenger-path",
                "/tmp/challenger",
                "--out",
                "/tmp/report.json",
                "--exact-solve-enabled",
            ]
            with self.assertRaises(SystemExit):
                parse_args()
        finally:
            sys.argv = original_argv

    def test_parse_args_requires_exact_solve_enabled_when_threshold_provided(self):
        from ml.alphazero_lite.mcts1200_baseline import parse_args

        original_argv = list(sys.argv)
        try:
            sys.argv = [
                "mcts1200_baseline.py",
                "--challenger-path",
                "/tmp/challenger",
                "--out",
                "/tmp/report.json",
                "--exact-solve-stone-threshold",
                "8",
            ]
            with self.assertRaises(SystemExit):
                parse_args()
        finally:
            sys.argv = original_argv

    def test_parse_args_rejects_negative_exact_solve_threshold(self):
        from ml.alphazero_lite.mcts1200_baseline import parse_args

        original_argv = list(sys.argv)
        try:
            sys.argv = [
                "mcts1200_baseline.py",
                "--challenger-path",
                "/tmp/challenger",
                "--out",
                "/tmp/report.json",
                "--exact-solve-enabled",
                "--exact-solve-stone-threshold",
                "-1",
            ]
            with self.assertRaises(SystemExit):
                parse_args()
        finally:
            sys.argv = original_argv

    def test_parse_args_rejects_non_positive_probe_when_dynamic_budget_enabled(self):
        from ml.alphazero_lite.mcts1200_baseline import parse_args

        original_argv = list(sys.argv)
        try:
            sys.argv = [
                "mcts1200_baseline.py",
                "--challenger-path",
                "/tmp/challenger",
                "--out",
                "/tmp/report.json",
                "--dynamic-budget-enabled",
                "--dynamic-budget-probe-simulations",
                "0",
            ]
            with self.assertRaises(SystemExit):
                parse_args()
        finally:
            sys.argv = original_argv

    def test_parse_args_rejects_probe_greater_than_or_equal_to_max_when_dynamic_budget_enabled(self):
        from ml.alphazero_lite.mcts1200_baseline import parse_args

        original_argv = list(sys.argv)
        try:
            sys.argv = [
                "mcts1200_baseline.py",
                "--challenger-path",
                "/tmp/challenger",
                "--out",
                "/tmp/report.json",
                "--dynamic-budget-enabled",
                "--dynamic-budget-probe-simulations",
                "16",
                "--dynamic-budget-max-simulations",
                "16",
            ]
            with self.assertRaises(SystemExit):
                parse_args()
        finally:
            sys.argv = original_argv

    def test_parse_args_rejects_max_less_than_min_when_dynamic_budget_enabled(self):
        from ml.alphazero_lite.mcts1200_baseline import parse_args

        original_argv = list(sys.argv)
        try:
            sys.argv = [
                "mcts1200_baseline.py",
                "--challenger-path",
                "/tmp/challenger",
                "--out",
                "/tmp/report.json",
                "--dynamic-budget-enabled",
                "--dynamic-budget-probe-simulations",
                "8",
                "--dynamic-budget-min-simulations",
                "20",
                "--dynamic-budget-max-simulations",
                "12",
            ]
            with self.assertRaises(SystemExit):
                parse_args()
        finally:
            sys.argv = original_argv

    def test_parse_args_rejects_non_finite_dynamic_budget_weight_when_enabled(self):
        from ml.alphazero_lite.mcts1200_baseline import parse_args

        original_argv = list(sys.argv)
        try:
            sys.argv = [
                "mcts1200_baseline.py",
                "--challenger-path",
                "/tmp/challenger",
                "--out",
                "/tmp/report.json",
                "--dynamic-budget-enabled",
                "--dynamic-budget-probe-simulations",
                "8",
                "--dynamic-budget-entropy-weight",
                "inf",
            ]
            with self.assertRaises(SystemExit):
                parse_args()
        finally:
            sys.argv = original_argv

    def test_parse_args_rejects_out_of_range_low_margin_threshold_when_enabled(self):
        from ml.alphazero_lite.mcts1200_baseline import parse_args

        original_argv = list(sys.argv)
        try:
            sys.argv = [
                "mcts1200_baseline.py",
                "--challenger-path",
                "/tmp/challenger",
                "--out",
                "/tmp/report.json",
                "--dynamic-budget-enabled",
                "--dynamic-budget-probe-simulations",
                "8",
                "--dynamic-budget-low-margin-threshold",
                "1.2",
            ]
            with self.assertRaises(SystemExit):
                parse_args()
        finally:
            sys.argv = original_argv

    def test_parse_args_rejects_negative_dynamic_budget_weight_when_enabled(self):
        from ml.alphazero_lite.mcts1200_baseline import parse_args

        original_argv = list(sys.argv)
        try:
            sys.argv = [
                "mcts1200_baseline.py",
                "--challenger-path",
                "/tmp/challenger",
                "--out",
                "/tmp/report.json",
                "--dynamic-budget-enabled",
                "--dynamic-budget-probe-simulations",
                "8",
                "--dynamic-budget-variance-weight",
                "-0.1",
            ]
            with self.assertRaises(SystemExit):
                parse_args()
        finally:
            sys.argv = original_argv

    def test_cli_writes_mcts1200_report_with_expected_schema(self):
        with tempfile.TemporaryDirectory(prefix="azlite-mcts1200-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "mcts1200_report.json"

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/mcts1200_baseline.py",
                    "--challenger-path",
                    str(tmp_path / "challenger"),
                    "--games",
                    "30",
                    "--seed",
                    "42",
                    "--az-base-simulations",
                    "640",
                    "--mcts-simulations",
                    "1200",
                    "--workers",
                    "6",
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "AZLITE_MCTS1200_BASELINE_STUB": "1"},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {
                    "schema",
                    "classic_mcts_mode",
                    "games",
                    "az_base_simulations",
                    "mcts_simulations",
                    "search_option_notes",
                    "search_profile",
                    "search_profile_hash",
                    "classic_mcts_dynamic_budget_config",
                    "az_wins",
                    "mcts_wins",
                    "draws",
                    "score",
                    "budget_summary",
                },
                set(report.keys()),
            )
            self.assertEqual("azlite_vs_mcts_v1", report["schema"])
            self.assertEqual(30, report["games"])
            self.assertEqual(640, report["az_base_simulations"])
            self.assertEqual(1200, report["mcts_simulations"])
            self.assertEqual("v1", report["search_profile"]["version"])
            self.assertEqual("mcts1200_baseline_eval", report["search_profile"]["kind"])
            self.assertEqual(report["search_profile"]["hash"], report["search_profile_hash"])
            self.assertNotIn("comparison_mode", report)
            self.assertNotIn("dynamic_budget_comparison", report)
            self.assertEqual("fixed_classic_mcts", report["search_profile"]["simulation_budget_policy"])
            self.assertEqual(1200, report["search_profile"]["simulation_budget_min"])
            self.assertEqual(1200, report["search_profile"]["simulation_budget_max"])
            self.assertEqual("fixed:constant", report["search_profile"]["simulation_budget_multipliers"])
            self.assertIn("search_option_notes", report)
            self.assertIn("ignored by ClassicMCTS", report["search_option_notes"])
            self.assertIn("mean_final_simulations", report["budget_summary"])
            self.assertIn("mean_root_latency_ms", report["budget_summary"])

    def test_partitioning_preserves_global_game_indexes(self):
        from ml.alphazero_lite.mcts1200_baseline import partition_counts, partition_starts

        self.assertEqual([5, 5, 5, 5, 5, 5], partition_counts(30, 6))
        self.assertEqual([0, 5, 10, 15, 20, 25], partition_starts([5, 5, 5, 5, 5, 5]))

        with tempfile.TemporaryDirectory(prefix="azlite-mcts1200-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "mcts1200_report.json"

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/mcts1200_baseline.py",
                    "--challenger-path",
                    str(tmp_path / "challenger"),
                    "--games",
                    "6",
                    "--seed",
                    "42",
                    "--az-base-simulations",
                    "640",
                    "--mcts-simulations",
                    "1200",
                    "--workers",
                    "4",
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "AZLITE_MCTS1200_BASELINE_STUB": "1"},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(6, report["games"])
            self.assertEqual(3, report["az_wins"])
            self.assertEqual(2, report["mcts_wins"])
            self.assertEqual(1, report["draws"])
            self.assertEqual(0.5833, report["score"])

    def test_build_report_includes_budget_summary_from_worker_results(self):
        from ml.alphazero_lite.mcts1200_baseline import build_report

        report = build_report(
            games=4,
            az_base_simulations=640,
            mcts_simulations=1200,
            search_options={
                "fpu_mode": "zero",
                "reuse_subtree": False,
                "normalize_values": False,
                "root_policy_mode": "deterministic",
                "tactical_root_bias": 0.1,
            },
            results=[
                {
                    "dynamic": {
                        "az_wins": 2,
                        "mcts_wins": 1,
                        "draws": 1,
                        "budget_summary": {
                            "mean_final_simulations": 128.0,
                            "mean_root_latency_ms": 6.5,
                        },
                    },
                    "fixed": {
                        "az_wins": 1,
                        "mcts_wins": 2,
                        "draws": 1,
                        "budget_summary": {
                            "mean_final_simulations": 96.0,
                            "mean_root_latency_ms": 6.3,
                        },
                    },
                }
            ],
            exact_solve_enabled=False,
            exact_solve_stone_threshold=None,
        )

        self.assertEqual(96.0, report["budget_summary"]["mean_final_simulations"])
        self.assertEqual(6.3, report["budget_summary"]["mean_root_latency_ms"])

    def test_build_report_fallback_averages_latency_without_final_simulations(self):
        from ml.alphazero_lite.mcts1200_baseline import build_report

        report = build_report(
            games=4,
            az_base_simulations=640,
            mcts_simulations=1200,
            search_options={
                "fpu_mode": "zero",
                "reuse_subtree": False,
                "normalize_values": False,
                "root_policy_mode": "deterministic",
                "tactical_root_bias": 0.1,
            },
            results=[
                {
                    "dynamic": {
                        "az_wins": 2,
                        "mcts_wins": 1,
                        "draws": 1,
                        "budget_summary": {
                            "mean_root_latency_ms": 6.5,
                        },
                    },
                    "fixed": {
                        "az_wins": 1,
                        "mcts_wins": 2,
                        "draws": 1,
                        "budget_summary": {
                            "mean_root_latency_ms": 6.3,
                        },
                    },
                }
            ],
            exact_solve_enabled=False,
            exact_solve_stone_threshold=None,
        )

        self.assertIsNone(report["budget_summary"]["mean_final_simulations"])
        self.assertEqual(6.3, report["budget_summary"]["mean_root_latency_ms"])

    def test_build_report_weights_fallback_budget_summary_by_worker_games(self):
        from ml.alphazero_lite.mcts1200_baseline import build_report

        report = build_report(
            games=5,
            az_base_simulations=640,
            mcts_simulations=1200,
            search_options={
                "fpu_mode": "zero",
                "reuse_subtree": False,
                "normalize_values": False,
                "root_policy_mode": "deterministic",
                "tactical_root_bias": 0.1,
            },
            results=[
                {
                    "games": 3,
                    "dynamic": {"games": 3, "az_wins": 1, "mcts_wins": 1, "draws": 1},
                    "fixed": {
                        "games": 3,
                        "az_wins": 2,
                        "mcts_wins": 1,
                        "draws": 0,
                        "budget_summary": {
                            "mean_final_simulations": 120.0,
                            "mean_root_latency_ms": 8.0,
                        },
                    },
                },
                {
                    "games": 2,
                    "dynamic": {"games": 2, "az_wins": 0, "mcts_wins": 2, "draws": 0},
                    "fixed": {
                        "games": 2,
                        "az_wins": 1,
                        "mcts_wins": 1,
                        "draws": 0,
                        "budget_summary": {
                            "mean_final_simulations": 60.0,
                            "mean_root_latency_ms": 4.0,
                        },
                    },
                },
            ],
            exact_solve_enabled=False,
            exact_solve_stone_threshold=None,
        )

        self.assertEqual(96.0, report["budget_summary"]["mean_final_simulations"])
        self.assertEqual(6.4, report["budget_summary"]["mean_root_latency_ms"])

    def test_build_report_marks_budget_summary_source_and_preserves_classic_mcts_dynamic_budget_config(self):
        from ml.alphazero_lite.mcts1200_baseline import build_report

        report = build_report(
            games=4,
            az_base_simulations=640,
            mcts_simulations=1200,
            search_options={
                "fpu_mode": "zero",
                "reuse_subtree": False,
                "normalize_values": False,
                "root_policy_mode": "deterministic",
                "tactical_root_bias": 0.1,
            },
            dynamic_budget_enabled=True,
            dynamic_budget_probe_simulations=12,
            dynamic_budget_min_simulations=24,
            dynamic_budget_max_simulations=96,
            dynamic_budget_entropy_weight=0.75,
            dynamic_budget_low_margin_threshold=0.18,
            dynamic_budget_low_margin_weight=1.25,
            dynamic_budget_variance_weight=1.1,
            results=[
                {
                    "dynamic": {
                        "az_wins": 2,
                        "mcts_wins": 1,
                        "draws": 1,
                        "budget_sample_count": 2,
                        "budget_total_final_simulations": 256.0,
                        "budget_total_root_latency_ms": 13.0,
                    },
                    "fixed": {
                        "az_wins": 3,
                        "mcts_wins": 1,
                        "draws": 0,
                        "budget_sample_count": 2,
                        "budget_total_final_simulations": 192.0,
                        "budget_total_root_latency_ms": 12.6,
                    },
                }
            ],
            exact_solve_enabled=False,
            exact_solve_stone_threshold=None,
        )

        self.assertEqual("classic_mcts_dynamic_runtime", report["budget_summary"]["source"])
        self.assertEqual("classic_dynamic_vs_fixed", report["comparison_mode"])
        self.assertEqual("dynamic", report["classic_mcts_mode"])
        self.assertEqual(
            {
                "enabled": True,
                "probe_simulations": 12,
                "min_simulations": 24,
                "max_simulations": 96,
                "entropy_weight": 0.75,
                "low_margin_threshold": 0.18,
                "low_margin_weight": 1.25,
                "variance_weight": 1.1,
            },
            report["classic_mcts_dynamic_budget_config"],
        )
        self.assertNotIn("dynamic_budget_config", report)
        self.assertEqual(
            {
                "comparison_mode": "classic_dynamic_vs_fixed",
                "runtime_target_ms": 6.3,
                "runtime_target_matched": True,
                "seat_bias_neutralized": True,
                "dynamic_mean_final_simulations": 128.0,
                "dynamic_mean_root_latency_ms": 6.5,
                "fixed_mean_final_simulations": 96.0,
                "fixed_mean_root_latency_ms": 6.3,
                "dynamic_score": 0.625,
                "fixed_score": 0.75,
            },
            report["dynamic_budget_comparison"],
        )

    def test_build_report_marks_fixed_classic_mcts_runtime_for_runtime_matched_comparison(self):
        from ml.alphazero_lite.mcts1200_baseline import build_report

        report = build_report(
            games=4,
            az_base_simulations=640,
            mcts_simulations=1200,
            search_options={
                "fpu_mode": "zero",
                "reuse_subtree": False,
                "normalize_values": False,
                "root_policy_mode": "deterministic",
                "tactical_root_bias": 0.1,
            },
            dynamic_budget_enabled=False,
            results=[
                {
                    "dynamic": {
                        "az_wins": 2,
                        "mcts_wins": 1,
                        "draws": 1,
                        "budget_sample_count": 2,
                        "budget_total_final_simulations": 256.0,
                        "budget_total_root_latency_ms": 13.0,
                    },
                    "fixed": {
                        "az_wins": 3,
                        "mcts_wins": 1,
                        "draws": 0,
                        "budget_sample_count": 2,
                        "budget_total_final_simulations": 192.0,
                        "budget_total_root_latency_ms": 12.6,
                    },
                }
            ],
            exact_solve_enabled=False,
            exact_solve_stone_threshold=None,
        )

        self.assertEqual("fixed", report["classic_mcts_mode"])
        self.assertEqual("classic_mcts_fixed_runtime", report["budget_summary"]["source"])
        self.assertEqual(96.0, report["budget_summary"]["mean_final_simulations"])
        self.assertEqual(6.3, report["budget_summary"]["mean_root_latency_ms"])
        self.assertNotIn("comparison_mode", report)
        self.assertNotIn("dynamic_budget_comparison", report)

    def test_build_report_does_not_label_fixed_only_run_as_dynamic_vs_fixed_comparison(self):
        from ml.alphazero_lite.mcts1200_baseline import build_report

        report = build_report(
            games=4,
            az_base_simulations=640,
            mcts_simulations=1200,
            search_options={
                "fpu_mode": "zero",
                "reuse_subtree": False,
                "normalize_values": False,
                "root_policy_mode": "deterministic",
                "tactical_root_bias": 0.1,
            },
            dynamic_budget_enabled=False,
            results=[
                {
                    "dynamic": {
                        "az_wins": 2,
                        "mcts_wins": 1,
                        "draws": 1,
                        "budget_sample_count": 2,
                        "budget_total_final_simulations": 256.0,
                        "budget_total_root_latency_ms": 13.0,
                    },
                    "fixed": {
                        "az_wins": 3,
                        "mcts_wins": 1,
                        "draws": 0,
                        "budget_sample_count": 2,
                        "budget_total_final_simulations": 192.0,
                        "budget_total_root_latency_ms": 12.6,
                    },
                }
            ],
            exact_solve_enabled=False,
            exact_solve_stone_threshold=None,
        )

        self.assertEqual("fixed", report["classic_mcts_mode"])
        self.assertNotEqual("classic_dynamic_vs_fixed", report.get("comparison_mode"))
        self.assertNotIn("dynamic_budget_comparison", report)

    def test_build_report_collapses_fixed_only_budget_bounds_to_mcts_simulations(self):
        from ml.alphazero_lite.mcts1200_baseline import build_report

        report = build_report(
            games=4,
            az_base_simulations=640,
            mcts_simulations=1200,
            search_options={
                "fpu_mode": "zero",
                "reuse_subtree": False,
                "normalize_values": False,
                "root_policy_mode": "deterministic",
                "tactical_root_bias": 0.1,
            },
            dynamic_budget_enabled=False,
            dynamic_budget_probe_simulations=12,
            dynamic_budget_min_simulations=24,
            dynamic_budget_max_simulations=96,
            results=[
                {
                    "dynamic": {
                        "az_wins": 2,
                        "mcts_wins": 1,
                        "draws": 1,
                        "budget_sample_count": 2,
                        "budget_total_final_simulations": 256.0,
                        "budget_total_root_latency_ms": 13.0,
                    },
                    "fixed": {
                        "az_wins": 3,
                        "mcts_wins": 1,
                        "draws": 0,
                        "budget_sample_count": 2,
                        "budget_total_final_simulations": 192.0,
                        "budget_total_root_latency_ms": 12.6,
                    },
                }
            ],
            exact_solve_enabled=False,
            exact_solve_stone_threshold=None,
        )

        self.assertEqual(1200, report["classic_mcts_dynamic_budget_config"]["min_simulations"])
        self.assertEqual(1200, report["classic_mcts_dynamic_budget_config"]["max_simulations"])
        self.assertEqual(1200, report["search_profile"]["simulation_budget_min"])
        self.assertEqual(1200, report["search_profile"]["simulation_budget_max"])
        self.assertEqual(1200, report["search_profile"]["dynamic_budget_min_simulations"])
        self.assertEqual(1200, report["search_profile"]["dynamic_budget_max_simulations"])

    def test_build_report_emits_canonical_fixed_only_dynamic_budget_config_metadata(self):
        from ml.alphazero_lite.mcts1200_baseline import build_report

        report = build_report(
            games=4,
            az_base_simulations=640,
            mcts_simulations=1200,
            search_options={
                "fpu_mode": "zero",
                "reuse_subtree": False,
                "normalize_values": False,
                "root_policy_mode": "deterministic",
                "tactical_root_bias": 0.1,
            },
            dynamic_budget_enabled=False,
            dynamic_budget_probe_simulations=12,
            dynamic_budget_min_simulations=24,
            dynamic_budget_max_simulations=96,
            dynamic_budget_entropy_weight=0.75,
            dynamic_budget_low_margin_threshold=0.18,
            dynamic_budget_low_margin_weight=1.25,
            dynamic_budget_variance_weight=1.1,
            results=[
                {
                    "dynamic": {
                        "az_wins": 2,
                        "mcts_wins": 1,
                        "draws": 1,
                        "budget_sample_count": 2,
                        "budget_total_final_simulations": 256.0,
                        "budget_total_root_latency_ms": 13.0,
                    },
                    "fixed": {
                        "az_wins": 3,
                        "mcts_wins": 1,
                        "draws": 0,
                        "budget_sample_count": 2,
                        "budget_total_final_simulations": 192.0,
                        "budget_total_root_latency_ms": 12.6,
                    },
                }
            ],
            exact_solve_enabled=False,
            exact_solve_stone_threshold=None,
        )

        self.assertEqual(
            {
                "enabled": False,
                "probe_simulations": 0,
                "min_simulations": 1200,
                "max_simulations": 1200,
                "entropy_weight": 0.0,
                "low_margin_threshold": 0.0,
                "low_margin_weight": 0.0,
                "variance_weight": 0.0,
            },
            report["classic_mcts_dynamic_budget_config"],
        )

    def test_build_report_normalizes_effective_dynamic_budget_min_max_metadata(self):
        from ml.alphazero_lite.mcts1200_baseline import build_report

        report = build_report(
            games=4,
            az_base_simulations=640,
            mcts_simulations=1200,
            search_options={
                "fpu_mode": "zero",
                "reuse_subtree": False,
                "normalize_values": False,
                "root_policy_mode": "deterministic",
                "tactical_root_bias": 0.1,
            },
            dynamic_budget_enabled=True,
            dynamic_budget_probe_simulations=12,
            dynamic_budget_min_simulations=None,
            dynamic_budget_max_simulations=None,
            results=[
                {
                    "dynamic": {
                        "az_wins": 2,
                        "mcts_wins": 1,
                        "draws": 1,
                        "budget_sample_count": 2,
                        "budget_total_final_simulations": 256.0,
                        "budget_total_root_latency_ms": 13.0,
                    },
                    "fixed": {
                        "az_wins": 3,
                        "mcts_wins": 1,
                        "draws": 0,
                        "budget_sample_count": 2,
                        "budget_total_final_simulations": 192.0,
                        "budget_total_root_latency_ms": 12.6,
                    },
                }
            ],
            exact_solve_enabled=False,
            exact_solve_stone_threshold=None,
        )

        self.assertEqual(1200, report["classic_mcts_dynamic_budget_config"]["min_simulations"])
        self.assertEqual(1200, report["classic_mcts_dynamic_budget_config"]["max_simulations"])
        self.assertEqual(1200, report["search_profile"]["dynamic_budget_min_simulations"])
        self.assertEqual(1200, report["search_profile"]["dynamic_budget_max_simulations"])

    def test_stub_cli_budget_summary_uses_configured_mcts_simulations_for_fixed_mode(self):
        with tempfile.TemporaryDirectory(prefix="azlite-mcts1200-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "mcts1200_report.json"

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/mcts1200_baseline.py",
                    "--challenger-path",
                    str(tmp_path / "challenger"),
                    "--games",
                    "8",
                    "--seed",
                    "42",
                    "--az-base-simulations",
                    "320",
                    "--mcts-simulations",
                    "1200",
                    "--workers",
                    "2",
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "AZLITE_MCTS1200_BASELINE_STUB": "1"},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(1200.0, report["budget_summary"]["mean_final_simulations"])

    def test_stub_cli_budget_summary_uses_configured_mcts_simulations_for_dynamic_mode(self):
        with tempfile.TemporaryDirectory(prefix="azlite-mcts1200-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "mcts1200_report.json"

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/mcts1200_baseline.py",
                    "--challenger-path",
                    str(tmp_path / "challenger"),
                    "--games",
                    "8",
                    "--seed",
                    "42",
                    "--az-base-simulations",
                    "320",
                    "--mcts-simulations",
                    "700",
                    "--dynamic-budget-enabled",
                    "--dynamic-budget-probe-simulations",
                    "8",
                    "--workers",
                    "2",
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "AZLITE_MCTS1200_BASELINE_STUB": "1"},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(700.0, report["dynamic_budget_comparison"]["fixed_mean_final_simulations"])

    def test_cli_rejects_missing_challenger_path(self):
        with tempfile.TemporaryDirectory(prefix="azlite-mcts1200-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "mcts1200_report.json"
            missing_path = tmp_path / "missing-challenger"

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/mcts1200_baseline.py",
                    "--challenger-path",
                    str(missing_path),
                    "--games",
                    "8",
                    "--seed",
                    "42",
                    "--az-base-simulations",
                    "320",
                    "--mcts-simulations",
                    "700",
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("challenger path not found", result.stderr)

    def test_run_worker_collects_actual_fixed_and_dynamic_classic_mcts_budget_metrics(self):
        from ml.alphazero_lite import mcts1200_baseline
        from ml.alphazero_lite.mcts1200_baseline import run_worker

        constructed = []
        player_order = []

        class FakeGame:
            def __init__(self):
                self.current_player = 1
                self.captured_seeds = [0, 0]
                self._moves = 0

            def over(self):
                return self._moves >= 2

            def pit_index(self, move):
                return move

            def move(self, pit_index):
                del pit_index
                self._moves += 1
                self.current_player = 1 - self.current_player
                if self._moves >= 2:
                    self.captured_seeds = [4, 2]
                return True

        class FakeMCTS:
            def __init__(self, game, **kwargs):
                player_order.append(game.current_player)
                constructed.append(kwargs)
                self._dynamic = bool(kwargs["dynamic_budget_enabled"])

            def choose_move(self):
                return 0

            def root_summary(self):
                if self._dynamic:
                    return {
                        "selected_move": 0,
                        "budget": {"final_simulations": 33, "root_latency_ms": 7.5},
                    }
                return {
                    "selected_move": 0,
                    "budget": {"final_simulations": 20, "root_latency_ms": 6.0},
                }

        with tempfile.TemporaryDirectory(prefix="azlite-mcts1200-") as tmp:
            artifact_dir = Path(tmp) / "artifact"
            artifact_dir.mkdir()
            (artifact_dir / "metadata.json").write_text(
                json.dumps({"input_encoding": "kalah_v1", "architecture": {"model_type": "mlp_v1"}}),
                encoding="utf-8",
            )
            (artifact_dir / "weights.json").write_text(
                json.dumps(
                    {
                        "w1": [[0.0] * 16 for _ in range(15)],
                        "b1": [0.0] * 16,
                        "w2": [[0.0] * 16 for _ in range(16)],
                        "b2": [0.0] * 16,
                        "w_policy": [[0.0] * 6 for _ in range(16)],
                        "b_policy": [0.0] * 6,
                        "w_value": [[0.0] for _ in range(16)],
                        "b_value": [0.0],
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(mcts1200_baseline, "MCTS", FakeMCTS), mock.patch.object(
                mcts1200_baseline,
                "initial_game",
                side_effect=lambda: FakeGame(),
            ):
                result = run_worker(
                    challenger_path=str(artifact_dir),
                    games=1,
                    start_index=0,
                    seed=42,
                    az_base_simulations=64,
                    mcts_simulations=64,
                    search_options={
                        "fpu_mode": "zero",
                        "reuse_subtree": False,
                        "normalize_values": False,
                        "root_policy_mode": "deterministic",
                        "tactical_root_bias": 0.1,
                    },
                    dynamic_budget_enabled=True,
                    dynamic_budget_probe_simulations=12,
                    dynamic_budget_min_simulations=24,
                    dynamic_budget_max_simulations=96,
                )

        self.assertEqual(2, len(constructed))
        self.assertFalse(constructed[0]["dynamic_budget_enabled"])
        self.assertTrue(constructed[1]["dynamic_budget_enabled"])
        self.assertEqual([1, 0], player_order)
        self.assertEqual(33.0, result["dynamic"]["budget_total_final_simulations"])
        self.assertEqual(7.5, result["dynamic"]["budget_total_root_latency_ms"])
        self.assertEqual(20.0, result["fixed"]["budget_total_final_simulations"])
        self.assertEqual(6.0, result["fixed"]["budget_total_root_latency_ms"])

    def test_run_worker_alternates_seats_between_dynamic_and_fixed_classic_mcts(self):
        from ml.alphazero_lite import mcts1200_baseline
        from ml.alphazero_lite.mcts1200_baseline import run_worker

        game_players = []

        class FakeGame:
            def __init__(self):
                self.current_player = 0
                self.captured_seeds = [0, 0]
                self._moves = 0

            def over(self):
                return self._moves >= 2

            def pit_index(self, move):
                return move

            def move(self, pit_index):
                del pit_index
                self._moves += 1
                self.current_player = 1 - self.current_player
                if self._moves >= 2:
                    self.captured_seeds = [5, 3]
                return True

        class FakeMCTS:
            def __init__(self, game, **kwargs):
                game_players.append((bool(kwargs["dynamic_budget_enabled"]), game.current_player))

            def root_summary(self):
                return {
                    "selected_move": 0,
                    "budget": {"final_simulations": 12, "root_latency_ms": 1.0},
                }

        with tempfile.TemporaryDirectory(prefix="azlite-mcts1200-") as tmp:
            artifact_dir = Path(tmp) / "artifact"
            artifact_dir.mkdir()
            (artifact_dir / "metadata.json").write_text(
                json.dumps({"input_encoding": "kalah_v1", "architecture": {"model_type": "mlp_v1"}}),
                encoding="utf-8",
            )
            (artifact_dir / "weights.json").write_text(
                json.dumps(
                    {
                        "w1": [[0.0] * 16 for _ in range(15)],
                        "b1": [0.0] * 16,
                        "w2": [[0.0] * 16 for _ in range(16)],
                        "b2": [0.0] * 16,
                        "w_policy": [[0.0] * 6 for _ in range(16)],
                        "b_policy": [0.0] * 6,
                        "w_value": [[0.0] for _ in range(16)],
                        "b_value": [0.0],
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(mcts1200_baseline, "MCTS", FakeMCTS), mock.patch.object(
                mcts1200_baseline,
                "initial_game",
                side_effect=lambda: FakeGame(),
            ):
                run_worker(
                    challenger_path=str(artifact_dir),
                    games=2,
                    start_index=0,
                    seed=42,
                    az_base_simulations=64,
                    mcts_simulations=64,
                    search_options={
                        "fpu_mode": "zero",
                        "reuse_subtree": False,
                        "normalize_values": False,
                        "root_policy_mode": "deterministic",
                        "tactical_root_bias": 0.1,
                    },
                    dynamic_budget_enabled=True,
                    dynamic_budget_probe_simulations=12,
                    dynamic_budget_min_simulations=24,
                    dynamic_budget_max_simulations=96,
                )

        self.assertEqual(
            [
                (True, 0),
                (False, 1),
                (False, 0),
                (True, 1),
            ],
            game_players,
        )

    def test_module_no_longer_exposes_ruby_worker_script(self):
        from ml.alphazero_lite import mcts1200_baseline

        self.assertFalse(hasattr(mcts1200_baseline, "ruby_worker_script"))

    def test_run_worker_stays_in_python(self):
        from ml.alphazero_lite import mcts1200_baseline
        from ml.alphazero_lite.mcts1200_baseline import run_worker

        with tempfile.TemporaryDirectory(prefix="azlite-mcts1200-") as tmp:
            artifact_dir = Path(tmp) / "artifact"
            artifact_dir.mkdir()
            (artifact_dir / "metadata.json").write_text(
                json.dumps({"input_encoding": "kalah_v1", "architecture": {"model_type": "mlp_v1"}}),
                encoding="utf-8",
            )
            (artifact_dir / "weights.json").write_text(
                json.dumps(
                    {
                        "w1": [[0.0] * 16 for _ in range(15)],
                        "b1": [0.0] * 16,
                        "w2": [[0.0] * 16 for _ in range(16)],
                        "b2": [0.0] * 16,
                        "w_policy": [[0.0] * 6 for _ in range(16)],
                        "b_policy": [0.0] * 6,
                        "w_value": [[0.0] for _ in range(16)],
                        "b_value": [0.0],
                    }
                ),
                encoding="utf-8",
            )

            self.assertFalse(hasattr(mcts1200_baseline, "subprocess"))
            result = run_worker(
                challenger_path=str(artifact_dir),
                games=1,
                start_index=0,
                seed=42,
                az_base_simulations=64,
                mcts_simulations=64,
                search_options={
                    "fpu_mode": "zero",
                    "reuse_subtree": False,
                    "normalize_values": False,
                    "root_policy_mode": "deterministic",
                    "tactical_root_bias": 0.1,
                },
            )

        self.assertEqual(1, result["games"])

    def test_run_worker_passes_tablebase_when_exact_solve_is_enabled(self):
        from ml.alphazero_lite import mcts1200_baseline
        from ml.alphazero_lite.mcts1200_baseline import run_worker

        constructed = []

        class FakeMCTS:
            def __init__(self, game, **kwargs):
                del game
                constructed.append(kwargs)

            def choose_move(self):
                return 0

        with tempfile.TemporaryDirectory(prefix="azlite-mcts1200-") as tmp:
            artifact_dir = Path(tmp) / "artifact"
            artifact_dir.mkdir()
            (artifact_dir / "metadata.json").write_text(
                json.dumps({"input_encoding": "kalah_v1", "architecture": {"model_type": "mlp_v1"}}),
                encoding="utf-8",
            )
            (artifact_dir / "weights.json").write_text(
                json.dumps(
                    {
                        "w1": [[0.0] * 16 for _ in range(15)],
                        "b1": [0.0] * 16,
                        "w2": [[0.0] * 16 for _ in range(16)],
                        "b2": [0.0] * 16,
                        "w_policy": [[0.0] * 6 for _ in range(16)],
                        "b_policy": [0.0] * 6,
                        "w_value": [[0.0] for _ in range(16)],
                        "b_value": [0.0],
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(mcts1200_baseline, "MCTS", FakeMCTS):
                run_worker(
                    challenger_path=str(artifact_dir),
                    games=1,
                    start_index=0,
                    seed=42,
                    az_base_simulations=64,
                    mcts_simulations=64,
                    search_options={
                        "fpu_mode": "zero",
                        "reuse_subtree": False,
                        "normalize_values": False,
                        "root_policy_mode": "deterministic",
                        "tactical_root_bias": 0.1,
                    },
                    exact_solve_enabled=True,
                    exact_solve_stone_threshold=10,
                )

        self.assertTrue(constructed)
        self.assertTrue(constructed[0]["exact_solve_enabled"])
        self.assertEqual(10, constructed[0]["exact_solve_stone_threshold"])
        self.assertIsNotNone(constructed[0]["endgame_tablebase"])

    def test_run_worker_passes_dynamic_budget_tuning_to_classic_mcts(self):
        from ml.alphazero_lite import mcts1200_baseline
        from ml.alphazero_lite.mcts1200_baseline import run_worker

        constructed = []

        class FakeGame:
            def __init__(self):
                self.current_player = 1
                self.captured_seeds = [0, 0]
                self._moves = 0

            def over(self):
                return self._moves >= 2

            def pit_index(self, move):
                return move

            def move(self, pit_index):
                del pit_index
                self._moves += 1
                self.current_player = 1 - self.current_player
                if self._moves >= 2:
                    self.captured_seeds = [3, 1]
                return True

        class FakeMCTS:
            def __init__(self, game, **kwargs):
                del game
                constructed.append(kwargs)

            def choose_move(self):
                return 0

        with tempfile.TemporaryDirectory(prefix="azlite-mcts1200-") as tmp:
            artifact_dir = Path(tmp) / "artifact"
            artifact_dir.mkdir()
            (artifact_dir / "metadata.json").write_text(
                json.dumps({"input_encoding": "kalah_v1", "architecture": {"model_type": "mlp_v1"}}),
                encoding="utf-8",
            )
            (artifact_dir / "weights.json").write_text(
                json.dumps(
                    {
                        "w1": [[0.0] * 16 for _ in range(15)],
                        "b1": [0.0] * 16,
                        "w2": [[0.0] * 16 for _ in range(16)],
                        "b2": [0.0] * 16,
                        "w_policy": [[0.0] * 6 for _ in range(16)],
                        "b_policy": [0.0] * 6,
                        "w_value": [[0.0] for _ in range(16)],
                        "b_value": [0.0],
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(mcts1200_baseline, "MCTS", FakeMCTS), mock.patch.object(
                mcts1200_baseline,
                "initial_game",
                side_effect=lambda: FakeGame(),
            ):
                run_worker(
                    challenger_path=str(artifact_dir),
                    games=1,
                    start_index=0,
                    seed=42,
                    az_base_simulations=64,
                    mcts_simulations=64,
                    search_options={
                        "fpu_mode": "zero",
                        "reuse_subtree": False,
                        "normalize_values": False,
                        "root_policy_mode": "deterministic",
                        "tactical_root_bias": 0.1,
                    },
                    dynamic_budget_enabled=True,
                    dynamic_budget_probe_simulations=12,
                    dynamic_budget_min_simulations=24,
                    dynamic_budget_max_simulations=96,
                    dynamic_budget_entropy_weight=0.75,
                    dynamic_budget_low_margin_threshold=0.18,
                    dynamic_budget_low_margin_weight=1.25,
                    dynamic_budget_variance_weight=1.1,
                )

        self.assertEqual(2, len(constructed))
        dynamic_kwargs = next(kwargs for kwargs in constructed if kwargs["dynamic_budget_enabled"])
        self.assertEqual(12, dynamic_kwargs["dynamic_budget_probe_simulations"])
        self.assertEqual(24, dynamic_kwargs["dynamic_budget_min_simulations"])
        self.assertEqual(96, dynamic_kwargs["dynamic_budget_max_simulations"])
        self.assertEqual(0.75, dynamic_kwargs["dynamic_budget_entropy_weight"])
        self.assertEqual(0.18, dynamic_kwargs["dynamic_budget_low_margin_threshold"])
        self.assertEqual(1.25, dynamic_kwargs["dynamic_budget_low_margin_weight"])
        self.assertEqual(1.1, dynamic_kwargs["dynamic_budget_variance_weight"])


if __name__ == "__main__":
    unittest.main()
