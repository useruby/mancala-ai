import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from ml.alphazero_lite.forensic_suite import summarize_bucket
from ml.alphazero_lite import search_ablation


class SearchAblationTest(unittest.TestCase):
    def test_build_mode_config_returns_expected_flags(self):
        self.assertEqual(
            {
                "name": "classic_only",
                "use_policy": False,
                "use_value": False,
                "use_classic": True,
            },
            search_ablation.build_mode_config("classic_only"),
        )
        self.assertEqual(
            {
                "name": "policy_only",
                "use_policy": True,
                "use_value": False,
                "use_classic": False,
            },
            search_ablation.build_mode_config("policy_only"),
        )
        self.assertEqual(
            {
                "name": "value_only",
                "use_policy": False,
                "use_value": True,
                "use_classic": False,
            },
            search_ablation.build_mode_config("value_only"),
        )
        self.assertEqual(
            {
                "name": "full",
                "use_policy": True,
                "use_value": True,
                "use_classic": False,
            },
            search_ablation.build_mode_config("full"),
        )

    def test_build_mode_config_rejects_unknown_modes(self):
        with self.assertRaisesRegex(ValueError, "unsupported ablation mode"):
            search_ablation.build_mode_config("unknown")


class SearchAblationHelperTest(unittest.TestCase):
    def test_flat_legal_priors_only_weight_legal_moves(self):
        priors = search_ablation.flat_legal_priors([1, 4])
        self.assertEqual((6,), priors.shape)
        self.assertEqual(np.float32, priors.dtype)
        np.testing.assert_allclose(
            np.asarray([0.0, 0.5, 0.0, 0.0, 0.5, 0.0], dtype=np.float32),
            priors,
        )

    def test_flat_legal_priors_returns_zeros_for_empty_legal_moves(self):
        priors = search_ablation.flat_legal_priors([])
        self.assertEqual((6,), priors.shape)
        self.assertEqual(np.float32, priors.dtype)
        np.testing.assert_allclose(np.zeros(6, dtype=np.float32), priors)

    def test_neutral_value_is_centered_zero(self):
        self.assertEqual(0.0, search_ablation.neutral_value())


class RunSearchAblationTest(unittest.TestCase):
    def test_parse_args_uses_schema_defaults(self):
        from ml.alphazero_lite import run_search_ablation

        args = run_search_ablation.parse_args(["--out", "/tmp/report.json"])

        self.assertEqual("model-artifact/current", args.artifact_path)
        self.assertEqual(
            "ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json",
            args.suite,
        )
        self.assertEqual("128,384,1200", args.budgets)
        self.assertEqual("/tmp/report.json", args.out)
        self.assertEqual(42, args.seed)

    def test_parse_budgets_returns_positive_unique_integers(self):
        from ml.alphazero_lite import run_search_ablation

        self.assertEqual([128, 384, 1200], run_search_ablation.parse_budgets("128,384,1200"))

    def test_parse_budgets_rejects_duplicates(self):
        from ml.alphazero_lite import run_search_ablation

        with self.assertRaisesRegex(ValueError, "duplicate budgets"):
            run_search_ablation.parse_budgets("128,384,128")

    def test_build_position_matrix_summarizes_overall_and_bucket_rows(self):
        from ml.alphazero_lite import run_search_ablation

        rows_by_budget_and_mode = {
            128: {
                "classic_only": [
                    {
                        "bucket": "opening_plies_1_8",
                        "agrees_top1": True,
                        "regret": 0.25,
                        "value_error": 0.2,
                    },
                    {
                        "bucket": "capture_available",
                        "agrees_top1": False,
                        "regret": 0.5,
                        "value_error": None,
                    },
                ],
                "full": [
                    {
                        "bucket": "opening_plies_1_8",
                        "agrees_top1": True,
                        "regret": 0.0,
                        "value_error": 0.1,
                    }
                ],
            },
            384: {
                "classic_only": [
                    {
                        "bucket": "capture_available",
                        "agrees_top1": True,
                        "regret": 0.1,
                        "value_error": 0.4,
                    }
                ],
                "full": [
                    {
                        "bucket": "opening_plies_1_8",
                        "agrees_top1": False,
                        "regret": 0.75,
                        "value_error": 0.6,
                    },
                    {
                        "bucket": "capture_available",
                        "agrees_top1": True,
                        "regret": 0.25,
                        "value_error": 0.3,
                    },
                ],
            },
        }

        matrix = run_search_ablation.build_position_matrix(rows_by_budget_and_mode)

        self.assertEqual(
            {
                "overall": {
                    128: {
                        "classic_only": summarize_bucket(rows_by_budget_and_mode[128]["classic_only"]),
                        "full": summarize_bucket(rows_by_budget_and_mode[128]["full"]),
                    },
                    384: {
                        "classic_only": summarize_bucket(rows_by_budget_and_mode[384]["classic_only"]),
                        "full": summarize_bucket(rows_by_budget_and_mode[384]["full"]),
                    },
                },
                "buckets": {
                    "capture_available": {
                        128: {
                            "classic_only": summarize_bucket([
                                rows_by_budget_and_mode[128]["classic_only"][1]
                            ]),
                            "full": summarize_bucket([]),
                        },
                        384: {
                            "classic_only": summarize_bucket(rows_by_budget_and_mode[384]["classic_only"]),
                            "full": summarize_bucket([
                                rows_by_budget_and_mode[384]["full"][1]
                            ]),
                        },
                    },
                    "opening_plies_1_8": {
                        128: {
                            "classic_only": summarize_bucket([
                                rows_by_budget_and_mode[128]["classic_only"][0]
                            ]),
                            "full": summarize_bucket(rows_by_budget_and_mode[128]["full"]),
                        },
                        384: {
                            "classic_only": summarize_bucket([]),
                            "full": summarize_bucket([
                                rows_by_budget_and_mode[384]["full"][0]
                            ]),
                        },
                    },
                },
            },
            matrix,
        )

    def test_build_attribution_summary_reports_pairwise_deltas_and_larger_contributor(self):
        from ml.alphazero_lite import run_search_ablation

        summary = run_search_ablation.build_attribution_summary(
            overall={
                128: {
                    "classic_only": {"score": 0.40},
                    "policy_only": {"score": 0.61},
                    "value_only": {"score": 0.53},
                    "full": {"score": 0.68},
                },
                384: {
                    "classic_only": {"score": 0.45},
                    "policy_only": {"score": 0.66},
                    "value_only": {"score": 0.57},
                    "full": {"score": 0.74},
                },
            },
            buckets={
                "capture_available": {
                    128: {
                        "classic_only": {"score": 0.48},
                        "policy_only": {"score": 0.46},
                        "value_only": {"score": 0.52},
                        "full": {"score": 0.58},
                    },
                    384: {
                        "classic_only": {"score": 0.50},
                        "policy_only": {"score": 0.47},
                        "value_only": {"score": 0.56},
                        "full": {"score": 0.60},
                    },
                }
            },
        )

        self.assertEqual(
            {
                128: {
                    "full_minus_classic_only": 0.28,
                    "full_minus_policy_only": 0.07,
                    "full_minus_value_only": 0.15,
                    "policy_only_minus_classic_only": 0.21,
                    "value_only_minus_classic_only": 0.13,
                    "larger_contributor": "policy",
                },
                384: {
                    "full_minus_classic_only": 0.29,
                    "full_minus_policy_only": 0.08,
                    "full_minus_value_only": 0.17,
                    "policy_only_minus_classic_only": 0.21,
                    "value_only_minus_classic_only": 0.12,
                    "larger_contributor": "policy",
                },
            },
            summary["overall"],
        )
        self.assertEqual(
            {
                "capture_available": {
                    128: {
                        "full_minus_classic_only": 0.10,
                        "full_minus_policy_only": 0.12,
                        "full_minus_value_only": 0.06,
                        "policy_only_minus_classic_only": -0.02,
                        "value_only_minus_classic_only": 0.04,
                        "larger_contributor": "value",
                    },
                    384: {
                        "full_minus_classic_only": 0.10,
                        "full_minus_policy_only": 0.13,
                        "full_minus_value_only": 0.04,
                        "policy_only_minus_classic_only": -0.03,
                        "value_only_minus_classic_only": 0.06,
                        "larger_contributor": "value",
                    },
                }
            },
            summary["buckets"],
        )

    def test_build_attribution_summary_reports_neither_when_learned_modes_trail_classic(self):
        from ml.alphazero_lite import run_search_ablation

        summary = run_search_ablation.build_attribution_summary(
            overall={
                128: {
                    "classic_only": {"score": 0.62},
                    "policy_only": {"score": 0.41},
                    "value_only": {"score": 0.43},
                    "full": {"score": 0.44},
                }
            },
            buckets={},
        )

        self.assertEqual(
            {
                "full_minus_classic_only": -0.18,
                "full_minus_policy_only": 0.03,
                "full_minus_value_only": 0.01,
                "policy_only_minus_classic_only": -0.21,
                "value_only_minus_classic_only": -0.19,
                "larger_contributor": "neither",
            },
            summary["overall"][128],
        )

    def test_build_real_rows_by_budget_and_mode_reuses_reference_across_modes(self):
        from ml.alphazero_lite import run_search_ablation

        fake_position = mock.Mock(
            id="opening-1",
            bucket="opening_plies_1_8",
            state={
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            },
            side_to_move=0,
            legal_moves=[0, 1, 2, 3, 4, 5],
            phase="opening",
            tags=["opening"],
            source="fixture",
        )
        args = run_search_ablation.parse_args(["--out", "/tmp/report.json", "--budgets", "128,384"])
        reference_calls = []

        def fake_run_reference(state, policy_simulations, value_simulations, seed, index):
            del state, seed
            reference_calls.append((policy_simulations, value_simulations, index))
            return {
                "selected_move": 0,
                "child_stats": [{"move": 0, "visits": policy_simulations, "win_rate": 0.75}],
                "teacher_value": 0.5,
            }

        def fake_evaluate_artifact_position(*, artifact_path, evaluator, state, simulations, seed, c_puct, search_options, ablation_mode):
            del artifact_path, evaluator, state, seed, c_puct, search_options, ablation_mode
            return {"selected_move": 0, "value": 0.5}

        with mock.patch.object(run_search_ablation, "load_suite", return_value=[fake_position]), mock.patch.object(
            run_search_ablation,
            "run_reference",
            side_effect=fake_run_reference,
        ), mock.patch.object(
            run_search_ablation,
            "evaluate_artifact_position",
            side_effect=fake_evaluate_artifact_position,
        ), mock.patch.object(run_search_ablation, "ArtifactEvaluator", return_value=mock.Mock()):
            rows = run_search_ablation.build_real_rows_by_budget_and_mode(args)

        self.assertEqual([(128, 128, 0), (384, 384, 0)], reference_calls)
        self.assertEqual(set(run_search_ablation.DEFAULT_MODES), set(rows[128].keys()))
        self.assertEqual(set(run_search_ablation.DEFAULT_MODES), set(rows[384].keys()))

    def test_main_writes_stubbed_report_with_expected_shape(self):
        runner = Path(__file__).with_name("run_search_ablation.py")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "report.json"
            env = os.environ.copy()
            env["AZLITE_SEARCH_ABLATION_STUB"] = "1"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    "--out",
                    str(out_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertTrue(out_path.exists(), completed.stderr)

            report = json.loads(out_path.read_text())

        self.assertEqual("search_ablation_report_v1", report["schema"])
        self.assertEqual("model-artifact/current", report["artifact_path"])
        self.assertEqual(
            "ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json",
            report["suite_path"],
        )
        self.assertEqual([128, 384, 1200], report["budgets"])
        self.assertEqual(
            ["classic_only", "policy_only", "value_only", "full"],
            report["modes"],
        )
        self.assertIn("reference", report)
        self.assertIn("overall", report)
        self.assertIn("buckets", report)
        self.assertIn("attribution_summary", report)

    def test_main_returns_error_when_artifact_path_is_invalid(self):
        from ml.alphazero_lite import run_search_ablation

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(run_search_ablation.__file__)),
                    "--artifact-path",
                    str(Path(tmpdir) / "missing-artifact"),
                    "--out",
                    str(out_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )

        self.assertNotEqual(0, completed.returncode)
        self.assertIn("missing weights.json", completed.stderr)
        self.assertFalse(out_path.exists())

    def test_default_modes_pin_issue_258_contract_order(self):
        from ml.alphazero_lite import run_search_ablation

        self.assertEqual(
            ["classic_only", "policy_only", "value_only", "full"],
            run_search_ablation.DEFAULT_MODES,
        )


class SearchAblationWrapperTest(unittest.TestCase):
    def test_dry_run_wrapper_points_at_model_artifact_current_and_default_budgets(self):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                str(repo_root / "script/ai/run_local_policy_value_ablation"),
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        payload = json.loads(result.stdout)
        command = payload["command"]
        self.assertEqual("ml/alphazero_lite/run_search_ablation.py", command[1])
        self.assertIn("model-artifact/current", command)
        self.assertIn("128,384,1200", command)
        self.assertIn("--suite", command)
        self.assertIn("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json", command)

    def test_write_up_contains_expected_headings(self):
        doc_path = Path(__file__).resolve().parents[2] / "docs/alphazero-lite-policy-value-ablation.md"
        doc = doc_path.read_text(encoding="utf-8")

        self.assertIn("# AlphaZero-lite Policy-Value Search Ablation", doc)
        self.assertIn("## Artifact", doc)
        self.assertIn("`model-artifact/current`", doc)
        self.assertIn("## Budgets", doc)
        self.assertIn("`128`, `384`, `1200`", doc)
        self.assertIn("## Overall Attribution", doc)
        self.assertIn("## Bucket Findings", doc)
        self.assertIn("## Answer", doc)
        self.assertIn("Overall larger contributor among the learned search signals: `value`.", doc)


class SearchAblationRealRunnerTest(unittest.TestCase):
    def test_main_writes_real_report_from_forensic_suite_inputs(self):
        from ml.alphazero_lite import run_search_ablation

        fake_position = mock.Mock(
            id="opening-1",
            bucket="opening_plies_1_8",
            state={
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            },
            side_to_move=0,
            legal_moves=[0, 1, 2, 3, 4, 5],
            phase="opening",
            tags=["opening"],
            source="fixture",
        )

        def fake_run_reference(state, policy_simulations, value_simulations, seed, index):
            del state, value_simulations, seed, index
            return {
                "selected_move": 0,
                "child_stats": [
                    {"move": 0, "visits": policy_simulations, "win_rate": 0.8},
                    {"move": 1, "visits": policy_simulations, "win_rate": 0.2},
                ],
                "teacher_value": 0.4,
            }

        def fake_evaluate_artifact_position(*, artifact_path, evaluator, state, simulations, seed, c_puct, search_options, ablation_mode):
            del artifact_path, evaluator, state, seed, c_puct, search_options
            selected_move = 0 if ablation_mode in {"full", "policy_only"} else 1
            return {
                "selected_move": selected_move,
                "value": 0.4 if ablation_mode in {"full", "value_only"} else 0.0,
            }

        with tempfile.TemporaryDirectory(prefix="azlite-search-ablation-real-") as tmpdir:
            out_path = Path(tmpdir) / "report.json"
            with mock.patch.object(run_search_ablation, "load_suite", return_value=[fake_position]), mock.patch.object(
                run_search_ablation,
                "run_reference",
                side_effect=fake_run_reference,
            ), mock.patch.object(
                run_search_ablation,
                "evaluate_artifact_position",
                side_effect=fake_evaluate_artifact_position,
            ), mock.patch.object(run_search_ablation, "ArtifactEvaluator", return_value=mock.Mock()):
                exit_code = run_search_ablation.main(["--out", str(out_path), "--budgets", "128"])

            self.assertEqual(0, exit_code)
            report = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual("search_ablation_report_v1", report["schema"])
        self.assertEqual([128], report["budgets"])
        self.assertEqual(["classic_only", "policy_only", "value_only", "full"], report["modes"])
        self.assertEqual(1, report["overall"]["128"]["full"]["positions"])
        self.assertEqual(0.0, report["overall"]["128"]["full"]["top1_agreement"])
        self.assertEqual(0.0, report["overall"]["128"]["full"]["average_regret"])
        self.assertEqual(0.0, report["overall"]["128"]["full"]["value_calibration_mae"])
        self.assertEqual(0.0, report["overall"]["128"]["classic_only"]["top1_agreement"])
        self.assertIn("opening_plies_1_8", report["buckets"])
        self.assertEqual("neither", report["attribution_summary"]["overall"]["128"]["larger_contributor"])


if __name__ == "__main__":
    unittest.main()
