import unittest
from pathlib import Path
import tempfile
from unittest import mock

from ml.alphazero_lite.runtime_root_sensitivity import (
    root_sensitivity_prefilter,
    runtime_sensitivity_diagnostic_for_opening_suite,
)


class RootSensitivityPrefilterTest(unittest.TestCase):
    def test_filters_lanes_below_thresholds(self):
        result = root_sensitivity_prefilter(
            runtime_sensitivity={
                "budgets": {
                    "384:256": {
                        "identity_ref": {
                            "move_change_rate": 0.0,
                            "mean_abs_value_delta": 0.0,
                        },
                        "lane_a": {
                            "move_change_rate": 0.0,
                            "mean_abs_value_delta": 0.0,
                        },
                        "lane_b": {
                            "move_change_rate": 0.02,
                            "mean_abs_value_delta": 0.0,
                        },
                    },
                    "768:768": {
                        "lane_a": {
                            "move_change_rate": 0.0,
                            "mean_abs_value_delta": 0.005,
                        },
                        "lane_b": {
                            "move_change_rate": 0.0,
                            "mean_abs_value_delta": 0.0,
                        },
                    },
                }
            },
            lane_names=["identity_ref", "lane_a", "lane_b"],
            default_name="identity_ref",
            min_move_change_rate=0.01,
            min_mean_abs_value_delta=0.01,
        )

        self.assertEqual(["identity_ref", "lane_b"], result["retain"])
        self.assertEqual(["lane_a"], result["filtered_out"])
        self.assertEqual(
            "reference_lane", result["decisions"]["identity_ref"]["reason"]
        )
        self.assertEqual(
            "below_root_sensitivity_threshold", result["decisions"]["lane_a"]["reason"]
        )
        self.assertEqual(
            "root_sensitivity_threshold_met", result["decisions"]["lane_b"]["reason"]
        )


class RuntimeSensitivityDiagnosticTest(unittest.TestCase):
    def test_prefers_search_root_value_over_raw_value(self):
        suite_entries = [{"state": {"current_player": 0}}]
        identity_summary = {
            "selected_move": 0,
            "value": 0.25,
            "search_root_value": 0.25,
        }
        transformed_summary = {
            "selected_move": 0,
            "value": 0.25,
            "search_root_value": -0.5,
        }

        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch(
                    "ml.alphazero_lite.runtime_root_sensitivity.load_suite_entries",
                    return_value=suite_entries,
                ),
                mock.patch(
                    "ml.alphazero_lite.runtime_root_sensitivity.ArtifactEvaluator"
                ),
                mock.patch(
                    "ml.alphazero_lite.runtime_root_sensitivity.sha256_file",
                    return_value="suite-sha",
                ),
                mock.patch(
                    "ml.alphazero_lite.runtime_root_sensitivity.evaluate_artifact_position",
                    side_effect=[identity_summary, transformed_summary],
                ),
            ):
                diagnostic = runtime_sensitivity_diagnostic_for_opening_suite(
                    current_path=Path("model-artifact/current"),
                    suite_path=Path("suite.jsonl"),
                    lane_specs=[
                        {"name": "identity_ref", "value_transform": None},
                        {
                            "name": "zero_value",
                            "value_transform": {
                                "name": "zero_value",
                                "kind": "diagnostic_phase_transform",
                                "phase_params": {
                                    "opening": {"mode": "zero"},
                                    "midgame": {"mode": "zero"},
                                    "late": {"mode": "zero"},
                                },
                            },
                        },
                    ],
                    default_c_puct=1.25,
                    cpuct_schedule={"768:768": 0.9},
                    seed=42,
                    workdir=Path(tmp),
                    budget_labels=["384:256"],
                )

        self.assertAlmostEqual(
            0.75,
            diagnostic["budgets"]["384:256"]["zero_value"]["mean_abs_value_delta"],
        )
