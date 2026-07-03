import unittest

from ml.alphazero_lite.runtime_root_sensitivity import root_sensitivity_prefilter


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
