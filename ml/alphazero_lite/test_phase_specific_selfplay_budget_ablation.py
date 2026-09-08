"""Contract tests for the phase-specific self-play budget ablation."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from ml.alphazero_lite.run_phase_specific_selfplay_budget_ablation import (
    lane_configs,
    option_value,
    self_play_step,
)


class PhaseSpecificBudgetAblationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = {
            "selected_replay": "/tmp/selected.jsonl",
            "controls_replay": "/tmp/controls.jsonl",
            "lanes": {
                "control_384_192": {
                    "simulations": 192,
                    "opening_min_simulations": 384,
                    "opening_min_simulations_plies": 8,
                },
                "opening1200_rest192": {
                    "simulations": 192,
                    "opening_min_simulations": 1200,
                    "opening_min_simulations_plies": 8,
                },
                "uniform1200": {
                    "simulations": 1200,
                    "opening_min_simulations": None,
                    "opening_min_simulations_plies": None,
                },
            },
        }
        self.base = {
            "run_id": "base",
            "steps": [
                {
                    "name": "self_play",
                    "command": [
                        "python",
                        "self_play.py",
                        "--simulations",
                        "192",
                        "--opening-min-simulations",
                        "384",
                        "--opening-min-simulations-plies",
                        "8",
                        "--games",
                        "1600",
                    ],
                }
            ],
        }

    def test_only_budget_options_differ_across_lanes(self) -> None:
        configs = lane_configs(self.plan, self.base, Path("/tmp/work"))
        for name, expected in self.plan["lanes"].items():
            command = self_play_step(configs[name])["command"]
            self.assertEqual(
                str(expected["simulations"]), option_value(command, "--simulations")
            )
            self.assertEqual(
                None
                if expected["opening_min_simulations"] is None
                else str(expected["opening_min_simulations"]),
                option_value(command, "--opening-min-simulations"),
            )
            self.assertEqual("1600", option_value(command, "--games"))

    def test_fixed_replay_weights_are_preserved(self) -> None:
        configs = lane_configs(self.plan, copy.deepcopy(self.base), Path("/tmp/work"))
        for config in configs.values():
            self.assertEqual(
                [
                    {"path": "/tmp/selected.jsonl", "weight": 1},
                    {"path": "/tmp/controls.jsonl", "weight": 2},
                ],
                config["fixed_replay_sources"],
            )


if __name__ == "__main__":
    unittest.main()
