"""Focused contract tests for the PR #286 uniform-1200 seed confirmation."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from ml.alphazero_lite.run_uniform1200_seed_confirmation import (
    LANES,
    classify,
    deterministic_exact_audit_sample,
    lane_config,
    non_budget_identity,
    self_play_step,
)
from ml.alphazero_lite.run_phase_specific_selfplay_budget_ablation import option_value


class Uniform1200SeedConfirmationTest(unittest.TestCase):
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
                "uniform1200": {
                    "simulations": 1200,
                    "opening_min_simulations": None,
                    "opening_min_simulations_plies": None,
                },
            },
        }
        self.base = {
            "steps": [
                {
                    "name": "self_play",
                    "command": [
                        "python",
                        "self_play.py",
                        "--seed",
                        "42",
                        "--seed-sweep",
                        "41,42,43",
                        "--simulations",
                        "192",
                        "--opening-min-simulations",
                        "384",
                        "--opening-min-simulations-plies",
                        "8",
                        "--games",
                        "1600",
                    ],
                },
                {
                    "name": "train",
                    "command": ["python", "train.py", "--seed", "442", "--epochs", "4"],
                },
            ]
        }

    def test_only_two_lanes_and_matched_seed_rendering(self) -> None:
        self.assertEqual(("control_384_192", "uniform1200"), LANES)
        configs = {
            lane: lane_config(
                self.plan, self.base, seed=44, lane=lane, workdir=Path("/tmp/work")
            )
            for lane in LANES
        }
        for config in configs.values():
            command = self_play_step(config)["command"]
            self.assertEqual("44", option_value(command, "--seed"))
            self.assertEqual("43,44,45", option_value(command, "--seed-sweep"))
            self.assertEqual(
                "44", option_value(config["steps"][1]["command"], "--seed")
            )
        self.assertEqual(
            non_budget_identity(configs["control_384_192"]),
            non_budget_identity(configs["uniform1200"]),
        )

    def test_replay_and_checkpoint_recipe_are_identical(self) -> None:
        left = lane_config(
            self.plan,
            copy.deepcopy(self.base),
            seed=45,
            lane=LANES[0],
            workdir=Path("/tmp/work"),
        )
        right = lane_config(
            self.plan,
            copy.deepcopy(self.base),
            seed=45,
            lane=LANES[1],
            workdir=Path("/tmp/work"),
        )
        self.assertEqual(left["fixed_replay_sources"], right["fixed_replay_sources"])
        self.assertEqual(left["steps"][1], right["steps"][1])

    def test_exact_audit_sample_is_deterministic_and_never_mutates_rows(self) -> None:
        rows = [
            {"state": [index], "move_index": index % 12, "policy": [1.0]}
            for index in range(500)
        ]
        original = copy.deepcopy(rows)
        self.assertEqual(
            deterministic_exact_audit_sample(rows, opening=True),
            deterministic_exact_audit_sample(list(reversed(rows)), opening=True),
        )
        self.assertEqual(rows, original)

    def test_pre_registered_classification(self) -> None:
        self.assertEqual(
            "uniform1200_confirmed",
            classify(
                [0.1, 0.2, -0.01],
                2,
                credible_regression=False,
                oracle_contradictory=False,
            ),
        )
        self.assertEqual(
            "uniform1200_seed_sensitive",
            classify(
                [0.1, -0.2, -0.1],
                2,
                credible_regression=True,
                oracle_contradictory=False,
            ),
        )
        self.assertEqual(
            "uniform1200_not_confirmed",
            classify(
                [0.1, 0.2, 0.1],
                1,
                credible_regression=False,
                oracle_contradictory=False,
            ),
        )


if __name__ == "__main__":
    unittest.main()
