from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from ml.alphazero_lite.run_seed48_uniform384_vs_1200 import (
    LANES,
    classify,
    lane_config,
    non_budget_identity,
    paired_bootstrap,
    preflight,
    seed_sweep,
)


ROOT = Path(__file__).resolve().parents[2]


class Seed48UniformBudgetAblationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = json.loads(
            (
                ROOT / "ml/alphazero_lite/configs/seed48_uniform384_vs_1200.json"
            ).read_text()
        )
        self.base = json.loads((ROOT / self.plan["base_config"]).read_text())

    def test_lanes_differ_only_by_simulations(self) -> None:
        configs = [
            lane_config(self.plan, self.base, 401, lane, ROOT / ".tmp/test")
            for lane in LANES
        ]
        self.assertEqual(
            non_budget_identity(configs[0]), non_budget_identity(configs[1])
        )
        self.assertNotIn("--opening-min-simulations", configs[0]["steps"][0]["command"])

    def test_parent_and_replay_pins_preflight(self) -> None:
        integrity = preflight(self.plan, self.base)
        self.assertEqual(
            self.plan["expected_parent_weights_sha256"],
            integrity["parent_weights_sha256"],
        )
        self.assertEqual(
            [4, 1, 8, 4], [row["weight"] for row in integrity["replay_sources"]]
        )

    def test_parent_pin_rejects_mismatch(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["expected_parent_weights_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "parent weights SHA"):
            preflight(plan, self.base)

    def test_replay_pin_rejects_mismatch(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["fixed_replay_sources"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(FileNotFoundError, "replay_provenance"):
            preflight(plan, self.base)

    def test_training_pairs_have_non_overlapping_sweeps(self) -> None:
        sweeps = [
            set(seed_sweep(seed).split(",")) for seed in self.plan["training_seeds"]
        ]
        self.assertEqual(18, len(set().union(*sweeps)))
        self.assertEqual([401, 407, 413, 419, 425, 431], self.plan["training_seeds"])

    def test_bootstrap_is_deterministic_at_training_seed_level(self) -> None:
        values = [0.01, -0.02, 0.03, -0.01, 0.0, 0.02]
        self.assertEqual(
            paired_bootstrap(values, seed=341, samples=100),
            paired_bootstrap(values, seed=341, samples=100),
        )

    def test_noninferiority_boundary_is_strict(self) -> None:
        aggregate = {
            "sibling_effect": {
                "values": [0] * 6,
                "ci95_lower": -0.0249,
                "ci95_upper": 0.01,
            },
            "parent_gain_difference": {"values": [0] * 6},
            "work_ratio_384_over_1200": 0.4,
        }
        self.assertEqual(
            "uniform384_training_noninferior_and_cheaper",
            classify(copy.deepcopy(aggregate), margin=-0.025),
        )
        aggregate["sibling_effect"]["ci95_lower"] = -0.025
        self.assertNotEqual(
            "uniform384_training_noninferior_and_cheaper",
            classify(aggregate, margin=-0.025),
        )


if __name__ == "__main__":
    unittest.main()
