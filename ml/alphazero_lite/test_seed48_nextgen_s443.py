from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest import mock

from ml.alphazero_lite.run_seed48_nextgen_s443 import (
    assert_effective_selfplay_contract,
    effective_selfplay_command,
    preflight,
    rendered_config,
)

ROOT = Path(__file__).resolve().parents[2]


class Seed48NextgenS443Test(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = json.loads(
            (ROOT / "ml/alphazero_lite/configs/seed48_nextgen_s443.json").read_text()
        )
        self.base = json.loads((ROOT / self.plan["base_config"]).read_text())

    def test_pinned_recipe_preflight(self) -> None:
        with mock.patch(
            "ml.alphazero_lite.run_seed48_nextgen_s443.seed_conflict",
            return_value=False,
        ):
            integrity = preflight(self.plan, self.base, ROOT / ".tmp/test-nextgen")
        self.assertEqual(
            self.plan["expected_parent_weights_sha256"],
            integrity["parent_weights_sha256"],
        )

    def test_seed_and_opening_budget_pins(self) -> None:
        config = rendered_config(self.plan, self.base, ROOT / ".tmp/test-nextgen")
        command = config["steps"][0]["command"]
        self.assertTrue(config["preserve_config_workers"])
        self.assertIn("443", command)
        self.assertIn("442,443,444", command)
        self.assertEqual("1200", command[command.index("--simulations") + 1])
        self.assertNotIn("--opening-min-simulations", command)

    def test_final_effective_worker_contract_blocks_before_execution(self) -> None:
        config = rendered_config(self.plan, self.base, ROOT / ".tmp/test-nextgen")
        command = effective_selfplay_command(config, ROOT / ".tmp/test-nextgen")
        assert_effective_selfplay_contract(command, self.plan)
        invalid = list(command)
        invalid[invalid.index("--workers") + 1] = "24"
        with self.assertRaisesRegex(ValueError, "self_play_worker_preflight_failed"):
            assert_effective_selfplay_contract(invalid, self.plan)

    def test_preflight_rejects_parent_seed_replay_and_suite_changes(self) -> None:
        for key, value, message in (
            ("training_seed", 444, "seed/sweep"),
            ("expected_parent_weights_sha256", "0" * 64, "parent_weights"),
        ):
            plan = copy.deepcopy(self.plan)
            plan[key] = value
            with mock.patch(
                "ml.alphazero_lite.run_seed48_nextgen_s443.seed_conflict",
                return_value=False,
            ):
                with self.assertRaisesRegex(ValueError, message):
                    preflight(plan, self.base, ROOT / ".tmp/test-nextgen")
        plan = copy.deepcopy(self.plan)
        plan["fixed_replay_sources"][0]["sha256"] = "0" * 64
        with mock.patch(
            "ml.alphazero_lite.run_seed48_nextgen_s443.seed_conflict",
            return_value=False,
        ):
            with self.assertRaisesRegex(FileNotFoundError, "replay_provenance"):
                preflight(plan, self.base, ROOT / ".tmp/test-nextgen")
        self.assertEqual(
            "5d1f5982c990d00bce0a57161c1ae710bed6b8ced831ac2c1551af5217d2716c",
            self.plan["canonical_gate"]["hard_suite_sha256"],
        )
        plan = copy.deepcopy(self.plan)
        plan["canonical_gate"]["hard_suite_sha256"] = "0" * 64
        with mock.patch(
            "ml.alphazero_lite.run_seed48_nextgen_s443.seed_conflict",
            return_value=False,
        ):
            with self.assertRaisesRegex(ValueError, "frozen_suite"):
                preflight(plan, self.base, ROOT / ".tmp/test-nextgen")

    def test_conflicting_seed_is_blocked(self) -> None:
        with mock.patch(
            "ml.alphazero_lite.run_seed48_nextgen_s443.seed_conflict", return_value=True
        ):
            with self.assertRaisesRegex(
                ValueError, "registered_training_seed_conflict"
            ):
                preflight(self.plan, self.base, ROOT / ".tmp/test-nextgen")


if __name__ == "__main__":
    unittest.main()
