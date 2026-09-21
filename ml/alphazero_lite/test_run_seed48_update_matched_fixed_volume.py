from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from ml.alphazero_lite import run_seed48_challenger_variance as variance
from ml.alphazero_lite import run_seed48_update_matched_fixed_volume as control


class UpdateMatchedFixedVolumeTest(unittest.TestCase):
    def plan(self) -> dict[str, Any]:
        return json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "docs/data/alphazero-lite-seed48-update-matched-fixed-volume/plan.json"
            ).read_text(encoding="utf-8")
        )

    def test_plan_validates_paired_seed_and_frozen_input_provenance(self):
        plan = self.plan()

        a, c = control.validate_plan(plan)

        self.assertEqual(control.TRAINING_SEEDS, tuple(sorted(a)))
        self.assertEqual(control.TRAINING_SEEDS, tuple(sorted(c)))
        self.assertEqual(3068, plan["training"]["max_optimizer_updates"])
        self.assertEqual("final", plan["training"]["final_checkpoint"])
        self.assertEqual(
            "a7016fe0e360f81ba98613c52f04510e1449352bcc6d9f20cbfbfa647706e400",
            plan["fixed_pool_sha256"],
        )

    def test_plan_rejects_canonical_promotion_or_selection(self):
        for flag in (
            "canonical_gate_run",
            "promotion_performed",
            "candidate_selection_performed",
        ):
            plan = self.plan()
            plan[flag] = True
            with self.assertRaisesRegex(ValueError, "nonpromotion_contract"):
                control.validate_plan(plan)

    def test_training_command_has_only_training_controls(self):
        plan = self.plan()
        command = variance.training_command(
            control.ROOT / plan["fixed_pool_path"],
            Path("checkpoint.npz"),
            plan,
            443,
        )

        self.assertNotIn("canonical", " ".join(command).lower())
        self.assertNotIn("promot", " ".join(command).lower())
        self.assertIn("--lr-scheduler", command)
        self.assertEqual("none", command[command.index("--lr-scheduler") + 1])


if __name__ == "__main__":
    unittest.main()
