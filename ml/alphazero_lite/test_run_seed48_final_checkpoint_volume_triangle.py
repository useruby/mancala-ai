from __future__ import annotations

import json
import unittest
from pathlib import Path

from ml.alphazero_lite import run_seed48_final_checkpoint_volume_triangle as triangle


class FinalCheckpointVolumeTriangleTest(unittest.TestCase):
    def test_plan_validates_final_only_triangle_and_reused_b(self):
        plan = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "docs/data/alphazero-lite-seed48-final-checkpoint-volume-triangle/plan.json"
            ).read_text(encoding="utf-8")
        )

        b = triangle.validate_plan(plan)

        self.assertEqual(triangle.TRAINING_SEEDS, tuple(sorted(b)))
        self.assertEqual(
            1084, plan["arms"]["a_final"]["training"]["max_optimizer_updates"]
        )
        self.assertEqual(
            3068, plan["arms"]["c_final"]["training"]["max_optimizer_updates"]
        )
        self.assertEqual(
            "final", plan["arms"]["a_final"]["training"]["final_checkpoint"]
        )
        self.assertEqual(
            "final", plan["arms"]["c_final"]["training"]["final_checkpoint"]
        )

    def test_command_uses_explicit_final_cap(self):
        plan = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "docs/data/alphazero-lite-seed48-final-checkpoint-volume-triangle/plan.json"
            ).read_text(encoding="utf-8")
        )

        command = triangle.command_for(
            plan["arms"]["a_final"], Path("checkpoint.npz"), plan, 443
        )

        self.assertEqual("1084", command[command.index("--max-optimizer-updates") + 1])
        self.assertEqual("final", command[command.index("--final-checkpoint") + 1])


if __name__ == "__main__":
    unittest.main()
