from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import run_fresh_replay_weight_ablation as ablation


class FreshReplayWeightAblationTest(unittest.TestCase):
    def plan(self):
        return json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "docs/data/alphazero-lite-fresh-replay-weight-ablation/plan.json"
            ).read_text(encoding="utf-8")
        )

    def test_source_sha_validation_and_parent_bytes_are_pinned(self):
        plan = self.plan()
        self.assertEqual(6, len(plan["selfplay_datasets"]))
        self.assertEqual(
            "935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c",
            plan["parent_weights_sha256"],
        )
        self.assertEqual(
            "a8046517cd6da07a7ddff8efa03fea27b5a53f5a9146bcf0632150a651e74895",
            plan["selfplay_datasets"][4]["sha256"],
        )
        self.assertEqual(
            "feb6ce5ebb748599a64bba2956e29b37716ddc540ea939fe4d50dbd3ce45ddc0",
            plan["selfplay_datasets"][5]["sha256"],
        )
        self.assertEqual(
            list(ablation.SOURCE_SEEDS),
            [item["seed"] for item in ablation.validate_plan(plan)],
        )

    def test_arms_differ_only_in_fresh_weight(self):
        self.assertEqual([1, 4, 1, 8, 4], ablation.ARMS["fresh_w1"])
        self.assertEqual([4, 4, 1, 8, 4], ablation.ARMS["fresh_w4"])
        self.assertEqual(ablation.ARMS["fresh_w1"][1:], ablation.ARMS["fresh_w4"][1:])

    def test_command_has_exact_updates_and_final_checkpoint_only(self):
        plan, dataset = self.plan(), self.plan()["selfplay_datasets"][0]
        command = ablation.command_for(
            dataset, "fresh_w4", Path("checkpoint.npz"), plan
        )
        self.assertEqual("1084", command[command.index("--max-optimizer-updates") + 1])
        self.assertEqual("final", command[command.index("--final-checkpoint") + 1])
        self.assertNotIn("canonical", " ".join(command).lower())
        self.assertNotIn("promot", " ".join(command).lower())

    @patch.object(ablation, "compact_row_count", side_effect=[10, 20, 30, 40, 50])
    def test_weighted_index_accounting_uses_compact_rows(self, _count):
        plan, dataset = self.plan(), self.plan()["selfplay_datasets"][0]
        accounting = ablation.source_accounting(
            dataset, plan, ablation.ARMS["fresh_w4"]
        )
        self.assertEqual(
            [40, 80, 30, 320, 200],
            [item["weighted_index_count"] for item in accounting],
        )
        self.assertEqual(40 / 670, accounting[0]["weighted_index_fraction"])

    def test_paired_bootstrap_is_deterministic(self):
        def cell(seed, value):
            return {
                "self_play_source": {"seed": seed},
                "diagnostic_arena": {"score": value},
                "exact": {
                    "raw": {"optimal_mass": value, "expected_regret": value},
                    "mcts_384": {"optimal_mass": value, "expected_regret": value},
                },
            }

        left, right = [cell(401, 0.1), cell(407, 0.3)], [cell(401, 0.2), cell(407, 0.7)]
        self.assertEqual(
            ablation.paired(left, right, "arena", 351),
            ablation.paired(left, right, "arena", 351),
        )

    def test_runner_has_no_canonical_promotion_path(self):
        source = Path(ablation.__file__).read_text(encoding="utf-8")
        self.assertNotIn("promotion_gate", source)
        self.assertNotIn("canonical_prefilter", source)


if __name__ == "__main__":
    unittest.main()
