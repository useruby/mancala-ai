import json
import unittest
from pathlib import Path

from ml.alphazero_lite.replay_source_attribution import (
    REQUIRED_TRAINING,
    ReplaySourceAttributionError,
    arm_sources,
    training_command,
    validate_arm_contract,
)


class ReplaySourceAttributionTest(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "docs/data/alphazero-lite-replay-source-attribution/plan.json"
            ).read_text()
        )

    def test_plan_has_fixed_work_and_historical_hashes(self):
        self.assertEqual(REQUIRED_TRAINING, self.plan["training"])
        self.assertEqual(5, len(self.plan["arms"]))
        self.assertEqual(1, self.plan["sources"][0]["weight"])
        self.assertEqual(64, len(self.plan["sources"][0]["sha256"]))

    def test_each_sibling_removes_exactly_one_retaining_weights(self):
        baseline = {
            "excluded_source": None,
            "training": REQUIRED_TRAINING,
            "sources": arm_sources(self.plan, None),
        }
        for excluded in (
            "generic_bootstrap",
            "random_teacher",
            "opening_disagreement",
            "stability",
        ):
            sibling = {
                "excluded_source": excluded,
                "training": REQUIRED_TRAINING,
                "sources": arm_sources(self.plan, excluded),
            }
            validate_arm_contract(baseline, sibling)

    def test_contract_mismatch_stops_an_arm(self):
        baseline = {
            "excluded_source": None,
            "training": REQUIRED_TRAINING,
            "sources": arm_sources(self.plan, None),
        }
        sibling = {
            "excluded_source": "stability",
            "training": {**REQUIRED_TRAINING, "batch_size": 256},
            "sources": arm_sources(self.plan, "stability"),
        }
        with self.assertRaisesRegex(ReplaySourceAttributionError, "contract_mismatch"):
            validate_arm_contract(baseline, sibling)

    def test_trainer_preserves_weights_without_renormalizing(self):
        arm = {
            "training": REQUIRED_TRAINING,
            "sources": arm_sources(self.plan, "random_teacher"),
        }
        command = training_command(self.plan, arm, Path("/tmp/checkpoint.npz"))
        self.assertEqual("1,4,8,4", command[command.index("--replay-weights") + 1])
        self.assertEqual("1052", command[command.index("--max-optimizer-updates") + 1])
        self.assertEqual("final", command[command.index("--final-checkpoint") + 1])


if __name__ == "__main__":
    unittest.main()
