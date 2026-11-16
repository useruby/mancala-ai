"""Synthetic contract checks for Seed 604 preparation guards."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite import seed604_replay_optimization as seed604


class Seed604PreparationTests(unittest.TestCase):
    def test_schedule_exactly_reproduces_seed599_prefix(self) -> None:
        root = seed604.ROOT
        prior = json.loads(
            (root / seed604.SEED599 / "batch-schedule-B.json").read_bytes()
        )
        schedule = seed604.batch_schedule()
        self.assertEqual(schedule[:512], prior)
        self.assertEqual(len(schedule), 2048)

    def test_authorization_gates_are_distinct_and_freeze_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / seed604.DATA
            data.mkdir(parents=True)
            self.assertFalse(seed604.training_authorized(root))
            self.assertFalse(seed604.evaluation_authorized(root))
            for action, checker, status, key in (
                (
                    "training",
                    seed604.training_authorized,
                    "authorized-for-training",
                    "training_may_start",
                ),
                (
                    "evaluation",
                    seed604.evaluation_authorized,
                    "authorized-for-evaluation",
                    "run_games",
                ),
            ):
                freeze_path = data / f"{action}-execution-freeze.json"
                auth_path = data / f"{action}-authorization.json"
                freeze_path.write_text(
                    json.dumps({"status": "frozen-awaiting-owner-authorization"})
                )
                digest = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
                auth_path.write_text(
                    json.dumps(
                        {
                            "status": status,
                            "execution_freeze_sha256": digest,
                            "authorization": {
                                key: True,
                                "authorized_by": "synthetic owner",
                            },
                        }
                    )
                )
                self.assertTrue(checker(root))
                freeze_path.write_text(freeze_path.read_text() + " ")
                self.assertFalse(checker(root))


if __name__ == "__main__":
    unittest.main()
