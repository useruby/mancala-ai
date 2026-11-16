"""Synthetic contract checks for Seed 604 preparation guards."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite import seed604_replay_optimization as seed604


class Seed604PreparationTests(unittest.TestCase):
    def test_membership_identity_must_be_64_hex_and_match_both_sources(self) -> None:
        valid = seed604.MEMBERSHIP_SHA256
        seed604.verify_membership_identity(valid, valid, valid)
        for invalid in (valid[:-1], "z" + valid[1:], "0" * 64):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    seed604.verify_membership_identity(invalid, valid, valid)
        with self.assertRaisesRegex(ValueError, "seed604_membership_sha256_mismatch"):
            seed604.verify_membership_identity(valid, "f" * 64, valid)
        with self.assertRaisesRegex(ValueError, "seed604_membership_sha256_mismatch"):
            seed604.verify_membership_identity(valid, valid, "e" * 64)

    def test_authenticated_membership_matches_seed599_configuration(self) -> None:
        prior = seed604.ROOT / seed604.SEED599
        config = json.loads((prior / "training-B.json").read_bytes())
        rows = json.loads((prior / "treatment-membership.json").read_bytes())
        calculated = hashlib.sha256(seed604.canonical_bytes(rows)).hexdigest()
        self.assertEqual(config["membership_sha256"], calculated)
        self.assertEqual(len(calculated), 64)
        self.assertEqual(calculated, seed604.MEMBERSHIP_SHA256)

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
                    json.dumps(
                        {
                            "schema": f"seed604-{action}-execution-freeze-v1",
                            "status": "frozen-awaiting-owner-authorization",
                            "execution_authorized": False,
                        }
                    )
                )
                digest = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
                auth_path.write_text(
                    json.dumps(
                        {
                            "schema": f"seed604-{action}-owner-authorization-v1",
                            "status": status,
                            "execution_freeze_sha256": digest,
                            "authorization": {
                                "type": "explicit-project-owner-authorization",
                                key: True,
                                "authorized_by": "project owner",
                            },
                        }
                    )
                )
                self.assertTrue(checker(root))
                freeze_path.write_text(freeze_path.read_text() + " ")
                self.assertFalse(checker(root))

    def test_preflight_reports_present_but_invalid_authorization_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / seed604.DATA
            data.mkdir(parents=True)
            (data / "training-execution-freeze.json").write_text("not json")
            (data / "training-authorization.json").write_text("not json")
            status = seed604.authorization_artifact_status(root)
            self.assertTrue(status["training_freeze_present"])
            self.assertTrue(status["training_authorization_present"])
            self.assertFalse(status["training_authorized"])
            self.assertFalse(status["evaluation_freeze_present"])
            self.assertFalse(status["evaluation_authorization_present"])
            self.assertFalse(status["evaluation_authorized"])


if __name__ == "__main__":
    unittest.main()
