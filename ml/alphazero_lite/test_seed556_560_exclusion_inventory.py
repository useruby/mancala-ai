"""Focused corruption tests for the read-only archive inventory."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.seed556_560_exclusion_inventory import (
    S556,
    S560,
    authenticate_556,
    jsonl,
    load_pinned_cohort,
    validate_alias_records,
)
from ml.alphazero_lite.verify_seed560_paired_first_action import validate_continuation


class InventoryCorruptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roots = json.loads((S560 / "cohort-manifest.json").read_text())["states"]
        cls.arms = jsonl(S560 / "continuations.jsonl")

    def test_changed_source_bytes_do_not_match_frozen_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            source_copy = Path(temporary)
            shutil.copyfile(
                S556 / "verification-receipt.json",
                source_copy / "verification-receipt.json",
            )
            shutil.copyfile(
                S556 / "registration.json", source_copy / "registration.json"
            )
            with (source_copy / "registration.json").open("ab") as stream:
                stream.write(b"\nchanged")
            with self.assertRaisesRegex(
                ValueError, "556_receipt_bytes_mismatch:registration.json"
            ):
                authenticate_556(source_copy, ["registration.json"])

    def test_missing_cohort_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            cohort = Path(temporary) / "cohort.json"
            cohort.write_text('{"states":[{"opening_index":1}]}')
            import hashlib

            expected = hashlib.sha256(cohort.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "557_cohort_state_missing:0"):
                load_pinned_cohort(cohort, expected, "557")

    def test_changed_cohort_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            cohort = Path(temporary) / "cohort.json"
            cohort.write_bytes((S556 / "registration.json").read_bytes())
            with self.assertRaisesRegex(ValueError, "557_cohort_bytes_mismatch"):
                load_pinned_cohort(
                    cohort,
                    "f62539cc3e1b1843a2bd9296453d51b895a1c5e693bd9edc3b3291f883398243",
                    "557",
                )

    def test_illegal_replay_action_is_rejected(self):
        arm = copy.deepcopy(self.arms[0])
        root = self.roots[arm["opening_index"]]
        arm["outcome"]["trajectory"][0]["action"] = 99
        with self.assertRaisesRegex(ValueError, "continuation_action_illegal"):
            validate_continuation(root, arm, arm["outcome"])

    def test_alias_with_conflicting_outcome_is_rejected(self):
        arm = copy.deepcopy(self.arms[0])
        alias = copy.deepcopy(arm)
        alias["outcome"]["conflicting_record_marker"] = True
        with self.assertRaisesRegex(ValueError, "560_alias_conflicting_outcomes"):
            validate_alias_records(
                [arm, alias], {arm["opening_index"]: self.roots[arm["opening_index"]]}
            )


if __name__ == "__main__":
    unittest.main()
