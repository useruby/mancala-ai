"""Focused corruption tests for the read-only archive inventory."""

from __future__ import annotations

import copy
import json
import unittest

from ml.alphazero_lite.seed556_560_exclusion_inventory import (
    S560,
    jsonl,
    validate_alias_records,
)
from ml.alphazero_lite.verify_seed560_paired_first_action import validate_continuation


class InventoryCorruptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roots = json.loads((S560 / "cohort-manifest.json").read_text())["states"]
        cls.arms = jsonl(S560 / "continuations.jsonl")

    def test_changed_source_bytes_do_not_match_frozen_hash(self):
        import hashlib

        frozen = "5c47e4d161077f3b438d737887da35b73ca3dddab7d60db22fed19bd6dd3d8a3"
        changed = (S560 / "continuations.jsonl").read_bytes() + b"\n"
        self.assertNotEqual(hashlib.sha256(changed).hexdigest(), frozen)

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
