"""Semantic regression tests for the paired first-action evidence contract."""

from __future__ import annotations

import json
import unittest

from ml.alphazero_lite.seed560_paired_first_action import OUT
from ml.alphazero_lite.verify_seed560_paired_first_action import validate_continuation


class Seed560PairedFirstActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohort = json.loads((OUT / "cohort-manifest.json").read_text())["states"]
        cls.roots = [
            json.loads(line)
            for line in (OUT / "root-searches.jsonl").read_text().splitlines()
        ]
        cls.outcomes = [
            json.loads(line)
            for line in (OUT / "continuations.jsonl").read_text().splitlines()
        ]

    def test_identical_actions_reuse_one_continuation(self):
        by_key = {(row["opening_index"], row["budget"]): row for row in self.outcomes}
        matches = [
            i
            for i, root in enumerate(self.roots)
            if root["actions"]["128"] == root["actions"]["512"]
        ]
        self.assertTrue(matches)
        for index in matches:
            self.assertEqual(
                by_key[(index, 128)]["outcome"], by_key[(index, 512)]["outcome"]
            )

    def test_extra_turn_trajectory_is_replayed_without_alternating_actor_assumption(
        self,
    ):
        by_key = {(row["opening_index"], row["budget"]): row for row in self.outcomes}
        selected = None
        for index, root in enumerate(self.roots):
            row = by_key[(index, 128)]
            actor = self.cohort[index]["state"]["current_player"]
            sequence = [
                actor,
                *[step["actor"] for step in row["outcome"]["trajectory"]],
            ]
            if any(left == right for left, right in zip(sequence, sequence[1:])):
                selected = (self.cohort[index], row)
                break
        self.assertIsNotNone(selected)
        validate_continuation(selected[0], selected[1], selected[1]["outcome"])

    def test_continuation_seed_is_bound_to_root_and_ply(self):
        row = next(row for row in self.outcomes if row["outcome"]["trajectory"])
        corrupted = json.loads(json.dumps(row))
        corrupted["outcome"]["trajectory"][0]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "seed_or_runtime_binding"):
            validate_continuation(
                self.cohort[row["opening_index"]], corrupted, corrupted["outcome"]
            )

    def test_contradictory_terminal_evidence_is_rejected(self):
        row = next(row for row in self.outcomes if row["outcome"]["trajectory"])
        corrupted = json.loads(json.dumps(row))
        corrupted["outcome"]["margin"] += 1
        with self.assertRaisesRegex(ValueError, "terminal_evidence_contradiction"):
            validate_continuation(
                self.cohort[row["opening_index"]], corrupted, corrupted["outcome"]
            )


if __name__ == "__main__":
    unittest.main()
