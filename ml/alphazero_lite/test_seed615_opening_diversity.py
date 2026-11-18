"""Synthetic fail-closed checks for Seed 615 source-group selection."""

from __future__ import annotations

import unittest

from ml.alphazero_lite.seed599_replay_coverage import (
    validate_training_heldout_disjointness,
)
from ml.alphazero_lite.seed615_opening_diversity import select_balanced


def row(identity: str, group: str, phase: str = "above_32") -> dict[str, object]:
    return {
        "canonical_identity": identity,
        "float32_identity": f"f32-{identity}",
        "source": "synthetic",
        "opening_id": group,
        "phase": phase,
    }


class Seed615SelectionTests(unittest.TestCase):
    def test_round_robin_balances_groups_and_is_deterministic(self) -> None:
        candidates = {
            f"state-{group}-{index}": row(f"state-{group}-{index}", group)
            for group, capacity in (("g0", 6), ("g1", 4), ("g2", 2))
            for index in range(capacity)
        }
        first = select_balanced(candidates, {"above_32": 7})
        second = select_balanced(candidates, {"above_32": 7})
        self.assertEqual(first, second)
        counts = {}
        for item in first:
            counts[item["opening_id"]] = counts.get(item["opening_id"], 0) + 1
        self.assertEqual(sorted(counts.values()), [2, 2, 3])

    def test_insufficient_total_phase_capacity_fails_closed(self) -> None:
        candidates = {"one": row("one", "g0")}
        with self.assertRaisesRegex(ValueError, "insufficient_phase_positions"):
            select_balanced(candidates, {"above_32": 2})

    def test_float32_collision_fails_closed(self) -> None:
        first = row("canonical-a", "g0")
        second = row("canonical-b", "g1")
        second["float32_identity"] = first["float32_identity"]
        with self.assertRaisesRegex(ValueError, "duplicate_float32_identity"):
            select_balanced({"a": first, "b": second}, {"above_32": 2})

    def test_canonical_duplicate_fails_closed(self) -> None:
        first = row("same", "g0")
        second = row("same", "g1")
        with self.assertRaisesRegex(ValueError, "duplicate_canonical_identity"):
            select_balanced({"a": first, "b": second}, {"above_32": 2})

    def test_heldout_canonical_overlap_fails_closed(self) -> None:
        candidate = [row("same", "train")]
        control = [row("control", "train")]
        heldout = [row("same", "heldout")]
        with self.assertRaisesRegex(ValueError, "heldout_treatment_canonical_overlap"):
            validate_training_heldout_disjointness(candidate, control, heldout)

    def test_heldout_float32_overlap_fails_closed(self) -> None:
        candidate = [row("candidate", "train")]
        control = [row("control", "train")]
        heldout = [row("heldout", "test")]
        heldout[0]["float32_identity"] = candidate[0]["float32_identity"]
        with self.assertRaisesRegex(ValueError, "heldout_treatment_float32_overlap"):
            validate_training_heldout_disjointness(candidate, control, heldout)


if __name__ == "__main__":
    unittest.main()
