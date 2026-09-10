"""Focused contracts for the immutable tier-20 unresolved feasibility run."""

from __future__ import annotations

import unittest
from pathlib import Path
import json

from ml.alphazero_lite.run_uniform1200_tier20_feasibility import (
    COMPLETION_RETRY_TIMEOUT_SECONDS,
    combined_coverage,
    preflight_tier20,
    tier20_cohort_identity,
    unresolved_rows,
    BASELINE,
    TIER19,
)


class Tier20FeasibilityTest(unittest.TestCase):
    @staticmethod
    def row(key: str, status: str = "exact_error", radius: int = 1) -> dict:
        return {
            "canonical_state_key": key,
            "oracle_status": status,
            "radius": radius,
            "terminal": False,
            "provenance": [{"anchor_id": "anchor-001"}],
            "state": {
                "player_pits": [1, 0, 0, 0, 0, 0],
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 23,
                "opponent_store": 23,
                "current_player": 0,
            },
        }

    def test_only_tier19_timeouts_form_the_request_cohort(self) -> None:
        baseline = [
            self.row("solved", "exact_solved"),
            self.row("pending"),
            self.row("other"),
        ]
        tier19 = [
            {"canonical_state_key": "solved", "status": "exact_solved"},
            {"canonical_state_key": "pending", "status": "exact_timeout"},
            {"canonical_state_key": "other", "status": "exact_solved"},
        ]
        with self.assertRaisesRegex(ValueError, "expected 53"):
            unresolved_rows(baseline, tier19)

    def test_committed_request_cohort_is_exactly_the_53_tier19_timeouts(self) -> None:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["cohort"]
        tier19 = json.loads(TIER19.read_text(encoding="utf-8"))["rows"]
        rows = unresolved_rows(baseline, tier19)
        self.assertEqual(53, len(rows))
        self.assertEqual(
            "171c16b86e722d0d7208b3ad6110de49c0877163b7df75330507080c6bbbf5d7",
            tier20_cohort_identity(rows),
        )
        tier19_solved = {
            row["canonical_state_key"]
            for row in tier19
            if row["status"] == "exact_solved"
        }
        self.assertFalse(tier19_solved & {row["canonical_state_key"] for row in rows})

    def test_cohort_hash_is_deterministic(self) -> None:
        rows = [self.row("a"), self.row("b")]
        for row in rows:
            row["tier19_status"] = "exact_timeout"
        self.assertEqual(
            tier20_cohort_identity(rows), tier20_cohort_identity(list(rows))
        )

    def test_earlier_solved_labels_take_precedence_in_coverage(self) -> None:
        baseline = [
            self.row("a", "exact_solved", 0),
            self.row("b", "exact_error", 1),
            self.row("c", "exact_error", 2),
        ]
        tier19 = [
            {"canonical_state_key": "b", "status": "exact_solved"},
            {"canonical_state_key": "c", "status": "exact_timeout"},
        ]
        rates = combined_coverage(baseline, tier19, {"c": {"status": "exact_solved"}})
        self.assertEqual(1.0, rates["0"]["solve_rate"])
        self.assertEqual(1.0, rates["1"]["solve_rate"])
        self.assertEqual(1.0, rates["2"]["solve_rate"])

    def test_timeout_and_pinned_preflight_contracts(self) -> None:
        self.assertEqual(120.0, COMPLETION_RETRY_TIMEOUT_SECONDS)
        with self.assertRaises(ValueError):
            preflight_tier20(Path("missing-probe"), Path("missing-tier20"))


if __name__ == "__main__":
    unittest.main()
