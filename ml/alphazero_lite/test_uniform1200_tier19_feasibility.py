"""Focused contracts for the immutable tier-19 unresolved feasibility run."""

from __future__ import annotations

import unittest
from pathlib import Path

from ml.alphazero_lite.run_uniform1200_tier19_feasibility import (
    COMPLETION_RETRY_TIMEOUT_SECONDS,
    preflight_tier19,
    shard_rows,
    unresolved_rows,
)


class Tier19FeasibilityTest(unittest.TestCase):
    @staticmethod
    def row(key: str, status: str, radius: int = 1) -> dict:
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

    def test_only_unresolved_nonterminal_rows_are_submitted(self) -> None:
        solved = self.row("solved", "exact_solved")
        terminal = self.row("terminal", "exact_error")
        terminal["terminal"] = True
        rows = unresolved_rows(
            [
                solved,
                terminal,
                self.row("timeout", "exact_timeout"),
                self.row("error", "exact_error"),
            ]
        )
        self.assertEqual(
            {"timeout", "error"}, {row["canonical_state_key"] for row in rows}
        )
        self.assertEqual("exact_error", rows[0]["oracle_status"])

    def test_anchor_shards_and_order_are_deterministic(self) -> None:
        rows = [
            self.row("z", "exact_error"),
            self.row("a", "exact_timeout"),
            self.row("b", "exact_error"),
        ]
        first = shard_rows(rows, 4)
        second = shard_rows(list(reversed(rows)), 4)
        self.assertEqual(
            {
                key: [row["canonical_state_key"] for row in shard]
                for key, shard in first.items()
            },
            {
                key: [row["canonical_state_key"] for row in shard]
                for key, shard in second.items()
            },
        )
        self.assertEqual(120.0, COMPLETION_RETRY_TIMEOUT_SECONDS)

    def test_preflight_rejects_unpinned_tier19_inputs(self) -> None:
        with self.assertRaises(ValueError):
            preflight_tier19(Path("missing-probe"), Path("missing-tier19"))


if __name__ == "__main__":
    unittest.main()
