"""Focused contracts for the evaluation-only exact neighborhood audit."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite.run_uniform1200_exact_neighborhood_audit import (
    family_classification,
    generate_neighborhood,
    load_anchor_ids,
    ordered_tasks,
    preflight_native_oracle,
    sample_radius2,
    solve_exact,
    coverage_by_radius,
)


class ExactNeighborhoodAuditTest(unittest.TestCase):
    @staticmethod
    def row(key: str, radius: int, status: str = "not_attempted") -> dict:
        return {
            "canonical_state_key": key,
            "radius": radius,
            "terminal": False,
            "state": {
                "player_pits": [1, 0, 0, 0, 0, 0],
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 23,
                "opponent_store": 23,
                "current_player": 0,
            },
            "oracle_status": status,
        }

    def test_pr289_anchor_loading_is_locked_to_expected_regression_count(self) -> None:
        regressions, improvements = load_anchor_ids()
        self.assertEqual(38, len(regressions))
        self.assertTrue(improvements)

    def test_radius_generation_uses_legal_game_transitions_and_marks_ineligible(
        self,
    ) -> None:
        state = {
            "player_pits": [0, 0, 0, 0, 0, 1],
            "opponent_pits": [0, 0, 0, 0, 0, 1],
            "player_store": 23,
            "opponent_store": 23,
            "current_player": 0,
        }
        rows = generate_neighborhood("test-001", state)
        self.assertEqual(1, sum(row["radius"] == 1 for row in rows))
        self.assertTrue(all(not row["training_eligible"] for row in rows))
        radius1 = next(row for row in rows if row["radius"] == 1)
        self.assertTrue(radius1["provenance"][0]["extra_turn"])

    def test_radius2_sampling_is_sha_deterministic(self) -> None:
        rows = [{"canonical_state_key": str(index), "radius": 2} for index in range(20)]
        first, counts = sample_radius2(rows, cap=7)
        second, _ = sample_radius2(list(reversed(rows)), cap=7)
        self.assertEqual(first, second)
        self.assertEqual(20, counts["radius2_full"])
        self.assertEqual(7, counts["radius2_sampled"])

    def test_family_thresholds_are_fixed(self) -> None:
        self.assertEqual(
            "stable_regression_family",
            family_classification(
                original_regression_seeds=2,
                local_rates_by_seed=[0.6, 0.7, 0.5],
                solved_states=10,
                same_direction_seeds=2,
            ),
        )
        self.assertEqual(
            "anchor_specific_regression",
            family_classification(
                original_regression_seeds=2,
                local_rates_by_seed=[0.1, 0.2, 0.3],
                solved_states=10,
                same_direction_seeds=2,
            ),
        )
        self.assertEqual(
            "mixed",
            family_classification(
                original_regression_seeds=2,
                local_rates_by_seed=[0.4, 0.5, 0.3],
                solved_states=10,
                same_direction_seeds=2,
            ),
        )

    def test_timeout_rows_are_skipped_unless_explicitly_retried(self) -> None:
        rows = [self.row("timeout", 0, "exact_timeout"), self.row("new", 2)]
        self.assertEqual(
            ["new"],
            [row["canonical_state_key"] for row in ordered_tasks(rows, False, False)],
        )
        self.assertEqual(
            {"timeout", "new"},
            {row["canonical_state_key"] for row in ordered_tasks(rows, True, False)},
        )

    def test_solved_rows_are_never_resubmitted_and_history_is_preserved(self) -> None:
        solved = self.row("solved", 0, "exact_solved")
        solved["exact_value"] = 42
        timeout = self.row("timeout", 1, "exact_timeout")
        timeout["oracle_attempt_history"] = [
            {"attempt_number": 1, "status": "exact_timeout"}
        ]
        result = {
            "status": "exact_solved",
            "failure_reason": None,
            "attempt_number": 2,
            "timeout_seconds": 120.0,
            "wall_duration_seconds": 0.1,
            "solver_identity": "native",
            "probe_sha256": "p",
            "tablebase_sha256": "t",
            "exact_value": 42,
            "exact_action_values": {0: 42},
            "exact_optimal_actions": [0],
            "exact_root_value": 1.0,
        }
        with patch(
            "ml.alphazero_lite.run_uniform1200_exact_neighborhood_audit._attempt",
            return_value=result,
        ) as attempt:
            solve_exact(
                [solved, timeout],
                Path("probe"),
                Path("tablebase"),
                120,
                retry_timeouts=True,
            )
        attempt.assert_called_once()
        self.assertEqual(42, solved["exact_value"])
        self.assertEqual(2, len(timeout["oracle_attempt_history"]))
        self.assertEqual("exact_solved", timeout["oracle_status"])

    def test_priority_and_coverage_are_deterministic(self) -> None:
        rows = [self.row("z", 2), self.row("a", 1), self.row("b", 0)]
        ordered = ordered_tasks(list(reversed(rows)), False, False)
        self.assertEqual([0, 1, 2], [row["radius"] for row in ordered])
        rows[0]["oracle_status"] = "exact_solved"
        rows[1]["oracle_status"] = "exact_timeout"
        rows[2]["oracle_status"] = "exact_solved"
        rates = coverage_by_radius(rows)
        self.assertEqual(1.0, rates["0"]["solve_rate"])
        self.assertEqual(0.0, rates["1"]["solve_rate"])
        self.assertEqual(1.0, rates["2"]["solve_rate"])

    def test_preflight_rejects_noncanonical_tablebase_before_requests(self) -> None:
        with self.assertRaises(ValueError):
            preflight_native_oracle(Path("missing-probe"), Path("missing-tablebase"))


if __name__ == "__main__":
    unittest.main()
