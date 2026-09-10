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
    preflight_python_label_path,
    sample_radius2,
    shard_for_row,
    shard_tasks,
    solve_exact,
    coverage_by_radius,
    original_coverage_gate,
    benchmark_cohort,
    benchmark_metrics,
    run_warm_vs_fresh_benchmark,
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
            "provenance": [{"anchor_id": "anchor-001"}],
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
        native_result = {
            "action_values": {"0": 42},
            "optimal_actions": [0],
            "exact_value": 42,
            "metrics": {"tt_hits": 2, "cumulative_cache": {"tt_hits": 2}},
        }
        with (
            patch(
                "ml.alphazero_lite.run_uniform1200_exact_neighborhood_audit.NativeHybridProcess"
            ) as process,
            patch(
                "ml.alphazero_lite.run_uniform1200_exact_neighborhood_audit.sha256_file",
                return_value="pinned",
            ),
        ):
            process.return_value.request.return_value = native_result
            solve_exact(
                [solved, timeout],
                Path("probe"),
                Path("tablebase"),
                120,
                retry_timeouts=True,
            )
        process.assert_called_once()
        process.return_value.request.assert_called_once()
        self.assertEqual(42, solved["exact_value"])
        self.assertEqual(2, len(timeout["oracle_attempt_history"]))
        self.assertEqual("exact_solved", timeout["oracle_status"])
        self.assertEqual(
            "persistent_warm_completion",
            timeout["oracle_attempt_history"][-1]["attempt_class"],
        )
        self.assertEqual(
            native_result["metrics"],
            timeout["oracle_attempt_history"][-1]["native_metrics"],
        )

    def test_sharding_and_order_are_deterministic(self) -> None:
        rows = [
            self.row("z", 2, "exact_error"),
            self.row("a", 1, "exact_timeout"),
            self.row("b", 1, "exact_error"),
        ]
        rows[0]["provenance"] = [{"anchor_id": "beta-001"}]
        rows[1]["provenance"] = [{"anchor_id": "alpha-001"}]
        rows[2]["provenance"] = [{"anchor_id": "alpha-001"}]
        first = shard_tasks(rows, True, True, 4)
        second = shard_tasks(list(reversed(rows)), True, True, 4)
        self.assertEqual(
            {
                key: [row["canonical_state_key"] for row in value]
                for key, value in first.items()
            },
            {
                key: [row["canonical_state_key"] for row in value]
                for key, value in second.items()
            },
        )
        self.assertEqual(shard_for_row(rows[1], 4), shard_for_row(rows[2], 4))

    def test_timeout_replaces_process_and_increments_generation(self) -> None:
        timeout = self.row("a", 1, "exact_error")
        error = self.row("b", 1, "exact_error")
        solved = self.row("c", 1, "exact_error")
        response = {
            "action_values": {"0": 1},
            "optimal_actions": [0],
            "exact_value": 1,
            "metrics": {},
        }
        with (
            patch(
                "ml.alphazero_lite.run_uniform1200_exact_neighborhood_audit.NativeHybridProcess"
            ) as process,
            patch(
                "ml.alphazero_lite.run_uniform1200_exact_neighborhood_audit.sha256_file",
                return_value="pinned",
            ),
        ):
            process.return_value.request.side_effect = [
                TimeoutError("late"),
                response,
                response,
            ]
            solve_exact(
                [timeout, error, solved],
                Path("probe"),
                Path("tablebase"),
                120,
                True,
                True,
            )
        self.assertEqual(2, process.call_count)
        attempts = [
            row["oracle_attempt_history"][-1] for row in (timeout, error, solved)
        ]
        timed_out = next(
            attempt for attempt in attempts if attempt["status"] == "exact_timeout"
        )
        self.assertEqual(1, timed_out["process_generation"])
        self.assertTrue(any(attempt["process_generation"] == 2 for attempt in attempts))
        self.assertTrue(
            any(attempt["process_reused_from_previous_success"] for attempt in attempts)
        )

    def test_python_preflight_uses_known_solved_label_without_solver_import(
        self,
    ) -> None:
        row = self.row("known", 1, "exact_solved")
        row.update(
            {
                "exact_action_values": {0: 1},
                "exact_optimal_actions": [0],
                "exact_value": 1,
            }
        )
        preflight_python_label_path(row)

    def test_benchmark_cohort_is_deterministic_and_records_nearest_rule(self) -> None:
        rows = [
            self.row(f"solved-{index}", index % 3, "exact_solved") for index in range(4)
        ]
        for row in rows:
            row["state"]["player_pits"] = [34, 0, 0, 0, 0, 0]
            row["state"]["opponent_pits"] = [0, 0, 0, 0, 0, 0]
        rows.extend(
            [
                self.row("timeout-0", 0, "exact_timeout"),
                self.row("timeout-1", 1, "exact_timeout"),
            ]
        )
        first = benchmark_cohort(rows)
        second = benchmark_cohort(list(reversed(rows)))
        self.assertEqual(first["cohort_sha256"], second["cohort_sha256"])
        self.assertEqual(4, len(first["solved"]))
        self.assertIn("nearest deterministic equivalent", first["selection_rule"])

    def test_benchmark_metrics_and_gate_use_native_metrics(self) -> None:
        metrics = {
            "wall_time_seconds": 2.0,
            "cpu_time_seconds": 1.0,
            "forward_search_nodes": 10,
            "tt_probes": 4,
            "tt_hits": 2,
        }
        summary = benchmark_metrics(
            [{"status": "exact_solved", "native_metrics": metrics}]
        )
        self.assertEqual(1.0, summary["solve_rate"])
        self.assertEqual(0.5, summary["median_tt_hit_rate"])

        solved = self.row("solved", 1, "exact_solved")
        solved["state"]["player_pits"] = [34, 0, 0, 0, 0, 0]
        solved.update(
            {
                "exact_action_values": {0: 1},
                "exact_optimal_actions": [0],
                "exact_value": 1,
            }
        )
        with (
            patch(
                "ml.alphazero_lite.run_uniform1200_exact_neighborhood_audit.NativeHybridProcess"
            ),
            patch(
                "ml.alphazero_lite.run_uniform1200_exact_neighborhood_audit.sha256_file",
                return_value="pinned",
            ),
            patch(
                "ml.alphazero_lite.run_uniform1200_exact_neighborhood_audit._benchmark_attempt",
                side_effect=[
                    {
                        "status": "exact_solved",
                        "native_metrics": metrics | {"wall_time_seconds": 4.0},
                    },
                ],
            ),
            patch(
                "ml.alphazero_lite.run_uniform1200_exact_neighborhood_audit._run_persistent_shard",
                side_effect=lambda worker_id, shard, *_: [
                    (
                        row,
                        {
                            "status": "exact_solved",
                            "native_metrics": metrics | {"wall_time_seconds": 1.0},
                        },
                    )
                    for row in shard
                ],
            ),
        ):
            report = run_warm_vs_fresh_benchmark(
                [solved], Path("probe"), Path("tablebase")
            )
        self.assertTrue(report["go"])
        self.assertEqual("persistent_warm_oracle_effective", report["classification"])

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
        rates["1"]["solve_rate"] = 1.0
        self.assertTrue(original_coverage_gate(rates))
        rates["0"]["solve_rate"] = 0.98
        self.assertFalse(original_coverage_gate(rates))

    def test_preflight_rejects_noncanonical_tablebase_before_requests(self) -> None:
        with self.assertRaises(ValueError):
            preflight_native_oracle(Path("missing-probe"), Path("missing-tablebase"))


if __name__ == "__main__":
    unittest.main()
