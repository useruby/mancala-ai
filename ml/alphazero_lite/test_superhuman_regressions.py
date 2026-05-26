import unittest
from pathlib import Path
from unittest import mock

from ml.alphazero_lite import superhuman_regressions


class SuperhumanRegressionsTest(unittest.TestCase):
    def test_evaluate_regression_positions_defaults_omitted_simulations_to_384(self):
        positions = [
            {
                "id": "capture-1",
                "description": "Prefer the capture move.",
                "state": {
                    "player_pits": [1, 8, 0, 1, 1, 0],
                    "opponent_pits": [1, 3, 5, 0, 0, 1],
                    "player_store": 20,
                    "opponent_store": 7,
                    "current_player": 1,
                },
                "expected_move": 1,
            }
        ]

        with mock.patch(
            "ml.alphazero_lite.superhuman_regressions.arena.evaluate_artifact_position",
            return_value={"selected_move": 1},
        ) as evaluate_position:
            results = superhuman_regressions.evaluate_regression_positions(
                positions=positions,
                artifact_path="candidate-artifact",
                simulations=None,
                seed=17,
                c_puct=1.25,
            )

        self.assertTrue(results[0]["passed"])
        self.assertEqual(384, evaluate_position.call_args.kwargs["simulations"])

    def test_load_regression_positions_reads_fixture(self):
        fixture_path = (
            Path(__file__).resolve().parents[2]
            / "test/fixtures/ai/superhuman_regression_positions.json"
        )

        positions = superhuman_regressions.load_regression_positions(fixture_path)

        matching_positions = [
            position
            for position in positions
            if position.get("id") == "missed_capture_f67bd4k0_move_28"
        ]

        self.assertEqual(1, len(matching_positions))
        self.assertEqual(1, matching_positions[0]["expected_move"])
        self.assertEqual([1], matching_positions[0]["acceptable_moves"])

    def test_evaluate_positions_builds_report_and_comparisons(self):
        positions = [
            {
                "id": "capture-1",
                "description": "Prefer the capture move.",
                "state": {
                    "player_pits": [1, 8, 0, 1, 1, 0],
                    "opponent_pits": [1, 3, 5, 0, 0, 1],
                    "player_store": 20,
                    "opponent_store": 7,
                    "current_player": 1,
                },
                "expected_move": 1,
                "acceptable_moves": [1],
                "token": "f67bd4k0",
                "move_number": 28,
            }
        ]
        search_options = superhuman_regressions.build_search_options(
            reuse_subtree=True,
            root_policy_mode="deterministic",
            tactical_root_bias=0.1,
        )

        with mock.patch(
            "ml.alphazero_lite.superhuman_regressions.arena.evaluate_artifact_position",
            side_effect=[
                {"selected_move": 0, "policy": [1.0, 0.0], "value": -0.25},
                {"selected_move": 1, "policy": [0.0, 1.0], "value": 0.5},
            ],
        ) as evaluate_position:
            baseline_results = superhuman_regressions.evaluate_regression_positions(
                positions=positions,
                artifact_path="baseline-artifact",
                simulations=384,
                seed=17,
                c_puct=1.25,
                search_options=search_options,
            )
            candidate_results = superhuman_regressions.evaluate_regression_positions(
                positions=positions,
                artifact_path="candidate-artifact",
                simulations=384,
                seed=17,
                c_puct=1.25,
                search_options=search_options,
            )

        report = superhuman_regressions.build_regression_report(
            artifact_path="baseline-artifact",
            positions_path="positions.json",
            results=baseline_results,
        )
        comparisons = superhuman_regressions.compare_regression_results(
            baseline_results=baseline_results,
            candidate_results=candidate_results,
        )

        self.assertEqual(2, evaluate_position.call_count)
        self.assertEqual("capture-1", baseline_results[0]["id"])
        self.assertFalse(baseline_results[0]["passed"])
        self.assertTrue(candidate_results[0]["passed"])
        self.assertEqual("f67bd4k0", baseline_results[0]["token"])
        self.assertEqual(28, baseline_results[0]["move_number"])
        self.assertEqual("baseline-artifact", report["artifact_path"])
        self.assertEqual("positions.json", report["positions_path"])
        self.assertFalse(report["passed"])
        self.assertEqual(0, comparisons[0]["baseline_selected_move"])
        self.assertEqual(1, comparisons[0]["candidate_selected_move"])
        self.assertFalse(comparisons[0]["baseline_passed"])
        self.assertTrue(comparisons[0]["candidate_passed"])
        self.assertTrue(comparisons[0]["improved"])
        self.assertFalse(comparisons[0]["regressed"])

    def test_compare_regression_results_rejects_duplicate_candidate_ids(self):
        baseline_results = [
            {
                "id": "capture-1",
                "description": "Prefer the capture move.",
                "expected_move": 1,
                "acceptable_moves": [1],
                "selected_move": 0,
                "passed": False,
            }
        ]
        candidate_results = [
            {
                "id": "capture-1",
                "description": "Prefer the capture move.",
                "expected_move": 1,
                "acceptable_moves": [1],
                "selected_move": 1,
                "passed": True,
            },
            {
                "id": "capture-1",
                "description": "Prefer the capture move.",
                "expected_move": 1,
                "acceptable_moves": [1],
                "selected_move": 1,
                "passed": True,
            },
        ]

        with self.assertRaisesRegex(ValueError, "duplicate regression result id"):
            superhuman_regressions.compare_regression_results(
                baseline_results=baseline_results,
                candidate_results=candidate_results,
            )

    def test_compare_regression_results_rejects_mismatched_id_sets(self):
        baseline_results = [
            {
                "id": "capture-1",
                "description": "Prefer the capture move.",
                "expected_move": 1,
                "acceptable_moves": [1],
                "selected_move": 0,
                "passed": False,
            }
        ]
        candidate_results = [
            {
                "id": "capture-2",
                "description": "Prefer the capture move.",
                "expected_move": 1,
                "acceptable_moves": [1],
                "selected_move": 1,
                "passed": True,
            }
        ]

        with self.assertRaisesRegex(ValueError, "mismatched regression result ids"):
            superhuman_regressions.compare_regression_results(
                baseline_results=baseline_results,
                candidate_results=candidate_results,
            )

    def test_compare_regression_results_rejects_same_id_metadata_mismatch(self):
        baseline_results = [
            {
                "id": "capture-1",
                "description": "Prefer the capture move.",
                "expected_move": 1,
                "acceptable_moves": [1],
                "selected_move": 0,
                "passed": False,
            }
        ]
        candidate_results = [
            {
                "id": "capture-1",
                "description": "Prefer a different line.",
                "expected_move": 2,
                "acceptable_moves": [2],
                "selected_move": 2,
                "passed": True,
            }
        ]

        with self.assertRaisesRegex(
            ValueError, "mismatched regression metadata for id"
        ):
            superhuman_regressions.compare_regression_results(
                baseline_results=baseline_results,
                candidate_results=candidate_results,
            )

    def test_compare_regression_results_rejects_same_id_token_mismatch(self):
        baseline_results = [
            {
                "id": "capture-1",
                "description": "Prefer the capture move.",
                "expected_move": 1,
                "acceptable_moves": [1],
                "token": "f67bd4k0",
                "move_number": 28,
                "selected_move": 0,
                "passed": False,
            }
        ]
        candidate_results = [
            {
                "id": "capture-1",
                "description": "Prefer the capture move.",
                "expected_move": 1,
                "acceptable_moves": [1],
                "token": "other-token",
                "move_number": 28,
                "selected_move": 1,
                "passed": True,
            }
        ]

        with self.assertRaisesRegex(
            ValueError, "mismatched regression metadata for id"
        ):
            superhuman_regressions.compare_regression_results(
                baseline_results=baseline_results,
                candidate_results=candidate_results,
            )

    def test_compare_regression_results_rejects_same_id_move_number_mismatch(self):
        baseline_results = [
            {
                "id": "capture-1",
                "description": "Prefer the capture move.",
                "expected_move": 1,
                "acceptable_moves": [1],
                "token": "f67bd4k0",
                "move_number": 28,
                "selected_move": 0,
                "passed": False,
            }
        ]
        candidate_results = [
            {
                "id": "capture-1",
                "description": "Prefer the capture move.",
                "expected_move": 1,
                "acceptable_moves": [1],
                "token": "f67bd4k0",
                "move_number": 29,
                "selected_move": 1,
                "passed": True,
            }
        ]

        with self.assertRaisesRegex(
            ValueError, "mismatched regression metadata for id"
        ):
            superhuman_regressions.compare_regression_results(
                baseline_results=baseline_results,
                candidate_results=candidate_results,
            )

    def test_compare_regression_results_treats_missing_optional_semantic_metadata_consistently(
        self,
    ):
        baseline_results = [
            {
                "id": "capture-1",
                "description": "Prefer the capture move.",
                "expected_move": 1,
                "acceptable_moves": [1],
                "selected_move": 0,
                "passed": False,
            }
        ]
        candidate_results = [
            {
                "id": "capture-1",
                "description": "Prefer the capture move.",
                "expected_move": 1,
                "acceptable_moves": [1],
                "token": "",
                "move_number": None,
                "selected_move": 1,
                "passed": True,
            }
        ]

        comparisons = superhuman_regressions.compare_regression_results(
            baseline_results=baseline_results,
            candidate_results=candidate_results,
        )

        self.assertEqual(1, len(comparisons))
        self.assertTrue(comparisons[0]["improved"])

    def test_build_regression_report_marks_empty_results_as_not_passed(self):
        report = superhuman_regressions.build_regression_report(
            artifact_path="baseline-artifact",
            positions_path="positions.json",
            results=[],
        )

        self.assertFalse(report["passed"])
        self.assertEqual([], report["results"])

    def test_compare_regression_results_returns_empty_comparisons_for_empty_inputs(
        self,
    ):
        comparisons = superhuman_regressions.compare_regression_results(
            baseline_results=[],
            candidate_results=[],
        )

        self.assertEqual([], comparisons)


if __name__ == "__main__":
    unittest.main()
