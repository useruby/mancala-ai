import json
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite import diagnose_search_interaction as module


class DiagnoseSearchInteractionTest(unittest.TestCase):
    def test_build_matrix_reads_opening_prior_and_search_fields(self):
        opening_report = {
            "rows": [
                {
                    "id": "capture_available-002",
                    "bucket": "capture_available",
                    "phase": "opening",
                    "reference_move": 2,
                    "candidate_prior_summary": {"selected_move": 2, "value": 0.5681},
                    "candidate_searched_summary": {"selected_move": 0, "value": 0.5681},
                    "current_prior_summary": {"selected_move": 0, "value": 0.1135},
                    "current_searched_summary": {"selected_move": 2, "value": 0.1135},
                }
            ]
        }
        current_rows = {
            "capture_available-002": {
                "system_value": 0.1135,
                "value_error": 0.3893,
                "selected_move": 5,
                "teacher_value": 0.5028,
            }
        }
        original_rows = {
            "capture_available-002": {
                "system_value": 0.5441,
                "value_error": 0.0413,
                "selected_move": 4,
                "teacher_value": 0.5028,
            }
        }
        rebalanced_rows = {
            "capture_available-002": {
                "system_value": 0.5681,
                "value_error": 0.0653,
                "selected_move": 3,
                "teacher_value": 0.5028,
            }
        }

        row = module.build_opening_matrix_row(
            row_id="capture_available-002",
            opening_row=opening_report["rows"][0],
            current_row=current_rows["capture_available-002"],
            original_row=original_rows["capture_available-002"],
            rebalanced_row=rebalanced_rows["capture_available-002"],
            original_opening_row={
                "candidate_prior_summary": {"selected_move": 0},
                "candidate_searched_summary": {"selected_move": 0},
            },
            rebalanced_opening_row=opening_report["rows"][0],
        )

        self.assertEqual("capture_available-002", row["row_id"])
        self.assertEqual(0, row["current_prior_move"])
        self.assertEqual(2, row["current_searched_move"])
        self.assertEqual(0, row["original_challenger_prior_move"])
        self.assertEqual(0, row["original_challenger_searched_move"])
        self.assertEqual(2, row["rebalanced_challenger_prior_move"])
        self.assertEqual(0, row["rebalanced_challenger_searched_move"])

    def test_classify_mechanism_marks_search_override_when_value_is_good(self):
        row = {
            "row_id": "capture_available-003",
            "current_prior_move": 1,
            "current_searched_move": 2,
            "original_challenger_prior_move": 2,
            "original_challenger_searched_move": 2,
            "rebalanced_challenger_prior_move": 2,
            "rebalanced_challenger_searched_move": 1,
            "reference_move": 2,
            "current_system": {"value_error": 0.3974},
            "original_challenger_system": {"value_error": 0.0243, "value": 0.5710},
            "rebalanced_challenger_system": {"value_error": 0.0242, "value": 0.5709},
            "teacher_value": 0.5467,
            "phase": "opening",
            "bucket": "capture_available",
        }

        self.assertEqual("search_overrides_prior", module.classify_mechanism(row))

    def test_classify_mechanism_marks_value_sign_miscalibration(self):
        row = {
            "row_id": "high_imbalance-010",
            "reference_move": 1,
            "current_searched_move": 1,
            "original_challenger_searched_move": 1,
            "rebalanced_challenger_searched_move": 5,
            "teacher_value": -0.3133,
            "current_system": {"value": -0.0257, "value_error": 0.2876},
            "original_challenger_system": {"value": 0.3105, "value_error": 0.6238},
            "rebalanced_challenger_system": {"value": 0.3354, "value_error": 0.6487},
            "phase": "opening",
            "bucket": "high_imbalance",
        }

        self.assertEqual("value_sign_miscalibration", module.classify_mechanism(row))

    def test_choose_next_branch_prefers_search_then_value_then_endgame(self):
        matrix = [
            {"mechanism": "search_overrides_prior", "row_id": "capture_available-002"},
            {"mechanism": "search_overrides_prior", "row_id": "capture_available-003"},
            {"mechanism": "value_sign_miscalibration", "row_id": "high_imbalance-010"},
            {"mechanism": "persistent_late_game_weakness", "row_id": "sparse_endgame-009"},
        ]

        summary = module.choose_next_branch(matrix)

        self.assertEqual("search_interaction_diagnostic", summary["next_branch"])
        self.assertEqual(["capture_available-002", "capture_available-003"], summary["priority_rows"])
        self.assertIn("high_imbalance-010", summary["followup_rows"])
        self.assertIn("sparse_endgame-009", summary["separate_track_rows"])

    def test_choose_next_branch_uses_value_branch_for_value_only_matrix(self):
        matrix = [
            {"mechanism": "value_sign_miscalibration", "row_id": "high_imbalance-010"},
            {"mechanism": "value_sign_miscalibration", "row_id": "high_imbalance-011"},
        ]

        summary = module.choose_next_branch(matrix)

        self.assertEqual("value_calibration_diagnostic", summary["next_branch"])
        self.assertEqual([], summary["priority_rows"])
        self.assertEqual(["high_imbalance-010", "high_imbalance-011"], summary["followup_rows"])
        self.assertEqual([], summary["separate_track_rows"])

    def test_choose_next_branch_uses_endgame_branch_for_late_only_matrix(self):
        matrix = [
            {"mechanism": "persistent_late_game_weakness", "row_id": "sparse_endgame-009"},
        ]

        summary = module.choose_next_branch(matrix)

        self.assertEqual("endgame_isolation_diagnostic", summary["next_branch"])
        self.assertEqual([], summary["priority_rows"])
        self.assertEqual([], summary["followup_rows"])
        self.assertEqual(["sparse_endgame-009"], summary["separate_track_rows"])

    def test_build_matrix_from_runs_scopes_to_targeted_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            original_run = tmp_path / "original"
            rebalanced_run = tmp_path / "rebalanced"
            (original_run / "final").mkdir(parents=True)
            (rebalanced_run / "final").mkdir(parents=True)

            current_rows = [
                {"id": "capture_available-002", "bucket": "capture_available", "phase": "opening", "reference_move": 2, "selected_move": 2, "teacher_value": 0.50, "system_value": 0.11, "value_error": 0.39},
                {"id": "capture_available-003", "bucket": "capture_available", "phase": "opening", "reference_move": 2, "selected_move": 2, "teacher_value": 0.54, "system_value": 0.15, "value_error": 0.40},
                {"id": "capture_available-007", "bucket": "capture_available", "phase": "opening", "reference_move": 2, "selected_move": 2, "teacher_value": 0.57, "system_value": 0.12, "value_error": 0.41},
                {"id": "early_extra_turn-014", "bucket": "early_extra_turn", "phase": "mid", "reference_move": 1, "selected_move": 5, "teacher_value": -0.29, "system_value": -0.01, "value_error": 0.29},
                {"id": "high_imbalance-010", "bucket": "high_imbalance", "phase": "opening", "reference_move": 1, "selected_move": 1, "teacher_value": -0.31, "system_value": -0.03, "value_error": 0.29},
                {"id": "high_imbalance-011", "bucket": "high_imbalance", "phase": "opening", "reference_move": 1, "selected_move": 1, "teacher_value": -0.10, "system_value": -0.20, "value_error": 0.09},
                {"id": "high_imbalance-019", "bucket": "high_imbalance", "phase": "opening", "reference_move": 3, "selected_move": 4, "teacher_value": -0.75, "system_value": -0.16, "value_error": 0.59},
                {"id": "incumbent_proxy_disagreement-031", "bucket": "incumbent_proxy_disagreement", "phase": "mid", "reference_move": 4, "selected_move": 4, "teacher_value": 0.65, "system_value": 0.17, "value_error": 0.49},
                {"id": "incumbent_proxy_disagreement-033", "bucket": "incumbent_proxy_disagreement", "phase": "mid", "reference_move": 4, "selected_move": 0, "teacher_value": 0.65, "system_value": 0.26, "value_error": 0.39},
                {"id": "opening_plies_1_8-010", "bucket": "opening_plies_1_8", "phase": "opening", "reference_move": 1, "selected_move": 2, "teacher_value": 0.41, "system_value": 0.06, "value_error": 0.35},
                {"id": "opening_plies_1_8-057", "bucket": "opening_plies_1_8", "phase": "opening", "reference_move": 5, "selected_move": 2, "teacher_value": 0.45, "system_value": 0.17, "value_error": 0.28},
                {"id": "sparse_endgame-009", "bucket": "sparse_endgame", "phase": "late", "reference_move": 5, "selected_move": 2, "teacher_value": 0.70, "system_value": -0.12, "value_error": 0.81},
                {"id": "sparse_endgame-024", "bucket": "sparse_endgame", "phase": "late", "reference_move": 1, "selected_move": 5, "teacher_value": 0.30, "system_value": 0.05, "value_error": 0.25},
            ]

            original_challenger_rows = [
                {"id": "capture_available-002", "bucket": "capture_available", "phase": "opening", "reference_move": 2, "selected_move": 0, "teacher_value": 0.50, "system_value": 0.54, "value_error": 0.04},
                {"id": "capture_available-003", "bucket": "capture_available", "phase": "opening", "reference_move": 2, "selected_move": 2, "teacher_value": 0.54, "system_value": 0.57, "value_error": 0.02},
                {"id": "capture_available-007", "bucket": "capture_available", "phase": "opening", "reference_move": 2, "selected_move": 2, "teacher_value": 0.57, "system_value": 0.58, "value_error": 0.03},
                {"id": "early_extra_turn-014", "bucket": "early_extra_turn", "phase": "mid", "reference_move": 1, "selected_move": 1, "teacher_value": -0.29, "system_value": 0.39, "value_error": 0.68},
                {"id": "high_imbalance-010", "bucket": "high_imbalance", "phase": "opening", "reference_move": 1, "selected_move": 1, "teacher_value": -0.31, "system_value": 0.31, "value_error": 0.62},
                {"id": "high_imbalance-011", "bucket": "high_imbalance", "phase": "opening", "reference_move": 1, "selected_move": 1, "teacher_value": -0.10, "system_value": 0.24, "value_error": 0.34},
                {"id": "high_imbalance-019", "bucket": "high_imbalance", "phase": "opening", "reference_move": 3, "selected_move": 3, "teacher_value": -0.75, "system_value": 0.28, "value_error": 1.03},
                {"id": "incumbent_proxy_disagreement-031", "bucket": "incumbent_proxy_disagreement", "phase": "mid", "reference_move": 4, "selected_move": 4, "teacher_value": 0.65, "system_value": 0.53, "value_error": 0.12},
                {"id": "incumbent_proxy_disagreement-033", "bucket": "incumbent_proxy_disagreement", "phase": "mid", "reference_move": 4, "selected_move": 4, "teacher_value": 0.65, "system_value": 0.57, "value_error": 0.08},
                {"id": "opening_plies_1_8-010", "bucket": "opening_plies_1_8", "phase": "opening", "reference_move": 1, "selected_move": 1, "teacher_value": 0.41, "system_value": 0.30, "value_error": 0.11},
                {"id": "opening_plies_1_8-057", "bucket": "opening_plies_1_8", "phase": "opening", "reference_move": 5, "selected_move": 5, "teacher_value": 0.45, "system_value": 0.45, "value_error": 0.00},
                {"id": "sparse_endgame-009", "bucket": "sparse_endgame", "phase": "late", "reference_move": 5, "selected_move": 0, "teacher_value": 0.70, "system_value": 0.32, "value_error": 0.37},
                {"id": "sparse_endgame-024", "bucket": "sparse_endgame", "phase": "late", "reference_move": 1, "selected_move": 5, "teacher_value": 0.30, "system_value": 0.33, "value_error": 0.02},
            ]

            rebalanced_challenger_rows = [
                {"id": "capture_available-002", "bucket": "capture_available", "phase": "opening", "reference_move": 2, "selected_move": 0, "teacher_value": 0.50, "system_value": 0.57, "value_error": 0.07},
                {"id": "capture_available-003", "bucket": "capture_available", "phase": "opening", "reference_move": 2, "selected_move": 1, "teacher_value": 0.54, "system_value": 0.57, "value_error": 0.02},
                {"id": "capture_available-007", "bucket": "capture_available", "phase": "opening", "reference_move": 2, "selected_move": 0, "teacher_value": 0.57, "system_value": 0.58, "value_error": 0.03},
                {"id": "early_extra_turn-014", "bucket": "early_extra_turn", "phase": "mid", "reference_move": 1, "selected_move": 0, "teacher_value": -0.29, "system_value": 0.38, "value_error": 0.67},
                {"id": "high_imbalance-010", "bucket": "high_imbalance", "phase": "opening", "reference_move": 1, "selected_move": 5, "teacher_value": -0.31, "system_value": 0.34, "value_error": 0.65},
                {"id": "high_imbalance-011", "bucket": "high_imbalance", "phase": "opening", "reference_move": 1, "selected_move": 3, "teacher_value": -0.10, "system_value": 0.25, "value_error": 0.36},
                {"id": "high_imbalance-019", "bucket": "high_imbalance", "phase": "opening", "reference_move": 3, "selected_move": 2, "teacher_value": -0.75, "system_value": 0.29, "value_error": 1.05},
                {"id": "incumbent_proxy_disagreement-031", "bucket": "incumbent_proxy_disagreement", "phase": "mid", "reference_move": 4, "selected_move": 2, "teacher_value": 0.65, "system_value": 0.57, "value_error": 0.09},
                {"id": "incumbent_proxy_disagreement-033", "bucket": "incumbent_proxy_disagreement", "phase": "mid", "reference_move": 4, "selected_move": 0, "teacher_value": 0.65, "system_value": 0.58, "value_error": 0.07},
                {"id": "opening_plies_1_8-010", "bucket": "opening_plies_1_8", "phase": "opening", "reference_move": 1, "selected_move": 5, "teacher_value": 0.41, "system_value": 0.33, "value_error": 0.08},
                {"id": "opening_plies_1_8-057", "bucket": "opening_plies_1_8", "phase": "opening", "reference_move": 5, "selected_move": 2, "teacher_value": 0.45, "system_value": 0.46, "value_error": 0.01},
                {"id": "sparse_endgame-009", "bucket": "sparse_endgame", "phase": "late", "reference_move": 5, "selected_move": 2, "teacher_value": 0.70, "system_value": 0.33, "value_error": 0.37},
                {"id": "sparse_endgame-024", "bucket": "sparse_endgame", "phase": "late", "reference_move": 1, "selected_move": 4, "teacher_value": 0.30, "system_value": 0.33, "value_error": 0.03},
            ]

            original_opening_rows = [
                {"id": "capture_available-002", "bucket": "capture_available", "phase": "opening", "reference_move": 2, "current_prior_summary": {"selected_move": 0}, "current_searched_summary": {"selected_move": 2}, "candidate_prior_summary": {"selected_move": 0}, "candidate_searched_summary": {"selected_move": 0}},
                {"id": "capture_available-003", "bucket": "capture_available", "phase": "opening", "reference_move": 2, "current_prior_summary": {"selected_move": 1}, "current_searched_summary": {"selected_move": 2}, "candidate_prior_summary": {"selected_move": 2}, "candidate_searched_summary": {"selected_move": 2}},
                {"id": "capture_available-007", "bucket": "capture_available", "phase": "opening", "reference_move": 2, "current_prior_summary": {"selected_move": 1}, "current_searched_summary": {"selected_move": 2}, "candidate_prior_summary": {"selected_move": 2}, "candidate_searched_summary": {"selected_move": 2}},
            ]
            rebalanced_opening_rows = [
                {"id": "capture_available-002", "bucket": "capture_available", "phase": "opening", "reference_move": 2, "current_prior_summary": {"selected_move": 0}, "current_searched_summary": {"selected_move": 2}, "candidate_prior_summary": {"selected_move": 2}, "candidate_searched_summary": {"selected_move": 0}},
                {"id": "capture_available-003", "bucket": "capture_available", "phase": "opening", "reference_move": 2, "current_prior_summary": {"selected_move": 1}, "current_searched_summary": {"selected_move": 2}, "candidate_prior_summary": {"selected_move": 2}, "candidate_searched_summary": {"selected_move": 1}},
                {"id": "capture_available-007", "bucket": "capture_available", "phase": "opening", "reference_move": 2, "current_prior_summary": {"selected_move": 1}, "current_searched_summary": {"selected_move": 2}, "candidate_prior_summary": {"selected_move": 2}, "candidate_searched_summary": {"selected_move": 0}},
            ]

            stable_summary = {
                "schema": "azlite_stable_failure_family_summary_v1",
                "capture_available": {"search_flipped_ids": ["capture_available-002", "capture_available-003"]},
            }

            original_forensics = {"systems": {"current": {"rows": current_rows}, "challenger": {"rows": original_challenger_rows}}}
            rebalanced_forensics = {"systems": {"current": {"rows": current_rows}, "challenger": {"rows": rebalanced_challenger_rows}}}

            (original_run / "final" / "selected_candidate_forensics.json").write_text(json.dumps(original_forensics), encoding="utf-8")
            (rebalanced_run / "final" / "selected_candidate_forensics.json").write_text(json.dumps(rebalanced_forensics), encoding="utf-8")
            (original_run / "final" / "opening_capture_family_report.json").write_text(json.dumps({"rows": original_opening_rows}), encoding="utf-8")
            (rebalanced_run / "final" / "opening_capture_family_report.json").write_text(json.dumps({"rows": rebalanced_opening_rows}), encoding="utf-8")
            (rebalanced_run / "final" / "stable_failure_family_summary.json").write_text(json.dumps(stable_summary), encoding="utf-8")

            matrix = module.build_matrix_from_runs(original_run=original_run, rebalanced_run=rebalanced_run)
            summary = module.choose_next_branch(matrix)

        self.assertEqual(
            [
                "capture_available-002",
                "capture_available-003",
                "high_imbalance-010",
                "high_imbalance-011",
                "high_imbalance-019",
                "incumbent_proxy_disagreement-031",
                "incumbent_proxy_disagreement-033",
                "opening_plies_1_8-057",
                "sparse_endgame-009",
            ],
            [row["row_id"] for row in matrix],
        )
        self.assertEqual(
            [
                "capture_available-002",
                "capture_available-003",
                "incumbent_proxy_disagreement-031",
                "incumbent_proxy_disagreement-033",
                "opening_plies_1_8-057",
            ],
            summary["priority_rows"],
        )
        self.assertEqual(["high_imbalance-010", "high_imbalance-011", "high_imbalance-019"], summary["followup_rows"])
        self.assertEqual(["sparse_endgame-009"], summary["separate_track_rows"])
