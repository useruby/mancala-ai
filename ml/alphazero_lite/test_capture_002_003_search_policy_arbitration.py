import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import capture_002_003_search_policy_arbitration as module


class Capture002003SearchPolicyArbitrationContractTest(unittest.TestCase):
    def test_schema_and_row_ids_are_stable(self):
        self.assertEqual(
            "azlite_capture_002_003_search_policy_arbitration_v1",
            module.SCHEMA,
        )
        self.assertEqual(
            ["capture_available-002", "capture_available-003"],
            module.ROW_IDS,
        )
        self.assertEqual(
            [
                "policy_prior_gap",
                "value_backup_gap",
                "search_amplification_gap",
                "state_specific_rule_collision",
                "unresolved",
            ],
            module.CLASSIFICATION_LABELS,
        )

    def test_decision_mapping_is_stable(self):
        self.assertEqual(
            {
                "policy_prior_gap": "write_search_adjustment_spec",
                "value_backup_gap": "write_value_backup_followup_spec",
                "search_amplification_gap": "write_search_adjustment_spec",
                "state_specific_rule_collision": "write_rule_collision_spec",
                "unresolved": "stop_unresolved",
            },
            module.CLASSIFICATION_DECISIONS,
        )

    def test_default_out_path_uses_rebalanced_final_directory(self):
        self.assertEqual(
            Path("/tmp/rebalanced/final/capture_002_003_search_policy_arbitration.json"),
            module.default_out_path(rebalanced_run_dir=Path("/tmp/rebalanced")),
        )


class Capture002003SearchPolicyArbitrationResolutionTest(unittest.TestCase):
    def write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_load_selected_artifact_requires_resolved_selection_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            (run_dir / "selection").mkdir(parents=True)
            self.write_json(
                run_dir / "selection" / "selection_manifest.json",
                {"selected_artifact": ""},
            )

            with self.assertRaisesRegex(ValueError, "selected artifact"):
                module.load_selected_artifact(run_dir=run_dir)

    def test_load_selected_artifact_persists_path_and_provenance_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            selected_path = Path(tmp) / "artifacts" / "candidate.bin"
            (run_dir / "selection").mkdir(parents=True)
            self.write_json(
                run_dir / "selection" / "selection_manifest.json",
                {
                    "selected_artifact": str(selected_path),
                    "selected_target": str(selected_path),
                    "selection_reason": "arena_winner",
                },
            )

            resolved = module.load_selected_artifact(run_dir=run_dir)

        self.assertEqual(
            {
                "path": str(selected_path),
                "provenance_source": "selection_manifest.selected_target",
                "selected_target": str(selected_path),
                "selected_artifact": str(selected_path),
            },
            resolved,
        )

    def test_load_selected_artifact_falls_back_to_selected_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            selected_path = Path(tmp) / "artifacts" / "candidate.bin"
            (run_dir / "selection").mkdir(parents=True)
            self.write_json(
                run_dir / "selection" / "selection_manifest.json",
                {
                    "selected_artifact": str(selected_path),
                    "selection_reason": "arena_winner",
                },
            )

            resolved = module.load_selected_artifact(run_dir=run_dir)

        self.assertEqual(
            {
                "path": str(selected_path),
                "provenance_source": "selection_manifest.selected_artifact",
                "selected_target": None,
                "selected_artifact": str(selected_path),
            },
            resolved,
        )

    def test_load_selected_artifact_rejects_conflicting_selected_target_and_selected_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            target_path = Path(tmp) / "artifacts" / "candidate-a.bin"
            artifact_path = Path(tmp) / "artifacts" / "candidate-b.bin"
            (run_dir / "selection").mkdir(parents=True)
            self.write_json(
                run_dir / "selection" / "selection_manifest.json",
                {
                    "selected_target": str(target_path),
                    "selected_artifact": str(artifact_path),
                },
            )

            resolved = module.load_selected_artifact(run_dir=run_dir)

        self.assertEqual(
            {
                "path": str(target_path),
                "provenance_source": "selection_manifest.selected_target",
                "selected_target": str(target_path),
                "selected_artifact": str(artifact_path),
            },
            resolved,
        )

    def test_load_selected_artifact_prefers_selected_target_when_paths_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            target_path = Path(tmp) / "versions" / "candidate-iter1"
            artifact_path = Path(tmp) / "selection" / "artifact"
            (run_dir / "selection").mkdir(parents=True)
            self.write_json(
                run_dir / "selection" / "selection_manifest.json",
                {
                    "selected_target": str(target_path),
                    "selected_artifact": str(artifact_path),
                },
            )

            resolved = module.load_selected_artifact(run_dir=run_dir)

        self.assertEqual(
            {
                "path": str(target_path),
                "provenance_source": "selection_manifest.selected_target",
                "selected_target": str(target_path),
                "selected_artifact": str(artifact_path),
            },
            resolved,
        )

    def test_resolve_rows_requires_exact_002_and_003_entries(self):
        rows = {
            "capture_available-002": {
                "id": "capture_available-002",
                "canonical_state": "state-002",
                "legal_moves": [0, 2, 4],
                "reference_move": 2,
                "state": {"row_id": "capture_available-002"},
            }
        }

        with self.assertRaisesRegex(ValueError, "capture_available-003"):
            module.resolve_rows(rows_by_id=rows)

    def test_resolve_rows_requires_capture_available_002_entry(self):
        rows = {
            "capture_available-003": {
                "id": "capture_available-003",
                "canonical_state": "state-003",
                "legal_moves": [1, 2, 5],
                "reference_move": 2,
                "state": {"row_id": "capture_available-003"},
            }
        }

        with self.assertRaisesRegex(ValueError, "capture_available-002"):
            module.resolve_rows(rows_by_id=rows)

    def test_resolve_rows_returns_exact_row_order_and_required_fields(self):
        rows = {
            "capture_available-003": {
                "id": "capture_available-003",
                "canonical_state": "state-003",
                "legal_moves": [1, 2, 5],
                "reference_move": 2,
                "state": {"row_id": "capture_available-003"},
            },
            "capture_available-002": {
                "id": "capture_available-002",
                "canonical_state": "state-002",
                "legal_moves": [0, 2, 4],
                "reference_move": 2,
                "state": {"row_id": "capture_available-002"},
            },
        }

        resolved = module.resolve_rows(rows_by_id=rows)

        self.assertEqual(
            [
                {
                    "id": "capture_available-002",
                    "canonical_state": "state-002",
                    "legal_moves": [0, 2, 4],
                    "reference_move": 2,
                    "state": {"row_id": "capture_available-002"},
                },
                {
                    "id": "capture_available-003",
                    "canonical_state": "state-003",
                    "legal_moves": [1, 2, 5],
                    "reference_move": 2,
                    "state": {"row_id": "capture_available-003"},
                },
            ],
            resolved,
        )

    def test_resolve_rows_rejects_row_id_field_mismatches(self):
        rows = {
            "capture_available-002": {
                "id": "capture_available-003",
                "canonical_state": "state-002",
                "legal_moves": [0, 2, 4],
                "reference_move": 2,
                "state": {"row_id": "capture_available-002"},
            },
            "capture_available-003": {
                "id": "capture_available-003",
                "canonical_state": "state-003",
                "legal_moves": [1, 2, 5],
                "reference_move": 2,
                "state": {"row_id": "capture_available-003"},
            },
        }

        with self.assertRaisesRegex(ValueError, "capture_available-002"):
            module.resolve_rows(rows_by_id=rows)

    def test_resolve_rows_requires_canonical_state_field(self):
        rows = {
            "capture_available-002": {
                "id": "capture_available-002",
                "legal_moves": [0, 2, 4],
                "reference_move": 2,
                "state": {"row_id": "capture_available-002"},
            },
            "capture_available-003": {
                "id": "capture_available-003",
                "canonical_state": "state-003",
                "legal_moves": [1, 2, 5],
                "reference_move": 2,
                "state": {"row_id": "capture_available-003"},
            },
        }

        with self.assertRaisesRegex(ValueError, "canonical_state"):
            module.resolve_rows(rows_by_id=rows)

    def test_resolve_rows_rejects_empty_legal_moves(self):
        rows = {
            "capture_available-002": {
                "id": "capture_available-002",
                "canonical_state": "state-002",
                "legal_moves": [],
                "reference_move": 2,
                "state": {"row_id": "capture_available-002"},
            },
            "capture_available-003": {
                "id": "capture_available-003",
                "canonical_state": "state-003",
                "legal_moves": [1, 2, 5],
                "reference_move": 2,
                "state": {"row_id": "capture_available-003"},
            },
        }

        with self.assertRaisesRegex(ValueError, "empty legal_moves"):
            module.resolve_rows(rows_by_id=rows)

    def test_resolve_rows_rejects_reference_move_outside_legal_moves(self):
        rows = {
            "capture_available-002": {
                "id": "capture_available-002",
                "canonical_state": "state-002",
                "legal_moves": [0, 2, 4],
                "reference_move": 3,
                "state": {"row_id": "capture_available-002"},
            },
            "capture_available-003": {
                "id": "capture_available-003",
                "canonical_state": "state-003",
                "legal_moves": [1, 2, 5],
                "reference_move": 2,
                "state": {"row_id": "capture_available-003"},
            },
        }

        with self.assertRaisesRegex(ValueError, "reference_move.*legal_moves"):
            module.resolve_rows(rows_by_id=rows)

    def test_load_rows_from_run_rejects_non_dict_row_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            self.write_json(
                run_dir / "final" / "selected_candidate_forensics.json",
                {"systems": {"current": {"rows": ["capture_available-002"]}}},
            )

            with self.assertRaisesRegex(ValueError, "forensic rows entries must be dicts"):
                module.load_rows_from_run(run_dir=run_dir)

    def test_load_rows_from_run_rejects_empty_row_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            self.write_json(
                run_dir / "final" / "selected_candidate_forensics.json",
                {"systems": {"current": {"rows": [{"id": ""}]}}},
            )

            with self.assertRaisesRegex(ValueError, "must contain non-empty id"):
                module.load_rows_from_run(run_dir=run_dir)

    def test_load_rows_from_run_rejects_duplicate_row_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            self.write_json(
                run_dir / "final" / "selected_candidate_forensics.json",
                {
                    "systems": {
                        "current": {
                            "rows": [
                                {"id": "capture_available-002"},
                                {"id": "capture_available-002"},
                            ]
                        }
                    }
                },
            )

            with self.assertRaisesRegex(ValueError, "duplicate forensic row id: capture_available-002"):
                module.load_rows_from_run(run_dir=run_dir)

    def test_resolve_rows_rejects_non_list_legal_moves(self):
        rows = {
            "capture_available-002": {
                "id": "capture_available-002",
                "canonical_state": "state-002",
                "legal_moves": "0,2,4",
                "reference_move": 2,
                "state": {"row_id": "capture_available-002"},
            },
            "capture_available-003": {
                "id": "capture_available-003",
                "canonical_state": "state-003",
                "legal_moves": [1, 2, 5],
                "reference_move": 2,
                "state": {"row_id": "capture_available-003"},
            },
        }

        with self.assertRaisesRegex(ValueError, "legal_moves must be a list"):
            module.resolve_rows(rows_by_id=rows)

    def test_resolve_rows_rejects_non_integer_legal_move_entries(self):
        rows = {
            "capture_available-002": {
                "id": "capture_available-002",
                "canonical_state": "state-002",
                "legal_moves": [0, "2", 4],
                "reference_move": 2,
                "state": {"row_id": "capture_available-002"},
            },
            "capture_available-003": {
                "id": "capture_available-003",
                "canonical_state": "state-003",
                "legal_moves": [1, 2, 5],
                "reference_move": 2,
                "state": {"row_id": "capture_available-003"},
            },
        }

        with self.assertRaisesRegex(ValueError, "legal_moves entries must be integers"):
            module.resolve_rows(rows_by_id=rows)

    def test_resolve_rows_rejects_boolean_legal_move_entries(self):
        rows = {
            "capture_available-002": {
                "id": "capture_available-002",
                "canonical_state": "state-002",
                "legal_moves": [0, True, 4],
                "reference_move": 4,
                "state": {"row_id": "capture_available-002"},
            },
            "capture_available-003": {
                "id": "capture_available-003",
                "canonical_state": "state-003",
                "legal_moves": [1, 2, 5],
                "reference_move": 2,
                "state": {"row_id": "capture_available-003"},
            },
        }

        with self.assertRaisesRegex(ValueError, "legal_moves entries must be integers"):
            module.resolve_rows(rows_by_id=rows)

    def test_resolve_rows_rejects_negative_legal_move_entries(self):
        rows = {
            "capture_available-002": {
                "id": "capture_available-002",
                "canonical_state": "state-002",
                "legal_moves": [-1, 2, 4],
                "reference_move": 2,
                "state": {"row_id": "capture_available-002"},
            },
            "capture_available-003": {
                "id": "capture_available-003",
                "canonical_state": "state-003",
                "legal_moves": [1, 2, 5],
                "reference_move": 2,
                "state": {"row_id": "capture_available-003"},
            },
        }

        with self.assertRaisesRegex(ValueError, "legal_moves entries must be non-negative"):
            module.resolve_rows(rows_by_id=rows)

    def test_resolve_rows_rejects_non_integer_reference_move(self):
        rows = {
            "capture_available-002": {
                "id": "capture_available-002",
                "canonical_state": "state-002",
                "legal_moves": [0, 2, 4],
                "reference_move": "2",
                "state": {"row_id": "capture_available-002"},
            },
            "capture_available-003": {
                "id": "capture_available-003",
                "canonical_state": "state-003",
                "legal_moves": [1, 2, 5],
                "reference_move": 2,
                "state": {"row_id": "capture_available-003"},
            },
        }

        with self.assertRaisesRegex(ValueError, "reference_move must be an integer"):
            module.resolve_rows(rows_by_id=rows)

    def test_resolve_rows_rejects_boolean_reference_move(self):
        rows = {
            "capture_available-002": {
                "id": "capture_available-002",
                "canonical_state": "state-002",
                "legal_moves": [0, 2, 4],
                "reference_move": True,
                "state": {"row_id": "capture_available-002"},
            },
            "capture_available-003": {
                "id": "capture_available-003",
                "canonical_state": "state-003",
                "legal_moves": [1, 2, 5],
                "reference_move": 2,
                "state": {"row_id": "capture_available-003"},
            },
        }

        with self.assertRaisesRegex(ValueError, "reference_move must be an integer"):
            module.resolve_rows(rows_by_id=rows)


class Capture002003SearchPolicyArbitrationViewsTest(unittest.TestCase):
    def test_build_row_views_extracts_policy_value_and_search_metrics(self):
        row_views = module.build_row_views(
            row={
                "id": "capture_available-002",
                "canonical_state": "state-002",
                "legal_moves": [0, 2, 4],
                "reference_move": 2,
            },
            probe_summary={
                "selected_move": 0,
                "value": 0.41,
                "policy": [0.45, 0.0, 0.35, 0.0, 0.20, 0.0],
                "visits": [28, 0, 20, 0, 12, 0],
                "child_stats": [
                    {"move": 0, "visits": 28, "q_value": 0.63},
                    {"move": 2, "visits": 20, "q_value": 0.51},
                    {"move": 4, "visits": 12, "q_value": 0.49},
                ],
            },
        )

        self.assertEqual(0, row_views["policy_view"]["top_move"])
        self.assertEqual(0.35, row_views["policy_view"]["reference_move_probability"])
        self.assertEqual(0.1, row_views["policy_view"]["selected_minus_reference_margin"])
        self.assertEqual(0.41, row_views["value_view"]["root_value_estimate"])
        self.assertEqual(0.51, row_views["value_view"]["reference_move_q_value"])
        self.assertEqual(0.63, row_views["value_view"]["selected_move_q_value"])
        self.assertEqual(0.12, row_views["value_view"]["selected_minus_reference_q_margin"])
        self.assertEqual(0, row_views["search_view"]["searched_selected_move"])
        self.assertEqual(0.3333, row_views["search_view"]["reference_move_visit_share"])
        self.assertEqual(0.4667, row_views["search_view"]["selected_move_visit_share"])
        self.assertTrue(row_views["search_view"]["child_stats_complete"])
        self.assertEqual([], row_views["search_view"]["missing_fields"])

    def test_build_row_views_preserves_root_telemetry(self):
        row_views = module.build_row_views(
            row={
                "id": "capture_available-002",
                "canonical_state": "state-002",
                "legal_moves": [0, 2, 4],
                "reference_move": 2,
            },
            probe_summary={
                "selected_move": 0,
                "value": 0.41,
                "policy": [0.45, 0.0, 0.35, 0.0, 0.20, 0.0],
                "visits": [28, 0, 20, 0, 12, 0],
                "child_stats": [
                    {"move": 0, "visits": 28, "q_value": 0.63},
                    {"move": 2, "visits": 20, "q_value": 0.51},
                    {"move": 4, "visits": 12, "q_value": 0.49},
                ],
                "selection_breakdown": {
                    "policy_top_move": 2,
                    "visit_top_move": 0,
                    "q_top_move": 0,
                },
                "visit_snapshots": [
                    {"simulation": 1, "selected_move": 2, "visits": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]},
                    {"simulation": 64, "selected_move": 0, "visits": [28.0, 0.0, 20.0, 0.0, 12.0, 0.0]},
                ],
            },
        )

        self.assertEqual(
            {
                "policy_top_move": 2,
                "visit_top_move": 0,
                "q_top_move": 0,
            },
            row_views["search_view"]["selection_breakdown"],
        )
        self.assertEqual(
            [
                {"simulation": 1, "selected_move": 2, "visits": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]},
                {"simulation": 64, "selected_move": 0, "visits": [28.0, 0.0, 20.0, 0.0, 12.0, 0.0]},
            ],
            row_views["search_view"]["visit_snapshots"],
        )

    def test_build_row_views_preserves_nulls_and_missing_fields_when_child_stats_are_missing(self):
        row_views = module.build_row_views(
            row={
                "id": "capture_available-003",
                "canonical_state": "state-003",
                "legal_moves": [1, 2, 5],
                "reference_move": 2,
            },
            probe_summary={
                "selected_move": 1,
                "value": 0.27,
                "policy": [0.0, 0.4, 0.35, 0.0, 0.0, 0.25],
                "visits": [0, 24, 21, 0, 0, 15],
                "child_stats": [],
            },
        )

        self.assertIsNone(row_views["value_view"]["per_child_q_values"])
        self.assertIsNone(row_views["value_view"]["reference_move_q_value"])
        self.assertIsNone(row_views["value_view"]["selected_move_q_value"])
        self.assertIsNone(row_views["value_view"]["selected_minus_reference_q_margin"])
        self.assertIsNone(row_views["search_view"]["child_stats"])
        self.assertFalse(row_views["search_view"]["child_stats_complete"])
        self.assertEqual(
            [
                "value_view.per_child_q_values",
                "value_view.reference_move_q_value",
                "value_view.selected_move_q_value",
                "value_view.selected_minus_reference_q_margin",
                "search_view.child_stats",
            ],
            row_views["search_view"]["missing_fields"],
        )

    def test_build_row_views_rejects_malformed_child_stats_entries(self):
        with self.assertRaisesRegex(ValueError, "child_stats.*q_value"):
            module.build_row_views(
                row={
                    "id": "capture_available-003",
                    "canonical_state": "state-003",
                    "legal_moves": [1, 2, 5],
                    "reference_move": 2,
                },
                probe_summary={
                    "selected_move": 1,
                    "value": 0.27,
                    "policy": [0.0, 0.4, 0.35, 0.0, 0.0, 0.25],
                    "visits": [0, 24, 21, 0, 0, 15],
                    "child_stats": [
                        {"move": 1, "visits": 24},
                        {"move": 2, "visits": 21, "q_value": 0.29},
                    ],
                },
            )

    def test_build_row_views_rejects_policy_container_that_is_not_list_or_tuple(self):
        with self.assertRaisesRegex(ValueError, "policy must be a list or tuple"):
            module.build_row_views(
                row={
                    "id": "capture_available-002",
                    "canonical_state": "state-002",
                    "legal_moves": [0, 2, 4],
                    "reference_move": 2,
                },
                probe_summary={
                    "selected_move": 0,
                    "value": 0.41,
                    "policy": {0: 0.45, 2: 0.35, 4: 0.2},
                    "visits": [28, 0, 20, 0, 12, 0],
                    "child_stats": [
                        {"move": 0, "visits": 28, "q_value": 0.63},
                        {"move": 2, "visits": 20, "q_value": 0.51},
                        {"move": 4, "visits": 12, "q_value": 0.49},
                    ],
                },
            )

    def test_build_row_views_rejects_visits_container_with_non_numeric_entries(self):
        with self.assertRaisesRegex(ValueError, "visits entries must be numeric or None"):
            module.build_row_views(
                row={
                    "id": "capture_available-002",
                    "canonical_state": "state-002",
                    "legal_moves": [0, 2, 4],
                    "reference_move": 2,
                },
                probe_summary={
                    "selected_move": 0,
                    "value": 0.41,
                    "policy": [0.45, 0.0, 0.35, 0.0, 0.20, 0.0],
                    "visits": [28, 0, "20", 0, 12, 0],
                    "child_stats": [
                        {"move": 0, "visits": 28, "q_value": 0.63},
                        {"move": 2, "visits": 20, "q_value": 0.51},
                        {"move": 4, "visits": 12, "q_value": 0.49},
                    ],
                },
            )

    def test_build_row_views_rejects_child_stats_container_that_is_not_a_list(self):
        with self.assertRaisesRegex(ValueError, "child_stats must be a list of dicts"):
            module.build_row_views(
                row={
                    "id": "capture_available-003",
                    "canonical_state": "state-003",
                    "legal_moves": [1, 2, 5],
                    "reference_move": 2,
                },
                probe_summary={
                    "selected_move": 1,
                    "value": 0.27,
                    "policy": [0.0, 0.4, 0.35, 0.0, 0.0, 0.25],
                    "visits": [0, 24, 21, 0, 0, 15],
                    "child_stats": {"move": 1, "visits": 24, "q_value": 0.31},
                },
            )

    def test_build_row_views_rejects_child_stats_entries_that_are_not_dicts(self):
        with self.assertRaisesRegex(ValueError, "child_stats entries must be dicts"):
            module.build_row_views(
                row={
                    "id": "capture_available-003",
                    "canonical_state": "state-003",
                    "legal_moves": [1, 2, 5],
                    "reference_move": 2,
                },
                probe_summary={
                    "selected_move": 1,
                    "value": 0.27,
                    "policy": [0.0, 0.4, 0.35, 0.0, 0.0, 0.25],
                    "visits": [0, 24, 21, 0, 0, 15],
                    "child_stats": [
                        {"move": 1, "visits": 24, "q_value": 0.31},
                        "bad-entry",
                    ],
                },
            )

    def test_build_row_views_rejects_selected_move_outside_legal_moves(self):
        with self.assertRaisesRegex(ValueError, "selected_move must be present in legal_moves"):
            module.build_row_views(
                row={
                    "id": "capture_available-002",
                    "canonical_state": "state-002",
                    "legal_moves": [0, 2, 4],
                    "reference_move": 2,
                },
                probe_summary={
                    "selected_move": 5,
                    "value": 0.41,
                    "policy": [0.45, 0.0, 0.35, 0.0, 0.20, 0.0],
                    "visits": [28, 0, 20, 0, 12, 0],
                    "child_stats": [
                        {"move": 0, "visits": 28, "q_value": 0.63},
                        {"move": 2, "visits": 20, "q_value": 0.51},
                        {"move": 4, "visits": 12, "q_value": 0.49},
                    ],
                },
            )

    def test_build_row_views_rejects_non_integer_selected_move(self):
        with self.assertRaisesRegex(ValueError, "selected_move must be an integer"):
            module.build_row_views(
                row={
                    "id": "capture_available-002",
                    "canonical_state": "state-002",
                    "legal_moves": [0, 2, 4],
                    "reference_move": 2,
                },
                probe_summary={
                    "selected_move": "0",
                    "value": 0.41,
                    "policy": [0.45, 0.0, 0.35, 0.0, 0.20, 0.0],
                    "visits": [28, 0, 20, 0, 12, 0],
                    "child_stats": [
                        {"move": 0, "visits": 28, "q_value": 0.63},
                        {"move": 2, "visits": 20, "q_value": 0.51},
                        {"move": 4, "visits": 12, "q_value": 0.49},
                    ],
                },
            )

    def test_build_row_views_rejects_boolean_selected_move(self):
        with self.assertRaisesRegex(ValueError, "selected_move must be an integer"):
            module.build_row_views(
                row={
                    "id": "capture_available-002",
                    "canonical_state": "state-002",
                    "legal_moves": [0, 2, 4],
                    "reference_move": 2,
                },
                probe_summary={
                    "selected_move": True,
                    "value": 0.41,
                    "policy": [0.45, 0.0, 0.35, 0.0, 0.20, 0.0],
                    "visits": [28, 0, 20, 0, 12, 0],
                    "child_stats": [
                        {"move": 0, "visits": 28, "q_value": 0.63},
                        {"move": 2, "visits": 20, "q_value": 0.51},
                        {"move": 4, "visits": 12, "q_value": 0.49},
                    ],
                },
            )

    def test_build_row_views_rejects_boolean_child_stat_visits(self):
        with self.assertRaisesRegex(ValueError, "child_stats entry at index 0 visits must be an integer"):
            module.build_row_views(
                row={
                    "id": "capture_available-002",
                    "canonical_state": "state-002",
                    "legal_moves": [0, 2, 4],
                    "reference_move": 2,
                },
                probe_summary={
                    "selected_move": 0,
                    "value": 0.41,
                    "policy": [0.45, 0.0, 0.35, 0.0, 0.20, 0.0],
                    "visits": [28, 0, 20, 0, 12, 0],
                    "child_stats": [
                        {"move": 0, "visits": True, "q_value": 0.63},
                        {"move": 2, "visits": 20, "q_value": 0.51},
                        {"move": 4, "visits": 12, "q_value": 0.49},
                    ],
                },
            )

    def test_build_row_views_rejects_fractional_child_stat_visits(self):
        with self.assertRaisesRegex(ValueError, "child_stats entry at index 0 visits must be an integer"):
            module.build_row_views(
                row={
                    "id": "capture_available-002",
                    "canonical_state": "state-002",
                    "legal_moves": [0, 2, 4],
                    "reference_move": 2,
                },
                probe_summary={
                    "selected_move": 0,
                    "value": 0.41,
                    "policy": [0.45, 0.0, 0.35, 0.0, 0.20, 0.0],
                    "visits": [28, 0, 20, 0, 12, 0],
                    "child_stats": [
                        {"move": 0, "visits": 28.5, "q_value": 0.63},
                        {"move": 2, "visits": 20, "q_value": 0.51},
                        {"move": 4, "visits": 12, "q_value": 0.49},
                    ],
                },
            )

    def test_build_row_views_rejects_negative_child_stat_visits(self):
        with self.assertRaisesRegex(ValueError, "child_stats entry at index 0 visits must be non-negative"):
            module.build_row_views(
                row={
                    "id": "capture_available-002",
                    "canonical_state": "state-002",
                    "legal_moves": [0, 2, 4],
                    "reference_move": 2,
                },
                probe_summary={
                    "selected_move": 0,
                    "value": 0.41,
                    "policy": [0.45, 0.0, 0.35, 0.0, 0.20, 0.0],
                    "visits": [28, 0, 20, 0, 12, 0],
                    "child_stats": [
                        {"move": 0, "visits": -1, "q_value": 0.63},
                        {"move": 2, "visits": 20, "q_value": 0.51},
                        {"move": 4, "visits": 12, "q_value": 0.49},
                    ],
                },
            )

    def test_build_row_views_rejects_child_stat_move_outside_legal_moves(self):
        with self.assertRaisesRegex(ValueError, "child_stats entry at index 1 move must be present in legal_moves"):
            module.build_row_views(
                row={
                    "id": "capture_available-002",
                    "canonical_state": "state-002",
                    "legal_moves": [0, 2, 4],
                    "reference_move": 2,
                },
                probe_summary={
                    "selected_move": 0,
                    "value": 0.41,
                    "policy": [0.45, 0.0, 0.35, 0.0, 0.20, 0.0],
                    "visits": [28, 0, 20, 0, 12, 0],
                    "child_stats": [
                        {"move": 0, "visits": 28, "q_value": 0.63},
                        {"move": 5, "visits": 20, "q_value": 0.51},
                        {"move": 4, "visits": 12, "q_value": 0.49},
                    ],
                },
            )

    def test_build_row_views_rejects_child_stats_missing_a_legal_move(self):
        row_views = module.build_row_views(
            row={
                "id": "capture_available-003",
                "canonical_state": "state-003",
                "legal_moves": [1, 2, 5],
                "reference_move": 2,
            },
            probe_summary={
                "selected_move": 1,
                "value": 0.27,
                "policy": [0.0, 0.4, 0.35, 0.0, 0.0, 0.25],
                "visits": [0, 24, 21, 0, 0, 15],
                "child_stats": [
                    {"move": 1, "visits": 24, "q_value": 0.31},
                    {"move": 2, "visits": 21, "q_value": 0.29},
                ],
            },
        )

        self.assertFalse(row_views["search_view"]["child_stats_complete"])
        self.assertIsNone(row_views["search_view"]["child_stats"])
        self.assertIn("search_view.child_stats", row_views["search_view"]["missing_fields"])
        self.assertIn("search_view.child_stats_partial", row_views["search_view"]["missing_fields"])

    def test_build_row_views_rejects_duplicate_child_stat_moves(self):
        row_views = module.build_row_views(
            row={
                "id": "capture_available-003",
                "canonical_state": "state-003",
                "legal_moves": [1, 2, 5],
                "reference_move": 2,
            },
            probe_summary={
                "selected_move": 1,
                "value": 0.27,
                "policy": [0.0, 0.4, 0.35, 0.0, 0.0, 0.25],
                "visits": [0, 24, 21, 0, 0, 15],
                "child_stats": [
                    {"move": 1, "visits": 24, "q_value": 0.31},
                    {"move": 1, "visits": 21, "q_value": 0.29},
                    {"move": 5, "visits": 15, "q_value": 0.18},
                ],
            },
        )

        self.assertFalse(row_views["search_view"]["child_stats_complete"])
        self.assertIsNone(row_views["search_view"]["child_stats"])
        self.assertIn("search_view.child_stats_partial", row_views["search_view"]["missing_fields"])

    def test_build_row_views_preserves_nulls_when_visit_totals_are_zero(self):
        row_views = module.build_row_views(
            row={
                "id": "capture_available-002",
                "canonical_state": "state-002",
                "legal_moves": [0, 2, 4],
                "reference_move": 2,
            },
            probe_summary={
                "selected_move": 0,
                "value": 0.18,
                "policy": [0.45, 0.0, 0.35, 0.0, 0.20, 0.0],
                "visits": [0, 0, 0, 0, 0, 0],
                "child_stats": [
                    {"move": 0, "visits": 0, "q_value": 0.22},
                    {"move": 2, "visits": 0, "q_value": 0.17},
                    {"move": 4, "visits": 0, "q_value": 0.11},
                ],
            },
        )

        self.assertEqual({"0": 0.0, "2": 0.0, "4": 0.0}, row_views["search_view"]["visit_distribution"])
        self.assertIsNone(row_views["search_view"]["reference_move_visit_share"])
        self.assertIsNone(row_views["search_view"]["selected_move_visit_share"])
        self.assertEqual(
            [
                "search_view.reference_move_visit_share",
                "search_view.selected_move_visit_share",
            ],
            row_views["search_view"]["missing_fields"],
        )

    def test_build_row_views_preserves_nulls_when_policy_or_visits_are_missing_or_short(self):
        row_views = module.build_row_views(
            row={
                "id": "capture_available-003",
                "canonical_state": "state-003",
                "legal_moves": [1, 2, 5],
                "reference_move": 2,
            },
            probe_summary={
                "selected_move": 1,
                "value": 0.27,
                "visits": [0.0, 24.0, 21.0],
                "child_stats": [
                    {"move": 1, "visits": 24, "q_value": 0.31},
                    {"move": 2, "visits": 21, "q_value": 0.29},
                    {"move": 5, "visits": 15, "q_value": 0.18},
                ],
            },
        )

        self.assertIsNone(row_views["policy_view"]["raw_policy_distribution"])
        self.assertIsNone(row_views["policy_view"]["top_move"])
        self.assertIsNone(row_views["policy_view"]["reference_move_probability"])
        self.assertIsNone(row_views["policy_view"]["selected_minus_reference_margin"])
        self.assertIsNone(row_views["search_view"]["visit_distribution"])
        self.assertIsNone(row_views["search_view"]["reference_move_visit_share"])
        self.assertIsNone(row_views["search_view"]["selected_move_visit_share"])
        self.assertEqual(
            [
                "policy_view.raw_policy_distribution",
                "search_view.visit_distribution",
                "search_view.reference_move_visit_share",
                "search_view.selected_move_visit_share",
            ],
            row_views["search_view"]["missing_fields"],
        )

    def test_build_row_views_preserves_nulls_when_value_is_missing(self):
        row_views = module.build_row_views(
            row={
                "id": "capture_available-002",
                "canonical_state": "state-002",
                "legal_moves": [0, 2, 4],
                "reference_move": 2,
            },
            probe_summary={
                "selected_move": 4,
                "policy": [0.45, 0.0, 0.35, 0.0, 0.20, 0.0],
                "visits": [28, 0, 20, 0, 12, 0],
                "child_stats": [
                    {"move": 0, "visits": 28, "q_value": 0.63},
                    {"move": 2, "visits": 20, "q_value": 0.51},
                    {"move": 4, "visits": 12, "q_value": 0.49},
                ],
            },
        )

        self.assertIsNone(row_views["value_view"]["root_value_estimate"])
        self.assertEqual(
            ["value_view.root_value_estimate"],
            row_views["search_view"]["missing_fields"],
        )

    def test_build_row_views_rejects_non_numeric_probe_value(self):
        with self.assertRaisesRegex(ValueError, "value must be numeric"):
            module.build_row_views(
                row={
                    "id": "capture_available-002",
                    "canonical_state": "state-002",
                    "legal_moves": [0, 2, 4],
                    "reference_move": 2,
                },
                probe_summary={
                    "selected_move": 0,
                    "value": True,
                    "policy": [0.45, 0.0, 0.35, 0.0, 0.20, 0.0],
                    "visits": [28, 0, 20, 0, 12, 0],
                    "child_stats": [
                        {"move": 0, "visits": 28, "q_value": 0.63},
                        {"move": 2, "visits": 20, "q_value": 0.51},
                        {"move": 4, "visits": 12, "q_value": 0.49},
                    ],
                },
            )

    def test_build_row_views_rejects_non_finite_probe_value(self):
        with self.assertRaisesRegex(ValueError, "value must be numeric"):
            module.build_row_views(
                row={
                    "id": "capture_available-002",
                    "canonical_state": "state-002",
                    "legal_moves": [0, 2, 4],
                    "reference_move": 2,
                },
                probe_summary={
                    "selected_move": 0,
                    "value": float("nan"),
                    "policy": [0.45, 0.0, 0.35, 0.0, 0.20, 0.0],
                    "visits": [28, 0, 20, 0, 12, 0],
                    "child_stats": [
                        {"move": 0, "visits": 28, "q_value": 0.63},
                        {"move": 2, "visits": 20, "q_value": 0.51},
                        {"move": 4, "visits": 12, "q_value": 0.49},
                    ],
                },
            )

    def test_build_row_views_rejects_non_finite_policy_entries(self):
        with self.assertRaisesRegex(ValueError, "policy entries must be numeric or None"):
            module.build_row_views(
                row={
                    "id": "capture_available-002",
                    "canonical_state": "state-002",
                    "legal_moves": [0, 2, 4],
                    "reference_move": 2,
                },
                probe_summary={
                    "selected_move": 0,
                    "value": 0.41,
                    "policy": [0.45, 0.0, float("inf"), 0.0, 0.20, 0.0],
                    "visits": [28, 0, 20, 0, 12, 0],
                    "child_stats": [
                        {"move": 0, "visits": 28, "q_value": 0.63},
                        {"move": 2, "visits": 20, "q_value": 0.51},
                        {"move": 4, "visits": 12, "q_value": 0.49},
                    ],
                },
            )


class Capture002003SearchPolicyArbitrationProbePayloadTest(unittest.TestCase):
    def test_validated_diagnostic_state_rejects_invalid_embedded_current_player_override(self):
        row = {
            "id": "capture_available-002",
            "canonical_state": json.dumps(
                {
                    "current_player": 1,
                    "player_pits": [0, 1, 0, 0, 0, 0],
                    "opponent_pits": [0, 0, 0, 2, 0, 1],
                }
            ),
            "state": {"current_player": "oops"},
        }

        with self.assertRaisesRegex(
            ValueError,
            "usable state.*current_player.*player_pits.*opponent_pits",
        ):
            module.validated_diagnostic_state(row=row)

    def test_validated_diagnostic_state_rejects_invalid_embedded_non_empty_pit_override(self):
        row = {
            "id": "capture_available-002",
            "canonical_state": json.dumps(
                {
                    "current_player": 1,
                    "player_pits": [0, 1, 0, 0, 0, 0],
                    "opponent_pits": [0, 0, 0, 2, 0, 1],
                }
            ),
            "state": {"player_pits": [0, -1, 0]},
        }

        with self.assertRaisesRegex(
            ValueError,
            "usable state.*current_player.*player_pits.*opponent_pits",
        ):
            module.validated_diagnostic_state(row=row)

    def test_validated_diagnostic_state_rejects_explicit_empty_embedded_player_pits_override(self):
        row = {
            "id": "capture_available-002",
            "canonical_state": json.dumps(
                {
                    "current_player": 1,
                    "player_pits": [0, 1, 0, 0, 0, 0],
                    "opponent_pits": [0, 0, 0, 2, 0, 1],
                }
            ),
            "state": {"player_pits": []},
        }

        with self.assertRaisesRegex(
            ValueError,
            "usable state.*current_player.*player_pits.*opponent_pits",
        ):
            module.validated_diagnostic_state(row=row)

    def test_validated_diagnostic_state_rejects_explicit_empty_embedded_opponent_pits_override(self):
        row = {
            "id": "capture_available-002",
            "canonical_state": json.dumps(
                {
                    "current_player": 1,
                    "player_pits": [0, 1, 0, 0, 0, 0],
                    "opponent_pits": [0, 0, 0, 2, 0, 1],
                }
            ),
            "state": {"opponent_pits": ()},
        }

        with self.assertRaisesRegex(
            ValueError,
            "usable state.*current_player.*player_pits.*opponent_pits",
        ):
            module.validated_diagnostic_state(row=row)

    def test_build_rows_payload_probes_both_rows_with_identical_settings(self):
        rows = [
            {
                "id": "capture_available-002",
                "canonical_state": "state-002",
                "legal_moves": [0, 2, 4],
                "reference_move": 2,
                "state": {
                    "current_player": 0,
                    "player_pits": [0, 2, 1, 0, 0, 0],
                    "opponent_pits": [1, 0, 3, 0, 0, 1],
                },
            },
            {
                "id": "capture_available-003",
                "canonical_state": "state-003",
                "legal_moves": [1, 2, 5],
                "reference_move": 2,
                "state": {
                    "current_player": 1,
                    "player_pits": [0, 1, 2, 0, 0, 1],
                    "opponent_pits": [1, 1, 0, 2, 0, 0],
                },
            },
        ]
        selected_artifact = {"path": "/tmp/artifacts/selected.bin"}
        search_options = {"dirichlet_alpha": 0.03, "temperature": 0.0}

        probe_summaries = [
            {
                "selected_move": 0,
                "value": 0.4,
                "policy": [0.45, 0.0, 0.35, 0.0, 0.2, 0.0],
                "visits": [27, 0, 21, 0, 12, 0],
                "child_stats": [
                    {"move": 0, "visits": 27, "q_value": 0.6},
                    {"move": 2, "visits": 21, "q_value": 0.51},
                    {"move": 4, "visits": 12, "q_value": 0.47},
                ],
            },
            {
                "selected_move": 0,
                "value": 0.1,
                "policy": [0.45, 0.0, 0.35, 0.0, 0.2, 0.0],
                "visits": [27, 0, 21, 0, 12, 0],
                "child_stats": [
                    {"move": 0, "visits": 27, "q_value": 0.6},
                    {"move": 2, "visits": 21, "q_value": 0.51},
                    {"move": 4, "visits": 12, "q_value": 0.47},
                ],
            },
            {
                "selected_move": 0,
                "value": 0.4,
                "policy": [0.45, 0.0, 0.35, 0.0, 0.2, 0.0],
                "visits": [27, 0, 21, 0, 12, 0],
                "child_stats": [
                    {"move": 0, "visits": 27, "q_value": 0.6},
                    {"move": 2, "visits": 21, "q_value": 0.51},
                    {"move": 4, "visits": 12, "q_value": 0.47},
                ],
            },
            {
                "selected_move": 2,
                "value": 0.3,
                "policy": [0.0, 0.25, 0.5, 0.0, 0.0, 0.25],
                "visits": [0, 18, 32, 0, 0, 10],
                "child_stats": [
                    {"move": 1, "visits": 18, "q_value": 0.21},
                    {"move": 2, "visits": 32, "q_value": 0.36},
                    {"move": 5, "visits": 10, "q_value": 0.19},
                ],
            },
            {
                "selected_move": 2,
                "value": 0.2,
                "policy": [0.0, 0.1, 0.75, 0.0, 0.0, 0.15],
                "visits": [0, 10, 40, 0, 0, 10],
                "child_stats": [
                    {"move": 1, "visits": 10, "q_value": 0.18},
                    {"move": 2, "visits": 40, "q_value": 0.44},
                    {"move": 5, "visits": 10, "q_value": 0.11},
                ],
            },
            {
                "selected_move": 2,
                "value": 0.3,
                "policy": [0.0, 0.25, 0.5, 0.0, 0.0, 0.25],
                "visits": [0, 18, 32, 0, 0, 10],
                "child_stats": [
                    {"move": 1, "visits": 18, "q_value": 0.21},
                    {"move": 2, "visits": 32, "q_value": 0.36},
                    {"move": 5, "visits": 10, "q_value": 0.19},
                ],
            },
        ]

        with (
            patch.object(module, "probe_artifact_position", side_effect=probe_summaries) as probe,
            patch.object(
                module,
                "compute_rule_features",
                side_effect=[
                    {"side_to_move": 0, "capture_legal": False},
                    {"side_to_move": 1, "capture_legal": True},
                ],
            ) as compute,
        ):
            payload = module.build_rows_payload(
                selected_artifact=selected_artifact,
                rows=rows,
                simulations=384,
                seed=99,
                c_puct=1.4,
                search_options=search_options,
            )

        self.assertEqual(6, probe.call_count)
        self.assertEqual(
            [
                {
                    "artifact_path": "/tmp/artifacts/selected.bin",
                    "state": {
                        "current_player": 0,
                        "player_pits": [0, 2, 1, 0, 0, 0],
                        "opponent_pits": [1, 0, 3, 0, 0, 1],
                    },
                    "simulations": 384,
                    "seed": 99,
                    "c_puct": 1.4,
                    "search_options": search_options,
                    "ablation_mode": "policy_only",
                },
                {
                    "artifact_path": "/tmp/artifacts/selected.bin",
                    "state": {
                        "current_player": 0,
                        "player_pits": [0, 2, 1, 0, 0, 0],
                        "opponent_pits": [1, 0, 3, 0, 0, 1],
                    },
                    "simulations": 384,
                    "seed": 99,
                    "c_puct": 1.4,
                    "search_options": search_options,
                    "ablation_mode": "value_only",
                },
                {
                    "artifact_path": "/tmp/artifacts/selected.bin",
                    "state": {
                        "current_player": 0,
                        "player_pits": [0, 2, 1, 0, 0, 0],
                        "opponent_pits": [1, 0, 3, 0, 0, 1],
                    },
                    "simulations": 384,
                    "seed": 99,
                    "c_puct": 1.4,
                    "search_options": search_options,
                    "ablation_mode": "full",
                },
                {
                    "artifact_path": "/tmp/artifacts/selected.bin",
                    "state": {
                        "current_player": 1,
                        "player_pits": [0, 1, 2, 0, 0, 1],
                        "opponent_pits": [1, 1, 0, 2, 0, 0],
                    },
                    "simulations": 384,
                    "seed": 99,
                    "c_puct": 1.4,
                    "search_options": search_options,
                    "ablation_mode": "policy_only",
                },
                {
                    "artifact_path": "/tmp/artifacts/selected.bin",
                    "state": {
                        "current_player": 1,
                        "player_pits": [0, 1, 2, 0, 0, 1],
                        "opponent_pits": [1, 1, 0, 2, 0, 0],
                    },
                    "simulations": 384,
                    "seed": 99,
                    "c_puct": 1.4,
                    "search_options": search_options,
                    "ablation_mode": "value_only",
                },
                {
                    "artifact_path": "/tmp/artifacts/selected.bin",
                    "state": {
                        "current_player": 1,
                        "player_pits": [0, 1, 2, 0, 0, 1],
                        "opponent_pits": [1, 1, 0, 2, 0, 0],
                    },
                    "simulations": 384,
                    "seed": 99,
                    "c_puct": 1.4,
                    "search_options": search_options,
                    "ablation_mode": "full",
                },
            ],
            [call.kwargs for call in probe.call_args_list],
        )
        self.assertEqual([{"row": rows[0]}, {"row": rows[1]}], [call.kwargs for call in compute.call_args_list])
        self.assertEqual(
            {
                "capture_available-002": {
                    "row_id": "capture_available-002",
                    "canonical_state": "state-002",
                    "legal_moves": [0, 2, 4],
                    "reference_move": 2,
                    "policy_view": {
                        "raw_policy_distribution": {"0": 0.45, "2": 0.35, "4": 0.2},
                        "top_move": 0,
                        "reference_move_probability": 0.35,
                        "selected_minus_reference_margin": 0.1,
                    },
                    "value_view": {
                        "root_value_estimate": 0.4,
                        "per_child_q_values": {"0": 0.6, "2": 0.51, "4": 0.47},
                        "reference_move_q_value": 0.51,
                        "selected_move_q_value": 0.6,
                        "selected_minus_reference_q_margin": 0.09,
                    },
                    "search_view": {
                        "searched_selected_move": 0,
                        "visit_distribution": {"0": 27.0, "2": 21.0, "4": 12.0},
                        "reference_move_visit_share": 0.35,
                        "selected_move_visit_share": 0.45,
                        "child_stats": [
                            {"move": 0, "visits": 27, "q_value": 0.6},
                            {"move": 2, "visits": 21, "q_value": 0.51},
                            {"move": 4, "visits": 12, "q_value": 0.47},
                        ],
                        "child_stats_complete": True,
                        "missing_fields": [],
                    },
                    "probe_views": {
                        "policy_only": {
                            "row_id": "capture_available-002",
                            "canonical_state": "state-002",
                            "legal_moves": [0, 2, 4],
                            "reference_move": 2,
                            "policy_view": {
                                "raw_policy_distribution": {"0": 0.45, "2": 0.35, "4": 0.2},
                                "top_move": 0,
                                "reference_move_probability": 0.35,
                                "selected_minus_reference_margin": 0.1,
                            },
                            "value_view": {
                                "root_value_estimate": 0.4,
                                "per_child_q_values": {"0": 0.6, "2": 0.51, "4": 0.47},
                                "reference_move_q_value": 0.51,
                                "selected_move_q_value": 0.6,
                                "selected_minus_reference_q_margin": 0.09,
                            },
                            "search_view": {
                                "searched_selected_move": 0,
                                "visit_distribution": {"0": 27.0, "2": 21.0, "4": 12.0},
                                "reference_move_visit_share": 0.35,
                                "selected_move_visit_share": 0.45,
                                "child_stats": [
                                    {"move": 0, "visits": 27, "q_value": 0.6},
                                    {"move": 2, "visits": 21, "q_value": 0.51},
                                    {"move": 4, "visits": 12, "q_value": 0.47},
                                ],
                                "child_stats_complete": True,
                                "missing_fields": [],
                            },
                        },
                        "value_only": {
                            "row_id": "capture_available-002",
                            "canonical_state": "state-002",
                            "legal_moves": [0, 2, 4],
                            "reference_move": 2,
                            "policy_view": {
                                "raw_policy_distribution": {"0": 0.45, "2": 0.35, "4": 0.2},
                                "top_move": 0,
                                "reference_move_probability": 0.35,
                                "selected_minus_reference_margin": 0.1,
                            },
                            "value_view": {
                                "root_value_estimate": 0.1,
                                "per_child_q_values": {"0": 0.6, "2": 0.51, "4": 0.47},
                                "reference_move_q_value": 0.51,
                                "selected_move_q_value": 0.6,
                                "selected_minus_reference_q_margin": 0.09,
                            },
                            "search_view": {
                                "searched_selected_move": 0,
                                "visit_distribution": {"0": 27.0, "2": 21.0, "4": 12.0},
                                "reference_move_visit_share": 0.35,
                                "selected_move_visit_share": 0.45,
                                "child_stats": [
                                    {"move": 0, "visits": 27, "q_value": 0.6},
                                    {"move": 2, "visits": 21, "q_value": 0.51},
                                    {"move": 4, "visits": 12, "q_value": 0.47},
                                ],
                                "child_stats_complete": True,
                                "missing_fields": [],
                            },
                        },
                        "full_search": {
                            "row_id": "capture_available-002",
                            "canonical_state": "state-002",
                            "legal_moves": [0, 2, 4],
                            "reference_move": 2,
                            "policy_view": {
                                "raw_policy_distribution": {"0": 0.45, "2": 0.35, "4": 0.2},
                                "top_move": 0,
                                "reference_move_probability": 0.35,
                                "selected_minus_reference_margin": 0.1,
                            },
                            "value_view": {
                                "root_value_estimate": 0.4,
                                "per_child_q_values": {"0": 0.6, "2": 0.51, "4": 0.47},
                                "reference_move_q_value": 0.51,
                                "selected_move_q_value": 0.6,
                                "selected_minus_reference_q_margin": 0.09,
                            },
                            "search_view": {
                                "searched_selected_move": 0,
                                "visit_distribution": {"0": 27.0, "2": 21.0, "4": 12.0},
                                "reference_move_visit_share": 0.35,
                                "selected_move_visit_share": 0.45,
                                "child_stats": [
                                    {"move": 0, "visits": 27, "q_value": 0.6},
                                    {"move": 2, "visits": 21, "q_value": 0.51},
                                    {"move": 4, "visits": 12, "q_value": 0.47},
                                ],
                                "child_stats_complete": True,
                                "missing_fields": [],
                            },
                        },
                    },
                    "rule_features": {"side_to_move": 0, "capture_legal": False},
                },
                "capture_available-003": {
                    "row_id": "capture_available-003",
                    "canonical_state": "state-003",
                    "legal_moves": [1, 2, 5],
                    "reference_move": 2,
                    "policy_view": {
                        "raw_policy_distribution": {"1": 0.25, "2": 0.5, "5": 0.25},
                        "top_move": 2,
                        "reference_move_probability": 0.5,
                        "selected_minus_reference_margin": 0.0,
                    },
                    "value_view": {
                        "root_value_estimate": 0.3,
                        "per_child_q_values": {"1": 0.21, "2": 0.36, "5": 0.19},
                        "reference_move_q_value": 0.36,
                        "selected_move_q_value": 0.36,
                        "selected_minus_reference_q_margin": 0.0,
                    },
                    "search_view": {
                        "searched_selected_move": 2,
                        "visit_distribution": {"1": 18.0, "2": 32.0, "5": 10.0},
                        "reference_move_visit_share": 0.5333,
                        "selected_move_visit_share": 0.5333,
                        "child_stats": [
                            {"move": 1, "visits": 18, "q_value": 0.21},
                            {"move": 2, "visits": 32, "q_value": 0.36},
                            {"move": 5, "visits": 10, "q_value": 0.19},
                        ],
                        "child_stats_complete": True,
                        "missing_fields": [],
                    },
                    "probe_views": {
                        "policy_only": {
                            "row_id": "capture_available-003",
                            "canonical_state": "state-003",
                            "legal_moves": [1, 2, 5],
                            "reference_move": 2,
                            "policy_view": {
                                "raw_policy_distribution": {"1": 0.25, "2": 0.5, "5": 0.25},
                                "top_move": 2,
                                "reference_move_probability": 0.5,
                                "selected_minus_reference_margin": 0.0,
                            },
                            "value_view": {
                                "root_value_estimate": 0.3,
                                "per_child_q_values": {"1": 0.21, "2": 0.36, "5": 0.19},
                                "reference_move_q_value": 0.36,
                                "selected_move_q_value": 0.36,
                                "selected_minus_reference_q_margin": 0.0,
                            },
                            "search_view": {
                                "searched_selected_move": 2,
                                "visit_distribution": {"1": 18.0, "2": 32.0, "5": 10.0},
                                "reference_move_visit_share": 0.5333,
                                "selected_move_visit_share": 0.5333,
                                "child_stats": [
                                    {"move": 1, "visits": 18, "q_value": 0.21},
                                    {"move": 2, "visits": 32, "q_value": 0.36},
                                    {"move": 5, "visits": 10, "q_value": 0.19},
                                ],
                                "child_stats_complete": True,
                                "missing_fields": [],
                            },
                        },
                        "value_only": {
                            "row_id": "capture_available-003",
                            "canonical_state": "state-003",
                            "legal_moves": [1, 2, 5],
                            "reference_move": 2,
                            "policy_view": {
                                "raw_policy_distribution": {"1": 0.1, "2": 0.75, "5": 0.15},
                                "top_move": 2,
                                "reference_move_probability": 0.75,
                                "selected_minus_reference_margin": 0.0,
                            },
                            "value_view": {
                                "root_value_estimate": 0.2,
                                "per_child_q_values": {"1": 0.18, "2": 0.44, "5": 0.11},
                                "reference_move_q_value": 0.44,
                                "selected_move_q_value": 0.44,
                                "selected_minus_reference_q_margin": 0.0,
                            },
                            "search_view": {
                                "searched_selected_move": 2,
                                "visit_distribution": {"1": 10.0, "2": 40.0, "5": 10.0},
                                "reference_move_visit_share": 0.6667,
                                "selected_move_visit_share": 0.6667,
                                "child_stats": [
                                    {"move": 1, "visits": 10, "q_value": 0.18},
                                    {"move": 2, "visits": 40, "q_value": 0.44},
                                    {"move": 5, "visits": 10, "q_value": 0.11},
                                ],
                                "child_stats_complete": True,
                                "missing_fields": [],
                            },
                        },
                        "full_search": {
                            "row_id": "capture_available-003",
                            "canonical_state": "state-003",
                            "legal_moves": [1, 2, 5],
                            "reference_move": 2,
                            "policy_view": {
                                "raw_policy_distribution": {"1": 0.25, "2": 0.5, "5": 0.25},
                                "top_move": 2,
                                "reference_move_probability": 0.5,
                                "selected_minus_reference_margin": 0.0,
                            },
                            "value_view": {
                                "root_value_estimate": 0.3,
                                "per_child_q_values": {"1": 0.21, "2": 0.36, "5": 0.19},
                                "reference_move_q_value": 0.36,
                                "selected_move_q_value": 0.36,
                                "selected_minus_reference_q_margin": 0.0,
                            },
                            "search_view": {
                                "searched_selected_move": 2,
                                "visit_distribution": {"1": 18.0, "2": 32.0, "5": 10.0},
                                "reference_move_visit_share": 0.5333,
                                "selected_move_visit_share": 0.5333,
                                "child_stats": [
                                    {"move": 1, "visits": 18, "q_value": 0.21},
                                    {"move": 2, "visits": 32, "q_value": 0.36},
                                    {"move": 5, "visits": 10, "q_value": 0.19},
                                ],
                                "child_stats_complete": True,
                                "missing_fields": [],
                            },
                        },
                    },
                    "rule_features": {"side_to_move": 1, "capture_legal": True},
                },
            },
            payload,
        )

    def test_build_rows_payload_resolves_partial_embedded_state_from_canonical_state(self):
        row = {
            "id": "capture_available-002",
            "canonical_state": json.dumps(
                {
                    "current_player": 1,
                    "player_pits": [0, 0, 0, 2, 0, 1],
                    "opponent_pits": [0, 1, 0, 0, 0, 0],
                }
            ),
            "legal_moves": [1],
            "reference_move": 1,
            "state": {"current_player": 1},
        }

        with patch.object(
            module,
            "probe_artifact_position",
            return_value={
                "selected_move": 1,
                "value": 0.3,
                "policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "visits": [0, 12, 0, 0, 0, 0],
                "child_stats": [{"move": 1, "visits": 12, "q_value": 0.3}],
            },
        ) as probe:
            payload = module.build_rows_payload(
                selected_artifact={"path": "/tmp/artifacts/selected.bin"},
                rows=[row],
                simulations=128,
                seed=7,
                c_puct=1.25,
                search_options={"temperature": 0.0},
            )

        self.assertEqual(
            {
                "current_player": 1,
                "player_pits": [0, 0, 0, 2, 0, 1],
                "opponent_pits": [0, 1, 0, 0, 0, 0],
            },
            probe.call_args.kwargs["state"],
        )
        self.assertEqual(
            {
                "side_to_move": 1,
                "capture_legal": True,
                "extra_turn_available": False,
                "starvation_shape": "sparse",
                "starvation_risk": True,
            },
            payload["capture_available-002"]["rule_features"],
        )

    def test_build_rows_payload_rejects_rows_without_usable_board_state_before_probing(self):
        row = {
            "id": "capture_available-002",
            "canonical_state": json.dumps({"current_player": 1}),
            "legal_moves": [1],
            "reference_move": 1,
            "state": {},
        }

        with patch.object(module, "probe_artifact_position") as probe:
            with self.assertRaisesRegex(
                ValueError,
                "usable state.*current_player.*player_pits.*opponent_pits",
            ):
                module.build_rows_payload(
                    selected_artifact={"path": "/tmp/artifacts/selected.bin"},
                    rows=[row],
                    simulations=128,
                    seed=7,
                    c_puct=1.25,
                    search_options={"temperature": 0.0},
                )

        probe.assert_not_called()


class Capture002003SearchPolicyArbitrationRuleFeaturesTest(unittest.TestCase):
    def test_compute_rule_features_handles_wraparound_extra_turns(self):
        features = module.compute_rule_features(
            row={
                "reference_move": 5,
                "state": {
                    "current_player": 0,
                    "player_pits": [0, 0, 0, 0, 0, 14],
                    "opponent_pits": [1, 1, 1, 0, 0, 0],
                },
            }
        )

        self.assertEqual(
            {
                "side_to_move": 0,
                "capture_legal": False,
                "extra_turn_available": True,
                "starvation_shape": "distributed",
                "starvation_risk": False,
            },
            features,
        )

    def test_compute_rule_features_uses_canonical_state_to_record_capture_and_starvation(self):
        features = module.compute_rule_features(
            row={
                "reference_move": 1,
                "canonical_state": json.dumps(
                    {
                        "current_player": 1,
                        "player_pits": [0, 0, 0, 2, 0, 1],
                        "opponent_pits": [0, 1, 0, 0, 0, 0],
                    }
                ),
            }
        )

        self.assertEqual(
            {
                "side_to_move": 1,
                "capture_legal": True,
                "extra_turn_available": False,
                "starvation_shape": "sparse",
                "starvation_risk": True,
            },
            features,
        )

    def test_compute_rule_features_uses_state_when_present_to_record_extra_turn_window(self):
        features = module.compute_rule_features(
            row={
                "reference_move": 1,
                "state": {
                    "current_player": 0,
                    "player_pits": [0, 5, 1, 0, 0, 0],
                    "opponent_pits": [2, 2, 1, 1, 0, 1],
                },
                "canonical_state": json.dumps(
                    {
                        "current_player": 1,
                        "player_pits": [0, 1, 0, 0, 0, 0],
                        "opponent_pits": [0, 0, 0, 2, 0, 1],
                    }
                ),
            }
        )

        self.assertEqual(
            {
                "side_to_move": 0,
                "capture_legal": False,
                "extra_turn_available": True,
                "starvation_shape": "distributed",
                "starvation_risk": False,
            },
            features,
        )

    def test_compute_rule_features_uses_current_player_side_for_reference_move(self):
        features = module.compute_rule_features(
            row={
                "reference_move": 1,
                "state": {
                    "current_player": 1,
                    "player_pits": [0, 0, 0, 2, 0, 1],
                    "opponent_pits": [0, 1, 0, 0, 0, 0],
                },
            }
        )

        self.assertEqual(
            {
                "side_to_move": 1,
                "capture_legal": True,
                "extra_turn_available": False,
                "starvation_shape": "sparse",
                "starvation_risk": True,
            },
            features,
        )

    def test_compute_rule_features_matches_engine_wraparound_order(self):
        features = module.compute_rule_features(
            row={
                "reference_move": 5,
                "state": {
                    "current_player": 0,
                    "player_pits": [0, 0, 0, 0, 0, 8],
                    "opponent_pits": [0, 0, 0, 0, 0, 1],
                },
            }
        )

        self.assertEqual(
            {
                "side_to_move": 0,
                "capture_legal": True,
                "extra_turn_available": False,
                "starvation_shape": "sparse",
                "starvation_risk": True,
            },
            features,
        )

    def test_compute_rule_features_rejects_incomplete_state_when_no_source_has_board_shape(self):
        with self.assertRaisesRegex(
            ValueError,
            "usable state.*current_player.*player_pits.*opponent_pits",
        ):
            module.compute_rule_features(
                row={
                    "reference_move": 1,
                    "state": {"current_player": 0, "player_pits": []},
                    "canonical_state": json.dumps({"opponent_pits": [0, 0, 0, 0, 0, 0]}),
                }
            )


class Capture002003SearchPolicyArbitrationClassificationTest(unittest.TestCase):
    FIXTURE_DIR = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "diagnostics"
        / "capture_002_003_search_policy_arbitration"
    )

    def load_fixture(self, name: str) -> dict:
        return json.loads((self.FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))

    def fixture_rows_with_reference_moves(self, fixture: dict) -> dict:
        return {
            row_id: {"reference_move": 2, **fixture["rows"][row_id]}
            for row_id in module.ROW_IDS
        }

    def assert_fixture_classification(self, name: str) -> None:
        fixture = self.load_fixture(name)
        expected_classification = fixture["expected_classification"]
        rows = self.fixture_rows_with_reference_moves(fixture)

        comparison = module.build_paired_comparison(rows=rows)

        self.assertEqual(
            {
                **fixture["expected_comparison"],
                "reference_moves": {
                    row_id: rows[row_id]["reference_move"] for row_id in module.ROW_IDS
                },
            },
            comparison,
        )

        classification = module.classify_paired_comparison(comparison=comparison)

        self.assertTrue(expected_classification["evidence_summary"])
        self.assertEqual(expected_classification, classification)
        self.assertTrue(classification["evidence_summary"])
        self.assertEqual(
            fixture["expected_decision"],
            module.decision_for_classification(classification["classification"]),
        )

    def test_classification_returns_unresolved_when_search_visit_shares_are_missing(self):
        comparison = {
            "row_ids": ["capture_available-002", "capture_available-003"],
            "reference_moves": {
                "capture_available-002": 2,
                "capture_available-003": 2,
            },
            "policy_top_moves": {
                "capture_available-002": 2,
                "capture_available-003": 2,
            },
            "searched_selected_moves": {
                "capture_available-002": 0,
                "capture_available-003": 2,
            },
            "q_margins": {
                "capture_available-002": 0.03,
                "capture_available-003": 0.01,
            },
            "visit_shares": {
                "capture_available-002": {
                    "reference": 0.27,
                    "selected": None,
                },
                "capture_available-003": {
                    "reference": 0.57,
                    "selected": 0.57,
                },
            },
            "rule_feature_differences": {},
            "missing_child_stats": {
                "capture_available-002": [],
                "capture_available-003": [],
            },
        }

        self.assertEqual(
            {
                "classification": "unresolved",
                "evidence_summary": "Required search-amplification inputs are missing, so the paired evidence cannot isolate a supported failure mechanism.",
            },
            module.classify_paired_comparison(comparison=comparison),
        )

    def test_policy_prior_gap_requires_002_policy_top_move_evidence(self):
        comparison = {
            "row_ids": ["capture_available-002", "capture_available-003"],
            "reference_moves": {
                "capture_available-002": 2,
                "capture_available-003": 2,
            },
            "policy_top_moves": {
                "capture_available-002": None,
                "capture_available-003": 2,
            },
            "searched_selected_moves": {
                "capture_available-002": 0,
                "capture_available-003": 2,
            },
            "q_margins": {
                "capture_available-002": 0.03,
                "capture_available-003": 0.01,
            },
            "visit_shares": {
                "capture_available-002": {
                    "reference": 0.27,
                    "selected": 0.49,
                },
                "capture_available-003": {
                    "reference": 0.57,
                    "selected": 0.57,
                },
            },
            "rule_feature_differences": {},
            "missing_child_stats": {
                "capture_available-002": [],
                "capture_available-003": [],
            },
        }

        self.assertEqual(
            {
                "classification": "unresolved",
                "evidence_summary": "Required data is present, but the paired evidence does not isolate a supported failure mechanism.",
            },
            module.classify_paired_comparison(comparison=comparison),
        )

    def test_value_backup_gap_requires_003_q_margin_evidence(self):
        comparison = {
            "row_ids": ["capture_available-002", "capture_available-003"],
            "reference_moves": {
                "capture_available-002": 2,
                "capture_available-003": 2,
            },
            "policy_top_moves": {
                "capture_available-002": 2,
                "capture_available-003": 2,
            },
            "searched_selected_moves": {
                "capture_available-002": 0,
                "capture_available-003": 2,
            },
            "q_margins": {
                "capture_available-002": 0.22,
                "capture_available-003": None,
            },
            "visit_shares": {
                "capture_available-002": {
                    "reference": 0.27,
                    "selected": 0.49,
                },
                "capture_available-003": {
                    "reference": 0.57,
                    "selected": 0.57,
                },
            },
            "rule_feature_differences": {},
            "missing_child_stats": {
                "capture_available-002": [],
                "capture_available-003": [],
            },
        }

        self.assertEqual(
            {
                "classification": "unresolved",
                "evidence_summary": "Required data is present, but the paired evidence does not isolate a supported failure mechanism.",
            },
            module.classify_paired_comparison(comparison=comparison),
        )

    def test_search_amplification_gap_requires_003_control_to_not_show_same_amplification(self):
        comparison = {
            "row_ids": ["capture_available-002", "capture_available-003"],
            "reference_moves": {
                "capture_available-002": 2,
                "capture_available-003": 2,
            },
            "policy_top_moves": {
                "capture_available-002": 2,
                "capture_available-003": 2,
            },
            "searched_selected_moves": {
                "capture_available-002": 0,
                "capture_available-003": 0,
            },
            "q_margins": {
                "capture_available-002": 0.03,
                "capture_available-003": 0.01,
            },
            "visit_shares": {
                "capture_available-002": {
                    "reference": 0.27,
                    "selected": 0.49,
                },
                "capture_available-003": {
                    "reference": 0.35,
                    "selected": 0.57,
                },
            },
            "rule_feature_differences": {},
            "missing_child_stats": {
                "capture_available-002": [],
                "capture_available-003": [],
            },
        }

        self.assertEqual(
            {
                "classification": "unresolved",
                "evidence_summary": "Required data is present, but the paired evidence does not isolate a supported failure mechanism.",
            },
            module.classify_paired_comparison(comparison=comparison),
        )

    def test_policy_prior_gap_fixture(self):
        self.assert_fixture_classification("policy_prior_gap")

    def test_build_paired_comparison_carries_reference_moves(self):
        rows = {
            "capture_available-002": {
                "reference_move": 4,
                "policy_view": {"top_move": 0},
                "search_view": {
                    "searched_selected_move": 0,
                    "reference_move_visit_share": 0.32,
                    "selected_move_visit_share": 0.52,
                    "missing_fields": [],
                },
                "value_view": {"selected_minus_reference_q_margin": 0.03},
                "rule_features": {},
            },
            "capture_available-003": {
                "reference_move": 4,
                "policy_view": {"top_move": 4},
                "search_view": {
                    "searched_selected_move": 4,
                    "reference_move_visit_share": 0.51,
                    "selected_move_visit_share": 0.51,
                    "missing_fields": [],
                },
                "value_view": {"selected_minus_reference_q_margin": 0.0},
                "rule_features": {},
            },
        }

        self.assertEqual(
            {
                "capture_available-002": 4,
                "capture_available-003": 4,
            },
            module.build_paired_comparison(rows=rows)["reference_moves"],
        )

    def test_classification_uses_paired_reference_move_for_policy_prior_gap(self):
        comparison = {
            "row_ids": ["capture_available-002", "capture_available-003"],
            "reference_moves": {
                "capture_available-002": 4,
                "capture_available-003": 4,
            },
            "policy_top_moves": {
                "capture_available-002": 0,
                "capture_available-003": 4,
            },
            "searched_selected_moves": {
                "capture_available-002": 0,
                "capture_available-003": 4,
            },
            "q_margins": {
                "capture_available-002": 0.03,
                "capture_available-003": 0.01,
            },
            "visit_shares": {
                "capture_available-002": {
                    "reference": 0.32,
                    "selected": 0.52,
                },
                "capture_available-003": {
                    "reference": 0.51,
                    "selected": 0.51,
                },
            },
            "rule_feature_differences": {},
            "missing_child_stats": {
                "capture_available-002": [],
                "capture_available-003": [],
            },
        }

        self.assertEqual(
            {
                "classification": "policy_prior_gap",
                "evidence_summary": "002 policy top move misses the reference move while 003 policy top move matches it.",
            },
            module.classify_paired_comparison(comparison=comparison),
        )

    def test_classification_uses_paired_reference_move_for_value_backup_gap(self):
        comparison = {
            "row_ids": ["capture_available-002", "capture_available-003"],
            "reference_moves": {
                "capture_available-002": 4,
                "capture_available-003": 4,
            },
            "policy_top_moves": {
                "capture_available-002": 4,
                "capture_available-003": 4,
            },
            "searched_selected_moves": {
                "capture_available-002": 0,
                "capture_available-003": 4,
            },
            "q_margins": {
                "capture_available-002": 0.22,
                "capture_available-003": 0.01,
            },
            "visit_shares": {
                "capture_available-002": {
                    "reference": 0.32,
                    "selected": 0.52,
                },
                "capture_available-003": {
                    "reference": 0.51,
                    "selected": 0.51,
                },
            },
            "rule_feature_differences": {},
            "missing_child_stats": {
                "capture_available-002": [],
                "capture_available-003": [],
            },
        }

        self.assertEqual(
            {
                "classification": "value_backup_gap",
                "evidence_summary": "002 policy supports the reference move, but 002 child Q values favor the wrong move much more strongly than 003.",
            },
            module.classify_paired_comparison(comparison=comparison),
        )

    def test_value_backup_gap_fixture(self):
        self.assert_fixture_classification("value_backup_gap")

    def test_search_amplification_gap_fixture(self):
        self.assert_fixture_classification("search_amplification_gap")

    def test_state_specific_rule_collision_fixture(self):
        self.assert_fixture_classification("state_specific_rule_collision")

    def test_unresolved_fixture(self):
        self.assert_fixture_classification("unresolved")


class Capture002003SearchPolicyArbitrationCliTest(unittest.TestCase):
    def write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def write_text(self, path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    def test_build_payload_persists_required_top_level_fields(self):
        selected_artifact = {
            "path": "/tmp/artifacts/selected.bin",
            "provenance_source": "selection_manifest.selected_target",
        }
        source_artifacts = {"base_config": "/tmp/base/config.yml"}
        settings = {
            "base_config_path": "/tmp/base/config.yml",
            "row_ids": list(module.ROW_IDS),
            "search_settings": {"c_puct": 1.25},
            "seeds": [17, 17],
            "simulation_count": 384,
        }
        rows = {
            "capture_available-002": {
                "reference_move": 2,
                "policy_view": {"top_move": 2},
                "search_view": {
                    "searched_selected_move": 0,
                    "reference_move_visit_share": 0.3,
                    "selected_move_visit_share": 0.5,
                    "missing_fields": [],
                },
                "value_view": {"selected_minus_reference_q_margin": 0.02},
                "rule_features": {},
            },
            "capture_available-003": {
                "reference_move": 2,
                "policy_view": {"top_move": 2},
                "search_view": {
                    "searched_selected_move": 2,
                    "reference_move_visit_share": 0.6,
                    "selected_move_visit_share": 0.6,
                    "missing_fields": [],
                },
                "value_view": {"selected_minus_reference_q_margin": 0.01},
                "rule_features": {},
            },
        }

        payload = module.build_payload(
            selected_artifact=selected_artifact,
            source_artifacts=source_artifacts,
            settings=settings,
            rows=rows,
        )

        self.assertEqual(module.SCHEMA, payload["schema"])
        self.assertEqual(selected_artifact, payload["selected_artifact"])
        self.assertEqual(source_artifacts, payload["source_artifacts"])
        self.assertEqual(settings, payload["settings"])
        self.assertEqual(rows, payload["rows"])
        self.assertEqual(
            {
                "classification": "search_amplification_gap",
                "evidence_summary": "002 policy and Q values are near neutral, but search visits amplify the wrong move into the final choice while 003 does not.",
            },
            payload["classification"],
        )
        self.assertEqual("write_search_adjustment_spec", payload["decision"])
        self.assertEqual(list(module.ROW_IDS), payload["paired_comparison"]["row_ids"])

    def test_main_fails_closed_when_required_rows_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            base_config = Path(tmp) / "configs" / "base.yml"
            out_path = Path(tmp) / "artifacts" / "diagnostic.json"
            self.write_text(base_config, "mcts: {}\n")
            self.write_json(
                run_dir / "selection" / "selection_manifest.json",
                {"selected_target": "/tmp/artifacts/selected.bin"},
            )
            self.write_json(
                run_dir / "final" / "selected_candidate_forensics.json",
                {
                    "systems": {
                        "current": {
                            "rows": [
                                {
                                    "id": "capture_available-002",
                                    "canonical_state": "state-002",
                                    "legal_moves": [0, 2, 4],
                                    "reference_move": 2,
                                    "state": {"current_player": 0},
                                }
                            ]
                        }
                    }
                },
            )

            with self.assertRaisesRegex(ValueError, "capture_available-003"):
                module.main(
                    [
                        "--run-dir",
                        str(run_dir),
                        "--base-config",
                        str(base_config),
                        "--out",
                        str(out_path),
                    ]
                )

            self.assertFalse(out_path.exists())

    def test_main_fails_closed_when_base_config_does_not_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()

            with self.assertRaisesRegex(ValueError, "base config.*readable"), redirect_stderr(stderr):
                module.main(
                    [
                        "--run-dir",
                        str(Path(tmp) / "run"),
                        "--base-config",
                        str(Path(tmp) / "configs" / "missing.yml"),
                        "--out",
                        str(Path(tmp) / "artifacts" / "diagnostic.json"),
                    ]
                )

    def test_main_fails_closed_when_base_config_is_unreadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_config = Path(tmp) / "configs" / "base.yml"
            self.write_text(base_config, "mcts: {}\n")
            base_config.chmod(0)

            try:
                with self.assertRaisesRegex(ValueError, "base config.*readable"):
                    module.main(
                        [
                            "--run-dir",
                            str(Path(tmp) / "run"),
                            "--base-config",
                            str(base_config),
                            "--out",
                            str(Path(tmp) / "artifacts" / "diagnostic.json"),
                        ]
                    )
            finally:
                base_config.chmod(0o644)

    def test_main_rejects_non_positive_simulations(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()

            with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                module.main(
                    [
                        "--run-dir",
                        str(Path(tmp) / "run"),
                        "--base-config",
                        str(Path(tmp) / "configs" / "base.yml"),
                        "--out",
                        str(Path(tmp) / "artifacts" / "diagnostic.json"),
                        "--simulations",
                        "0",
                    ]
                )

            self.assertEqual(2, error.exception.code)
            self.assertIn("--simulations must be > 0", stderr.getvalue())

    def test_main_rejects_non_positive_c_puct(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()

            with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                module.main(
                    [
                        "--run-dir",
                        str(Path(tmp) / "run"),
                        "--base-config",
                        str(Path(tmp) / "configs" / "base.yml"),
                        "--out",
                        str(Path(tmp) / "artifacts" / "diagnostic.json"),
                        "--c-puct",
                        "0",
                    ]
                )

            self.assertEqual(2, error.exception.code)
            self.assertIn("--c-puct must be > 0", stderr.getvalue())

    def test_main_rejects_non_finite_c_puct(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()

            with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                module.main(
                    [
                        "--run-dir",
                        str(Path(tmp) / "run"),
                        "--base-config",
                        str(Path(tmp) / "configs" / "base.yml"),
                        "--out",
                        str(Path(tmp) / "artifacts" / "diagnostic.json"),
                        "--c-puct",
                        "nan",
                    ]
                )

            self.assertEqual(2, error.exception.code)
            self.assertIn("--c-puct must be finite", stderr.getvalue())

    def test_main_writes_cli_artifact_output_with_expected_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            base_config = Path(tmp) / "configs" / "base.yml"
            out_path = Path(tmp) / "artifacts" / "diagnostic.json"
            self.write_text(base_config, "mcts: {}\n")
            fake_arena = type(
                "FakeArena",
                (),
                {
                    "build_eval_search_options": staticmethod(
                        lambda: {
                            "fpu_mode": "parent_q",
                            "reuse_subtree": True,
                            "normalize_values": True,
                            "root_policy_mode": "visit_count",
                            "tactical_root_bias": 0.1,
                        }
                    )
                },
            )
            self.write_json(
                run_dir / "selection" / "selection_manifest.json",
                {"selected_target": "/tmp/artifacts/selected.bin"},
            )
            self.write_json(
                run_dir / "final" / "selected_candidate_forensics.json",
                {
                    "systems": {
                        "current": {
                            "rows": [
                                {
                                    "id": "capture_available-002",
                                    "canonical_state": "state-002",
                                    "legal_moves": [0, 2, 4],
                                    "reference_move": 2,
                                    "state": {"current_player": 0},
                                },
                                {
                                    "id": "capture_available-003",
                                    "canonical_state": "state-003",
                                    "legal_moves": [1, 2, 5],
                                    "reference_move": 2,
                                    "state": {"current_player": 1},
                                },
                            ]
                        }
                    }
                },
            )
            payload_rows = {
                "capture_available-002": {
                    "reference_move": 2,
                    "policy_view": {"top_move": 2},
                    "search_view": {
                        "searched_selected_move": 0,
                        "reference_move_visit_share": 0.3,
                        "selected_move_visit_share": 0.5,
                        "missing_fields": [],
                    },
                    "value_view": {"selected_minus_reference_q_margin": 0.02},
                    "rule_features": {},
                },
                "capture_available-003": {
                    "reference_move": 2,
                    "policy_view": {"top_move": 2},
                    "search_view": {
                        "searched_selected_move": 2,
                        "reference_move_visit_share": 0.6,
                        "selected_move_visit_share": 0.6,
                        "missing_fields": [],
                    },
                    "value_view": {"selected_minus_reference_q_margin": 0.01},
                    "rule_features": {},
                },
            }

            stdout = io.StringIO()
            with (
                patch.object(module, "load_arena_module", return_value=fake_arena),
                patch.object(module, "build_rows_payload", return_value=payload_rows),
                redirect_stdout(stdout),
            ):
                exit_code = module.main(
                    [
                        "--run-dir",
                        str(run_dir),
                        "--base-config",
                        str(base_config),
                        "--out",
                        str(out_path),
                        "--simulations",
                        "512",
                        "--seed",
                        "23",
                        "--c-puct",
                        "1.5",
                    ]
                )
            self.assertEqual(0, exit_code)
            artifact = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(module.SCHEMA, artifact["schema"])
            self.assertEqual(
                {
                    "path": "/tmp/artifacts/selected.bin",
                    "provenance_source": "selection_manifest.selected_target",
                    "selected_artifact": None,
                    "selected_target": "/tmp/artifacts/selected.bin",
                },
                artifact["selected_artifact"],
            )
            self.assertEqual({"base_config": str(base_config)}, artifact["source_artifacts"])
            self.assertEqual(
                {
                    "base_config_path": str(base_config),
                    "row_ids": list(module.ROW_IDS),
                    "search_settings": {
                        "c_puct": 1.5,
                        "fpu_mode": "parent_q",
                        "normalize_values": True,
                        "reuse_subtree": False,
                        "root_policy_mode": "visit_count",
                        "tactical_root_bias": 0.1,
                    },
                    "seeds": [23, 23],
                    "simulation_count": 512,
                },
                artifact["settings"],
            )
            self.assertEqual(payload_rows, artifact["rows"])
            self.assertEqual("write_search_adjustment_spec", artifact["decision"])
            self.assertEqual(
                {
                    "artifact_path": str(out_path),
                    "schema": module.SCHEMA,
                    "decision": "write_search_adjustment_spec",
                },
                json.loads(stdout.getvalue().strip()),
            )

    def test_main_builds_eval_search_options_and_overrides_reuse_subtree(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            base_config = Path(tmp) / "configs" / "base.yml"
            out_path = Path(tmp) / "artifacts" / "diagnostic.json"
            self.write_text(base_config, "mcts: {}\n")

            captured = {}

            fake_arena = type(
                "FakeArena",
                (),
                {
                    "build_eval_search_options": staticmethod(
                        lambda: {
                            "fpu_mode": "parent_q",
                            "reuse_subtree": True,
                            "normalize_values": True,
                            "root_policy_mode": "visit_count",
                            "tactical_root_bias": 1.5,
                        }
                    )
                },
            )

            def fake_build_rows_payload(**kwargs):
                captured.update(kwargs)
                return {
                    "capture_available-002": {
                        "reference_move": 2,
                        "policy_view": {"top_move": 2},
                        "value_view": {"selected_minus_reference_q_margin": 0.0},
                        "search_view": {
                            "searched_selected_move": 2,
                            "reference_move_visit_share": 0.5,
                            "selected_move_visit_share": 0.5,
                            "missing_fields": [],
                        },
                        "rule_features": {},
                    },
                    "capture_available-003": {
                        "reference_move": 2,
                        "policy_view": {"top_move": 2},
                        "value_view": {"selected_minus_reference_q_margin": 0.0},
                        "search_view": {
                            "searched_selected_move": 2,
                            "reference_move_visit_share": 0.5,
                            "selected_move_visit_share": 0.5,
                            "missing_fields": [],
                        },
                        "rule_features": {},
                    },
                }

            with (
                patch.object(module, "load_selected_artifact", return_value={"path": "/tmp/artifacts/selected.bin"}),
                patch.object(module, "load_rows_from_run", return_value={}),
                patch.object(module, "resolve_rows", return_value=[]),
                patch.object(module, "load_arena_module", return_value=fake_arena),
                patch.object(module, "build_rows_payload", side_effect=fake_build_rows_payload),
                patch.object(module, "build_payload", return_value={"decision": "stop_unresolved"}),
            ):
                exit_code = module.main(
                    [
                        "--run-dir",
                        str(run_dir),
                        "--base-config",
                        str(base_config),
                        "--out",
                        str(out_path),
                    ]
                )

        self.assertEqual(0, exit_code)
        self.assertEqual(
            {
                "fpu_mode": "parent_q",
                "reuse_subtree": False,
                "normalize_values": True,
                "root_policy_mode": "visit_count",
                "tactical_root_bias": 1.5,
            },
            captured["search_options"],
        )
