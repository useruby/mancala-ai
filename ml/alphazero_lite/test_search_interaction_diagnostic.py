import json
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ml.alphazero_lite import search_interaction_diagnostic as module


class SearchInteractionDiagnosticTest(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_resolve_target_rows_uses_matrix_priority_rows_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_run = Path(tmp) / "original"
            rebalanced_run = Path(tmp) / "rebalanced"
            original_run.mkdir()
            rebalanced_run.mkdir()

            with mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.build_matrix_payload",
                return_value={
                    "summary": {
                        "priority_rows": [
                            "capture_available-002",
                            "high_imbalance-010",
                            "sparse_endgame-009",
                            "opening_plies_1_8-057",
                        ]
                    }
                },
            ):
                rows = module.resolve_target_rows(
                    original_run_dir=original_run,
                    rebalanced_run_dir=rebalanced_run,
                    explicit_rows=None,
                )

        self.assertEqual(["capture_available-002", "opening_plies_1_8-057"], rows)

    def test_resolve_target_rows_prefers_explicit_rows(self):
        rows = module.resolve_target_rows(
            original_run_dir=Path("/tmp/original"),
            rebalanced_run_dir=Path("/tmp/rebalanced"),
            explicit_rows=[
                "capture_available-003",
                "high_imbalance-010",
                "sparse_endgame-009",
                "incumbent_proxy_disagreement-031",
            ],
        )

        self.assertEqual(["capture_available-003", "incumbent_proxy_disagreement-031"], rows)

    def test_resolve_target_rows_rejects_explicit_rows_when_all_are_excluded(self):
        with self.assertRaisesRegex(ValueError, "resolved no target rows"):
            module.resolve_target_rows(
                original_run_dir=Path("/tmp/original"),
                rebalanced_run_dir=Path("/tmp/rebalanced"),
                explicit_rows=["high_imbalance-010", "sparse_endgame-009"],
            )

    def test_load_selected_artifact_path_prefers_selected_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            selection_dir = run_dir / "selection"
            selection_dir.mkdir(parents=True)
            (selection_dir / "selection_manifest.json").write_text(
                json.dumps(
                    {
                        "selected_artifact": str(selection_dir / "artifact"),
                        "selected_target": "/artifacts/real-selected",
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual("/artifacts/real-selected", module.load_selected_artifact_path(run_dir))

    def test_load_row_context_reads_current_original_rebalanced_and_optional_opening_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            original_run = tmp_path / "original"
            rebalanced_run = tmp_path / "rebalanced"

            self._write_json(
                original_run / "final" / "selected_candidate_forensics.json",
                {
                    "systems": {
                        "current": {
                            "rows": [
                                {
                                    "id": "capture_available-002",
                                    "bucket": "capture_available",
                                    "phase": "opening",
                                    "reference_move": 2,
                                    "teacher_value": 0.5,
                                    "system_value": 0.1,
                                    "value_error": 0.4,
                                }
                            ]
                        },
                        "challenger": {
                            "rows": [
                                {
                                    "id": "capture_available-002",
                                    "bucket": "capture_available",
                                    "phase": "opening",
                                    "reference_move": 2,
                                    "teacher_value": 0.5,
                                    "system_value": 0.54,
                                    "value_error": 0.04,
                                }
                            ]
                        },
                    }
                },
            )
            self._write_json(
                rebalanced_run / "final" / "selected_candidate_forensics.json",
                {
                    "systems": {
                        "current": {
                            "rows": [
                                {
                                    "id": "capture_available-002",
                                    "bucket": "capture_available",
                                    "phase": "opening",
                                    "reference_move": 2,
                                    "teacher_value": 0.5,
                                    "system_value": 0.1,
                                    "value_error": 0.4,
                                }
                            ]
                        },
                        "challenger": {
                            "rows": [
                                {
                                    "id": "capture_available-002",
                                    "bucket": "capture_available",
                                    "phase": "opening",
                                    "reference_move": 2,
                                    "teacher_value": 0.5,
                                    "system_value": 0.57,
                                    "value_error": 0.07,
                                }
                            ]
                        },
                    }
                },
            )
            self._write_json(
                original_run / "final" / "opening_capture_family_report.json",
                {
                    "rows": [
                        {
                            "id": "capture_available-002",
                            "candidate_prior_summary": {"selected_move": 0},
                            "candidate_searched_summary": {"selected_move": 0},
                        }
                    ]
                },
            )
            self._write_json(
                rebalanced_run / "final" / "opening_capture_family_report.json",
                {
                    "rows": [
                        {
                            "id": "capture_available-002",
                            "candidate_prior_summary": {"selected_move": 2},
                            "candidate_searched_summary": {"selected_move": 0},
                        }
                    ]
                },
            )

            context = module.load_row_context(
                row_id="capture_available-002",
                original_run_dir=original_run,
                rebalanced_run_dir=rebalanced_run,
            )

        self.assertEqual("capture_available", context["bucket"])
        self.assertEqual(2, context["reference_move"])
        self.assertEqual(0.5, context["teacher_value"])
        self.assertEqual(0, context["original_opening_row"]["candidate_searched_summary"]["selected_move"])
        self.assertEqual(2, context["rebalanced_opening_row"]["candidate_prior_summary"]["selected_move"])

    def test_build_artifact_row_normalizes_policy_visits_q_values_and_child_stats(self):
        artifact_row = module.build_artifact_row(
            artifact_path="model-artifact/current",
            row_id="capture_available-002",
            bucket="capture_available",
            phase="opening",
            reference_move=2,
            system_value=0.5681,
            teacher_value=0.5028,
            value_error=0.0653,
            probe_summary={
                "selected_move": 0,
                "value": 0.57,
                "policy": [0.05, 0.15, 0.52, 0.18, 0.10, 0.0],
                "visits": [6.0, 8.0, 22.0, 18.0, 10.0, 0.0],
                "child_stats": [
                    {"move": 0, "visits": 6, "q_value": 0.38},
                    {"move": 2, "visits": 22, "q_value": 0.44},
                    {"move": 3, "visits": 18, "q_value": 0.58},
                ],
            },
            legal_moves=[0, 1, 2, 3, 4],
        )

        self.assertEqual(0, artifact_row["selected_move"])
        self.assertEqual({"0": 0.05, "1": 0.15, "2": 0.52, "3": 0.18, "4": 0.1}, artifact_row["raw_policy_distribution"])
        self.assertEqual({"0": 6, "2": 22, "3": 18}, artifact_row["per_move_visits"])
        self.assertEqual({"0": 0.38, "2": 0.44, "3": 0.58}, artifact_row["per_move_q_values"])
        self.assertTrue(artifact_row["child_stats_available"])
        self.assertEqual([], artifact_row["missing_fields"])

    def test_build_artifact_row_persists_missing_probe_fields_explicitly(self):
        artifact_row = module.build_artifact_row(
            artifact_path="model-artifact/current",
            row_id="incumbent_proxy_disagreement-031",
            bucket="incumbent_proxy_disagreement",
            phase="mid",
            reference_move=4,
            system_value=0.57,
            teacher_value=0.65,
            value_error=0.08,
            probe_summary={"selected_move": 2, "policy": [], "visits": [], "child_stats": []},
            legal_moves=[0, 2, 4],
        )

        self.assertEqual({"0": 0.0, "2": 0.0, "4": 0.0}, artifact_row["raw_policy_distribution"])
        self.assertIsNone(artifact_row["searched_visit_distribution"])
        self.assertIsNone(artifact_row["per_move_visits"])
        self.assertIsNone(artifact_row["per_move_q_values"])
        self.assertIsNone(artifact_row["probe_value"])
        self.assertIsNone(artifact_row["probe_value_error"])
        self.assertFalse(artifact_row["child_stats_available"])
        self.assertIsNone(artifact_row["child_stats"])
        self.assertEqual(
            [
                "probe_value",
                "probe_value_error",
                "searched_visit_distribution",
                "per_move_visits",
                "per_move_q_values",
                "child_stats",
            ],
            artifact_row["missing_fields"],
        )

    def test_build_artifact_row_retains_selection_breakdown_and_visit_snapshots(self):
        artifact_row = module.build_artifact_row(
            artifact_path="model-artifact/current",
            row_id="capture_available-003",
            bucket="capture_available",
            phase="opening",
            reference_move=2,
            system_value=0.5681,
            teacher_value=0.5028,
            value_error=0.0653,
            probe_summary={
                "selected_move": 0,
                "value": 0.57,
                "policy": [0.05, 0.15, 0.52, 0.18, 0.10, 0.0],
                "visits": [6.0, 8.0, 22.0, 18.0, 10.0, 0.0],
                "child_stats": [
                    {"move": 0, "visits": 6, "q_value": 0.38},
                    {"move": 2, "visits": 22, "q_value": 0.44},
                ],
                "selection_breakdown": {
                    "policy_top_move": 2,
                    "visit_top_move": 0,
                    "q_top_move": 0,
                },
                "visit_snapshots": [
                    {"simulation": 32, "top_move": 2, "top_visits": 13},
                    {"simulation": 64, "top_move": 0, "top_visits": 22},
                ],
            },
            legal_moves=[0, 1, 2, 3, 4],
        )

        self.assertEqual(
            {
                "policy_top_move": 2,
                "visit_top_move": 0,
                "q_top_move": 0,
            },
            artifact_row["selection_breakdown"],
        )
        self.assertEqual(
            [
                {"simulation": 32, "top_move": 2, "top_visits": 13},
                {"simulation": 64, "top_move": 0, "top_visits": 22},
            ],
            artifact_row["visit_snapshots"],
        )

    def test_build_artifact_row_preserves_missing_vs_empty_visit_snapshots(self):
        missing_snapshots_row = module.build_artifact_row(
            artifact_path="model-artifact/current",
            row_id="capture_available-003",
            bucket="capture_available",
            phase="opening",
            reference_move=2,
            system_value=0.5681,
            teacher_value=0.5028,
            value_error=0.0653,
            probe_summary={
                "selected_move": 0,
                "value": 0.57,
                "policy": [0.05, 0.15, 0.52, 0.18, 0.10, 0.0],
                "visits": [6.0, 8.0, 22.0, 18.0, 10.0, 0.0],
                "child_stats": [{"move": 0, "visits": 6, "q_value": 0.38}],
            },
            legal_moves=[0, 1, 2, 3, 4],
        )
        empty_snapshots_row = module.build_artifact_row(
            artifact_path="model-artifact/current",
            row_id="capture_available-003",
            bucket="capture_available",
            phase="opening",
            reference_move=2,
            system_value=0.5681,
            teacher_value=0.5028,
            value_error=0.0653,
            probe_summary={
                "selected_move": 0,
                "value": 0.57,
                "policy": [0.05, 0.15, 0.52, 0.18, 0.10, 0.0],
                "visits": [6.0, 8.0, 22.0, 18.0, 10.0, 0.0],
                "child_stats": [{"move": 0, "visits": 6, "q_value": 0.38}],
                "visit_snapshots": [],
            },
            legal_moves=[0, 1, 2, 3, 4],
        )

        self.assertIsNone(missing_snapshots_row["visit_snapshots"])
        self.assertEqual([], empty_snapshots_row["visit_snapshots"])

    def test_summarize_row_mechanism_prefers_selection_breakdown_and_distinguishes_snapshot_state(self):
        self.assertEqual(
            "policy leans to 4, visits finish on 2, q-values favor 2, snapshots empty",
            module.summarize_row_mechanism(
                {
                    "rebalanced_challenger": {
                        "selection_breakdown": {
                            "policy_top_move": 4,
                            "visit_top_move": 2,
                            "q_top_move": 2,
                        },
                        "raw_policy_distribution": {"2": 0.8, "4": 0.2},
                        "searched_visit_distribution": {"2": 0.1, "4": 0.9},
                        "per_move_q_values": {"2": 0.3, "4": 0.7},
                        "visit_snapshots": [],
                    }
                }
            ),
        )
        self.assertEqual(
            "policy leans to 1, visits finish on 1, q-values favor 1, snapshots missing",
            module.summarize_row_mechanism(
                {
                    "rebalanced_challenger": {
                        "raw_policy_distribution": {"1": 0.7, "2": 0.3},
                        "searched_visit_distribution": {"1": 0.6, "2": 0.4},
                        "per_move_q_values": {"1": 0.5, "2": 0.2},
                        "visit_snapshots": None,
                    }
                }
            ),
        )

    def test_decide_row_marks_search_overrides_prior_when_prior_is_correct_but_search_drifts(self):
        row_payload = {
            "reference_move": 2,
            "current": {"selected_move": 2, "child_stats_available": True},
            "original_challenger": {
                "selected_move": 2,
                "raw_policy_distribution": {"2": 0.51},
                "searched_visit_distribution": {"2": 0.55},
                "per_move_q_values": {"2": 0.54},
                "child_stats_available": True,
            },
            "rebalanced_challenger": {
                "selected_move": 1,
                "raw_policy_distribution": {"2": 0.48, "1": 0.23},
                "searched_visit_distribution": {"2": 0.19, "1": 0.61},
                "per_move_q_values": {"2": 0.44, "1": 0.58},
                "child_stats_available": True,
            },
        }

        self.assertEqual("search_overrides_prior", module.decide_row(row_payload))

    def test_decide_row_marks_q_value_backup_issue_when_backup_favors_wrong_move(self):
        row_payload = {
            "reference_move": 4,
            "current": {"selected_move": 4, "child_stats_available": True},
            "original_challenger": {"selected_move": 4, "child_stats_available": True},
            "rebalanced_challenger": {
                "selected_move": 2,
                "raw_policy_distribution": {"4": 0.14, "2": 0.16},
                "searched_visit_distribution": {"4": 0.21, "2": 0.57},
                "per_move_q_values": {"4": 0.22, "2": 0.71},
                "child_stats_available": True,
            },
        }

        self.assertEqual("q_value_backup_issue", module.decide_row(row_payload))

    def test_build_artifact_row_separates_forensic_and_probe_values(self):
        artifact_row = module.build_artifact_row(
            artifact_path="model-artifact/current",
            row_id="capture_available-002",
            bucket="capture_available",
            phase="opening",
            reference_move=2,
            system_value=0.5681,
            teacher_value=0.5028,
            value_error=0.0653,
            probe_summary={
                "selected_move": 2,
                "value": 0.75,
                "policy": [0.05, 0.15, 0.52, 0.18, 0.10, 0.0],
                "visits": [6.0, 8.0, 22.0, 18.0, 10.0, 0.0],
                "child_stats": [{"move": 2, "visits": 22, "q_value": 0.44}],
            },
            legal_moves=[0, 1, 2, 3, 4],
        )

        self.assertEqual(0.5681, artifact_row["forensic_system_value"])
        self.assertEqual(0.0653, artifact_row["forensic_value_error"])
        self.assertEqual(0.75, artifact_row["probe_value"])
        self.assertEqual(0.2472, artifact_row["probe_value_error"])

    def test_decide_row_requires_child_stats_for_all_compared_artifacts(self):
        row_payload = {
            "reference_move": 2,
            "current": {"selected_move": 2, "child_stats_available": False},
            "original_challenger": {"selected_move": 2, "child_stats_available": True},
            "rebalanced_challenger": {
                "selected_move": 1,
                "raw_policy_distribution": {"2": 0.48, "1": 0.23},
                "searched_visit_distribution": {"2": 0.19, "1": 0.61},
                "per_move_q_values": {"2": 0.44, "1": 0.58},
                "child_stats_available": True,
            },
        }

        self.assertEqual("insufficient_child_stats", module.decide_row(row_payload))

    def test_decide_row_preserves_source_behavior_when_probe_keys_are_partial(self):
        row_payload = {
            "reference_move": 2,
            "current": {"selected_move": 2, "child_stats_available": True},
            "original_challenger": {"selected_move": 2, "child_stats_available": True},
            "rebalanced_challenger": {
                "selected_move": 1,
                "raw_policy_distribution": {"2": 0.48},
                "searched_visit_distribution": {"2": 0.19, "1": 0.61},
                "per_move_q_values": {"2": 0.44, "1": 0.58},
                "child_stats_available": True,
            },
        }

        self.assertEqual("search_overrides_prior", module.decide_row(row_payload))

    def test_decide_row_preserves_source_behavior_when_selected_move_is_missing(self):
        row_payload = {
            "reference_move": 2,
            "current": {"selected_move": 2, "child_stats_available": True},
            "original_challenger": {"selected_move": 2, "child_stats_available": True},
            "rebalanced_challenger": {
                "selected_move": None,
                "raw_policy_distribution": {"2": 0.48},
                "searched_visit_distribution": {"2": 0.19, "1": 0.61},
                "per_move_q_values": {"2": 0.44, "1": 0.58},
                "child_stats_available": True,
            },
        }

        self.assertEqual("search_overrides_prior", module.decide_row(row_payload))

    def test_build_search_value_rows_marks_partial_probe_data_as_insufficient_child_stats(self):
        rows = module._build_search_value_rows(
            {
                "capture_available-002": {
                    "reference_move": 2,
                    "decision": "search_overrides_prior",
                    "current": {"selected_move": 2, "child_stats_available": True},
                    "original_challenger": {"selected_move": 2, "child_stats_available": True},
                    "rebalanced_challenger": {
                        "selected_move": 1,
                        "raw_policy_distribution": {"2": 0.48},
                        "searched_visit_distribution": {"2": 0.19, "1": 0.61},
                        "per_move_q_values": {"2": 0.44, "1": 0.58},
                        "child_stats_available": True,
                        "visit_snapshots": None,
                    },
                    "notes": ["prior corrected but searched move still wrong"],
                }
            }
        )

        self.assertEqual("insufficient_child_stats", rows["capture_available-002"]["decision"])
        self.assertEqual(
            ["child stats unavailable for one or more compared artifacts"],
            rows["capture_available-002"]["notes"],
        )

    def test_build_search_value_interaction_payload_populates_rows_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_run = Path(tmp) / "original"
            rebalanced_run = Path(tmp) / "rebalanced"
            original_run.mkdir()
            rebalanced_run.mkdir()

            with mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.resolve_target_rows",
                return_value=[
                    "capture_available-002",
                    "capture_available-003",
                    "incumbent_proxy_disagreement-031",
                    "incumbent_proxy_disagreement-033",
                    "opening_plies_1_8-057",
                ],
            ), mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.load_row_context",
                side_effect=[
                    {
                        "row_id": "capture_available-002",
                        "bucket": "capture_available",
                        "phase": "opening",
                        "reference_move": 2,
                        "teacher_value": 0.5028,
                        "current_row": {
                            "state": {"player_pits": [4, 4, 4, 4, 4, 4]},
                            "system_value": 0.1135,
                            "value_error": 0.3893,
                            "legal_moves": [0, 1, 2, 3, 4],
                        },
                        "original_row": {"system_value": 0.5441, "value_error": 0.0413},
                        "rebalanced_row": {"system_value": 0.5681, "value_error": 0.0653},
                    },
                    {
                        "row_id": "capture_available-003",
                        "bucket": "capture_available",
                        "phase": "opening",
                        "reference_move": 2,
                        "teacher_value": 0.55,
                        "current_row": {
                            "state": {"player_pits": [4, 4, 4, 4, 4, 4]},
                            "system_value": 0.18,
                            "value_error": 0.37,
                            "legal_moves": [0, 1, 2, 3, 4],
                        },
                        "original_row": {"system_value": 0.5, "value_error": 0.05},
                        "rebalanced_row": {"system_value": 0.52, "value_error": 0.03},
                    },
                    {
                        "row_id": "incumbent_proxy_disagreement-031",
                        "bucket": "incumbent_proxy_disagreement",
                        "phase": "mid",
                        "reference_move": 4,
                        "teacher_value": 0.65,
                        "current_row": {
                            "state": {"player_pits": [4, 4, 4, 4, 4, 4]},
                            "system_value": 0.17,
                            "value_error": 0.49,
                            "legal_moves": [0, 2, 4],
                        },
                        "original_row": {"system_value": 0.53, "value_error": 0.12},
                        "rebalanced_row": {"system_value": 0.57, "value_error": 0.09},
                    },
                    {
                        "row_id": "incumbent_proxy_disagreement-033",
                        "bucket": "incumbent_proxy_disagreement",
                        "phase": "mid",
                        "reference_move": 4,
                        "teacher_value": 0.61,
                        "current_row": {
                            "state": {"player_pits": [4, 4, 4, 4, 4, 4]},
                            "system_value": 0.15,
                            "value_error": 0.46,
                            "legal_moves": [0, 2, 4],
                        },
                        "original_row": {"system_value": 0.48, "value_error": 0.13},
                        "rebalanced_row": {"system_value": 0.51, "value_error": 0.1},
                    },
                    {
                        "row_id": "opening_plies_1_8-057",
                        "bucket": "opening_plies_1_8",
                        "phase": "opening",
                        "reference_move": 1,
                        "teacher_value": 0.12,
                        "current_row": {
                            "state": {"player_pits": [4, 4, 4, 4, 4, 4]},
                            "system_value": 0.07,
                            "value_error": 0.05,
                            "legal_moves": [0, 1, 2],
                        },
                        "original_row": {"system_value": 0.11, "value_error": 0.01},
                        "rebalanced_row": {"system_value": 0.09, "value_error": 0.03},
                    },
                ],
            ), mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.load_selected_artifact_path",
                side_effect=[
                    "/artifacts/original-selected",
                    "/artifacts/rebalanced-selected",
                ],
            ), mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.load_arena_module",
                return_value=type(
                    "FakeArena",
                    (),
                    {
                        "ArtifactEvaluator": staticmethod(lambda path: f"evaluator:{path}"),
                        "build_eval_search_options": staticmethod(lambda: {"root_policy_mode": "deterministic"}),
                        "evaluate_artifact_position": staticmethod(
                            lambda **kwargs: {
                                "selected_move": kwargs["state"].get("selected_move", 0),
                                "policy": kwargs["state"].get("policy", []),
                                "visits": kwargs["state"].get("visits", []),
                                "child_stats": kwargs["state"].get("child_stats", []),
                            }
                        ),
                    },
                ),
            ), mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.probe_artifact_position",
                side_effect=[
                    {"selected_move": 2, "value": 0.5, "policy": [0.1, 0.1, 0.6, 0.1, 0.1, 0.0], "visits": [2, 2, 20, 4, 4, 0], "child_stats": [{"move": 2, "visits": 20, "q_value": 0.5}], "selection_breakdown": {"policy_top_move": 2, "visit_top_move": 2, "q_top_move": 2}, "visit_snapshots": [{"simulation": 64, "top_move": 2, "top_visits": 20}]},
                    {"selected_move": 2, "value": 0.52, "policy": [0.1, 0.1, 0.6, 0.1, 0.1, 0.0], "visits": [2, 2, 20, 4, 4, 0], "child_stats": [{"move": 2, "visits": 20, "q_value": 0.5}], "selection_breakdown": {"policy_top_move": 2, "visit_top_move": 2, "q_top_move": 2}, "visit_snapshots": [{"simulation": 64, "top_move": 2, "top_visits": 20}]},
                    {"selected_move": 0, "value": 0.61, "policy": [0.08, 0.1, 0.51, 0.16, 0.15, 0.0], "visits": [22, 4, 7, 9, 6, 0], "child_stats": [{"move": 0, "visits": 22, "q_value": 0.61}, {"move": 2, "visits": 7, "q_value": 0.43}], "selection_breakdown": {"policy_top_move": 2, "visit_top_move": 0, "q_top_move": 0}, "visit_snapshots": [{"simulation": 32, "top_move": 2, "top_visits": 7}, {"simulation": 64, "top_move": 0, "top_visits": 22}]},
                    {"selected_move": 2, "value": 0.54, "policy": [0.1, 0.1, 0.55, 0.15, 0.1, 0.0], "visits": [3, 3, 18, 5, 3, 0], "child_stats": [{"move": 2, "visits": 18, "q_value": 0.49}], "selection_breakdown": {"policy_top_move": 2, "visit_top_move": 2, "q_top_move": 2}, "visit_snapshots": [{"simulation": 64, "top_move": 2, "top_visits": 18}]},
                    {"selected_move": 2, "value": 0.55, "policy": [0.1, 0.1, 0.55, 0.15, 0.1, 0.0], "visits": [3, 3, 18, 5, 3, 0], "child_stats": [{"move": 2, "visits": 18, "q_value": 0.49}], "selection_breakdown": {"policy_top_move": 2, "visit_top_move": 2, "q_top_move": 2}, "visit_snapshots": [{"simulation": 64, "top_move": 2, "top_visits": 18}]},
                    {"selected_move": 3, "value": 0.56, "policy": [0.1, 0.1, 0.56, 0.14, 0.1, 0.0], "visits": [2, 2, 6, 19, 3, 0], "child_stats": [{"move": 2, "visits": 6, "q_value": 0.44}, {"move": 3, "visits": 19, "q_value": 0.6}], "selection_breakdown": {"policy_top_move": 2, "visit_top_move": 3, "q_top_move": 3}, "visit_snapshots": [{"simulation": 32, "top_move": 2, "top_visits": 6}, {"simulation": 64, "top_move": 3, "top_visits": 19}]},
                    {"selected_move": 4, "value": 0.63, "policy": [0.1, 0.0, 0.2, 0.0, 0.7, 0.0], "visits": [1, 0, 2, 0, 21, 0], "child_stats": [{"move": 4, "visits": 21, "q_value": 0.62}], "selection_breakdown": {"policy_top_move": 4, "visit_top_move": 4, "q_top_move": 4}, "visit_snapshots": []},
                    {"selected_move": 4, "value": 0.62, "policy": [0.1, 0.0, 0.2, 0.0, 0.7, 0.0], "visits": [1, 0, 2, 0, 21, 0], "child_stats": [{"move": 4, "visits": 21, "q_value": 0.62}], "selection_breakdown": {"policy_top_move": 4, "visit_top_move": 4, "q_top_move": 4}, "visit_snapshots": []},
                    {"selected_move": 2, "value": 0.57, "policy": [0.1, 0.0, 0.2, 0.0, 0.7, 0.0], "visits": [0, 0, 18, 0, 6, 0], "child_stats": [{"move": 2, "visits": 18, "q_value": 0.71}, {"move": 4, "visits": 6, "q_value": 0.24}], "selection_breakdown": {"policy_top_move": 4, "visit_top_move": 2, "q_top_move": 2}, "visit_snapshots": [{"simulation": 64, "top_move": 2, "top_visits": 18}]},
                    {"selected_move": 4, "value": 0.58, "policy": [0.1, 0.0, 0.18, 0.0, 0.72, 0.0], "visits": [0, 0, 3, 0, 21, 0], "child_stats": [{"move": 4, "visits": 21, "q_value": 0.59}], "selection_breakdown": {"policy_top_move": 4, "visit_top_move": 4, "q_top_move": 4}, "visit_snapshots": []},
                    {"selected_move": 4, "value": 0.59, "policy": [0.1, 0.0, 0.18, 0.0, 0.72, 0.0], "visits": [0, 0, 3, 0, 21, 0], "child_stats": [{"move": 4, "visits": 21, "q_value": 0.59}], "selection_breakdown": {"policy_top_move": 4, "visit_top_move": 4, "q_top_move": 4}, "visit_snapshots": []},
                    {"selected_move": 4, "value": 0.6, "policy": [0.1, 0.0, 0.2, 0.0, 0.7, 0.0], "visits": [0, 0, 6, 0, 18, 0], "child_stats": [{"move": 4, "visits": 18, "q_value": 0.58}, {"move": 2, "visits": 6, "q_value": 0.31}], "selection_breakdown": {"policy_top_move": 4, "visit_top_move": 4, "q_top_move": 4}, "visit_snapshots": []},
                    {"selected_move": 1, "value": 0.12, "policy": [0.2, 0.5, 0.3], "visits": [4, 18, 10], "child_stats": [{"move": 1, "visits": 18, "q_value": 0.13}, {"move": 2, "visits": 10, "q_value": 0.11}], "selection_breakdown": {"policy_top_move": 1, "visit_top_move": 1, "q_top_move": 1}, "visit_snapshots": []},
                    {"selected_move": 1, "value": 0.11, "policy": [0.2, 0.5, 0.3], "visits": [4, 18, 10], "child_stats": [{"move": 1, "visits": 18, "q_value": 0.13}, {"move": 2, "visits": 10, "q_value": 0.11}], "selection_breakdown": {"policy_top_move": 1, "visit_top_move": 1, "q_top_move": 1}, "visit_snapshots": []},
                    {"selected_move": 2, "value": 0.09, "policy": [0.24, 0.46, 0.3], "visits": [5, 11, 16], "child_stats": [{"move": 1, "visits": 11, "q_value": 0.1}, {"move": 2, "visits": 16, "q_value": 0.14}], "selection_breakdown": {"policy_top_move": 1, "visit_top_move": 2, "q_top_move": 2}, "visit_snapshots": [{"simulation": 64, "top_move": 2, "top_visits": 16}]},
                ],
            ):
                payload = module.build_search_value_interaction_payload(
                    original_run_dir=original_run,
                    rebalanced_run_dir=rebalanced_run,
                    current_artifact_path="model-artifact/current",
                    explicit_rows=None,
                )

        self.assertEqual(module.SEARCH_VALUE_INTERACTION_SCHEMA, payload["schema"])
        self.assertEqual(
            str(module.diagnostic_out_path(rebalanced_run_dir=rebalanced_run)),
            payload["source_diagnostic_path"],
        )
        self.assertEqual(
            [
                "capture_available-002",
                "capture_available-003",
                "incumbent_proxy_disagreement-031",
                "incumbent_proxy_disagreement-033",
            ],
            payload["primary_row_ids"],
        )
        self.assertEqual(["opening_plies_1_8-057"], payload["comparator_row_ids"])
        self.assertEqual(
            {
                "bad_priors": 0,
                "search_overrides_prior": 2,
                "q_value_backup_issue": 1,
                "insufficient_child_stats": 0,
                "mixed": 2,
            },
            payload["summary"]["decision_counts"],
        )
        self.assertEqual("search_value_interaction_investigation", payload["summary"]["next_branch"])
        self.assertEqual(4, payload["summary"]["primary_row_count"])
        self.assertEqual(1, payload["summary"]["comparator_row_count"])
        self.assertEqual("search_overrides_prior", payload["rows"]["capture_available-002"]["decision"])
        self.assertEqual("search_overrides_prior", payload["rows"]["capture_available-003"]["decision"])
        self.assertEqual("q_value_backup_issue", payload["rows"]["incumbent_proxy_disagreement-031"]["decision"])
        self.assertEqual("mixed", payload["rows"]["incumbent_proxy_disagreement-033"]["decision"])
        self.assertEqual("mixed", payload["rows"]["opening_plies_1_8-057"]["decision"])
        self.assertEqual(
            "policy leans to 2, visits finish on 0, q-values favor 0, snapshots available",
            payload["rows"]["capture_available-002"]["row_mechanism_summary"],
        )
        self.assertEqual(
            "policy leans to 4, visits finish on 2, q-values favor 2, snapshots available",
            payload["rows"]["incumbent_proxy_disagreement-031"]["row_mechanism_summary"],
        )
        self.assertEqual(
            "policy leans to 1, visits finish on 2, q-values favor 2, snapshots available",
            payload["rows"]["opening_plies_1_8-057"]["row_mechanism_summary"],
        )
        self.assertEqual([], payload["rows"]["incumbent_proxy_disagreement-033"]["rebalanced_challenger"]["visit_snapshots"])
        self.assertEqual(
            "policy leans to 4, visits finish on 4, q-values favor 4, snapshots empty",
            payload["rows"]["incumbent_proxy_disagreement-033"]["row_mechanism_summary"],
        )

    def test_build_search_value_interaction_payload_reuses_one_evaluator_per_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_run = Path(tmp) / "original"
            rebalanced_run = Path(tmp) / "rebalanced"
            original_run.mkdir()
            rebalanced_run.mkdir()
            evaluator_creations = []
            evaluate_calls = []

            fake_arena = type(
                "FakeArena",
                (),
                {
                    "ArtifactEvaluator": staticmethod(lambda path: evaluator_creations.append(str(path)) or f"evaluator:{path}"),
                    "build_eval_search_options": staticmethod(lambda: {"root_policy_mode": "deterministic"}),
                    "evaluate_artifact_position": staticmethod(
                        lambda **kwargs: evaluate_calls.append(kwargs["evaluator"]) or {
                            "selected_move": 2,
                            "value": 0.5,
                            "policy": [0.1, 0.1, 0.6, 0.1, 0.1, 0.0],
                            "visits": [2, 2, 20, 4, 4, 0],
                            "child_stats": [{"move": 2, "visits": 20, "q_value": 0.5}],
                        }
                    ),
                },
            )

            with mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.resolve_target_rows",
                return_value=["capture_available-002", "capture_available-003"],
            ), mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.load_row_context",
                side_effect=[
                    {
                        "row_id": "capture_available-002",
                        "bucket": "capture_available",
                        "phase": "opening",
                        "reference_move": 2,
                        "teacher_value": 0.5,
                        "current_row": {"state": {"player_pits": [4, 4, 4, 4, 4, 4]}, "system_value": 0.1, "value_error": 0.4, "legal_moves": [0, 1, 2, 3, 4]},
                        "original_row": {"system_value": 0.54, "value_error": 0.04},
                        "rebalanced_row": {"system_value": 0.57, "value_error": 0.07},
                    },
                    {
                        "row_id": "capture_available-003",
                        "bucket": "capture_available",
                        "phase": "opening",
                        "reference_move": 2,
                        "teacher_value": 0.5,
                        "current_row": {"state": {"player_pits": [4, 4, 4, 4, 4, 4]}, "system_value": 0.1, "value_error": 0.4, "legal_moves": [0, 1, 2, 3, 4]},
                        "original_row": {"system_value": 0.54, "value_error": 0.04},
                        "rebalanced_row": {"system_value": 0.57, "value_error": 0.07},
                    },
                ],
            ), mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.load_selected_artifact_path",
                side_effect=["/artifacts/original", "/artifacts/rebalanced", "/artifacts/original", "/artifacts/rebalanced"],
            ), mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.load_arena_module",
                return_value=fake_arena,
            ):
                module.build_search_value_interaction_payload(
                    original_run_dir=original_run,
                    rebalanced_run_dir=rebalanced_run,
                    current_artifact_path="model-artifact/current",
                    explicit_rows=None,
                )

        self.assertEqual(
            ["model-artifact/current", "/artifacts/original", "/artifacts/rebalanced"],
            evaluator_creations,
        )
        self.assertEqual(6, len(evaluate_calls))

    def test_main_writes_original_and_sibling_search_interaction_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            original_run = tmp_path / "original"
            rebalanced_run = tmp_path / "rebalanced"
            (original_run / "final").mkdir(parents=True)
            (rebalanced_run / "final").mkdir(parents=True)
            stdout = io.StringIO()

            with mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.build_search_interaction_payload",
                return_value={"schema": module.SEARCH_INTERACTION_SCHEMA, "rows": {}, "summary": {}, "row_source": {"resolved_rows": []}},
            ), mock.patch("sys.stdout", stdout):
                exit_code = module.main(
                    [
                        "--original-run",
                        str(original_run),
                        "--rebalanced-run",
                        str(rebalanced_run),
                        "--current-artifact",
                        "model-artifact/current",
                    ]
                )

            source_path = module.diagnostic_out_path(rebalanced_run_dir=rebalanced_run)
            sibling_path = rebalanced_run / "final" / "search_value_interaction_diagnostic.json"
            source_written = json.loads(source_path.read_text(encoding="utf-8"))
            written = json.loads(sibling_path.read_text(encoding="utf-8"))
            reported = json.loads(stdout.getvalue())

            self.assertEqual(0, exit_code)
            self.assertTrue(source_path.exists())
            self.assertEqual(module.SEARCH_VALUE_INTERACTION_SCHEMA, written["schema"])
            self.assertEqual(module.SEARCH_INTERACTION_SCHEMA, source_written["schema"])
            self.assertEqual(str(source_path), written["source_diagnostic_path"])
            self.assertTrue(sibling_path.exists())
            self.assertEqual(str(source_path), reported["artifact_path"])
            self.assertEqual(module.SEARCH_INTERACTION_SCHEMA, reported["schema"])
            self.assertEqual(str(sibling_path), reported["search_value_interaction_artifact_path"])

    def test_main_derives_sibling_payload_from_already_built_source_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            original_run = tmp_path / "original"
            rebalanced_run = tmp_path / "rebalanced"
            (original_run / "final").mkdir(parents=True)
            (rebalanced_run / "final").mkdir(parents=True)

            source_payload = {
                "schema": module.SEARCH_INTERACTION_SCHEMA,
                "rows": {"capture_available-002": {"decision": "mixed"}},
                "summary": {},
                "row_source": {"resolved_rows": ["capture_available-002"]},
            }
            sibling_payload = {
                "schema": module.SEARCH_VALUE_INTERACTION_SCHEMA,
                "source_diagnostic_path": str(module.diagnostic_out_path(rebalanced_run_dir=rebalanced_run)),
                "rows": {
                    "capture_available-002": {
                        "decision": "mixed",
                        "row_mechanism_summary": "policy leans to 2, visits finish on 2, q-values favor 2, snapshots missing",
                    }
                },
                "summary": {"primary_row_count": 1, "comparator_row_count": 0},
                "row_source": {"resolved_rows": ["capture_available-002"]},
            }

            with mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.build_search_interaction_payload",
                return_value=source_payload,
            ), mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.build_search_value_interaction_payload",
                side_effect=AssertionError("main should not rebuild sibling from scratch"),
            ), mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.build_search_value_interaction_payload_from_source_payload",
                return_value=sibling_payload,
            ) as build_from_source:
                module.main(
                    [
                        "--original-run",
                        str(original_run),
                        "--rebalanced-run",
                        str(rebalanced_run),
                        "--current-artifact",
                        "model-artifact/current",
                    ]
                )

            sibling_path = rebalanced_run / "final" / "search_value_interaction_diagnostic.json"
            written = json.loads(sibling_path.read_text(encoding="utf-8"))
            build_from_source.assert_called_once_with(
                payload=source_payload,
                source_diagnostic_path=str(module.diagnostic_out_path(rebalanced_run_dir=rebalanced_run)),
            )
            self.assertEqual(sibling_payload, written)

    def test_build_search_value_interaction_payload_recomputes_summary_from_sibling_rows(self):
        payload = {
            "schema": module.SEARCH_INTERACTION_SCHEMA,
            "row_source": {"resolved_rows": ["capture_available-002"]},
            "rows": {
                "capture_available-002": {
                    "reference_move": 2,
                    "decision": "search_overrides_prior",
                    "current": {"selected_move": 2, "child_stats_available": True},
                    "original_challenger": {"selected_move": 2, "child_stats_available": True},
                    "rebalanced_challenger": {
                        "selected_move": 1,
                        "raw_policy_distribution": {"2": 0.48},
                        "searched_visit_distribution": {"2": 0.19, "1": 0.61},
                        "per_move_q_values": {"2": 0.44, "1": 0.58},
                        "child_stats_available": True,
                        "visit_snapshots": None,
                    },
                    "notes": ["prior corrected but searched move still wrong"],
                }
            },
            "summary": {
                "decision_counts": {
                    "bad_priors": 0,
                    "search_overrides_prior": 1,
                    "q_value_backup_issue": 0,
                    "insufficient_child_stats": 0,
                    "mixed": 0,
                },
                "next_branch": "search_value_interaction_investigation",
            },
        }

        sibling_payload = module.build_search_value_interaction_payload_from_source_payload(
            payload=payload,
            source_diagnostic_path="/tmp/source.json",
        )

        self.assertEqual(
            {
                "bad_priors": 0,
                "search_overrides_prior": 0,
                "q_value_backup_issue": 0,
                "insufficient_child_stats": 1,
                "mixed": 0,
            },
            sibling_payload["summary"]["decision_counts"],
        )
        self.assertEqual("broaden_failure_surface_review", sibling_payload["summary"]["next_branch"])

    def test_module_cli_executes_main_entrypoint(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ml.alphazero_lite.search_interaction_diagnostic",
                "--help",
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode)
        self.assertIn("--rebalanced-run", result.stdout)
