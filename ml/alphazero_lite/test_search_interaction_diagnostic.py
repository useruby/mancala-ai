import json
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

    def test_build_search_interaction_payload_populates_rows_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_run = Path(tmp) / "original"
            rebalanced_run = Path(tmp) / "rebalanced"
            original_run.mkdir()
            rebalanced_run.mkdir()

            fake_arena = type(
                "FakeArena",
                (),
                {
                    "ArtifactEvaluator": staticmethod(lambda path: f"evaluator:{path}"),
                    "build_eval_search_options": staticmethod(lambda: {"root_policy_mode": "deterministic"}),
                    "evaluate_artifact_position": staticmethod(lambda **kwargs: kwargs["probe_summary"]),
                },
            )

            with mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.resolve_target_rows",
                return_value=["capture_available-002", "incumbent_proxy_disagreement-031"],
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
                    {"selected_move": 2, "policy": [0.1, 0.1, 0.6, 0.1, 0.1, 0.0], "visits": [2, 2, 20, 4, 4, 0], "child_stats": [{"move": 2, "visits": 20, "q_value": 0.5}]},
                    {"selected_move": 2, "policy": [0.1, 0.1, 0.6, 0.1, 0.1, 0.0], "visits": [2, 2, 20, 4, 4, 0], "child_stats": [{"move": 2, "visits": 20, "q_value": 0.5}]},
                    {"selected_move": 0, "policy": [0.08, 0.1, 0.51, 0.16, 0.15, 0.0], "visits": [22, 4, 7, 9, 6, 0], "child_stats": [{"move": 0, "visits": 22, "q_value": 0.61}, {"move": 2, "visits": 7, "q_value": 0.43}]},
                    {"selected_move": 4, "policy": [0.1, 0.0, 0.2, 0.0, 0.7, 0.0], "visits": [1, 0, 2, 0, 21, 0], "child_stats": [{"move": 4, "visits": 21, "q_value": 0.62}]},
                    {"selected_move": 4, "policy": [0.1, 0.0, 0.2, 0.0, 0.7, 0.0], "visits": [1, 0, 2, 0, 21, 0], "child_stats": [{"move": 4, "visits": 21, "q_value": 0.62}]},
                    {"selected_move": 2, "policy": [0.1, 0.0, 0.2, 0.0, 0.7, 0.0], "visits": [0, 0, 18, 0, 6, 0], "child_stats": [{"move": 2, "visits": 18, "q_value": 0.71}, {"move": 4, "visits": 6, "q_value": 0.24}]},
                ],
            ):
                payload = module.build_search_interaction_payload(
                    original_run_dir=original_run,
                    rebalanced_run_dir=rebalanced_run,
                    current_artifact_path="model-artifact/current",
                    explicit_rows=None,
                )

        self.assertEqual(module.SEARCH_INTERACTION_SCHEMA, payload["schema"])
        self.assertEqual(
            {
                "bad_priors": 0,
                "search_overrides_prior": 1,
                "q_value_backup_issue": 1,
                "insufficient_child_stats": 0,
                "mixed": 0,
            },
            payload["summary"]["decision_counts"],
        )
        self.assertEqual("search_value_interaction_investigation", payload["summary"]["next_branch"])
        self.assertEqual("search_overrides_prior", payload["rows"]["capture_available-002"]["decision"])
        self.assertEqual("q_value_backup_issue", payload["rows"]["incumbent_proxy_disagreement-031"]["decision"])

    def test_build_search_interaction_payload_reuses_one_evaluator_per_artifact(self):
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
                module.build_search_interaction_payload(
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

    def test_main_writes_search_interaction_artifact_under_rebalanced_final_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            original_run = tmp_path / "original"
            rebalanced_run = tmp_path / "rebalanced"
            (original_run / "final").mkdir(parents=True)
            (rebalanced_run / "final").mkdir(parents=True)

            with mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.build_search_interaction_payload",
                return_value={
                    "schema": module.SEARCH_INTERACTION_SCHEMA,
                    "rows": {},
                    "summary": {},
                    "row_source": {"resolved_rows": []},
                },
            ):
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

            written = json.loads((rebalanced_run / "final" / "search_interaction_diagnostic.json").read_text(encoding="utf-8"))

        self.assertEqual(0, exit_code)
        self.assertEqual(module.SEARCH_INTERACTION_SCHEMA, written["schema"])
