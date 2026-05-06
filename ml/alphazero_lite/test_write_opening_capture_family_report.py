import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class WriteOpeningCaptureFamilyReportTest(unittest.TestCase):
    def opening_row(self, row_id, *, reference_move=3, legal_moves=None, bucket="capture_available", phase="opening"):
        return {
            "id": row_id,
            "state": {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            },
            "legal_moves": [0, 1, 2, 3, 4] if legal_moves is None else legal_moves,
            "bucket": bucket,
            "phase": phase,
            "reference_move": reference_move,
        }

    def test_select_tracked_family_rows_enforces_task_three_signature(self):
        from ml.alphazero_lite import write_opening_capture_family_report as module

        rows = [
            self.opening_row("capture_available-017"),
            self.opening_row("capture_available-019", legal_moves=[0, 1, 2, 3]),
            self.opening_row("capture_available-024", reference_move=2),
            self.opening_row("capture_available-031", phase="midgame"),
            self.opening_row("high_imbalance-001"),
            self.opening_row("capture_available-099"),
        ]

        selected = module.select_tracked_family_rows(rows)

        self.assertEqual(["capture_available-017", "capture_available-099"], [row["id"] for row in selected])

    def test_build_report_includes_current_and_candidate_prior_and_search_summaries(self):
        from ml.alphazero_lite import write_opening_capture_family_report as module

        current_artifact = Path("/tmp/current-artifact")
        candidate_artifact = Path("/tmp/candidate-artifact")
        suite_rows = [
            self.opening_row("capture_available-017"),
            self.opening_row("capture_available-099"),
            self.opening_row("high_imbalance-002"),
        ]

        def fake_evaluate_artifact_position(*, artifact_path, evaluator, state, simulations, seed, c_puct, search_options, ablation_mode="full"):
            del state, simulations, seed, c_puct, search_options, ablation_mode
            self.assertEqual(artifact_path, evaluator)
            if artifact_path == current_artifact:
                return {
                    "selected_move": 1,
                    "legal_moves": [0, 1, 2, 3, 4],
                    "policy": [0.05, 0.15, 0.2, 0.5, 0.1, 0.0],
                    "visits": [3.0, 5.0, 7.0, 31.0, 4.0, 0.0],
                    "value": 0.2,
                    "child_stats": [
                        {"move": 0, "visits": 3, "q_value": 0.05},
                        {"move": 1, "visits": 5, "q_value": 0.15},
                        {"move": 2, "visits": 7, "q_value": 0.2},
                        {"move": 3, "visits": 31, "q_value": 0.55},
                        {"move": 4, "visits": 4, "q_value": 0.1},
                    ],
                }
            return {
                "selected_move": 3,
                "legal_moves": [0, 1, 2, 3, 4],
                "policy": [0.1, 0.1, 0.2, 0.45, 0.15, 0.0],
                "visits": [4.0, 4.0, 8.0, 20.0, 6.0, 0.0],
                "value": 0.35,
                "child_stats": [
                    {"move": 0, "visits": 4, "q_value": 0.1},
                    {"move": 1, "visits": 4, "q_value": 0.1},
                    {"move": 2, "visits": 8, "q_value": 0.2},
                    {"move": 3, "visits": 20, "q_value": 0.6},
                    {"move": 4, "visits": 6, "q_value": 0.25},
                    ],
                }

        fake_arena = type(
            "FakeArena",
            (),
            {
                "ArtifactEvaluator": staticmethod(lambda path: path),
                "build_eval_search_options": staticmethod(lambda: {"root_policy_mode": "deterministic"}),
                "evaluate_artifact_position": staticmethod(fake_evaluate_artifact_position),
            },
        )

        with mock.patch("ml.alphazero_lite.write_opening_capture_family_report.load_suite_rows", return_value=suite_rows), mock.patch(
            "ml.alphazero_lite.write_opening_capture_family_report.load_arena_module",
            return_value=fake_arena,
        ):
            report = module.build_report(
                suite_path=Path("/tmp/opening-suite.json"),
                current_artifact_path=current_artifact,
                candidate_artifact_path=candidate_artifact,
                artifact_simulations=64,
                c_puct=1.25,
                seed=7,
            )

        self.assertEqual("opening_capture_family_report_v1", report["schema"])
        self.assertEqual(str(current_artifact), report["current_artifact_path"])
        self.assertEqual(str(candidate_artifact), report["candidate_artifact_path"])
        self.assertEqual(2, len(report["rows"]))
        self.assertEqual(["capture_available-017", "capture_available-099"], [row["id"] for row in report["rows"]])
        first_row = report["rows"][0]
        self.assertEqual(3, first_row["reference_move"])
        self.assertEqual([0, 1, 2, 3, 4], first_row["legal_moves"])
        self.assertEqual(
            {"selected_move", "value", "early_mass", "move_3_mass", "move_3_margin", "reference_move"},
            set(first_row["current_prior_summary"]),
        )
        self.assertEqual(0.2, first_row["current_prior_summary"]["early_mass"])
        self.assertEqual(0.3, first_row["current_prior_summary"]["move_3_margin"])
        self.assertEqual(1, first_row["current_searched_summary"]["selected_move"])
        self.assertEqual(0.16, first_row["current_searched_summary"]["early_mass"])
        self.assertEqual(0.46, first_row["current_searched_summary"]["move_3_margin"])
        self.assertEqual(3, first_row["candidate_prior_summary"]["selected_move"])
        self.assertEqual(0.2, first_row["candidate_prior_summary"]["early_mass"])
        self.assertEqual(0.25, first_row["candidate_prior_summary"]["move_3_margin"])
        self.assertEqual(3, first_row["candidate_searched_summary"]["selected_move"])
        self.assertEqual(0.1905, first_row["candidate_searched_summary"]["early_mass"])
        self.assertEqual(0.2857, first_row["candidate_searched_summary"]["move_3_margin"])

    def test_summarize_prior_uses_prior_best_move_not_search_selected_move(self):
        from ml.alphazero_lite import write_opening_capture_family_report as module

        summary = module.summarize_prior(
            {
                "selected_move": 1,
                "value": 0.25,
                "policy": [0.05, 0.15, 0.2, 0.5, 0.1, 0.0],
            },
            reference_move=3,
            legal_moves=[0, 1, 2, 3, 4],
        )

        self.assertEqual(3, summary["selected_move"])

    def test_summarize_prior_includes_real_summary_shape(self):
        from ml.alphazero_lite import write_opening_capture_family_report as module

        summary = module.summarize_prior(
            {
                "selected_move": 1,
                "value": 0.25,
                "policy": [0.05, 0.15, 0.2, 0.5, 0.1, 0.0],
            },
            reference_move=3,
            legal_moves=[0, 1, 2, 3, 4],
        )

        self.assertEqual(
            {"selected_move", "value", "early_mass", "move_3_mass", "move_3_margin", "reference_move"},
            set(summary),
        )

    def test_write_report_persists_json_payload(self):
        from ml.alphazero_lite import write_opening_capture_family_report as module

        payload = {"schema": "opening_capture_family_report_v1", "rows": []}

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "reports" / "opening_capture_family_report.json"
            module.write_report(out_path, payload)
            self.assertEqual(payload, json.loads(out_path.read_text(encoding="utf-8")))

    def test_enrich_suite_rows_supplies_reference_move_for_tracked_family_rows(self):
        from ml.alphazero_lite import write_opening_capture_family_report as module

        tracked_state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        suite_rows = [
            {
                "id": "capture_available-017",
                "state": tracked_state,
                "legal_moves": [0, 1, 2, 3, 4],
                "bucket": "capture_available",
                "phase": "opening",
                "reference_move": None,
            },
            {
                "id": "capture_available-024",
                "state": tracked_state,
                "legal_moves": [0, 1, 2, 3, 4],
                "bucket": "capture_available",
                "phase": "opening",
                "reference_move": None,
            },
        ]

        enriched = module.enrich_suite_rows_with_reference_moves(
            suite_rows,
            {
                module.canonical_state_key(tracked_state): 3,
            },
        )

        self.assertEqual([3, 3], [row["reference_move"] for row in enriched])
        self.assertEqual(["capture_available-017", "capture_available-024"], [row["id"] for row in module.select_tracked_family_rows(enriched)])

    def test_enrich_suite_rows_prefers_shared_reference_move_over_stale_suite_value(self):
        from ml.alphazero_lite import write_opening_capture_family_report as module

        tracked_state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        suite_rows = [
            {
                "id": "capture_available-017",
                "state": tracked_state,
                "legal_moves": [0, 1, 2, 3, 4],
                "bucket": "capture_available",
                "phase": "opening",
                "reference_move": 1,
            }
        ]

        enriched = module.enrich_suite_rows_with_reference_moves(
            suite_rows,
            {module.canonical_state_key(tracked_state): 3},
        )

        self.assertEqual(3, enriched[0]["reference_move"])

    def test_build_report_uses_same_seed_for_current_and_candidate_search(self):
        from ml.alphazero_lite import write_opening_capture_family_report as module

        current_artifact = Path("/tmp/current-artifact")
        candidate_artifact = Path("/tmp/candidate-artifact")
        suite_rows = [self.opening_row("capture_available-017")]
        calls = []

        def fake_evaluate_artifact_position(*, artifact_path, evaluator, state, simulations, seed, c_puct, search_options, ablation_mode="full"):
            del evaluator, state, simulations, c_puct, search_options, ablation_mode
            calls.append((artifact_path, seed))
            return {
                "selected_move": 3,
                "legal_moves": [0, 1, 2, 3, 4],
                "policy": [0.1, 0.1, 0.2, 0.45, 0.15, 0.0],
                "visits": [4.0, 4.0, 8.0, 20.0, 6.0, 0.0],
                "value": 0.35,
            }

        fake_arena = type(
            "FakeArena",
            (),
            {
                "ArtifactEvaluator": staticmethod(lambda path: path),
                "build_eval_search_options": staticmethod(lambda: {"root_policy_mode": "deterministic"}),
                "evaluate_artifact_position": staticmethod(fake_evaluate_artifact_position),
            },
        )

        with mock.patch("ml.alphazero_lite.write_opening_capture_family_report.load_suite_rows", return_value=suite_rows), mock.patch(
            "ml.alphazero_lite.write_opening_capture_family_report.load_arena_module",
            return_value=fake_arena,
        ):
            module.build_report(
                suite_path=Path("/tmp/opening-suite.json"),
                current_artifact_path=current_artifact,
                candidate_artifact_path=candidate_artifact,
                artifact_simulations=64,
                c_puct=1.25,
                seed=7,
            )

        self.assertEqual([(current_artifact, 7), (candidate_artifact, 7)], calls)

    def test_build_report_records_missing_reference_move_failure(self):
        from ml.alphazero_lite import write_opening_capture_family_report as module

        tracked_state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }

        with mock.patch(
            "ml.alphazero_lite.write_opening_capture_family_report.load_suite_rows",
            return_value=[
                {
                    "id": "capture_available-017",
                    "state": tracked_state,
                    "legal_moves": [0, 1, 2, 3, 4],
                    "bucket": "capture_available",
                    "phase": "opening",
                    "reference_move": None,
                }
            ],
        ), mock.patch(
            "ml.alphazero_lite.write_opening_capture_family_report.load_reference_moves",
            return_value={},
        ):
            report = module.build_report(
                suite_path=Path("/tmp/opening-suite.json"),
                current_artifact_path=Path("/tmp/current-artifact"),
                candidate_artifact_path=Path("/tmp/candidate-artifact"),
                artifact_simulations=64,
                c_puct=1.25,
                seed=7,
            )

        self.assertEqual("opening_capture_family_report_v1", report["schema"])
        self.assertEqual([], report["rows"])
        self.assertEqual(
            [
                {
                    "code": "missing_reference_move",
                    "id": "capture_available-017",
                }
            ],
            report["missing_references"],
        )

    def test_build_report_records_invalid_reference_move_failure_when_tracked_row_has_stale_value_without_shared_reference(self):
        from ml.alphazero_lite import write_opening_capture_family_report as module

        tracked_state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }

        with mock.patch(
            "ml.alphazero_lite.write_opening_capture_family_report.load_suite_rows",
            return_value=[
                {
                    "id": "capture_available-017",
                    "state": tracked_state,
                    "legal_moves": [0, 1, 2, 3, 4],
                    "bucket": "capture_available",
                    "phase": "opening",
                    "reference_move": 1,
                }
            ],
        ), mock.patch(
            "ml.alphazero_lite.write_opening_capture_family_report.load_reference_moves",
            return_value={},
        ):
            report = module.build_report(
                suite_path=Path("/tmp/opening-suite.json"),
                current_artifact_path=Path("/tmp/current-artifact"),
                candidate_artifact_path=Path("/tmp/candidate-artifact"),
                artifact_simulations=64,
                c_puct=1.25,
                seed=7,
            )

        self.assertEqual([], report["rows"])
        self.assertEqual(
            [{"code": "invalid_reference_move", "id": "capture_available-017", "reference_move": 1}],
            report["missing_references"],
        )

    def test_load_reference_moves_reads_shared_reference_artifact_rows(self):
        from ml.alphazero_lite import write_opening_capture_family_report as module

        tracked_state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }

        with tempfile.TemporaryDirectory() as tmp:
            reference_path = Path(tmp) / "reference_moves.json"
            reference_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_forensic_references_v1",
                        "rows": [
                            {
                                "id": "capture_available-017",
                                "canonical_state": module.canonical_state_key(tracked_state),
                                "state": tracked_state,
                                "reference_move": 3,
                                "teacher_value": 0.4183,
                                "reference_unstable": False,
                                "observed_reference_moves": [3],
                                "seed_samples": [{"seed": 2100, "reference_move": 3, "teacher_value": 0.4183}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            reference_moves = module.load_reference_moves(reference_path)

        self.assertEqual({module.canonical_state_key(tracked_state): 3}, reference_moves)

    def test_load_reference_moves_rejects_malformed_reference_artifact(self):
        from ml.alphazero_lite import write_opening_capture_family_report as module

        with tempfile.TemporaryDirectory() as tmp:
            reference_path = Path(tmp) / "reference_moves.json"
            reference_path.write_text("{not-json", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "reference artifact is not valid JSON"):
                module.load_reference_moves(reference_path)

    def test_load_reference_moves_rejects_missing_explicit_reference_artifact(self):
        from ml.alphazero_lite import write_opening_capture_family_report as module

        missing_path = Path("/tmp/does-not-exist-reference-moves.json")

        with self.assertRaisesRegex(ValueError, "reference artifact does not exist"):
            module.load_reference_moves(missing_path)

    def test_load_reference_moves_rejects_wrong_reference_artifact_schema(self):
        from ml.alphazero_lite import write_opening_capture_family_report as module

        with tempfile.TemporaryDirectory() as tmp:
            reference_path = Path(tmp) / "reference_moves.json"
            reference_path.write_text(
                json.dumps({"schema": "azlite_forensic_suite_v1", "rows": []}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "reference artifact must use schema azlite_forensic_references_v1"):
                module.load_reference_moves(reference_path)

    def test_load_reference_moves_rejects_reference_artifact_without_rows_list(self):
        from ml.alphazero_lite import write_opening_capture_family_report as module

        with tempfile.TemporaryDirectory() as tmp:
            reference_path = Path(tmp) / "reference_moves.json"
            reference_path.write_text(
                json.dumps({"schema": "azlite_forensic_references_v1", "rows": {}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "reference artifact must contain a rows list"):
                module.load_reference_moves(reference_path)

    def test_load_reference_moves_ignores_unstable_shared_reference_rows_without_reference_move(self):
        from ml.alphazero_lite import write_opening_capture_family_report as module

        tracked_state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }

        with tempfile.TemporaryDirectory() as tmp:
            reference_path = Path(tmp) / "reference_moves.json"
            reference_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_forensic_references_v1",
                        "rows": [
                            {
                                "id": "capture_available-017",
                                "canonical_state": module.canonical_state_key(tracked_state),
                                "state": tracked_state,
                                "reference_move": None,
                                "teacher_value": None,
                                "reference_unstable": True,
                                "observed_reference_moves": [3, 4],
                                "seed_samples": [
                                    {"seed": 2100, "reference_move": 3, "teacher_value": 0.4183},
                                    {"seed": 2200, "reference_move": 4, "teacher_value": 0.4012},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            reference_moves = module.load_reference_moves(reference_path)

        self.assertEqual({}, reference_moves)

    def test_build_report_uses_reference_forensics_to_enrich_tracked_rows(self):
        from ml.alphazero_lite import write_opening_capture_family_report as module

        current_artifact = Path("/tmp/current-artifact")
        candidate_artifact = Path("/tmp/candidate-artifact")
        tracked_state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        suite_rows = [
            {
                "id": "capture_available-017",
                "state": tracked_state,
                "legal_moves": [0, 1, 2, 3, 4],
                "bucket": "capture_available",
                "phase": "opening",
                "reference_move": None,
            },
            {
                "id": "capture_available-019",
                "state": tracked_state,
                "legal_moves": [0, 1, 2, 3, 4],
                "bucket": "capture_available",
                "phase": "opening",
                "reference_move": None,
            },
            {
                "id": "capture_available-024",
                "state": tracked_state,
                "legal_moves": [0, 1, 2, 3, 4],
                "bucket": "capture_available",
                "phase": "opening",
                "reference_move": None,
            },
        ]

        fake_summary = {
            "selected_move": 3,
            "legal_moves": [0, 1, 2, 3, 4],
            "policy": [0.1, 0.1, 0.2, 0.45, 0.15, 0.0],
            "visits": [4.0, 4.0, 8.0, 20.0, 6.0, 0.0],
            "value": 0.35,
        }
        fake_arena = type(
            "FakeArena",
            (),
            {
                "ArtifactEvaluator": staticmethod(lambda path: path),
                "build_eval_search_options": staticmethod(lambda: {"root_policy_mode": "deterministic"}),
                "evaluate_artifact_position": staticmethod(lambda **kwargs: fake_summary),
            },
        )

        with mock.patch("ml.alphazero_lite.write_opening_capture_family_report.load_suite_rows", return_value=suite_rows), mock.patch(
            "ml.alphazero_lite.write_opening_capture_family_report.load_reference_moves",
            return_value={module.canonical_state_key(tracked_state): 3},
        ), mock.patch(
            "ml.alphazero_lite.write_opening_capture_family_report.load_arena_module",
            return_value=fake_arena,
        ):
            report = module.build_report(
                suite_path=Path("/tmp/opening-suite.json"),
                current_artifact_path=current_artifact,
                candidate_artifact_path=candidate_artifact,
                artifact_simulations=64,
                c_puct=1.25,
                seed=7,
                reference_path=Path("/tmp/baseline_candidate_forensics.json"),
            )

        self.assertEqual(
            ["capture_available-017", "capture_available-019", "capture_available-024"],
            [row["id"] for row in report["rows"]],
        )
        self.assertEqual([], report["missing_references"])


if __name__ == "__main__":
    unittest.main()
