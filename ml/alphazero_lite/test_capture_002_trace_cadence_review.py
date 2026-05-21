import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ml.alphazero_lite import (
    capture_002_selection_score_trace as selection_score_module,
)
from ml.alphazero_lite import capture_002_trace_cadence_review as module


class Capture002TraceCadenceReviewContractTest(unittest.TestCase):
    def test_contract_constants_are_stable(self):
        self.assertEqual("azlite_capture_002_trace_cadence_review_v1", module.SCHEMA)
        self.assertEqual(
            "azlite_capture_002_trace_capture_v1", module.SOURCE_TRACE_CAPTURE_SCHEMA
        )
        self.assertEqual(
            "azlite_capture_002_selection_score_trace_v1",
            module.SOURCE_SELECTION_SCORE_SCHEMA,
        )
        self.assertEqual(
            {
                "trace_too_sparse": "write_002_trace_cadence_capture_spec",
                "cadence_adequate": "continue_002_threshold_too_strict_check",
            },
            module.CLASSIFICATION_DECISIONS,
        )

    def test_parse_args_reads_required_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_capture_path = Path(tmp) / "capture_002_trace_capture.json"
            selection_score_path = Path(tmp) / "capture_002_selection_score_trace.json"
            out_path = Path(tmp) / "capture_002_trace_cadence_review.json"

            args = module.parse_args(
                [
                    "--source-trace-capture-artifact",
                    str(trace_capture_path),
                    "--source-selection-score-artifact",
                    str(selection_score_path),
                    "--out",
                    str(out_path),
                ]
            )

        self.assertEqual(trace_capture_path, args.source_trace_capture_artifact)
        self.assertEqual(selection_score_path, args.source_selection_score_artifact)
        self.assertEqual(out_path, args.out)

    def test_parse_args_requires_all_paths(self):
        with self.assertRaises(SystemExit):
            module.parse_args(
                [
                    "--source-selection-score-artifact",
                    "/tmp/selection.json",
                    "--out",
                    "/tmp/out.json",
                ]
            )
        with self.assertRaises(SystemExit):
            module.parse_args(
                [
                    "--source-trace-capture-artifact",
                    "/tmp/trace.json",
                    "--out",
                    "/tmp/out.json",
                ]
            )
        with self.assertRaises(SystemExit):
            module.parse_args(
                [
                    "--source-trace-capture-artifact",
                    "/tmp/trace.json",
                    "--source-selection-score-artifact",
                    "/tmp/selection.json",
                ]
            )


class Capture002TraceCadenceReviewBuildPayloadTest(unittest.TestCase):
    def write_json(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def trace_capture_artifact(self, *, simulations: list[float]) -> dict:
        trace_points = []
        selected_moves = [2, 2, 0, 0][: len(simulations)]
        visits_by_simulation = {
            1.0: [0.0, 0.0, 1.0, 0.0, 0.0],
            8.0: [4.0, 0.0, 5.0, 0.0, 0.0],
            12.0: [7.0, 0.0, 5.0, 0.0, 0.0],
            16.0: [8.0, 4.0, 7.0, 2.0, 1.0],
        }
        for simulation, selected_move in zip(simulations, selected_moves):
            trace_points.append(
                {
                    "simulation": simulation,
                    "selected_move": selected_move,
                    "reference_move_by_prior": 2,
                    "visits": visits_by_simulation[simulation],
                    "moves": [
                        {"move": 0, "selection_score": 0.10, "q_value": 0.05},
                        {"move": 2, "selection_score": 0.14, "q_value": 0.07},
                    ],
                }
            )
        return {
            "schema": module.SOURCE_TRACE_CAPTURE_SCHEMA,
            "trace_origin": "extracted",
            "row_id": "capture_available-002",
            "reference_move": 2,
            "full_search_selected_move": 0,
            "trace_points": trace_points,
            "insufficiency_reasons": [],
        }

    def selection_score_artifact(
        self, *, visit_share: float, unresolved: bool = True
    ) -> dict:
        return {
            "schema": module.SOURCE_SELECTION_SCORE_SCHEMA,
            "trace_origin": "extracted",
            "source_artifact": {
                "row_id": "capture_available-002",
                "reference_move": 2,
                "full_search_selected_move": 0,
            },
            "classification": {
                "classification": "unresolved"
                if unresolved
                else "selection_score_pressure_confirmed",
                "evidence_summary": "Selection-score and Q-support timing do not cleanly separate for capture 002.",
            },
            "decision": "write_002_unresolved_trace_review_spec"
            if unresolved
            else "write_002_selection_pressure_ablation_spec",
            "insufficiency_reasons": [],
            "final_selected_minus_reference_visit_share": visit_share,
            "first_selected_material_visit_share_snapshot": None,
            "first_selected_meaningful_q_support_snapshot": None,
            "first_selected_selection_score_overtake_snapshot": None,
        }

    def dense_cadence_capture_artifact(
        self,
        *,
        provenance_passed: bool = True,
        trace_points: list[dict] | None = None,
        trace_points_summary: dict | None = None,
    ) -> dict:
        default_trace_points = [
            {
                "simulation": 1.0,
                "selected_move": 2,
                "reference_move_by_prior": 2,
                "visits": [0.0, 0.0, 1.0, 0.0, 0.0],
                "moves": [],
            },
            {
                "simulation": 4.0,
                "selected_move": 2,
                "reference_move_by_prior": 2,
                "visits": [2.0, 0.0, 3.0, 0.0, 0.0],
                "moves": [],
            },
            {
                "simulation": 8.0,
                "selected_move": 0,
                "reference_move_by_prior": 2,
                "visits": [4.0, 0.0, 5.0, 0.0, 0.0],
                "moves": [],
            },
            {
                "simulation": 12.0,
                "selected_move": 0,
                "reference_move_by_prior": 2,
                "visits": [7.0, 0.0, 5.0, 0.0, 0.0],
                "moves": [],
            },
            {
                "simulation": 16.0,
                "selected_move": 0,
                "reference_move_by_prior": 2,
                "visits": [8.0, 4.0, 7.0, 2.0, 1.0],
                "moves": [],
            },
        ]
        trace_points = default_trace_points if trace_points is None else trace_points
        default_trace_points_summary = {
            "unique_simulation_checkpoints": [1.0, 4.0, 8.0, 12.0, 16.0],
            "unique_simulation_checkpoint_count": 5,
            "duplicate_root_snapshot_count": 0,
            "first_divergent_selected_move_index": 2,
            "has_additional_checkpoint_between_divergence_and_final": True,
        }
        return {
            "schema": "azlite_capture_002_trace_cadence_capture_v1",
            "trace_origin": "dense_rerun",
            "row_id": "capture_available-002",
            "reference_move": 2,
            "full_search_selected_move": 0,
            "trace_points": trace_points,
            "insufficiency_reasons": [],
            "trace_points_summary": default_trace_points_summary
            if trace_points_summary is None
            else trace_points_summary,
            "provenance_guard": {
                "passed": provenance_passed,
                "failures": []
                if provenance_passed
                else ["selected_artifact_provenance_mismatch"],
            },
        }

    def generated_selection_score_artifact(self) -> dict:
        def trace_point(
            *,
            simulation: float,
            selected_move: int,
            visits: list[float],
            moves: list[dict],
        ) -> dict:
            return {
                "simulation": simulation,
                "selected_move": selected_move,
                "reference_move_by_prior": 2,
                "visits": list(visits),
                "moves": moves,
            }

        return selection_score_module.build_payload(
            {
                "artifact_path": "/tmp/source-artifacts/shared_drift.json",
                "schema": selection_score_module.SOURCE_SHARED_DRIFT_SCHEMA,
                "decision": selection_score_module.EXPECTED_SOURCE_DECISION,
                "classification": {"classification": "shared_mechanism_disproved"},
                "row": {
                    "row_id": "capture_available-002",
                    "reference_move": 2,
                    "full_search_selected_move": 0,
                    "legal_moves": [0, 1, 2, 3, 4],
                    "root_start": trace_point(
                        simulation=1.0,
                        selected_move=2,
                        visits=[0.0, 0.0, 1.0, 0.0, 0.0],
                        moves=[
                            {"move": 0, "selection_score": 0.40, "q_value": 0.01},
                            {"move": 2, "selection_score": 0.44, "q_value": 0.00},
                        ],
                    ),
                    "snapshots": [
                        trace_point(
                            simulation=8.0,
                            selected_move=0,
                            visits=[7.0, 0.0, 5.0, 0.0, 0.0],
                            moves=[
                                {"move": 0, "selection_score": 0.48, "q_value": 0.01},
                                {"move": 2, "selection_score": 0.46, "q_value": 0.00},
                            ],
                        ),
                        trace_point(
                            simulation=16.0,
                            selected_move=0,
                            visits=[11.0, 0.0, 5.0, 0.0, 0.0],
                            moves=[
                                {"move": 0, "selection_score": 0.56, "q_value": -0.01},
                                {"move": 2, "selection_score": 0.45, "q_value": -0.03},
                            ],
                        ),
                        trace_point(
                            simulation=32.0,
                            selected_move=0,
                            visits=[23.0, 0.0, 9.0, 0.0, 0.0],
                            moves=[
                                {"move": 0, "selection_score": 0.59, "q_value": 0.04},
                                {"move": 2, "selection_score": 0.43, "q_value": 0.00},
                            ],
                        ),
                    ],
                },
            }
        )

    def test_build_payload_classifies_sparse_trace_when_only_root_and_final_checkpoints_exist(
        self,
    ):
        payload = module.build_payload(
            self.trace_capture_artifact(simulations=[1.0, 1.0, 16.0]),
            self.selection_score_artifact(visit_share=0.04545454545454547),
            trace_capture_artifact_path="/tmp/trace_capture.json",
            selection_score_artifact_path="/tmp/selection_score.json",
        )

        self.assertEqual(
            "trace_too_sparse", payload["classification"]["classification"]
        )
        self.assertEqual("write_002_trace_cadence_capture_spec", payload["decision"])
        self.assertEqual([1.0, 16.0], payload["unique_simulation_checkpoints"])
        self.assertEqual(2, payload["unique_simulation_checkpoint_count"])
        self.assertIn(
            "selected_move_changed_without_captured_crossing",
            payload["ambiguity_signals"],
        )
        self.assertIn(
            "near_material_visit_share_threshold", payload["ambiguity_signals"]
        )

    def test_build_payload_marks_cadence_adequate_when_intermediate_checkpoints_exist(
        self,
    ):
        payload = module.build_payload(
            self.trace_capture_artifact(simulations=[1.0, 8.0, 12.0, 16.0]),
            self.selection_score_artifact(visit_share=0.02),
            trace_capture_artifact_path="/tmp/trace_capture.json",
            selection_score_artifact_path="/tmp/selection_score.json",
        )

        self.assertEqual(
            "cadence_adequate", payload["classification"]["classification"]
        )
        self.assertEqual("continue_002_threshold_too_strict_check", payload["decision"])
        self.assertEqual(
            [1.0, 8.0, 12.0, 16.0], payload["unique_simulation_checkpoints"]
        )
        self.assertEqual(4, payload["unique_simulation_checkpoint_count"])

    def test_load_selection_score_artifact_requires_finite_visit_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection_score.json"
            artifact = self.selection_score_artifact(visit_share=0.02)
            del artifact["final_selected_minus_reference_visit_share"]
            self.write_json(path, artifact)

            with self.assertRaisesRegex(
                ValueError,
                r"selection score artifact final_selected_minus_reference_visit_share must be a finite number",
            ):
                module.load_selection_score_artifact(path)

    def test_load_selection_score_artifact_requires_source_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection_score.json"
            artifact = self.selection_score_artifact(visit_share=0.02)
            del artifact["source_artifact"]
            self.write_json(path, artifact)

            with self.assertRaisesRegex(
                ValueError,
                r"selection score artifact source_artifact must be an object",
            ):
                module.load_selection_score_artifact(path)

    def test_load_selection_score_artifact_requires_source_artifact_identity_fields(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection_score.json"
            artifact = self.selection_score_artifact(visit_share=0.02)
            del artifact["source_artifact"]["row_id"]
            self.write_json(path, artifact)

            with self.assertRaisesRegex(
                ValueError,
                r"selection score artifact source_artifact.row_id is required",
            ):
                module.load_selection_score_artifact(path)

    def test_load_selection_score_artifact_rejects_null_source_identity_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection_score.json"
            artifact = self.selection_score_artifact(visit_share=0.02)
            artifact["source_artifact"]["reference_move"] = None
            self.write_json(path, artifact)

            with self.assertRaisesRegex(
                ValueError,
                r"selection score artifact source_artifact.reference_move must be an integer",
            ):
                module.load_selection_score_artifact(path)

    def test_load_selection_score_artifact_rejects_non_002_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection_score.json"
            artifact = self.selection_score_artifact(visit_share=0.02)
            artifact["source_artifact"]["row_id"] = "capture_available-003"
            self.write_json(path, artifact)

            with self.assertRaisesRegex(
                ValueError,
                r"selection score artifact source_artifact.row_id must be capture_available-002",
            ):
                module.load_selection_score_artifact(path)

    def test_load_selection_score_artifact_accepts_generated_artifact_shape_with_move_identity_fields(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection_score.json"
            self.write_json(path, self.generated_selection_score_artifact())

            artifact = module.load_selection_score_artifact(path)

        self.assertEqual("capture_available-002", artifact["source_artifact"]["row_id"])
        self.assertEqual(2, artifact["source_artifact"]["reference_move"])
        self.assertEqual(0, artifact["source_artifact"]["full_search_selected_move"])
        payload = module.build_payload(
            self.trace_capture_artifact(simulations=[1.0, 8.0, 12.0, 16.0]),
            artifact,
            trace_capture_artifact_path="/tmp/trace_capture.json",
            selection_score_artifact_path="/tmp/selection_score.json",
        )
        self.assertEqual(
            "cadence_adequate", payload["classification"]["classification"]
        )

    def test_load_selection_score_artifact_rejects_boolean_visit_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection_score.json"
            artifact = self.selection_score_artifact(visit_share=0.02)
            artifact["final_selected_minus_reference_visit_share"] = True
            self.write_json(path, artifact)

            with self.assertRaisesRegex(
                ValueError,
                r"selection score artifact final_selected_minus_reference_visit_share must be a finite number",
            ):
                module.load_selection_score_artifact(path)

    def test_load_trace_capture_artifact_rejects_non_finite_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace_capture.json"
            artifact = self.trace_capture_artifact(simulations=[1.0, 8.0])
            artifact["trace_points"][1]["simulation"] = "nan"
            self.write_json(path, artifact)

            with self.assertRaisesRegex(
                ValueError,
                r"trace capture artifact trace_points\[1\] simulation must be a finite number",
            ):
                module.load_trace_capture_artifact(path)

    def test_load_trace_capture_artifact_rejects_boolean_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace_capture.json"
            artifact = self.trace_capture_artifact(simulations=[1.0, 8.0])
            artifact["trace_points"][1]["simulation"] = False
            self.write_json(path, artifact)

            with self.assertRaisesRegex(
                ValueError,
                r"trace capture artifact trace_points\[1\] simulation must be a finite number",
            ):
                module.load_trace_capture_artifact(path)

    def test_load_trace_capture_artifact_rejects_dense_cadence_capture_with_failed_provenance_guard(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace_capture.json"
            artifact = self.dense_cadence_capture_artifact(provenance_passed=False)
            self.write_json(path, artifact)

            with self.assertRaisesRegex(ValueError, r"provenance_guard"):
                module.load_trace_capture_artifact(path)

    def test_load_trace_capture_artifact_accepts_valid_dense_cadence_capture_schema(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace_capture.json"
            artifact = self.dense_cadence_capture_artifact()
            self.write_json(path, artifact)

            loaded = module.load_trace_capture_artifact(path)

        self.assertEqual(
            "azlite_capture_002_trace_cadence_capture_v1", loaded["schema"]
        )
        self.assertEqual("dense_rerun", loaded["trace_origin"])
        self.assertEqual(5, len(loaded["trace_points"]))

    def test_load_trace_capture_artifact_rejects_forged_dense_summary_without_actual_intermediate_checkpoint(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace_capture.json"
            artifact = self.dense_cadence_capture_artifact(
                trace_points=[
                    {
                        "simulation": 1.0,
                        "selected_move": 2,
                        "reference_move_by_prior": 2,
                        "visits": [0.0, 0.0, 1.0, 0.0, 0.0],
                        "moves": [],
                    },
                    {
                        "simulation": 4.0,
                        "selected_move": 2,
                        "reference_move_by_prior": 2,
                        "visits": [2.0, 0.0, 3.0, 0.0, 0.0],
                        "moves": [],
                    },
                    {
                        "simulation": 8.0,
                        "selected_move": 2,
                        "reference_move_by_prior": 2,
                        "visits": [4.0, 0.0, 5.0, 0.0, 0.0],
                        "moves": [],
                    },
                    {
                        "simulation": 16.0,
                        "selected_move": 0,
                        "reference_move_by_prior": 2,
                        "visits": [8.0, 4.0, 7.0, 2.0, 1.0],
                        "moves": [],
                    },
                ],
                trace_points_summary={
                    "unique_simulation_checkpoints": [1.0, 4.0, 8.0, 16.0],
                    "unique_simulation_checkpoint_count": 4,
                    "duplicate_root_snapshot_count": 0,
                    "first_divergent_selected_move_index": 3,
                    "has_additional_checkpoint_between_divergence_and_final": True,
                },
            )
            self.write_json(path, artifact)

            with self.assertRaisesRegex(
                ValueError,
                r"has_additional_checkpoint_between_divergence_and_final must match trace_points",
            ):
                module.load_trace_capture_artifact(path)

    def test_main_writes_sorted_payload_and_prints_compact_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_capture_path = Path(tmp) / "trace_capture.json"
            selection_score_path = Path(tmp) / "selection_score.json"
            out_path = Path(tmp) / "trace_cadence_review.json"
            self.write_json(
                trace_capture_path,
                self.trace_capture_artifact(simulations=[1.0, 1.0, 16.0]),
            )
            self.write_json(
                selection_score_path,
                self.selection_score_artifact(visit_share=0.04545454545454547),
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = module.main(
                    [
                        "--source-trace-capture-artifact",
                        str(trace_capture_path),
                        "--source-selection-score-artifact",
                        str(selection_score_path),
                        "--out",
                        str(out_path),
                    ]
                )

            written_text = out_path.read_text(encoding="utf-8")
            written = json.loads(written_text)
            printed_text = stdout.getvalue()
            printed = json.loads(printed_text)

        self.assertEqual(0, exit_code)
        self.assertEqual(module.SCHEMA, written["schema"])
        expected_printed = {
            "artifact_path": str(out_path),
            "schema": module.SCHEMA,
            "decision": "write_002_trace_cadence_capture_spec",
        }
        self.assertEqual(expected_printed, printed)
        self.assertEqual(
            json.dumps(written, indent=2, sort_keys=True) + "\n", written_text
        )
        self.assertEqual(json.dumps(expected_printed) + "\n", printed_text)
