import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ml.alphazero_lite import capture_002_selection_score_trace as selection_score_module
from ml.alphazero_lite import capture_002_trace_cadence_review as trace_cadence_module
from ml.alphazero_lite import capture_002_nonseparable_review as module


class Capture002NonseparableReviewContractTest(unittest.TestCase):
    def test_contract_constants_are_stable(self):
        self.assertEqual("azlite_capture_002_nonseparable_review_v1", module.SCHEMA)
        self.assertEqual("azlite_capture_002_selection_score_trace_v1", module.SOURCE_SELECTION_SCORE_SCHEMA)
        self.assertEqual(
            "azlite_capture_002_trace_cadence_review_v1",
            module.SOURCE_TRACE_CADENCE_REVIEW_SCHEMA,
        )
        self.assertEqual(
            {
                "genuinely_not_separable": "stop_002_unresolved",
                "prerequisite_preempted": None,
            },
            module.CLASSIFICATION_DECISIONS,
        )

    def test_parse_args_reads_required_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection_score_path = Path(tmp) / "capture_002_selection_score_trace.json"
            cadence_review_path = Path(tmp) / "capture_002_trace_cadence_review.json"
            threshold_review_path = Path(tmp) / "capture_002_threshold_review.json"
            out_path = Path(tmp) / "capture_002_nonseparable_review.json"

            args = module.parse_args(
                [
                    "--source-selection-score-artifact",
                    str(selection_score_path),
                    "--source-trace-cadence-review-artifact",
                    str(cadence_review_path),
                    "--source-threshold-review-artifact",
                    str(threshold_review_path),
                    "--out",
                    str(out_path),
                ]
            )

        self.assertEqual(selection_score_path, args.source_selection_score_artifact)
        self.assertEqual(cadence_review_path, args.source_trace_cadence_review_artifact)
        self.assertEqual(threshold_review_path, args.source_threshold_review_artifact)
        self.assertEqual(out_path, args.out)


class Capture002NonseparableReviewBuildPayloadTest(unittest.TestCase):
    def write_json(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def default_selection_score_artifact(self) -> dict:
        return {
            "schema": module.SOURCE_SELECTION_SCORE_SCHEMA,
            "classification": {
                "classification": "unresolved",
                "evidence_summary": "Selection-score and Q-support timing do not cleanly separate for capture 002.",
            },
            "decision": "write_002_unresolved_trace_review_spec",
            "insufficiency_reasons": [],
            "source_artifact": {
                "row_id": "capture_available-002",
                "reference_move": 2,
                "full_search_selected_move": 0,
                "selected_artifact": {
                    "path": "/tmp/source/selected",
                    "selected_artifact": "/tmp/source/selected_artifact",
                    "selected_target": "/tmp/source/selected",
                    "provenance_source": "selection_manifest.selected_target",
                },
            },
            "thresholds": {
                "meaningful_q_margin": 0.03,
                "material_selection_score_margin": 0.05,
                "material_visit_share_margin": 0.05,
            },
            "final_selected_minus_reference_q": -0.02,
            "final_selected_minus_reference_selection_score": -0.04,
            "final_selected_minus_reference_visit_share": 0.04545454545454547,
        }

    def generated_default_selection_score_artifact(self) -> dict:
        def trace_point(*, simulation: float, selected_move: int, visits: list[float], moves: list[dict]) -> dict:
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
                "selected_artifact": {
                    "path": "/tmp/source/selected",
                    "selected_artifact": "/tmp/source/selected_artifact",
                    "selected_target": "/tmp/source/selected",
                    "provenance_source": "selection_manifest.selected_target",
                },
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

    def generated_trace_cadence_review_artifact(self, selection_score_artifact: dict) -> dict:
        return trace_cadence_module.build_payload(
            {
                "schema": trace_cadence_module.SOURCE_TRACE_CAPTURE_SCHEMA,
                "trace_origin": "extracted",
                "row_id": "capture_available-002",
                "reference_move": 2,
                "full_search_selected_move": 0,
                "trace_points": [
                    {
                        "simulation": 1.0,
                        "selected_move": 2,
                        "reference_move_by_prior": 2,
                        "visits": [0.0, 0.0, 1.0, 0.0, 0.0],
                        "moves": [
                            {"move": 0, "selection_score": 0.10, "q_value": 0.05},
                            {"move": 2, "selection_score": 0.14, "q_value": 0.07},
                        ],
                    },
                    {
                        "simulation": 8.0,
                        "selected_move": 2,
                        "reference_move_by_prior": 2,
                        "visits": [4.0, 0.0, 5.0, 0.0, 0.0],
                        "moves": [
                            {"move": 0, "selection_score": 0.10, "q_value": 0.05},
                            {"move": 2, "selection_score": 0.14, "q_value": 0.07},
                        ],
                    },
                    {
                        "simulation": 12.0,
                        "selected_move": 0,
                        "reference_move_by_prior": 2,
                        "visits": [7.0, 0.0, 5.0, 0.0, 0.0],
                        "moves": [
                            {"move": 0, "selection_score": 0.10, "q_value": 0.05},
                            {"move": 2, "selection_score": 0.14, "q_value": 0.07},
                        ],
                    },
                    {
                        "simulation": 16.0,
                        "selected_move": 0,
                        "reference_move_by_prior": 2,
                        "visits": [8.0, 4.0, 7.0, 2.0, 1.0],
                        "moves": [
                            {"move": 0, "selection_score": 0.10, "q_value": 0.05},
                            {"move": 2, "selection_score": 0.14, "q_value": 0.07},
                        ],
                    },
                ],
                "insufficiency_reasons": [],
            },
            selection_score_artifact,
            trace_capture_artifact_path="/tmp/trace_capture.json",
            selection_score_artifact_path="/tmp/default_selection.json",
        )

    def cadence_review_artifact(self, *, decision: str) -> dict:
        classification = "cadence_adequate" if decision == "continue_002_threshold_too_strict_check" else "trace_too_sparse"
        return {
            "schema": module.SOURCE_TRACE_CADENCE_REVIEW_SCHEMA,
            "classification": {"classification": classification, "evidence_summary": "cadence review"},
            "decision": decision,
            "trace_capture_excerpt": {
                "row_id": "capture_available-002",
                "reference_move": 2,
                "full_search_selected_move": 0,
            },
            "selection_score_excerpt": {
                "final_selected_minus_reference_visit_share": 0.04545454545454547,
                "first_selected_material_visit_share_snapshot": None,
                "first_selected_meaningful_q_support_snapshot": None,
                "first_selected_selection_score_overtake_snapshot": None,
            },
        }

    def threshold_review_artifact(
        self,
        *,
        classification: str,
        decision: str,
        material_visit_share_margin: float = 0.04,
    ) -> dict:
        return {
            "schema": module.SOURCE_SELECTION_SCORE_SCHEMA,
            "classification": {"classification": classification, "evidence_summary": "threshold review"},
            "decision": decision,
            "insufficiency_reasons": [],
            "source_artifact": {
                "row_id": "capture_available-002",
                "reference_move": 2,
                "full_search_selected_move": 0,
                "selected_artifact": {
                    "path": "/tmp/source/selected",
                    "selected_artifact": "/tmp/source/selected_artifact",
                    "selected_target": "/tmp/source/selected",
                    "provenance_source": "selection_manifest.selected_target",
                },
            },
            "thresholds": {
                "meaningful_q_margin": 0.03,
                "material_selection_score_margin": 0.05,
                "material_visit_share_margin": material_visit_share_margin,
            },
            "final_selected_minus_reference_q": -0.02,
            "final_selected_minus_reference_selection_score": -0.04,
            "final_selected_minus_reference_visit_share": 0.04545454545454547,
        }

    def test_build_payload_supports_genuinely_not_separable_when_prerequisites_hold_and_relaxation_still_unresolved(self):
        payload = module.build_payload(
            self.default_selection_score_artifact(),
            self.cadence_review_artifact(decision="continue_002_threshold_too_strict_check"),
            self.threshold_review_artifact(
                classification="unresolved",
                decision="write_002_unresolved_trace_review_spec",
            ),
            source_selection_score_artifact_path="/tmp/default_selection.json",
            source_trace_cadence_review_artifact_path="/tmp/cadence_review.json",
            source_threshold_review_artifact_path="/tmp/threshold_review.json",
        )

        self.assertEqual("genuinely_not_separable", payload["classification"]["classification"])
        self.assertEqual("stop_002_unresolved", payload["decision"])
        self.assertEqual(-0.02, payload["final_margin_summary"]["default_q_margin"])
        self.assertEqual(0.04, payload["thresholds_evaluated"]["relaxed_material_visit_share_margin"])

    def test_build_payload_preempts_when_cadence_review_already_supports_sparse_branch(self):
        payload = module.build_payload(
            self.default_selection_score_artifact(),
            self.cadence_review_artifact(decision="write_002_trace_cadence_capture_spec"),
            self.threshold_review_artifact(
                classification="unresolved",
                decision="write_002_unresolved_trace_review_spec",
            ),
            source_selection_score_artifact_path="/tmp/default_selection.json",
            source_trace_cadence_review_artifact_path="/tmp/cadence_review.json",
            source_threshold_review_artifact_path="/tmp/threshold_review.json",
        )

        self.assertEqual("prerequisite_preempted", payload["classification"]["classification"])
        self.assertEqual("genuinely_not_separable", payload["hypothesis"])
        self.assertEqual("write_002_trace_cadence_capture_spec", payload["decision"])

    def test_build_payload_preempts_when_threshold_review_already_supports_selection_score_pressure(self):
        payload = module.build_payload(
            self.default_selection_score_artifact(),
            self.cadence_review_artifact(decision="continue_002_threshold_too_strict_check"),
            self.threshold_review_artifact(
                classification="selection_score_pressure_confirmed",
                decision="write_002_selection_pressure_ablation_spec",
            ),
            source_selection_score_artifact_path="/tmp/default_selection.json",
            source_trace_cadence_review_artifact_path="/tmp/cadence_review.json",
            source_threshold_review_artifact_path="/tmp/threshold_review.json",
        )

        self.assertEqual("prerequisite_preempted", payload["classification"]["classification"])
        self.assertEqual("genuinely_not_separable", payload["hypothesis"])
        self.assertEqual("write_002_selection_pressure_ablation_spec", payload["decision"])

    def test_build_payload_accepts_valid_relaxed_threshold_unresolved_artifact(self):
        payload = module.build_payload(
            self.default_selection_score_artifact(),
            self.cadence_review_artifact(decision="continue_002_threshold_too_strict_check"),
            self.threshold_review_artifact(
                classification="unresolved",
                decision="write_002_unresolved_trace_review_spec",
            ),
            source_selection_score_artifact_path="/tmp/default_selection.json",
            source_trace_cadence_review_artifact_path="/tmp/cadence_review.json",
            source_threshold_review_artifact_path="/tmp/threshold_review.json",
        )

        self.assertEqual("genuinely_not_separable", payload["classification"]["classification"])
        self.assertEqual("stop_002_unresolved", payload["decision"])

    def test_build_payload_accepts_generated_default_selection_artifact_shape_via_loaders(self):
        with tempfile.TemporaryDirectory() as tmp:
            default_path = Path(tmp) / "default_selection.json"
            cadence_path = Path(tmp) / "cadence_review.json"
            threshold_path = Path(tmp) / "threshold_review.json"

            default_artifact = self.generated_default_selection_score_artifact()
            cadence_artifact = self.generated_trace_cadence_review_artifact(default_artifact)
            threshold_artifact = self.threshold_review_artifact(
                classification="unresolved",
                decision="write_002_unresolved_trace_review_spec",
            )

            self.write_json(default_path, default_artifact)
            self.write_json(cadence_path, cadence_artifact)
            self.write_json(threshold_path, threshold_artifact)

            loaded_default_artifact = module.load_selection_score_artifact(default_path)
            loaded_cadence_artifact = module.load_trace_cadence_review_artifact(cadence_path)
            loaded_threshold_artifact = module.load_threshold_review_artifact(threshold_path)

        self.assertEqual(2, loaded_default_artifact["source_artifact"]["reference_move"])
        self.assertEqual(0, loaded_default_artifact["source_artifact"]["full_search_selected_move"])
        payload = module.build_payload(
            loaded_default_artifact,
            loaded_cadence_artifact,
            loaded_threshold_artifact,
            source_selection_score_artifact_path="/tmp/default_selection.json",
            source_trace_cadence_review_artifact_path="/tmp/cadence_review.json",
            source_threshold_review_artifact_path="/tmp/threshold_review.json",
        )
        self.assertEqual("genuinely_not_separable", payload["classification"]["classification"])

    def test_build_payload_rejects_contradictory_threshold_classification_and_decision(self):
        with self.assertRaisesRegex(ValueError, "threshold review artifact"):
            module.build_payload(
                self.default_selection_score_artifact(),
                self.cadence_review_artifact(decision="continue_002_threshold_too_strict_check"),
                self.threshold_review_artifact(
                    classification="selection_score_pressure_confirmed",
                    decision="write_002_unresolved_trace_review_spec",
                ),
                source_selection_score_artifact_path="/tmp/default_selection.json",
                source_trace_cadence_review_artifact_path="/tmp/cadence_review.json",
                source_threshold_review_artifact_path="/tmp/threshold_review.json",
            )

    def test_build_payload_rejects_default_selection_score_artifact_that_already_supports_another_branch(self):
        default_artifact = self.default_selection_score_artifact()
        default_artifact["classification"] = {
            "classification": "selection_score_pressure_confirmed",
            "evidence_summary": "Selection-score pressure appears before meaningful child-Q support.",
        }
        default_artifact["decision"] = "write_002_selection_pressure_ablation_spec"

        with self.assertRaisesRegex(ValueError, "default selection score artifact"):
            module.build_payload(
                default_artifact,
                self.cadence_review_artifact(decision="continue_002_threshold_too_strict_check"),
                self.threshold_review_artifact(
                    classification="unresolved",
                    decision="write_002_unresolved_trace_review_spec",
                ),
                source_selection_score_artifact_path="/tmp/default_selection.json",
                source_trace_cadence_review_artifact_path="/tmp/cadence_review.json",
                source_threshold_review_artifact_path="/tmp/threshold_review.json",
            )

    def test_build_payload_rejects_relaxed_threshold_artifact_as_default_baseline_input(self):
        with self.assertRaisesRegex(ValueError, "default selection score artifact"):
            module.build_payload(
                self.threshold_review_artifact(
                    classification="unresolved",
                    decision="write_002_unresolved_trace_review_spec",
                ),
                self.cadence_review_artifact(decision="continue_002_threshold_too_strict_check"),
                self.threshold_review_artifact(
                    classification="unresolved",
                    decision="write_002_unresolved_trace_review_spec",
                ),
                source_selection_score_artifact_path="/tmp/default_selection.json",
                source_trace_cadence_review_artifact_path="/tmp/cadence_review.json",
                source_threshold_review_artifact_path="/tmp/threshold_review.json",
            )

    def test_build_payload_rejects_malformed_cadence_artifact_for_direct_callers(self):
        with self.assertRaisesRegex(ValueError, "trace cadence review artifact"):
            module.build_payload(
                self.default_selection_score_artifact(),
                {
                    "schema": module.SOURCE_TRACE_CADENCE_REVIEW_SCHEMA,
                    "classification": {
                        "classification": "cadence_adequate",
                        "evidence_summary": "cadence review",
                    },
                },
                self.threshold_review_artifact(
                    classification="unresolved",
                    decision="write_002_unresolved_trace_review_spec",
                ),
                source_selection_score_artifact_path="/tmp/default_selection.json",
                source_trace_cadence_review_artifact_path="/tmp/cadence_review.json",
                source_threshold_review_artifact_path="/tmp/threshold_review.json",
            )

    def test_build_payload_rejects_non_baseline_default_artifact_even_on_cadence_preempt_path(self):
        default_artifact = self.default_selection_score_artifact()
        default_artifact["classification"] = {
            "classification": "selection_score_pressure_confirmed",
            "evidence_summary": "Selection-score pressure appears before meaningful child-Q support.",
        }
        default_artifact["decision"] = "write_002_selection_pressure_ablation_spec"

        with self.assertRaisesRegex(ValueError, "default selection score artifact"):
            module.build_payload(
                default_artifact,
                self.cadence_review_artifact(decision="write_002_trace_cadence_capture_spec"),
                self.threshold_review_artifact(
                    classification="unresolved",
                    decision="write_002_unresolved_trace_review_spec",
                ),
                source_selection_score_artifact_path="/tmp/default_selection.json",
                source_trace_cadence_review_artifact_path="/tmp/cadence_review.json",
                source_threshold_review_artifact_path="/tmp/threshold_review.json",
            )

    def test_build_payload_rejects_wrong_schemas_for_direct_inputs(self):
        wrong_default = self.default_selection_score_artifact()
        wrong_default["schema"] = "wrong_schema"

        with self.assertRaisesRegex(ValueError, "selection score artifact"):
            module.build_payload(
                wrong_default,
                self.cadence_review_artifact(decision="continue_002_threshold_too_strict_check"),
                self.threshold_review_artifact(
                    classification="unresolved",
                    decision="write_002_unresolved_trace_review_spec",
                ),
                source_selection_score_artifact_path="/tmp/default_selection.json",
                source_trace_cadence_review_artifact_path="/tmp/cadence_review.json",
                source_threshold_review_artifact_path="/tmp/threshold_review.json",
            )

        wrong_threshold = self.threshold_review_artifact(
            classification="unresolved",
            decision="write_002_unresolved_trace_review_spec",
        )
        wrong_threshold["schema"] = "wrong_schema"

        with self.assertRaisesRegex(ValueError, "selection score artifact"):
            module.build_payload(
                self.default_selection_score_artifact(),
                self.cadence_review_artifact(decision="continue_002_threshold_too_strict_check"),
                wrong_threshold,
                source_selection_score_artifact_path="/tmp/default_selection.json",
                source_trace_cadence_review_artifact_path="/tmp/cadence_review.json",
                source_threshold_review_artifact_path="/tmp/threshold_review.json",
            )

        wrong_cadence = self.cadence_review_artifact(
            decision="continue_002_threshold_too_strict_check"
        )
        wrong_cadence["schema"] = "wrong_schema"

        with self.assertRaisesRegex(ValueError, "trace cadence review artifact"):
            module.build_payload(
                self.default_selection_score_artifact(),
                wrong_cadence,
                self.threshold_review_artifact(
                    classification="unresolved",
                    decision="write_002_unresolved_trace_review_spec",
                ),
                source_selection_score_artifact_path="/tmp/default_selection.json",
                source_trace_cadence_review_artifact_path="/tmp/cadence_review.json",
                source_threshold_review_artifact_path="/tmp/threshold_review.json",
            )

    def test_build_payload_rejects_non_empty_insufficiency_reasons_on_selection_score_inputs(self):
        bad_default = self.default_selection_score_artifact()
        bad_default["insufficiency_reasons"] = ["trace_missing"]

        with self.assertRaisesRegex(ValueError, "selection score artifact"):
            module.build_payload(
                bad_default,
                self.cadence_review_artifact(decision="continue_002_threshold_too_strict_check"),
                self.threshold_review_artifact(
                    classification="unresolved",
                    decision="write_002_unresolved_trace_review_spec",
                ),
                source_selection_score_artifact_path="/tmp/default_selection.json",
                source_trace_cadence_review_artifact_path="/tmp/cadence_review.json",
                source_threshold_review_artifact_path="/tmp/threshold_review.json",
            )

    def test_build_payload_rejects_mismatched_default_and_threshold_source_identity(self):
        threshold_artifact = self.threshold_review_artifact(
            classification="unresolved",
            decision="write_002_unresolved_trace_review_spec",
        )
        threshold_artifact["source_artifact"]["full_search_selected_move"] = 1

        with self.assertRaisesRegex(ValueError, "selection score source identities must match"):
            module.build_payload(
                self.default_selection_score_artifact(),
                self.cadence_review_artifact(decision="continue_002_threshold_too_strict_check"),
                threshold_artifact,
                source_selection_score_artifact_path="/tmp/default_selection.json",
                source_trace_cadence_review_artifact_path="/tmp/cadence_review.json",
                source_threshold_review_artifact_path="/tmp/threshold_review.json",
            )

    def test_build_payload_rejects_mismatched_cadence_trace_excerpt(self):
        cadence_artifact = self.cadence_review_artifact(decision="continue_002_threshold_too_strict_check")
        cadence_artifact["trace_capture_excerpt"]["row_id"] = "capture_available-003"

        with self.assertRaisesRegex(ValueError, "trace cadence review artifact trace_capture_excerpt"):
            module.build_payload(
                self.default_selection_score_artifact(),
                cadence_artifact,
                self.threshold_review_artifact(
                    classification="unresolved",
                    decision="write_002_unresolved_trace_review_spec",
                ),
                source_selection_score_artifact_path="/tmp/default_selection.json",
                source_trace_cadence_review_artifact_path="/tmp/cadence_review.json",
                source_threshold_review_artifact_path="/tmp/threshold_review.json",
            )

    def test_build_payload_rejects_mismatched_cadence_selection_score_excerpt(self):
        cadence_artifact = self.cadence_review_artifact(decision="continue_002_threshold_too_strict_check")
        cadence_artifact["selection_score_excerpt"]["final_selected_minus_reference_visit_share"] = 0.02

        with self.assertRaisesRegex(ValueError, "trace cadence review artifact selection_score_excerpt"):
            module.build_payload(
                self.default_selection_score_artifact(),
                cadence_artifact,
                self.threshold_review_artifact(
                    classification="unresolved",
                    decision="write_002_unresolved_trace_review_spec",
                ),
                source_selection_score_artifact_path="/tmp/default_selection.json",
                source_trace_cadence_review_artifact_path="/tmp/cadence_review.json",
                source_threshold_review_artifact_path="/tmp/threshold_review.json",
            )

        bad_threshold = self.threshold_review_artifact(
            classification="unresolved",
            decision="write_002_unresolved_trace_review_spec",
        )
        bad_threshold["insufficiency_reasons"] = ["trace_missing"]

        with self.assertRaisesRegex(ValueError, "selection score artifact"):
            module.build_payload(
                self.default_selection_score_artifact(),
                self.cadence_review_artifact(decision="continue_002_threshold_too_strict_check"),
                bad_threshold,
                source_selection_score_artifact_path="/tmp/default_selection.json",
                source_trace_cadence_review_artifact_path="/tmp/cadence_review.json",
                source_threshold_review_artifact_path="/tmp/threshold_review.json",
            )

    def test_build_payload_preempts_when_threshold_review_supports_q_support_branch(self):
        payload = module.build_payload(
            self.default_selection_score_artifact(),
            self.cadence_review_artifact(decision="continue_002_threshold_too_strict_check"),
            self.threshold_review_artifact(
                classification="q_support_precedes_selection_score",
                decision="write_002_child_value_audit_spec",
            ),
            source_selection_score_artifact_path="/tmp/default_selection.json",
            source_trace_cadence_review_artifact_path="/tmp/cadence_review.json",
            source_threshold_review_artifact_path="/tmp/threshold_review.json",
        )

        self.assertEqual("prerequisite_preempted", payload["classification"]["classification"])
        self.assertEqual("write_002_child_value_audit_spec", payload["decision"])

    def test_main_writes_sorted_payload_and_prints_compact_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            default_path = Path(tmp) / "default_selection.json"
            cadence_path = Path(tmp) / "cadence_review.json"
            threshold_path = Path(tmp) / "threshold_review.json"
            out_path = Path(tmp) / "nonseparable_review.json"
            self.write_json(default_path, self.default_selection_score_artifact())
            self.write_json(cadence_path, self.cadence_review_artifact(decision="continue_002_threshold_too_strict_check"))
            self.write_json(
                threshold_path,
                self.threshold_review_artifact(
                    classification="unresolved",
                    decision="write_002_unresolved_trace_review_spec",
                ),
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = module.main(
                    [
                        "--source-selection-score-artifact",
                        str(default_path),
                        "--source-trace-cadence-review-artifact",
                        str(cadence_path),
                        "--source-threshold-review-artifact",
                        str(threshold_path),
                        "--out",
                        str(out_path),
                    ]
                )

            written = json.loads(out_path.read_text(encoding="utf-8"))
            printed = json.loads(stdout.getvalue())

        self.assertEqual(0, exit_code)
        self.assertEqual(module.SCHEMA, written["schema"])
        self.assertEqual(
            {
                "artifact_path": str(out_path),
                "schema": module.SCHEMA,
                "decision": "stop_002_unresolved",
            },
            printed,
        )


class Capture002NonseparableReviewThresholdArtifactValidationTest(unittest.TestCase):
    def test_load_threshold_review_artifact_rejects_wrong_relaxed_threshold_state(self):
        artifact = {
            "schema": module.SOURCE_SELECTION_SCORE_SCHEMA,
            "classification": {
                "classification": "unresolved",
                "evidence_summary": "Selection-score and Q-support timing do not cleanly separate for capture 002.",
            },
            "decision": "write_002_unresolved_trace_review_spec",
            "insufficiency_reasons": [],
            "thresholds": {
                "meaningful_q_margin": 0.03,
                "material_selection_score_margin": 0.05,
                "material_visit_share_margin": 0.05,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "threshold_review.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "threshold review artifact"):
                module.load_threshold_review_artifact(path)

    def test_load_threshold_review_artifact_rejects_supported_mechanism_at_default_thresholds(self):
        artifact = {
            "schema": module.SOURCE_SELECTION_SCORE_SCHEMA,
            "classification": {
                "classification": "selection_score_pressure_confirmed",
                "evidence_summary": "Selection-score pressure appears before meaningful child-Q support.",
            },
            "decision": "write_002_selection_pressure_ablation_spec",
            "insufficiency_reasons": [],
            "thresholds": {
                "meaningful_q_margin": 0.03,
                "material_selection_score_margin": 0.05,
                "material_visit_share_margin": 0.05,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "threshold_review.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "threshold review artifact"):
                module.load_threshold_review_artifact(path)


class Capture002NonseparableReviewCadenceArtifactValidationTest(unittest.TestCase):
    def test_load_trace_cadence_review_artifact_rejects_malformed_classification_decision_pair(self):
        artifact = {
            "schema": module.SOURCE_TRACE_CADENCE_REVIEW_SCHEMA,
            "classification": {
                "classification": "cadence_adequate",
                "evidence_summary": "cadence review",
            },
            "decision": "write_002_trace_cadence_capture_spec",
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cadence_review.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "trace cadence review artifact"):
                module.load_trace_cadence_review_artifact(path)

    def test_load_trace_cadence_review_artifact_rejects_missing_classification_or_decision(self):
        artifacts = [
            {
                "schema": module.SOURCE_TRACE_CADENCE_REVIEW_SCHEMA,
                "classification": {"evidence_summary": "cadence review"},
                "decision": "continue_002_threshold_too_strict_check",
            },
            {
                "schema": module.SOURCE_TRACE_CADENCE_REVIEW_SCHEMA,
                "classification": {
                    "classification": "cadence_adequate",
                    "evidence_summary": "cadence review",
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            for index, artifact in enumerate(artifacts):
                path = Path(tmp) / f"cadence_review_{index}.json"
                path.write_text(json.dumps(artifact), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "trace cadence review artifact"):
                    module.load_trace_cadence_review_artifact(path)
