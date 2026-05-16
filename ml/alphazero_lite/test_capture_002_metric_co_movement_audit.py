import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ml.alphazero_lite import capture_002_metric_co_movement_audit as module


class Capture002MetricCoMovementAuditContractTest(unittest.TestCase):
    def test_contract_constants_are_stable(self):
        self.assertEqual("azlite_capture_002_metric_co_movement_audit_v1", module.SCHEMA)
        self.assertEqual("azlite_capture_002_nonseparable_decomposition_v1", module.SOURCE_DECOMPOSITION_SCHEMA)
        self.assertEqual("azlite_capture_002_selection_score_trace_v1", module.SOURCE_SELECTION_SCORE_SCHEMA)
        self.assertEqual("capture_available-002", module.ROW_ID)
        self.assertEqual("metric_co_movement", module.EXPECTED_DECOMPOSITION_CLASSIFICATION)
        self.assertEqual("stop_002_mechanism_not_isolated", module.EXPECTED_DECOMPOSITION_DECISION)
        self.assertEqual("unresolved", module.EXPECTED_TRACE_CLASSIFICATION)
        self.assertEqual("write_002_unresolved_trace_review_spec", module.EXPECTED_TRACE_DECISION)
        self.assertEqual(1e-12, module.FLOAT_TOLERANCE)
        self.assertEqual(
            {
                "q": "meaningful_q_margin",
                "selection_score": "material_selection_score_margin",
                "visit_share": "material_visit_share_margin",
            },
            module.METRIC_THRESHOLDS,
        )
        self.assertEqual(
            {
                "weak_aligned_drift": "write_002_low_confidence_policy_value_interaction_spec",
                "early_selection_score_only": "write_002_selection_score_component_audit_spec",
                "late_visit_share_only": "write_002_visit_accumulation_audit_spec",
                "mixed_low_confidence_signal": "write_002_low_confidence_trace_comparison_spec",
                "metric_audit_inconclusive": "stop_002_metric_audit_inconclusive",
            },
            module.CLASSIFICATION_DECISIONS,
        )

    def test_parse_args_reads_required_paths(self):
        args = module.parse_args(
            [
                "--source-decomposition-artifact",
                "/tmp/decomposition.json",
                "--source-selection-score-artifact",
                "/tmp/default.json",
                "--source-threshold-relaxed-selection-score-artifact",
                "/tmp/relaxed.json",
                "--out",
                "/tmp/out.json",
            ]
        )

        self.assertEqual(Path("/tmp/decomposition.json"), args.source_decomposition_artifact)
        self.assertEqual(Path("/tmp/default.json"), args.source_selection_score_artifact)
        self.assertEqual(
            Path("/tmp/relaxed.json"),
            args.source_threshold_relaxed_selection_score_artifact,
        )
        self.assertEqual(Path("/tmp/out.json"), args.out)


class Capture002MetricCoMovementAuditBuildPayloadTest(unittest.TestCase):
    def selected_artifact(self) -> dict:
        return {
            "path": "/tmp/source/selected",
            "selected_artifact": "/tmp/source/selection/artifact",
            "selected_target": "/tmp/source/selected",
            "provenance_source": "selection_manifest.selected_target",
        }

    def source_artifact(self) -> dict:
        return {
            "row_id": "capture_available-002",
            "reference_move": 2,
            "full_search_selected_move": 0,
            "selected_artifact": self.selected_artifact(),
        }

    def move_entry(self, move: int, *, q_value: float | None, selection_score: float | None) -> dict:
        entry = {"move": move}
        if q_value is not None:
            entry["q_value"] = q_value
        if selection_score is not None:
            entry["selection_score"] = selection_score
        return entry

    def trace_point(
        self,
        *,
        simulation: float,
        selected_visits: float,
        reference_visits: float,
        selected_q: float | None,
        reference_q: float | None,
        selected_selection_score: float | None,
        reference_selection_score: float | None,
    ) -> dict:
        return {
            "simulation": simulation,
            "selected_move": 2,
            "reference_move_by_prior": 2,
            "visits": [selected_visits, 0.0, reference_visits, 0.0, 0.0],
            "moves": [
                self.move_entry(0, q_value=selected_q, selection_score=selected_selection_score),
                self.move_entry(2, q_value=reference_q, selection_score=reference_selection_score),
            ],
        }

    def trace_artifact(self, *, thresholds: dict, trace_points: list[dict], trace_origin: str = "extracted") -> dict:
        return {
            "schema": module.SOURCE_SELECTION_SCORE_SCHEMA,
            "classification": {"classification": "unresolved", "evidence_summary": "unresolved"},
            "decision": "write_002_unresolved_trace_review_spec",
            "insufficiency_reasons": [],
            "trace_origin": trace_origin,
            "source_artifact": self.source_artifact(),
            "thresholds": copy.deepcopy(thresholds),
            "trace_points": copy.deepcopy(trace_points),
        }

    def default_thresholds(self) -> dict:
        return {
            "meaningful_q_margin": 0.03,
            "material_selection_score_margin": 0.05,
            "material_visit_share_margin": 0.05,
        }

    def relaxed_thresholds(self) -> dict:
        return {
            "meaningful_q_margin": 0.03,
            "material_selection_score_margin": 0.05,
            "material_visit_share_margin": 0.04,
        }

    def weak_trace_points(self) -> list[dict]:
        return [
            self.trace_point(
                simulation=1.0,
                selected_visits=0.0,
                reference_visits=1.0,
                selected_q=0.0,
                reference_q=0.02,
                selected_selection_score=0.10,
                reference_selection_score=0.15,
            ),
            self.trace_point(
                simulation=2.0,
                selected_visits=2.0,
                reference_visits=3.0,
                selected_q=0.04,
                reference_q=0.03,
                selected_selection_score=0.17,
                reference_selection_score=0.16,
            ),
        ]

    def material_selection_score_trace(self) -> list[dict]:
        return [
            self.trace_point(
                simulation=1.0,
                selected_visits=1.0,
                reference_visits=1.0,
                selected_q=0.0,
                reference_q=0.0,
                selected_selection_score=0.22,
                reference_selection_score=0.16,
            ),
            self.trace_point(
                simulation=2.0,
                selected_visits=1.0,
                reference_visits=1.0,
                selected_q=0.01,
                reference_q=0.0,
                selected_selection_score=0.25,
                reference_selection_score=0.18,
            ),
        ]

    def material_visit_share_trace(self) -> list[dict]:
        return [
            self.trace_point(
                simulation=1.0,
                selected_visits=1.0,
                reference_visits=1.0,
                selected_q=0.0,
                reference_q=0.0,
                selected_selection_score=0.10,
                reference_selection_score=0.10,
            ),
            self.trace_point(
                simulation=2.0,
                selected_visits=3.0,
                reference_visits=2.0,
                selected_q=0.01,
                reference_q=0.0,
                selected_selection_score=0.11,
                reference_selection_score=0.10,
            ),
        ]

    def simultaneous_material_trace(self) -> list[dict]:
        return [
            self.trace_point(
                simulation=1.0,
                selected_visits=3.0,
                reference_visits=2.0,
                selected_q=0.05,
                reference_q=0.0,
                selected_selection_score=0.22,
                reference_selection_score=0.16,
            )
        ]

    def decomposition_artifact(
        self,
        *,
        classification: str = "metric_co_movement",
        decision: str = "stop_002_mechanism_not_isolated",
    ) -> dict:
        return {
            "schema": module.SOURCE_DECOMPOSITION_SCHEMA,
            "hypothesis": "genuinely_not_separable_decomposition",
            "classification": {"classification": classification, "evidence_summary": "metric co-movement"},
            "decision": decision,
            "input_artifacts": {
                "source_selection_score_artifact_path": "/tmp/default.json",
                "source_threshold_review_artifact_path": "/tmp/relaxed.json",
                "source_nonseparable_review_artifact_path": "/tmp/nonseparable.json",
            },
            "source_artifact": self.source_artifact(),
            "thresholds_evaluated": {
                "default": self.default_thresholds(),
                "relaxed": self.relaxed_thresholds(),
                "threshold_boundary_band": 0.01,
            },
            "source_snapshots": {
                "nonseparable_classification": {"classification": "genuinely_not_separable"},
                "default_classification": {"classification": "unresolved"},
                "threshold_classification": {"classification": "unresolved"},
            },
        }

    def valid_inputs(self) -> tuple[dict, dict, dict]:
        trace_points = self.weak_trace_points()
        return (
            self.decomposition_artifact(),
            self.trace_artifact(thresholds=self.default_thresholds(), trace_points=trace_points),
            self.trace_artifact(thresholds=self.relaxed_thresholds(), trace_points=trace_points),
        )

    def build_payload(self, decomposition=None, default=None, relaxed=None) -> dict:
        valid_decomposition, valid_default, valid_relaxed = self.valid_inputs()
        return module.build_payload(
            decomposition or valid_decomposition,
            default or valid_default,
            relaxed or valid_relaxed,
            source_decomposition_artifact_path="/tmp/decomposition.json",
            source_selection_score_artifact_path="/tmp/default.json",
            source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
        )

    def test_build_payload_rejects_wrong_decomposition_schema(self):
        decomposition, default, relaxed = self.valid_inputs()
        decomposition["schema"] = "wrong"

        with self.assertRaisesRegex(ValueError, "decomposition artifact has wrong schema"):
            self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

    def test_build_payload_rejects_non_metric_co_movement_decomposition(self):
        decomposition, default, relaxed = self.valid_inputs()
        decomposition["classification"]["classification"] = "signal_absent"

        with self.assertRaisesRegex(
            ValueError,
            "decomposition artifact classification must be metric_co_movement",
        ):
            self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

    def test_build_payload_rejects_wrong_decomposition_decision(self):
        decomposition, default, relaxed = self.valid_inputs()
        decomposition["decision"] = "stop_002_unresolved"

        with self.assertRaisesRegex(
            ValueError,
            "decomposition artifact decision must be stop_002_mechanism_not_isolated",
        ):
            self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

    def test_build_payload_rejects_wrong_trace_schema(self):
        decomposition, default, relaxed = self.valid_inputs()
        default["schema"] = "wrong"

        with self.assertRaisesRegex(ValueError, "selection score artifact has wrong schema"):
            self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

    def test_build_payload_rejects_missing_thresholds(self):
        decomposition, default, relaxed = self.valid_inputs()
        del relaxed["thresholds"]["material_visit_share_margin"]

        with self.assertRaisesRegex(ValueError, "thresholds must contain material_visit_share_margin"):
            self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

    def test_build_payload_rejects_negative_thresholds(self):
        decomposition, default, relaxed = self.valid_inputs()
        relaxed["thresholds"]["material_visit_share_margin"] = -0.01
        decomposition["thresholds_evaluated"]["relaxed"]["material_visit_share_margin"] = -0.01

        with self.assertRaisesRegex(
            ValueError,
            "relaxed trace artifact thresholds.material_visit_share_margin must be finite non-negative",
        ):
            self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

    def test_build_payload_fails_closed_for_insufficient_trace(self):
        decomposition, default, relaxed = self.valid_inputs()
        relaxed["insufficiency_reasons"] = ["too_few_trace_points"]

        with self.assertRaisesRegex(ValueError, r"relaxed trace artifact insufficiency_reasons must be \[\]"):
            self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

    def test_build_payload_rejects_mismatched_thresholds(self):
        decomposition, default, relaxed = self.valid_inputs()
        decomposition["thresholds_evaluated"]["relaxed"]["material_visit_share_margin"] = 0.03

        with self.assertRaisesRegex(
            ValueError,
            "decomposition thresholds_evaluated.relaxed must match relaxed thresholds",
        ):
            self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

    def test_build_payload_rejects_mismatched_source_artifact(self):
        decomposition, default, relaxed = self.valid_inputs()
        relaxed["source_artifact"]["selected_artifact"]["selected_target"] = "/tmp/other"

        with self.assertRaisesRegex(ValueError, "source artifacts must match"):
            self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

    def test_build_payload_rejects_mismatched_input_paths(self):
        decomposition, default, relaxed = self.valid_inputs()
        decomposition["input_artifacts"]["source_selection_score_artifact_path"] = "/tmp/other.json"

        with self.assertRaisesRegex(
            ValueError,
            "decomposition input_artifacts source_selection_score_artifact_path must match",
        ):
            self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

    def test_build_payload_rejects_mixed_trace_origins(self):
        decomposition, default, relaxed = self.valid_inputs()
        relaxed["trace_origin"] = "rerun"

        with self.assertRaisesRegex(ValueError, "trace_origin must match"):
            self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

    def test_build_payload_rejects_empty_checkpoint_sequences(self):
        decomposition, default, relaxed = self.valid_inputs()
        default["trace_points"] = []
        relaxed["trace_points"] = []

        with self.assertRaisesRegex(ValueError, "trace_points must be a non-empty list"):
            self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

    def test_build_payload_rejects_mismatched_checkpoint_sequences(self):
        decomposition, default, relaxed = self.valid_inputs()
        relaxed["trace_points"][1]["simulation"] = 3.0

        with self.assertRaisesRegex(ValueError, "checkpoint sequences must match"):
            self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

    def test_build_payload_rejects_duplicate_checkpoint_sequences(self):
        decomposition, default, relaxed = self.valid_inputs()
        default["trace_points"][1]["simulation"] = 1.0
        relaxed["trace_points"][1]["simulation"] = 1.0

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint sequences must not contain duplicates",
        ):
            self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

    def test_build_payload_rejects_out_of_order_checkpoint_sequences(self):
        decomposition, default, relaxed = self.valid_inputs()
        default["trace_points"] = [default["trace_points"][1], default["trace_points"][0]]
        relaxed["trace_points"] = [relaxed["trace_points"][1], relaxed["trace_points"][0]]

        with self.assertRaisesRegex(ValueError, "checkpoint sequences must be strictly increasing"):
            self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

    def test_build_payload_computes_checkpoint_margins_for_default_and_relaxed_traces(self):
        decomposition, default, relaxed = self.valid_inputs()

        payload = self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

        self.assertEqual(
            [
                {
                    "simulation": 1.0,
                    "default": {
                        "selected_minus_reference_q": -0.02,
                        "selected_minus_reference_selection_score": -0.04999999999999999,
                        "selected_minus_reference_visit_share": -1.0,
                    },
                    "relaxed": {
                        "selected_minus_reference_q": -0.02,
                        "selected_minus_reference_selection_score": -0.04999999999999999,
                        "selected_minus_reference_visit_share": -1.0,
                    },
                },
                {
                    "simulation": 2.0,
                    "default": {
                        "selected_minus_reference_q": 0.010000000000000002,
                        "selected_minus_reference_selection_score": 0.010000000000000009,
                        "selected_minus_reference_visit_share": -0.19999999999999996,
                    },
                    "relaxed": {
                        "selected_minus_reference_q": 0.010000000000000002,
                        "selected_minus_reference_selection_score": 0.010000000000000009,
                        "selected_minus_reference_visit_share": -0.19999999999999996,
                    },
                },
            ],
            payload["checkpoint_audit"],
        )

    def test_build_payload_preserves_null_margins_when_move_metrics_are_missing(self):
        decomposition, default, relaxed = self.valid_inputs()
        default["trace_points"][1]["moves"] = [{"move": 0}, {"move": 2}]
        relaxed["trace_points"][1]["moves"] = [{"move": 0}, {"move": 2}]

        payload = self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

        self.assertIsNone(payload["checkpoint_audit"][1]["default"]["selected_minus_reference_q"])
        self.assertIsNone(
            payload["checkpoint_audit"][1]["default"]["selected_minus_reference_selection_score"]
        )
        self.assertEqual(
            -0.19999999999999996,
            payload["checkpoint_audit"][1]["default"]["selected_minus_reference_visit_share"],
        )

    def test_build_payload_derives_first_positive_and_first_material_checkpoints(self):
        decomposition, default, relaxed = self.valid_inputs()
        relaxed["trace_points"] = [
            self.trace_point(
                simulation=1.0,
                selected_visits=0.0,
                reference_visits=1.0,
                selected_q=0.0,
                reference_q=0.02,
                selected_selection_score=0.10,
                reference_selection_score=0.15,
            ),
            self.trace_point(
                simulation=2.0,
                selected_visits=3.0,
                reference_visits=2.0,
                selected_q=0.04,
                reference_q=0.03,
                selected_selection_score=0.22,
                reference_selection_score=0.16,
            ),
        ]

        payload = self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

        self.assertEqual(
            {"simulation": 2.0, "margin": 0.010000000000000002},
            payload["first_positive_checkpoints"]["relaxed"]["q"],
        )
        self.assertEqual(
            {"simulation": 2.0, "margin": 0.06},
            payload["first_material_checkpoints"]["relaxed"]["selection_score"],
        )
        self.assertIsNone(payload["first_material_checkpoints"]["default"]["q"])

    def test_build_payload_classifies_weak_aligned_drift(self):
        decomposition, default, relaxed = self.valid_inputs()
        relaxed["trace_points"] = [
            self.trace_point(
                simulation=1.0,
                selected_visits=14.0,
                reference_visits=13.0,
                selected_q=0.02,
                reference_q=0.01,
                selected_selection_score=0.13,
                reference_selection_score=0.12,
            )
        ]
        default["trace_points"] = copy.deepcopy(relaxed["trace_points"])

        payload = self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

        self.assertEqual("weak_aligned_drift", payload["classification"]["classification"])
        self.assertEqual("write_002_low_confidence_policy_value_interaction_spec", payload["decision"])

    def test_build_payload_classifies_early_selection_score_only(self):
        decomposition, default, relaxed = self.valid_inputs()
        relaxed["trace_points"] = self.material_selection_score_trace()
        default["trace_points"] = copy.deepcopy(relaxed["trace_points"])

        payload = self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

        self.assertEqual("early_selection_score_only", payload["classification"]["classification"])
        self.assertEqual("write_002_selection_score_component_audit_spec", payload["decision"])

    def test_build_payload_classifies_late_visit_share_only(self):
        decomposition, default, relaxed = self.valid_inputs()
        relaxed["trace_points"] = self.material_visit_share_trace()
        default["trace_points"] = copy.deepcopy(relaxed["trace_points"])

        payload = self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

        self.assertEqual("late_visit_share_only", payload["classification"]["classification"])
        self.assertEqual("write_002_visit_accumulation_audit_spec", payload["decision"])

    def test_build_payload_classifies_mixed_low_confidence_signal_for_simultaneous_material_support(self):
        decomposition, default, relaxed = self.valid_inputs()
        relaxed["trace_points"] = self.simultaneous_material_trace()
        default["trace_points"] = copy.deepcopy(relaxed["trace_points"])

        payload = self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

        self.assertEqual("mixed_low_confidence_signal", payload["classification"]["classification"])
        self.assertEqual("write_002_low_confidence_trace_comparison_spec", payload["decision"])

    def test_build_payload_keeps_simultaneous_first_material_support_out_of_weak_aligned_drift(self):
        decomposition, default, relaxed = self.valid_inputs()
        relaxed["trace_points"] = [
            self.trace_point(
                simulation=1.0,
                selected_visits=2.0,
                reference_visits=2.0,
                selected_q=0.05,
                reference_q=0.0,
                selected_selection_score=0.22,
                reference_selection_score=0.16,
            ),
            self.trace_point(
                simulation=2.0,
                selected_visits=14.0,
                reference_visits=13.0,
                selected_q=0.02,
                reference_q=0.01,
                selected_selection_score=0.13,
                reference_selection_score=0.12,
            ),
        ]
        default["trace_points"] = copy.deepcopy(relaxed["trace_points"])

        payload = self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

        self.assertEqual("mixed_low_confidence_signal", payload["classification"]["classification"])
        self.assertEqual("write_002_low_confidence_trace_comparison_spec", payload["decision"])

    def test_build_payload_classifies_metric_audit_inconclusive_when_no_signal_is_usable(self):
        decomposition, default, relaxed = self.valid_inputs()
        relaxed["trace_points"] = [
            self.trace_point(
                simulation=1.0,
                selected_visits=0.0,
                reference_visits=1.0,
                selected_q=None,
                reference_q=None,
                selected_selection_score=None,
                reference_selection_score=None,
            )
        ]
        default["trace_points"] = copy.deepcopy(relaxed["trace_points"])

        payload = self.build_payload(decomposition=decomposition, default=default, relaxed=relaxed)

        self.assertEqual("metric_audit_inconclusive", payload["classification"]["classification"])
        self.assertEqual("stop_002_metric_audit_inconclusive", payload["decision"])

    def test_main_writes_sorted_payload_and_prints_compact_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            decomposition_path = tmp_path / "decomposition.json"
            default_path = tmp_path / "default.json"
            relaxed_path = tmp_path / "relaxed.json"
            out_path = tmp_path / "diagnostics" / "metric_audit.json"
            decomposition, default, relaxed = self.valid_inputs()
            decomposition["input_artifacts"]["source_selection_score_artifact_path"] = str(default_path)
            decomposition["input_artifacts"]["source_threshold_review_artifact_path"] = str(relaxed_path)

            decomposition_path.write_text(json.dumps(decomposition), encoding="utf-8")
            default_path.write_text(json.dumps(default), encoding="utf-8")
            relaxed_path.write_text(json.dumps(relaxed), encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = module.main(
                    [
                        "--source-decomposition-artifact",
                        str(decomposition_path),
                        "--source-selection-score-artifact",
                        str(default_path),
                        "--source-threshold-relaxed-selection-score-artifact",
                        str(relaxed_path),
                        "--out",
                        str(out_path),
                    ]
                )

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            summary = json.loads(stdout.getvalue())

        self.assertEqual(0, exit_code)
        self.assertEqual(module.SCHEMA, payload["schema"])
        self.assertEqual(str(out_path), summary["artifact_path"])
        self.assertEqual(module.SCHEMA, summary["schema"])
        self.assertEqual(payload["classification"]["classification"], summary["classification"])
        self.assertEqual(payload["decision"], summary["decision"])


if __name__ == "__main__":
    unittest.main()
