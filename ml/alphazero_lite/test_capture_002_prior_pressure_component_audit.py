import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ml.alphazero_lite import capture_002_prior_pressure_component_audit as module


class Capture002PriorPressureComponentAuditContractTest(unittest.TestCase):
    def test_contract_constants_are_stable(self):
        self.assertEqual("azlite_capture_002_prior_pressure_component_audit_v1", module.SCHEMA)
        self.assertEqual(
            "azlite_capture_002_selection_score_component_audit_v1",
            module.SOURCE_SELECTION_SCORE_COMPONENT_AUDIT_SCHEMA,
        )
        self.assertEqual("azlite_capture_002_metric_co_movement_audit_v1", module.SOURCE_METRIC_AUDIT_SCHEMA)
        self.assertEqual("azlite_capture_002_selection_score_trace_v1", module.SOURCE_SELECTION_SCORE_SCHEMA)
        self.assertEqual(
            "azlite_capture_002_trace_checkpoint_canonicalization_v1",
            module.SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA,
        )
        self.assertEqual("capture_available-002", module.ROW_ID)
        self.assertEqual("prior_pressure_lead", module.EXPECTED_COMPONENT_AUDIT_CLASSIFICATION)
        self.assertEqual(
            "write_002_prior_pressure_component_spec",
            module.EXPECTED_COMPONENT_AUDIT_DECISION,
        )
        self.assertEqual(
            {
                "selection_score_residual_lead": "write_002_selection_score_residual_spec",
                "visit_alignment_pressure": "write_002_visit_alignment_pressure_spec",
                "mixed_prior_pressure_signal": "write_002_prior_pressure_mixed_signal_spec",
                "prior_pressure_component_inconclusive": "stop_002_prior_pressure_component_inconclusive",
            },
            module.CLASSIFICATION_DECISIONS,
        )

    def test_parse_args_reads_required_paths(self):
        args = module.parse_args(
            [
                "--source-selection-score-component-audit-artifact",
                "/tmp/selection_score_component_audit.json",
                "--source-metric-audit-artifact",
                "/tmp/metric_audit.json",
                "--source-selection-score-artifact",
                "/tmp/default.json",
                "--source-threshold-relaxed-selection-score-artifact",
                "/tmp/relaxed.json",
                "--out",
                "/tmp/out.json",
            ]
        )

        self.assertEqual(
            Path("/tmp/selection_score_component_audit.json"),
            args.source_selection_score_component_audit_artifact,
        )
        self.assertEqual(Path("/tmp/metric_audit.json"), args.source_metric_audit_artifact)
        self.assertEqual(Path("/tmp/default.json"), args.source_selection_score_artifact)
        self.assertEqual(
            Path("/tmp/relaxed.json"),
            args.source_threshold_relaxed_selection_score_artifact,
        )
        self.assertIsNone(args.source_checkpoint_canonicalization_artifact)
        self.assertEqual(Path("/tmp/out.json"), args.out)

    def test_parse_args_accepts_optional_checkpoint_canonicalization_path(self):
        args = module.parse_args(
            [
                "--source-selection-score-component-audit-artifact",
                "/tmp/selection_score_component_audit.json",
                "--source-metric-audit-artifact",
                "/tmp/metric_audit.json",
                "--source-selection-score-artifact",
                "/tmp/default.json",
                "--source-threshold-relaxed-selection-score-artifact",
                "/tmp/relaxed.json",
                "--source-checkpoint-canonicalization-artifact",
                "/tmp/canonicalization.json",
                "--out",
                "/tmp/out.json",
            ]
        )

        self.assertEqual(
            Path("/tmp/canonicalization.json"),
            args.source_checkpoint_canonicalization_artifact,
        )

    def test_cli_test_case_only_exposes_cli_specific_tests(self):
        self.assertEqual(
            ["test_main_writes_sorted_json_and_prints_compact_summary_json"],
            unittest.defaultTestLoader.getTestCaseNames(Capture002PriorPressureComponentAuditCliTest),
        )


class Capture002PriorPressureComponentAuditTestSupport:
    def selected_artifact(self) -> dict:
        return {
            "path": "/tmp/source/selected",
            "selected_artifact": "/tmp/source/selection/artifact",
            "selected_target": "/tmp/source/selected",
            "provenance_source": "selection_manifest.selected_target",
        }

    def source_artifact_with_provenance(self) -> dict:
        return {
            "artifact_path": "/tmp/source/upstream.json",
            "schema": "azlite_shared_full_search_drift_diagnostic_v1",
            "classification": {
                "classification": "shared_mechanism_disproved",
                "evidence_summary": "upstream evidence",
            },
            "decision": "write_row_split_followup_spec",
            "row_id": "capture_available-002",
            "reference_move": 2,
            "full_search_selected_move": 0,
            "selected_artifact": self.selected_artifact(),
        }

    def trace_point(
        self,
        *,
        simulation: float,
        q_margin: float | None,
        selection_score_margin: float | None,
        selected_visits: float,
        reference_visits: float,
    ) -> dict:
        moves = [{"move": 0}, {"move": 2}]
        if q_margin is not None:
            if q_margin >= 0:
                moves[0]["q_value"] = q_margin
                moves[1]["q_value"] = 0.0
            else:
                moves[0]["q_value"] = 0.0
                moves[1]["q_value"] = -q_margin
        if selection_score_margin is not None:
            if selection_score_margin >= 0:
                moves[0]["selection_score"] = selection_score_margin
                moves[1]["selection_score"] = 0.0
            else:
                moves[0]["selection_score"] = 0.0
                moves[1]["selection_score"] = -selection_score_margin
        return {
            "simulation": simulation,
            "selected_move": 0,
            "reference_move_by_prior": 2,
            "visits": [selected_visits, 0.0, reference_visits, 0.0, 0.0],
            "moves": moves,
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

    def trace_artifact(
        self,
        *,
        thresholds: dict,
        trace_points: list[dict],
        trace_origin: str = "extracted",
        source_artifact: dict | None = None,
    ) -> dict:
        return {
            "schema": module.SOURCE_SELECTION_SCORE_SCHEMA,
            "classification": {"classification": "unresolved", "evidence_summary": "unresolved"},
            "decision": "write_002_unresolved_trace_review_spec",
            "insufficiency_reasons": [],
            "trace_origin": trace_origin,
            "source_artifact": copy.deepcopy(source_artifact or self.source_artifact_with_provenance()),
            "thresholds": copy.deepcopy(thresholds),
            "trace_points": copy.deepcopy(trace_points),
        }

    def metric_audit_artifact(
        self,
        *,
        classification: str = "early_selection_score_only",
        decision: str = "write_002_selection_score_component_audit_spec",
        source_artifact: dict | None = None,
    ) -> dict:
        source_artifact = copy.deepcopy(source_artifact or self.source_artifact_with_provenance())
        return {
            "schema": module.SOURCE_METRIC_AUDIT_SCHEMA,
            "hypothesis": "metric_co_movement_audit",
            "classification": {"classification": classification, "evidence_summary": "selection-score lead"},
            "decision": decision,
            "input_artifacts": {
                "source_decomposition_artifact_path": "/tmp/decomposition.json",
                "source_selection_score_artifact_path": "/tmp/default.json",
                "source_threshold_relaxed_selection_score_artifact_path": "/tmp/relaxed.json",
            },
            "source_artifact": source_artifact,
            "thresholds_evaluated": {
                "default": self.default_thresholds(),
                "relaxed": self.relaxed_thresholds(),
            },
            "checkpoint_audit": [],
            "first_positive_checkpoints": {
                "default": {"q": None, "selection_score": None, "visit_share": None},
                "relaxed": {"q": None, "selection_score": None, "visit_share": None},
            },
            "first_material_checkpoints": {
                "default": {"q": None, "selection_score": {"simulation": 1.0, "margin": 0.06}, "visit_share": None},
                "relaxed": {"q": None, "selection_score": {"simulation": 1.0, "margin": 0.06}, "visit_share": None},
            },
            "final_margin_summary": {
                "default_q_margin": 0.0,
                "default_selection_score_margin": 0.06,
                "default_visit_share_margin": 0.0,
                "relaxed_q_margin": 0.0,
                "relaxed_selection_score_margin": 0.06,
                "relaxed_visit_share_margin": 0.0,
            },
            "source_snapshots": {
                "decomposition_classification": {"classification": "metric_co_movement"},
                "default_trace_classification": {"classification": "unresolved"},
                "relaxed_trace_classification": {"classification": "unresolved"},
                "default_trace_origin": "extracted",
                "relaxed_trace_origin": "extracted",
            },
        }

    def component_audit_artifact(
        self,
        *,
        classification: str = "prior_pressure_lead",
        decision: str = "write_002_prior_pressure_component_spec",
        source_artifact: dict | None = None,
    ) -> dict:
        source_artifact = copy.deepcopy(source_artifact or self.source_artifact_with_provenance())
        return {
            "schema": module.SOURCE_SELECTION_SCORE_COMPONENT_AUDIT_SCHEMA,
            "hypothesis": "selection_score_component_audit",
            "classification": {
                "classification": classification,
                "evidence_summary": "prior-pressure lead",
            },
            "decision": decision,
            "input_artifacts": {
                "source_metric_audit_artifact_path": "/tmp/metric_audit.json",
                "source_selection_score_artifact_path": "/tmp/default.json",
                "source_threshold_relaxed_selection_score_artifact_path": "/tmp/relaxed.json",
            },
            "source_artifact": source_artifact,
            "thresholds_evaluated": {
                "selection_score": self.default_thresholds()["material_selection_score_margin"],
                "meaningful_q": self.default_thresholds()["meaningful_q_margin"],
            },
            "checkpoint_audit": [],
            "first_positive_checkpoints": {
                "default": {"prior_pressure": {"simulation": 1.0}, "child_q_lift": None},
                "relaxed": {"prior_pressure": {"simulation": 1.0}, "child_q_lift": None},
            },
            "first_material_checkpoints": {
                "default": {
                    "prior_pressure": {
                        "simulation": 1.0,
                        "selection_score_margin": 0.06,
                        "q_margin": -0.02,
                    },
                    "child_q_lift": None,
                },
                "relaxed": {
                    "prior_pressure": {
                        "simulation": 1.0,
                        "selection_score_margin": 0.06,
                        "q_margin": -0.02,
                    },
                    "child_q_lift": None,
                },
            },
            "selection_score_support_signatures": {
                "default": {},
                "relaxed": {},
                "branch_level_disagreement": {
                    "branch_signature_disagreement": False,
                    "first_positive_first_material_conflict": False,
                    "first_positive_signatures": {
                        "default": "prior_pressure_lead",
                        "relaxed": "prior_pressure_lead",
                    },
                    "first_material_signatures": {
                        "default": "prior_pressure_lead",
                        "relaxed": "prior_pressure_lead",
                    },
                },
            },
            "source_snapshots": {
                "metric_audit_classification": {"classification": "early_selection_score_only"},
                "default_trace_classification": {"classification": "unresolved"},
                "relaxed_trace_classification": {"classification": "unresolved"},
                "default_trace_origin": "extracted",
                "relaxed_trace_origin": "extracted",
            },
        }

    def canonicalization_artifact(
        self,
        *,
        schema: str | None = None,
        source_artifact: dict | None = None,
        default_sequence: list[float] | None = None,
        relaxed_sequence: list[float] | None = None,
        thresholds_evaluated: dict | None = None,
        trace_origin: str = "extracted",
    ) -> dict:
        if default_sequence is None:
            default_sequence = [1.0, 2.0]
        if relaxed_sequence is None:
            relaxed_sequence = copy.deepcopy(default_sequence)
        if thresholds_evaluated is None:
            thresholds_evaluated = {
                "default": self.default_thresholds(),
                "relaxed": self.relaxed_thresholds(),
            }
        return {
            "schema": schema or module.SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA,
            "decision": "write_002_metric_audit_canonical_input_spec",
            "input_artifacts": {
                "source_selection_score_artifact_path": "/tmp/default.json",
                "source_threshold_relaxed_selection_score_artifact_path": "/tmp/relaxed.json",
            },
            "source_artifact": copy.deepcopy(source_artifact or self.source_artifact_with_provenance()),
            "canonicalization_status": {"safe_for_followup_spec": True},
            "canonical_sequences_match": True,
            "canonical_checkpoint_sequences": {
                "default": copy.deepcopy(default_sequence),
                "relaxed": copy.deepcopy(relaxed_sequence),
            },
            "thresholds_evaluated": copy.deepcopy(thresholds_evaluated),
            "trace_origin": trace_origin,
        }

    def valid_inputs(self) -> tuple[dict, dict, dict, dict]:
        trace_points = [
            self.trace_point(
                simulation=1.0,
                q_margin=-0.02,
                selection_score_margin=0.06,
                selected_visits=1.0,
                reference_visits=1.0,
            ),
            self.trace_point(
                simulation=2.0,
                q_margin=-0.01,
                selection_score_margin=0.07,
                selected_visits=2.0,
                reference_visits=1.0,
            ),
        ]
        source_artifact = self.source_artifact_with_provenance()
        metric_audit = self.metric_audit_artifact(source_artifact=source_artifact)
        default_trace = self.trace_artifact(
            thresholds=self.default_thresholds(),
            trace_points=trace_points,
            source_artifact=source_artifact,
        )
        relaxed_trace = self.trace_artifact(
            thresholds=self.relaxed_thresholds(),
            trace_points=trace_points,
            source_artifact=source_artifact,
        )
        component_audit = self.component_audit_artifact(source_artifact=source_artifact)
        return component_audit, metric_audit, default_trace, relaxed_trace

    def build_payload(
        self,
        component_audit=None,
        metric_audit=None,
        default_trace=None,
        relaxed_trace=None,
        checkpoint_canonicalization_artifact=None,
        source_checkpoint_canonicalization_artifact_path=None,
    ) -> dict:
        valid_component_audit, valid_metric_audit, valid_default_trace, valid_relaxed_trace = self.valid_inputs()
        return module.build_payload(
            component_audit or valid_component_audit,
            metric_audit or valid_metric_audit,
            default_trace or valid_default_trace,
            relaxed_trace or valid_relaxed_trace,
            source_selection_score_component_audit_artifact_path="/tmp/selection_score_component_audit.json",
            source_metric_audit_artifact_path="/tmp/metric_audit.json",
            source_selection_score_artifact_path="/tmp/default.json",
            source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
            checkpoint_canonicalization_artifact=checkpoint_canonicalization_artifact,
            source_checkpoint_canonicalization_artifact_path=source_checkpoint_canonicalization_artifact_path,
        )

    def replace_branch_prior_pressure_checkpoint(
        self,
        component_audit: dict,
        *,
        selection_score_margin: float | None,
        q_margin: float | None,
    ) -> None:
        for branch in ("default", "relaxed"):
            component_audit["first_material_checkpoints"][branch]["prior_pressure"] = {
                "simulation": 1.0,
                "selection_score_margin": selection_score_margin,
                "q_margin": q_margin,
            }

    def replace_branch_prior_pressure_checkpoint_for_branch(
        self,
        component_audit: dict,
        *,
        branch: str,
        selection_score_margin: float | None,
        q_margin: float | None,
    ) -> None:
        component_audit["first_material_checkpoints"][branch]["prior_pressure"] = {
            "simulation": 1.0,
            "selection_score_margin": selection_score_margin,
            "q_margin": q_margin,
        }

    def replace_first_trace_point(
        self,
        default_trace: dict,
        relaxed_trace: dict,
        *,
        q_margin: float | None,
        selection_score_margin: float | None,
        selected_visits: float,
        reference_visits: float,
    ) -> None:
        trace_point = self.trace_point(
            simulation=1.0,
            q_margin=q_margin,
            selection_score_margin=selection_score_margin,
            selected_visits=selected_visits,
            reference_visits=reference_visits,
        )
        default_trace["trace_points"][0] = copy.deepcopy(trace_point)
        relaxed_trace["trace_points"][0] = copy.deepcopy(trace_point)

    def replace_first_trace_point_for_branch(
        self,
        trace_artifact: dict,
        *,
        q_margin: float | None,
        selection_score_margin: float | None,
        selected_visits: float,
        reference_visits: float,
    ) -> None:
        trace_artifact["trace_points"][0] = self.trace_point(
            simulation=1.0,
            q_margin=q_margin,
            selection_score_margin=selection_score_margin,
            selected_visits=selected_visits,
            reference_visits=reference_visits,
        )

    def assert_build_payload_error(
        self,
        message: str,
        *,
        component_audit=None,
        metric_audit=None,
        default_trace=None,
        relaxed_trace=None,
    ) -> None:
        with self.assertRaisesRegex(ValueError, message):
            self.build_payload(
                component_audit=component_audit,
                metric_audit=metric_audit,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_non_dict_component_audit_classification(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["classification"] = "prior_pressure_lead"

        with self.assertRaisesRegex(
            ValueError,
            "prior-pressure component audit upstream classification must be an object",
        ):
            self.build_payload(
                component_audit=component_audit,
                metric_audit=metric_audit,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_non_dict_component_audit_artifact(self):
        _, metric_audit, default_trace, relaxed_trace = self.valid_inputs()

        with self.assertRaisesRegex(
            ValueError,
            "prior-pressure component audit upstream artifact must be an object",
        ):
            self.build_payload(
                component_audit="invalid",
                metric_audit=metric_audit,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_wrong_component_audit_classification(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["classification"]["classification"] = "mixed_selection_score_signal"

        with self.assertRaisesRegex(
            ValueError,
            "prior-pressure component audit upstream classification must be prior_pressure_lead",
        ):
            self.build_payload(
                component_audit=component_audit,
                metric_audit=metric_audit,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_wrong_component_audit_schema(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["schema"] = "wrong"

        with self.assertRaisesRegex(
            ValueError,
            "prior-pressure component audit upstream artifact has wrong schema",
        ):
            self.build_payload(
                component_audit=component_audit,
                metric_audit=metric_audit,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_wrong_component_audit_decision(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["decision"] = "write_002_mixed_selection_score_component_spec"

        with self.assertRaisesRegex(
            ValueError,
            "prior-pressure component audit upstream decision must be write_002_prior_pressure_component_spec",
        ):
            self.build_payload(
                component_audit=component_audit,
                metric_audit=metric_audit,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_component_audit_wrong_row_id(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["source_artifact"]["row_id"] = "capture_available-003"

        self.assert_build_payload_error(
            "prior-pressure component audit upstream source_artifact.row_id must be capture_available-002",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_component_audit_non_integer_reference_move(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["source_artifact"]["reference_move"] = "2"

        self.assert_build_payload_error(
            "prior-pressure component audit upstream source_artifact.reference_move must be an integer",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_component_audit_non_integer_full_search_selected_move(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["source_artifact"]["full_search_selected_move"] = "0"

        self.assert_build_payload_error(
            "prior-pressure component audit upstream source_artifact.full_search_selected_move must be an integer",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_component_audit_non_dict_selected_artifact(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["source_artifact"]["selected_artifact"] = "/tmp/not-an-object"

        self.assert_build_payload_error(
            "prior-pressure component audit upstream source_artifact.selected_artifact must be an object",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_metric_audit_reference_move_mismatch_vs_component_audit(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        metric_audit["source_artifact"]["reference_move"] = 1

        self.assert_build_payload_error(
            "metric audit artifact source_artifact.reference_move must match prior-pressure component audit upstream source_artifact.reference_move",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_non_dict_metric_audit_artifact(self):
        component_audit, _, default_trace, relaxed_trace = self.valid_inputs()

        self.assert_build_payload_error(
            "metric audit artifact must be an object",
            component_audit=component_audit,
            metric_audit="invalid",
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_wrong_metric_audit_schema(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        metric_audit["schema"] = "wrong"

        self.assert_build_payload_error(
            "metric audit artifact has wrong schema",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_wrong_metric_audit_classification(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        metric_audit["classification"]["classification"] = "mixed_selection_score_signal"

        self.assert_build_payload_error(
            "metric audit artifact classification must be early_selection_score_only",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_wrong_metric_audit_decision(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        metric_audit["decision"] = "stop_002_metric_audit_inconclusive"

        self.assert_build_payload_error(
            "metric audit artifact decision must be write_002_selection_score_component_audit_spec",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_metric_audit_wrong_row_id(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        metric_audit["source_artifact"]["row_id"] = "capture_available-003"

        self.assert_build_payload_error(
            "metric audit artifact source_artifact.row_id must be capture_available-002",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_mismatched_metric_audit_canonicalization_path(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["input_artifacts"]["source_checkpoint_canonicalization_artifact_path"] = (
            "/tmp/canonicalization.json"
        )
        metric_audit["input_artifacts"]["source_checkpoint_canonicalization_artifact_path"] = "/tmp/other.json"

        with self.assertRaisesRegex(
            ValueError,
            "metric audit input_artifacts source_checkpoint_canonicalization_artifact_path must match source path",
        ):
            self.build_payload(
                component_audit=component_audit,
                metric_audit=metric_audit,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
                checkpoint_canonicalization_artifact=self.canonicalization_artifact(),
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_unexpected_metric_audit_canonicalization_path_outside_canonical_mode(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        metric_audit["input_artifacts"]["source_checkpoint_canonicalization_artifact_path"] = "/tmp/canonicalization.json"

        self.assert_build_payload_error(
            "metric audit input_artifacts source_checkpoint_canonicalization_artifact_path must match source path",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_non_dict_default_trace_artifact(self):
        component_audit, metric_audit, _, relaxed_trace = self.valid_inputs()

        self.assert_build_payload_error(
            "default trace artifact must be an object",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace="invalid",
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_wrong_default_trace_schema(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["schema"] = "wrong"

        self.assert_build_payload_error(
            "default trace artifact has wrong schema",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_wrong_default_trace_classification(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["classification"]["classification"] = "mixed_selection_score_signal"

        self.assert_build_payload_error(
            "default trace artifact classification must be unresolved",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_wrong_default_trace_decision(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["decision"] = "write_002_selection_pressure_ablation_spec"

        self.assert_build_payload_error(
            "default trace artifact decision must be write_002_unresolved_trace_review_spec",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_default_trace_wrong_row_id(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["source_artifact"]["row_id"] = "capture_available-003"

        self.assert_build_payload_error(
            "default trace artifact source_artifact.row_id must be capture_available-002",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_default_trace_full_search_selected_move_mismatch_vs_component_audit(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["source_artifact"]["full_search_selected_move"] = 2

        self.assert_build_payload_error(
            "default trace artifact source_artifact.full_search_selected_move must match prior-pressure component audit upstream source_artifact.full_search_selected_move",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_non_dict_relaxed_trace_artifact(self):
        component_audit, metric_audit, default_trace, _ = self.valid_inputs()

        self.assert_build_payload_error(
            "relaxed trace artifact must be an object",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace="invalid",
        )

    def test_build_payload_rejects_wrong_relaxed_trace_schema(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        relaxed_trace["schema"] = "wrong"

        self.assert_build_payload_error(
            "relaxed trace artifact has wrong schema",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_wrong_relaxed_trace_classification(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        relaxed_trace["classification"]["classification"] = "mixed_selection_score_signal"

        self.assert_build_payload_error(
            "relaxed trace artifact classification must be unresolved",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_wrong_relaxed_trace_decision(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        relaxed_trace["decision"] = "write_002_selection_pressure_ablation_spec"

        self.assert_build_payload_error(
            "relaxed trace artifact decision must be write_002_unresolved_trace_review_spec",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_relaxed_trace_wrong_row_id(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        relaxed_trace["source_artifact"]["row_id"] = "capture_available-003"

        self.assert_build_payload_error(
            "relaxed trace artifact source_artifact.row_id must be capture_available-002",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_relaxed_trace_selected_artifact_mismatch_vs_component_audit(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        relaxed_trace["source_artifact"]["selected_artifact"]["path"] = "/tmp/other-selected"

        self.assert_build_payload_error(
            "relaxed trace artifact source_artifact.selected_artifact must match prior-pressure component audit upstream source_artifact.selected_artifact",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_preserves_canonicalization_input_path(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["input_artifacts"]["source_checkpoint_canonicalization_artifact_path"] = (
            "/tmp/canonicalization.json"
        )
        metric_audit["input_artifacts"]["source_checkpoint_canonicalization_artifact_path"] = (
            "/tmp/canonicalization.json"
        )

        payload = self.build_payload(
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
            checkpoint_canonicalization_artifact=self.canonicalization_artifact(),
            source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json"
        )

        self.assertEqual(
            "/tmp/canonicalization.json",
            payload["input_artifacts"]["source_checkpoint_canonicalization_artifact_path"],
        )

    def test_build_payload_rejects_canonicalization_path_without_artifact(self):
        with self.assertRaisesRegex(
            ValueError,
            "canonical mode requires checkpoint_canonicalization_artifact",
        ):
            self.build_payload(source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json")

    def test_build_payload_rejects_canonicalization_artifact_without_path(self):
        with self.assertRaisesRegex(
            ValueError,
            "canonical mode requires source_checkpoint_canonicalization_artifact_path",
        ):
            self.build_payload(checkpoint_canonicalization_artifact=self.canonicalization_artifact())

    def test_build_payload_rejects_wrong_canonicalization_schema(self):
        with self.assertRaisesRegex(
            ValueError,
            "canonicalization artifact has wrong schema",
        ):
            self.build_payload(
                checkpoint_canonicalization_artifact=self.canonicalization_artifact(schema="wrong"),
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_non_dict_canonicalization_artifact(self):
        with self.assertRaisesRegex(
            ValueError,
            "checkpoint_canonicalization_artifact must be an object",
        ):
            self.build_payload(
                checkpoint_canonicalization_artifact="invalid",
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_mismatched_component_audit_canonicalization_path(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["input_artifacts"]["source_checkpoint_canonicalization_artifact_path"] = "/tmp/other.json"

        with self.assertRaisesRegex(
            ValueError,
            "prior-pressure component audit upstream input_artifacts source_checkpoint_canonicalization_artifact_path must match source path",
        ):
            self.build_payload(
                component_audit=component_audit,
                metric_audit=metric_audit,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
                checkpoint_canonicalization_artifact=self.canonicalization_artifact(),
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_unexpected_component_audit_canonicalization_path_outside_canonical_mode(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["input_artifacts"]["source_checkpoint_canonicalization_artifact_path"] = "/tmp/canonicalization.json"

        self.assert_build_payload_error(
            "prior-pressure component audit upstream input_artifacts source_checkpoint_canonicalization_artifact_path must be absent outside canonical mode",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_accepts_duplicate_equivalent_root_snapshots_with_valid_canonicalization_artifact(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["input_artifacts"]["source_checkpoint_canonicalization_artifact_path"] = (
            "/tmp/canonicalization.json"
        )
        metric_audit["input_artifacts"]["source_checkpoint_canonicalization_artifact_path"] = (
            "/tmp/canonicalization.json"
        )
        duplicate_default_point = copy.deepcopy(default_trace["trace_points"][0])
        duplicate_relaxed_point = copy.deepcopy(relaxed_trace["trace_points"][0])
        default_trace["trace_points"] = [
            default_trace["trace_points"][0],
            duplicate_default_point,
            default_trace["trace_points"][1],
        ]
        relaxed_trace["trace_points"] = [
            relaxed_trace["trace_points"][0],
            duplicate_relaxed_point,
            relaxed_trace["trace_points"][1],
        ]

        payload = self.build_payload(
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
            checkpoint_canonicalization_artifact=self.canonicalization_artifact(),
            source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
        )

        self.assertEqual(
            "/tmp/canonicalization.json",
            payload["input_artifacts"]["source_checkpoint_canonicalization_artifact_path"],
        )
        self.assertEqual(1.0, payload["branch_level_evidence"]["default"]["upstream_checkpoint_echo"]["simulation"])

    def test_build_payload_rejects_duplicate_checkpoint_sequences_without_canonicalization_artifact(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        duplicate_default_point = copy.deepcopy(default_trace["trace_points"][0])
        duplicate_relaxed_point = copy.deepcopy(relaxed_trace["trace_points"][0])
        default_trace["trace_points"] = [
            default_trace["trace_points"][0],
            duplicate_default_point,
            default_trace["trace_points"][1],
        ]
        relaxed_trace["trace_points"] = [
            relaxed_trace["trace_points"][0],
            duplicate_relaxed_point,
            relaxed_trace["trace_points"][1],
        ]

        with self.assertRaisesRegex(ValueError, "checkpoint sequences must not contain duplicates"):
            self.build_payload(
                component_audit=component_audit,
                metric_audit=metric_audit,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_conflicting_skipped_duplicates_in_canonical_mode(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["input_artifacts"]["source_checkpoint_canonicalization_artifact_path"] = (
            "/tmp/canonicalization.json"
        )
        metric_audit["input_artifacts"]["source_checkpoint_canonicalization_artifact_path"] = (
            "/tmp/canonicalization.json"
        )
        duplicate_default_point = copy.deepcopy(default_trace["trace_points"][0])
        duplicate_default_point["selected_move"] = 2
        duplicate_relaxed_point = copy.deepcopy(relaxed_trace["trace_points"][0])
        default_trace["trace_points"] = [
            default_trace["trace_points"][0],
            duplicate_default_point,
            default_trace["trace_points"][1],
        ]
        relaxed_trace["trace_points"] = [
            relaxed_trace["trace_points"][0],
            duplicate_relaxed_point,
            relaxed_trace["trace_points"][1],
        ]

        with self.assertRaisesRegex(
            ValueError,
            "skipped duplicate checkpoint must match kept checkpoint contents",
        ):
            self.build_payload(
                component_audit=component_audit,
                metric_audit=metric_audit,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
                checkpoint_canonicalization_artifact=self.canonicalization_artifact(),
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_canonical_sequence_mismatch(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["input_artifacts"]["source_checkpoint_canonicalization_artifact_path"] = (
            "/tmp/canonicalization.json"
        )
        metric_audit["input_artifacts"]["source_checkpoint_canonicalization_artifact_path"] = (
            "/tmp/canonicalization.json"
        )
        duplicate_default_point = copy.deepcopy(default_trace["trace_points"][0])
        duplicate_relaxed_point = copy.deepcopy(relaxed_trace["trace_points"][0])
        default_trace["trace_points"] = [
            default_trace["trace_points"][0],
            duplicate_default_point,
            default_trace["trace_points"][1],
        ]
        relaxed_trace["trace_points"] = [
            relaxed_trace["trace_points"][0],
            duplicate_relaxed_point,
            relaxed_trace["trace_points"][1],
        ]

        with self.assertRaisesRegex(
            ValueError,
            "collapsed original checkpoint sequence must match canonical checkpoint sequence",
        ):
            self.build_payload(
                component_audit=component_audit,
                metric_audit=metric_audit,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
                checkpoint_canonicalization_artifact=self.canonicalization_artifact(
                    default_sequence=[1.0, 2.0, 3.0],
                    relaxed_sequence=[1.0, 2.0, 3.0],
                ),
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_missing_default_prior_pressure_checkpoint(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["first_material_checkpoints"]["default"]["prior_pressure"] = None

        self.assert_build_payload_error(
            "prior-pressure component audit upstream default first_material_checkpoints.prior_pressure must be present",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_missing_relaxed_prior_pressure_checkpoint(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["first_material_checkpoints"]["relaxed"]["prior_pressure"] = None

        self.assert_build_payload_error(
            "prior-pressure component audit upstream relaxed first_material_checkpoints.prior_pressure must be present",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_default_prior_pressure_checkpoint_missing_from_trace(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["first_material_checkpoints"]["default"]["prior_pressure"]["simulation"] = 9.0

        self.assert_build_payload_error(
            "prior-pressure component audit upstream default prior_pressure checkpoint simulation 9.0 is missing from validated trace",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_default_prior_pressure_selection_score_margin_mismatch(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["first_material_checkpoints"]["default"]["prior_pressure"]["selection_score_margin"] = 0.08

        self.assert_build_payload_error(
            "prior-pressure component audit upstream default prior_pressure checkpoint selection_score_margin must match validated trace",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_relaxed_prior_pressure_q_margin_mismatch(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["first_material_checkpoints"]["relaxed"]["prior_pressure"]["q_margin"] = -0.03

        self.assert_build_payload_error(
            "prior-pressure component audit upstream relaxed prior_pressure checkpoint q_margin must match validated trace",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_component_audit_mismatched_source_metric_path(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["input_artifacts"]["source_metric_audit_artifact_path"] = "/tmp/other_metric_audit.json"

        self.assert_build_payload_error(
            "prior-pressure component audit upstream input_artifacts source_metric_audit_artifact_path must match source path",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_component_audit_mismatched_source_default_trace_path(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["input_artifacts"]["source_selection_score_artifact_path"] = "/tmp/other_default.json"

        self.assert_build_payload_error(
            "prior-pressure component audit upstream input_artifacts source_selection_score_artifact_path must match source path",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_component_audit_mismatched_source_relaxed_trace_path(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["input_artifacts"]["source_threshold_relaxed_selection_score_artifact_path"] = "/tmp/other_relaxed.json"

        self.assert_build_payload_error(
            "prior-pressure component audit upstream input_artifacts source_threshold_relaxed_selection_score_artifact_path must match source path",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_missing_source_artifact(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["source_artifact"] = None

        with self.assertRaisesRegex(
            ValueError,
            "prior-pressure component audit upstream source_artifact must be an object",
        ):
            self.build_payload(
                component_audit=component_audit,
                metric_audit=metric_audit,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_returns_source_artifact_copy(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()

        payload = self.build_payload(
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

        payload["source_artifact"]["classification"]["classification"] = "mutated"
        payload["source_artifact"]["selected_artifact"]["path"] = "/tmp/mutated"

        self.assertEqual(
            "shared_mechanism_disproved",
            component_audit["source_artifact"]["classification"]["classification"],
        )
        self.assertEqual(
            "/tmp/source/selected",
            component_audit["source_artifact"]["selected_artifact"]["path"],
        )

    def test_build_payload_includes_branch_level_evidence(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()

        payload = self.build_payload(
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

        self.assertIn("branch_level_evidence", payload)
        self.assertEqual({"default", "relaxed"}, set(payload["branch_level_evidence"].keys()))

        for branch, visit_share_threshold in (("default", 0.05), ("relaxed", 0.04)):
            evidence = payload["branch_level_evidence"][branch]
            self.assertEqual(
                component_audit["first_material_checkpoints"][branch]["prior_pressure"],
                evidence["upstream_checkpoint_echo"],
            )
            self.assertEqual(0.06, evidence["selection_score_margin"])
            self.assertEqual(-0.02, evidence["q_margin"])
            self.assertEqual(0.06, evidence["selection_score_residual_margin"])
            self.assertEqual(1.0, evidence["selected_visits"])
            self.assertEqual(1.0, evidence["reference_visits"])
            self.assertEqual(0.5, evidence["selected_visit_share"])
            self.assertEqual(0.5, evidence["reference_visit_share"])
            self.assertEqual(0.05, evidence["selection_score_residual_threshold"])
            self.assertEqual(visit_share_threshold, evidence["visit_share_threshold"])
            self.assertEqual("selection_score_residual_lead", evidence["explanation_candidate"])

    def test_build_payload_rejects_malformed_branch_evidence_threshold_value(self):
        for threshold_key in ("material_selection_score_margin", "material_visit_share_margin"):
            with self.subTest(threshold_key=threshold_key):
                component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
                if threshold_key == "material_selection_score_margin":
                    self.replace_branch_prior_pressure_checkpoint(
                        component_audit,
                        selection_score_margin=None,
                        q_margin=None,
                    )
                    self.replace_first_trace_point(
                        default_trace,
                        relaxed_trace,
                        q_margin=None,
                        selection_score_margin=None,
                        selected_visits=1.0,
                        reference_visits=1.0,
                    )
                else:
                    self.replace_first_trace_point(
                        default_trace,
                        relaxed_trace,
                        q_margin=-0.02,
                        selection_score_margin=0.06,
                        selected_visits=1.0,
                        reference_visits=1.0,
                    )
                    default_trace["trace_points"][0]["visits"] = "invalid"
                default_trace["thresholds"][threshold_key] = "invalid"

                with self.assertRaisesRegex(
                    ValueError,
                    rf"default branch evidence thresholds\.{threshold_key} must be a finite non-negative number",
                ):
                    self.build_payload(
                        component_audit=component_audit,
                        metric_audit=metric_audit,
                        default_trace=default_trace,
                        relaxed_trace=relaxed_trace,
                    )

    def test_build_payload_sets_visit_alignment_pressure_branch_candidate(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        self.replace_branch_prior_pressure_checkpoint(
            component_audit,
            selection_score_margin=0.06,
            q_margin=0.02,
        )
        self.replace_first_trace_point(
            default_trace,
            relaxed_trace,
            q_margin=0.02,
            selection_score_margin=0.06,
            selected_visits=4.0,
            reference_visits=1.0,
        )

        payload = self.build_payload(
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

        self.assertEqual("visit_alignment_pressure", payload["branch_level_evidence"]["default"]["explanation_candidate"])
        self.assertEqual("visit_alignment_pressure", payload["branch_level_evidence"]["relaxed"]["explanation_candidate"])

    def test_build_payload_sets_mixed_prior_pressure_signal_branch_candidate(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        self.replace_branch_prior_pressure_checkpoint(
            component_audit,
            selection_score_margin=0.05,
            q_margin=0.02,
        )
        self.replace_first_trace_point(
            default_trace,
            relaxed_trace,
            q_margin=0.02,
            selection_score_margin=0.05,
            selected_visits=51.0,
            reference_visits=49.0,
        )

        payload = self.build_payload(
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

        self.assertEqual(
            "prior_pressure_component_inconclusive",
            payload["branch_level_evidence"]["default"]["explanation_candidate"],
        )
        self.assertEqual(
            "prior_pressure_component_inconclusive",
            payload["branch_level_evidence"]["relaxed"]["explanation_candidate"],
        )

    def test_build_payload_rejects_non_earliest_material_default_prior_pressure_checkpoint(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["trace_points"][0] = self.trace_point(
            simulation=1.0,
            q_margin=-0.02,
            selection_score_margin=0.06,
            selected_visits=1.0,
            reference_visits=1.0,
        )
        default_trace["trace_points"][1] = self.trace_point(
            simulation=2.0,
            q_margin=-0.01,
            selection_score_margin=0.07,
            selected_visits=2.0,
            reference_visits=1.0,
        )
        component_audit["first_material_checkpoints"]["default"]["prior_pressure"] = {
            "simulation": 2.0,
            "selection_score_margin": 0.07,
            "q_margin": -0.01,
        }

        with self.assertRaisesRegex(
            ValueError,
            "prior-pressure component audit upstream default prior_pressure checkpoint must match earliest material prior-pressure trace checkpoint",
        ):
            self.build_payload(
                component_audit=component_audit,
                metric_audit=metric_audit,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_keeps_residual_absent_when_q_margin_is_missing(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        self.replace_branch_prior_pressure_checkpoint(
            component_audit,
            selection_score_margin=0.06,
            q_margin=None,
        )
        self.replace_first_trace_point(
            default_trace,
            relaxed_trace,
            q_margin=None,
            selection_score_margin=0.06,
            selected_visits=1.0,
            reference_visits=1.0,
        )

        payload = self.build_payload(
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

        self.assertIsNone(payload["branch_level_evidence"]["default"]["selection_score_residual_margin"])
        self.assertEqual("prior_pressure_component_inconclusive", payload["classification"]["classification"])

    def test_build_payload_uses_float_tolerance_residual_threshold_without_upstream_override(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["thresholds_evaluated"] = {
            "meaningful_q": self.default_thresholds()["meaningful_q_margin"],
        }
        near_threshold_margin = self.default_thresholds()["material_selection_score_margin"] - (module.FLOAT_TOLERANCE / 2)
        self.replace_branch_prior_pressure_checkpoint(
            component_audit,
            selection_score_margin=near_threshold_margin,
            q_margin=-0.02,
        )
        self.replace_first_trace_point(
            default_trace,
            relaxed_trace,
            q_margin=-0.02,
            selection_score_margin=near_threshold_margin,
            selected_visits=1.0,
            reference_visits=1.0,
        )

        payload = self.build_payload(
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

        self.assertEqual(module.FLOAT_TOLERANCE, payload["branch_level_evidence"]["default"]["selection_score_residual_threshold"])
        self.assertEqual(module.FLOAT_TOLERANCE, payload["branch_level_evidence"]["relaxed"]["selection_score_residual_threshold"])
        self.assertEqual("selection_score_residual_lead", payload["classification"]["classification"])

    def test_build_payload_uses_stronger_upstream_residual_threshold_when_exposed(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        component_audit["thresholds_evaluated"] = {
            "selection_score": 0.08,
            "meaningful_q": self.default_thresholds()["meaningful_q_margin"],
        }

        payload = self.build_payload(
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

        self.assertEqual(0.08, payload["branch_level_evidence"]["default"]["selection_score_residual_threshold"])
        self.assertEqual(0.08, payload["branch_level_evidence"]["relaxed"]["selection_score_residual_threshold"])
        self.assertEqual(
            "prior_pressure_component_inconclusive",
            payload["classification"]["classification"],
        )

    def test_build_payload_accepts_upstream_checkpoint_with_float_tolerance_boundary(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        near_threshold_margin = self.default_thresholds()["material_selection_score_margin"] - (module.FLOAT_TOLERANCE / 2)
        self.replace_branch_prior_pressure_checkpoint(
            component_audit,
            selection_score_margin=near_threshold_margin,
            q_margin=-0.02,
        )
        self.replace_first_trace_point(
            default_trace,
            relaxed_trace,
            q_margin=-0.02,
            selection_score_margin=near_threshold_margin,
            selected_visits=1.0,
            reference_visits=1.0,
        )

        payload = self.build_payload(
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

        self.assertEqual(near_threshold_margin, payload["branch_level_evidence"]["default"]["selection_score_margin"])

    def test_build_payload_includes_checkpoint_audit(self):
        payload = self.build_payload()

        self.assertIn("checkpoint_audit", payload)
        self.assertEqual(2, len(payload["checkpoint_audit"]))
        self.assertEqual(1.0, payload["checkpoint_audit"][0]["simulation"])
        self.assertEqual(2.0, payload["checkpoint_audit"][1]["simulation"])

    def test_build_payload_uses_relaxed_trace_values_in_checkpoint_audit(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        relaxed_trace["trace_points"][1] = self.trace_point(
            simulation=2.0,
            q_margin=0.42,
            selection_score_margin=0.43,
            selected_visits=2.0,
            reference_visits=1.0,
        )

        payload = self.build_payload(
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

        self.assertEqual(0.07, payload["checkpoint_audit"][1]["default"]["selection_score_margin"])
        self.assertEqual(-0.01, payload["checkpoint_audit"][1]["default"]["q_margin"])
        self.assertEqual(0.43, payload["checkpoint_audit"][1]["relaxed"]["selection_score_margin"])
        self.assertEqual(0.42, payload["checkpoint_audit"][1]["relaxed"]["q_margin"])

    def test_build_payload_sets_prior_pressure_component_inconclusive_branch_candidate(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        self.replace_branch_prior_pressure_checkpoint(
            component_audit,
            selection_score_margin=0.05,
            q_margin=0.02,
        )
        self.replace_first_trace_point(
            default_trace,
            relaxed_trace,
            q_margin=0.02,
            selection_score_margin=0.05,
            selected_visits=1.0,
            reference_visits=2.0,
        )

        payload = self.build_payload(
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

        self.assertEqual(
            "prior_pressure_component_inconclusive",
            payload["branch_level_evidence"]["default"]["explanation_candidate"],
        )
        self.assertEqual(
            "prior_pressure_component_inconclusive",
            payload["branch_level_evidence"]["relaxed"]["explanation_candidate"],
        )

    def test_build_payload_sets_overall_classification_selection_score_residual_lead(self):
        payload = self.build_payload()

        self.assertEqual(
            "selection_score_residual_lead",
            payload["classification"]["classification"],
        )
        self.assertEqual(
            module.CLASSIFICATION_DECISIONS["selection_score_residual_lead"],
            payload["decision"],
        )
        self.assertEqual(
            {
                "default_explanation_candidate": "selection_score_residual_lead",
                "relaxed_explanation_candidate": "selection_score_residual_lead",
                "branches_disagree": False,
                "residual_vs_visit_material_conflict": False,
                "material_conflict_branches": [],
            },
            payload["branch_disagreement_summary"],
        )

    def test_build_payload_sets_overall_classification_visit_alignment_pressure(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        self.replace_branch_prior_pressure_checkpoint(
            component_audit,
            selection_score_margin=0.06,
            q_margin=0.02,
        )
        self.replace_first_trace_point(
            default_trace,
            relaxed_trace,
            q_margin=0.02,
            selection_score_margin=0.06,
            selected_visits=4.0,
            reference_visits=1.0,
        )

        payload = self.build_payload(
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

        self.assertEqual("visit_alignment_pressure", payload["classification"]["classification"])
        self.assertEqual(
            module.CLASSIFICATION_DECISIONS["visit_alignment_pressure"],
            payload["decision"],
        )
        self.assertEqual(
            {
                "default_explanation_candidate": "visit_alignment_pressure",
                "relaxed_explanation_candidate": "visit_alignment_pressure",
                "branches_disagree": False,
                "residual_vs_visit_material_conflict": False,
                "material_conflict_branches": [],
            },
            payload["branch_disagreement_summary"],
        )

    def test_build_payload_sets_overall_classification_mixed_when_branches_disagree(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        self.replace_branch_prior_pressure_checkpoint_for_branch(
            component_audit,
            branch="relaxed",
            selection_score_margin=0.06,
            q_margin=0.02,
        )
        self.replace_first_trace_point_for_branch(
            relaxed_trace,
            q_margin=0.02,
            selection_score_margin=0.06,
            selected_visits=4.0,
            reference_visits=1.0,
        )

        payload = self.build_payload(
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

        self.assertEqual("selection_score_residual_lead", payload["branch_level_evidence"]["default"]["explanation_candidate"])
        self.assertEqual("visit_alignment_pressure", payload["branch_level_evidence"]["relaxed"]["explanation_candidate"])
        self.assertEqual("mixed_prior_pressure_signal", payload["classification"]["classification"])
        self.assertEqual(
            module.CLASSIFICATION_DECISIONS["mixed_prior_pressure_signal"],
            payload["decision"],
        )
        self.assertEqual(
            {
                "default_explanation_candidate": "selection_score_residual_lead",
                "relaxed_explanation_candidate": "visit_alignment_pressure",
                "branches_disagree": True,
                "residual_vs_visit_material_conflict": False,
                "material_conflict_branches": [],
            },
            payload["branch_disagreement_summary"],
        )

    def test_build_payload_sets_overall_classification_mixed_when_residual_and_visit_conflict_materially(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        self.replace_branch_prior_pressure_checkpoint(
            component_audit,
            selection_score_margin=0.20,
            q_margin=0.02,
        )
        self.replace_first_trace_point(
            default_trace,
            relaxed_trace,
            q_margin=0.02,
            selection_score_margin=0.20,
            selected_visits=4.0,
            reference_visits=1.0,
        )

        payload = self.build_payload(
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

        self.assertEqual("mixed_prior_pressure_signal", payload["branch_level_evidence"]["default"]["explanation_candidate"])
        self.assertEqual("mixed_prior_pressure_signal", payload["branch_level_evidence"]["relaxed"]["explanation_candidate"])
        self.assertEqual("mixed_prior_pressure_signal", payload["classification"]["classification"])
        self.assertEqual(
            module.CLASSIFICATION_DECISIONS["mixed_prior_pressure_signal"],
            payload["decision"],
        )
        self.assertEqual(
            {
                "default_explanation_candidate": "mixed_prior_pressure_signal",
                "relaxed_explanation_candidate": "mixed_prior_pressure_signal",
                "branches_disagree": False,
                "residual_vs_visit_material_conflict": True,
                "material_conflict_branches": ["default", "relaxed"],
            },
            payload["branch_disagreement_summary"],
        )

    def test_build_payload_sets_overall_classification_inconclusive(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        self.replace_branch_prior_pressure_checkpoint(
            component_audit,
            selection_score_margin=0.05,
            q_margin=0.02,
        )
        self.replace_first_trace_point(
            default_trace,
            relaxed_trace,
            q_margin=0.02,
            selection_score_margin=0.05,
            selected_visits=1.0,
            reference_visits=2.0,
        )

        payload = self.build_payload(
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

        self.assertEqual(
            "prior_pressure_component_inconclusive",
            payload["classification"]["classification"],
        )
        self.assertEqual(
            module.CLASSIFICATION_DECISIONS["prior_pressure_component_inconclusive"],
            payload["decision"],
        )
        self.assertEqual(
            {
                "default_explanation_candidate": "prior_pressure_component_inconclusive",
                "relaxed_explanation_candidate": "prior_pressure_component_inconclusive",
                "branches_disagree": False,
                "residual_vs_visit_material_conflict": False,
                "material_conflict_branches": [],
            },
            payload["branch_disagreement_summary"],
        )

    def test_build_payload_accepts_valid_prior_pressure_chain(self):
        payload = self.build_payload()

        self.assertEqual(module.SCHEMA, payload["schema"])
        self.assertEqual(
            "selection_score_residual_lead",
            payload["classification"]["classification"],
        )
        self.assertEqual(module.CLASSIFICATION_DECISIONS["selection_score_residual_lead"], payload["decision"])
        self.assertEqual(
            {
                "source_selection_score_component_audit_artifact_path": "/tmp/selection_score_component_audit.json",
                "source_metric_audit_artifact_path": "/tmp/metric_audit.json",
                "source_selection_score_artifact_path": "/tmp/default.json",
                "source_threshold_relaxed_selection_score_artifact_path": "/tmp/relaxed.json",
            },
            payload["input_artifacts"],
        )
        self.assertEqual(self.source_artifact_with_provenance(), payload["source_artifact"])

    def test_build_payload_includes_final_payload_metadata(self):
        payload = self.build_payload()

        self.assertEqual("prior_pressure_component_audit", payload["hypothesis"])
        self.assertEqual(
            {
                "default": self.default_thresholds(),
                "relaxed": self.relaxed_thresholds(),
            },
            payload["thresholds_evaluated"],
        )
        self.assertEqual(
            {
                "component_audit_classification": {"classification": "prior_pressure_lead", "evidence_summary": "prior-pressure lead"},
                "metric_audit_classification": {"classification": "early_selection_score_only", "evidence_summary": "selection-score lead"},
                "default_trace_classification": {"classification": "unresolved", "evidence_summary": "unresolved"},
                "relaxed_trace_classification": {"classification": "unresolved", "evidence_summary": "unresolved"},
                "default_trace_origin": "extracted",
                "relaxed_trace_origin": "extracted",
            },
            payload["source_snapshots"],
        )

    def test_build_payload_rejects_malformed_trace_origin_used_by_source_snapshots(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["trace_origin"] = ""

        self.assert_build_payload_error(
            "default trace artifact trace_origin must be a non-empty string",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_empty_trace_points_before_payload_metadata_assembly(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["trace_points"] = []

        self.assert_build_payload_error(
            "default trace artifact trace_points must be a non-empty list",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

    def test_build_payload_rejects_malformed_thresholds_metadata_used_by_final_payload(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["thresholds"]["meaningful_q_margin"] = "invalid"

        self.assert_build_payload_error(
            "default branch evidence thresholds.meaningful_q_margin must be a finite non-negative number",
            component_audit=component_audit,
            metric_audit=metric_audit,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )


class Capture002PriorPressureComponentAuditBuildPayloadTest(
    Capture002PriorPressureComponentAuditTestSupport, unittest.TestCase
):
    pass


class Capture002PriorPressureComponentAuditCliTest(
    Capture002PriorPressureComponentAuditTestSupport, unittest.TestCase
):
    def test_main_writes_sorted_json_and_prints_compact_summary_json(self):
        component_audit, metric_audit, default_trace, relaxed_trace = self.valid_inputs()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            component_path = tmpdir_path / "selection_score_component_audit.json"
            metric_path = tmpdir_path / "metric_audit.json"
            default_path = tmpdir_path / "default.json"
            relaxed_path = tmpdir_path / "relaxed.json"
            out_path = tmpdir_path / "artifacts" / "prior_pressure_component_audit.json"

            component_audit = copy.deepcopy(component_audit)
            metric_audit = copy.deepcopy(metric_audit)
            component_audit["input_artifacts"]["source_metric_audit_artifact_path"] = str(metric_path)
            component_audit["input_artifacts"]["source_selection_score_artifact_path"] = str(default_path)
            component_audit["input_artifacts"]["source_threshold_relaxed_selection_score_artifact_path"] = str(relaxed_path)
            metric_audit["input_artifacts"]["source_selection_score_artifact_path"] = str(default_path)
            metric_audit["input_artifacts"]["source_threshold_relaxed_selection_score_artifact_path"] = str(relaxed_path)

            component_path.write_text(json.dumps(component_audit), encoding="utf-8")
            metric_path.write_text(json.dumps(metric_audit), encoding="utf-8")
            default_path.write_text(json.dumps(default_trace), encoding="utf-8")
            relaxed_path.write_text(json.dumps(relaxed_trace), encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = module.main(
                    [
                        "--source-selection-score-component-audit-artifact",
                        str(component_path),
                        "--source-metric-audit-artifact",
                        str(metric_path),
                        "--source-selection-score-artifact",
                        str(default_path),
                        "--source-threshold-relaxed-selection-score-artifact",
                        str(relaxed_path),
                        "--out",
                        str(out_path),
                    ]
                )

            self.assertEqual(0, result)
            expected_payload = module.build_payload(
                component_audit,
                metric_audit,
                default_trace,
                relaxed_trace,
                source_selection_score_component_audit_artifact_path=str(component_path),
                source_metric_audit_artifact_path=str(metric_path),
                source_selection_score_artifact_path=str(default_path),
                source_threshold_relaxed_selection_score_artifact_path=str(relaxed_path),
            )
            self.assertEqual(
                json.dumps(expected_payload, indent=2, sort_keys=True) + "\n",
                out_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                json.dumps(
                    {
                        "artifact_path": str(out_path),
                        "schema": expected_payload["schema"],
                        "classification": expected_payload["classification"]["classification"],
                        "decision": expected_payload["decision"],
                    }
                )
                + "\n",
                stdout.getvalue(),
            )


for _name in unittest.defaultTestLoader.getTestCaseNames(Capture002PriorPressureComponentAuditBuildPayloadTest):
    if _name != "test_main_writes_sorted_json_and_prints_compact_summary_json":
        setattr(Capture002PriorPressureComponentAuditCliTest, _name, None)
