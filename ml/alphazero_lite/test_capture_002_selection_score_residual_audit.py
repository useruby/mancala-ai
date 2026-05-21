import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ml.alphazero_lite import capture_002_selection_score_residual_audit as module


class Capture002SelectionScoreResidualAuditContractTest(unittest.TestCase):
    def test_contract_constants_are_stable(self):
        self.assertEqual(
            "azlite_capture_002_selection_score_residual_audit_v1", module.SCHEMA
        )
        self.assertEqual(
            "azlite_capture_002_prior_pressure_component_audit_v1",
            module.SOURCE_PRIOR_PRESSURE_AUDIT_SCHEMA,
        )
        self.assertEqual(
            "azlite_capture_002_selection_score_trace_v1",
            module.SOURCE_SELECTION_SCORE_SCHEMA,
        )
        self.assertEqual(
            "azlite_capture_002_trace_checkpoint_canonicalization_v1",
            module.SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA,
        )
        self.assertEqual("capture_available-002", module.ROW_ID)
        self.assertEqual(
            "selection_score_residual_lead",
            module.EXPECTED_PRIOR_PRESSURE_CLASSIFICATION,
        )
        self.assertEqual(
            "write_002_selection_score_residual_spec",
            module.EXPECTED_PRIOR_PRESSURE_DECISION,
        )
        self.assertEqual(2.0, module.EXPECTED_CANONICAL_SIMULATION)
        self.assertEqual("unresolved", module.EXPECTED_TRACE_CLASSIFICATION)
        self.assertEqual(
            "write_002_unresolved_trace_review_spec",
            module.EXPECTED_TRACE_DECISION,
        )
        self.assertEqual(
            "write_002_metric_audit_canonical_input_spec",
            module.EXPECTED_CANONICALIZATION_DECISION,
        )
        self.assertEqual(1e-12, module.FLOAT_TOLERANCE)
        self.assertEqual(
            {
                "stable_selected_residual_advantage": "write_002_residual_ablation_spec",
                "reference_or_other_move_residual_competes": "write_002_competing_residual_review_spec",
                "tiny_count_residual_ambiguous": "write_002_tiny_count_residual_review_spec",
                "selection_score_residual_inconclusive": "stop_002_selection_score_residual_inconclusive",
            },
            module.CLASSIFICATION_DECISIONS,
        )

    def test_parse_args_reads_required_paths(self):
        args = module.parse_args(
            [
                "--source-prior-pressure-audit-artifact",
                "/tmp/prior_pressure_audit.json",
                "--source-selection-score-artifact",
                "/tmp/default.json",
                "--source-threshold-relaxed-selection-score-artifact",
                "/tmp/relaxed.json",
                "--out",
                "/tmp/out.json",
            ]
        )

        self.assertEqual(
            Path("/tmp/prior_pressure_audit.json"),
            args.source_prior_pressure_audit_artifact,
        )
        self.assertEqual(
            Path("/tmp/default.json"), args.source_selection_score_artifact
        )
        self.assertEqual(
            Path("/tmp/relaxed.json"),
            args.source_threshold_relaxed_selection_score_artifact,
        )
        self.assertIsNone(args.source_checkpoint_canonicalization_artifact)
        self.assertEqual(Path("/tmp/out.json"), args.out)

    def test_parse_args_accepts_optional_checkpoint_canonicalization_path(self):
        args = module.parse_args(
            [
                "--source-prior-pressure-audit-artifact",
                "/tmp/prior_pressure_audit.json",
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


class Capture002SelectionScoreResidualAuditTestSupport:
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
        self, *, simulation: float, moves: list[dict], visits: list[float]
    ) -> dict:
        return {
            "simulation": simulation,
            "selected_move": 0,
            "reference_move_by_prior": 2,
            "moves": copy.deepcopy(moves),
            "visits": copy.deepcopy(visits),
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
            "classification": {
                "classification": "unresolved",
                "evidence_summary": "unresolved",
            },
            "decision": "write_002_unresolved_trace_review_spec",
            "insufficiency_reasons": [],
            "trace_origin": trace_origin,
            "source_artifact": copy.deepcopy(
                source_artifact or self.source_artifact_with_provenance()
            ),
            "thresholds": copy.deepcopy(thresholds),
            "trace_points": copy.deepcopy(trace_points),
        }

    def canonicalization_artifact(
        self,
        *,
        canonical_trace_point: dict | None = None,
        decision: str | None = None,
        input_artifacts: dict | None = None,
        source_artifact: dict | None = None,
        source_path: str | None = None,
        thresholds_evaluated: dict | None = None,
        trace_origin: str | None = None,
        canonicalization_status: dict | None = None,
        canonical_sequences_match: bool = True,
    ) -> dict:
        canonical_trace_point = copy.deepcopy(
            canonical_trace_point
            or self.trace_point(
                simulation=2.0,
                moves=[
                    {"move": 0, "selection_score": 0.07, "q_value": -0.01},
                    {"move": 2, "selection_score": 0.0, "q_value": 0.0},
                ],
                visits=[2.0, 0.0, 1.0, 0.0, 0.0],
            )
        )
        artifact = {
            "schema": module.SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA,
            "decision": decision or module.EXPECTED_CANONICALIZATION_DECISION,
            "input_artifacts": copy.deepcopy(
                input_artifacts
                or {
                    "source_selection_score_artifact_path": "/tmp/default.json",
                    "source_threshold_relaxed_selection_score_artifact_path": "/tmp/relaxed.json",
                }
            ),
            "canonicalization_status": copy.deepcopy(
                canonicalization_status or {"safe_for_followup_spec": True}
            ),
            "canonical_sequences_match": canonical_sequences_match,
            "canonical_checkpoint_sequences": {
                "default": [1.0, 2.0],
                "relaxed": [1.0, 2.0],
            },
            "thresholds_evaluated": copy.deepcopy(
                thresholds_evaluated
                or {
                    "default": self.default_thresholds(),
                    "relaxed": self.relaxed_thresholds(),
                }
            ),
            "trace_origin": trace_origin or "extracted",
            "source_artifact": copy.deepcopy(
                source_artifact or self.source_artifact_with_provenance()
            ),
        }
        if source_path is not None:
            artifact["source_path"] = source_path
        return artifact

    def prior_pressure_artifact(self, *, source_artifact: dict | None = None) -> dict:
        source_artifact = copy.deepcopy(
            source_artifact or self.source_artifact_with_provenance()
        )
        return {
            "schema": module.SOURCE_PRIOR_PRESSURE_AUDIT_SCHEMA,
            "hypothesis": "prior_pressure_component_audit",
            "classification": {
                "classification": module.EXPECTED_PRIOR_PRESSURE_CLASSIFICATION,
                "evidence_summary": "selection-score residual lead",
            },
            "decision": module.EXPECTED_PRIOR_PRESSURE_DECISION,
            "input_artifacts": {
                "source_selection_score_component_audit_artifact_path": "/tmp/selection_score_component_audit.json",
                "source_metric_audit_artifact_path": "/tmp/metric_audit.json",
                "source_selection_score_artifact_path": "/tmp/default.json",
                "source_threshold_relaxed_selection_score_artifact_path": "/tmp/relaxed.json",
            },
            "source_artifact": source_artifact,
            "thresholds_evaluated": {
                "selection_score": self.default_thresholds()[
                    "material_selection_score_margin"
                ],
                "meaningful_q": self.default_thresholds()["meaningful_q_margin"],
            },
            "source_snapshots": {
                "metric_audit_classification": {
                    "classification": "early_selection_score_only"
                },
                "default_trace_classification": {"classification": "unresolved"},
                "relaxed_trace_classification": {"classification": "unresolved"},
                "default_trace_origin": "extracted",
                "relaxed_trace_origin": "extracted",
            },
            "branch_level_evidence": {
                "default": {
                    "selection_score_residual_threshold": 0.05,
                    "upstream_checkpoint_echo": {
                        "simulation": 2.0,
                        "selection_score_margin": 0.07,
                        "q_margin": -0.01,
                    },
                },
                "relaxed": {
                    "selection_score_residual_threshold": 0.05,
                    "upstream_checkpoint_echo": {
                        "simulation": 2.0,
                        "selection_score_margin": 0.07,
                        "q_margin": -0.01,
                    },
                },
            },
        }

    def valid_inputs(self) -> tuple[dict, dict, dict]:
        trace_points = [
            self.trace_point(
                simulation=1.0,
                moves=[
                    {"move": 0, "selection_score": 0.06, "q_value": -0.02},
                    {"move": 2, "selection_score": 0.0, "q_value": 0.0},
                ],
                visits=[1.0, 0.0, 1.0, 0.0, 0.0],
            ),
            self.trace_point(
                simulation=2.0,
                moves=[
                    {"move": 0, "selection_score": 0.07, "q_value": -0.01},
                    {"move": 2, "selection_score": 0.0, "q_value": 0.0},
                ],
                visits=[2.0, 0.0, 1.0, 0.0, 0.0],
            ),
        ]
        source_artifact = self.source_artifact_with_provenance()
        prior_pressure = self.prior_pressure_artifact(source_artifact=source_artifact)
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
        return prior_pressure, default_trace, relaxed_trace


class Capture002SelectionScoreResidualAuditBuildPayloadHappyPathTest(
    Capture002SelectionScoreResidualAuditTestSupport,
    unittest.TestCase,
):
    def test_build_payload_returns_happy_path_payload_skeleton(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()

        payload = module.build_payload(
            prior_pressure,
            default_trace,
            relaxed_trace,
            source_prior_pressure_audit_artifact_path="/tmp/prior_pressure_audit.json",
            source_selection_score_artifact_path="/tmp/default.json",
            source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
        )

        self.assertEqual(module.SCHEMA, payload["schema"])
        self.assertEqual("selection_score_residual_audit", payload["hypothesis"])
        self.assertEqual(
            "stable selected residual advantage",
            payload["classification"]["evidence_summary"],
        )
        self.assertEqual("capture_available-002", payload["source_artifact"]["row_id"])

    def test_build_payload_includes_required_contract_fields(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()

        payload = module.build_payload(
            prior_pressure,
            default_trace,
            relaxed_trace,
            source_prior_pressure_audit_artifact_path="/tmp/prior_pressure_audit.json",
            source_selection_score_artifact_path="/tmp/default.json",
            source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
        )

        self.assertEqual(
            {
                "default": {
                    **self.default_thresholds(),
                    "selection_score_residual_threshold": 0.05,
                },
                "relaxed": {
                    **self.relaxed_thresholds(),
                    "selection_score_residual_threshold": 0.05,
                },
                "float_tolerance": module.FLOAT_TOLERANCE,
            },
            payload["thresholds_evaluated"],
        )
        self.assertEqual(
            {
                "canonical_simulation": 2.0,
                "default_upstream_checkpoint_echo": prior_pressure[
                    "branch_level_evidence"
                ]["default"]["upstream_checkpoint_echo"],
                "relaxed_upstream_checkpoint_echo": prior_pressure[
                    "branch_level_evidence"
                ]["relaxed"]["upstream_checkpoint_echo"],
            },
            payload["checkpoint"],
        )
        self.assertEqual(
            {
                "default": {
                    "canonical_simulation": 2.0,
                    "raw_match_count": 1,
                    "all_raw_matches_canonical": True,
                },
                "relaxed": {
                    "canonical_simulation": 2.0,
                    "raw_match_count": 1,
                    "all_raw_matches_canonical": True,
                },
            },
            payload["duplicate_equivalence_audit"],
        )
        self.assertEqual(
            {
                "all_branches_agree": True,
                "default_branch_candidate": "stable_selected_residual_advantage",
                "relaxed_branch_candidate": "stable_selected_residual_advantage",
            },
            payload["branch_disagreement_summary"],
        )
        self.assertEqual(
            [0, 2],
            payload["branch_residual_evidence"]["default"]["legal_moves"],
        )
        self.assertEqual(
            [],
            payload["branch_residual_evidence"]["default"][
                "missing_residual_inputs_moves"
            ],
        )
        self.assertEqual(
            [],
            payload["branch_residual_evidence"]["default"]["competing_moves"],
        )


class Capture002SelectionScoreResidualAuditValidationTest(
    Capture002SelectionScoreResidualAuditTestSupport,
    unittest.TestCase,
):
    def build_payload(
        self,
        *,
        prior_pressure=None,
        default_trace=None,
        relaxed_trace=None,
        canonicalization=None,
        canonicalization_path=None,
    ):
        valid_prior_pressure, valid_default_trace, valid_relaxed_trace = (
            self.valid_inputs()
        )
        return module.build_payload(
            prior_pressure or valid_prior_pressure,
            default_trace or valid_default_trace,
            relaxed_trace or valid_relaxed_trace,
            source_prior_pressure_audit_artifact_path="/tmp/prior_pressure_audit.json",
            source_selection_score_artifact_path="/tmp/default.json",
            source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
            checkpoint_canonicalization_artifact=canonicalization,
            source_checkpoint_canonicalization_artifact_path=canonicalization_path,
        )

    def test_build_payload_rejects_wrong_prior_pressure_schema(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["schema"] = "wrong"

        with self.assertRaisesRegex(
            ValueError, "prior-pressure audit artifact has wrong schema"
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_non_object_prior_pressure_artifact(self):
        _, default_trace, relaxed_trace = self.valid_inputs()

        with self.assertRaisesRegex(
            ValueError, "prior-pressure audit artifact must be an object"
        ):
            self.build_payload(
                prior_pressure="not-a-dict",
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_prior_pressure_source_artifact_wrong_row_id(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["source_artifact"]["row_id"] = "capture_available-999"

        with self.assertRaisesRegex(
            ValueError,
            f"prior-pressure audit artifact source_artifact.row_id must be {module.ROW_ID}",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_boolean_prior_pressure_source_artifact_selected_move(
        self,
    ):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["source_artifact"]["full_search_selected_move"] = True

        with self.assertRaisesRegex(
            ValueError,
            "prior-pressure audit artifact source_artifact.full_search_selected_move must be an integer",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_wrong_prior_pressure_classification(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["classification"]["classification"] = "wrong"

        with self.assertRaisesRegex(
            ValueError,
            f"prior-pressure audit artifact classification must be {module.EXPECTED_PRIOR_PRESSURE_CLASSIFICATION}",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_wrong_prior_pressure_decision(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["decision"] = "wrong"

        with self.assertRaisesRegex(
            ValueError,
            f"prior-pressure audit artifact decision must be {module.EXPECTED_PRIOR_PRESSURE_DECISION}",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_missing_branch_threshold(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        del prior_pressure["branch_level_evidence"]["default"][
            "selection_score_residual_threshold"
        ]

        with self.assertRaisesRegex(
            ValueError,
            "prior-pressure audit branch_level_evidence.default.selection_score_residual_threshold must be finite numeric",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_non_canonical_upstream_checkpoint(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["branch_level_evidence"]["default"]["upstream_checkpoint_echo"][
            "simulation"
        ] = 1.0

        with self.assertRaisesRegex(
            ValueError,
            "prior-pressure audit default upstream checkpoint simulation must be 2.0",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_mismatched_duplicate_canonical_projection(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        duplicate_default_point = copy.deepcopy(default_trace["trace_points"][1])
        duplicate_default_point["moves"][0]["selection_score"] = 0.08
        default_trace["trace_points"].append(duplicate_default_point)
        prior_pressure["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"

        with self.assertRaisesRegex(
            ValueError,
            "default raw duplicate 2.0 checkpoint must match canonical projection",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
                canonicalization=self.canonicalization_artifact(),
                canonicalization_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_conflicting_duplicate_canonical_projection_without_canonicalization(
        self,
    ):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        duplicate_default_point = copy.deepcopy(default_trace["trace_points"][1])
        duplicate_default_point["moves"][0]["selection_score"] = 0.08
        default_trace["trace_points"].append(duplicate_default_point)

        with self.assertRaisesRegex(
            ValueError,
            "default raw duplicate 2.0 checkpoint requires canonicalization artifact",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_requires_canonicalization_artifact_when_path_supplied(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        duplicate_default_point = copy.deepcopy(default_trace["trace_points"][1])
        default_trace["trace_points"].append(duplicate_default_point)
        prior_pressure["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"

        with self.assertRaisesRegex(
            ValueError,
            "canonicalization path requires checkpoint_canonicalization_artifact",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
                canonicalization_path="/tmp/canonicalization.json",
            )

    def test_build_payload_requires_canonicalization_path_when_artifact_supplied(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint_canonicalization_artifact requires canonicalization path",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
                canonicalization=self.canonicalization_artifact(),
            )

    def test_build_payload_requires_prior_pressure_canonicalization_path_in_canonical_mode(
        self,
    ):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()

        with self.assertRaisesRegex(
            ValueError,
            "prior-pressure audit upstream input_artifacts source_checkpoint_canonicalization_artifact_path must match source path",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
                canonicalization=self.canonicalization_artifact(),
                canonicalization_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_upstream_checkpoint_mismatch_against_canonical_trace(
        self,
    ):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["branch_level_evidence"]["default"]["upstream_checkpoint_echo"][
            "selection_score_margin"
        ] = 0.08

        with self.assertRaisesRegex(
            ValueError,
            "prior-pressure audit default upstream checkpoint must match canonical 2.0 trace point",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_accepts_upstream_checkpoint_tiny_float_noise(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["branch_level_evidence"]["default"]["upstream_checkpoint_echo"][
            "selection_score_margin"
        ] += module.FLOAT_TOLERANCE / 2
        prior_pressure["branch_level_evidence"]["default"]["upstream_checkpoint_echo"][
            "q_margin"
        ] -= module.FLOAT_TOLERANCE / 2

        payload = self.build_payload(
            prior_pressure=prior_pressure,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

        self.assertEqual(module.SCHEMA, payload["schema"])

    def test_build_payload_accepts_numeric_equivalent_duplicate_canonical_projection(
        self,
    ):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        duplicate_default_point = copy.deepcopy(default_trace["trace_points"][1])
        duplicate_default_point["simulation"] = 2
        duplicate_default_point["visits"] = [2, 0, 1, 0, 0]
        duplicate_default_point["moves"][0]["selection_score"] = 0.07
        duplicate_default_point["moves"][0]["q_value"] = -0.01
        duplicate_default_point["moves"][1]["selection_score"] = 0
        duplicate_default_point["moves"][1]["q_value"] = 0
        default_trace["trace_points"].append(duplicate_default_point)
        prior_pressure["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"

        payload = self.build_payload(
            prior_pressure=prior_pressure,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
            canonicalization=self.canonicalization_artifact(),
            canonicalization_path="/tmp/canonicalization.json",
        )

        self.assertEqual(module.SCHEMA, payload["schema"])

    def test_build_payload_rejects_non_object_default_trace_artifact(self):
        prior_pressure, _, relaxed_trace = self.valid_inputs()

        with self.assertRaisesRegex(
            ValueError, "default trace artifact must be an object"
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace="not-a-dict",
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_wrong_relaxed_trace_schema(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        relaxed_trace["schema"] = "wrong"

        with self.assertRaisesRegex(
            ValueError,
            f"relaxed trace artifact has wrong schema; expected {module.SOURCE_SELECTION_SCORE_SCHEMA}",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_wrong_default_trace_classification(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["classification"]["classification"] = "wrong"

        with self.assertRaisesRegex(
            ValueError,
            f"default trace artifact classification must be {module.EXPECTED_TRACE_CLASSIFICATION}",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_missing_default_trace_trace_origin(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        del default_trace["trace_origin"]

        with self.assertRaisesRegex(
            ValueError,
            "default trace artifact trace_origin must be a non-empty string",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_blank_default_trace_trace_origin(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["trace_origin"] = "   "

        with self.assertRaisesRegex(
            ValueError,
            "default trace artifact trace_origin must be a non-empty string",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_non_object_default_trace_thresholds(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["thresholds"] = "not-an-object"

        with self.assertRaisesRegex(
            ValueError, "default trace artifact thresholds must be an object"
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_non_numeric_relaxed_trace_material_selection_score_margin(
        self,
    ):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        relaxed_trace["thresholds"]["material_selection_score_margin"] = "wrong"

        with self.assertRaisesRegex(
            ValueError,
            "relaxed trace artifact thresholds.material_selection_score_margin must be finite non-negative numeric",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_wrong_relaxed_trace_decision(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        relaxed_trace["decision"] = "wrong"

        with self.assertRaisesRegex(
            ValueError,
            f"relaxed trace artifact decision must be {module.EXPECTED_TRACE_DECISION}",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_wrong_canonicalization_decision_when_present(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"

        with self.assertRaisesRegex(
            ValueError,
            f"checkpoint canonicalization artifact decision must be {module.EXPECTED_CANONICALIZATION_DECISION}",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
                canonicalization=self.canonicalization_artifact(decision="wrong"),
                canonicalization_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_missing_canonicalization_decision(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"
        canonicalization = self.canonicalization_artifact()
        del canonicalization["decision"]

        with self.assertRaisesRegex(
            ValueError,
            f"checkpoint canonicalization artifact decision must be {module.EXPECTED_CANONICALIZATION_DECISION}",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
                canonicalization=canonicalization,
                canonicalization_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_canonicalization_artifact_input_path_mismatch(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint canonicalization artifact input_artifacts source_selection_score_artifact_path must match source path",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
                canonicalization=self.canonicalization_artifact(
                    input_artifacts={
                        "source_selection_score_artifact_path": "/tmp/other-default.json",
                        "source_threshold_relaxed_selection_score_artifact_path": "/tmp/relaxed.json",
                    }
                ),
                canonicalization_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_canonicalization_artifact_source_artifact_mismatch(
        self,
    ):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"
        mismatched_source_artifact = self.source_artifact_with_provenance()
        mismatched_source_artifact["artifact_path"] = "/tmp/source/other-upstream.json"

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint canonicalization artifact source_artifact must match prior-pressure audit artifact source_artifact",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
                canonicalization=self.canonicalization_artifact(
                    source_artifact=mismatched_source_artifact
                ),
                canonicalization_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_canonicalization_thresholds_evaluated_mismatch(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint canonicalization artifact thresholds_evaluated.default must match validated trace thresholds",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
                canonicalization={
                    **self.canonicalization_artifact(),
                    "thresholds_evaluated": {
                        "default": {
                            **self.default_thresholds(),
                            "material_selection_score_margin": 0.99,
                        },
                        "relaxed": self.relaxed_thresholds(),
                    },
                    "trace_origin": "extracted",
                },
                canonicalization_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_missing_canonicalization_thresholds_evaluated(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"
        canonicalization = self.canonicalization_artifact()
        del canonicalization["thresholds_evaluated"]

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint canonicalization artifact thresholds_evaluated must be an object",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
                canonicalization=canonicalization,
                canonicalization_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_canonicalization_trace_origin_mismatch(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint canonicalization artifact trace_origin must match trace artifacts",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
                canonicalization={
                    **self.canonicalization_artifact(),
                    "thresholds_evaluated": {
                        "default": self.default_thresholds(),
                        "relaxed": self.relaxed_thresholds(),
                    },
                    "trace_origin": "other-origin",
                },
                canonicalization_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_missing_canonicalization_trace_origin(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"
        canonicalization = self.canonicalization_artifact()
        del canonicalization["trace_origin"]

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint canonicalization artifact trace_origin must match trace artifacts",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
                canonicalization=canonicalization,
                canonicalization_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_missing_canonicalization_safe_for_followup_spec(
        self,
    ):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"
        canonicalization = self.canonicalization_artifact()
        del canonicalization["canonicalization_status"]["safe_for_followup_spec"]

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint canonicalization artifact canonicalization_status.safe_for_followup_spec must be true",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
                canonicalization=canonicalization,
                canonicalization_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_false_canonicalization_safe_for_followup_spec(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint canonicalization artifact canonicalization_status.safe_for_followup_spec must be true",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
                canonicalization=self.canonicalization_artifact(
                    canonicalization_status={"safe_for_followup_spec": False}
                ),
                canonicalization_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_malformed_canonicalization_artifact_shape_in_canonical_mode(
        self,
    ):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint canonicalization artifact canonical_checkpoint_sequences must be an object",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
                canonicalization={
                    **self.canonicalization_artifact(),
                    "canonical_checkpoint_sequences": "wrong",
                },
                canonicalization_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_mismatched_source_artifact_between_prior_pressure_and_default_trace(
        self,
    ):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["source_artifact"]["artifact_path"] = "/tmp/source/other.json"

        with self.assertRaisesRegex(
            ValueError,
            "default trace artifact source_artifact must match prior-pressure audit artifact source_artifact",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_uses_source_artifact_move_ids_when_canonical_trace_point_ids_differ(
        self,
    ):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        for trace_artifact in (default_trace, relaxed_trace):
            trace_artifact["trace_points"][1]["selected_move"] = 2
            trace_artifact["trace_points"][1]["reference_move_by_prior"] = 0

        payload = self.build_payload(
            prior_pressure=prior_pressure,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

        self.assertEqual(
            0, payload["branch_residual_evidence"]["default"]["selected_move"]
        )
        self.assertEqual(
            2, payload["branch_residual_evidence"]["default"]["reference_move"]
        )
        self.assertEqual(
            0.07, payload["branch_residual_evidence"]["default"]["selected_residual"]
        )
        self.assertEqual(
            "stable_selected_residual_advantage",
            payload["classification"]["classification"],
        )

    def test_build_payload_rejects_when_source_artifact_selected_move_missing_from_canonical_trace_moves(
        self,
    ):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["source_artifact"]["full_search_selected_move"] = 1
        default_trace["source_artifact"]["full_search_selected_move"] = 1
        relaxed_trace["source_artifact"]["full_search_selected_move"] = 1

        with self.assertRaisesRegex(
            ValueError,
            "default canonical trace point.moves must include move 1",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_mismatched_prior_pressure_default_input_path(self):
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["input_artifacts"]["source_selection_score_artifact_path"] = (
            "/tmp/other-default.json"
        )

        with self.assertRaisesRegex(
            ValueError,
            "prior-pressure audit upstream input_artifacts source_selection_score_artifact_path must match source path",
        ):
            self.build_payload(
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )


class Capture002SelectionScoreResidualAuditBranchClassificationTest(
    Capture002SelectionScoreResidualAuditTestSupport,
    unittest.TestCase,
):
    def canonical_trace_point(
        self,
        *,
        selected_selection_score: float,
        selected_q_value: float,
        reference_selection_score: float,
        reference_q_value: float,
        visits: list[float],
        extra_moves: list[dict] | None = None,
    ) -> dict:
        moves = [
            {
                "move": 0,
                "selection_score": selected_selection_score,
                "q_value": selected_q_value,
            },
            {
                "move": 2,
                "selection_score": reference_selection_score,
                "q_value": reference_q_value,
            },
        ]
        if extra_moves:
            moves.extend(copy.deepcopy(extra_moves))
        return self.trace_point(
            simulation=2.0,
            moves=moves,
            visits=visits,
        )

    def build_payload_for_canonical_trace_point(self, trace_point: dict):
        source_artifact = self.source_artifact_with_provenance()
        prior_pressure = self.prior_pressure_artifact(source_artifact=source_artifact)
        checkpoint_echo = {
            "simulation": 2.0,
            "selection_score_margin": (
                trace_point["moves"][0]["selection_score"]
                - trace_point["moves"][1]["selection_score"]
            ),
            "q_margin": trace_point["moves"][0]["q_value"]
            - trace_point["moves"][1]["q_value"],
        }
        for branch in ("default", "relaxed"):
            prior_pressure["branch_level_evidence"][branch][
                "upstream_checkpoint_echo"
            ] = copy.deepcopy(checkpoint_echo)
        default_trace = self.trace_artifact(
            thresholds=self.default_thresholds(),
            trace_points=[trace_point],
            source_artifact=source_artifact,
        )
        relaxed_trace = self.trace_artifact(
            thresholds=self.relaxed_thresholds(),
            trace_points=[copy.deepcopy(trace_point)],
            source_artifact=source_artifact,
        )
        return module.build_payload(
            prior_pressure,
            default_trace,
            relaxed_trace,
            source_prior_pressure_audit_artifact_path="/tmp/prior_pressure_audit.json",
            source_selection_score_artifact_path="/tmp/default.json",
            source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
        )

    def test_build_payload_classifies_competing_reference_residual_before_inconclusive(
        self,
    ):
        payload = self.build_payload_for_canonical_trace_point(
            self.canonical_trace_point(
                selected_selection_score=0.07,
                selected_q_value=-0.01,
                reference_selection_score=0.07,
                reference_q_value=0.0,
                visits=[2.0, 0.0, 1.0, 0.0, 0.0],
            )
        )

        self.assertEqual(
            "reference_or_other_move_residual_competes",
            payload["branch_residual_evidence"]["default"]["branch_candidate"],
        )
        self.assertEqual(
            "reference_or_other_move_residual_competes",
            payload["classification"]["classification"],
        )

    def test_build_payload_prefers_competing_residual_over_missing_other_move_data(
        self,
    ):
        payload = self.build_payload_for_canonical_trace_point(
            self.canonical_trace_point(
                selected_selection_score=0.07,
                selected_q_value=-0.01,
                reference_selection_score=0.07,
                reference_q_value=0.0,
                visits=[2.0, 0.0, 1.0, 0.0, 1.0],
            )
        )

        self.assertEqual(
            "reference_or_other_move_residual_competes",
            payload["branch_residual_evidence"]["default"]["branch_candidate"],
        )

    def test_build_payload_uses_moves_list_for_legal_move_universe_when_visit_is_zero(
        self,
    ):
        payload = self.build_payload_for_canonical_trace_point(
            self.canonical_trace_point(
                selected_selection_score=0.07,
                selected_q_value=-0.01,
                reference_selection_score=0.0,
                reference_q_value=0.0,
                visits=[2.0, 0.0, 1.0, 0.0, 0.0],
                extra_moves=[
                    {
                        "move": 4,
                        "selection_score": 0.09,
                        "q_value": 0.0,
                    }
                ],
            )
        )

        self.assertEqual(
            [0, 2, 4],
            payload["branch_residual_evidence"]["default"]["legal_moves"],
        )
        self.assertEqual(
            "reference_or_other_move_residual_competes",
            payload["branch_residual_evidence"]["default"]["branch_candidate"],
        )
        self.assertEqual(
            [4],
            payload["branch_residual_evidence"]["default"]["competing_moves"],
        )

    def test_build_payload_marks_missing_non_selected_residual_input_ambiguous(self):
        payload = self.build_payload_for_canonical_trace_point(
            self.canonical_trace_point(
                selected_selection_score=0.12,
                selected_q_value=-0.01,
                reference_selection_score=0.01,
                reference_q_value=0.0,
                visits=[2.0, 0.0, 1.0, 0.0, 1.0],
            )
        )

        self.assertEqual(
            "tiny_count_residual_ambiguous",
            payload["branch_residual_evidence"]["default"]["branch_candidate"],
        )
        self.assertFalse(
            payload["branch_residual_evidence"]["default"]["visit_summary"]["usable"]
        )

    def test_build_payload_rejects_boolean_move_ids_without_changing_competitor_precedence(
        self,
    ):
        payload = self.build_payload_for_canonical_trace_point(
            self.canonical_trace_point(
                selected_selection_score=0.07,
                selected_q_value=-0.01,
                reference_selection_score=0.07,
                reference_q_value=0.0,
                visits=[2.0, 1.0, 1.0, 0.0, 0.0],
                extra_moves=[
                    {
                        "move": True,
                        "selection_score": 0.99,
                        "q_value": -0.01,
                    }
                ],
            )
        )

        self.assertFalse(
            payload["branch_residual_evidence"]["default"]["visit_summary"]["usable"]
        )
        self.assertEqual(
            "reference_or_other_move_residual_competes",
            payload["branch_residual_evidence"]["default"]["branch_candidate"],
        )

    def test_build_payload_rejects_negative_visit_counts_from_stable_branch(self):
        payload = self.build_payload_for_canonical_trace_point(
            self.canonical_trace_point(
                selected_selection_score=0.12,
                selected_q_value=-0.01,
                reference_selection_score=0.01,
                reference_q_value=0.0,
                visits=[2.0, -1.0, 1.0, 0.0, 0.0],
            )
        )

        self.assertFalse(
            payload["branch_residual_evidence"]["default"]["visit_summary"]["usable"]
        )
        self.assertEqual(
            "tiny_count_residual_ambiguous",
            payload["branch_residual_evidence"]["default"]["branch_candidate"],
        )

    def test_build_payload_classifies_stable_selected_residual_advantage_when_both_branches_stable(
        self,
    ):
        payload = self.build_payload_for_canonical_trace_point(
            self.canonical_trace_point(
                selected_selection_score=0.12,
                selected_q_value=-0.01,
                reference_selection_score=0.01,
                reference_q_value=0.0,
                visits=[2.0, 0.0, 1.0, 0.0, 0.0],
            )
        )

        self.assertEqual(
            "stable_selected_residual_advantage",
            payload["branch_residual_evidence"]["default"]["branch_candidate"],
        )
        self.assertEqual(
            "stable_selected_residual_advantage",
            payload["branch_residual_evidence"]["relaxed"]["branch_candidate"],
        )
        self.assertEqual(
            "stable_selected_residual_advantage",
            payload["classification"]["classification"],
        )
        self.assertEqual(
            "write_002_residual_ablation_spec",
            payload["decision"],
        )

    def test_build_payload_classifies_tiny_count_residual_ambiguous_when_any_branch_is_ambiguity_limited(
        self,
    ):
        source_artifact = self.source_artifact_with_provenance()
        stable_trace_point = self.canonical_trace_point(
            selected_selection_score=0.12,
            selected_q_value=-0.01,
            reference_selection_score=0.01,
            reference_q_value=0.0,
            visits=[2.0, 0.0, 1.0, 0.0, 0.0],
        )
        ambiguous_trace_point = self.canonical_trace_point(
            selected_selection_score=0.12,
            selected_q_value=-0.01,
            reference_selection_score=0.01,
            reference_q_value=0.0,
            visits=[2.0, 0.0, 1.0, 0.0, 1.0],
        )
        prior_pressure = self.prior_pressure_artifact(source_artifact=source_artifact)
        for branch, trace_point in (
            ("default", stable_trace_point),
            ("relaxed", ambiguous_trace_point),
        ):
            prior_pressure["branch_level_evidence"][branch][
                "upstream_checkpoint_echo"
            ] = {
                "simulation": 2.0,
                "selection_score_margin": (
                    trace_point["moves"][0]["selection_score"]
                    - trace_point["moves"][1]["selection_score"]
                ),
                "q_margin": trace_point["moves"][0]["q_value"]
                - trace_point["moves"][1]["q_value"],
            }

        payload = module.build_payload(
            prior_pressure,
            self.trace_artifact(
                thresholds=self.default_thresholds(),
                trace_points=[stable_trace_point],
                source_artifact=source_artifact,
            ),
            self.trace_artifact(
                thresholds=self.relaxed_thresholds(),
                trace_points=[ambiguous_trace_point],
                source_artifact=source_artifact,
            ),
            source_prior_pressure_audit_artifact_path="/tmp/prior_pressure_audit.json",
            source_selection_score_artifact_path="/tmp/default.json",
            source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
        )

        self.assertEqual(
            "tiny_count_residual_ambiguous",
            payload["classification"]["classification"],
        )
        self.assertEqual(
            module.CLASSIFICATION_DECISIONS["tiny_count_residual_ambiguous"],
            payload["decision"],
        )


class Capture002SelectionScoreResidualAuditCliTest(
    Capture002SelectionScoreResidualAuditTestSupport,
    unittest.TestCase,
):
    def write_cli_inputs(self, tmpdir_path: Path) -> tuple[Path, Path, Path]:
        prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure_path = tmpdir_path / "prior_pressure.json"
        default_trace_path = tmpdir_path / "default_trace.json"
        relaxed_trace_path = tmpdir_path / "relaxed_trace.json"
        prior_pressure["input_artifacts"]["source_selection_score_artifact_path"] = str(
            default_trace_path
        )
        prior_pressure["input_artifacts"][
            "source_threshold_relaxed_selection_score_artifact_path"
        ] = str(relaxed_trace_path)
        prior_pressure_path.write_text(json.dumps(prior_pressure), encoding="utf-8")
        default_trace_path.write_text(json.dumps(default_trace), encoding="utf-8")
        relaxed_trace_path.write_text(json.dumps(relaxed_trace), encoding="utf-8")
        return prior_pressure_path, default_trace_path, relaxed_trace_path

    def test_main_writes_sorted_json_and_prints_compact_summary_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            prior_pressure_path, default_trace_path, relaxed_trace_path = (
                self.write_cli_inputs(tmpdir_path)
            )
            output_path = tmpdir_path / "out.json"

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = module.main(
                    [
                        "--source-prior-pressure-audit-artifact",
                        str(prior_pressure_path),
                        "--source-selection-score-artifact",
                        str(default_trace_path),
                        "--source-threshold-relaxed-selection-score-artifact",
                        str(relaxed_trace_path),
                        "--out",
                        str(output_path),
                    ]
                )

            self.assertEqual(0, exit_code)
            written_text = output_path.read_text(encoding="utf-8")
            self.assertTrue(written_text.endswith("\n"))
            self.assertEqual(
                json.dumps(json.loads(written_text), indent=2, sort_keys=True) + "\n",
                written_text,
            )
            self.assertEqual(
                json.dumps(
                    {
                        "classification": "stable_selected_residual_advantage",
                        "decision": "write_002_residual_ablation_spec",
                    },
                    sort_keys=True,
                ),
                stdout.getvalue().strip(),
            )

    def test_main_creates_nested_output_parent_before_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            prior_pressure_path, default_trace_path, relaxed_trace_path = (
                self.write_cli_inputs(tmpdir_path)
            )
            output_path = tmpdir_path / "nested" / "deeper" / "out.json"

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = module.main(
                    [
                        "--source-prior-pressure-audit-artifact",
                        str(prior_pressure_path),
                        "--source-selection-score-artifact",
                        str(default_trace_path),
                        "--source-threshold-relaxed-selection-score-artifact",
                        str(relaxed_trace_path),
                        "--out",
                        str(output_path),
                    ]
                )

            self.assertEqual(0, exit_code)
            self.assertTrue(output_path.exists())
