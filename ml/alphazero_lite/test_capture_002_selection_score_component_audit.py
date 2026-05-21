import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ml.alphazero_lite import capture_002_selection_score_component_audit as module


class Capture002SelectionScoreComponentAuditContractTest(unittest.TestCase):
    def test_contract_constants_are_stable(self):
        self.assertEqual(
            "azlite_capture_002_selection_score_component_audit_v1", module.SCHEMA
        )
        self.assertEqual(
            "azlite_capture_002_metric_co_movement_audit_v1",
            module.SOURCE_METRIC_AUDIT_SCHEMA,
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
            "early_selection_score_only", module.EXPECTED_METRIC_AUDIT_CLASSIFICATION
        )
        self.assertEqual(
            "write_002_selection_score_component_audit_spec",
            module.EXPECTED_METRIC_AUDIT_DECISION,
        )
        self.assertEqual(
            {
                "prior_pressure_lead": "write_002_prior_pressure_component_spec",
                "child_q_lift_lead": "write_002_child_q_lift_component_spec",
                "mixed_selection_score_signal": "write_002_mixed_selection_score_component_spec",
                "selection_score_component_inconclusive": "stop_002_selection_score_component_inconclusive",
            },
            module.CLASSIFICATION_DECISIONS,
        )

    def test_parse_args_reads_required_paths(self):
        args = module.parse_args(
            [
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
            Path("/tmp/metric_audit.json"), args.source_metric_audit_artifact
        )
        self.assertEqual(
            Path("/tmp/default.json"), args.source_selection_score_artifact
        )
        self.assertEqual(
            Path("/tmp/relaxed.json"),
            args.source_threshold_relaxed_selection_score_artifact,
        )
        self.assertEqual(Path("/tmp/out.json"), args.out)
        self.assertIsNone(args.source_checkpoint_canonicalization_artifact)

    def test_parse_args_accepts_optional_checkpoint_canonicalization_path(self):
        args = module.parse_args(
            [
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


class Capture002SelectionScoreComponentAuditBuildPayloadTest(unittest.TestCase):
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

    def metric_audit_artifact(
        self,
        *,
        classification: str = "early_selection_score_only",
        decision: str = "write_002_selection_score_component_audit_spec",
        source_artifact: dict | None = None,
    ) -> dict:
        source_artifact = copy.deepcopy(
            source_artifact or self.source_artifact_with_provenance()
        )
        return {
            "schema": module.SOURCE_METRIC_AUDIT_SCHEMA,
            "hypothesis": "metric_co_movement_audit",
            "classification": {
                "classification": classification,
                "evidence_summary": "selection-score lead",
            },
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
                "default": {
                    "q": None,
                    "selection_score": {"simulation": 1.0, "margin": 0.06},
                    "visit_share": None,
                },
                "relaxed": {
                    "q": None,
                    "selection_score": {"simulation": 1.0, "margin": 0.06},
                    "visit_share": None,
                },
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
                "decomposition_classification": {
                    "classification": "metric_co_movement"
                },
                "default_trace_classification": {"classification": "unresolved"},
                "relaxed_trace_classification": {"classification": "unresolved"},
                "default_trace_origin": "extracted",
                "relaxed_trace_origin": "extracted",
            },
        }

    def canonicalization_artifact(
        self,
        *,
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
            "schema": module.SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA,
            "decision": "write_002_metric_audit_canonical_input_spec",
            "input_artifacts": {
                "source_selection_score_artifact_path": "/tmp/default.json",
                "source_threshold_relaxed_selection_score_artifact_path": "/tmp/relaxed.json",
            },
            "source_artifact": copy.deepcopy(
                source_artifact or self.source_artifact_with_provenance()
            ),
            "canonicalization_status": {"safe_for_followup_spec": True},
            "canonical_sequences_match": True,
            "canonical_checkpoint_sequences": {
                "default": copy.deepcopy(default_sequence),
                "relaxed": copy.deepcopy(relaxed_sequence),
            },
            "thresholds_evaluated": copy.deepcopy(thresholds_evaluated),
            "trace_origin": trace_origin,
        }

    def valid_inputs(self) -> tuple[dict, dict, dict]:
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
                q_margin=0.01,
                selection_score_margin=0.07,
                selected_visits=2.0,
                reference_visits=1.0,
            ),
        ]
        source_artifact = self.source_artifact_with_provenance()
        return (
            self.metric_audit_artifact(source_artifact=source_artifact),
            self.trace_artifact(
                thresholds=self.default_thresholds(),
                trace_points=trace_points,
                source_artifact=source_artifact,
            ),
            self.trace_artifact(
                thresholds=self.relaxed_thresholds(),
                trace_points=trace_points,
                source_artifact=source_artifact,
            ),
        )

    def build_payload(
        self,
        metric_audit=None,
        default=None,
        relaxed=None,
        canonicalization=None,
        source_checkpoint_canonicalization_artifact_path=None,
    ) -> dict:
        valid_metric_audit, valid_default, valid_relaxed = self.valid_inputs()
        kwargs = {
            "source_metric_audit_artifact_path": "/tmp/metric_audit.json",
            "source_selection_score_artifact_path": "/tmp/default.json",
            "source_threshold_relaxed_selection_score_artifact_path": "/tmp/relaxed.json",
        }
        if canonicalization is not None:
            kwargs["checkpoint_canonicalization_artifact"] = canonicalization
        if source_checkpoint_canonicalization_artifact_path is not None:
            kwargs["source_checkpoint_canonicalization_artifact_path"] = (
                source_checkpoint_canonicalization_artifact_path
            )
        return module.build_payload(
            metric_audit or valid_metric_audit,
            default or valid_default,
            relaxed or valid_relaxed,
            **kwargs,
        )

    def test_build_payload_rejects_wrong_metric_audit_schema(self):
        metric_audit, default, relaxed = self.valid_inputs()
        metric_audit["schema"] = "wrong"

        with self.assertRaisesRegex(
            ValueError, "metric audit artifact has wrong schema"
        ):
            self.build_payload(
                metric_audit=metric_audit, default=default, relaxed=relaxed
            )

    def test_build_payload_rejects_wrong_metric_audit_classification(self):
        metric_audit, default, relaxed = self.valid_inputs()
        metric_audit["classification"]["classification"] = "mixed_low_confidence_signal"

        with self.assertRaisesRegex(
            ValueError,
            "metric audit artifact classification must be early_selection_score_only",
        ):
            self.build_payload(
                metric_audit=metric_audit, default=default, relaxed=relaxed
            )

    def test_build_payload_rejects_wrong_metric_audit_decision(self):
        metric_audit, default, relaxed = self.valid_inputs()
        metric_audit["decision"] = "stop_002_metric_audit_inconclusive"

        with self.assertRaisesRegex(
            ValueError,
            "metric audit artifact decision must be write_002_selection_score_component_audit_spec",
        ):
            self.build_payload(
                metric_audit=metric_audit, default=default, relaxed=relaxed
            )

    def test_build_payload_rejects_wrong_trace_schema(self):
        metric_audit, default, relaxed = self.valid_inputs()
        default["schema"] = "wrong"

        with self.assertRaisesRegex(
            ValueError, "selection score artifact has wrong schema"
        ):
            self.build_payload(
                metric_audit=metric_audit, default=default, relaxed=relaxed
            )

    def test_build_payload_rejects_wrong_trace_decision(self):
        metric_audit, default, relaxed = self.valid_inputs()
        relaxed["decision"] = "write_002_selection_pressure_ablation_spec"

        with self.assertRaisesRegex(
            ValueError,
            "relaxed trace artifact decision must be write_002_unresolved_trace_review_spec",
        ):
            self.build_payload(
                metric_audit=metric_audit, default=default, relaxed=relaxed
            )

    def test_build_payload_rejects_wrong_row_identity(self):
        metric_audit, default, relaxed = self.valid_inputs()
        relaxed["source_artifact"]["row_id"] = "capture_available-003"

        with self.assertRaisesRegex(
            ValueError, "source_artifact.row_id must be capture_available-002"
        ):
            self.build_payload(
                metric_audit=metric_audit, default=default, relaxed=relaxed
            )

    def test_build_payload_classifies_prior_pressure_lead(self):
        metric_audit, default, relaxed = self.valid_inputs()

        payload = self.build_payload(
            metric_audit=metric_audit, default=default, relaxed=relaxed
        )

        self.assertEqual(
            "prior_pressure_lead", payload["classification"]["classification"]
        )
        self.assertEqual("write_002_prior_pressure_component_spec", payload["decision"])

    def test_build_payload_classifies_child_q_lift_lead(self):
        metric_audit, default, relaxed = self.valid_inputs()
        default["trace_points"] = [
            self.trace_point(
                simulation=1.0,
                q_margin=0.05,
                selection_score_margin=0.06,
                selected_visits=1.0,
                reference_visits=1.0,
            ),
            self.trace_point(
                simulation=2.0,
                q_margin=0.06,
                selection_score_margin=0.07,
                selected_visits=2.0,
                reference_visits=1.0,
            ),
        ]
        relaxed["trace_points"] = copy.deepcopy(default["trace_points"])

        payload = self.build_payload(
            metric_audit=metric_audit, default=default, relaxed=relaxed
        )

        self.assertEqual(
            "child_q_lift_lead", payload["classification"]["classification"]
        )
        self.assertEqual("write_002_child_q_lift_component_spec", payload["decision"])

    def test_build_payload_classifies_mixed_signal_when_default_and_relaxed_disagree(
        self,
    ):
        metric_audit, default, relaxed = self.valid_inputs()
        default["trace_points"] = [
            self.trace_point(
                simulation=1.0,
                q_margin=-0.01,
                selection_score_margin=0.06,
                selected_visits=1.0,
                reference_visits=1.0,
            )
        ]
        relaxed["trace_points"] = [
            self.trace_point(
                simulation=1.0,
                q_margin=0.05,
                selection_score_margin=0.06,
                selected_visits=1.0,
                reference_visits=1.0,
            )
        ]

        payload = self.build_payload(
            metric_audit=metric_audit, default=default, relaxed=relaxed
        )

        self.assertEqual(
            "mixed_selection_score_signal", payload["classification"]["classification"]
        )
        self.assertEqual(
            "write_002_mixed_selection_score_component_spec", payload["decision"]
        )

    def test_build_payload_classifies_mixed_signal_when_only_one_branch_has_material_support(
        self,
    ):
        metric_audit, default, relaxed = self.valid_inputs()
        default["trace_points"] = [
            self.trace_point(
                simulation=1.0,
                q_margin=-0.01,
                selection_score_margin=0.02,
                selected_visits=1.0,
                reference_visits=1.0,
            )
        ]
        relaxed["trace_points"] = [
            self.trace_point(
                simulation=1.0,
                q_margin=-0.01,
                selection_score_margin=0.06,
                selected_visits=1.0,
                reference_visits=1.0,
            )
        ]

        payload = self.build_payload(
            metric_audit=metric_audit, default=default, relaxed=relaxed
        )

        self.assertEqual(
            "mixed_selection_score_signal", payload["classification"]["classification"]
        )
        self.assertEqual(
            "write_002_mixed_selection_score_component_spec", payload["decision"]
        )

    def test_build_payload_classifies_mixed_signal_for_first_positive_first_material_conflict(
        self,
    ):
        metric_audit, default, relaxed = self.valid_inputs()
        default["trace_points"] = [
            self.trace_point(
                simulation=1.0,
                q_margin=0.01,
                selection_score_margin=0.06,
                selected_visits=1.0,
                reference_visits=1.0,
            ),
            self.trace_point(
                simulation=2.0,
                q_margin=0.05,
                selection_score_margin=0.07,
                selected_visits=2.0,
                reference_visits=1.0,
            ),
        ]
        relaxed["trace_points"] = copy.deepcopy(default["trace_points"])

        payload = self.build_payload(
            metric_audit=metric_audit, default=default, relaxed=relaxed
        )

        self.assertEqual(
            "mixed_selection_score_signal", payload["classification"]["classification"]
        )

    def test_build_payload_classifies_inconclusive_when_selection_score_support_never_materializes_cleanly(
        self,
    ):
        metric_audit, default, relaxed = self.valid_inputs()
        default["trace_points"] = [
            self.trace_point(
                simulation=1.0,
                q_margin=0.01,
                selection_score_margin=0.02,
                selected_visits=1.0,
                reference_visits=1.0,
            )
        ]
        relaxed["trace_points"] = copy.deepcopy(default["trace_points"])

        payload = self.build_payload(
            metric_audit=metric_audit, default=default, relaxed=relaxed
        )

        self.assertEqual(
            "selection_score_component_inconclusive",
            payload["classification"]["classification"],
        )
        self.assertEqual(
            "stop_002_selection_score_component_inconclusive", payload["decision"]
        )

    def test_build_payload_rejects_duplicate_checkpoint_sequences_without_canonicalization_artifact(
        self,
    ):
        metric_audit, default, relaxed = self.valid_inputs()
        duplicate_default_point = copy.deepcopy(default["trace_points"][0])
        duplicate_relaxed_point = copy.deepcopy(relaxed["trace_points"][0])
        default["trace_points"] = [
            default["trace_points"][0],
            duplicate_default_point,
            default["trace_points"][1],
        ]
        relaxed["trace_points"] = [
            relaxed["trace_points"][0],
            duplicate_relaxed_point,
            relaxed["trace_points"][1],
        ]

        with self.assertRaisesRegex(
            ValueError, "checkpoint sequences must not contain duplicates"
        ):
            self.build_payload(
                metric_audit=metric_audit, default=default, relaxed=relaxed
            )

    def test_build_payload_accepts_duplicate_equivalent_root_snapshots_with_valid_canonicalization_artifact(
        self,
    ):
        metric_audit, default, relaxed = self.valid_inputs()
        duplicate_default_point = copy.deepcopy(default["trace_points"][0])
        duplicate_relaxed_point = copy.deepcopy(relaxed["trace_points"][0])
        metric_audit["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"
        default["trace_points"] = [
            default["trace_points"][0],
            duplicate_default_point,
            default["trace_points"][1],
        ]
        relaxed["trace_points"] = [
            relaxed["trace_points"][0],
            duplicate_relaxed_point,
            relaxed["trace_points"][1],
        ]
        canonicalization = self.canonicalization_artifact(
            default_sequence=[1.0, 2.0], relaxed_sequence=[1.0, 2.0]
        )

        payload = self.build_payload(
            metric_audit=metric_audit,
            default=default,
            relaxed=relaxed,
            canonicalization=canonicalization,
            source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
        )

        self.assertEqual(
            "/tmp/canonicalization.json",
            payload["input_artifacts"][
                "source_checkpoint_canonicalization_artifact_path"
            ],
        )

    def test_build_payload_omits_canonicalization_path_without_canonical_mode(self):
        metric_audit, default, relaxed = self.valid_inputs()

        payload = self.build_payload(
            metric_audit=metric_audit, default=default, relaxed=relaxed
        )

        self.assertNotIn(
            "source_checkpoint_canonicalization_artifact_path",
            payload["input_artifacts"],
        )

    def test_build_payload_rejects_canonical_artifact_without_canonicalization_path(
        self,
    ):
        metric_audit, default, relaxed = self.valid_inputs()
        canonicalization = self.canonicalization_artifact()

        with self.assertRaisesRegex(
            ValueError,
            "canonical mode requires source_checkpoint_canonicalization_artifact_path",
        ):
            self.build_payload(
                metric_audit=metric_audit,
                default=default,
                relaxed=relaxed,
                canonicalization=canonicalization,
            )

    def test_build_payload_rejects_canonicalization_path_without_canonical_artifact(
        self,
    ):
        metric_audit, default, relaxed = self.valid_inputs()
        metric_audit["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"

        with self.assertRaisesRegex(
            ValueError,
            "canonical mode requires checkpoint_canonicalization_artifact",
        ):
            self.build_payload(
                metric_audit=metric_audit,
                default=default,
                relaxed=relaxed,
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_mismatched_upstream_metric_audit_canonicalization_path(
        self,
    ):
        metric_audit, default, relaxed = self.valid_inputs()
        metric_audit["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/stale-canonicalization.json"
        canonicalization = self.canonicalization_artifact()

        with self.assertRaisesRegex(
            ValueError,
            "metric audit input_artifacts source_checkpoint_canonicalization_artifact_path must match source path",
        ):
            self.build_payload(
                metric_audit=metric_audit,
                default=default,
                relaxed=relaxed,
                canonicalization=canonicalization,
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_non_canonical_metric_audit_that_claims_canonicalization_input(
        self,
    ):
        metric_audit, default, relaxed = self.valid_inputs()
        metric_audit["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"

        with self.assertRaisesRegex(
            ValueError,
            "metric audit input_artifacts source_checkpoint_canonicalization_artifact_path must match source path",
        ):
            self.build_payload(
                metric_audit=metric_audit, default=default, relaxed=relaxed
            )

    def test_build_payload_rejects_conflicting_skipped_duplicates_in_canonical_mode(
        self,
    ):
        metric_audit, default, relaxed = self.valid_inputs()
        metric_audit["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"
        duplicate_default_point = copy.deepcopy(default["trace_points"][0])
        duplicate_default_point["selected_move"] = 2
        duplicate_relaxed_point = copy.deepcopy(relaxed["trace_points"][0])
        default["trace_points"] = [
            default["trace_points"][0],
            duplicate_default_point,
            default["trace_points"][1],
        ]
        relaxed["trace_points"] = [
            relaxed["trace_points"][0],
            duplicate_relaxed_point,
            relaxed["trace_points"][1],
        ]
        canonicalization = self.canonicalization_artifact(
            default_sequence=[1.0, 2.0], relaxed_sequence=[1.0, 2.0]
        )

        with self.assertRaisesRegex(
            ValueError,
            "skipped duplicate checkpoint must match kept checkpoint contents",
        ):
            self.build_payload(
                metric_audit=metric_audit,
                default=default,
                relaxed=relaxed,
                canonicalization=canonicalization,
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_canonical_sequence_mismatch(self):
        metric_audit, default, relaxed = self.valid_inputs()
        metric_audit["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"
        duplicate_default_point = copy.deepcopy(default["trace_points"][0])
        duplicate_relaxed_point = copy.deepcopy(relaxed["trace_points"][0])
        default["trace_points"] = [
            default["trace_points"][0],
            duplicate_default_point,
            default["trace_points"][1],
        ]
        relaxed["trace_points"] = [
            relaxed["trace_points"][0],
            duplicate_relaxed_point,
            relaxed["trace_points"][1],
        ]
        canonicalization = self.canonicalization_artifact(
            default_sequence=[1.0, 2.0, 3.0], relaxed_sequence=[1.0, 2.0, 3.0]
        )

        with self.assertRaisesRegex(
            ValueError,
            "collapsed original checkpoint sequence must match canonical checkpoint sequence",
        ):
            self.build_payload(
                metric_audit=metric_audit,
                default=default,
                relaxed=relaxed,
                canonicalization=canonicalization,
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
            )

    def test_build_payload_accepts_full_provenance_chain_in_canonical_mode(self):
        metric_audit, default, relaxed = self.valid_inputs()
        full_source_artifact = self.source_artifact_with_provenance()
        metric_audit["source_artifact"] = copy.deepcopy(full_source_artifact)
        metric_audit["input_artifacts"][
            "source_checkpoint_canonicalization_artifact_path"
        ] = "/tmp/canonicalization.json"
        default["source_artifact"] = copy.deepcopy(full_source_artifact)
        relaxed["source_artifact"] = copy.deepcopy(full_source_artifact)
        duplicate_default_point = copy.deepcopy(default["trace_points"][0])
        duplicate_relaxed_point = copy.deepcopy(relaxed["trace_points"][0])
        default["trace_points"] = [
            default["trace_points"][0],
            duplicate_default_point,
            default["trace_points"][1],
        ]
        relaxed["trace_points"] = [
            relaxed["trace_points"][0],
            duplicate_relaxed_point,
            relaxed["trace_points"][1],
        ]
        canonicalization = self.canonicalization_artifact(
            source_artifact=full_source_artifact,
            default_sequence=[1.0, 2.0],
            relaxed_sequence=[1.0, 2.0],
        )

        payload = self.build_payload(
            metric_audit=metric_audit,
            default=default,
            relaxed=relaxed,
            canonicalization=canonicalization,
            source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
        )

        self.assertEqual(full_source_artifact, payload["source_artifact"])

    def test_main_writes_sorted_payload_and_prints_compact_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metric_audit_path = tmp_path / "metric_audit.json"
            default_path = tmp_path / "default.json"
            relaxed_path = tmp_path / "relaxed.json"
            out_path = tmp_path / "diagnostics" / "selection_score_component_audit.json"
            metric_audit, default, relaxed = self.valid_inputs()
            metric_audit["input_artifacts"]["source_selection_score_artifact_path"] = (
                str(default_path)
            )
            metric_audit["input_artifacts"][
                "source_threshold_relaxed_selection_score_artifact_path"
            ] = str(relaxed_path)

            metric_audit_path.write_text(json.dumps(metric_audit), encoding="utf-8")
            default_path.write_text(json.dumps(default), encoding="utf-8")
            relaxed_path.write_text(json.dumps(relaxed), encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = module.main(
                    [
                        "--source-metric-audit-artifact",
                        str(metric_audit_path),
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
        self.assertEqual(payload["decision"], summary["decision"])


if __name__ == "__main__":
    unittest.main()
