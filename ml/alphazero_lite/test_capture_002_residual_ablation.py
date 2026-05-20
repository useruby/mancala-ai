import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from ml.alphazero_lite import capture_002_residual_ablation as module


class Capture002ResidualAblationContractTest(unittest.TestCase):
    def test_contract_constants_are_stable(self):
        self.assertEqual("azlite_capture_002_residual_ablation_v1", module.SCHEMA)
        self.assertEqual(
            "azlite_capture_002_selection_score_residual_audit_v1",
            module.SOURCE_SELECTION_SCORE_RESIDUAL_AUDIT_SCHEMA,
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
            "stable_selected_residual_advantage",
            module.EXPECTED_RESIDUAL_AUDIT_CLASSIFICATION,
        )
        self.assertEqual(
            "write_002_residual_ablation_spec",
            module.EXPECTED_RESIDUAL_AUDIT_DECISION,
        )
        self.assertEqual(
            "selection_score_residual_lead",
            module.EXPECTED_PRIOR_PRESSURE_CLASSIFICATION,
        )
        self.assertEqual(
            "write_002_selection_score_residual_spec",
            module.EXPECTED_PRIOR_PRESSURE_DECISION,
        )
        self.assertEqual("unresolved", module.EXPECTED_TRACE_CLASSIFICATION)
        self.assertEqual(
            "write_002_unresolved_trace_review_spec",
            module.EXPECTED_TRACE_DECISION,
        )
        self.assertEqual(
            "write_002_metric_audit_canonical_input_spec",
            module.EXPECTED_CANONICALIZATION_DECISION,
        )
        self.assertEqual(2.0, module.EXPECTED_CANONICAL_SIMULATION)
        self.assertEqual(
            (
                "baseline_replay",
                "selected_residual_neutralized",
                "all_residuals_flattened",
            ),
            module.MODES,
        )
        self.assertEqual(
            {
                "selected_move_residual_sensitive": "write_002_residual_sensitive_intervention_spec",
                "selected_move_residual_insensitive": "write_002_non_residual_mechanism_review_spec",
                "selected_move_residual_ablation_inconclusive": "stop_002_residual_ablation_inconclusive",
            },
            module.CLASSIFICATION_DECISIONS,
        )

    def test_parse_args_reads_required_paths(self):
        args = module.parse_args(
            [
                "--source-selection-score-residual-audit-artifact",
                "/tmp/residual_audit.json",
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
            Path("/tmp/residual_audit.json"),
            args.source_selection_score_residual_audit_artifact,
        )
        self.assertEqual(
            Path("/tmp/prior_pressure_audit.json"),
            args.source_prior_pressure_audit_artifact,
        )
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
                "--source-selection-score-residual-audit-artifact",
                "/tmp/residual_audit.json",
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

    def test_parse_args_requires_all_required_flags(self):
        valid_argv = [
            "--source-selection-score-residual-audit-artifact",
            "/tmp/residual_audit.json",
            "--source-prior-pressure-audit-artifact",
            "/tmp/prior_pressure_audit.json",
            "--source-selection-score-artifact",
            "/tmp/default.json",
            "--source-threshold-relaxed-selection-score-artifact",
            "/tmp/relaxed.json",
            "--out",
            "/tmp/out.json",
        ]

        required_flags = [
            "--source-selection-score-residual-audit-artifact",
            "--source-prior-pressure-audit-artifact",
            "--source-selection-score-artifact",
            "--source-threshold-relaxed-selection-score-artifact",
            "--out",
        ]

        for required_flag in required_flags:
            with self.subTest(required_flag=required_flag):
                missing_flag_argv = list(valid_argv)
                index = missing_flag_argv.index(required_flag)
                del missing_flag_argv[index : index + 2]

                with self.assertRaises(SystemExit) as error:
                    module.parse_args(missing_flag_argv)

                self.assertEqual(2, error.exception.code)


class Capture002ResidualAblationTestSupport:
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
            "row_id": module.ROW_ID,
            "reference_move": 2,
            "full_search_selected_move": 0,
            "selected_artifact": self.selected_artifact(),
        }

    def trace_point(self, *, simulation: float, moves: list[dict], visits: list[float]) -> dict:
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
            "classification": {"classification": module.EXPECTED_TRACE_CLASSIFICATION},
            "decision": module.EXPECTED_TRACE_DECISION,
            "insufficiency_reasons": [],
            "trace_origin": trace_origin,
            "source_artifact": copy.deepcopy(source_artifact or self.source_artifact_with_provenance()),
            "thresholds": copy.deepcopy(thresholds),
            "trace_points": copy.deepcopy(trace_points),
        }

    def prior_pressure_artifact(self, *, source_artifact: dict | None = None) -> dict:
        return {
            "schema": module.SOURCE_PRIOR_PRESSURE_AUDIT_SCHEMA,
            "classification": {
                "classification": module.EXPECTED_PRIOR_PRESSURE_CLASSIFICATION,
                "evidence_summary": "selection-score residual lead",
            },
            "decision": module.EXPECTED_PRIOR_PRESSURE_DECISION,
            "source_artifact": copy.deepcopy(source_artifact or self.source_artifact_with_provenance()),
        }

    def residual_audit_artifact(self, *, source_artifact: dict | None = None) -> dict:
        return {
            "schema": module.SOURCE_SELECTION_SCORE_RESIDUAL_AUDIT_SCHEMA,
            "hypothesis": "selection_score_residual_audit",
            "classification": {
                "classification": module.EXPECTED_RESIDUAL_AUDIT_CLASSIFICATION,
                "evidence_summary": "stable selected residual advantage",
            },
            "decision": module.EXPECTED_RESIDUAL_AUDIT_DECISION,
            "input_artifacts": {
                "source_prior_pressure_audit_artifact_path": "/tmp/prior_pressure.json",
                "source_selection_score_artifact_path": "/tmp/default.json",
                "source_threshold_relaxed_selection_score_artifact_path": "/tmp/relaxed.json",
            },
            "source_artifact": copy.deepcopy(source_artifact or self.source_artifact_with_provenance()),
            "checkpoint": {
                "canonical_simulation": module.EXPECTED_CANONICAL_SIMULATION,
                "default_upstream_checkpoint_echo": {
                    "simulation": module.EXPECTED_CANONICAL_SIMULATION,
                    "selection_score_margin": 0.07,
                    "q_margin": -0.01,
                },
                "relaxed_upstream_checkpoint_echo": {
                    "simulation": module.EXPECTED_CANONICAL_SIMULATION,
                    "selection_score_margin": 0.07,
                    "q_margin": -0.01,
                },
            },
            "thresholds_evaluated": {
                "default": {
                    **self.default_thresholds(),
                    "selection_score_residual_threshold": 0.05,
                },
                "relaxed": {
                    **self.relaxed_thresholds(),
                    "selection_score_residual_threshold": 0.05,
                },
                "float_tolerance": 1e-12,
            },
            "source_snapshots": {
                "metric_audit_classification": {"classification": "early_selection_score_only"},
                "default_trace_classification": {"classification": "unresolved"},
                "relaxed_trace_classification": {"classification": "unresolved"},
            },
        }

    def valid_inputs(self) -> tuple[dict, dict, dict, dict]:
        source_artifact = self.source_artifact_with_provenance()
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
        return (
            self.residual_audit_artifact(source_artifact=source_artifact),
            self.prior_pressure_artifact(source_artifact=source_artifact),
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


class Capture002ResidualAblationBuildPayloadHappyPathTest(
    Capture002ResidualAblationTestSupport,
    unittest.TestCase,
):
    def test_build_payload_returns_initial_summary_skeleton(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()

        payload = module.build_payload(
            residual_audit,
            prior_pressure,
            default_trace,
            relaxed_trace,
            source_selection_score_residual_audit_artifact_path="/tmp/residual_audit.json",
            source_prior_pressure_audit_artifact_path="/tmp/prior_pressure.json",
            source_selection_score_artifact_path="/tmp/default.json",
            source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
        )

        self.assertEqual(module.SCHEMA, payload["schema"])
        self.assertEqual("residual_ablation", payload["hypothesis"])
        self.assertEqual(
            "selected_move_residual_insensitive",
            payload["classification"]["classification"],
        )
        self.assertEqual(
            ["baseline_replay", "selected_residual_neutralized", "all_residuals_flattened"],
            [entry["mode"] for entry in payload["mode_results"]],
        )
        self.assertEqual(
            {
                "mode",
                "validation_status",
                "selected_move",
                "preserved_move_zero",
                "evidence_summary",
                "failure_reason",
                "applied_edit_summary",
                "branch_selected_moves",
                "branches_agree",
            },
            set(payload["mode_results"][0].keys()),
        )
        self.assertEqual("ok", payload["mode_results"][0]["validation_status"])
        self.assertTrue(payload["mode_results"][0]["preserved_move_zero"])
        self.assertIsNone(payload["mode_results"][0]["failure_reason"])
        self.assertEqual("no ablation edit applied", payload["mode_results"][0]["applied_edit_summary"])
        self.assertEqual(0, payload["mode_comparison"]["baseline_selected_move"])
        self.assertEqual(0, payload["mode_comparison"]["selected_residual_neutralized_selected_move"])
        self.assertEqual(0, payload["mode_comparison"]["all_residuals_flattened_selected_move"])
        self.assertFalse(payload["mode_comparison"]["selected_residual_neutralized_changed_away_from_baseline"])
        self.assertFalse(payload["mode_comparison"]["all_residuals_flattened_changed_away_from_baseline"])
        self.assertEqual(
            {
                "schema",
                "hypothesis",
                "classification",
                "decision",
                "input_artifacts",
                "source_artifact",
                "checkpoint",
                "thresholds_evaluated",
                "mode_results",
                "mode_comparison",
                "source_snapshots",
            },
            set(payload.keys()),
        )

    def test_build_payload_rebuilds_checkpoint_payload_from_canonical_trace_lookup(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        residual_audit["checkpoint"]["default_upstream_checkpoint_echo"]["selection_score_margin"] = 999.0
        default_trace["trace_points"][1]["moves"][0]["selection_score"] = 0.11
        relaxed_trace["trace_points"][1]["moves"][0]["selection_score"] = 0.12

        payload = module.build_payload(
            residual_audit,
            prior_pressure,
            default_trace,
            relaxed_trace,
            source_selection_score_residual_audit_artifact_path="/tmp/residual_audit.json",
            source_prior_pressure_audit_artifact_path="/tmp/prior_pressure.json",
            source_selection_score_artifact_path="/tmp/default.json",
            source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
        )

        self.assertEqual(
            {
                "canonical_simulation": module.EXPECTED_CANONICAL_SIMULATION,
                "default_upstream_checkpoint_echo": {
                    "simulation": module.EXPECTED_CANONICAL_SIMULATION,
                    "selection_score_margin": 0.11,
                    "q_margin": -0.01,
                },
                "relaxed_upstream_checkpoint_echo": {
                    "simulation": module.EXPECTED_CANONICAL_SIMULATION,
                    "selection_score_margin": 0.12,
                    "q_margin": -0.01,
                },
            },
            payload["checkpoint"],
        )


class Capture002ResidualAblationModeExecutionTest(
    Capture002ResidualAblationTestSupport,
    unittest.TestCase,
):
    def canonical_trace_point(
        self,
        *,
        selected_selection_score: float,
        selected_q_value: float,
        reference_selection_score: float,
        reference_q_value: float,
        extra_moves: list[dict] | None = None,
        visits: list[float] | None = None,
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
            simulation=module.EXPECTED_CANONICAL_SIMULATION,
            moves=moves,
            visits=visits or [2.0, 0.0, 1.0, 0.0, 0.0],
        )

    def build_payload_for_trace_points(self, *, default_trace_point: dict, relaxed_trace_point: dict | None = None):
        source_artifact = self.source_artifact_with_provenance()
        residual_audit = self.residual_audit_artifact(source_artifact=source_artifact)
        prior_pressure = self.prior_pressure_artifact(source_artifact=source_artifact)
        default_trace = self.trace_artifact(
            thresholds=self.default_thresholds(),
            trace_points=[default_trace_point],
            source_artifact=source_artifact,
        )
        relaxed_trace = self.trace_artifact(
            thresholds=self.relaxed_thresholds(),
            trace_points=[copy.deepcopy(relaxed_trace_point or default_trace_point)],
            source_artifact=source_artifact,
        )
        return module.build_payload(
            residual_audit,
            prior_pressure,
            default_trace,
            relaxed_trace,
            source_selection_score_residual_audit_artifact_path="/tmp/residual_audit.json",
            source_prior_pressure_audit_artifact_path="/tmp/prior_pressure.json",
            source_selection_score_artifact_path="/tmp/default.json",
            source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
        )

    def test_build_payload_classifies_selected_move_residual_sensitive_when_ablation_changes_move(self):
        payload = self.build_payload_for_trace_points(
            default_trace_point=self.canonical_trace_point(
                selected_selection_score=0.09,
                selected_q_value=-0.01,
                reference_selection_score=0.08,
                reference_q_value=0.08,
            )
        )

        self.assertEqual("selected_move_residual_sensitive", payload["classification"]["classification"])
        self.assertEqual(
            "write_002_residual_sensitive_intervention_spec",
            payload["decision"],
        )
        self.assertEqual(0, payload["mode_comparison"]["baseline_selected_move"])
        self.assertEqual(2, payload["mode_comparison"]["selected_residual_neutralized_selected_move"])
        self.assertEqual(2, payload["mode_comparison"]["all_residuals_flattened_selected_move"])

    def test_build_payload_classifies_selected_move_residual_insensitive_when_all_modes_preserve_move_zero(self):
        payload = self.build_payload_for_trace_points(
            default_trace_point=self.canonical_trace_point(
                selected_selection_score=0.07,
                selected_q_value=-0.01,
                reference_selection_score=0.0,
                reference_q_value=0.0,
            )
        )

        self.assertEqual("selected_move_residual_insensitive", payload["classification"]["classification"])
        self.assertEqual(
            "write_002_non_residual_mechanism_review_spec",
            payload["decision"],
        )
        self.assertEqual(
            [0, 0, 0],
            [entry["selected_move"] for entry in payload["mode_results"]],
        )

    def test_build_payload_classifies_selected_move_residual_ablation_inconclusive_when_baseline_selects_non_zero(self):
        payload = self.build_payload_for_trace_points(
            default_trace_point=self.canonical_trace_point(
                selected_selection_score=0.05,
                selected_q_value=-0.01,
                reference_selection_score=0.06,
                reference_q_value=0.01,
            )
        )

        self.assertEqual(
            "selected_move_residual_ablation_inconclusive",
            payload["classification"]["classification"],
        )
        self.assertEqual(
            "stop_002_residual_ablation_inconclusive",
            payload["decision"],
        )
        self.assertEqual(2, payload["mode_comparison"]["baseline_selected_move"])

    def test_build_payload_selected_residual_neutralized_still_edits_move_zero_when_baseline_drifts(self):
        payload = self.build_payload_for_trace_points(
            default_trace_point=self.canonical_trace_point(
                selected_selection_score=0.05,
                selected_q_value=0.04,
                reference_selection_score=0.06,
                reference_q_value=0.0,
            )
        )

        self.assertEqual(2, payload["mode_comparison"]["baseline_selected_move"])
        self.assertEqual(2, payload["mode_comparison"]["selected_residual_neutralized_selected_move"])
        self.assertEqual(
            "selected_residual_neutralized changed away from move 0",
            payload["mode_results"][1]["evidence_summary"],
        )

    def test_build_payload_rejects_all_residuals_flattened_when_any_legal_move_lacks_usable_residual_evidence(self):
        with self.assertRaisesRegex(
            ValueError,
            "all_residuals_flattened requires usable residual evidence for every legal move",
        ):
            self.build_payload_for_trace_points(
                default_trace_point=self.canonical_trace_point(
                    selected_selection_score=0.09,
                    selected_q_value=-0.01,
                    reference_selection_score=0.0,
                    reference_q_value=0.0,
                    extra_moves=[{"move": 4, "selection_score": 0.03, "q_value": None}],
                )
            )

    def test_build_payload_rejects_non_row_002_selected_move_identity(self):
        source_artifact = self.source_artifact_with_provenance()
        source_artifact["full_search_selected_move"] = 2

        trace_point = self.trace_point(
            simulation=module.EXPECTED_CANONICAL_SIMULATION,
            moves=[
                {"move": 0, "selection_score": 0.01, "q_value": 0.0},
                {"move": 2, "selection_score": 0.07, "q_value": 0.07},
            ],
            visits=[1.0, 0.0, 2.0, 0.0, 0.0],
        )

        with self.assertRaisesRegex(
            ValueError,
            "selection-score residual audit artifact source_artifact.full_search_selected_move must be 0",
        ):
            module.build_payload(
                self.residual_audit_artifact(source_artifact=source_artifact),
                self.prior_pressure_artifact(source_artifact=source_artifact),
                self.trace_artifact(
                    thresholds=self.default_thresholds(),
                    trace_points=[trace_point],
                    source_artifact=source_artifact,
                ),
                self.trace_artifact(
                    thresholds=self.relaxed_thresholds(),
                    trace_points=[copy.deepcopy(trace_point)],
                    source_artifact=source_artifact,
                ),
                source_selection_score_residual_audit_artifact_path="/tmp/residual_audit.json",
                source_prior_pressure_audit_artifact_path="/tmp/prior_pressure.json",
                source_selection_score_artifact_path="/tmp/default.json",
                source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
            )

    def test_build_payload_classifies_inconclusive_when_branches_disagree_on_baseline(self):
        payload = self.build_payload_for_trace_points(
            default_trace_point=self.canonical_trace_point(
                selected_selection_score=0.07,
                selected_q_value=-0.01,
                reference_selection_score=0.0,
                reference_q_value=0.0,
            ),
            relaxed_trace_point=self.canonical_trace_point(
                selected_selection_score=0.01,
                selected_q_value=0.0,
                reference_selection_score=0.08,
                reference_q_value=0.0,
            ),
        )

        self.assertEqual(
            "selected_move_residual_ablation_inconclusive",
            payload["classification"]["classification"],
        )
        self.assertIsNone(payload["mode_results"][0]["selected_move"])


class Capture002ResidualAblationValidationTest(
    Capture002ResidualAblationTestSupport,
    unittest.TestCase,
):
    def build_payload(
        self,
        *,
        residual_audit=None,
        prior_pressure=None,
        default_trace=None,
        relaxed_trace=None,
    ):
        (
            valid_residual_audit,
            valid_prior_pressure,
            valid_default_trace,
            valid_relaxed_trace,
        ) = self.valid_inputs()
        return module.build_payload(
            residual_audit or valid_residual_audit,
            prior_pressure or valid_prior_pressure,
            default_trace or valid_default_trace,
            relaxed_trace or valid_relaxed_trace,
            source_selection_score_residual_audit_artifact_path="/tmp/residual_audit.json",
            source_prior_pressure_audit_artifact_path="/tmp/prior_pressure.json",
            source_selection_score_artifact_path="/tmp/default.json",
            source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
        )

    def test_build_payload_rejects_wrong_residual_audit_schema(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        residual_audit["schema"] = "wrong"

        with self.assertRaisesRegex(ValueError, "selection-score residual audit artifact has wrong schema"):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )
    def test_build_payload_rejects_wrong_residual_audit_classification(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        residual_audit["classification"]["classification"] = "wrong"

        with self.assertRaisesRegex(
            ValueError,
            f"selection-score residual audit artifact classification must be {module.EXPECTED_RESIDUAL_AUDIT_CLASSIFICATION}",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_non_canonical_residual_audit_checkpoint(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        residual_audit["checkpoint"]["canonical_simulation"] = 1.0

        with self.assertRaisesRegex(
            ValueError,
            f"selection-score residual audit artifact checkpoint.canonical_simulation must be {module.EXPECTED_CANONICAL_SIMULATION}",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_wrong_residual_audit_decision(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        residual_audit["decision"] = "wrong"

        with self.assertRaisesRegex(
            ValueError,
            f"selection-score residual audit artifact decision must be {module.EXPECTED_RESIDUAL_AUDIT_DECISION}",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_residual_audit_input_prior_pressure_path_mismatch(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        residual_audit["input_artifacts"]["source_prior_pressure_audit_artifact_path"] = "/tmp/other.json"

        with self.assertRaisesRegex(
            ValueError,
            "selection-score residual audit artifact input_artifacts source_prior_pressure_audit_artifact_path must match source path",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rebuilds_thresholds_and_trace_snapshots_from_validated_inputs_only(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        residual_audit["thresholds_evaluated"] = {"stale": True}
        residual_audit["source_snapshots"] = {
            "stale": True,
            "metric_audit_classification": {"classification": "early_selection_score_only"},
        }
        default_trace["trace_origin"] = "default-live"
        relaxed_trace["trace_origin"] = "relaxed-live"

        payload = self.build_payload(
            residual_audit=residual_audit,
            prior_pressure=prior_pressure,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

        self.assertEqual(
            {
                "default": self.default_thresholds(),
                "relaxed": self.relaxed_thresholds(),
                "float_tolerance": module.FLOAT_TOLERANCE,
            },
            payload["thresholds_evaluated"],
        )
        self.assertEqual(
            {
                "metric_audit_classification": {"classification": "early_selection_score_only"},
                "default_trace_classification": default_trace["classification"],
                "relaxed_trace_classification": relaxed_trace["classification"],
                "default_trace_origin": "default-live",
                "relaxed_trace_origin": "relaxed-live",
            },
            payload["source_snapshots"],
        )

    def test_build_payload_rejects_wrong_prior_pressure_schema(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["schema"] = "wrong"

        with self.assertRaisesRegex(ValueError, "prior-pressure audit artifact has wrong schema"):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_wrong_prior_pressure_classification(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["classification"] = {"classification": "wrong"}

        with self.assertRaisesRegex(
            ValueError,
            f"prior-pressure audit artifact classification must be {module.EXPECTED_PRIOR_PRESSURE_CLASSIFICATION}",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_wrong_prior_pressure_decision(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["decision"] = "wrong"

        with self.assertRaisesRegex(
            ValueError,
            f"prior-pressure audit artifact decision must be {module.EXPECTED_PRIOR_PRESSURE_DECISION}",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_wrong_default_trace_classification(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["classification"]["classification"] = "wrong"

        with self.assertRaisesRegex(
            ValueError,
            f"default trace artifact classification must be {module.EXPECTED_TRACE_CLASSIFICATION}",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_wrong_relaxed_trace_decision(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        relaxed_trace["decision"] = "wrong"

        with self.assertRaisesRegex(
            ValueError,
            f"relaxed trace artifact decision must be {module.EXPECTED_TRACE_DECISION}",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_missing_default_trace_trace_origin(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        del default_trace["trace_origin"]

        with self.assertRaisesRegex(
            ValueError,
            "default trace artifact trace_origin must be a non-empty string",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_non_numeric_relaxed_trace_material_selection_score_margin(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        relaxed_trace["thresholds"]["material_selection_score_margin"] = "wrong"

        with self.assertRaisesRegex(
            ValueError,
            "relaxed trace artifact thresholds.material_selection_score_margin must be finite non-negative numeric",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_default_trace_missing_canonical_checkpoint(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["trace_points"] = [
            trace_point
            for trace_point in default_trace["trace_points"]
            if trace_point["simulation"] != module.EXPECTED_CANONICAL_SIMULATION
        ]

        with self.assertRaisesRegex(
            ValueError,
            f"default trace must include canonical simulation {module.EXPECTED_CANONICAL_SIMULATION}",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_conflicting_duplicate_non_canonical_checkpoint(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        duplicate_default_trace_point = copy.deepcopy(default_trace["trace_points"][0])
        duplicate_default_trace_point["moves"][0]["selection_score"] = 0.08
        default_trace["trace_points"].insert(1, duplicate_default_trace_point)

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint sequences must be strictly increasing",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_conflicting_duplicate_canonical_checkpoint_without_canonicalization(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        duplicate_default_trace_point = copy.deepcopy(default_trace["trace_points"][1])
        duplicate_default_trace_point["moves"][0]["selection_score"] = 0.08
        default_trace["trace_points"].append(duplicate_default_trace_point)

        with self.assertRaisesRegex(
            ValueError,
            "default raw duplicate 2.0 checkpoint requires canonicalization artifact",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_equivalent_duplicate_canonical_checkpoint_without_canonicalization(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["trace_points"].append(copy.deepcopy(default_trace["trace_points"][1]))

        with self.assertRaisesRegex(
            ValueError,
            "default raw duplicate 2.0 checkpoint requires canonicalization artifact",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_accepts_default_canonical_trace_point_selected_move_metadata_mismatch(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["trace_points"][1]["selected_move"] = 2

        payload = self.build_payload(
            residual_audit=residual_audit,
            prior_pressure=prior_pressure,
            default_trace=default_trace,
            relaxed_trace=relaxed_trace,
        )

        self.assertEqual(module.SCHEMA, payload["schema"])

    def test_build_payload_rejects_relaxed_canonical_trace_point_reference_move_mismatch(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        relaxed_trace["trace_points"][1]["reference_move_by_prior"] = 0

        with self.assertRaisesRegex(
            ValueError,
            "relaxed canonical trace point.reference_move_by_prior must be 2",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_path_only_canonicalization_hint(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        duplicate_default_trace_point = copy.deepcopy(default_trace["trace_points"][1])
        duplicate_default_trace_point["moves"][0]["selection_score"] = 0.08
        default_trace["trace_points"].append(duplicate_default_trace_point)

        with self.assertRaisesRegex(
            ValueError,
            "source_checkpoint_canonicalization_artifact_path requires checkpoint canonicalization artifact",
        ):
            module.build_payload(
                residual_audit,
                prior_pressure,
                default_trace,
                relaxed_trace,
                source_selection_score_residual_audit_artifact_path="/tmp/residual_audit.json",
                source_prior_pressure_audit_artifact_path="/tmp/prior_pressure.json",
                source_selection_score_artifact_path="/tmp/default.json",
                source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_prior_pressure_source_artifact_mismatch(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["source_artifact"]["selected_artifact"]["selected_target"] = "/tmp/source/other-selected"

        with self.assertRaisesRegex(
            ValueError,
            "prior-pressure audit artifact source_artifact must match selection-score residual audit artifact source_artifact",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_prior_pressure_source_artifact_provenance_mismatch(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["source_artifact"]["artifact_path"] = "/tmp/other-upstream.json"

        with self.assertRaisesRegex(
            ValueError,
            "prior-pressure audit artifact source_artifact must match selection-score residual audit artifact source_artifact",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_default_trace_source_artifact_mismatch(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["source_artifact"]["reference_move"] = 4

        with self.assertRaisesRegex(
            ValueError,
            "default trace artifact source_artifact.reference_move must be 2",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_relaxed_trace_source_artifact_mismatch(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        relaxed_trace["source_artifact"]["full_search_selected_move"] = 4

        with self.assertRaisesRegex(
            ValueError,
            "relaxed trace artifact source_artifact.full_search_selected_move must be 0",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_supplied_canonicalization_artifact_wrong_schema(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()

        with self.assertRaisesRegex(ValueError, "checkpoint canonicalization artifact has wrong schema"):
            module.build_payload(
                residual_audit,
                prior_pressure,
                default_trace,
                relaxed_trace,
                source_selection_score_residual_audit_artifact_path="/tmp/residual_audit.json",
                source_prior_pressure_audit_artifact_path="/tmp/prior_pressure.json",
                source_selection_score_artifact_path="/tmp/default.json",
                source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
                checkpoint_canonicalization_artifact={"schema": "wrong"},
            )

    def test_build_payload_echoes_supplied_canonicalization_path_into_input_artifacts(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()

        payload = module.build_payload(
            residual_audit,
            prior_pressure,
            default_trace,
            relaxed_trace,
            source_selection_score_residual_audit_artifact_path="/tmp/residual_audit.json",
            source_prior_pressure_audit_artifact_path="/tmp/prior_pressure.json",
            source_selection_score_artifact_path="/tmp/default.json",
            source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
            checkpoint_canonicalization_artifact={
                "schema": module.SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA,
                "decision": module.EXPECTED_CANONICALIZATION_DECISION,
                "source_artifact": copy.deepcopy(residual_audit["source_artifact"]),
                "canonical_checkpoint_sequences": {"default": [1.0, 2.0], "relaxed": [1.0, 2.0]},
                "canonicalization_status": {"safe_for_followup_spec": True},
                "canonical_sequences_match": True,
            },
            source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
        )

        self.assertEqual(
            "/tmp/canonicalization.json",
            payload["input_artifacts"]["source_checkpoint_canonicalization_artifact_path"],
        )

    def test_build_payload_rejects_canonicalization_path_without_canonicalization_artifact(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()

        with self.assertRaisesRegex(
            ValueError,
            "source_checkpoint_canonicalization_artifact_path requires checkpoint canonicalization artifact",
        ):
            module.build_payload(
                residual_audit,
                prior_pressure,
                default_trace,
                relaxed_trace,
                source_selection_score_residual_audit_artifact_path="/tmp/residual_audit.json",
                source_prior_pressure_audit_artifact_path="/tmp/prior_pressure.json",
                source_selection_score_artifact_path="/tmp/default.json",
                source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
            )

    def test_build_payload_rejects_canonicalization_artifact_without_canonicalization_path(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint canonicalization artifact requires source_checkpoint_canonicalization_artifact_path",
        ):
            module.build_payload(
                residual_audit,
                prior_pressure,
                default_trace,
                relaxed_trace,
                source_selection_score_residual_audit_artifact_path="/tmp/residual_audit.json",
                source_prior_pressure_audit_artifact_path="/tmp/prior_pressure.json",
                source_selection_score_artifact_path="/tmp/default.json",
                source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
                checkpoint_canonicalization_artifact={
                    "schema": module.SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA,
                    "decision": module.EXPECTED_CANONICALIZATION_DECISION,
                    "source_artifact": copy.deepcopy(residual_audit["source_artifact"]),
                    "canonical_checkpoint_sequences": {"default": [1.0, 2.0], "relaxed": [1.0, 2.0]},
                    "canonicalization_status": {"safe_for_followup_spec": True},
                    "canonical_sequences_match": True,
                },
            )

    def test_build_payload_rejects_wrong_canonicalization_decision(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()

        with self.assertRaisesRegex(
            ValueError,
            f"checkpoint canonicalization artifact decision must be {module.EXPECTED_CANONICALIZATION_DECISION}",
        ):
            module.build_payload(
                residual_audit,
                prior_pressure,
                default_trace,
                relaxed_trace,
                source_selection_score_residual_audit_artifact_path="/tmp/residual_audit.json",
                source_prior_pressure_audit_artifact_path="/tmp/prior_pressure.json",
                source_selection_score_artifact_path="/tmp/default.json",
                source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
                checkpoint_canonicalization_artifact={
                    "schema": module.SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA,
                    "decision": "wrong",
                    "source_artifact": copy.deepcopy(residual_audit["source_artifact"]),
                    "canonical_checkpoint_sequences": {"default": [1.0, 2.0], "relaxed": [1.0, 2.0]},
                },
            )

    def test_build_payload_rejects_canonicalization_source_identity_mismatch(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        mismatched_source_artifact = copy.deepcopy(residual_audit["source_artifact"])
        mismatched_source_artifact["selected_artifact"]["selected_target"] = "/tmp/source/mismatch"

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint canonicalization artifact source_artifact must match selection-score residual audit artifact source_artifact",
        ):
            module.build_payload(
                residual_audit,
                prior_pressure,
                default_trace,
                relaxed_trace,
                source_selection_score_residual_audit_artifact_path="/tmp/residual_audit.json",
                source_prior_pressure_audit_artifact_path="/tmp/prior_pressure.json",
                source_selection_score_artifact_path="/tmp/default.json",
                source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
                checkpoint_canonicalization_artifact={
                    "schema": module.SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA,
                    "decision": module.EXPECTED_CANONICALIZATION_DECISION,
                    "source_artifact": mismatched_source_artifact,
                    "canonical_checkpoint_sequences": {"default": [1.0, 2.0], "relaxed": [1.0, 2.0]},
                },
            )

    def test_build_payload_rejects_canonicalization_artifact_missing_canonical_2_sequence(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint canonicalization artifact canonical_checkpoint_sequences must include canonical simulation 2.0",
        ):
            module.build_payload(
                residual_audit,
                prior_pressure,
                default_trace,
                relaxed_trace,
                source_selection_score_residual_audit_artifact_path="/tmp/residual_audit.json",
                source_prior_pressure_audit_artifact_path="/tmp/prior_pressure.json",
                source_selection_score_artifact_path="/tmp/default.json",
                source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
                checkpoint_canonicalization_artifact={
                    "schema": module.SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA,
                    "decision": module.EXPECTED_CANONICALIZATION_DECISION,
                    "source_artifact": copy.deepcopy(residual_audit["source_artifact"]),
                    "canonical_checkpoint_sequences": {"default": [1.0], "relaxed": [1.0]},
                    "canonicalization_status": {"safe_for_followup_spec": True},
                    "canonical_sequences_match": True,
                },
            )

    def test_build_payload_rejects_missing_canonicalization_safe_for_followup_spec(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint canonicalization artifact canonicalization_status.safe_for_followup_spec must be true",
        ):
            module.build_payload(
                residual_audit,
                prior_pressure,
                default_trace,
                relaxed_trace,
                source_selection_score_residual_audit_artifact_path="/tmp/residual_audit.json",
                source_prior_pressure_audit_artifact_path="/tmp/prior_pressure.json",
                source_selection_score_artifact_path="/tmp/default.json",
                source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
                checkpoint_canonicalization_artifact={
                    "schema": module.SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA,
                    "decision": module.EXPECTED_CANONICALIZATION_DECISION,
                    "source_artifact": copy.deepcopy(residual_audit["source_artifact"]),
                    "canonical_checkpoint_sequences": {"default": [1.0, 2.0], "relaxed": [1.0, 2.0]},
                    "canonicalization_status": {},
                    "canonical_sequences_match": True,
                },
            )

    def test_build_payload_rejects_false_canonical_sequences_match(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint canonicalization artifact canonical_sequences_match must be true",
        ):
            module.build_payload(
                residual_audit,
                prior_pressure,
                default_trace,
                relaxed_trace,
                source_selection_score_residual_audit_artifact_path="/tmp/residual_audit.json",
                source_prior_pressure_audit_artifact_path="/tmp/prior_pressure.json",
                source_selection_score_artifact_path="/tmp/default.json",
                source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
                checkpoint_canonicalization_artifact={
                    "schema": module.SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA,
                    "decision": module.EXPECTED_CANONICALIZATION_DECISION,
                    "source_artifact": copy.deepcopy(residual_audit["source_artifact"]),
                    "canonical_checkpoint_sequences": {"default": [1.0, 2.0], "relaxed": [1.0, 2.0]},
                    "canonicalization_status": {"safe_for_followup_spec": True},
                    "canonical_sequences_match": False,
                },
            )

    def test_build_payload_rejects_canonical_sequences_that_do_not_align_with_supplied_traces(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint canonicalization artifact canonical checkpoint sequences must align with supplied traces",
        ):
            module.build_payload(
                residual_audit,
                prior_pressure,
                default_trace,
                relaxed_trace,
                source_selection_score_residual_audit_artifact_path="/tmp/residual_audit.json",
                source_prior_pressure_audit_artifact_path="/tmp/prior_pressure.json",
                source_selection_score_artifact_path="/tmp/default.json",
                source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
                checkpoint_canonicalization_artifact={
                    "schema": module.SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA,
                    "decision": module.EXPECTED_CANONICALIZATION_DECISION,
                    "source_artifact": copy.deepcopy(residual_audit["source_artifact"]),
                    "canonical_checkpoint_sequences": {"default": [2.0], "relaxed": [2.0]},
                    "canonicalization_status": {"safe_for_followup_spec": True},
                    "canonical_sequences_match": True,
                },
            )

    def test_build_payload_rejects_conflicting_duplicate_checkpoint_with_valid_canonicalization(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        duplicate_default_trace_point = copy.deepcopy(default_trace["trace_points"][1])
        duplicate_default_trace_point["moves"][0]["selection_score"] = 0.08
        default_trace["trace_points"].append(duplicate_default_trace_point)

        with self.assertRaisesRegex(
            ValueError,
            "default raw duplicate 2.0 checkpoint must match canonical projection",
        ):
            module.build_payload(
                residual_audit,
                prior_pressure,
                default_trace,
                relaxed_trace,
                source_selection_score_residual_audit_artifact_path="/tmp/residual_audit.json",
                source_prior_pressure_audit_artifact_path="/tmp/prior_pressure.json",
                source_selection_score_artifact_path="/tmp/default.json",
                source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
                checkpoint_canonicalization_artifact={
                    "schema": module.SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA,
                    "decision": module.EXPECTED_CANONICALIZATION_DECISION,
                    "source_artifact": copy.deepcopy(residual_audit["source_artifact"]),
                    "canonical_checkpoint_sequences": {"default": [1.0, 2.0], "relaxed": [1.0, 2.0]},
                    "canonicalization_status": {"safe_for_followup_spec": True},
                    "canonical_sequences_match": True,
                },
            )

    def test_build_payload_accepts_equivalent_duplicate_checkpoint_with_valid_canonicalization(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        duplicate_default_trace_point = copy.deepcopy(default_trace["trace_points"][1])
        duplicate_default_trace_point["simulation"] = 2
        duplicate_default_trace_point["visits"] = [2, 0, 1, 0, 0]
        duplicate_default_trace_point["moves"][0]["selection_score"] = 0.07
        duplicate_default_trace_point["moves"][0]["q_value"] = -0.01
        duplicate_default_trace_point["moves"][1]["selection_score"] = 0
        duplicate_default_trace_point["moves"][1]["q_value"] = 0
        default_trace["trace_points"].append(duplicate_default_trace_point)

        payload = module.build_payload(
            residual_audit,
            prior_pressure,
            default_trace,
            relaxed_trace,
            source_selection_score_residual_audit_artifact_path="/tmp/residual_audit.json",
            source_prior_pressure_audit_artifact_path="/tmp/prior_pressure.json",
            source_selection_score_artifact_path="/tmp/default.json",
            source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
            source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
            checkpoint_canonicalization_artifact={
                "schema": module.SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA,
                "decision": module.EXPECTED_CANONICALIZATION_DECISION,
                "source_artifact": copy.deepcopy(residual_audit["source_artifact"]),
                "canonical_checkpoint_sequences": {"default": [1.0, 2.0], "relaxed": [1.0, 2.0]},
                "canonicalization_status": {"safe_for_followup_spec": True},
                "canonical_sequences_match": True,
            },
        )

        self.assertEqual(module.SCHEMA, payload["schema"])

    def test_build_payload_rejects_non_contiguous_duplicate_checkpoint_with_valid_canonicalization(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["trace_points"] = [
            default_trace["trace_points"][0],
            default_trace["trace_points"][1],
            self.trace_point(
                simulation=3.0,
                moves=[
                    {"move": 0, "selection_score": 0.08, "q_value": -0.01},
                    {"move": 2, "selection_score": 0.0, "q_value": 0.0},
                ],
                visits=[3.0, 0.0, 1.0, 0.0, 0.0],
            ),
            copy.deepcopy(default_trace["trace_points"][1]),
        ]

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint sequences must be strictly increasing",
        ):
            module.build_payload(
                residual_audit,
                prior_pressure,
                default_trace,
                relaxed_trace,
                source_selection_score_residual_audit_artifact_path="/tmp/residual_audit.json",
                source_prior_pressure_audit_artifact_path="/tmp/prior_pressure.json",
                source_selection_score_artifact_path="/tmp/default.json",
                source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
                source_checkpoint_canonicalization_artifact_path="/tmp/canonicalization.json",
                checkpoint_canonicalization_artifact={
                    "schema": module.SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA,
                    "decision": module.EXPECTED_CANONICALIZATION_DECISION,
                    "source_artifact": copy.deepcopy(residual_audit["source_artifact"]),
                    "canonical_checkpoint_sequences": {"default": [1.0, 2.0, 3.0], "relaxed": [1.0, 2.0]},
                    "canonicalization_status": {"safe_for_followup_spec": True},
                    "canonical_sequences_match": True,
                },
            )

    def test_build_payload_rejects_source_artifact_metadata_drift(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["source_artifact"]["metadata_note"] = "harmless"
        default_trace["source_artifact"]["trace_label"] = "default-copy"
        relaxed_trace["source_artifact"]["trace_label"] = "relaxed-copy"

        with self.assertRaisesRegex(
            ValueError,
            "prior-pressure audit artifact source_artifact must match selection-score residual audit artifact source_artifact",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_boolean_prior_pressure_reference_move(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["source_artifact"]["reference_move"] = False

        with self.assertRaisesRegex(
            ValueError,
            "prior-pressure audit artifact source_artifact.reference_move must be an integer",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_boolean_default_trace_reference_move(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        default_trace["source_artifact"]["reference_move"] = False

        with self.assertRaisesRegex(
            ValueError,
            "default trace artifact source_artifact.reference_move must be an integer",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_boolean_prior_pressure_full_search_selected_move(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        prior_pressure["source_artifact"]["full_search_selected_move"] = True

        with self.assertRaisesRegex(
            ValueError,
            "prior-pressure audit artifact source_artifact.full_search_selected_move must be an integer",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )

    def test_build_payload_rejects_boolean_relaxed_trace_full_search_selected_move(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()
        relaxed_trace["source_artifact"]["full_search_selected_move"] = True

        with self.assertRaisesRegex(
            ValueError,
            "relaxed trace artifact source_artifact.full_search_selected_move must be an integer",
        ):
            self.build_payload(
                residual_audit=residual_audit,
                prior_pressure=prior_pressure,
                default_trace=default_trace,
                relaxed_trace=relaxed_trace,
            )


class Capture002ResidualAblationMainTest(
    Capture002ResidualAblationTestSupport,
    unittest.TestCase,
):
    def write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_main_writes_sorted_json_to_out_and_prints_compact_summary(self):
        residual_audit, prior_pressure, default_trace, relaxed_trace = self.valid_inputs()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            residual_audit_path = tmpdir_path / "residual_audit.json"
            prior_pressure_path = tmpdir_path / "prior_pressure.json"
            default_trace_path = tmpdir_path / "default_trace.json"
            relaxed_trace_path = tmpdir_path / "relaxed_trace.json"
            out_path = tmpdir_path / "nested" / "out.json"
            residual_audit["input_artifacts"]["source_prior_pressure_audit_artifact_path"] = str(
                prior_pressure_path
            )
            residual_audit["input_artifacts"]["source_selection_score_artifact_path"] = str(
                default_trace_path
            )
            residual_audit["input_artifacts"][
                "source_threshold_relaxed_selection_score_artifact_path"
            ] = str(relaxed_trace_path)
            self.write_json(residual_audit_path, residual_audit)
            self.write_json(prior_pressure_path, prior_pressure)
            self.write_json(default_trace_path, default_trace)
            self.write_json(relaxed_trace_path, relaxed_trace)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = module.main(
                    [
                        "--source-selection-score-residual-audit-artifact",
                        str(residual_audit_path),
                        "--source-prior-pressure-audit-artifact",
                        str(prior_pressure_path),
                        "--source-selection-score-artifact",
                        str(default_trace_path),
                        "--source-threshold-relaxed-selection-score-artifact",
                        str(relaxed_trace_path),
                        "--out",
                        str(out_path),
                    ]
                )

            written_text = out_path.read_text(encoding="utf-8")

        self.assertEqual(0, exit_code)
        self.assertEqual(
            json.dumps(json.loads(written_text), indent=2, sort_keys=True) + "\n",
            written_text,
        )
        self.assertEqual(
            {
                "artifact_path": str(out_path),
                "classification": "selected_move_residual_insensitive",
                "decision": "write_002_non_residual_mechanism_review_spec",
                "schema": module.SCHEMA,
            },
            json.loads(stdout.getvalue()),
        )
        self.assertEqual(stdout.getvalue().strip(), stdout.getvalue().rstrip("\n"))
        self.assertNotIn("\n", stdout.getvalue().strip())


class Capture002ResidualAblationLiveArtifactSmokeTest(unittest.TestCase):
    def _discover_live_residual_audit_artifact_paths(self) -> list[Path]:
        repo_root = Path(__file__).resolve().parents[2]
        search_roots = [repo_root / "artifacts", repo_root / "tmp"]
        matching_paths = []

        for root in search_roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if (
                    isinstance(payload, dict)
                    and payload.get("schema") == module.SOURCE_SELECTION_SCORE_RESIDUAL_AUDIT_SCHEMA
                    and payload.get("source_artifact", {}).get("row_id") == module.ROW_ID
                ):
                    matching_paths.append(path)

        return matching_paths

    def test_live_row_002_residual_audit_artifact_is_discoverable(self):
        matching_paths = self._discover_live_residual_audit_artifact_paths()

        if not matching_paths:
            self.skipTest(
                "Missing live row-002 residual-audit artifact; expected one under artifacts/ or tmp/."
            )

        self.assertTrue(any(path.exists() for path in matching_paths))

    def test_live_row_002_residual_audit_artifact_has_allowed_harness_outcome(self):
        matching_paths = self._discover_live_residual_audit_artifact_paths()

        if not matching_paths:
            self.skipTest(
                "Missing live row-002 residual-audit artifact; expected one under artifacts/ or tmp/."
            )

        allowed_classifications = {
            "selected_move_residual_sensitive",
            "selected_move_residual_insensitive",
            "selected_move_residual_ablation_inconclusive",
        }
        allowed_validation_error_fragments = (
            "requires canonicalization artifact",
            "must match selection-score residual audit artifact source_artifact",
            "must be 2",
        )

        for residual_audit_path in matching_paths:
            with self.subTest(residual_audit_path=str(residual_audit_path)):
                residual_audit = json.loads(residual_audit_path.read_text(encoding="utf-8"))
                input_artifacts = residual_audit.get("input_artifacts", {})
                try:
                    payload = module.build_payload(
                        residual_audit,
                        module.load_json(Path(input_artifacts["source_prior_pressure_audit_artifact_path"])),
                        module.load_json(Path(input_artifacts["source_selection_score_artifact_path"])),
                        module.load_json(Path(input_artifacts["source_threshold_relaxed_selection_score_artifact_path"])),
                        source_selection_score_residual_audit_artifact_path=str(residual_audit_path),
                        source_prior_pressure_audit_artifact_path=input_artifacts[
                            "source_prior_pressure_audit_artifact_path"
                        ],
                        source_selection_score_artifact_path=input_artifacts[
                            "source_selection_score_artifact_path"
                        ],
                        source_threshold_relaxed_selection_score_artifact_path=input_artifacts[
                            "source_threshold_relaxed_selection_score_artifact_path"
                        ],
                    )
                except (FileNotFoundError, KeyError, ValueError) as error:
                    self.assertTrue(
                        isinstance(error, (FileNotFoundError, KeyError))
                        or any(fragment in str(error) for fragment in allowed_validation_error_fragments),
                        str(error),
                    )
                    continue

                self.assertIn(payload["classification"]["classification"], allowed_classifications)


class Capture002ResidualAblationLiveArtifactSmokeBehaviorTest(unittest.TestCase):
    def test_live_artifact_smoke_test_skips_when_no_artifact_is_available(self):
        smoke_test = Capture002ResidualAblationLiveArtifactSmokeTest(
            "test_live_row_002_residual_audit_artifact_is_discoverable"
        )

        with mock.patch.object(Path, "exists", return_value=False):
            with self.assertRaises(unittest.SkipTest):
                smoke_test.test_live_row_002_residual_audit_artifact_is_discoverable()

    def test_live_artifact_smoke_test_does_not_allow_selected_move_metadata_mismatch(self):
        smoke_test = Capture002ResidualAblationLiveArtifactSmokeTest(
            "test_live_row_002_residual_audit_artifact_has_allowed_harness_outcome"
        )
        fake_path = Path("/tmp/residual_audit.json")
        residual_audit = {
            "input_artifacts": {
                "source_prior_pressure_audit_artifact_path": "/tmp/prior_pressure.json",
                "source_selection_score_artifact_path": "/tmp/default.json",
                "source_threshold_relaxed_selection_score_artifact_path": "/tmp/relaxed.json",
            }
        }

        with mock.patch.object(
            Capture002ResidualAblationLiveArtifactSmokeTest,
            "_discover_live_residual_audit_artifact_paths",
            return_value=[fake_path],
        ):
            with mock.patch.object(Path, "read_text", return_value=json.dumps(residual_audit)):
                with mock.patch.object(module, "load_json", return_value={}):
                    with mock.patch.object(
                        module,
                        "build_payload",
                        side_effect=ValueError("default canonical trace point.selected_move must be 0"),
                    ):
                        with self.assertRaises(AssertionError):
                            smoke_test.test_live_row_002_residual_audit_artifact_has_allowed_harness_outcome()


class Capture002ResidualAblationDiscoveryGuardTest(unittest.TestCase):
    def test_validation_suite_retains_expected_test_methods(self):
        validation_methods = {
            name
            for name in dir(Capture002ResidualAblationValidationTest)
            if name.startswith("test_")
        }

        self.assertIn(
            "test_build_payload_rejects_wrong_prior_pressure_schema",
            validation_methods,
        )


if __name__ == "__main__":
    unittest.main()
