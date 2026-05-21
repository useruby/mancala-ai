import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import capture_002_trace_cadence_capture as module


class Capture002TraceCadenceCaptureContractTest(unittest.TestCase):
    def test_contract_constants_are_stable(self):
        self.assertEqual("azlite_capture_002_trace_cadence_capture_v1", module.SCHEMA)
        self.assertEqual(
            "azlite_capture_002_trace_capture_v1", module.SOURCE_TRACE_CAPTURE_SCHEMA
        )
        self.assertEqual(
            "azlite_capture_002_selection_score_trace_v1",
            module.SOURCE_SELECTION_SCORE_SCHEMA,
        )
        self.assertEqual(
            "azlite_capture_002_trace_cadence_review_v1",
            module.SOURCE_TRACE_CADENCE_REVIEW_SCHEMA,
        )
        self.assertEqual(
            "azlite_capture_002_nonseparable_review_v1",
            module.SOURCE_NONSEPARABLE_REVIEW_SCHEMA,
        )
        self.assertEqual("capture_available-002", module.ROW_ID)
        self.assertEqual(
            {
                "trace_cadence_unresolved": "stop_002_trace_cadence_unresolved",
                "selection_score_pressure_confirmed": "write_002_selection_pressure_ablation_spec",
                "q_support_precedes_selection_score": "write_002_child_value_audit_spec",
                "genuinely_not_separable": "stop_002_unresolved",
            },
            module.CLASSIFICATION_DECISIONS,
        )

    def test_parse_args_reads_required_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_capture_path = Path(tmp) / "capture_002_trace_capture.json"
            selection_score_path = Path(tmp) / "capture_002_selection_score_trace.json"
            cadence_review_path = Path(tmp) / "capture_002_trace_cadence_review.json"
            nonseparable_path = Path(tmp) / "capture_002_nonseparable_review.json"
            out_path = Path(tmp) / "capture_002_trace_cadence_capture.json"

            args = module.parse_args(
                [
                    "--source-trace-capture-artifact",
                    str(trace_capture_path),
                    "--source-selection-score-artifact",
                    str(selection_score_path),
                    "--source-trace-cadence-review-artifact",
                    str(cadence_review_path),
                    "--source-nonseparable-review-artifact",
                    str(nonseparable_path),
                    "--out",
                    str(out_path),
                ]
            )

        self.assertEqual(trace_capture_path, args.source_trace_capture_artifact)
        self.assertEqual(selection_score_path, args.source_selection_score_artifact)
        self.assertEqual(cadence_review_path, args.source_trace_cadence_review_artifact)
        self.assertEqual(nonseparable_path, args.source_nonseparable_review_artifact)
        self.assertEqual(out_path, args.out)

    def test_parse_args_requires_baseline_inputs_and_out(self):
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
                    "--source-trace-cadence-review-artifact",
                    "/tmp/cadence.json",
                ]
            )


class Capture002TraceCadenceCaptureBaselineTest(unittest.TestCase):
    def write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def trace_capture_artifact(self) -> dict:
        return {
            "schema": module.SOURCE_TRACE_CAPTURE_SCHEMA,
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
                    "moves": [],
                },
                {
                    "simulation": 1.0,
                    "selected_move": 2,
                    "reference_move_by_prior": 2,
                    "visits": [0.0, 0.0, 1.0, 0.0, 0.0],
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
            "insufficiency_reasons": [],
            "row_context": {
                "row_id": "capture_available-002",
                "canonical_state": '{"player_pits":[1,0,7,6,6,5],"opponent_pits":[5,4,4,4,4,0]}',
                "legal_moves": [0, 1, 2, 3, 4],
                "reference_move": 2,
                "full_search_selected_move": 0,
            },
            "upstream_inputs": {
                "seed": 17,
                "simulation_count": 384,
                "search_settings": {
                    "c_puct": 1.25,
                    "fpu_mode": "zero",
                    "normalize_values": False,
                    "reuse_subtree": False,
                    "root_policy_mode": "deterministic",
                    "tactical_root_bias": 0.1,
                },
                "reason": None,
            },
            "source_shared_drift_artifact": {
                "artifact_path": "/tmp/source/shared_drift.json",
                "artifact_sha256": "source-sha",
                "selected_artifact": {
                    "path": "/tmp/source/selected",
                    "selected_artifact": "/tmp/source/selected_artifact",
                    "selected_target": "/tmp/source/selected",
                    "provenance_source": "selection_manifest.selected_target",
                },
                "source_payload": {
                    "schema": "azlite_shared_full_search_drift_diagnostic_v1",
                    "classification": {"classification": "shared_mechanism_disproved"},
                    "decision": "write_row_split_followup_spec",
                    "selected_artifact": {
                        "path": "/tmp/source/selected",
                        "selected_artifact": "/tmp/source/selected_artifact",
                        "selected_target": "/tmp/source/selected",
                        "provenance_source": "selection_manifest.selected_target",
                    },
                    "settings": {
                        "seed": 17,
                        "simulation_count": 384,
                        "search_settings": {
                            "c_puct": 1.25,
                            "fpu_mode": "zero",
                            "normalize_values": False,
                            "reuse_subtree": False,
                            "root_policy_mode": "deterministic",
                            "tactical_root_bias": 0.1,
                        },
                    },
                    "paired_summary": {
                        "probe_mode_failure_paths": {
                            "capture_available-002": {
                                "policy_only": "reference_kept",
                                "value_only": "reference_kept",
                                "full_search": "full_search_drift",
                            },
                            "capture_available-003": {
                                "policy_only": "reference_kept",
                                "value_only": "diverged_before_full_search",
                                "full_search": "diverged_before_full_search",
                            },
                        },
                        "probe_mode_selected_moves": {
                            "capture_available-002": {
                                "policy_only": 2,
                                "value_only": 2,
                                "full_search": 0,
                            },
                            "capture_available-003": {
                                "policy_only": 2,
                                "value_only": 1,
                                "full_search": 1,
                            },
                        },
                        "shared_mechanism_supported": False,
                    },
                    "rows": {
                        "capture_available-002": {
                            "row_id": "capture_available-002",
                            "canonical_state": '{"player_pits":[1,0,7,6,6,5],"opponent_pits":[5,4,4,4,4,0]}',
                            "legal_moves": [0, 1, 2, 3, 4],
                            "reference_move": 2,
                            "full_search_selected_move": 0,
                            "root_start": {
                                "simulation": 1.0,
                                "selected_move": 2,
                                "reference_move_by_prior": 2,
                                "visits": [0.0, 0.0, 1.0, 0.0, 0.0],
                                "moves": [],
                            },
                            "snapshots": [
                                {
                                    "simulation": 1.0,
                                    "selected_move": 2,
                                    "reference_move_by_prior": 2,
                                    "visits": [0.0, 0.0, 1.0, 0.0, 0.0],
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
                            "final_deltas": {
                                "selected_visits": 8.0,
                                "reference_visits": 6.0,
                            },
                            "missing_fields": [],
                            "probe_mode_traces": {
                                "policy_only": {"selected_move": 2},
                                "value_only": {"selected_move": 2},
                                "full_search": {"selected_move": 0},
                            },
                        },
                        "capture_available-003": {
                            "row_id": "capture_available-003",
                            "canonical_state": '{"player_pits":[1,6,0,6,6,5],"opponent_pits":[5,5,4,4,4,0]}',
                            "legal_moves": [0, 1, 2, 3, 4],
                            "reference_move": 2,
                            "full_search_selected_move": 1,
                            "root_start": {
                                "simulation": 1.0,
                                "selected_move": 2,
                                "reference_move_by_prior": 2,
                                "visits": [0.0, 0.0, 1.0, 0.0, 0.0],
                                "moves": [],
                            },
                            "snapshots": [
                                {
                                    "simulation": 16.0,
                                    "selected_move": 1,
                                    "reference_move_by_prior": 2,
                                    "visits": [2.0, 8.0, 7.0, 1.0, 1.0],
                                    "moves": [],
                                }
                            ],
                            "final_deltas": {
                                "selected_visits": 8.0,
                                "reference_visits": 7.0,
                            },
                            "missing_fields": [],
                            "probe_mode_traces": {
                                "policy_only": {"selected_move": 2},
                                "value_only": {"selected_move": 1},
                                "full_search": {"selected_move": 1},
                            },
                        },
                    },
                },
            },
        }

    def selection_score_artifact(self) -> dict:
        return {
            "schema": module.SOURCE_SELECTION_SCORE_SCHEMA,
            "trace_origin": "extracted",
            "source_artifact": {
                "artifact_path": "/tmp/source/shared_drift.json",
                "schema": "azlite_shared_full_search_drift_diagnostic_v1",
                "decision": "write_row_split_followup_spec",
                "classification": {"classification": "shared_mechanism_disproved"},
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
            "classification": {
                "classification": "unresolved",
                "evidence_summary": "Selection-score and Q-support timing do not cleanly separate for capture 002.",
            },
            "decision": "write_002_unresolved_trace_review_spec",
            "insufficiency_reasons": [],
            "final_selected_minus_reference_selection_score": -0.01,
            "final_selected_minus_reference_q": 0.01,
            "final_selected_minus_reference_visit_share": 0.04545454545454547,
        }

    def trace_cadence_review_artifact(self) -> dict:
        return {
            "schema": module.SOURCE_TRACE_CADENCE_REVIEW_SCHEMA,
            "classification": {
                "classification": "trace_too_sparse",
                "evidence_summary": "Trace cadence is too sparse to localize the first crossing event for the validated 002 trace.",
            },
            "decision": "write_002_trace_cadence_capture_spec",
            "unique_simulation_checkpoints": [1.0, 16.0],
            "duplicate_root_snapshot_count": 1,
            "ambiguity_signals": [
                "selected_move_changed_without_captured_crossing",
                "near_material_visit_share_threshold",
            ],
        }

    def nonseparable_review_artifact(self) -> dict:
        return {
            "schema": module.SOURCE_NONSEPARABLE_REVIEW_SCHEMA,
            "hypothesis": "genuinely_not_separable",
            "classification": {
                "classification": "prerequisite_preempted",
                "evidence_summary": "Cadence review already supported a narrower branch before threshold or non-separable review.",
            },
            "decision": "write_002_trace_cadence_capture_spec",
        }

    def test_load_baseline_artifacts_accepts_sparse_trigger_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace_capture.json"
            selection_path = Path(tmp) / "selection_score.json"
            cadence_path = Path(tmp) / "cadence_review.json"
            nonseparable_path = Path(tmp) / "nonseparable_review.json"
            self.write_json(trace_path, self.trace_capture_artifact())
            self.write_json(selection_path, self.selection_score_artifact())
            self.write_json(cadence_path, self.trace_cadence_review_artifact())
            self.write_json(nonseparable_path, self.nonseparable_review_artifact())

            baseline = module.load_baseline_inputs(
                trace_capture_artifact_path=trace_path,
                selection_score_artifact_path=selection_path,
                trace_cadence_review_artifact_path=cadence_path,
                nonseparable_review_artifact_path=nonseparable_path,
            )

        self.assertEqual("capture_available-002", baseline["row_context"]["row_id"])
        self.assertEqual(
            [1.0, 16.0], baseline["baseline_trigger"]["baseline_unique_checkpoint_list"]
        )
        self.assertEqual(
            [
                "selected_move_changed_without_captured_crossing",
                "near_material_visit_share_threshold",
            ],
            baseline["baseline_trigger"]["baseline_ambiguity_signals"],
        )

    def test_load_baseline_artifacts_preserves_zero_selected_move_from_row_context(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace_capture.json"
            selection_path = Path(tmp) / "selection_score.json"
            cadence_path = Path(tmp) / "cadence_review.json"
            trace_artifact = self.trace_capture_artifact()
            trace_artifact["full_search_selected_move"] = 4
            self.write_json(trace_path, trace_artifact)
            self.write_json(selection_path, self.selection_score_artifact())
            self.write_json(cadence_path, self.trace_cadence_review_artifact())

            baseline = module.load_baseline_inputs(
                trace_capture_artifact_path=trace_path,
                selection_score_artifact_path=selection_path,
                trace_cadence_review_artifact_path=cadence_path,
                nonseparable_review_artifact_path=None,
            )

        self.assertEqual(0, baseline["row_context"]["full_search_selected_move"])

    def test_load_baseline_artifacts_rejects_selection_score_source_artifact_move_pair_mismatch(
        self,
    ):
        for mismatched_field, mismatched_value in (
            ("reference_move", 3),
            ("full_search_selected_move", 4),
        ):
            with self.subTest(mismatched_field=mismatched_field):
                with tempfile.TemporaryDirectory() as tmp:
                    trace_path = Path(tmp) / "trace_capture.json"
                    selection_path = Path(tmp) / "selection_score.json"
                    cadence_path = Path(tmp) / "cadence_review.json"
                    selection_artifact = self.selection_score_artifact()
                    selection_artifact["source_artifact"][mismatched_field] = (
                        mismatched_value
                    )
                    self.write_json(trace_path, self.trace_capture_artifact())
                    self.write_json(selection_path, selection_artifact)
                    self.write_json(cadence_path, self.trace_cadence_review_artifact())

                    with self.assertRaisesRegex(
                        ValueError,
                        "selection score artifact source_artifact move pair must match trace capture baseline",
                    ):
                        module.load_baseline_inputs(
                            trace_capture_artifact_path=trace_path,
                            selection_score_artifact_path=selection_path,
                            trace_cadence_review_artifact_path=cadence_path,
                            nonseparable_review_artifact_path=None,
                        )

    def test_load_baseline_artifacts_rejects_missing_selection_score_source_row_id(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace_capture.json"
            selection_path = Path(tmp) / "selection_score.json"
            cadence_path = Path(tmp) / "cadence_review.json"
            selection_artifact = self.selection_score_artifact()
            selection_artifact["source_artifact"]["row_id"] = None
            self.write_json(trace_path, self.trace_capture_artifact())
            self.write_json(selection_path, selection_artifact)
            self.write_json(cadence_path, self.trace_cadence_review_artifact())

            with self.assertRaisesRegex(
                ValueError,
                "selection score artifact source_artifact.row_id must be capture_available-002",
            ):
                module.load_baseline_inputs(
                    trace_capture_artifact_path=trace_path,
                    selection_score_artifact_path=selection_path,
                    trace_cadence_review_artifact_path=cadence_path,
                    nonseparable_review_artifact_path=None,
                )

    def test_load_baseline_artifacts_rejects_non_sparse_cadence_trigger(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace_capture.json"
            selection_path = Path(tmp) / "selection_score.json"
            cadence_path = Path(tmp) / "cadence_review.json"
            self.write_json(trace_path, self.trace_capture_artifact())
            self.write_json(selection_path, self.selection_score_artifact())
            cadence_artifact = self.trace_cadence_review_artifact()
            cadence_artifact["classification"]["classification"] = "cadence_adequate"
            cadence_artifact["decision"] = "continue_002_threshold_too_strict_check"
            self.write_json(cadence_path, cadence_artifact)

            with self.assertRaisesRegex(
                ValueError,
                "trace cadence review artifact must classify trace_too_sparse",
            ):
                module.load_baseline_inputs(
                    trace_capture_artifact_path=trace_path,
                    selection_score_artifact_path=selection_path,
                    trace_cadence_review_artifact_path=cadence_path,
                    nonseparable_review_artifact_path=None,
                )

    def test_load_baseline_artifacts_rejects_non_002_row_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace_capture.json"
            selection_path = Path(tmp) / "selection_score.json"
            cadence_path = Path(tmp) / "cadence_review.json"
            trace_artifact = self.trace_capture_artifact()
            trace_artifact["row_id"] = "capture_available-003"
            trace_artifact["row_context"]["row_id"] = "capture_available-003"
            self.write_json(trace_path, trace_artifact)
            self.write_json(selection_path, self.selection_score_artifact())
            self.write_json(cadence_path, self.trace_cadence_review_artifact())

            with self.assertRaisesRegex(ValueError, "capture_available-002"):
                module.load_baseline_inputs(
                    trace_capture_artifact_path=trace_path,
                    selection_score_artifact_path=selection_path,
                    trace_cadence_review_artifact_path=cadence_path,
                    nonseparable_review_artifact_path=None,
                )

    def test_load_baseline_artifacts_rejects_preempted_nonseparable_review_with_wrong_decision(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace_capture.json"
            selection_path = Path(tmp) / "selection_score.json"
            cadence_path = Path(tmp) / "cadence_review.json"
            nonseparable_path = Path(tmp) / "nonseparable_review.json"
            self.write_json(trace_path, self.trace_capture_artifact())
            self.write_json(selection_path, self.selection_score_artifact())
            self.write_json(cadence_path, self.trace_cadence_review_artifact())
            nonseparable_artifact = self.nonseparable_review_artifact()
            nonseparable_artifact["decision"] = "stop_002_unresolved"
            self.write_json(nonseparable_path, nonseparable_artifact)

            with self.assertRaisesRegex(
                ValueError, "write_002_trace_cadence_capture_spec"
            ):
                module.load_baseline_inputs(
                    trace_capture_artifact_path=trace_path,
                    selection_score_artifact_path=selection_path,
                    trace_cadence_review_artifact_path=cadence_path,
                    nonseparable_review_artifact_path=nonseparable_path,
                )

    def test_provenance_guard_fails_when_selected_move_or_search_settings_change(self):
        baseline = {
            "row_context": {
                "row_id": "capture_available-002",
                "canonical_state": "state-002",
                "reference_move": 2,
                "full_search_selected_move": 0,
            },
            "source_selected_artifact": {
                "path": "/tmp/source/selected",
                "selected_artifact": "/tmp/source/selected_artifact",
                "selected_target": "/tmp/source/selected",
                "provenance_source": "selection_manifest.selected_target",
            },
            "search_settings": {
                "c_puct": 1.25,
                "fpu_mode": "zero",
                "normalize_values": False,
                "reuse_subtree": False,
                "root_policy_mode": "deterministic",
                "tactical_root_bias": 0.1,
            },
        }
        dense_context = {
            "row_id": "capture_available-002",
            "canonical_state": "state-002",
            "reference_move": 2,
            "full_search_selected_move": 1,
        }
        dense_settings = {
            "c_puct": 1.5,
            "fpu_mode": "zero",
            "normalize_values": False,
            "reuse_subtree": False,
            "root_policy_mode": "deterministic",
            "tactical_root_bias": 0.1,
        }

        guard = module.build_provenance_guard(
            baseline,
            dense_row_context=dense_context,
            dense_selected_artifact=baseline["source_selected_artifact"],
            dense_search_settings=dense_settings,
            checkpoint_capture_policy={
                "capture_mode": "dense_full",
                "checkpoint_schedule": [1, 4, 8, 12, 16],
            },
        )

        self.assertFalse(guard["passed"])
        self.assertIn("full_search_selected_move_mismatch", guard["failures"])
        self.assertIn(
            "decision_relevant_search_setting_changed:c_puct", guard["failures"]
        )

    def test_provenance_guard_fails_when_legal_moves_change_order(self):
        baseline = {
            "row_context": {
                "row_id": "capture_available-002",
                "canonical_state": "state-002",
                "legal_moves": [0, 1, 2, 3, 4],
                "reference_move": 2,
                "full_search_selected_move": 0,
            },
            "source_selected_artifact": {
                "path": "/tmp/source/selected",
                "selected_artifact": "/tmp/source/selected_artifact",
                "selected_target": "/tmp/source/selected",
                "provenance_source": "selection_manifest.selected_target",
            },
            "search_settings": {
                "c_puct": 1.25,
                "fpu_mode": "zero",
                "normalize_values": False,
                "reuse_subtree": False,
                "root_policy_mode": "deterministic",
                "tactical_root_bias": 0.1,
            },
        }

        guard = module.build_provenance_guard(
            baseline,
            dense_row_context={
                "row_id": "capture_available-002",
                "canonical_state": "state-002",
                "legal_moves": [1, 0, 2, 3, 4],
                "reference_move": 2,
                "full_search_selected_move": 0,
            },
            dense_selected_artifact=baseline["source_selected_artifact"],
            dense_search_settings=baseline["search_settings"],
            checkpoint_capture_policy={
                "capture_mode": "dense_full",
                "checkpoint_schedule": [1, 4, 8, 12, 16],
            },
        )

        self.assertFalse(guard["passed"])
        self.assertIn("legal_moves_mismatch", guard["failures"])


class Capture002TraceCadenceCaptureDenseTraceTest(
    Capture002TraceCadenceCaptureBaselineTest
):
    def dense_trace_points(self) -> list[dict]:
        return [
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

    def duplicate_root_dense_trace_points(self) -> list[dict]:
        return [
            {
                "simulation": 1.0,
                "selected_move": 2,
                "reference_move_by_prior": 2,
                "visits": [0.0, 0.0, 1.0, 0.0, 0.0],
                "moves": [],
            },
            {
                "simulation": 1.0,
                "selected_move": 2,
                "reference_move_by_prior": 2,
                "visits": [0.0, 0.0, 1.0, 0.0, 0.0],
                "moves": [],
            },
            {
                "simulation": 1.0,
                "selected_move": 2,
                "reference_move_by_prior": 2,
                "visits": [0.0, 0.0, 1.0, 0.0, 0.0],
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

    def sparse_dense_trace_points(self) -> list[dict]:
        return [
            {
                "simulation": 1.0,
                "selected_move": 2,
                "reference_move_by_prior": 2,
                "visits": [0.0, 0.0, 1.0, 0.0, 0.0],
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

    def test_capture_dense_trace_records_additional_checkpoint_between_divergence_and_final(
        self,
    ):
        baseline = {
            "row_context": self.trace_capture_artifact()["row_context"],
            "source_selected_artifact": self.trace_capture_artifact()[
                "source_shared_drift_artifact"
            ]["selected_artifact"],
            "search_settings": self.trace_capture_artifact()["upstream_inputs"][
                "search_settings"
            ],
        }

        dense_capture = module.build_dense_trace(
            baseline,
            dense_trace_points=self.dense_trace_points(),
            checkpoint_capture_policy={
                "capture_mode": "dense_full",
                "checkpoint_schedule": [1, 4, 8, 12, 16],
                "root_snapshot_deduplicated": True,
            },
        )

        self.assertEqual(
            [1.0, 4.0, 8.0, 12.0, 16.0], dense_capture["unique_simulation_checkpoints"]
        )
        self.assertTrue(
            dense_capture["has_additional_checkpoint_between_divergence_and_final"]
        )
        self.assertEqual([], dense_capture["insufficiency_reasons"])

    def test_capture_dense_trace_fails_when_no_additional_checkpoint_exists_between_divergence_and_final(
        self,
    ):
        baseline = {
            "row_context": self.trace_capture_artifact()["row_context"],
            "source_selected_artifact": self.trace_capture_artifact()[
                "source_shared_drift_artifact"
            ]["selected_artifact"],
            "search_settings": self.trace_capture_artifact()["upstream_inputs"][
                "search_settings"
            ],
        }

        dense_capture = module.build_dense_trace(
            baseline,
            dense_trace_points=self.sparse_dense_trace_points(),
            checkpoint_capture_policy={
                "capture_mode": "dense_full",
                "checkpoint_schedule": [1, 16],
                "root_snapshot_deduplicated": True,
            },
        )

        self.assertIn(
            "no additional checkpoint exists between the first divergent selected move and the final snapshot",
            dense_capture["insufficiency_reasons"],
        )

    def test_capture_dense_trace_handles_duplicate_root_checkpoints_before_divergence(
        self,
    ):
        baseline = {
            "row_context": self.trace_capture_artifact()["row_context"],
            "source_selected_artifact": self.trace_capture_artifact()[
                "source_shared_drift_artifact"
            ]["selected_artifact"],
            "search_settings": self.trace_capture_artifact()["upstream_inputs"][
                "search_settings"
            ],
        }

        dense_capture = module.build_dense_trace(
            baseline,
            dense_trace_points=self.duplicate_root_dense_trace_points(),
            checkpoint_capture_policy={
                "capture_mode": "dense_full",
                "checkpoint_schedule": [1, 8, 12, 16],
                "root_snapshot_deduplicated": True,
            },
        )

        self.assertTrue(
            dense_capture["has_additional_checkpoint_between_divergence_and_final"]
        )
        self.assertNotIn(
            "no additional checkpoint exists between the first divergent selected move and the final snapshot",
            dense_capture["insufficiency_reasons"],
        )

    def test_build_regenerated_shared_drift_artifact_preserves_row_context_and_dense_trace(
        self,
    ):
        baseline = self.load_baseline_inputs_for_direct_call()
        dense_trace = module.build_dense_trace(
            baseline,
            dense_trace_points=self.dense_trace_points(),
            checkpoint_capture_policy={
                "capture_mode": "dense_full",
                "checkpoint_schedule": [1, 4, 8, 12, 16],
                "root_snapshot_deduplicated": True,
            },
        )

        regenerated = module.build_regenerated_shared_drift_artifact(
            baseline,
            dense_trace=dense_trace,
            trace_cadence_capture_artifact_path="/tmp/capture_002_trace_cadence_capture.json",
            trace_cadence_capture_artifact_sha256="dense-sha",
        )

        self.assertEqual(
            "capture_available-002", regenerated["rows"][module.ROW_ID]["row_id"]
        )
        self.assertEqual(
            self.dense_trace_points()[0],
            regenerated["rows"][module.ROW_ID]["root_start"],
        )
        self.assertEqual(
            self.dense_trace_points()[1:],
            regenerated["rows"][module.ROW_ID]["snapshots"],
        )
        self.assertEqual(
            {
                "trace_capture_schema": module.SCHEMA,
                "trace_origin": "dense_rerun",
                "row_id": module.ROW_ID,
                "trace_capture_artifact_path": "/tmp/capture_002_trace_cadence_capture.json",
                "trace_capture_artifact_sha256": "dense-sha",
                "source_shared_drift_artifact_path": "/tmp/source/shared_drift.json",
                "source_shared_drift_artifact_sha256": "source-sha",
            },
            regenerated["trace_capture_provenance"],
        )

    def test_build_regenerated_shared_drift_artifact_uses_reloaded_source_payload_and_sha_when_provided(
        self,
    ):
        baseline = self.load_baseline_inputs_for_direct_call()
        dense_source_artifact = copy.deepcopy(
            baseline["trace_capture_artifact"]["source_shared_drift_artifact"]
        )
        dense_source_artifact["artifact_sha256"] = "reloaded-source-sha"
        dense_source_artifact["source_payload"]["rows"][module.ROW_ID][
            "canonical_state"
        ] = "reloaded-state"
        dense_trace = module.build_dense_trace(
            baseline,
            dense_trace_points=self.dense_trace_points(),
            checkpoint_capture_policy={
                "capture_mode": "dense_full",
                "checkpoint_schedule": [1, 4, 8, 12, 16],
                "root_snapshot_deduplicated": True,
            },
        )

        regenerated = module.build_regenerated_shared_drift_artifact(
            baseline,
            dense_trace=dense_trace,
            source_shared_drift_artifact=dense_source_artifact,
            trace_cadence_capture_artifact_path="/tmp/capture_002_trace_cadence_capture.json",
            trace_cadence_capture_artifact_sha256="dense-sha",
        )

        self.assertEqual(
            "reloaded-state", regenerated["rows"][module.ROW_ID]["canonical_state"]
        )
        self.assertEqual(
            "reloaded-source-sha",
            regenerated["trace_capture_provenance"][
                "source_shared_drift_artifact_sha256"
            ],
        )

    def load_baseline_inputs_for_direct_call(self) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace_capture.json"
            selection_path = Path(tmp) / "selection_score.json"
            cadence_path = Path(tmp) / "cadence_review.json"
            self.write_json(trace_path, self.trace_capture_artifact())
            self.write_json(selection_path, self.selection_score_artifact())
            self.write_json(cadence_path, self.trace_cadence_review_artifact())
            return module.load_baseline_inputs(
                trace_capture_artifact_path=trace_path,
                selection_score_artifact_path=selection_path,
                trace_cadence_review_artifact_path=cadence_path,
                nonseparable_review_artifact_path=None,
            )


class Capture002TraceCadenceCaptureDownstreamTest(
    Capture002TraceCadenceCaptureDenseTraceTest
):
    def stable_trace_capture_payload_sha256(self, payload: dict) -> str:
        normalized = json.loads(json.dumps(payload))
        normalized["trace_capture_artifact_sha256"] = None
        normalized["artifact_write_summary"]["trace_capture_sha256"] = None
        normalized["regenerated_shared_drift_artifact"]["trace_capture_provenance"][
            "trace_capture_artifact_sha256"
        ] = None
        return hashlib.sha256(
            (json.dumps(normalized, indent=2, sort_keys=True) + "\n").encode("utf-8")
        ).hexdigest()

    def test_build_payload_emits_dense_trace_capture_artifact_for_downstream_review(
        self,
    ):
        baseline = self.load_baseline_inputs_for_direct_call()
        dense_trace = module.build_dense_trace(
            baseline,
            dense_trace_points=self.dense_trace_points(),
            checkpoint_capture_policy={
                "capture_mode": "dense_full",
                "checkpoint_schedule": [1, 4, 8, 12, 16],
                "root_snapshot_deduplicated": True,
            },
        )
        provenance_guard = module.build_provenance_guard(
            baseline,
            dense_row_context=baseline["row_context"],
            dense_selected_artifact=baseline["source_selected_artifact"],
            dense_search_settings=baseline["search_settings"],
            checkpoint_capture_policy=dense_trace["checkpoint_capture_policy"],
        )

        payload = module.build_payload(
            baseline=baseline,
            dense_trace=dense_trace,
            provenance_guard=provenance_guard,
            trace_cadence_capture_artifact_path="/tmp/capture_002_trace_cadence_capture.json",
        )

        self.assertEqual(module.SCHEMA, payload["schema"])
        self.assertEqual("dense_rerun", payload["trace_origin"])
        self.assertEqual(
            [1.0, 4.0, 8.0, 12.0, 16.0],
            payload["trace_points_summary"]["unique_simulation_checkpoints"],
        )
        self.assertEqual([], payload["insufficiency_reasons"])
        self.assertEqual(
            dense_trace["checkpoint_capture_policy"],
            payload["checkpoint_capture_policy"],
        )
        self.assertTrue(payload["provenance_guard"]["passed"])
        self.assertEqual(
            "azlite_shared_full_search_drift_diagnostic_v1",
            payload["regenerated_shared_drift_artifact"]["schema"],
        )
        self.assertEqual(
            {
                "trace_capture_schema": module.SCHEMA,
                "trace_origin": "dense_rerun",
                "row_id": module.ROW_ID,
                "trace_capture_artifact_path": "/tmp/capture_002_trace_cadence_capture.json",
                "trace_capture_artifact_sha256": payload[
                    "trace_capture_artifact_sha256"
                ],
                "source_shared_drift_artifact_path": "/tmp/source/shared_drift.json",
                "source_shared_drift_artifact_sha256": "source-sha",
            },
            payload["regenerated_shared_drift_artifact"]["trace_capture_provenance"],
        )

    def test_build_payload_embeds_sha_for_full_final_payload_shape(self):
        baseline = self.load_baseline_inputs_for_direct_call()
        dense_trace = module.build_dense_trace(
            baseline,
            dense_trace_points=self.dense_trace_points(),
            checkpoint_capture_policy={
                "capture_mode": "dense_full",
                "checkpoint_schedule": [1, 4, 8, 12, 16],
                "root_snapshot_deduplicated": True,
            },
        )
        provenance_guard = module.build_provenance_guard(
            baseline,
            dense_row_context=baseline["row_context"],
            dense_selected_artifact=baseline["source_selected_artifact"],
            dense_search_settings=baseline["search_settings"],
            checkpoint_capture_policy=dense_trace["checkpoint_capture_policy"],
        )

        payload = module.build_payload(
            baseline=baseline,
            dense_trace=dense_trace,
            provenance_guard=provenance_guard,
            trace_cadence_capture_artifact_path="/tmp/capture_002_trace_cadence_capture.json",
        )

        expected_sha = self.stable_trace_capture_payload_sha256(payload)

        self.assertEqual(expected_sha, payload["trace_capture_artifact_sha256"])
        self.assertEqual(
            expected_sha, payload["artifact_write_summary"]["trace_capture_sha256"]
        )
        self.assertEqual(
            expected_sha,
            payload["regenerated_shared_drift_artifact"]["trace_capture_provenance"][
                "trace_capture_artifact_sha256"
            ],
        )

    def test_build_payload_records_provenance_guard_failures_and_dense_trace_insufficiency(
        self,
    ):
        baseline = self.load_baseline_inputs_for_direct_call()
        dense_trace = module.build_dense_trace(
            baseline,
            dense_trace_points=self.sparse_dense_trace_points(),
            checkpoint_capture_policy={
                "capture_mode": "dense_full",
                "checkpoint_schedule": [1, 16],
                "root_snapshot_deduplicated": True,
            },
        )
        provenance_guard = module.build_provenance_guard(
            baseline,
            dense_row_context={
                **baseline["row_context"],
                "full_search_selected_move": 1,
            },
            dense_selected_artifact=baseline["source_selected_artifact"],
            dense_search_settings=baseline["search_settings"],
            checkpoint_capture_policy=dense_trace["checkpoint_capture_policy"],
        )

        payload = module.build_payload(
            baseline=baseline,
            dense_trace=dense_trace,
            provenance_guard=provenance_guard,
            trace_cadence_capture_artifact_path="/tmp/capture_002_trace_cadence_capture.json",
        )

        self.assertIn(
            "dense trace still collapses to only root/final checkpoints after deduplication",
            payload["insufficiency_reasons"],
        )
        self.assertIn(
            "no additional checkpoint exists between the first divergent selected move and the final snapshot",
            payload["insufficiency_reasons"],
        )
        self.assertFalse(payload["provenance_guard"]["passed"])
        self.assertIn(
            "full_search_selected_move_mismatch",
            payload["provenance_guard"]["failures"],
        )

    def test_build_payload_does_not_regenerate_shared_drift_artifact_when_provenance_fails(
        self,
    ):
        baseline = self.load_baseline_inputs_for_direct_call()
        dense_trace = module.build_dense_trace(
            baseline,
            dense_trace_points=self.dense_trace_points(),
            checkpoint_capture_policy={
                "capture_mode": "dense_full",
                "checkpoint_schedule": [1, 4, 8, 12, 16],
                "root_snapshot_deduplicated": True,
            },
        )
        provenance_guard = module.build_provenance_guard(
            baseline,
            dense_row_context={
                **baseline["row_context"],
                "full_search_selected_move": 1,
            },
            dense_selected_artifact=baseline["source_selected_artifact"],
            dense_search_settings=baseline["search_settings"],
            checkpoint_capture_policy=dense_trace["checkpoint_capture_policy"],
        )

        payload = module.build_payload(
            baseline=baseline,
            dense_trace=dense_trace,
            provenance_guard=provenance_guard,
            trace_cadence_capture_artifact_path="/tmp/capture_002_trace_cadence_capture.json",
        )

        self.assertFalse(payload["cadence_validation"]["passed"])
        self.assertIsNone(payload["regenerated_shared_drift_artifact"])


class Capture002TraceCadenceCaptureCliTest(Capture002TraceCadenceCaptureDenseTraceTest):
    def dense_nonseparable_review_artifact(self) -> dict:
        return {
            "schema": module.SOURCE_NONSEPARABLE_REVIEW_SCHEMA,
            "classification": {
                "classification": "genuinely_not_separable",
                "evidence_summary": "dense nonseparable review",
            },
            "decision": "stop_002_unresolved",
        }

    def cadence_adequate_review_artifact(self) -> dict:
        return {
            "schema": module.SOURCE_TRACE_CADENCE_REVIEW_SCHEMA,
            "classification": {
                "classification": "cadence_adequate",
                "evidence_summary": "dense cadence review",
            },
            "decision": "continue_002_threshold_too_strict_check",
            "unique_simulation_checkpoints": [1.0, 4.0, 8.0, 12.0, 16.0],
            "unique_simulation_checkpoint_count": 5,
            "duplicate_root_snapshot_count": 0,
            "ambiguity_signals": [],
            "trace_origin": "dense_rerun",
            "trace_capture_excerpt": {
                "row_id": module.ROW_ID,
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

    def test_run_downstream_reruns_stops_early_when_dense_trace_is_still_insufficient(
        self,
    ):
        baseline = self.load_baseline_inputs_for_direct_call()
        dense_trace = module.build_dense_trace(
            baseline,
            dense_trace_points=self.sparse_dense_trace_points(),
            checkpoint_capture_policy={
                "capture_mode": "dense_full",
                "checkpoint_schedule": [1, 16],
                "root_snapshot_deduplicated": True,
            },
        )

        outputs = module.run_downstream_reruns(
            baseline,
            dense_trace=dense_trace,
            provenance_guard={"passed": True, "failures": []},
            out_path=Path("/tmp/out.json"),
        )

        self.assertIsNone(outputs["regenerated_shared_drift_artifact"])
        self.assertIsNone(outputs["dense_selection_score_artifact"])
        self.assertEqual(
            {"stopped_early": True, "stop_reason": "trace_cadence_unresolved"},
            outputs["downstream_rerun_summary"],
        )

    def test_run_downstream_reruns_stops_early_when_provenance_guard_fails(self):
        baseline = self.load_baseline_inputs_for_direct_call()
        dense_trace = module.build_dense_trace(
            baseline,
            dense_trace_points=self.dense_trace_points(),
            checkpoint_capture_policy={
                "capture_mode": "dense_full",
                "checkpoint_schedule": [1, 4, 8, 12, 16],
                "root_snapshot_deduplicated": True,
            },
        )

        outputs = module.run_downstream_reruns(
            baseline,
            dense_trace=dense_trace,
            provenance_guard={
                "passed": False,
                "failures": ["selected_artifact_provenance_mismatch"],
            },
            out_path=Path("/tmp/out.json"),
        )

        self.assertIsNone(outputs["regenerated_shared_drift_artifact"])
        self.assertEqual(
            {"stopped_early": True, "stop_reason": "provenance_guard_failed"},
            outputs["downstream_rerun_summary"],
        )

    def test_capture_dense_rerun_result_exposes_actual_rerun_provenance_context(self):
        baseline = self.load_baseline_inputs_for_direct_call()
        source_artifact = self.trace_capture_artifact()["source_shared_drift_artifact"]
        source_artifact["source_payload"]["rows"][module.ROW_ID]["canonical_state"] = (
            "changed-state"
        )

        with (
            patch.object(
                module.trace_capture,
                "load_source_shared_drift_artifact",
                return_value={
                    "artifact_path": source_artifact["artifact_path"],
                    "artifact_sha256": source_artifact["artifact_sha256"],
                    "row": source_artifact["source_payload"]["rows"][module.ROW_ID],
                    "selected_artifact": source_artifact["selected_artifact"],
                    "settings": source_artifact["source_payload"]["settings"],
                    "source_payload": source_artifact["source_payload"],
                },
            ),
            patch.object(
                module.trace_capture,
                "_default_rerun_result",
                return_value={"trace_points": self.dense_trace_points()},
            ),
        ):
            result = module.capture_dense_rerun_result(baseline)

        self.assertEqual("changed-state", result["row_context"]["canonical_state"])
        self.assertEqual(
            source_artifact["selected_artifact"], result["selected_artifact"]
        )
        self.assertEqual(
            source_artifact["source_payload"],
            result["source_shared_drift_artifact"]["source_payload"],
        )
        self.assertEqual(
            source_artifact["source_payload"]["settings"]["search_settings"],
            result["search_settings"],
        )

    def test_run_downstream_reruns_builds_threshold_and_nonseparable_only_after_unresolved_dense_results(
        self,
    ):
        baseline = self.load_baseline_inputs_for_direct_call()
        dense_trace = module.build_dense_trace(
            baseline,
            dense_trace_points=self.dense_trace_points(),
            checkpoint_capture_policy={
                "capture_mode": "dense_full",
                "checkpoint_schedule": [1, 4, 8, 12, 16],
                "root_snapshot_deduplicated": True,
            },
        )
        dense_selection_score_artifact = self.selection_score_artifact()
        dense_trace_cadence_review_artifact = self.cadence_adequate_review_artifact()
        dense_threshold_relaxed_selection_score_artifact = (
            self.selection_score_artifact()
        )
        dense_nonseparable_review_artifact = self.dense_nonseparable_review_artifact()

        with (
            patch.object(
                module,
                "build_regenerated_shared_drift_artifact",
                return_value={
                    "schema": "azlite_shared_full_search_drift_diagnostic_v1"
                },
            ),
            patch.object(
                module.selection_score_trace,
                "load_source_shared_drift_artifact_document",
                return_value={"artifact_path": "<dense>"},
            ),
            patch.object(
                module.selection_score_trace,
                "build_payload",
                side_effect=[
                    dense_selection_score_artifact,
                    dense_threshold_relaxed_selection_score_artifact,
                ],
            ) as build_selection_payload,
            patch.object(
                module.trace_cadence_review,
                "build_payload",
                return_value=dense_trace_cadence_review_artifact,
            ),
            patch.object(
                module.nonseparable_review,
                "build_payload",
                return_value=dense_nonseparable_review_artifact,
            ),
        ):
            outputs = module.run_downstream_reruns(
                baseline,
                dense_trace=dense_trace,
                provenance_guard={"passed": True, "failures": []},
                out_path=Path("/tmp/capture_002_trace_cadence_capture.json"),
            )

        self.assertEqual(
            "azlite_shared_full_search_drift_diagnostic_v1",
            outputs["regenerated_shared_drift_artifact"]["schema"],
        )
        self.assertEqual(
            "write_002_unresolved_trace_review_spec",
            outputs["dense_selection_score_artifact"]["decision"],
        )
        self.assertEqual(
            "continue_002_threshold_too_strict_check",
            outputs["dense_trace_cadence_review_artifact"]["decision"],
        )
        self.assertEqual(
            "write_002_unresolved_trace_review_spec",
            outputs["dense_threshold_relaxed_selection_score_artifact"]["decision"],
        )
        self.assertEqual(
            "stop_002_unresolved",
            outputs["dense_nonseparable_review_artifact"]["decision"],
        )
        self.assertEqual(2, build_selection_payload.call_count)
        self.assertFalse(outputs["downstream_rerun_summary"]["stopped_early"])
        self.assertIsNone(outputs["downstream_rerun_summary"]["stop_reason"])

    def test_main_writes_dense_artifact_and_prints_compact_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace_capture.json"
            selection_path = Path(tmp) / "selection_score.json"
            cadence_path = Path(tmp) / "trace_cadence_review.json"
            out_path = Path(tmp) / "trace_cadence_capture.json"
            self.write_json(trace_path, self.trace_capture_artifact())
            self.write_json(selection_path, self.selection_score_artifact())
            self.write_json(cadence_path, self.trace_cadence_review_artifact())

            dense_trace_points = self.dense_trace_points()
            downstream_outputs = {
                "regenerated_shared_drift_artifact": {
                    "schema": "azlite_shared_full_search_drift_diagnostic_v1"
                },
                "dense_selection_score_artifact": self.selection_score_artifact(),
                "dense_trace_cadence_review_artifact": self.cadence_adequate_review_artifact(),
                "dense_threshold_relaxed_selection_score_artifact": self.selection_score_artifact(),
                "dense_nonseparable_review_artifact": self.dense_nonseparable_review_artifact(),
                "downstream_rerun_summary": {
                    "stopped_early": False,
                    "stop_reason": None,
                },
            }

            with (
                patch.object(
                    module,
                    "capture_dense_rerun_result",
                    return_value={
                        "trace_points": dense_trace_points,
                        "row_context": self.trace_capture_artifact()["row_context"],
                        "selected_artifact": self.trace_capture_artifact()[
                            "source_shared_drift_artifact"
                        ]["selected_artifact"],
                        "search_settings": self.trace_capture_artifact()[
                            "upstream_inputs"
                        ]["search_settings"],
                        "source_shared_drift_artifact": self.trace_capture_artifact()[
                            "source_shared_drift_artifact"
                        ],
                        "insufficiency_reasons": [],
                    },
                ),
                patch.object(
                    module,
                    "run_downstream_reruns",
                    return_value=downstream_outputs,
                ),
            ):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = module.main(
                        [
                            "--source-trace-capture-artifact",
                            str(trace_path),
                            "--source-selection-score-artifact",
                            str(selection_path),
                            "--source-trace-cadence-review-artifact",
                            str(cadence_path),
                            "--out",
                            str(out_path),
                        ]
                    )

            written = json.loads(out_path.read_text(encoding="utf-8"))
            printed = json.loads(stdout.getvalue())

        self.assertEqual(0, exit_code)
        self.assertEqual(module.SCHEMA, written["schema"])
        self.assertEqual(str(out_path), printed["artifact_path"])
        self.assertEqual(module.SCHEMA, printed["schema"])
        self.assertEqual(written["decision"], printed["decision"])

    def test_main_fails_closed_when_dense_cadence_sufficiency_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace_capture.json"
            selection_path = Path(tmp) / "selection_score.json"
            cadence_path = Path(tmp) / "trace_cadence_review.json"
            out_path = Path(tmp) / "trace_cadence_capture.json"
            self.write_json(trace_path, self.trace_capture_artifact())
            self.write_json(selection_path, self.selection_score_artifact())
            self.write_json(cadence_path, self.trace_cadence_review_artifact())

            with patch.object(
                module,
                "capture_dense_rerun_result",
                return_value={
                    "trace_points": self.sparse_dense_trace_points(),
                    "row_context": self.trace_capture_artifact()["row_context"],
                    "selected_artifact": self.trace_capture_artifact()[
                        "source_shared_drift_artifact"
                    ]["selected_artifact"],
                    "search_settings": self.trace_capture_artifact()["upstream_inputs"][
                        "search_settings"
                    ],
                    "source_shared_drift_artifact": self.trace_capture_artifact()[
                        "source_shared_drift_artifact"
                    ],
                    "insufficiency_reasons": [],
                },
            ):
                exit_code = module.main(
                    [
                        "--source-trace-capture-artifact",
                        str(trace_path),
                        "--source-selection-score-artifact",
                        str(selection_path),
                        "--source-trace-cadence-review-artifact",
                        str(cadence_path),
                        "--out",
                        str(out_path),
                    ]
                )

            written = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual(0, exit_code)
        self.assertEqual("trace_cadence_unresolved", written["classification"])
        self.assertEqual("stop_002_trace_cadence_unresolved", written["decision"])

    def test_main_fails_closed_when_dense_rerun_returns_no_trace_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace_capture.json"
            selection_path = Path(tmp) / "selection_score.json"
            cadence_path = Path(tmp) / "trace_cadence_review.json"
            out_path = Path(tmp) / "trace_cadence_capture.json"
            self.write_json(trace_path, self.trace_capture_artifact())
            self.write_json(selection_path, self.selection_score_artifact())
            self.write_json(cadence_path, self.trace_cadence_review_artifact())

            with patch.object(
                module,
                "capture_dense_rerun_result",
                return_value={
                    "trace_points": [],
                    "row_context": self.trace_capture_artifact()["row_context"],
                    "selected_artifact": self.trace_capture_artifact()[
                        "source_shared_drift_artifact"
                    ]["selected_artifact"],
                    "search_settings": self.trace_capture_artifact()["upstream_inputs"][
                        "search_settings"
                    ],
                    "source_shared_drift_artifact": self.trace_capture_artifact()[
                        "source_shared_drift_artifact"
                    ],
                    "insufficiency_reasons": ["deterministic_rerun_inputs_incomplete"],
                },
            ):
                exit_code = module.main(
                    [
                        "--source-trace-capture-artifact",
                        str(trace_path),
                        "--source-selection-score-artifact",
                        str(selection_path),
                        "--source-trace-cadence-review-artifact",
                        str(cadence_path),
                        "--out",
                        str(out_path),
                    ]
                )

            written = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual(0, exit_code)
        self.assertEqual("trace_cadence_unresolved", written["classification"])
        self.assertEqual("stop_002_trace_cadence_unresolved", written["decision"])
        self.assertIn(
            "no additional checkpoint exists between the first divergent selected move and the final snapshot",
            written["insufficiency_reasons"],
        )
