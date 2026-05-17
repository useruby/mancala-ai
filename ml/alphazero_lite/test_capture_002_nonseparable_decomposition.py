import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ml.alphazero_lite import capture_002_nonseparable_decomposition as module


class Capture002NonseparableDecompositionContractTest(unittest.TestCase):
    def test_contract_constants_are_stable(self):
        self.assertEqual("azlite_capture_002_nonseparable_decomposition_v1", module.SCHEMA)
        self.assertEqual("azlite_capture_002_selection_score_trace_v1", module.SOURCE_SELECTION_SCORE_SCHEMA)
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
                "metric_co_movement": "stop_002_mechanism_not_isolated",
                "threshold_boundary_ambiguity": "write_002_confidence_band_spec",
                "signal_absent": "write_002_child_value_source_audit_spec",
                "decomposition_inconclusive": "stop_002_decomposition_inconclusive",
            },
            module.CLASSIFICATION_DECISIONS,
        )

    def test_parse_args_reads_required_paths(self):
        args = module.parse_args(
            [
                "--source-selection-score-artifact",
                "/tmp/default.json",
                "--source-threshold-review-artifact",
                "/tmp/relaxed.json",
                "--source-trace-cadence-review-artifact",
                "/tmp/cadence.json",
                "--source-nonseparable-review-artifact",
                "/tmp/nonseparable.json",
                "--out",
                "/tmp/out.json",
            ]
        )

        self.assertEqual(Path("/tmp/default.json"), args.source_selection_score_artifact)
        self.assertEqual(Path("/tmp/relaxed.json"), args.source_threshold_review_artifact)
        self.assertEqual(Path("/tmp/cadence.json"), args.source_trace_cadence_review_artifact)
        self.assertEqual(
            Path("/tmp/nonseparable.json"),
            args.source_nonseparable_review_artifact,
        )
        self.assertEqual(Path("/tmp/out.json"), args.out)


class Capture002NonseparableDecompositionBuildPayloadTest(unittest.TestCase):
    def write_json(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def selected_artifact(self) -> dict:
        return {
            "path": "/tmp/source/selected",
            "selected_artifact": "/tmp/source/selected_artifact",
            "selected_target": "/tmp/source/selected",
            "provenance_source": "selection_manifest.selected_target",
        }

    def source_artifact(self) -> dict:
        return {
            "artifact_path": "/tmp/source-artifacts/shared_drift.json",
            "classification": {"classification": "shared_mechanism_disproved"},
            "decision": "write_row_split_followup_spec",
            "full_search_selected_move": 0,
            "reference_move": 2,
            "row_id": "capture_available-002",
            "schema": "azlite_shared_full_search_drift_diagnostic_v1",
            "selected_artifact": self.selected_artifact(),
        }

    def first_selection_score_overtake_snapshot(self) -> dict:
        return {
            "simulation": 2.0,
            "selected_move": 2,
            "reference_move_by_prior": 2,
            "visits": [0.0, 0.0, 2.0, 0.0, 0.0],
            "moves": [
                {"move": 0, "selection_score": 0.47, "q_value": 0.0},
                {"move": 2, "selection_score": 0.23, "q_value": 0.04},
            ],
        }

    def default_selection_score_artifact(self, **overrides) -> dict:
        artifact = {
            "schema": module.SOURCE_SELECTION_SCORE_SCHEMA,
            "classification": {"classification": "unresolved", "evidence_summary": "default unresolved"},
            "decision": "write_002_unresolved_trace_review_spec",
            "insufficiency_reasons": [],
            "source_artifact": self.source_artifact(),
            "thresholds": {
                "meaningful_q_margin": 0.03,
                "material_selection_score_margin": 0.05,
                "material_visit_share_margin": 0.05,
            },
            "final_selected_minus_reference_q": 0.018,
            "final_selected_minus_reference_selection_score": 0.002,
            "final_selected_minus_reference_visit_share": 0.015625,
            "first_selected_material_visit_share_snapshot": None,
            "first_selected_meaningful_q_support_snapshot": None,
            "first_selected_selection_score_overtake_snapshot": self.first_selection_score_overtake_snapshot(),
        }
        artifact.update(overrides)
        return artifact

    def threshold_review_artifact(self, **overrides) -> dict:
        artifact = self.default_selection_score_artifact(
            classification={"classification": "unresolved", "evidence_summary": "relaxed unresolved"},
            thresholds={
                "meaningful_q_margin": 0.03,
                "material_selection_score_margin": 0.05,
                "material_visit_share_margin": 0.04,
            },
        )
        artifact.update(overrides)
        return artifact

    def trace_cadence_review_artifact(self, selection_score_artifact: dict | None = None, **overrides) -> dict:
        selection_score_artifact = selection_score_artifact or self.default_selection_score_artifact()
        artifact = {
            "schema": module.SOURCE_TRACE_CADENCE_REVIEW_SCHEMA,
            "classification": {"classification": "cadence_adequate", "evidence_summary": "adequate cadence"},
            "decision": "continue_002_threshold_too_strict_check",
            "trace_capture_excerpt": {
                "row_id": "capture_available-002",
                "reference_move": 2,
                "full_search_selected_move": 0,
            },
            "selection_score_excerpt": {
                "final_selected_minus_reference_visit_share": selection_score_artifact[
                    "final_selected_minus_reference_visit_share"
                ],
                "first_selected_material_visit_share_snapshot": selection_score_artifact[
                    "first_selected_material_visit_share_snapshot"
                ],
                "first_selected_meaningful_q_support_snapshot": selection_score_artifact[
                    "first_selected_meaningful_q_support_snapshot"
                ],
                "first_selected_selection_score_overtake_snapshot": selection_score_artifact[
                    "first_selected_selection_score_overtake_snapshot"
                ],
            },
            "input_artifacts": {
                "selection_score_artifact_path": "/tmp/default.json",
                "trace_capture_artifact_path": "/tmp/trace_capture.json",
            },
            "unique_simulation_checkpoint_count": 4,
            "unique_simulation_checkpoints": [1.0, 2.0, 192.0, 384.0],
        }
        artifact.update(overrides)
        return artifact

    def nonseparable_review_artifact(
        self,
        default_selection_score_artifact: dict | None = None,
        threshold_review_artifact: dict | None = None,
        trace_cadence_review_artifact: dict | None = None,
        **overrides,
    ) -> dict:
        default_selection_score_artifact = default_selection_score_artifact or self.default_selection_score_artifact()
        threshold_review_artifact = threshold_review_artifact or self.threshold_review_artifact()
        trace_cadence_review_artifact = trace_cadence_review_artifact or self.trace_cadence_review_artifact(
            default_selection_score_artifact
        )
        artifact = {
            "schema": module.SOURCE_NONSEPARABLE_REVIEW_SCHEMA,
            "hypothesis": "genuinely_not_separable",
            "classification": {
                "classification": "genuinely_not_separable",
                "evidence_summary": "nonseparable",
            },
            "decision": "stop_002_unresolved",
            "input_artifacts": {
                "source_selection_score_artifact_path": "/tmp/default.json",
                "source_threshold_review_artifact_path": "/tmp/relaxed.json",
                "source_trace_cadence_review_artifact_path": "/tmp/cadence.json",
            },
            "thresholds_evaluated": {
                "default_material_visit_share_margin": default_selection_score_artifact["thresholds"][
                    "material_visit_share_margin"
                ],
                "relaxed_material_visit_share_margin": threshold_review_artifact["thresholds"][
                    "material_visit_share_margin"
                ],
            },
            "final_margin_summary": {
                "default_q_margin": default_selection_score_artifact[
                    "final_selected_minus_reference_q"
                ],
                "default_selection_score_margin": default_selection_score_artifact[
                    "final_selected_minus_reference_selection_score"
                ],
                "default_visit_share_margin": default_selection_score_artifact[
                    "final_selected_minus_reference_visit_share"
                ],
                "relaxed_q_margin": threshold_review_artifact[
                    "final_selected_minus_reference_q"
                ],
                "relaxed_selection_score_margin": threshold_review_artifact[
                    "final_selected_minus_reference_selection_score"
                ],
                "relaxed_visit_share_margin": threshold_review_artifact[
                    "final_selected_minus_reference_visit_share"
                ],
            },
            "source_snapshots": {
                "default_classification": copy.deepcopy(default_selection_score_artifact["classification"]),
                "cadence_classification": copy.deepcopy(trace_cadence_review_artifact["classification"]),
                "threshold_classification": copy.deepcopy(threshold_review_artifact["classification"]),
            },
        }
        artifact.update(overrides)
        return artifact

    def build_payload(self, default=None, cadence=None, threshold=None, nonseparable=None) -> dict:
        default = default or self.default_selection_score_artifact()
        threshold = threshold or self.threshold_review_artifact()
        cadence = cadence or self.trace_cadence_review_artifact(default)
        nonseparable = nonseparable or self.nonseparable_review_artifact(default, threshold, cadence)
        return module.build_payload(
            default,
            cadence,
            threshold,
            nonseparable,
            source_selection_score_artifact_path="/tmp/default.json",
            source_trace_cadence_review_artifact_path="/tmp/cadence.json",
            source_threshold_review_artifact_path="/tmp/relaxed.json",
            source_nonseparable_review_artifact_path="/tmp/nonseparable.json",
        )

    def test_build_payload_classifies_metric_co_movement_when_final_margins_move_together(self):
        payload = self.build_payload()

        self.assertEqual("azlite_capture_002_nonseparable_decomposition_v1", payload["schema"])
        self.assertEqual("metric_co_movement", payload["classification"]["classification"])
        self.assertEqual("stop_002_mechanism_not_isolated", payload["decision"])
        self.assertEqual(0.018, payload["final_margin_summary"]["relaxed_q_margin"])
        self.assertEqual(4, payload["cadence_summary"]["unique_simulation_checkpoint_count"])
        self.assertIsNotNone(payload["first_support_summary"]["selection_score_overtake_snapshot"])
        self.assertEqual("capture_available-002", payload["source_artifact"]["row_id"])

    def test_build_payload_preserves_full_source_artifact_provenance(self):
        default = self.default_selection_score_artifact()

        payload = self.build_payload(default=default)

        self.assertEqual(default["source_artifact"], payload["source_artifact"])

    def test_build_payload_classifies_threshold_boundary_ambiguity_near_relaxed_visit_threshold(self):
        default = self.default_selection_score_artifact(
            final_selected_minus_reference_q=-0.002,
            final_selected_minus_reference_selection_score=-0.001,
            final_selected_minus_reference_visit_share=0.035,
            first_selected_selection_score_overtake_snapshot=None,
        )
        threshold = self.threshold_review_artifact(
            final_selected_minus_reference_q=-0.002,
            final_selected_minus_reference_selection_score=-0.001,
            final_selected_minus_reference_visit_share=0.035,
            first_selected_selection_score_overtake_snapshot=None,
        )
        cadence = self.trace_cadence_review_artifact(default)

        payload = self.build_payload(default=default, threshold=threshold, cadence=cadence)

        self.assertEqual("threshold_boundary_ambiguity", payload["classification"]["classification"])
        self.assertEqual("write_002_confidence_band_spec", payload["decision"])

    def test_build_payload_classifies_signal_absent_when_all_final_margins_are_absent(self):
        default = self.default_selection_score_artifact(
            final_selected_minus_reference_q=-0.02,
            final_selected_minus_reference_selection_score=-0.04,
            final_selected_minus_reference_visit_share=0.01,
            first_selected_selection_score_overtake_snapshot=None,
        )
        threshold = self.threshold_review_artifact(
            final_selected_minus_reference_q=-0.02,
            final_selected_minus_reference_selection_score=-0.04,
            final_selected_minus_reference_visit_share=0.01,
            first_selected_selection_score_overtake_snapshot=None,
        )
        cadence = self.trace_cadence_review_artifact(default)

        payload = self.build_payload(default=default, threshold=threshold, cadence=cadence)

        self.assertEqual("signal_absent", payload["classification"]["classification"])
        self.assertEqual("write_002_child_value_source_audit_spec", payload["decision"])

    def test_build_payload_classifies_inconclusive_for_valid_unmatched_evidence(self):
        default = self.default_selection_score_artifact(
            final_selected_minus_reference_q=0.01,
            final_selected_minus_reference_selection_score=-0.01,
            final_selected_minus_reference_visit_share=-0.02,
            first_selected_selection_score_overtake_snapshot=None,
        )
        threshold = self.threshold_review_artifact(
            final_selected_minus_reference_q=0.01,
            final_selected_minus_reference_selection_score=-0.01,
            final_selected_minus_reference_visit_share=-0.02,
            first_selected_selection_score_overtake_snapshot=None,
        )
        cadence = self.trace_cadence_review_artifact(default)

        payload = self.build_payload(default=default, threshold=threshold, cadence=cadence)

        self.assertEqual("decomposition_inconclusive", payload["classification"]["classification"])
        self.assertEqual("stop_002_decomposition_inconclusive", payload["decision"])

    def test_build_payload_rejects_wrong_schemas(self):
        default = self.default_selection_score_artifact(schema="wrong")

        with self.assertRaisesRegex(ValueError, "selection score artifact has wrong schema"):
            self.build_payload(default=default)

    def test_build_payload_rejects_mismatched_source_identity(self):
        threshold = self.threshold_review_artifact()
        threshold["source_artifact"]["reference_move"] = 3

        with self.assertRaisesRegex(ValueError, "selection score source identities must match"):
            self.build_payload(threshold=threshold)

    def test_build_payload_rejects_mismatched_selected_artifact(self):
        threshold = self.threshold_review_artifact()
        threshold["source_artifact"]["selected_artifact"]["selected_target"] = "/tmp/other"

        with self.assertRaisesRegex(ValueError, "selection score source identities must match"):
            self.build_payload(threshold=threshold)

    def test_build_payload_rejects_mismatched_cadence_trace_identity(self):
        cadence = self.trace_cadence_review_artifact()
        cadence["trace_capture_excerpt"]["full_search_selected_move"] = 1

        with self.assertRaisesRegex(ValueError, "trace cadence review artifact trace_capture_excerpt"):
            self.build_payload(cadence=cadence)

    def test_build_payload_rejects_mismatched_cadence_selection_excerpt(self):
        cadence = self.trace_cadence_review_artifact()
        cadence["selection_score_excerpt"]["final_selected_minus_reference_visit_share"] = 0.99

        with self.assertRaisesRegex(ValueError, "selection_score_excerpt"):
            self.build_payload(cadence=cadence)

    def test_build_payload_rejects_mismatched_cadence_selection_score_input_path(self):
        cadence = self.trace_cadence_review_artifact()
        cadence["input_artifacts"]["selection_score_artifact_path"] = "/tmp/other-default.json"

        with self.assertRaisesRegex(ValueError, "trace cadence review artifact input_artifacts"):
            self.build_payload(cadence=cadence)

    def test_build_payload_rejects_non_adequate_cadence(self):
        cadence = self.trace_cadence_review_artifact(
            classification={"classification": "trace_too_sparse"},
            decision="write_002_trace_cadence_capture_spec",
        )

        with self.assertRaisesRegex(ValueError, "trace cadence review artifact must represent adequate cadence"):
            self.build_payload(cadence=cadence)

    def test_build_payload_rejects_non_genuine_nonseparable_review(self):
        nonseparable = self.nonseparable_review_artifact(
            classification={"classification": "prerequisite_preempted"},
            decision="write_002_trace_cadence_capture_spec",
        )

        with self.assertRaisesRegex(ValueError, "nonseparable review artifact must stop unresolved"):
            self.build_payload(nonseparable=nonseparable)

    def test_build_payload_rejects_mismatched_nonseparable_summary(self):
        nonseparable = self.nonseparable_review_artifact()
        nonseparable["final_margin_summary"]["relaxed_q_margin"] = 0.99

        with self.assertRaisesRegex(ValueError, "nonseparable review artifact final_margin_summary"):
            self.build_payload(nonseparable=nonseparable)

    def test_main_writes_sorted_payload_and_prints_compact_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            default_path = tmp_path / "default.json"
            threshold_path = tmp_path / "threshold.json"
            cadence_path = tmp_path / "cadence.json"
            nonseparable_path = tmp_path / "nonseparable.json"
            out_path = tmp_path / "diagnostics" / "decomposition.json"

            default = self.default_selection_score_artifact()
            threshold = self.threshold_review_artifact()
            cadence = self.trace_cadence_review_artifact(default)
            cadence["input_artifacts"]["selection_score_artifact_path"] = str(default_path)
            nonseparable = self.nonseparable_review_artifact(default, threshold, cadence)
            nonseparable["input_artifacts"] = {
                "source_selection_score_artifact_path": str(default_path),
                "source_threshold_review_artifact_path": str(threshold_path),
                "source_trace_cadence_review_artifact_path": str(cadence_path),
            }
            self.write_json(default_path, default)
            self.write_json(threshold_path, threshold)
            self.write_json(cadence_path, cadence)
            self.write_json(nonseparable_path, nonseparable)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = module.main(
                    [
                        "--source-selection-score-artifact",
                        str(default_path),
                        "--source-threshold-review-artifact",
                        str(threshold_path),
                        "--source-trace-cadence-review-artifact",
                        str(cadence_path),
                        "--source-nonseparable-review-artifact",
                        str(nonseparable_path),
                        "--out",
                        str(out_path),
                    ]
                )

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            summary = json.loads(stdout.getvalue())

        self.assertEqual(0, exit_code)
        self.assertEqual("metric_co_movement", payload["classification"]["classification"])
        self.assertEqual(str(out_path), summary["artifact_path"])
        self.assertEqual(module.SCHEMA, summary["schema"])
        self.assertEqual("metric_co_movement", summary["classification"])
        self.assertEqual("stop_002_mechanism_not_isolated", summary["decision"])


if __name__ == "__main__":
    unittest.main()
