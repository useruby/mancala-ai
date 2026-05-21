import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ml.alphazero_lite import capture_002_trace_checkpoint_canonicalization as module


class Capture002TraceCheckpointCanonicalizationContractTest(unittest.TestCase):
    def test_contract_constants_are_stable(self):
        self.assertEqual(
            "azlite_capture_002_trace_checkpoint_canonicalization_v1", module.SCHEMA
        )
        self.assertEqual(
            "azlite_capture_002_selection_score_trace_v1",
            module.SOURCE_SELECTION_SCORE_SCHEMA,
        )
        self.assertEqual("capture_available-002", module.ROW_ID)
        self.assertEqual("unresolved", module.EXPECTED_TRACE_CLASSIFICATION)
        self.assertEqual(
            "write_002_unresolved_trace_review_spec", module.EXPECTED_TRACE_DECISION
        )
        self.assertEqual(
            {
                "duplicate_root_snapshot_only": "write_002_metric_audit_canonical_input_spec",
                "duplicate_equivalent_checkpoint": "write_002_metric_audit_canonical_input_spec",
                "duplicate_conflicting_checkpoint": "stop_002_duplicate_checkpoint_conflict",
                "checkpoint_shape_mismatch": "stop_002_checkpoint_shape_mismatch",
                "checkpoint_canonicalization_inconclusive": "stop_002_checkpoint_canonicalization_inconclusive",
            },
            module.CLASSIFICATION_DECISIONS,
        )

    def test_parse_args_reads_required_paths(self):
        args = module.parse_args(
            [
                "--source-selection-score-artifact",
                "/tmp/default.json",
                "--source-threshold-relaxed-selection-score-artifact",
                "/tmp/relaxed.json",
                "--out",
                "/tmp/out.json",
            ]
        )

        self.assertEqual(
            Path("/tmp/default.json"), args.source_selection_score_artifact
        )
        self.assertEqual(
            Path("/tmp/relaxed.json"),
            args.source_threshold_relaxed_selection_score_artifact,
        )
        self.assertEqual(Path("/tmp/out.json"), args.out)


class Capture002TraceCheckpointCanonicalizationBuildPayloadTest(unittest.TestCase):
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

    def move_entry(self, move: int, *, q_value: float, selection_score: float) -> dict:
        return {
            "move": move,
            "q_value": q_value,
            "selection_score": selection_score,
        }

    def trace_point(
        self,
        *,
        simulation: float,
        selected_move: int,
        reference_move_by_prior: int = 2,
        visits: list[float],
        selected_q: float,
        reference_q: float,
        selected_selection_score: float,
        reference_selection_score: float,
    ) -> dict:
        return {
            "simulation": simulation,
            "selected_move": selected_move,
            "reference_move_by_prior": reference_move_by_prior,
            "visits": list(visits),
            "moves": [
                self.move_entry(
                    0, q_value=selected_q, selection_score=selected_selection_score
                ),
                self.move_entry(
                    2, q_value=reference_q, selection_score=reference_selection_score
                ),
            ],
        }

    def trace_artifact(
        self,
        *,
        thresholds: dict,
        trace_points: list[dict],
        trace_origin: str = "extracted",
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

    def safe_trace_points(self) -> list[dict]:
        root = self.trace_point(
            simulation=1.0,
            selected_move=2,
            visits=[0.0, 0.0, 1.0, 0.0, 0.0],
            selected_q=0.0,
            reference_q=0.5,
            selected_selection_score=0.33,
            reference_selection_score=0.71,
        )
        return [
            root,
            copy.deepcopy(root),
            self.trace_point(
                simulation=2.0,
                selected_move=2,
                visits=[0.0, 0.0, 2.0, 0.0, 0.0],
                selected_q=0.0,
                reference_q=0.04,
                selected_selection_score=0.47,
                reference_selection_score=0.23,
            ),
            self.trace_point(
                simulation=192.0,
                selected_move=2,
                visits=[34.0, 44.0, 61.0, 25.0, 28.0],
                selected_q=0.03,
                reference_q=0.08,
                selected_selection_score=0.16,
                reference_selection_score=0.17,
            ),
            self.trace_point(
                simulation=384.0,
                selected_move=0,
                visits=[120.0, 65.0, 114.0, 34.0, 51.0],
                selected_q=0.03,
                reference_q=0.01,
                selected_selection_score=0.08,
                reference_selection_score=0.08,
            ),
        ]

    def valid_inputs(self) -> tuple[dict, dict]:
        trace_points = self.safe_trace_points()
        return (
            self.trace_artifact(
                thresholds=self.default_thresholds(), trace_points=trace_points
            ),
            self.trace_artifact(
                thresholds=self.relaxed_thresholds(), trace_points=trace_points
            ),
        )

    def build_payload(self, default=None, relaxed=None) -> dict:
        valid_default, valid_relaxed = self.valid_inputs()
        return module.build_payload(
            default or valid_default,
            relaxed or valid_relaxed,
            source_selection_score_artifact_path="/tmp/default.json",
            source_threshold_relaxed_selection_score_artifact_path="/tmp/relaxed.json",
        )

    def test_build_payload_classifies_duplicate_root_snapshot_only(self):
        payload = self.build_payload()

        self.assertEqual(
            "duplicate_root_snapshot_only", payload["classification"]["classification"]
        )
        self.assertEqual(
            "write_002_metric_audit_canonical_input_spec", payload["decision"]
        )
        self.assertTrue(payload["canonical_sequences_match"])
        self.assertEqual(
            [1.0, 2.0, 192.0, 384.0],
            payload["canonical_checkpoint_sequences"]["default"],
        )
        self.assertEqual(
            [1.0, 2.0, 192.0, 384.0],
            payload["canonical_checkpoint_sequences"]["relaxed"],
        )

    def test_build_payload_classifies_duplicate_equivalent_checkpoint_when_later_duplicate_exists(
        self,
    ):
        default, relaxed = self.valid_inputs()
        later_duplicate = copy.deepcopy(default["trace_points"][2])
        default["trace_points"].insert(3, copy.deepcopy(later_duplicate))
        relaxed["trace_points"].insert(3, copy.deepcopy(later_duplicate))

        payload = self.build_payload(default=default, relaxed=relaxed)

        self.assertEqual(
            "duplicate_equivalent_checkpoint",
            payload["classification"]["classification"],
        )
        self.assertEqual(
            "write_002_metric_audit_canonical_input_spec", payload["decision"]
        )
        self.assertTrue(payload["canonical_sequences_match"])

    def test_build_payload_classifies_duplicate_conflicting_checkpoint_when_duplicate_group_differs(
        self,
    ):
        default, relaxed = self.valid_inputs()
        default["trace_points"][1]["moves"][0]["selection_score"] = 0.99

        payload = self.build_payload(default=default, relaxed=relaxed)

        self.assertEqual(
            "duplicate_conflicting_checkpoint",
            payload["classification"]["classification"],
        )
        self.assertEqual("stop_002_duplicate_checkpoint_conflict", payload["decision"])
        self.assertFalse(payload["canonical_sequences_match"])

    def test_build_payload_classifies_duplicate_conflicting_checkpoint_when_duplicate_move_order_differs(
        self,
    ):
        default, relaxed = self.valid_inputs()
        default["trace_points"][1]["moves"] = list(
            reversed(default["trace_points"][1]["moves"])
        )
        relaxed["trace_points"][1]["moves"] = list(
            reversed(relaxed["trace_points"][1]["moves"])
        )

        payload = self.build_payload(default=default, relaxed=relaxed)

        self.assertEqual(
            "duplicate_conflicting_checkpoint",
            payload["classification"]["classification"],
        )
        self.assertEqual("stop_002_duplicate_checkpoint_conflict", payload["decision"])
        self.assertFalse(payload["canonical_sequences_match"])

    def test_build_payload_classifies_checkpoint_shape_mismatch_when_safe_sequences_differ(
        self,
    ):
        default, relaxed = self.valid_inputs()
        relaxed["trace_points"] = [
            relaxed["trace_points"][0],
            copy.deepcopy(relaxed["trace_points"][0]),
            relaxed["trace_points"][3],
            relaxed["trace_points"][4],
        ]

        payload = self.build_payload(default=default, relaxed=relaxed)

        self.assertEqual(
            "checkpoint_shape_mismatch", payload["classification"]["classification"]
        )
        self.assertEqual("stop_002_checkpoint_shape_mismatch", payload["decision"])
        self.assertFalse(payload["canonical_sequences_match"])

    def test_build_payload_rejects_mismatched_trace_origin(self):
        default, relaxed = self.valid_inputs()
        relaxed["trace_origin"] = "rerun"

        with self.assertRaisesRegex(
            ValueError, "default and relaxed trace_origin must match"
        ):
            self.build_payload(default=default, relaxed=relaxed)

    def test_main_writes_sorted_payload_and_prints_compact_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            default_path = tmp_path / "default.json"
            relaxed_path = tmp_path / "relaxed.json"
            out_path = tmp_path / "diagnostics" / "canonicalization.json"
            default, relaxed = self.valid_inputs()

            default_path.write_text(json.dumps(default), encoding="utf-8")
            relaxed_path.write_text(json.dumps(relaxed), encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = module.main(
                    [
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
        self.assertEqual(
            payload["classification"]["classification"], summary["classification"]
        )
        self.assertEqual(payload["decision"], summary["decision"])


if __name__ == "__main__":
    unittest.main()
