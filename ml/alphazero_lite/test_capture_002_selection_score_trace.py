import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ml.alphazero_lite import capture_002_selection_score_trace as module


class Capture002SelectionScoreTraceContractTest(unittest.TestCase):
    def test_contract_constants_are_stable(self):
        self.assertEqual(
            "azlite_capture_002_selection_score_trace_v1",
            module.SCHEMA,
        )
        self.assertEqual(
            "azlite_shared_full_search_drift_diagnostic_v1",
            module.SOURCE_SHARED_DRIFT_SCHEMA,
        )
        self.assertEqual("capture_available-002", module.ROW_ID)
        self.assertEqual(
            {
                "meaningful_q_margin": 0.03,
                "material_selection_score_margin": 0.05,
                "material_visit_share_margin": 0.05,
            },
            module.THRESHOLDS,
        )
        self.assertEqual(
            {
                "selection_score_pressure_confirmed": "write_002_selection_pressure_ablation_spec",
                "q_support_precedes_selection_score": "write_002_child_value_audit_spec",
                "trace_insufficient": "write_002_trace_capture_spec",
                "unresolved": "write_002_unresolved_trace_review_spec",
            },
            module.CLASSIFICATION_DECISIONS,
        )
        self.assertEqual(
            {"extracted", "rerun", "insufficient"},
            module.TRACE_ORIGINS,
        )

    def test_load_json_reads_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.json"
            payload = {"schema": module.SOURCE_SHARED_DRIFT_SCHEMA}
            source_path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertEqual(payload, module.load_json(source_path))

    def test_parse_args_reads_required_paths_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.json"
            out_path = Path(tmp) / "nested" / "diagnostic.json"

            args = module.parse_args(
                [
                    "--source-shared-drift-artifact",
                    str(source_path),
                    "--out",
                    str(out_path),
                ]
            )

        self.assertEqual(source_path, args.source_shared_drift_artifact)
        self.assertEqual(out_path, args.out)
        self.assertFalse(args.allow_rerun_capture)
        self.assertIsNone(args.meaningful_q_margin)
        self.assertIsNone(args.material_selection_score_margin)
        self.assertIsNone(args.material_visit_share_margin)

    def test_parse_args_accepts_allow_rerun_capture_flag(self):
        args = module.parse_args(
            [
                "--source-shared-drift-artifact",
                "/tmp/source.json",
                "--out",
                "/tmp/out.json",
                "--allow-rerun-capture",
            ]
        )

        self.assertTrue(args.allow_rerun_capture)

    def test_parse_args_accepts_threshold_overrides(self):
        args = module.parse_args(
            [
                "--source-shared-drift-artifact",
                "/tmp/source.json",
                "--out",
                "/tmp/out.json",
                "--meaningful-q-margin",
                "0.12",
                "--material-selection-score-margin",
                "0.18",
                "--material-visit-share-margin",
                "0.22",
            ]
        )

        self.assertEqual(0.12, args.meaningful_q_margin)
        self.assertEqual(0.18, args.material_selection_score_margin)
        self.assertEqual(0.22, args.material_visit_share_margin)

    def test_parse_args_requires_source_shared_drift_artifact(self):
        with self.assertRaises(SystemExit):
            module.parse_args(["--out", "/tmp/diagnostic.json"])

    def test_parse_args_requires_out(self):
        with self.assertRaises(SystemExit):
            module.parse_args(["--source-shared-drift-artifact", "/tmp/source.json"])


class Capture002SelectionScoreTraceSourceArtifactTest(unittest.TestCase):
    def write_json(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def valid_trace_point(
        self,
        *,
        selected_move: int,
        simulation: float,
        visits: list[float],
        reference_move_by_prior: int | None = 2,
    ) -> dict:
        return {
            "selected_move": selected_move,
            "simulation": simulation,
            "visits": list(visits),
            "reference_move_by_prior": reference_move_by_prior,
        }

    def valid_source_artifact(self) -> dict:
        legal_moves = [0, 1, 2, 3, 4]
        root_start = self.valid_trace_point(
            selected_move=2,
            simulation=1.0,
            visits=[0.0, 0.0, 1.0, 0.0, 0.0],
        )
        snapshots = [
            copy.deepcopy(root_start),
            self.valid_trace_point(
                selected_move=0,
                simulation=16.0,
                visits=[8.0, 4.0, 7.0, 2.0, 1.0],
            ),
        ]
        return {
            "schema": module.SOURCE_SHARED_DRIFT_SCHEMA,
            "selected_artifact": {
                "path": "/tmp/artifacts/selected.bin",
                "selected_target": "/tmp/artifacts/selected.bin",
                "selected_artifact": None,
                "provenance_source": "selection_manifest.selected_target",
            },
            "settings": {
                "search_settings": {
                    "c_puct": 1.25,
                    "fpu_mode": "zero",
                    "normalize_values": True,
                    "reuse_subtree": True,
                    "root_policy_mode": "deterministic",
                    "tactical_root_bias": 0.1,
                },
                "seed": 23,
                "simulation_count": 32,
            },
            "classification": {
                "classification": "shared_mechanism_disproved",
                "evidence_summary": "Rows diverge before a single shared mechanism can be supported.",
            },
            "decision": "write_row_split_followup_spec",
            "rows": {
                "capture_available-002": {
                    "row_id": "capture_available-002",
                    "canonical_state": '{"player_pits":[1,0,7,6,6],"opponent_pits":[5,4,4,4,0]}',
                    "legal_moves": list(legal_moves),
                    "reference_move": 2,
                    "full_search_selected_move": 0,
                    "root_start": copy.deepcopy(root_start),
                    "snapshots": copy.deepcopy(snapshots),
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
                }
            },
        }

    def test_load_source_shared_drift_artifact_accepts_valid_capture_available_002_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            self.write_json(source_path, self.valid_source_artifact())

            artifact = module.load_source_shared_drift_artifact(source_path)

        self.assertEqual(module.SOURCE_SHARED_DRIFT_SCHEMA, artifact["schema"])
        self.assertEqual("write_row_split_followup_spec", artifact["decision"])
        self.assertEqual("capture_available-002", artifact["row"]["row_id"])
        self.assertEqual(
            {
                "simulation": 1.0,
                "selected_move": 2,
                "reference_move_by_prior": 2,
                "visits": [0.0, 0.0, 1.0, 0.0, 0.0],
            },
            artifact["row"]["root_start"],
        )
        self.assertEqual(
            [1.0, 16.0],
            [snapshot["simulation"] for snapshot in artifact["row"]["snapshots"]],
        )
        self.assertEqual(
            {
                "path": "/tmp/artifacts/selected.bin",
                "selected_target": "/tmp/artifacts/selected.bin",
                "selected_artifact": None,
                "provenance_source": "selection_manifest.selected_target",
            },
            artifact["selected_artifact"],
        )
        self.assertEqual(
            {
                "search_settings": {
                    "c_puct": 1.25,
                    "fpu_mode": "zero",
                    "normalize_values": True,
                    "reuse_subtree": True,
                    "root_policy_mode": "deterministic",
                    "tactical_root_bias": 0.1,
                },
                "seed": 23,
                "simulation_count": 32,
            },
            artifact["settings"],
        )

    def test_load_source_shared_drift_artifact_preserves_empty_snapshots_for_fail_closed_handling(self):
        artifact = self.valid_source_artifact()
        artifact["rows"]["capture_available-002"]["snapshots"] = []
        artifact["rows"]["capture_available-002"]["root_start"] = None

        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            self.write_json(source_path, artifact)

            loaded_artifact = module.load_source_shared_drift_artifact(source_path)

        self.assertIsNone(loaded_artifact["row"]["root_start"])
        self.assertEqual([], loaded_artifact["row"]["snapshots"])

    def test_load_source_shared_drift_artifact_rejects_missing_root_start_when_snapshots_exist(self):
        artifact = self.valid_source_artifact()
        artifact["rows"]["capture_available-002"]["root_start"] = None

        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            self.write_json(source_path, artifact)

            with self.assertRaisesRegex(ValueError, "root_start"):
                module.load_source_shared_drift_artifact(source_path)

    def test_load_source_shared_drift_artifact_rejects_non_list_snapshots(self):
        artifact = self.valid_source_artifact()
        artifact["rows"]["capture_available-002"]["snapshots"] = {"simulation": 1.0}

        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            self.write_json(source_path, artifact)

            with self.assertRaisesRegex(ValueError, "snapshots"):
                module.load_source_shared_drift_artifact(source_path)

    def test_load_source_shared_drift_artifact_rejects_non_dict_root_start(self):
        artifact = self.valid_source_artifact()
        artifact["rows"]["capture_available-002"]["root_start"] = []

        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            self.write_json(source_path, artifact)

            with self.assertRaisesRegex(ValueError, "root_start"):
                module.load_source_shared_drift_artifact(source_path)

    def test_load_source_shared_drift_artifact_rejects_snapshots_out_of_order_by_simulation(self):
        artifact = self.valid_source_artifact()
        artifact["rows"]["capture_available-002"]["snapshots"] = [
            artifact["rows"]["capture_available-002"]["snapshots"][1],
            artifact["rows"]["capture_available-002"]["snapshots"][0],
        ]

        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            self.write_json(source_path, artifact)

            with self.assertRaisesRegex(ValueError, "ordered by simulation"):
                module.load_source_shared_drift_artifact(source_path)

    def test_load_source_shared_drift_artifact_rejects_root_start_later_than_first_snapshot(self):
        artifact = self.valid_source_artifact()
        artifact["rows"]["capture_available-002"]["root_start"]["simulation"] = 17.0

        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            self.write_json(source_path, artifact)

            with self.assertRaisesRegex(ValueError, "root_start.*ordered by simulation"):
                module.load_source_shared_drift_artifact(source_path)

    def test_load_source_shared_drift_artifact_rejects_snapshot_without_upstream_fields(self):
        artifact = self.valid_source_artifact()
        del artifact["rows"]["capture_available-002"]["snapshots"][0]["visits"]

        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            self.write_json(source_path, artifact)

            with self.assertRaisesRegex(ValueError, "visits"):
                module.load_source_shared_drift_artifact(source_path)

    def test_load_source_shared_drift_artifact_rejects_null_reference_move(self):
        artifact = self.valid_source_artifact()
        artifact["rows"]["capture_available-002"]["reference_move"] = None

        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            self.write_json(source_path, artifact)

            with self.assertRaisesRegex(ValueError, "reference_move must be a legal move int"):
                module.load_source_shared_drift_artifact(source_path)

    def test_load_source_shared_drift_artifact_rejects_wrong_type_full_search_selected_move(self):
        artifact = self.valid_source_artifact()
        artifact["rows"]["capture_available-002"]["full_search_selected_move"] = "0"

        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            self.write_json(source_path, artifact)

            with self.assertRaisesRegex(ValueError, "full_search_selected_move must be a legal move int"):
                module.load_source_shared_drift_artifact(source_path)

    def test_load_source_shared_drift_artifact_rejects_illegal_reference_move(self):
        artifact = self.valid_source_artifact()
        artifact["rows"]["capture_available-002"]["reference_move"] = 9

        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            self.write_json(source_path, artifact)

            with self.assertRaisesRegex(ValueError, "reference_move must be a legal move int"):
                module.load_source_shared_drift_artifact(source_path)


class Capture002SelectionScoreTraceBuildPayloadTest(unittest.TestCase):
    def write_json(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def trace_point(
        self,
        *,
        simulation: float,
        selected_move: int,
        reference_move_by_prior: int,
        moves: list[dict],
        visits: list[float],
    ) -> dict:
        return {
            "simulation": simulation,
            "selected_move": selected_move,
            "reference_move_by_prior": reference_move_by_prior,
            "moves": copy.deepcopy(moves),
            "visits": list(visits),
        }

    def source_artifact(self, *, root_start=None, snapshots=None, selected_artifact=None) -> dict:
        return {
            "artifact_path": "/tmp/source-artifacts/shared_drift.json",
            "schema": module.SOURCE_SHARED_DRIFT_SCHEMA,
            "decision": module.EXPECTED_SOURCE_DECISION,
            "classification": {"classification": "shared_mechanism_disproved"},
            "selected_artifact": selected_artifact,
            "row": {
                "row_id": module.ROW_ID,
                "reference_move": 2,
                "full_search_selected_move": 0,
                "legal_moves": [0, 1, 2, 3, 4],
                "root_start": copy.deepcopy(root_start),
                "snapshots": copy.deepcopy(snapshots or []),
            },
        }

    def source_artifact_document(self, *, root_start=None, snapshots=None, selected_artifact=None) -> dict:
        return {
            "schema": module.SOURCE_SHARED_DRIFT_SCHEMA,
            "classification": {"classification": "shared_mechanism_disproved"},
            "decision": module.EXPECTED_SOURCE_DECISION,
            "selected_artifact": copy.deepcopy(selected_artifact),
            "rows": {
                module.ROW_ID: {
                    "row_id": module.ROW_ID,
                    "canonical_state": "state-002",
                    "legal_moves": [0, 1, 2, 3, 4],
                    "reference_move": 2,
                    "full_search_selected_move": 0,
                    "root_start": copy.deepcopy(root_start),
                    "snapshots": copy.deepcopy(snapshots or []),
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
                }
            },
        }

    def test_build_payload_prefers_selection_score_overtake_before_meaningful_q_support(self):
        root_start = self.trace_point(
            simulation=1.0,
            selected_move=2,
            reference_move_by_prior=2,
            visits=[0.0, 0.0, 1.0, 0.0, 0.0],
            moves=[
                {"move": 0, "selection_score": 0.40, "q_value": 0.01},
                {"move": 2, "selection_score": 0.44, "q_value": 0.00},
            ],
        )
        snapshots = [
            self.trace_point(
                simulation=8.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[4.0, 0.0, 5.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.48, "q_value": 0.01},
                    {"move": 2, "selection_score": 0.46, "q_value": 0.00},
                ],
            ),
            self.trace_point(
                simulation=16.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[11.0, 0.0, 5.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.56, "q_value": -0.01},
                    {"move": 2, "selection_score": 0.45, "q_value": -0.03},
                ],
            ),
            self.trace_point(
                simulation=32.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[23.0, 0.0, 9.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.59, "q_value": 0.04},
                    {"move": 2, "selection_score": 0.43, "q_value": 0.00},
                ],
            ),
        ]

        payload = module.build_payload(self.source_artifact(root_start=root_start, snapshots=snapshots))

        self.assertEqual("extracted", payload["trace_origin"])
        self.assertEqual("selection_score_pressure_confirmed", payload["classification"]["classification"])
        self.assertEqual(
            module.CLASSIFICATION_DECISIONS["selection_score_pressure_confirmed"],
            payload["decision"],
        )
        self.assertEqual(
            {
                "artifact_path": "/tmp/source-artifacts/shared_drift.json",
                "schema": module.SOURCE_SHARED_DRIFT_SCHEMA,
                "decision": module.EXPECTED_SOURCE_DECISION,
                "classification": {"classification": "shared_mechanism_disproved"},
                "row_id": module.ROW_ID,
                "reference_move": 2,
                "full_search_selected_move": 0,
            },
            payload["source_artifact"],
        )
        self.assertEqual(16.0, payload["first_selected_selection_score_overtake_snapshot"]["simulation"])
        self.assertEqual(32.0, payload["first_selected_meaningful_q_support_snapshot"]["simulation"])
        self.assertAlmostEqual(0.16, payload["final_selected_minus_reference_selection_score"])
        self.assertEqual(0.04, payload["final_selected_minus_reference_q"])
        self.assertAlmostEqual(0.4375, payload["final_selected_minus_reference_visit_share"])

    def test_build_payload_compares_against_row_target_before_snapshot_selected_move_flips(self):
        root_start = self.trace_point(
            simulation=1.0,
            selected_move=2,
            reference_move_by_prior=2,
            visits=[0.0, 0.0, 1.0, 0.0, 0.0],
            moves=[
                {"move": 0, "selection_score": 0.40, "q_value": 0.00},
                {"move": 2, "selection_score": 0.44, "q_value": 0.00},
            ],
        )
        snapshots = [
            self.trace_point(
                simulation=8.0,
                selected_move=2,
                reference_move_by_prior=2,
                visits=[4.0, 0.0, 5.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.52, "q_value": 0.01},
                    {"move": 2, "selection_score": 0.46, "q_value": 0.00},
                ],
            ),
            self.trace_point(
                simulation=10.0,
                selected_move=2,
                reference_move_by_prior=2,
                visits=[5.0, 0.0, 6.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.53, "q_value": 0.04},
                    {"move": 2, "selection_score": 0.46, "q_value": 0.00},
                ],
            ),
            self.trace_point(
                simulation=12.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[8.0, 0.0, 5.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.55, "q_value": 0.04},
                    {"move": 2, "selection_score": 0.45, "q_value": 0.00},
                ],
            ),
        ]

        payload = module.build_payload(self.source_artifact(root_start=root_start, snapshots=snapshots))

        self.assertEqual(8.0, payload["first_selected_selection_score_overtake_snapshot"]["simulation"])
        self.assertEqual(10.0, payload["first_selected_meaningful_q_support_snapshot"]["simulation"])
        self.assertEqual("selection_score_pressure_confirmed", payload["classification"]["classification"])

    def test_build_payload_does_not_confirm_selection_score_when_visit_share_leads_first(self):
        root_start = self.trace_point(
            simulation=1.0,
            selected_move=2,
            reference_move_by_prior=2,
            visits=[0.0, 0.0, 1.0, 0.0, 0.0],
            moves=[
                {"move": 0, "selection_score": 0.40, "q_value": 0.01},
                {"move": 2, "selection_score": 0.44, "q_value": 0.00},
            ],
        )
        snapshots = [
            self.trace_point(
                simulation=8.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[7.0, 0.0, 5.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.48, "q_value": 0.01},
                    {"move": 2, "selection_score": 0.46, "q_value": 0.00},
                ],
            ),
            self.trace_point(
                simulation=16.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[11.0, 0.0, 5.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.56, "q_value": -0.01},
                    {"move": 2, "selection_score": 0.45, "q_value": -0.03},
                ],
            ),
            self.trace_point(
                simulation=32.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[23.0, 0.0, 9.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.59, "q_value": 0.04},
                    {"move": 2, "selection_score": 0.43, "q_value": 0.00},
                ],
            ),
        ]

        payload = module.build_payload(self.source_artifact(root_start=root_start, snapshots=snapshots))

        self.assertEqual(8.0, payload["first_selected_material_visit_share_snapshot"]["simulation"])
        self.assertEqual(16.0, payload["first_selected_selection_score_overtake_snapshot"]["simulation"])
        self.assertEqual("unresolved", payload["classification"]["classification"])
        self.assertEqual(module.CLASSIFICATION_DECISIONS["unresolved"], payload["decision"])

    def test_build_payload_routes_unresolved_trace_to_review_spec(self):
        root_start = self.trace_point(
            simulation=1.0,
            selected_move=2,
            reference_move_by_prior=2,
            visits=[0.0, 0.0, 1.0, 0.0, 0.0],
            moves=[
                {"move": 0, "selection_score": 0.40, "q_value": 0.01},
                {"move": 2, "selection_score": 0.44, "q_value": 0.00},
            ],
        )
        snapshots = [
            self.trace_point(
                simulation=12.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[7.0, 0.0, 5.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.48, "q_value": 0.01},
                    {"move": 2, "selection_score": 0.46, "q_value": 0.00},
                ],
            ),
            self.trace_point(
                simulation=16.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[11.0, 0.0, 5.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.56, "q_value": -0.01},
                    {"move": 2, "selection_score": 0.45, "q_value": -0.03},
                ],
            ),
            self.trace_point(
                simulation=32.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[23.0, 0.0, 9.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.59, "q_value": 0.04},
                    {"move": 2, "selection_score": 0.43, "q_value": 0.00},
                ],
            ),
        ]

        payload = module.build_payload(self.source_artifact(root_start=root_start, snapshots=snapshots))

        self.assertEqual("unresolved", payload["classification"]["classification"])
        self.assertEqual("write_002_unresolved_trace_review_spec", payload["decision"])

    def test_build_payload_uses_loaded_extracted_trace_metrics(self):
        root_start = self.trace_point(
            simulation=1.0,
            selected_move=2,
            reference_move_by_prior=2,
            visits=[0.0, 0.0, 1.0, 0.0, 0.0],
            moves=[
                {"move": 0, "selection_score": 0.40, "q_value": 0.01},
                {"move": 2, "selection_score": 0.44, "q_value": 0.00},
            ],
        )
        snapshots = [
            self.trace_point(
                simulation=16.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[11.0, 0.0, 5.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.56, "q_value": -0.01},
                    {"move": 2, "selection_score": 0.45, "q_value": -0.03},
                ],
            ),
            self.trace_point(
                simulation=32.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[23.0, 0.0, 9.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.59, "q_value": 0.04},
                    {"move": 2, "selection_score": 0.43, "q_value": 0.00},
                ],
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            self.write_json(
                source_path,
                self.source_artifact_document(root_start=root_start, snapshots=snapshots),
            )

            loaded_artifact = module.load_source_shared_drift_artifact(source_path)
            payload = module.build_payload(loaded_artifact)

        self.assertEqual("extracted", payload["trace_origin"])
        self.assertEqual("selection_score_pressure_confirmed", payload["classification"]["classification"])
        self.assertEqual(16.0, payload["first_selected_selection_score_overtake_snapshot"]["simulation"])
        self.assertEqual(32.0, payload["first_selected_meaningful_q_support_snapshot"]["simulation"])

    def test_build_payload_fails_closed_for_insufficient_extracted_trace(self):
        root_start = self.trace_point(
            simulation=1.0,
            selected_move=2,
            reference_move_by_prior=2,
            visits=[0.0, 0.0, 1.0, 0.0, 0.0],
            moves=[
                {"move": 2, "selection_score": 0.44, "q_value": 0.00},
            ],
        )
        snapshots = [
            self.trace_point(
                simulation=8.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[4.0, 0.0, 5.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.48, "q_value": 0.02},
                ],
            )
        ]
        selected_artifact = {
            "path": "/tmp/selected/model.bin",
            "selected_target": "/tmp/selected/model.bin",
            "selected_artifact": None,
            "provenance_source": "selection_manifest.selected_target",
        }

        payload = module.build_payload(
            self.source_artifact(
                root_start=root_start,
                snapshots=snapshots,
                selected_artifact=selected_artifact,
            )
        )

        self.assertEqual("insufficient", payload["trace_origin"])
        self.assertEqual("trace_insufficient", payload["classification"]["classification"])
        self.assertEqual(module.CLASSIFICATION_DECISIONS["trace_insufficient"], payload["decision"])
        self.assertEqual(module.ROW_ID, payload["source_artifact"]["row_id"])
        self.assertEqual(2, payload["source_artifact"]["reference_move"])
        self.assertEqual(0, payload["source_artifact"]["full_search_selected_move"])
        self.assertEqual(selected_artifact, payload["source_artifact"]["selected_artifact"])
        self.assertEqual(1.0, payload["trace_points"][0]["simulation"])
        self.assertEqual(8.0, payload["trace_points"][1]["simulation"])
        self.assertTrue(payload["insufficiency_reasons"])

    def test_build_payload_preserves_loaded_selected_artifact_for_insufficient_trace(self):
        selected_artifact = {
            "path": "/tmp/selected/model.bin",
            "selected_target": "/tmp/selected/model.bin",
            "selected_artifact": None,
            "provenance_source": "selection_manifest.selected_target",
        }
        root_start = self.trace_point(
            simulation=1.0,
            selected_move=2,
            reference_move_by_prior=2,
            visits=[0.0, 0.0, 1.0, 0.0, 0.0],
            moves=[
                {"move": 2, "selection_score": 0.44, "q_value": 0.00},
            ],
        )
        snapshots = [
            self.trace_point(
                simulation=8.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[4.0, 0.0, 5.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.48, "q_value": 0.02},
                ],
            )
        ]

        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            self.write_json(
                source_path,
                self.source_artifact_document(
                    root_start=root_start,
                    snapshots=snapshots,
                    selected_artifact=selected_artifact,
                ),
            )

            loaded_artifact = module.load_source_shared_drift_artifact(source_path)
            payload = module.build_payload(loaded_artifact)

        self.assertEqual("insufficient", payload["trace_origin"])
        self.assertEqual(selected_artifact, payload["source_artifact"]["selected_artifact"])
        self.assertEqual("trace_insufficient", payload["classification"]["classification"])

    def test_build_payload_uses_rerun_trace_when_allowed_and_extracted_trace_is_insufficient(self):
        extracted = self.source_artifact(
            root_start=self.trace_point(
                simulation=1.0,
                selected_move=2,
                reference_move_by_prior=2,
                visits=[0.0, 0.0, 1.0, 0.0, 0.0],
                moves=[
                    {"move": 2, "selection_score": 0.44, "q_value": 0.00},
                ],
            ),
            snapshots=[],
        )
        rerun_trace_points = [
            self.trace_point(
                simulation=1.0,
                selected_move=2,
                reference_move_by_prior=2,
                visits=[0.0, 0.0, 1.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.40, "q_value": 0.00},
                    {"move": 2, "selection_score": 0.44, "q_value": 0.00},
                ],
            ),
            self.trace_point(
                simulation=12.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[7.0, 0.0, 5.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.54, "q_value": 0.01},
                    {"move": 2, "selection_score": 0.46, "q_value": 0.00},
                ],
            ),
            self.trace_point(
                simulation=24.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[16.0, 0.0, 8.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.58, "q_value": 0.05},
                    {"move": 2, "selection_score": 0.45, "q_value": 0.00},
                ],
            ),
        ]
        rerun_settings = {"seed": 23, "simulation_count": 24}

        rerun_calls = []

        def rerun_capture(source_artifact):
            rerun_calls.append(source_artifact)
            return {
                "trace_points": copy.deepcopy(rerun_trace_points),
                "settings": dict(rerun_settings),
            }

        payload = module.build_payload(
            extracted,
            allow_rerun_capture=True,
            rerun_capture=rerun_capture,
        )

        self.assertEqual([extracted], rerun_calls)
        self.assertEqual("rerun", payload["trace_origin"])
        self.assertEqual(rerun_settings, payload["settings"])
        self.assertEqual(rerun_trace_points, payload["trace_points"])
        self.assertEqual("selection_score_pressure_confirmed", payload["classification"]["classification"])

    def test_build_payload_fails_closed_for_out_of_order_rerun_trace(self):
        extracted = self.source_artifact(
            root_start=self.trace_point(
                simulation=1.0,
                selected_move=2,
                reference_move_by_prior=2,
                visits=[0.0, 0.0, 1.0, 0.0, 0.0],
                moves=[
                    {"move": 2, "selection_score": 0.44, "q_value": 0.00},
                ],
            ),
            snapshots=[],
        )
        rerun_trace_points = [
            self.trace_point(
                simulation=12.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[7.0, 0.0, 5.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.54, "q_value": 0.01},
                    {"move": 2, "selection_score": 0.46, "q_value": 0.00},
                ],
            ),
            self.trace_point(
                simulation=1.0,
                selected_move=2,
                reference_move_by_prior=2,
                visits=[0.0, 0.0, 1.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.40, "q_value": 0.00},
                    {"move": 2, "selection_score": 0.44, "q_value": 0.00},
                ],
            ),
        ]

        payload = module.build_payload(
            extracted,
            allow_rerun_capture=True,
            rerun_capture=lambda _source_artifact: {
                "trace_points": copy.deepcopy(rerun_trace_points),
                "settings": {"seed": 41, "simulation_count": 12},
            },
        )

        self.assertEqual("insufficient", payload["trace_origin"])
        self.assertEqual("trace_insufficient", payload["classification"]["classification"])
        self.assertIn("rerun_trace_points_out_of_order", payload["insufficiency_reasons"])

    def test_build_payload_fails_closed_when_extracted_trace_drifts_to_different_move_pair(self):
        root_start = self.trace_point(
            simulation=1.0,
            selected_move=2,
            reference_move_by_prior=2,
            visits=[0.0, 0.0, 1.0, 0.0, 0.0],
            moves=[
                {"move": 0, "selection_score": 0.40, "q_value": 0.01},
                {"move": 2, "selection_score": 0.44, "q_value": 0.00},
            ],
        )
        snapshots = [
            self.trace_point(
                simulation=12.0,
                selected_move=1,
                reference_move_by_prior=2,
                visits=[4.0, 7.0, 5.0, 0.0, 0.0],
                moves=[
                    {"move": 1, "selection_score": 0.54, "q_value": 0.01},
                    {"move": 2, "selection_score": 0.46, "q_value": 0.00},
                ],
            ),
            self.trace_point(
                simulation=24.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[16.0, 0.0, 8.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.58, "q_value": 0.05},
                    {"move": 2, "selection_score": 0.45, "q_value": 0.00},
                ],
            ),
        ]

        payload = module.build_payload(self.source_artifact(root_start=root_start, snapshots=snapshots))

        self.assertEqual("insufficient", payload["trace_origin"])
        self.assertEqual("trace_insufficient", payload["classification"]["classification"])
        self.assertIn("trace_points_pair_mismatch", payload["insufficiency_reasons"])

    def test_build_payload_fails_closed_when_rerun_trace_drifts_to_different_move_pair(self):
        extracted = self.source_artifact(
            root_start=self.trace_point(
                simulation=1.0,
                selected_move=2,
                reference_move_by_prior=2,
                visits=[0.0, 0.0, 1.0, 0.0, 0.0],
                moves=[
                    {"move": 2, "selection_score": 0.44, "q_value": 0.00},
                ],
            ),
            snapshots=[],
        )
        rerun_trace_points = [
            self.trace_point(
                simulation=1.0,
                selected_move=2,
                reference_move_by_prior=2,
                visits=[0.0, 0.0, 1.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.40, "q_value": 0.00},
                    {"move": 2, "selection_score": 0.44, "q_value": 0.00},
                ],
            ),
            self.trace_point(
                simulation=12.0,
                selected_move=0,
                reference_move_by_prior=1,
                visits=[7.0, 0.0, 5.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.54, "q_value": 0.01},
                    {"move": 1, "selection_score": 0.46, "q_value": 0.00},
                ],
            ),
        ]

        payload = module.build_payload(
            extracted,
            allow_rerun_capture=True,
            rerun_capture=lambda _source_artifact: {
                "trace_points": copy.deepcopy(rerun_trace_points),
                "settings": {"seed": 41, "simulation_count": 12},
            },
        )

        self.assertEqual("insufficient", payload["trace_origin"])
        self.assertEqual("trace_insufficient", payload["classification"]["classification"])
        self.assertIn("rerun_trace_points_pair_mismatch", payload["insufficiency_reasons"])

    def test_build_payload_fails_closed_when_rerun_trace_is_still_insufficient(self):
        extracted = self.source_artifact(
            root_start=self.trace_point(
                simulation=1.0,
                selected_move=2,
                reference_move_by_prior=2,
                visits=[0.0, 0.0, 1.0, 0.0, 0.0],
                moves=[
                    {"move": 2, "selection_score": 0.44, "q_value": 0.00},
                ],
            ),
            snapshots=[],
        )
        rerun_trace_points = [
            self.trace_point(
                simulation=1.0,
                selected_move=2,
                reference_move_by_prior=2,
                visits=[0.0, 0.0, 1.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.40, "q_value": 0.00},
                    {"move": 2, "selection_score": 0.44, "q_value": 0.00},
                ],
            )
        ]
        rerun_settings = {"seed": 99, "simulation_count": 1}

        payload = module.build_payload(
            extracted,
            allow_rerun_capture=True,
            rerun_capture=lambda _source_artifact: {
                "trace_points": copy.deepcopy(rerun_trace_points),
                "settings": dict(rerun_settings),
            },
        )

        self.assertEqual("insufficient", payload["trace_origin"])
        self.assertEqual(rerun_settings, payload["settings"])
        self.assertEqual(rerun_trace_points, payload["trace_points"])
        self.assertEqual("trace_insufficient", payload["classification"]["classification"])
        self.assertIn("too_few_trace_points", payload["insufficiency_reasons"])

    def test_build_payload_fails_closed_when_rerun_callback_returns_malformed_trace_points(self):
        extracted = self.source_artifact(
            root_start=self.trace_point(
                simulation=1.0,
                selected_move=2,
                reference_move_by_prior=2,
                visits=[0.0, 0.0, 1.0, 0.0, 0.0],
                moves=[
                    {"move": 2, "selection_score": 0.44, "q_value": 0.00},
                ],
            ),
            snapshots=[],
        )

        payload = module.build_payload(
            extracted,
            allow_rerun_capture=True,
            rerun_capture=lambda _source_artifact: {
                "trace_points": {"bad": 1},
                "settings": {"seed": 17, "simulation_count": 32},
            },
        )

        self.assertEqual("insufficient", payload["trace_origin"])
        self.assertEqual("trace_insufficient", payload["classification"]["classification"])
        self.assertEqual([], payload["trace_points"])
        self.assertEqual({"seed": 17, "simulation_count": 32}, payload["settings"])
        self.assertIn("rerun_trace_points_malformed", payload["insufficiency_reasons"])

    def test_build_payload_rejects_nan_threshold_override(self):
        with self.assertRaisesRegex(ValueError, "meaningful_q_margin"):
            module.build_payload(
                self.source_artifact(),
                meaningful_q_margin=float("nan"),
            )

    def test_build_payload_rejects_negative_threshold_override(self):
        with self.assertRaisesRegex(ValueError, "material_selection_score_margin"):
            module.build_payload(
                self.source_artifact(),
                material_selection_score_margin=-0.01,
            )

    def test_build_payload_fails_closed_when_reference_move_is_missing(self):
        artifact = self.source_artifact()
        artifact["row"]["reference_move"] = None

        payload = module.build_payload(artifact)

        self.assertEqual("insufficient", payload["trace_origin"])
        self.assertIn("missing_reference_move", payload["insufficiency_reasons"])

    def test_build_payload_fails_closed_when_full_search_selected_move_is_missing(self):
        artifact = self.source_artifact()
        artifact["row"]["full_search_selected_move"] = None

        payload = module.build_payload(artifact)

        self.assertEqual("insufficient", payload["trace_origin"])
        self.assertIn("missing_full_search_selected_move", payload["insufficiency_reasons"])


class Capture002SelectionScoreTraceFixtureClassificationTest(unittest.TestCase):
    FIXTURE_DIR = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "diagnostics"
        / "capture_002_selection_score_trace"
    )
    FIXTURE_CASES = {
        "selection_score_pressure_confirmed": "selection_score_pressure_confirmed",
        "q_support_precedes_selection_score": "q_support_precedes_selection_score",
        "trace_insufficient": "trace_insufficient",
        "unresolved": "unresolved",
        "same_snapshot_material_overtakes": "unresolved",
        "final_divergence_without_overtakes": "unresolved",
        "extracted_insufficient_rerun_disallowed": "trace_insufficient",
        "extracted_insufficient_rerun_allowed": "selection_score_pressure_confirmed",
    }

    def load_fixture(self, name: str) -> dict:
        return json.loads((self.FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))

    def test_checked_in_task_4_fixtures_exist(self):
        for name in self.FIXTURE_CASES:
            self.assertTrue((self.FIXTURE_DIR / f"{name}.json").exists(), name)

    def test_fixture_cases_classify_as_expected(self):
        for name, expected_classification in self.FIXTURE_CASES.items():
            with self.subTest(name=name):
                fixture = self.load_fixture(name)
                classification = module.classify_fixture_payload(
                    fixture["payload"],
                    fixture.get("thresholds", module.THRESHOLDS),
                )

                self.assertEqual(expected_classification, classification["classification"]["classification"])
                self.assertEqual(
                    fixture["expected_decision"],
                    classification["decision"],
                )

    def test_fixture_classifier_uses_trace_origin_insufficient(self):
        fixture = self.load_fixture("trace_insufficient")

        classification = module.classify_fixture_payload(fixture["payload"], module.THRESHOLDS)

        self.assertEqual("insufficient", fixture["payload"]["trace_origin"])
        self.assertEqual("trace_insufficient", classification["classification"]["classification"])

    def test_fixture_classifier_keeps_same_snapshot_meaningful_q_and_selection_score_unresolved(self):
        fixture = self.load_fixture("same_snapshot_material_overtakes")

        classification = module.classify_fixture_payload(fixture["payload"], module.THRESHOLDS)

        self.assertEqual("unresolved", classification["classification"]["classification"])

    def test_unresolved_fixture_routes_to_review_spec(self):
        fixture = self.load_fixture("unresolved")

        classification = module.classify_fixture_payload(fixture["payload"], module.THRESHOLDS)

        self.assertEqual("unresolved", classification["classification"]["classification"])
        self.assertEqual("write_002_unresolved_trace_review_spec", classification["decision"])

    def test_fixture_classifier_tracks_runtime_classification_logic(self):
        payload = {
            "trace_origin": "extracted",
            "first_selected_selection_score_overtake_snapshot": {"simulation": 8.0},
            "first_selected_meaningful_q_support_snapshot": {"simulation": 16.0},
            "first_selected_material_visit_share_snapshot": {"simulation": 12.0},
        }

        fixture_classification = module.classify_fixture_payload(payload, module.THRESHOLDS)
        runtime_classification = module._classify_payload(payload)

        self.assertEqual(runtime_classification, fixture_classification)

    def test_fixture_classifier_selection_score_pressure_summary_allows_same_snapshot_visit_share(self):
        classification = module.classify_fixture_payload(
            {
                "trace_origin": "extracted",
                "first_selected_selection_score_overtake_snapshot": {"simulation": 8.0},
                "first_selected_meaningful_q_support_snapshot": {"simulation": 16.0},
                "first_selected_material_visit_share_snapshot": {"simulation": 8.0},
            },
            module.THRESHOLDS,
        )

        self.assertEqual("selection_score_pressure_confirmed", classification["classification"]["classification"])
        self.assertEqual(
            "Selection-score pressure appears before meaningful child-Q support and visit share is already present or follows.",
            classification["classification"]["evidence_summary"],
        )

    def test_build_payload_treats_q_margin_equal_to_threshold_as_meaningful(self):
        root_start = {
            "simulation": 1.0,
            "selected_move": 2,
            "reference_move_by_prior": 2,
            "moves": [
                {"move": 0, "selection_score": 0.375, "q_value": 0.0},
                {"move": 2, "selection_score": 0.4375, "q_value": 0.0},
            ],
            "visits": [0.0, 0.0, 1.0, 0.0, 0.0],
        }
        snapshots = [
            {
                "simulation": 8.0,
                "selected_move": 0,
                "reference_move_by_prior": 2,
                "moves": [
                    {"move": 0, "selection_score": 0.40625, "q_value": 0.03125},
                    {"move": 2, "selection_score": 0.390625, "q_value": 0.0},
                ],
                "visits": [4.0, 0.0, 5.0, 0.0, 0.0],
            },
            {
                "simulation": 12.0,
                "selected_move": 0,
                "reference_move_by_prior": 2,
                "moves": [
                    {"move": 0, "selection_score": 0.5, "q_value": 0.03125},
                    {"move": 2, "selection_score": 0.4375, "q_value": 0.0},
                ],
                "visits": [7.0, 0.0, 5.0, 0.0, 0.0],
            },
            {
                "simulation": 16.0,
                "selected_move": 0,
                "reference_move_by_prior": 2,
                "moves": [
                    {"move": 0, "selection_score": 0.5625, "q_value": 0.03125},
                    {"move": 2, "selection_score": 0.46875, "q_value": 0.0},
                ],
                "visits": [11.0, 0.0, 5.0, 0.0, 0.0],
            },
        ]

        payload = module.build_payload(
            {
                "artifact_path": "/tmp/source-artifacts/shared_drift.json",
                "schema": module.SOURCE_SHARED_DRIFT_SCHEMA,
                "decision": module.EXPECTED_SOURCE_DECISION,
                "classification": {"classification": "shared_mechanism_disproved"},
                "selected_artifact": None,
                "row": {
                    "row_id": module.ROW_ID,
                    "reference_move": 2,
                    "full_search_selected_move": 0,
                    "legal_moves": [0, 1, 2, 3, 4],
                    "root_start": root_start,
                    "snapshots": snapshots,
                },
            },
            meaningful_q_margin=0.03125,
            material_selection_score_margin=0.0625,
            material_visit_share_margin=0.05,
        )

        self.assertEqual(8.0, payload["first_selected_meaningful_q_support_snapshot"]["simulation"])
        self.assertEqual(12.0, payload["first_selected_selection_score_overtake_snapshot"]["simulation"])
        self.assertEqual("q_support_precedes_selection_score", payload["classification"]["classification"])

    def test_build_payload_treats_selection_score_margin_equal_to_threshold_as_material(self):
        root_start = {
            "simulation": 1.0,
            "selected_move": 2,
            "reference_move_by_prior": 2,
            "moves": [
                {"move": 0, "selection_score": 0.375, "q_value": 0.0},
                {"move": 2, "selection_score": 0.4375, "q_value": 0.0},
            ],
            "visits": [0.0, 0.0, 1.0, 0.0, 0.0],
        }
        snapshots = [
            {
                "simulation": 8.0,
                "selected_move": 0,
                "reference_move_by_prior": 2,
                "moves": [
                    {"move": 0, "selection_score": 0.5, "q_value": 0.015625},
                    {"move": 2, "selection_score": 0.4375, "q_value": 0.0},
                ],
                "visits": [4.0, 0.0, 5.0, 0.0, 0.0],
            },
            {
                "simulation": 12.0,
                "selected_move": 0,
                "reference_move_by_prior": 2,
                "moves": [
                    {"move": 0, "selection_score": 0.53125, "q_value": 0.015625},
                    {"move": 2, "selection_score": 0.453125, "q_value": 0.0},
                ],
                "visits": [7.0, 0.0, 5.0, 0.0, 0.0],
            },
            {
                "simulation": 16.0,
                "selected_move": 0,
                "reference_move_by_prior": 2,
                "moves": [
                    {"move": 0, "selection_score": 0.5625, "q_value": 0.03125},
                    {"move": 2, "selection_score": 0.46875, "q_value": 0.0},
                ],
                "visits": [11.0, 0.0, 5.0, 0.0, 0.0],
            },
        ]

        payload = module.build_payload(
            {
                "artifact_path": "/tmp/source-artifacts/shared_drift.json",
                "schema": module.SOURCE_SHARED_DRIFT_SCHEMA,
                "decision": module.EXPECTED_SOURCE_DECISION,
                "classification": {"classification": "shared_mechanism_disproved"},
                "selected_artifact": None,
                "row": {
                    "row_id": module.ROW_ID,
                    "reference_move": 2,
                    "full_search_selected_move": 0,
                    "legal_moves": [0, 1, 2, 3, 4],
                    "root_start": root_start,
                    "snapshots": snapshots,
                },
            },
            meaningful_q_margin=0.03125,
            material_selection_score_margin=0.0625,
            material_visit_share_margin=0.05,
        )

        self.assertEqual(8.0, payload["first_selected_selection_score_overtake_snapshot"]["simulation"])
        self.assertEqual(16.0, payload["first_selected_meaningful_q_support_snapshot"]["simulation"])
        self.assertEqual("selection_score_pressure_confirmed", payload["classification"]["classification"])


class Capture002SelectionScoreTraceCliTest(unittest.TestCase):
    def write_json(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def trace_point(
        self,
        *,
        simulation: float,
        selected_move: int,
        reference_move_by_prior: int,
        moves: list[dict],
        visits: list[float],
    ) -> dict:
        return {
            "simulation": simulation,
            "selected_move": selected_move,
            "reference_move_by_prior": reference_move_by_prior,
            "moves": copy.deepcopy(moves),
            "visits": list(visits),
        }

    def source_artifact_document(self, *, root_start=None, snapshots=None, selected_artifact=None) -> dict:
        return {
            "schema": module.SOURCE_SHARED_DRIFT_SCHEMA,
            "classification": {"classification": "shared_mechanism_disproved"},
            "decision": module.EXPECTED_SOURCE_DECISION,
            "selected_artifact": copy.deepcopy(selected_artifact),
            "rows": {
                module.ROW_ID: {
                    "row_id": module.ROW_ID,
                    "canonical_state": "state-002",
                    "legal_moves": [0, 1, 2, 3, 4],
                    "reference_move": 2,
                    "full_search_selected_move": 0,
                    "root_start": copy.deepcopy(root_start),
                    "snapshots": copy.deepcopy(snapshots or []),
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
                }
            },
        }

    def test_thresholds_from_args_uses_defaults_and_overrides(self):
        defaults = module.parse_args(
            [
                "--source-shared-drift-artifact",
                "/tmp/source.json",
                "--out",
                "/tmp/out.json",
            ]
        )
        overrides = module.parse_args(
            [
                "--source-shared-drift-artifact",
                "/tmp/source.json",
                "--out",
                "/tmp/out.json",
                "--meaningful-q-margin",
                "0.12",
                "--material-selection-score-margin",
                "0.18",
                "--material-visit-share-margin",
                "0.22",
            ]
        )

        self.assertEqual(module.THRESHOLDS, module._thresholds_from_args(defaults))
        self.assertEqual(
            {
                "meaningful_q_margin": 0.12,
                "material_selection_score_margin": 0.18,
                "material_visit_share_margin": 0.22,
            },
            module._thresholds_from_args(overrides),
        )

    def test_thresholds_from_args_rejects_non_finite_and_negative_values(self):
        with self.assertRaisesRegex(ValueError, "meaningful_q_margin"):
            module._thresholds_from_args(
                module.parse_args(
                    [
                        "--source-shared-drift-artifact",
                        "/tmp/source.json",
                        "--out",
                        "/tmp/out.json",
                        "--meaningful-q-margin",
                        "nan",
                    ]
                )
            )

        with self.assertRaisesRegex(ValueError, "material_selection_score_margin"):
            module._thresholds_from_args(
                module.parse_args(
                    [
                        "--source-shared-drift-artifact",
                        "/tmp/source.json",
                        "--out",
                        "/tmp/out.json",
                        "--material-selection-score-margin",
                        "-0.01",
                    ]
                )
            )

        with self.assertRaisesRegex(ValueError, "material_visit_share_margin"):
            module._thresholds_from_args(
                module.parse_args(
                    [
                        "--source-shared-drift-artifact",
                        "/tmp/source.json",
                        "--out",
                        "/tmp/out.json",
                        "--material-visit-share-margin",
                        "inf",
                    ]
                )
            )

    def test_main_writes_sorted_artifact_and_prints_compact_json(self):
        root_start = self.trace_point(
            simulation=1.0,
            selected_move=2,
            reference_move_by_prior=2,
            visits=[0.0, 0.0, 1.0, 0.0, 0.0],
            moves=[
                {"move": 0, "selection_score": 0.40, "q_value": 0.01},
                {"move": 2, "selection_score": 0.44, "q_value": 0.00},
            ],
        )
        snapshots = [
            self.trace_point(
                simulation=8.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[4.0, 0.0, 5.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.48, "q_value": 0.01},
                    {"move": 2, "selection_score": 0.46, "q_value": 0.00},
                ],
            ),
            self.trace_point(
                simulation=16.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[11.0, 0.0, 5.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.56, "q_value": -0.01},
                    {"move": 2, "selection_score": 0.45, "q_value": -0.03},
                ],
            ),
            self.trace_point(
                simulation=32.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[23.0, 0.0, 9.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.59, "q_value": 0.04},
                    {"move": 2, "selection_score": 0.43, "q_value": 0.00},
                ],
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            out_path = Path(tmp) / "diagnostics" / "selection_score_trace.json"
            self.write_json(
                source_path,
                self.source_artifact_document(root_start=root_start, snapshots=snapshots),
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = module.main(
                    [
                        "--source-shared-drift-artifact",
                        str(source_path),
                        "--out",
                        str(out_path),
                        "--meaningful-q-margin",
                        "0.03",
                        "--material-selection-score-margin",
                        "0.05",
                        "--material-visit-share-margin",
                        "0.05",
                    ]
                )

            written = out_path.read_text(encoding="utf-8")
            payload = json.loads(written)
            printed = json.loads(stdout.getvalue())

        self.assertEqual(0, exit_code)
        self.assertTrue(written.endswith("\n"))
        self.assertEqual(module.SCHEMA, payload["schema"])
        self.assertEqual("selection_score_pressure_confirmed", payload["classification"]["classification"])
        self.assertEqual(
            {
                "artifact_path": str(out_path),
                "schema": module.SCHEMA,
                "decision": module.CLASSIFICATION_DECISIONS["selection_score_pressure_confirmed"],
            },
            printed,
        )

    def test_main_allow_rerun_capture_changes_insufficient_result_when_rerun_trace_available(self):
        root_start = self.trace_point(
            simulation=1.0,
            selected_move=2,
            reference_move_by_prior=2,
            visits=[0.0, 0.0, 1.0, 0.0, 0.0],
            moves=[
                {"move": 2, "selection_score": 0.44, "q_value": 0.00},
            ],
        )
        snapshots = []
        rerun_trace_points = [
            self.trace_point(
                simulation=1.0,
                selected_move=2,
                reference_move_by_prior=2,
                visits=[0.0, 0.0, 1.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.40, "q_value": 0.00},
                    {"move": 2, "selection_score": 0.44, "q_value": 0.00},
                ],
            ),
            self.trace_point(
                simulation=12.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[7.0, 0.0, 5.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.54, "q_value": 0.01},
                    {"move": 2, "selection_score": 0.46, "q_value": 0.00},
                ],
            ),
            self.trace_point(
                simulation=24.0,
                selected_move=0,
                reference_move_by_prior=2,
                visits=[16.0, 0.0, 8.0, 0.0, 0.0],
                moves=[
                    {"move": 0, "selection_score": 0.58, "q_value": 0.05},
                    {"move": 2, "selection_score": 0.45, "q_value": 0.00},
                ],
            ),
        ]

        document = self.source_artifact_document(root_start=root_start, snapshots=snapshots)
        document["selected_artifact"] = {
            "path": "/tmp/artifacts/selected.bin",
            "selected_target": "/tmp/artifacts/selected.bin",
            "selected_artifact": None,
            "provenance_source": "selection_manifest.selected_target",
        }
        document["settings"] = {
            "search_settings": {
                "c_puct": 1.25,
                "fpu_mode": "zero",
                "normalize_values": True,
                "reuse_subtree": True,
                "root_policy_mode": "deterministic",
                "tactical_root_bias": 0.1,
            },
            "seed": 23,
            "simulation_count": 24,
            "rerun_trace_points": copy.deepcopy(rerun_trace_points),
        }

        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            without_rerun_out = Path(tmp) / "diagnostics" / "without_rerun.json"
            with_rerun_out = Path(tmp) / "diagnostics" / "with_rerun.json"
            self.write_json(source_path, document)

            module.main(
                [
                    "--source-shared-drift-artifact",
                    str(source_path),
                    "--out",
                    str(without_rerun_out),
                ]
            )
            module.main(
                [
                    "--source-shared-drift-artifact",
                    str(source_path),
                    "--out",
                    str(with_rerun_out),
                    "--allow-rerun-capture",
                ]
            )

            without_rerun = json.loads(without_rerun_out.read_text(encoding="utf-8"))
            with_rerun = json.loads(with_rerun_out.read_text(encoding="utf-8"))

        self.assertEqual("insufficient", without_rerun["trace_origin"])
        self.assertEqual("trace_insufficient", without_rerun["classification"]["classification"])
        self.assertEqual("rerun", with_rerun["trace_origin"])
        self.assertEqual("selection_score_pressure_confirmed", with_rerun["classification"]["classification"])
        self.assertEqual(rerun_trace_points, with_rerun["trace_points"])


if __name__ == "__main__":
    unittest.main()
