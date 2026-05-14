import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import capture_002_selection_score_trace as selection_score_trace_module
from ml.alphazero_lite import capture_002_trace_capture as module


class Capture002TraceCaptureContractTest(unittest.TestCase):
    def test_contract_constants_are_stable(self):
        self.assertEqual("azlite_capture_002_trace_capture_v1", module.SCHEMA)
        self.assertEqual("azlite_shared_full_search_drift_diagnostic_v1", module.SOURCE_SHARED_DRIFT_SCHEMA)
        self.assertEqual("capture_available-002", module.ROW_ID)
        self.assertEqual({"extract_only", "extract_then_rerun"}, module.CAPTURE_MODES)
        self.assertEqual({"extracted", "rerun", "insufficient"}, module.TRACE_ORIGINS)

    def test_parse_args_reads_required_paths_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "shared.json"
            out_path = Path(tmp) / "trace_capture.json"

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
        self.assertEqual("extract_then_rerun", args.capture_mode)
    def test_parse_args_requires_source_artifact_and_out(self):
        with self.assertRaises(SystemExit):
            module.parse_args(["--out", "/tmp/out.json"])
        with self.assertRaises(SystemExit):
            module.parse_args(["--source-shared-drift-artifact", "/tmp/shared.json"])


class Capture002TraceCaptureSourceArtifactTest(unittest.TestCase):
    def write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def valid_move_entry(self, *, move: int, visit_count: float, selection_score: float, used_fpu: bool) -> dict:
        return {
            "move": move,
            "prior": 0.2 + (0.01 * move),
            "q_value": 0.05 + (0.01 * move),
            "selection_score": selection_score,
            "used_fpu": used_fpu,
            "visit_count": visit_count,
        }

    def valid_trace_point(self, *, selected_move: int, simulation: float, visits: list[float], legal_moves: list[int]) -> dict:
        return {
            "selected_move": selected_move,
            "simulation": simulation,
            "visits": list(visits),
            "moves": [
                self.valid_move_entry(
                    move=move,
                    visit_count=float(visits[move]),
                    selection_score=0.1 + (0.02 * move),
                    used_fpu=(move != selected_move),
                )
                for move in legal_moves
            ],
        }

    def valid_probe_trace(
        self,
        *,
        selected_move: int,
        legal_moves: list[int],
        root_start_selected_move: int,
        root_start_visits: list[float],
        final_visits: list[float],
    ) -> dict:
        root_start = self.valid_trace_point(
            selected_move=root_start_selected_move,
            simulation=1.0,
            visits=root_start_visits,
            legal_moves=legal_moves,
        )
        snapshots = [
            self.valid_trace_point(
                selected_move=root_start_selected_move,
                simulation=1.0,
                visits=root_start_visits,
                legal_moves=legal_moves,
            ),
            self.valid_trace_point(
                selected_move=selected_move,
                simulation=16.0,
                visits=final_visits,
                legal_moves=legal_moves,
            ),
        ]
        return {
            "selected_move": selected_move,
            "root_start": root_start,
            "snapshots": snapshots,
            "final_deltas": {
                "selected_visits": float(final_visits[selected_move] - root_start_visits[selected_move]),
                "reference_visits": float(max(final_visits) - min(root_start_visits)),
            },
        }

    def valid_source_artifact(self) -> dict:
        legal_moves = [0, 1, 2, 3, 4]
        row_002_full_search = self.valid_probe_trace(
            selected_move=0,
            legal_moves=legal_moves,
            root_start_selected_move=2,
            root_start_visits=[0.0, 0.0, 1.0, 0.0, 0.0],
            final_visits=[8.0, 4.0, 7.0, 2.0, 1.0],
        )
        row_003_policy_only = self.valid_probe_trace(
            selected_move=2,
            legal_moves=legal_moves,
            root_start_selected_move=2,
            root_start_visits=[0.0, 0.0, 1.0, 0.0, 0.0],
            final_visits=[3.0, 2.0, 9.0, 1.0, 1.0],
        )
        row_003_value_only = self.valid_probe_trace(
            selected_move=1,
            legal_moves=legal_moves,
            root_start_selected_move=0,
            root_start_visits=[1.0, 0.0, 0.0, 0.0, 0.0],
            final_visits=[4.0, 7.0, 3.0, 1.0, 1.0],
        )
        row_003_full_search = self.valid_probe_trace(
            selected_move=1,
            legal_moves=legal_moves,
            root_start_selected_move=2,
            root_start_visits=[0.0, 0.0, 1.0, 0.0, 0.0],
            final_visits=[2.0, 8.0, 7.0, 1.0, 1.0],
        )
        return copy.deepcopy(
            {
                "schema": module.SOURCE_SHARED_DRIFT_SCHEMA,
                "classification": {
                    "classification": "shared_mechanism_disproved",
                    "evidence_summary": "Rows diverge before a single shared mechanism can be supported.",
                },
                "decision": "write_row_split_followup_spec",
                "selected_artifact": {
                    "path": "/tmp/artifacts/selected",
                    "provenance_source": "selection_manifest.selected_target",
                    "selected_artifact": "/tmp/artifacts/selection/artifact",
                    "selected_target": "/tmp/artifacts/selected",
                },
                "settings": {
                    "search_settings": {
                        "c_puct": 1.25,
                        "fpu_mode": "zero",
                        "normalize_values": False,
                        "reuse_subtree": False,
                        "root_policy_mode": "deterministic",
                        "tactical_root_bias": 0.1,
                    },
                    "seed": 17,
                    "simulation_count": 384,
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
                        "legal_moves": list(legal_moves),
                        "reference_move": 2,
                        "full_search_selected_move": 0,
                        "root_start": copy.deepcopy(row_002_full_search["root_start"]),
                        "snapshots": copy.deepcopy(row_002_full_search["snapshots"]),
                        "final_deltas": copy.deepcopy(row_002_full_search["final_deltas"]),
                        "missing_fields": [],
                        "probe_mode_traces": {
                            "policy_only": self.valid_probe_trace(
                                selected_move=2,
                                legal_moves=legal_moves,
                                root_start_selected_move=2,
                                root_start_visits=[0.0, 0.0, 1.0, 0.0, 0.0],
                                final_visits=[2.0, 1.0, 8.0, 1.0, 1.0],
                            ),
                            "value_only": self.valid_probe_trace(
                                selected_move=2,
                                legal_moves=legal_moves,
                                root_start_selected_move=2,
                                root_start_visits=[0.0, 0.0, 1.0, 0.0, 0.0],
                                final_visits=[1.0, 1.0, 9.0, 1.0, 1.0],
                            ),
                            "full_search": row_002_full_search,
                        },
                    },
                    "capture_available-003": {
                        "row_id": "capture_available-003",
                        "canonical_state": '{"player_pits":[1,6,0,6,6,5],"opponent_pits":[5,5,4,4,4,0]}',
                        "legal_moves": list(legal_moves),
                        "reference_move": 2,
                        "full_search_selected_move": 1,
                        "root_start": copy.deepcopy(row_003_full_search["root_start"]),
                        "snapshots": copy.deepcopy(row_003_full_search["snapshots"]),
                        "final_deltas": copy.deepcopy(row_003_full_search["final_deltas"]),
                        "missing_fields": [],
                        "probe_mode_traces": {
                            "policy_only": row_003_policy_only,
                            "value_only": row_003_value_only,
                            "full_search": row_003_full_search,
                        },
                    },
                },
            }
        )

    def test_load_source_artifact_preserves_row_context_and_path_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "fixtures" / "shared_drift.json"
            payload = self.valid_source_artifact()
            self.write_json(source_path, payload)
            expected_sha = module.sha256_file(source_path)

            artifact = module.load_source_shared_drift_artifact(source_path, allow_fixture_provenance=False)

        self.assertEqual(str(source_path), artifact["artifact_path"])
        self.assertEqual(expected_sha, artifact["artifact_sha256"])
        self.assertEqual("artifact", artifact["source_provenance"]["type"])
        self.assertEqual(str(source_path), artifact["source_provenance"]["artifact_path"])
        self.assertEqual(expected_sha, artifact["source_provenance"]["artifact_sha256"])
        self.assertEqual("capture_available-002", artifact["row"]["row_id"])
        self.assertEqual(payload, artifact["source_payload"])

    def test_materialized_fixture_provenance_requires_explicit_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "materialized_shared_drift.json"
            payload = self.valid_source_artifact()
            payload["selected_artifact"]["provenance_source"] = "materialized_fixture"
            self.write_json(source_path, payload)

            with self.assertRaisesRegex(ValueError, "allow_fixture_provenance"):
                module.load_source_shared_drift_artifact(source_path, allow_fixture_provenance=False)

            artifact = module.load_source_shared_drift_artifact(source_path, allow_fixture_provenance=True)

        self.assertEqual("fixture", artifact["source_provenance"]["type"])

    def test_non_object_top_level_json_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must be a JSON object"):
                module._source_provenance_type(source_path)

            with self.assertRaisesRegex(ValueError, "must be a JSON object"):
                module.load_source_shared_drift_artifact(source_path, allow_fixture_provenance=False)


class Capture002TraceCaptureExtractionTest(Capture002TraceCaptureSourceArtifactTest):
    def load_source_artifact(self, payload: dict) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            self.write_json(source_path, payload)
            return module.load_source_shared_drift_artifact(source_path, allow_fixture_provenance=False)

    def test_build_artifact_prefers_extracted_trace_when_pair_aligned_and_sufficient(self):
        payload = self.valid_source_artifact()

        artifact = self.load_source_artifact(payload)

        built = module.build_trace_capture_artifact(artifact, capture_mode="extract_only")

        self.assertEqual(module.SCHEMA, built["schema"])
        self.assertEqual("extract_only", built["capture_mode"])
        self.assertEqual("extracted", built["trace_origin"])
        self.assertEqual([], built["insufficiency_reasons"])
        self.assertEqual(3, len(built["trace_points"]))
        self.assertEqual(1.0, built["trace_points"][0]["simulation"])
        self.assertEqual(16.0, built["trace_points"][-1]["simulation"])
        self.assertEqual(payload, built["source_shared_drift_artifact"]["source_payload"])

    def test_build_artifact_keeps_extracted_trace_when_extract_then_rerun_already_has_sufficient_extraction(self):
        payload = self.valid_source_artifact()

        artifact = self.load_source_artifact(payload)

        built = module.build_trace_capture_artifact(artifact, capture_mode="extract_then_rerun")

        self.assertEqual("extract_then_rerun", built["capture_mode"])
        self.assertEqual("extracted", built["trace_origin"])
        self.assertEqual(3, len(built["trace_points"]))

    def test_build_artifact_emits_required_artifact_one_contract_sections(self):
        payload = self.valid_source_artifact()

        artifact = self.load_source_artifact(payload)

        built = module.build_trace_capture_artifact(artifact, capture_mode="extract_only")

        self.assertEqual(
            {
                "row_id",
                "canonical_state",
                "legal_moves",
                "reference_move",
                "full_search_selected_move",
            },
            set(built["row_context"].keys()),
        )
        self.assertEqual(
            {"seed": 17, "simulation_count": 384, "search_settings": copy.deepcopy(payload["settings"]["search_settings"]), "reason": None},
            built["upstream_context"],
        )
        self.assertEqual(built["trace_points"], built["extracted_trace"]["trace_points"])
        self.assertIsNone(built["rerun_trace"])
        self.assertEqual(
            {
                "trace_origin": "extracted",
                "trace_points": built["trace_points"],
                "insufficiency_reasons": [],
            },
            built["final_trace"],
        )
        self.assertIn("extracted", built["trace_validation"])
        self.assertIn("final", built["trace_validation"])
        self.assertEqual(
            {
                "trace_origin_changed": False,
                "trace_points_changed": False,
                "selected_move_changed": False,
                "simulation_sequence_changed": False,
                "root_start_changed": False,
                "snapshots_changed": False,
                "extracted_trace_point_count": 3,
                "final_trace_point_count": 3,
                "trace_point_count_delta": 0,
                "extracted_first_simulation": 1.0,
                "final_first_simulation": 1.0,
                "extracted_final_selected_move": 0,
                "final_final_selected_move": 0,
                "extracted_final_simulation": 16.0,
                "final_final_simulation": 16.0,
                "full_search_selected_move": 0,
                "final_trace_matches_full_search_selected_move": True,
                "field_change_counts": {
                    "added_fields": 0,
                    "removed_fields": 0,
                    "changed_fields": 0,
                },
                "field_changes": {
                    "added_fields": [],
                    "removed_fields": [],
                    "changed_fields": [],
                },
            },
            built["trace_diff_summary"],
        )
        self.assertEqual(module.SCHEMA, built["provenance"]["trace_capture_schema"])
        self.assertEqual(str(artifact["artifact_path"]), built["provenance"]["source_shared_drift_artifact_path"])

    def test_build_artifact_records_trace_points_pair_mismatch_when_extraction_drifts(self):
        payload = self.valid_source_artifact()
        payload["rows"][module.ROW_ID]["snapshots"][0]["selected_move"] = 1
        payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["snapshots"][0]["selected_move"] = 1

        artifact = self.load_source_artifact(payload)

        built = module.build_trace_capture_artifact(artifact, capture_mode="extract_only")

        self.assertEqual("insufficient", built["trace_origin"])
        self.assertIn("trace_points_pair_mismatch", built["insufficiency_reasons"])
        self.assertEqual(3, len(built["trace_points"]))

    def test_build_artifact_preserves_extracted_reference_move_by_prior_for_validation(self):
        payload = self.valid_source_artifact()
        payload["rows"][module.ROW_ID]["snapshots"][0]["reference_move_by_prior"] = 1
        payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["snapshots"][0]["reference_move_by_prior"] = 1

        artifact = self.load_source_artifact(payload)

        built = module.build_trace_capture_artifact(artifact, capture_mode="extract_only")

        self.assertEqual(1, built["trace_points"][1]["reference_move_by_prior"])
        self.assertEqual("insufficient", built["trace_origin"])
        self.assertIn("trace_points_pair_mismatch", built["insufficiency_reasons"])

    def test_build_artifact_requires_trace_to_reach_full_search_selected_move(self):
        payload = self.valid_source_artifact()
        payload["rows"][module.ROW_ID]["full_search_selected_move"] = 0
        payload["rows"][module.ROW_ID]["snapshots"][0]["selected_move"] = 2
        payload["rows"][module.ROW_ID]["snapshots"][1]["selected_move"] = 2
        payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["selected_move"] = 0
        payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["snapshots"][0]["selected_move"] = 2
        payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["snapshots"][1]["selected_move"] = 2

        artifact = self.load_source_artifact(payload)

        built = module.build_trace_capture_artifact(artifact, capture_mode="extract_only")

        self.assertEqual("insufficient", built["trace_origin"])
        self.assertIn("trace_never_reaches_full_search_selected_move", built["insufficiency_reasons"])

    def test_build_artifact_requires_final_trace_point_to_match_full_search_selected_move(self):
        payload = self.valid_source_artifact()
        payload["rows"][module.ROW_ID]["snapshots"].insert(
            1,
            self.valid_trace_point(
                selected_move=0,
                simulation=8.0,
                visits=[4.0, 2.0, 5.0, 1.0, 0.0],
                legal_moves=payload["rows"][module.ROW_ID]["legal_moves"],
            ),
        )
        payload["rows"][module.ROW_ID]["snapshots"][-1]["selected_move"] = 2
        payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["snapshots"] = copy.deepcopy(
            payload["rows"][module.ROW_ID]["snapshots"]
        )

        artifact = self.load_source_artifact(payload)

        built = module.build_trace_capture_artifact(artifact, capture_mode="extract_only")

        self.assertEqual("insufficient", built["trace_origin"])
        self.assertIn("final_trace_selected_move_mismatch", built["insufficiency_reasons"])
        self.assertEqual(2, built["trace_points"][-1]["selected_move"])
        self.assertEqual(0, built["full_search_selected_move"])


class Capture002TraceCaptureRerunTest(Capture002TraceCaptureSourceArtifactTest):
    def load_source_artifact(self, payload: dict) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            self.write_json(source_path, payload)
            return module.load_source_shared_drift_artifact(source_path, allow_fixture_provenance=False)

    def valid_rerun_trace_points(self) -> list[dict]:
        return [
            self.valid_trace_point(
                selected_move=2,
                simulation=1.0,
                visits=[0.0, 0.0, 1.0, 0.0, 0.0],
                legal_moves=[0, 1, 2, 3, 4],
            ),
            self.valid_trace_point(
                selected_move=0,
                simulation=16.0,
                visits=[8.0, 4.0, 7.0, 2.0, 1.0],
                legal_moves=[0, 1, 2, 3, 4],
            ),
        ]

    def test_build_artifact_uses_rerun_trace_when_extraction_is_insufficient(self):
        payload = self.valid_source_artifact()
        payload["rows"][module.ROW_ID]["snapshots"][0]["selected_move"] = 1
        payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["snapshots"][0]["selected_move"] = 1
        artifact = self.load_source_artifact(payload)
        rerun_trace_points = self.valid_rerun_trace_points()
        expected_trace_points = copy.deepcopy(rerun_trace_points)
        for trace_point in expected_trace_points:
            trace_point["reference_move_by_prior"] = payload["rows"][module.ROW_ID]["reference_move"]

        built = module.build_trace_capture_artifact(
            artifact,
            capture_mode="extract_then_rerun",
            rerun_capture=lambda _source_artifact: {"trace_points": copy.deepcopy(rerun_trace_points)},
        )

        self.assertEqual("rerun", built["trace_origin"])
        self.assertEqual(expected_trace_points, built["trace_points"])
        self.assertEqual([], built["insufficiency_reasons"])
        self.assertEqual(
            {
                "seed": 17,
                "simulation_count": 384,
                "search_settings": copy.deepcopy(payload["settings"]["search_settings"]),
                "reason": None,
            },
            built["upstream_inputs"],
        )

    def test_build_artifact_uses_null_upstream_inputs_with_reason_when_rerun_returns_no_trace(self):
        payload = self.valid_source_artifact()
        payload["rows"][module.ROW_ID]["snapshots"][0]["selected_move"] = 1
        payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["snapshots"][0]["selected_move"] = 1
        artifact = self.load_source_artifact(payload)

        built = module.build_trace_capture_artifact(
            artifact,
            capture_mode="extract_then_rerun",
            rerun_capture=lambda _source_artifact: {},
        )

        self.assertEqual("insufficient", built["trace_origin"])
        self.assertIn("trace_points_pair_mismatch", built["insufficiency_reasons"])
        self.assertEqual(
            {
                "seed": None,
                "simulation_count": None,
                "search_settings": None,
                "reason": "rerun_unusable_missing_trace_points",
            },
            built["upstream_inputs"],
        )

    def test_build_artifact_preserves_explicit_default_rerun_failure_reason(self):
        payload = self.valid_source_artifact()
        payload["rows"][module.ROW_ID]["snapshots"][0]["selected_move"] = 1
        payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["snapshots"][0]["selected_move"] = 1
        artifact = self.load_source_artifact(payload)

        built = module.build_trace_capture_artifact(
            artifact,
            capture_mode="extract_then_rerun",
            rerun_capture=lambda _source_artifact: {"insufficiency_reasons": ["deterministic_rerun_inputs_incomplete"]},
        )

        self.assertEqual("insufficient", built["trace_origin"])
        self.assertEqual(
            {
                "seed": None,
                "simulation_count": None,
                "search_settings": None,
                "reason": "deterministic_rerun_inputs_incomplete",
            },
            built["upstream_inputs"],
        )
        self.assertEqual(
            {
                "trace_origin": "insufficient",
                "trace_points": [],
                "insufficiency_reasons": ["deterministic_rerun_inputs_incomplete"],
            },
            built["rerun_trace"],
        )

    def test_build_artifact_uses_null_upstream_inputs_with_reason_when_rerun_is_still_insufficient(self):
        payload = self.valid_source_artifact()
        payload["rows"][module.ROW_ID]["snapshots"][0]["selected_move"] = 1
        payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["snapshots"][0]["selected_move"] = 1
        artifact = self.load_source_artifact(payload)
        rerun_trace_points = [self.valid_rerun_trace_points()[0]]

        built = module.build_trace_capture_artifact(
            artifact,
            capture_mode="extract_then_rerun",
            rerun_capture=lambda _source_artifact: {"trace_points": copy.deepcopy(rerun_trace_points)},
        )

        self.assertEqual("insufficient", built["trace_origin"])
        self.assertIn("too_few_trace_points", built["insufficiency_reasons"])
        self.assertEqual(
            {
                "seed": None,
                "simulation_count": None,
                "search_settings": None,
                "reason": "rerun_unusable_trace_insufficient",
            },
            built["upstream_inputs"],
        )

    def test_build_artifact_rejects_rerun_trace_that_is_still_trace_insufficient_downstream(self):
        payload = self.valid_source_artifact()
        payload["rows"][module.ROW_ID]["snapshots"][0]["selected_move"] = 1
        payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["snapshots"][0]["selected_move"] = 1
        artifact = self.load_source_artifact(payload)

        built = module.build_trace_capture_artifact(
            artifact,
            capture_mode="extract_then_rerun",
            rerun_capture=lambda _source_artifact: {
                "trace_points": copy.deepcopy(
                    Capture002TraceCaptureRegeneratedArtifactTest.downstream_trace_insufficient_rerun_trace_points(self)
                )
            },
        )

        self.assertEqual("insufficient", built["trace_origin"])
        self.assertIn("missing_final_selection_score_margin", built["insufficiency_reasons"])
        self.assertEqual(
            {
                "seed": None,
                "simulation_count": None,
                "search_settings": None,
                "reason": "rerun_unusable_trace_insufficient",
            },
            built["upstream_inputs"],
        )

    def test_build_artifact_uses_null_upstream_inputs_with_reason_when_rerun_blocked(self):
        payload = self.valid_source_artifact()
        payload["rows"][module.ROW_ID]["snapshots"][0]["selected_move"] = 1
        payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["snapshots"][0]["selected_move"] = 1
        artifact = self.load_source_artifact(payload)

        built = module.build_trace_capture_artifact(artifact, capture_mode="extract_then_rerun")

        self.assertEqual("insufficient", built["trace_origin"])
        self.assertIn("trace_points_pair_mismatch", built["insufficiency_reasons"])
        self.assertEqual(
            {
                "seed": None,
                "simulation_count": None,
                "search_settings": None,
                "reason": "rerun_blocked",
            },
            built["upstream_inputs"],
        )

    def test_default_rerun_result_uses_deterministic_probe_trace_points(self):
        payload = self.valid_source_artifact()
        artifact = self.load_source_artifact(payload)
        captured_call = {}
        expected_trace_points = [
            {
                "simulation": 1,
                "selected_move": 2,
                "reference_move_by_prior": 2,
                "visits": [0.0, 0.0, 1.0, 0.0, 0.0],
                "moves": [
                    {
                        "move": 0,
                        "prior": 0.2,
                        "visit_count": 0,
                        "q_value": 0.1,
                        "selection_score": 0.2,
                        "used_fpu": True,
                    },
                    {
                        "move": 2,
                        "prior": 0.6,
                        "visit_count": 1,
                        "q_value": 0.3,
                        "selection_score": 0.8,
                        "used_fpu": False,
                    },
                ],
            },
            {
                "simulation": 16,
                "selected_move": 0,
                "reference_move_by_prior": 2,
                "visits": [8.0, 4.0, 7.0, 2.0, 1.0],
                "moves": [
                    {
                        "move": 0,
                        "prior": 0.2,
                        "visit_count": 8,
                        "q_value": 0.7,
                        "selection_score": 1.1,
                        "used_fpu": False,
                    },
                    {
                        "move": 2,
                        "prior": 0.6,
                        "visit_count": 7,
                        "q_value": 0.5,
                        "selection_score": 0.9,
                        "used_fpu": False,
                    },
                ],
            },
        ]

        def fake_probe_artifact_position(*, artifact_path, state, simulations, seed, c_puct, evaluator=None, search_options=None, ablation_mode="full"):
            del evaluator
            captured_call.update(
                {
                    "artifact_path": artifact_path,
                    "state": copy.deepcopy(state),
                    "simulations": simulations,
                    "seed": seed,
                    "c_puct": c_puct,
                    "search_options": copy.deepcopy(search_options),
                    "ablation_mode": ablation_mode,
                }
            )
            return {"visit_snapshots": copy.deepcopy(expected_trace_points)}

        with patch("ml.alphazero_lite.capture_002_trace_capture.search_policy_arbitration.probe_artifact_position", side_effect=fake_probe_artifact_position):
            rerun_result = module._default_rerun_result(artifact, capture_mode="extract_then_rerun")

        self.assertEqual({"trace_points": expected_trace_points}, rerun_result)
        self.assertEqual(payload["selected_artifact"]["path"], captured_call["artifact_path"])
        self.assertEqual(
            {
                "player_pits": [1, 0, 7, 6, 6, 5],
                "opponent_pits": [5, 4, 4, 4, 4, 0],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            },
            captured_call["state"],
        )
        self.assertEqual(payload["settings"]["simulation_count"], captured_call["simulations"])
        self.assertEqual(payload["settings"]["seed"], captured_call["seed"])
        self.assertEqual(payload["settings"]["search_settings"]["c_puct"], captured_call["c_puct"])
        self.assertEqual(payload["settings"]["search_settings"], captured_call["search_options"])
        self.assertEqual("full", captured_call["ablation_mode"])

    def test_default_rerun_result_converts_probe_failure_to_insufficiency(self):
        payload = self.valid_source_artifact()
        artifact = self.load_source_artifact(payload)

        with patch(
            "ml.alphazero_lite.capture_002_trace_capture.search_policy_arbitration.probe_artifact_position",
            side_effect=RuntimeError("probe failed"),
        ):
            rerun_result = module._default_rerun_result(artifact, capture_mode="extract_then_rerun")

        self.assertEqual({"insufficiency_reasons": ["deterministic_rerun_failed"]}, rerun_result)


class Capture002TraceCaptureRegeneratedArtifactTest(Capture002TraceCaptureSourceArtifactTest):
    def load_source_artifact(self, payload: dict) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            self.write_json(source_path, payload)
            return module.load_source_shared_drift_artifact(source_path, allow_fixture_provenance=False)

    def valid_rerun_trace_points(self) -> list[dict]:
        return [
            self.valid_trace_point(
                selected_move=2,
                simulation=1.0,
                visits=[0.0, 0.0, 1.0, 0.0, 0.0],
                legal_moves=[0, 1, 2, 3, 4],
            ),
            self.valid_trace_point(
                selected_move=0,
                simulation=16.0,
                visits=[8.0, 4.0, 7.0, 2.0, 1.0],
                legal_moves=[0, 1, 2, 3, 4],
            ),
        ]

    def transient_hit_then_drift_away_rerun_trace_points(self) -> list[dict]:
        return [
            self.valid_trace_point(
                selected_move=2,
                simulation=1.0,
                visits=[0.0, 0.0, 1.0, 0.0, 0.0],
                legal_moves=[0, 1, 2, 3, 4],
            ),
            self.valid_trace_point(
                selected_move=0,
                simulation=8.0,
                visits=[4.0, 2.0, 5.0, 1.0, 0.0],
                legal_moves=[0, 1, 2, 3, 4],
            ),
            self.valid_trace_point(
                selected_move=2,
                simulation=16.0,
                visits=[7.0, 4.0, 8.0, 2.0, 1.0],
                legal_moves=[0, 1, 2, 3, 4],
            ),
        ]

    def downstream_trace_insufficient_rerun_trace_points(self) -> list[dict]:
        return [
            {
                "selected_move": 2,
                "simulation": 1.0,
                "visits": [0.0, 0.0, 1.0, 0.0, 0.0],
                "moves": [],
            },
            {
                "selected_move": 0,
                "simulation": 16.0,
                "visits": [8.0, 4.0, 7.0, 2.0, 1.0],
                "moves": [],
            },
        ]

    def downstream_invalid_rerun_trace_points(self) -> list[dict]:
        return [
            {
                "selected_move": 2,
                "simulation": 1.0,
                "visits": [0.0, 0.0, 1.0],
                "moves": [],
            },
            {
                "selected_move": 0,
                "simulation": 16.0,
                "visits": [8.0, 4.0, 7.0],
                "moves": [],
            },
        ]

    def test_regenerated_shared_drift_artifact_is_emitted_only_when_downstream_ready(self):
        payload = self.valid_source_artifact()
        payload["rows"][module.ROW_ID]["snapshots"][0]["selected_move"] = 1
        payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["snapshots"][0]["selected_move"] = 1
        artifact = self.load_source_artifact(payload)

        built = module.build_trace_capture_artifact(
            artifact,
            capture_mode="extract_then_rerun",
            rerun_capture=lambda _source_artifact: {"trace_points": copy.deepcopy(self.valid_rerun_trace_points())},
        )

        regenerated = module.build_regenerated_shared_drift_artifact(built)
        downstream_artifact = selection_score_trace_module.load_source_shared_drift_artifact_document(
            regenerated,
            artifact_path="/tmp/regenerated-shared-drift.json",
        )

        self.assertIsNotNone(regenerated)
        self.assertEqual(module.SOURCE_SHARED_DRIFT_SCHEMA, regenerated["schema"])
        self.assertEqual(payload["classification"], regenerated["classification"])
        self.assertEqual(payload["decision"], regenerated["decision"])
        self.assertEqual(payload["selected_artifact"], regenerated["selected_artifact"])
        self.assertEqual(payload["rows"]["capture_available-003"], regenerated["rows"]["capture_available-003"])
        self.assertEqual(built["trace_points"][0], regenerated["rows"][module.ROW_ID]["root_start"])
        self.assertEqual(built["trace_points"][1:], regenerated["rows"][module.ROW_ID]["snapshots"])
        self.assertEqual(0, regenerated["rows"][module.ROW_ID]["full_search_selected_move"])
        self.assertEqual(
            {"selected_visits": 8.0, "reference_visits": 6.0},
            regenerated["rows"][module.ROW_ID]["final_deltas"],
        )
        self.assertEqual(
            regenerated["rows"][module.ROW_ID]["final_deltas"],
            regenerated["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["final_deltas"],
        )
        self.assertEqual(0, regenerated["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["selected_move"])
        self.assertEqual(0, regenerated["paired_summary"]["probe_mode_selected_moves"][module.ROW_ID]["full_search"])
        expected_downstream_trace_points = selection_score_trace_module.adapt_trace_points_for_downstream_shared_drift_artifact(
            built["trace_points"],
            legal_moves=payload["rows"][module.ROW_ID]["legal_moves"],
        )
        self.assertEqual(expected_downstream_trace_points[0], downstream_artifact["row"]["root_start"])
        self.assertEqual(expected_downstream_trace_points[1:], downstream_artifact["row"]["snapshots"])
        self.assertEqual(
            {
                "trace_capture_schema": module.SCHEMA,
                "trace_origin": "rerun",
                "row_id": module.ROW_ID,
                "trace_capture_artifact_path": None,
                "trace_capture_artifact_sha256": None,
                "source_shared_drift_artifact_path": artifact["artifact_path"],
                "source_shared_drift_artifact_sha256": artifact["artifact_sha256"],
            },
            regenerated["trace_capture_provenance"],
        )
        self.assertEqual(
            built["trace_diff_summary"],
            regenerated["trace_capture_diff_summary"],
        )

    def test_regenerated_shared_drift_artifact_is_not_emitted_for_insufficient_trace(self):
        payload = self.valid_source_artifact()
        payload["rows"][module.ROW_ID]["snapshots"][0]["selected_move"] = 1
        payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["snapshots"][0]["selected_move"] = 1
        artifact = self.load_source_artifact(payload)

        built = module.build_trace_capture_artifact(artifact, capture_mode="extract_then_rerun")

        self.assertEqual("insufficient", built["trace_origin"])
        self.assertIsNone(module.build_regenerated_shared_drift_artifact(built))

    def test_regenerated_shared_drift_artifact_is_not_emitted_when_downstream_classifies_trace_insufficient(self):
        payload = self.valid_source_artifact()
        payload["rows"][module.ROW_ID]["snapshots"][0]["selected_move"] = 1
        payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["snapshots"][0]["selected_move"] = 1
        artifact = self.load_source_artifact(payload)

        built = module.build_trace_capture_artifact(
            artifact,
            capture_mode="extract_then_rerun",
            rerun_capture=lambda _source_artifact: {
                "trace_points": copy.deepcopy(self.downstream_trace_insufficient_rerun_trace_points())
            },
        )

        self.assertEqual("insufficient", built["trace_origin"])
        self.assertIn("missing_final_selection_score_margin", built["insufficiency_reasons"])
        self.assertIsNone(module.build_regenerated_shared_drift_artifact(built))

    def test_regenerated_shared_drift_artifact_is_not_emitted_when_downstream_validation_raises(self):
        payload = self.valid_source_artifact()
        payload["rows"][module.ROW_ID]["snapshots"][0]["selected_move"] = 1
        payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["snapshots"][0]["selected_move"] = 1
        artifact = self.load_source_artifact(payload)

        built = module.build_trace_capture_artifact(
            artifact,
            capture_mode="extract_then_rerun",
            rerun_capture=lambda _source_artifact: {
                "trace_points": copy.deepcopy(self.downstream_invalid_rerun_trace_points())
            },
        )

        self.assertEqual("insufficient", built["trace_origin"])
        self.assertIn("missing_final_selection_score_margin", built["insufficiency_reasons"])
        self.assertIsNone(module.build_regenerated_shared_drift_artifact(built))

    def test_regenerated_shared_drift_artifact_is_not_emitted_when_canonical_shared_drift_contract_rejects_it(self):
        payload = self.valid_source_artifact()
        payload["rows"][module.ROW_ID]["snapshots"][0]["selected_move"] = 1
        payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["snapshots"][0]["selected_move"] = 1
        artifact = self.load_source_artifact(payload)

        built = module.build_trace_capture_artifact(
            artifact,
            capture_mode="extract_then_rerun",
            rerun_capture=lambda _source_artifact: {"trace_points": copy.deepcopy(self.valid_rerun_trace_points())},
        )
        built["source_shared_drift_artifact"]["source_payload"]["rows"]["capture_available-003"]["probe_mode_traces"][
            "full_search"
        ]["root_start"] = None

        self.assertEqual("rerun", built["trace_origin"])
        self.assertEqual([], built["insufficiency_reasons"])
        self.assertIsNone(module.build_regenerated_shared_drift_artifact(built))

    def test_regenerated_shared_drift_artifact_is_not_emitted_when_trace_hits_target_then_drifts_away(self):
        payload = self.valid_source_artifact()
        payload["rows"][module.ROW_ID]["snapshots"][0]["selected_move"] = 1
        payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["snapshots"][0]["selected_move"] = 1
        artifact = self.load_source_artifact(payload)

        built = module.build_trace_capture_artifact(
            artifact,
            capture_mode="extract_then_rerun",
            rerun_capture=lambda _source_artifact: {
                "trace_points": copy.deepcopy(self.transient_hit_then_drift_away_rerun_trace_points())
            },
        )

        self.assertEqual("insufficient", built["trace_origin"])
        self.assertIn("final_trace_selected_move_mismatch", built["insufficiency_reasons"])
        self.assertEqual(0, built["full_search_selected_move"])
        self.assertEqual(2, built["trace_points"][-1]["selected_move"])
        self.assertIsNone(module.build_regenerated_shared_drift_artifact(built))


class Capture002TraceCaptureCliTest(Capture002TraceCaptureSourceArtifactTest):
    def valid_rerun_trace_points(self) -> list[dict]:
        return [
            self.valid_trace_point(
                selected_move=2,
                simulation=1.0,
                visits=[0.0, 0.0, 1.0, 0.0, 0.0],
                legal_moves=[0, 1, 2, 3, 4],
            ),
            self.valid_trace_point(
                selected_move=0,
                simulation=16.0,
                visits=[8.0, 4.0, 7.0, 2.0, 1.0],
                legal_moves=[0, 1, 2, 3, 4],
            ),
        ]

    def test_main_uses_distinct_artifact_two_path_when_out_already_has_rehydrated_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "shared_drift.json"
            out_path = Path(tmp) / "capture_002_trace_rehydrated_shared_drift.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-002"]["snapshots"][0]["selected_move"] = 1
            payload["rows"]["capture_available-002"]["probe_mode_traces"]["full_search"]["snapshots"][0][
                "selected_move"
            ] = 1
            self.write_json(source_path, payload)

            with patch(
                "ml.alphazero_lite.capture_002_trace_capture.search_policy_arbitration.probe_artifact_position",
                return_value={"visit_snapshots": copy.deepcopy(self.valid_rerun_trace_points())},
            ):
                module.main(
                    [
                        "--source-shared-drift-artifact",
                        str(source_path),
                        "--out",
                        str(out_path),
                        "--capture-mode",
                        "extract_then_rerun",
                    ]
                )

            written = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertTrue(written["artifact_write_summary"]["regenerated_shared_drift_written"])
        self.assertNotEqual(str(out_path), written["artifact_write_summary"]["regenerated_shared_drift_path"])

    def test_main_writes_trace_capture_and_regenerated_artifact_when_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "shared_drift.json"
            out_path = Path(tmp) / "capture_002_trace_capture.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-002"]["snapshots"][0]["selected_move"] = 1
            payload["rows"]["capture_available-002"]["probe_mode_traces"]["full_search"]["snapshots"][0][
                "selected_move"
            ] = 1
            self.write_json(source_path, payload)

            stdout = io.StringIO()
            real_write_text = Path.write_text
            write_calls = []

            def recording_write_text(path_obj, data, *args, **kwargs):
                write_calls.append(str(path_obj))
                return real_write_text(path_obj, data, *args, **kwargs)

            with patch(
                "ml.alphazero_lite.capture_002_trace_capture.search_policy_arbitration.probe_artifact_position",
                return_value={"visit_snapshots": copy.deepcopy(self.valid_rerun_trace_points())},
            ):
                with patch("pathlib.Path.write_text", autospec=True, side_effect=recording_write_text):
                    with redirect_stdout(stdout):
                        exit_code = module.main(
                            [
                                "--source-shared-drift-artifact",
                                str(source_path),
                                "--out",
                                str(out_path),
                                "--capture-mode",
                                "extract_then_rerun",
                            ]
                        )

            written = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(0, exit_code)
            self.assertEqual(1, write_calls.count(str(out_path)))
            self.assertTrue(written["artifact_write_summary"]["regenerated_shared_drift_written"])
            self.assertTrue(Path(written["artifact_write_summary"]["regenerated_shared_drift_path"]).exists())
            self.assertEqual(module.sha256_file(out_path), written["artifact_write_summary"]["trace_capture_sha256"])
            self.assertEqual(str(out_path), json.loads(stdout.getvalue())["artifact_path"])

    def test_main_writes_only_artifact_one_when_trace_remains_insufficient(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "shared_drift.json"
            out_path = Path(tmp) / "capture_002_trace_capture.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-002"]["snapshots"][0]["selected_move"] = 1
            payload["rows"]["capture_available-002"]["probe_mode_traces"]["full_search"]["snapshots"][0][
                "selected_move"
            ] = 1
            self.write_json(source_path, payload)

            real_write_text = Path.write_text
            write_calls = []

            def recording_write_text(path_obj, data, *args, **kwargs):
                write_calls.append(str(path_obj))
                return real_write_text(path_obj, data, *args, **kwargs)

            with patch("pathlib.Path.write_text", autospec=True, side_effect=recording_write_text):
                module.main(
                    [
                        "--source-shared-drift-artifact",
                        str(source_path),
                        "--out",
                        str(out_path),
                        "--capture-mode",
                        "extract_only",
                    ]
                )

            written = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(1, write_calls.count(str(out_path)))
            self.assertFalse(written["artifact_write_summary"]["regenerated_shared_drift_written"])
            self.assertIsNone(written["artifact_write_summary"]["regenerated_shared_drift_path"])
            self.assertEqual(module.sha256_file(out_path), written["artifact_write_summary"]["trace_capture_sha256"])

    def test_main_accepts_allow_fixture_provenance_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "materialized_shared_drift.json"
            out_path = Path(tmp) / "capture_002_trace_capture.json"
            payload = self.valid_source_artifact()
            payload["selected_artifact"]["provenance_source"] = "materialized_fixture"
            self.write_json(source_path, payload)

            with self.assertRaisesRegex(ValueError, "allow_fixture_provenance"):
                module.main(
                    [
                        "--source-shared-drift-artifact",
                        str(source_path),
                        "--out",
                        str(out_path),
                    ]
                )

            exit_code = module.main(
                [
                    "--source-shared-drift-artifact",
                    str(source_path),
                    "--out",
                    str(out_path),
                    "--allow-fixture-provenance",
                ]
            )

        self.assertEqual(0, exit_code)

    def test_main_records_trace_capture_path_and_hash_in_regenerated_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "shared_drift.json"
            out_path = Path(tmp) / "capture_002_trace_capture.json"
            payload = self.valid_source_artifact()
            payload["rows"][module.ROW_ID]["snapshots"][0]["selected_move"] = 1
            payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["snapshots"][0]["selected_move"] = 1
            self.write_json(source_path, payload)

            with patch(
                "ml.alphazero_lite.capture_002_trace_capture.search_policy_arbitration.probe_artifact_position",
                return_value={"visit_snapshots": copy.deepcopy(self.valid_rerun_trace_points())},
            ):
                module.main(
                    [
                        "--source-shared-drift-artifact",
                        str(source_path),
                        "--out",
                        str(out_path),
                        "--capture-mode",
                        "extract_then_rerun",
                    ]
                )

            written = json.loads(out_path.read_text(encoding="utf-8"))
            regenerated_path = Path(written["artifact_write_summary"]["regenerated_shared_drift_path"])
            regenerated = json.loads(regenerated_path.read_text(encoding="utf-8"))

        self.assertEqual(str(out_path), regenerated["trace_capture_provenance"]["trace_capture_artifact_path"])
        self.assertEqual(
            written["artifact_write_summary"]["trace_capture_sha256"],
            regenerated["trace_capture_provenance"]["trace_capture_artifact_sha256"],
        )


class Capture002TraceCaptureDiffSummaryTest(Capture002TraceCaptureSourceArtifactTest):
    def load_source_artifact(self, payload: dict) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "shared_drift.json"
            self.write_json(source_path, payload)
            return module.load_source_shared_drift_artifact(source_path, allow_fixture_provenance=False)

    def test_build_artifact_emits_explicit_trace_diff_summary_fields(self):
        payload = self.valid_source_artifact()
        payload["rows"][module.ROW_ID]["snapshots"][0]["selected_move"] = 1
        payload["rows"][module.ROW_ID]["probe_mode_traces"]["full_search"]["snapshots"][0]["selected_move"] = 1
        artifact = self.load_source_artifact(payload)
        rerun_trace_points = Capture002TraceCaptureRerunTest.valid_rerun_trace_points(self)
        rerun_trace_points[1]["reference_move_by_prior"] = 1
        rerun_trace_points[1]["moves"][0]["diagnostic_note"] = "added"
        del rerun_trace_points[1]["moves"][1]["used_fpu"]

        built = module.build_trace_capture_artifact(
            artifact,
            capture_mode="extract_then_rerun",
            rerun_capture=lambda _source_artifact: {"trace_points": copy.deepcopy(rerun_trace_points)},
        )

        self.assertEqual(
            {
                "trace_origin_changed": True,
                "trace_points_changed": True,
                "selected_move_changed": False,
                "simulation_sequence_changed": True,
                "root_start_changed": False,
                "snapshots_changed": True,
                "extracted_trace_point_count": 3,
                "final_trace_point_count": 2,
                "trace_point_count_delta": -1,
                "extracted_first_simulation": 1.0,
                "final_first_simulation": 1.0,
                "extracted_final_selected_move": 0,
                "final_final_selected_move": 0,
                "extracted_final_simulation": 16.0,
                "final_final_simulation": 16.0,
                "full_search_selected_move": 0,
                "final_trace_matches_full_search_selected_move": True,
                "field_change_counts": {
                    "added_fields": 1,
                    "removed_fields": 1,
                    "changed_fields": 1,
                },
                "field_changes": {
                    "added_fields": ["moves.diagnostic_note"],
                    "removed_fields": ["moves.used_fpu"],
                    "changed_fields": ["reference_move_by_prior"],
                },
            },
            built["trace_diff_summary"],
        )
