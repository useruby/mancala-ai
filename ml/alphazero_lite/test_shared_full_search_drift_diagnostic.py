import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import shared_full_search_drift_diagnostic as module


class SharedFullSearchDriftDiagnosticContractTest(unittest.TestCase):
    def test_schema_row_ids_thresholds_and_classification_decisions_are_stable(self):
        self.assertEqual(
            "azlite_shared_full_search_drift_diagnostic_v1",
            module.SCHEMA,
        )
        self.assertEqual(
            "azlite_capture_002_003_search_policy_arbitration_v1",
            module.SOURCE_ARBITRATION_SCHEMA,
        )
        self.assertEqual(
            ["capture_available-002", "capture_available-003"],
            module.ROW_IDS,
        )
        self.assertEqual(
            {
                "meaningful_q_margin": 0.03,
                "small_q_margin": 0.03,
                "meaningful_visit_share_overtake": 0.05,
                "material_prior_distortion": 0.05,
                "early_snapshot_fraction": 0.10,
                "minimum_early_snapshot_count": 1,
            },
            module.THRESHOLDS,
        )
        self.assertEqual(
            {
                "shared_mechanism_disproved": "write_row_split_followup_spec",
                "root_prior_decay": "write_fpu_root_pressure_spec",
                "child_value_override": "write_child_value_override_spec",
                "backup_accumulation_drift": "write_backup_accumulation_spec",
                "fpu_or_unvisited_child_pressure": "write_fpu_root_pressure_spec",
                "tactical_root_bias_interaction": "write_tactical_root_bias_spec",
                "unresolved": "stop_unresolved",
            },
            module.CLASSIFICATION_DECISIONS,
        )


class SharedFullSearchDriftDiagnosticSourceArtifactTest(unittest.TestCase):
    REAL_SOURCE_ARTIFACT_PATH = Path(
        "/tmp/opencode/stable-failure-family-runs/tactical-stable-failure-family-rebalance-20260508/final/capture_002_003_search_policy_arbitration.json"
    )
    REQUIRED_SEARCH_SETTINGS = {
        "c_puct": 1.25,
        "fpu_mode": "zero",
        "normalize_values": True,
        "reuse_subtree": True,
        "root_policy_mode": "deterministic",
        "tactical_root_bias": 0.1,
    }

    def write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def write_raw(self, path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    def valid_source_artifact(self) -> dict:
        return {
            "schema": module.SOURCE_ARBITRATION_SCHEMA,
            "selected_artifact": {
                "path": "/tmp/artifacts/selected.bin",
                "selected_target": "/tmp/artifacts/selected.bin",
                "selected_artifact": None,
                "provenance_source": "selection_manifest.selected_target",
            },
            "rows": {
                "capture_available-002": {
                    "id": "capture_available-002",
                    "reference_move": 2,
                    "policy_only": {"selected_move": 2},
                    "value_only": {"selected_move": 2},
                    "full_search": {"selected_move": 0},
                },
                "capture_available-003": {
                    "id": "capture_available-003",
                    "reference_move": 4,
                    "policy_only": {"selected_move": 4},
                    "value_only": {"selected_move": 4},
                    "full_search": {"selected_move": 1},
                },
            },
            "settings": {
                "search_settings": dict(self.REQUIRED_SEARCH_SETTINGS),
                "seeds": [23, 23],
                "simulation_count": 512,
            },
            "classification": {"classification": "root_prior_decay"},
            "decision": "write_fpu_root_pressure_spec",
        }

    def valid_nested_probe_views_source_artifact(self) -> dict:
        return {
            "schema": module.SOURCE_ARBITRATION_SCHEMA,
            "selected_artifact": {
                "path": "/tmp/artifacts/selected.bin",
                "selected_target": "/tmp/artifacts/selected.bin",
                "selected_artifact": None,
                "provenance_source": "selection_manifest.selected_target",
            },
            "rows": {
                "capture_available-002": {
                    "row_id": "capture_available-002",
                    "reference_move": 2,
                    "probe_views": {
                        "policy_only": {
                            "search_view": {
                                "searched_selected_move": 2,
                                "visit_distribution": {"0": 4.0, "1": 5.0, "2": 7.0},
                                "reference_move_visit_share": 0.4375,
                                "selected_move_visit_share": 0.4375,
                                "child_stats": [
                                    {"move": 0, "q_value": 0.0, "visits": 4.0},
                                    {"move": 1, "q_value": 0.0, "visits": 5.0},
                                    {"move": 2, "q_value": 0.0, "visits": 7.0},
                                ],
                            },
                            "value_view": {"selected_minus_reference_q_margin": 0.0},
                        },
                        "value_only": {
                            "search_view": {
                                "searched_selected_move": 2,
                                "visit_distribution": {"0": 3.0, "1": 4.0, "2": 9.0},
                                "reference_move_visit_share": 0.5625,
                                "selected_move_visit_share": 0.5625,
                                "child_stats": [
                                    {"move": 0, "q_value": 0.11, "visits": 3.0},
                                    {"move": 1, "q_value": 0.08, "visits": 4.0},
                                    {"move": 2, "q_value": 0.13, "visits": 9.0},
                                ],
                            },
                            "value_view": {"selected_minus_reference_q_margin": 0.0},
                        },
                        "full_search": {
                            "search_view": {
                                "searched_selected_move": 0,
                                "visit_distribution": {"0": 8.0, "1": 1.0, "2": 7.0},
                                "reference_move_visit_share": 0.4375,
                                "selected_move_visit_share": 0.5,
                                "visit_snapshots": [
                                    {"simulation": 1, "selected_move": 2, "visits": [0.0, 0.0, 1.0]},
                                    {"simulation": 16, "selected_move": 0, "visits": [8.0, 1.0, 7.0]},
                                ],
                                "child_stats": [
                                    {"move": 0, "q_value": 0.14, "visits": 8.0},
                                    {"move": 1, "q_value": 0.01, "visits": 1.0},
                                    {"move": 2, "q_value": 0.11, "visits": 7.0},
                                ],
                            },
                            "value_view": {"selected_minus_reference_q_margin": 0.03},
                        },
                    },
                },
                "capture_available-003": {
                    "row_id": "capture_available-003",
                    "reference_move": 4,
                    "probe_views": {
                        "policy_only": {
                            "search_view": {
                                "searched_selected_move": 4,
                                "visit_distribution": {"0": 1.0, "1": 2.0, "2": 3.0, "3": 4.0, "4": 6.0},
                                "reference_move_visit_share": 0.375,
                                "selected_move_visit_share": 0.375,
                                "child_stats": [
                                    {"move": 0, "q_value": 0.0, "visits": 1.0},
                                    {"move": 1, "q_value": 0.0, "visits": 2.0},
                                    {"move": 2, "q_value": 0.0, "visits": 3.0},
                                    {"move": 3, "q_value": 0.0, "visits": 4.0},
                                    {"move": 4, "q_value": 0.0, "visits": 6.0},
                                ],
                            },
                            "value_view": {"selected_minus_reference_q_margin": 0.0},
                        },
                        "value_only": {
                            "search_view": {
                                "searched_selected_move": 4,
                                "visit_distribution": {"0": 0.0, "1": 2.0, "2": 1.0, "3": 3.0, "4": 10.0},
                                "reference_move_visit_share": 0.625,
                                "selected_move_visit_share": 0.625,
                                "child_stats": [
                                    {"move": 0, "q_value": 0.02, "visits": 0.0},
                                    {"move": 1, "q_value": 0.03, "visits": 2.0},
                                    {"move": 2, "q_value": 0.01, "visits": 1.0},
                                    {"move": 3, "q_value": 0.04, "visits": 3.0},
                                    {"move": 4, "q_value": 0.09, "visits": 10.0},
                                ],
                            },
                            "value_view": {"selected_minus_reference_q_margin": 0.0},
                        },
                        "full_search": {
                            "search_view": {
                                "searched_selected_move": 1,
                                "visit_distribution": {"0": 0.0, "1": 9.0, "2": 1.0, "3": 2.0, "4": 4.0},
                                "reference_move_visit_share": 0.25,
                                "selected_move_visit_share": 0.5625,
                                "child_stats": [
                                    {"move": 0, "q_value": 0.02, "visits": 0.0},
                                    {"move": 1, "q_value": 0.12, "visits": 9.0},
                                    {"move": 2, "q_value": 0.01, "visits": 1.0},
                                    {"move": 3, "q_value": 0.03, "visits": 2.0},
                                    {"move": 4, "q_value": 0.06, "visits": 4.0},
                                ],
                            },
                            "value_view": {"selected_minus_reference_q_margin": 0.06},
                        },
                    },
                },
            },
            "settings": {
                "search_settings": dict(self.REQUIRED_SEARCH_SETTINGS),
                "seeds": [23, 23],
                "simulation_count": 512,
            },
            "classification": {"classification": "root_prior_decay"},
            "decision": "write_fpu_root_pressure_spec",
        }

    def test_load_json_reads_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            payload = {"schema": "example"}
            self.write_json(path, payload)

            self.assertEqual(payload, module.load_json(path))

    def test_load_source_arbitration_artifact_returns_fail_closed_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            self.write_json(path, self.valid_source_artifact())

            artifact = module.load_source_arbitration_artifact(path)

        self.assertEqual(
            {
                "artifact_path": str(path),
                "schema": module.SOURCE_ARBITRATION_SCHEMA,
                "row_ids": ["capture_available-002", "capture_available-003"],
                "rows": {
                    "capture_available-002": {
                        "id": "capture_available-002",
                        "reference_move": 2,
                        "policy_only": {"selected_move": 2},
                        "value_only": {"selected_move": 2},
                        "full_search": {"selected_move": 0},
                    },
                    "capture_available-003": {
                        "id": "capture_available-003",
                        "reference_move": 4,
                        "policy_only": {"selected_move": 4},
                        "value_only": {"selected_move": 4},
                        "full_search": {"selected_move": 1},
                    },
                },
                "selected_artifact": {
                    "path": "/tmp/artifacts/selected.bin",
                    "selected_target": "/tmp/artifacts/selected.bin",
                    "selected_artifact": None,
                    "provenance_source": "selection_manifest.selected_target",
                },
                "source_settings": {
                    "search_settings": dict(self.REQUIRED_SEARCH_SETTINGS),
                    "seeds": [23, 23],
                    "simulation_count": 512,
                },
                "search_settings": dict(self.REQUIRED_SEARCH_SETTINGS),
                "seed": 23,
                "simulation_count": 512,
                "classification": {"classification": "root_prior_decay"},
                "decision": "write_fpu_root_pressure_spec",
            },
            artifact,
        )

    def test_load_source_arbitration_artifact_fails_closed_for_wrong_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["schema"] = "wrong_schema"
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "wrong schema"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_non_dict_top_level_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            self.write_raw(path, '[{"schema":"not-an-object-artifact"}]')

            with self.assertRaisesRegex(ValueError, "must be a JSON object"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_missing_required_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            del payload["rows"]["capture_available-003"]
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "capture_available-003"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_missing_full_search_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["settings"] = {}
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "search_settings"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_negative_flat_shape_selected_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"]["selected_move"] = -1
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "selected_move"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_missing_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            del payload["settings"]["seeds"]
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "seeds"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_non_uniform_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["settings"]["seeds"] = [23, 29]
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "identical"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_malformed_selected_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["selected_artifact"] = {
                "path": 17,
                "selected_target": "/tmp/artifacts/selected.bin",
                "selected_artifact": None,
                "provenance_source": None,
            }
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "selected_artifact"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_missing_reference_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            del payload["rows"]["capture_available-002"]["reference_move"]
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "reference_move"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_non_dict_probe_mode_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"] = []
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "full_search"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_malformed_visit_snapshots_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"]["visit_snapshots"] = {
                "simulation": 6
            }
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "visit_snapshots"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_malformed_snapshot_simulation_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"]["visit_snapshots"] = [
                {"simulation": "six", "visits": [0.0, 3.0, 0.0, 0.0, 5.0, 0.0]}
            ]
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "simulation"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_non_numeric_visit_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"]["visits"] = [0.0, "three", 0.0, 0.0, 5.0, 0.0]
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "visits"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_selected_move_out_of_range_for_visits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"]["selected_move"] = 7
            payload["rows"]["capture_available-003"]["full_search"]["visits"] = [0.0, 3.0, 0.0, 0.0, 5.0]
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "selected_move"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_snapshot_reference_move_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"]["visit_snapshots"] = [
                {"simulation": 6, "visits": [0.0, 3.0, 0.0]}
            ]
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "reference_move"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_malformed_prior_reference_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"]["prior_reference_move"] = "4"
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "prior_reference_move"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_malformed_prior_selected_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"]["prior_selected_move"] = "1"
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "prior_selected_move"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_malformed_child_stats_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"]["child_stats"] = [
                {"move": "1", "q_value": 0.25, "visits": 3.0}
            ]
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "child_stats"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_malformed_selected_minus_reference_q_margin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"]["selected_minus_reference_q_margin"] = "0.12"
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "selected_minus_reference_q_margin"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_malformed_selected_minus_reference_visit_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"]["selected_minus_reference_visit_share"] = "0.12"
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "selected_minus_reference_visit_share"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_malformed_child_stats_visits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"]["child_stats"] = [
                {"move": 1, "q_value": 0.25, "visits": "3"}
            ]
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "child_stats"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_malformed_snapshot_selected_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"]["visit_snapshots"] = [
                {"simulation": 6, "selected_move": "1", "visits": [0.0, 3.0, 0.0, 0.0, 5.0, 0.0]}
            ]
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "selected_move"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_malformed_snapshot_reference_move_by_prior(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"]["visit_snapshots"] = [
                {"simulation": 6, "reference_move_by_prior": "4", "visits": [0.0, 3.0, 0.0, 0.0, 5.0, 0.0]}
            ]
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "reference_move_by_prior"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_malformed_snapshot_rank_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"]["visit_snapshots"] = [
                {
                    "simulation": 6,
                    "reference_move_rank_by_visits": "1",
                    "reference_move_rank_by_q": "2",
                    "reference_move_rank_by_selection_score": "3",
                    "visits": [0.0, 3.0, 0.0, 0.0, 5.0, 0.0],
                }
            ]
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "reference_move_rank_by_visits"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_rejects_bool_where_integer_expected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["reference_move"] = True
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "reference_move"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_rejects_bool_where_numeric_expected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"]["selected_minus_reference_q_margin"] = True
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "selected_minus_reference_q_margin"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_malformed_child_stats_q_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"]["child_stats"] = [
                {"move": 1, "q_value": True, "visits": 3.0}
            ]
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "child_stats"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_rejects_bool_child_stats_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"]["child_stats"] = [
                {"move": True, "q_value": 0.25, "visits": 3.0}
            ]
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "child_stats"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_rejects_non_bool_tactical_bias_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["rows"]["capture_available-003"]["full_search"]["tactical_bias_applied"] = "yes"
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "tactical_bias_applied"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_missing_required_full_search_setting_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            del payload["settings"]["search_settings"]["tactical_root_bias"]
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "tactical_root_bias"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_invalid_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["classification"] = {"classification": "policy_prior_gap"}
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "unsupported classification"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_mismatched_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_source_artifact()
            payload["decision"] = "write_row_split_followup_spec"
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "decision"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_nested_non_integer_selected_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_nested_probe_views_source_artifact()
            payload["rows"]["capture_available-003"]["probe_views"]["full_search"]["search_view"]["searched_selected_move"] = "1"
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "selected_move"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_nested_non_numeric_visit_distribution_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_nested_probe_views_source_artifact()
            payload["rows"]["capture_available-003"]["probe_views"]["full_search"]["search_view"]["visit_distribution"]["1"] = "9"
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, r"search_view\.visit_distribution"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_negative_nested_visit_distribution_move_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_nested_probe_views_source_artifact()
            payload["rows"]["capture_available-003"]["probe_views"]["full_search"]["search_view"]["visit_distribution"] = {
                "-1": 9.0,
                "4": 4.0,
            }
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, r"search_view\.visit_distribution"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_out_of_range_nested_move_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_nested_probe_views_source_artifact()
            payload["rows"]["capture_available-003"]["probe_views"]["full_search"]["search_view"]["searched_selected_move"] = 999999
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "must be between 0 and 5"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_out_of_range_nested_visit_distribution_move_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_nested_probe_views_source_artifact()
            payload["rows"]["capture_available-003"]["probe_views"]["full_search"]["search_view"]["visit_distribution"] = {
                "999999": 9.0,
                "4": 4.0,
            }
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, r"search_view\.visit_distribution.*between 0 and 5"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_nested_non_numeric_visit_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_nested_probe_views_source_artifact()
            payload["rows"]["capture_available-003"]["probe_views"]["full_search"]["search_view"]["reference_move_visit_share"] = "0.25"
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, r"search_view\.reference_move_visit_share"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_nested_non_numeric_q_margin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_nested_probe_views_source_artifact()
            payload["rows"]["capture_available-003"]["probe_views"]["full_search"]["value_view"]["selected_minus_reference_q_margin"] = "0.06"
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "selected_minus_reference_q_margin"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_fails_closed_for_nested_missing_search_view_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            payload = self.valid_nested_probe_views_source_artifact()
            payload["rows"]["capture_available-003"]["probe_views"]["full_search"]["search_view"] = None
            self.write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "search_view"):
                module.load_source_arbitration_artifact(path)

    def test_load_source_arbitration_artifact_accepts_current_real_artifact_shape(self):
        if not self.REAL_SOURCE_ARTIFACT_PATH.exists():
            self.skipTest(f"real artifact not available: {self.REAL_SOURCE_ARTIFACT_PATH}")

        artifact = module.load_source_arbitration_artifact(self.REAL_SOURCE_ARTIFACT_PATH)

        self.assertEqual(str(self.REAL_SOURCE_ARTIFACT_PATH), artifact["artifact_path"])
        self.assertEqual(
            {
                "path": "/tmp/opencode/stable-failure-family-runs/tactical-stable-failure-family-rebalance-20260508/versions/tactical-stable-failure-family-rebalance-20260508-iter1",
                "provenance_source": "selection_manifest.selected_target",
                "selected_artifact": "/tmp/opencode/stable-failure-family-runs/tactical-stable-failure-family-rebalance-20260508/selection/artifact",
                "selected_target": "/tmp/opencode/stable-failure-family-runs/tactical-stable-failure-family-rebalance-20260508/versions/tactical-stable-failure-family-rebalance-20260508-iter1",
            },
            artifact["selected_artifact"],
        )

        normalized_row = artifact["rows"]["capture_available-003"]

        self.assertEqual(2, normalized_row["reference_move"])
        self.assertEqual("capture_available-003", normalized_row["row_id"])
        self.assertEqual(
            {
                "selected_move": 2,
                "visits": [44.0, 80.0, 167.0, 54.0, 39.0],
                "selected_minus_reference_q_margin": 0.0,
                "selected_minus_reference_visit_share": 0.0,
                "child_stats": [
                    {"move": 0, "q_value": 0.0, "visits": 44},
                    {"move": 1, "q_value": 0.0, "visits": 80},
                    {"move": 2, "q_value": 0.0, "visits": 167},
                    {"move": 3, "q_value": 0.0, "visits": 54},
                    {"move": 4, "q_value": 0.0, "visits": 39},
                ],
            },
            normalized_row["policy_only"],
        )
        self.assertEqual(
            {
                "selected_move": 1,
                "visits": [28.0, 164.0, 101.0, 44.0, 47.0],
                "selected_minus_reference_q_margin": -0.0111,
                "child_stats": [
                    {"move": 0, "q_value": -0.0977, "visits": 28},
                    {"move": 1, "q_value": 0.0627, "visits": 164},
                    {"move": 2, "q_value": 0.0738, "visits": 101},
                    {"move": 3, "q_value": 0.0115, "visits": 44},
                    {"move": 4, "q_value": 0.018, "visits": 47},
                ],
            },
            {
                key: value
                for key, value in normalized_row["value_only"].items()
                if key != "selected_minus_reference_visit_share"
            },
        )
        self.assertAlmostEqual(0.1641, normalized_row["value_only"]["selected_minus_reference_visit_share"])
        self.assertEqual(
            {
                "selected_move": 1,
                "visits": [22.0, 156.0, 150.0, 30.0, 26.0],
                "selected_minus_reference_q_margin": 0.0388,
                "child_stats": [
                    {"move": 0, "q_value": 0.0052, "visits": 22},
                    {"move": 1, "q_value": 0.1059, "visits": 156},
                    {"move": 2, "q_value": 0.0671, "visits": 150},
                    {"move": 3, "q_value": 0.0113, "visits": 30},
                    {"move": 4, "q_value": 0.0469, "visits": 26},
                ],
            },
            {
                key: value
                for key, value in normalized_row["full_search"].items()
                if key != "selected_minus_reference_visit_share"
            },
        )
        self.assertAlmostEqual(0.0156, normalized_row["full_search"]["selected_minus_reference_visit_share"])

    def test_load_source_arbitration_artifact_preserves_nested_visit_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            self.write_json(path, self.valid_nested_probe_views_source_artifact())

            artifact = module.load_source_arbitration_artifact(path)

        self.assertEqual(
            [
                {"simulation": 1, "selected_move": 2, "visits": [0.0, 0.0, 1.0]},
                {"simulation": 16, "selected_move": 0, "visits": [8.0, 1.0, 7.0]},
            ],
            artifact["rows"]["capture_available-002"]["full_search"]["visit_snapshots"],
        )


class SharedFullSearchDriftDiagnosticTraceTest(unittest.TestCase):
    def test_build_row_trace_persists_spec_required_full_search_fields(self):
        trace = module.build_row_trace(
            row_id="capture_available-002",
            canonical_state='{"board":[0,1,0,0,0,0],"current_player":0}',
            legal_moves=[0, 1, 2, 3, 4, 5],
            reference_move=2,
            probe_modes={
                "policy_only": {"selected_move": 2, "visits": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]},
                "value_only": {"selected_move": 2, "visits": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]},
                "full_search": {
                    "selected_move": 0,
                    "visits": [24.0, 8.0, 4.0, 0.0, 0.0, 0.0],
                    "visit_snapshots": [
                        {
                            "simulation": 16,
                            "selected_move": 2,
                            "reference_move_by_prior": 2,
                            "visits": [2.0, 1.0, 4.0, 0.0, 0.0, 0.0],
                        },
                        {
                            "simulation": 64,
                            "selected_move": 0,
                            "reference_move_by_prior": 2,
                            "visits": [24.0, 8.0, 4.0, 0.0, 0.0, 0.0],
                        },
                    ],
                },
            },
        )

        self.assertEqual(
            '{"board":[0,1,0,0,0,0],"current_player":0}',
            trace["canonical_state"],
        )
        self.assertEqual([0, 1, 2, 3, 4, 5], trace["legal_moves"])
        self.assertEqual(0, trace["full_search_selected_move"])
        self.assertEqual(
            {
                "simulation": 16,
                "selected_move": 2,
                "reference_move_by_prior": 2,
                "visits": [2.0, 1.0, 4.0, 0.0, 0.0, 0.0],
            },
            trace["root_start"],
        )
        self.assertEqual(
            [
                {
                    "simulation": 16,
                    "selected_move": 2,
                    "reference_move_by_prior": 2,
                    "visits": [2.0, 1.0, 4.0, 0.0, 0.0, 0.0],
                },
                {
                    "simulation": 64,
                    "selected_move": 0,
                    "reference_move_by_prior": 2,
                    "visits": [24.0, 8.0, 4.0, 0.0, 0.0, 0.0],
                },
            ],
            trace["snapshots"],
        )
        self.assertEqual(
            {"selected_visits": 22.0, "reference_visits": 0.0},
            trace["final_deltas"],
        )
        self.assertEqual([], trace["missing_fields"])
        self.assertEqual(2, trace["reference_move"])
        self.assertIn("probe_mode_traces", trace)
        self.assertAlmostEqual(
            20.0 / 36.0,
            trace["probe_mode_traces"]["full_search"]["selected_minus_reference_visit_share"],
        )

    def test_build_row_trace_records_missing_full_search_fields(self):
        trace = module.build_row_trace(
            row_id="capture_available-002",
            canonical_state=None,
            legal_moves=None,
            reference_move=2,
            probe_modes={
                "policy_only": {"selected_move": 2},
                "value_only": {"selected_move": 2},
                "full_search": {},
            },
        )

        self.assertIsNone(trace["canonical_state"])
        self.assertEqual([], trace["legal_moves"])
        self.assertIsNone(trace["full_search_selected_move"])
        self.assertIsNone(trace["root_start"])
        self.assertEqual([], trace["snapshots"])
        self.assertEqual({"selected_visits": None, "reference_visits": None}, trace["final_deltas"])
        self.assertEqual(["canonical_state", "legal_moves", "full_search_selected_move"], trace["missing_fields"])

    def test_build_paired_summary_persists_probe_mode_selected_moves_by_row_then_mode(self):
        row_traces = [
            module.build_row_trace(
                row_id="capture_available-002",
                canonical_state="state-002",
                legal_moves=[0, 1, 2, 3, 4, 5],
                reference_move=2,
                probe_modes={
                    "policy_only": {"selected_move": 2},
                    "value_only": {"selected_move": 2},
                    "full_search": {"selected_move": 0},
                },
            ),
            module.build_row_trace(
                row_id="capture_available-003",
                canonical_state="state-003",
                legal_moves=[0, 1, 2, 3, 4, 5],
                reference_move=4,
                probe_modes={
                    "policy_only": {"selected_move": 4},
                    "value_only": {"selected_move": 4},
                    "full_search": {"selected_move": 2},
                },
            ),
        ]

        paired_summary = module.build_paired_summary(row_traces)

        self.assertEqual(
            {
                "capture_available-002": {
                    "policy_only": 2,
                    "value_only": 2,
                    "full_search": 0,
                },
                "capture_available-003": {
                    "policy_only": 4,
                    "value_only": 4,
                    "full_search": 2,
                },
            },
            paired_summary["probe_mode_selected_moves"],
        )

    def test_build_paired_summary_persists_failure_paths_and_flags_split_value_only_behavior(self):
        row_traces = [
            module.build_row_trace(
                row_id="capture_available-002",
                canonical_state="state-002",
                legal_moves=[0, 1, 2, 3, 4, 5],
                reference_move=2,
                probe_modes={
                    "policy_only": {"selected_move": 2},
                    "value_only": {"selected_move": 2},
                    "full_search": {"selected_move": 2},
                },
            ),
            module.build_row_trace(
                row_id="capture_available-003",
                canonical_state="state-003",
                legal_moves=[0, 1, 2, 3, 4, 5],
                reference_move=4,
                probe_modes={
                    "policy_only": {"selected_move": 4},
                    "value_only": {"selected_move": 2},
                    "full_search": {"selected_move": 2},
                },
            ),
        ]

        paired_summary = module.build_paired_summary(row_traces)

        self.assertEqual(
            {
                "capture_available-002": {
                    "policy_only": "reference_kept",
                    "value_only": "reference_kept",
                    "full_search": "reference_kept",
                },
                "capture_available-003": {
                    "policy_only": "reference_kept",
                    "value_only": "diverged_before_full_search",
                    "full_search": "diverged_before_full_search",
                },
            },
            paired_summary["probe_mode_failure_paths"],
        )
        self.assertFalse(paired_summary["shared_mechanism_supported"])

    def test_build_paired_summary_marks_full_search_after_value_divergence_as_diverged_before_full_search(self):
        row_traces = [
            module.build_row_trace(
                row_id="capture_available-002",
                canonical_state="state-002",
                legal_moves=[0, 1, 2, 3, 4, 5],
                reference_move=2,
                probe_modes={
                    "policy_only": {"selected_move": 2},
                    "value_only": {"selected_move": 2},
                    "full_search": {"selected_move": 0},
                },
            ),
            module.build_row_trace(
                row_id="capture_available-003",
                canonical_state="state-003",
                legal_moves=[0, 1, 2, 3, 4, 5],
                reference_move=2,
                probe_modes={
                    "policy_only": {"selected_move": 2},
                    "value_only": {"selected_move": 1},
                    "full_search": {"selected_move": 1},
                },
            ),
        ]

        paired_summary = module.build_paired_summary(row_traces)

        self.assertEqual(
            "diverged_before_full_search",
            paired_summary["probe_mode_failure_paths"]["capture_available-003"]["full_search"],
        )

    def test_build_paired_summary_uses_explicit_full_search_drift_path(self):
        row_traces = [
            module.build_row_trace(
                row_id="capture_available-002",
                canonical_state="state-002",
                legal_moves=[0, 1, 2, 3, 4, 5],
                reference_move=2,
                probe_modes={
                    "policy_only": {"selected_move": 2},
                    "value_only": {"selected_move": 2},
                    "full_search": {"selected_move": 0},
                },
            ),
            module.build_row_trace(
                row_id="capture_available-003",
                canonical_state="state-003",
                legal_moves=[0, 1, 2, 3, 4, 5],
                reference_move=4,
                probe_modes={
                    "policy_only": {"selected_move": 4},
                    "value_only": {"selected_move": 4},
                    "full_search": {"selected_move": 4},
                },
            ),
        ]

        paired_summary = module.build_paired_summary(row_traces)

        self.assertEqual(
            {
                "capture_available-002": {
                    "policy_only": "reference_kept",
                    "value_only": "reference_kept",
                    "full_search": "full_search_drift",
                },
                "capture_available-003": {
                    "policy_only": "reference_kept",
                    "value_only": "reference_kept",
                    "full_search": "reference_kept",
                },
            },
            paired_summary["probe_mode_failure_paths"],
        )
        self.assertFalse(paired_summary["shared_mechanism_supported"])

    def test_build_paired_summary_marks_incomplete_row_as_not_applicable(self):
        row_traces = [
            module.build_row_trace(
                row_id="capture_available-002",
                canonical_state="state-002",
                legal_moves=[0, 1, 2, 3, 4, 5],
                reference_move=2,
                probe_modes={
                    "policy_only": {"selected_move": 2},
                    "value_only": {"selected_move": 2},
                    "full_search": {},
                },
            ),
            module.build_row_trace(
                row_id="capture_available-003",
                canonical_state="state-003",
                legal_moves=[0, 1, 2, 3, 4, 5],
                reference_move=4,
                probe_modes={
                    "policy_only": {"selected_move": 4},
                    "value_only": {"selected_move": 4},
                    "full_search": {"selected_move": 4},
                },
            ),
        ]

        paired_summary = module.build_paired_summary(row_traces)

        self.assertEqual(
            {
                "capture_available-002": {
                    "policy_only": "reference_kept",
                    "value_only": "reference_kept",
                    "full_search": "not_applicable",
                },
                "capture_available-003": {
                    "policy_only": "reference_kept",
                    "value_only": "reference_kept",
                    "full_search": "reference_kept",
                },
            },
            paired_summary["probe_mode_failure_paths"],
        )

    def test_build_paired_summary_does_not_support_shared_mechanism_with_incomplete_data(self):
        row_traces = [
            module.build_row_trace(
                row_id="capture_available-002",
                canonical_state="state-002",
                legal_moves=[0, 1, 2, 3, 4, 5],
                reference_move=2,
                probe_modes={
                    "policy_only": {"selected_move": 2},
                    "value_only": {"selected_move": 2},
                    "full_search": {},
                },
            ),
            module.build_row_trace(
                row_id="capture_available-003",
                canonical_state="state-003",
                legal_moves=[0, 1, 2, 3, 4, 5],
                reference_move=4,
                probe_modes={
                    "policy_only": {"selected_move": 4},
                    "value_only": {"selected_move": 4},
                    "full_search": {},
                },
            ),
        ]

        paired_summary = module.build_paired_summary(row_traces)

        self.assertEqual(
            {
                "capture_available-002": {
                    "policy_only": "reference_kept",
                    "value_only": "reference_kept",
                    "full_search": "not_applicable",
                },
                "capture_available-003": {
                    "policy_only": "reference_kept",
                    "value_only": "reference_kept",
                    "full_search": "not_applicable",
                },
            },
            paired_summary["probe_mode_failure_paths"],
        )
        self.assertFalse(paired_summary["shared_mechanism_supported"])

    def test_build_paired_summary_fails_closed_for_missing_required_row(self):
        row_traces = [
            module.build_row_trace(
                row_id="capture_available-002",
                canonical_state="state-002",
                legal_moves=[0, 1, 2, 3, 4, 5],
                reference_move=2,
                probe_modes={
                    "policy_only": {"selected_move": 2},
                    "value_only": {"selected_move": 2},
                    "full_search": {"selected_move": 2},
                },
            )
        ]

        with self.assertRaisesRegex(ValueError, "exact required row ids"):
            module.build_paired_summary(row_traces)

    def test_build_paired_summary_fails_closed_for_non_exact_row_id_set(self):
        row_traces = [
            module.build_row_trace(
                row_id="capture_available-002",
                canonical_state="state-002",
                legal_moves=[0, 1, 2, 3, 4, 5],
                reference_move=2,
                probe_modes={
                    "policy_only": {"selected_move": 2},
                    "value_only": {"selected_move": 2},
                    "full_search": {"selected_move": 2},
                },
            ),
            module.build_row_trace(
                row_id="capture_available-002",
                canonical_state="state-002b",
                legal_moves=[0, 1, 2, 3, 4, 5],
                reference_move=2,
                probe_modes={
                    "policy_only": {"selected_move": 2},
                    "value_only": {"selected_move": 2},
                    "full_search": {"selected_move": 2},
                },
            ),
        ]

        with self.assertRaisesRegex(ValueError, "exact required row ids"):
            module.build_paired_summary(row_traces)

    def test_build_row_trace_fails_closed_for_negative_selected_move(self):
        with self.assertRaisesRegex(ValueError, "selected_move"):
            module.build_row_trace(
                row_id="capture_available-002",
                canonical_state="state-002",
                legal_moves=[0, 1, 2, 3, 4, 5],
                reference_move=2,
                probe_modes={
                    "policy_only": {"selected_move": 2},
                    "value_only": {"selected_move": 2},
                    "full_search": {
                        "selected_move": -1,
                        "visits": [24.0, 8.0, 4.0, 0.0, 0.0, 0.0],
                        "visit_snapshots": [
                            {
                                "simulation": 16,
                                "selected_move": 2,
                                "reference_move_by_prior": 2,
                                "visits": [2.0, 1.0, 4.0, 0.0, 0.0, 0.0],
                            }
                        ],
                    },
                },
            )


class SharedFullSearchDriftDiagnosticClassificationTest(unittest.TestCase):
    FIXTURE_DIR = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "diagnostics"
        / "shared_full_search_drift"
    )

    def load_fixture(self, name: str) -> dict:
        return json.loads((self.FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))

    def assert_fixture_classification(self, name: str) -> None:
        fixture = self.load_fixture(name)

        classification = module.classify_paired_summary(
            rows=fixture["rows"],
            paired_summary=fixture["paired_summary"],
            thresholds=fixture.get("thresholds", module.THRESHOLDS),
        )

        self.assertEqual(fixture["expected_classification"], classification)
        self.assertEqual(
            fixture["expected_decision"],
            module.decision_for_classification(classification["classification"]),
        )

    def test_decision_for_classification_matches_contract(self):
        for classification, decision in module.CLASSIFICATION_DECISIONS.items():
            self.assertEqual(decision, module.decision_for_classification(classification))

    def test_shared_mechanism_disproved_fixture(self):
        self.assert_fixture_classification("shared_mechanism_disproved")

    def test_root_prior_decay_fixture(self):
        self.assert_fixture_classification("root_prior_decay")

    def test_child_value_override_fixture(self):
        self.assert_fixture_classification("child_value_override")

    def test_backup_accumulation_drift_fixture(self):
        self.assert_fixture_classification("backup_accumulation_drift")

    def test_fpu_or_unvisited_child_pressure_fixture(self):
        self.assert_fixture_classification("fpu_or_unvisited_child_pressure")

    def test_tactical_root_bias_interaction_fixture(self):
        self.assert_fixture_classification("tactical_root_bias_interaction")

    def test_unresolved_fixture(self):
        self.assert_fixture_classification("unresolved")

    def test_classification_precedence_prefers_shared_mechanism_disproved(self):
        rows = {
            "capture_available-002": {
                "policy_only": {"selected_move": 2},
                "value_only": {"selected_move": 2},
                "full_search": {
                    "selected_move": 0,
                    "prior_reference_move": 2,
                    "prior_selected_move": 0,
                    "selected_minus_reference_q_margin": 0.22,
                    "selected_minus_reference_visit_share": 0.18,
                    "visit_snapshots": [
                        {"simulation": 8, "visits": [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]},
                        {"simulation": 64, "visits": [14.0, 0.0, 4.0, 0.0, 0.0, 0.0]},
                    ],
                    "child_stats": [
                        {"move": 0, "q_value": 0.72, "visits": 14.0},
                        {"move": 2, "q_value": 0.50, "visits": 4.0},
                    ],
                },
            },
            "capture_available-003": {
                "policy_only": {"selected_move": 4},
                "value_only": {"selected_move": 1},
                "full_search": {
                    "selected_move": 3,
                    "prior_reference_move": 4,
                    "prior_selected_move": 3,
                    "selected_minus_reference_q_margin": 0.24,
                    "selected_minus_reference_visit_share": 0.21,
                    "visit_snapshots": [
                        {"simulation": 8, "visits": [0.0, 0.0, 0.0, 4.0, 0.0, 0.0]},
                        {"simulation": 64, "visits": [0.0, 1.0, 0.0, 15.0, 4.0, 0.0]},
                    ],
                    "child_stats": [
                        {"move": 3, "q_value": 0.71, "visits": 15.0},
                        {"move": 4, "q_value": 0.47, "visits": 4.0},
                    ],
                },
            },
        }
        paired_summary = {
            "probe_mode_failure_paths": {
                "capture_available-002": "full_search_drift",
                "capture_available-003": "diverged_before_full_search",
            },
            "shared_mechanism_supported": False,
        }

        self.assertEqual(
            {
                "classification": "shared_mechanism_disproved",
                "evidence_summary": "The paired rows do not share the same failure path, so a single shared full-search drift mechanism is not supported.",
            },
            module.classify_paired_summary(rows=rows, paired_summary=paired_summary, thresholds=module.THRESHOLDS),
        )

    def test_missing_q_margin_fails_closed_to_unresolved(self):
        rows = {
            "capture_available-002": {
                "full_search": {
                    "selected_move": 0,
                    "prior_reference_move": 2,
                    "prior_selected_move": 2,
                    "selected_minus_reference_q_margin": None,
                    "selected_minus_reference_visit_share": 0.12,
                    "visit_snapshots": [
                        {"simulation": 6, "visits": [1.0, 0.0, 5.0, 0.0, 0.0, 0.0]},
                        {"simulation": 64, "visits": [11.0, 0.0, 6.0, 0.0, 0.0, 0.0]},
                    ],
                }
            },
            "capture_available-003": {
                "full_search": {
                    "selected_move": 1,
                    "prior_reference_move": 4,
                    "prior_selected_move": 4,
                    "selected_minus_reference_q_margin": 0.01,
                    "selected_minus_reference_visit_share": 0.11,
                    "visit_snapshots": [
                        {"simulation": 6, "visits": [0.0, 1.0, 0.0, 0.0, 5.0, 0.0]},
                        {"simulation": 64, "visits": [0.0, 10.0, 0.0, 0.0, 6.0, 0.0]},
                    ],
                }
            },
        }
        paired_summary = {
            "probe_mode_failure_paths": {
                "capture_available-002": "full_search_drift",
                "capture_available-003": "full_search_drift",
            },
            "shared_mechanism_supported": True,
        }

        self.assertEqual(
            {
                "classification": "unresolved",
                "evidence_summary": "Shared full-search drift is supported, but the paired evidence does not isolate one approved mechanism.",
            },
            module.classify_paired_summary(rows=rows, paired_summary=paired_summary, thresholds=module.THRESHOLDS),
        )

    def test_non_numeric_snapshot_metadata_fails_closed_to_unresolved_for_early_evidence_branch(self):
        rows = {
            "capture_available-002": {
                "full_search": {
                    "selected_move": 0,
                    "prior_reference_move": 2,
                    "prior_selected_move": 0,
                    "selected_minus_reference_q_margin": 0.01,
                    "selected_minus_reference_visit_share": 0.14,
                    "visit_snapshots": [
                        {"simulation": "six", "visits": [3.0, 0.0, 5.0, 0.0, 0.0, 0.0]},
                        {"simulation": 64, "visits": [16.0, 0.0, 5.0, 0.0, 0.0, 0.0]},
                    ],
                }
            },
            "capture_available-003": {
                "full_search": {
                    "selected_move": 1,
                    "prior_reference_move": 4,
                    "prior_selected_move": 1,
                    "selected_minus_reference_q_margin": 0.01,
                    "selected_minus_reference_visit_share": 0.12,
                    "visit_snapshots": [
                        {"simulation": 6, "visits": [0.0, 3.0, 0.0, 0.0, 5.0, 0.0]},
                        {"simulation": 64, "visits": [0.0, 15.0, 0.0, 0.0, 5.0, 0.0]},
                    ],
                }
            },
        }
        paired_summary = {
            "probe_mode_failure_paths": {
                "capture_available-002": "full_search_drift",
                "capture_available-003": "full_search_drift",
            },
            "shared_mechanism_supported": True,
        }

        self.assertEqual(
            {
                "classification": "unresolved",
                "evidence_summary": "Shared full-search drift is supported, but the paired evidence does not isolate one approved mechanism.",
            },
            module.classify_paired_summary(rows=rows, paired_summary=paired_summary, thresholds=module.THRESHOLDS),
        )

    def test_single_snapshot_rows_can_supply_valid_early_evidence(self):
        rows = {
            "capture_available-002": {
                "full_search": {
                    "selected_move": 0,
                    "prior_reference_move": 2,
                    "prior_selected_move": 0,
                    "selected_minus_reference_q_margin": 0.01,
                    "selected_minus_reference_visit_share": 0.14,
                    "visit_snapshots": [
                        {"simulation": 6, "visits": [3.0, 0.0, 5.0, 0.0, 0.0, 0.0]}
                    ],
                }
            },
            "capture_available-003": {
                "full_search": {
                    "selected_move": 1,
                    "prior_reference_move": 4,
                    "prior_selected_move": 1,
                    "selected_minus_reference_q_margin": 0.01,
                    "selected_minus_reference_visit_share": 0.12,
                    "visit_snapshots": [
                        {"simulation": 6, "visits": [0.0, 3.0, 0.0, 0.0, 5.0, 0.0]}
                    ],
                }
            },
        }
        paired_summary = {
            "probe_mode_failure_paths": {
                "capture_available-002": "full_search_drift",
                "capture_available-003": "full_search_drift",
            },
            "shared_mechanism_supported": True,
        }

        self.assertEqual(
            {
                "classification": "root_prior_decay",
                "evidence_summary": "Both rows drift only after full search, early visits already tilt away from the prior reference move, and child Q margins stay below the meaningful threshold.",
            },
            module.classify_paired_summary(
                rows=rows,
                paired_summary=paired_summary,
                thresholds=module.THRESHOLDS,
                simulation_count=64,
            ),
        )

    def test_single_late_snapshot_rows_do_not_qualify_as_early_evidence(self):
        rows = {
            "capture_available-002": {
                "full_search": {
                    "selected_move": 0,
                    "prior_reference_move": 2,
                    "prior_selected_move": 0,
                    "selected_minus_reference_q_margin": 0.01,
                    "selected_minus_reference_visit_share": 0.14,
                    "visit_snapshots": [
                        {"simulation": 96, "visits": [3.0, 0.0, 5.0, 0.0, 0.0, 0.0]}
                    ],
                }
            },
            "capture_available-003": {
                "full_search": {
                    "selected_move": 1,
                    "prior_reference_move": 4,
                    "prior_selected_move": 1,
                    "selected_minus_reference_q_margin": 0.01,
                    "selected_minus_reference_visit_share": 0.12,
                    "visit_snapshots": [
                        {"simulation": 96, "visits": [0.0, 3.0, 0.0, 0.0, 5.0, 0.0]}
                    ],
                }
            },
        }
        paired_summary = {
            "probe_mode_failure_paths": {
                "capture_available-002": "full_search_drift",
                "capture_available-003": "full_search_drift",
            },
            "shared_mechanism_supported": True,
        }

        self.assertEqual(
            {
                "classification": "unresolved",
                "evidence_summary": "Shared full-search drift is supported, but the paired evidence does not isolate one approved mechanism.",
            },
            module.classify_paired_summary(
                rows=rows,
                paired_summary=paired_summary,
                thresholds=module.THRESHOLDS,
                simulation_count=128,
            ),
        )


class SharedFullSearchDriftDiagnosticCliTest(unittest.TestCase):
    REQUIRED_SEARCH_SETTINGS = {
        "c_puct": 1.25,
        "fpu_mode": "zero",
        "normalize_values": True,
        "reuse_subtree": True,
        "root_policy_mode": "deterministic",
        "tactical_root_bias": 0.1,
    }

    def write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def valid_source_artifact(self) -> dict:
        return {
            "schema": module.SOURCE_ARBITRATION_SCHEMA,
            "selected_artifact": {
                "path": "/tmp/artifacts/selected.bin",
                "selected_target": "/tmp/artifacts/selected.bin",
                "selected_artifact": None,
                "provenance_source": "selection_manifest.selected_target",
            },
            "rows": {
                "capture_available-002": {
                    "reference_move": 2,
                    "full_search": {
                        "selected_move": 0,
                        "prior_reference_move": 2,
                        "prior_selected_move": 0,
                        "selected_minus_reference_q_margin": 0.01,
                        "selected_minus_reference_visit_share": 0.14,
                        "visit_snapshots": [
                            {"simulation": 6, "visits": [3.0, 0.0, 5.0, 0.0, 0.0, 0.0]}
                        ],
                    },
                    "policy_only": {"selected_move": 2},
                    "value_only": {"selected_move": 2},
                },
                "capture_available-003": {
                    "reference_move": 4,
                    "full_search": {
                        "selected_move": 1,
                        "prior_reference_move": 4,
                        "prior_selected_move": 1,
                        "selected_minus_reference_q_margin": 0.01,
                        "selected_minus_reference_visit_share": 0.12,
                        "visit_snapshots": [
                            {"simulation": 6, "visits": [0.0, 3.0, 0.0, 0.0, 5.0, 0.0]}
                        ],
                    },
                    "policy_only": {"selected_move": 4},
                    "value_only": {"selected_move": 4},
                },
            },
            "settings": {
                "search_settings": dict(self.REQUIRED_SEARCH_SETTINGS),
                "seeds": [23, 23],
                "simulation_count": 512,
            },
            "classification": {"classification": "root_prior_decay"},
            "decision": "write_fpu_root_pressure_spec",
        }

    def test_main_writes_self_contained_artifact_using_source_arbitration_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source_artifacts" / "arbitration.json"
            out_path = Path(tmp) / "artifacts" / "diagnostic.json"
            self.write_json(source_path, self.valid_source_artifact())

            payload_rows = {
                "capture_available-002": {
                    "row_id": "capture_available-002",
                    "canonical_state": "state-002",
                    "legal_moves": [0, 1, 2, 3, 4, 5],
                    "reference_move": 2,
                    "full_search_selected_move": 0,
                    "root_start": None,
                    "snapshots": [],
                    "final_deltas": {"selected_visits": None, "reference_visits": None},
                    "missing_fields": [],
                    "probe_mode_traces": {
                        "policy_only": {"selected_move": 2},
                        "value_only": {"selected_move": 2},
                        "full_search": {"selected_move": 0},
                    },
                },
                "capture_available-003": {
                    "row_id": "capture_available-003",
                    "canonical_state": "state-003",
                    "legal_moves": [0, 1, 2, 3, 4, 5],
                    "reference_move": 4,
                    "full_search_selected_move": 1,
                    "root_start": None,
                    "snapshots": [],
                    "final_deltas": {"selected_visits": None, "reference_visits": None},
                    "missing_fields": [],
                    "probe_mode_traces": {
                        "policy_only": {"selected_move": 4},
                        "value_only": {"selected_move": 4},
                        "full_search": {"selected_move": 1},
                    },
                },
            }
            stdout = io.StringIO()
            with (
                patch.object(module, "build_rows_payload", return_value=payload_rows),
                redirect_stdout(stdout),
            ):
                exit_code = module.main([
                    "--source-arbitration-artifact",
                    str(source_path),
                    "--out",
                    str(out_path),
                ])

            self.assertEqual(0, exit_code)
            artifact = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(module.SCHEMA, artifact["schema"])
            self.assertEqual(
                {
                    "artifact_path": str(source_path),
                    "schema": module.SOURCE_ARBITRATION_SCHEMA,
                    "row_ids": list(module.ROW_IDS),
                    "selected_artifact": self.valid_source_artifact()["selected_artifact"],
                    "settings": {
                        "search_settings": dict(self.REQUIRED_SEARCH_SETTINGS),
                        "seeds": [23, 23],
                        "simulation_count": 512,
                    },
                    "classification": {"classification": "root_prior_decay"},
                    "decision": "write_fpu_root_pressure_spec",
                },
                artifact["source_arbitration_artifact"],
            )
            self.assertEqual(self.valid_source_artifact()["selected_artifact"], artifact["selected_artifact"])
            self.assertEqual(module.THRESHOLDS, artifact["thresholds"])
            self.assertEqual(
                {
                    "search_settings": dict(self.REQUIRED_SEARCH_SETTINGS),
                    "seed": 23,
                    "simulation_count": 512,
                },
                artifact["settings"],
            )
            self.assertEqual(payload_rows, artifact["rows"])
            self.assertEqual(
                {
                    "probe_mode_selected_moves": {
                        "capture_available-002": {
                            "policy_only": 2,
                            "value_only": 2,
                            "full_search": 0,
                        },
                        "capture_available-003": {
                            "policy_only": 4,
                            "value_only": 4,
                            "full_search": 1,
                        },
                    },
                    "probe_mode_failure_paths": {
                        "capture_available-002": {
                            "policy_only": "reference_kept",
                            "value_only": "reference_kept",
                            "full_search": "full_search_drift",
                        },
                        "capture_available-003": {
                            "policy_only": "reference_kept",
                            "value_only": "reference_kept",
                            "full_search": "full_search_drift",
                        },
                    },
                    "shared_mechanism_supported": True,
                },
                artifact["paired_summary"],
            )
            self.assertEqual(
                {
                    "classification": "root_prior_decay",
                    "evidence_summary": "Both rows drift only after full search, early visits already tilt away from the prior reference move, and child Q margins stay below the meaningful threshold.",
                },
                artifact["classification"],
            )
            self.assertEqual("write_fpu_root_pressure_spec", artifact["decision"])
            self.assertEqual(
                {
                    "artifact_path": str(out_path),
                    "schema": module.SCHEMA,
                    "decision": "write_fpu_root_pressure_spec",
                },
                json.loads(stdout.getvalue().strip()),
            )
