import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ml.alphazero_lite import capture_002_003_row_split_followup as module


class Capture002003RowSplitFollowupContractTest(unittest.TestCase):
    def test_schema_thresholds_row_ids_and_decision_maps_are_stable(self):
        self.assertEqual(
            "azlite_capture_002_003_row_split_followup_v1",
            module.SCHEMA,
        )
        self.assertEqual(
            "azlite_shared_full_search_drift_diagnostic_v1",
            module.SOURCE_SHARED_DRIFT_SCHEMA,
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
                "early_snapshot_fraction": 0.10,
                "minimum_early_snapshot_count": 1,
                "material_selection_score_margin": 0.05,
                "material_prior_margin": 0.05,
            },
            module.THRESHOLDS,
        )
        self.assertEqual(
            {
                "selection_score_overtake": "write_002_selection_score_trace_spec",
                "early_fpu_pressure": "write_002_fpu_pressure_ablation_spec",
                "early_child_value_override": "write_002_child_value_audit_spec",
                "backup_accumulation_drift": "write_002_backup_accumulation_spec",
                "prior_pressure_with_small_q": "write_002_root_pressure_spec",
                "unresolved": "stop_002_unresolved",
            },
            module.LANE_002_DECISIONS,
        )
        self.assertEqual(
            {
                "value_only_child_q_prefers_wrong_move": "write_003_child_value_audit_spec",
                "value_only_visit_amplification_without_q": "write_003_value_only_visit_trace_spec",
                "policy_value_conflict": "write_003_policy_value_conflict_spec",
                "rule_feature_value_collision": "write_003_rule_value_collision_spec",
                "insufficient_value_trace": "write_003_value_trace_capture_spec",
                "unresolved": "stop_003_unresolved",
            },
            module.LANE_003_DECISIONS,
        )
        self.assertEqual(
            [
                "write_002_selection_score_trace_spec",
                "write_002_fpu_pressure_ablation_spec",
                "write_002_child_value_audit_spec",
                "write_002_backup_accumulation_spec",
                "write_002_root_pressure_spec",
                "write_003_child_value_audit_spec",
                "write_003_value_only_visit_trace_spec",
                "write_003_policy_value_conflict_spec",
                "write_003_rule_value_collision_spec",
                "write_003_value_trace_capture_spec",
                "write_parallel_row_followup_specs",
                "stop_row_split_unresolved",
            ],
            module.TOP_LEVEL_DECISIONS,
        )

    def test_load_json_reads_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.json"
            payload = {"schema": module.SOURCE_SHARED_DRIFT_SCHEMA}
            source_path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertEqual(payload, module.load_json(source_path))

    def test_parse_args_reads_required_paths(self):
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

    def test_parse_args_requires_source_shared_drift_artifact(self):
        with self.assertRaises(SystemExit):
            module.parse_args(["--out", "/tmp/diagnostic.json"])

    def test_parse_args_requires_out(self):
        with self.assertRaises(SystemExit):
            module.parse_args(["--source-shared-drift-artifact", "/tmp/source.json"])


class Capture002003RowSplitFollowupSourceArtifactTest(unittest.TestCase):
    def write_json(self, path: Path, payload) -> None:
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

    def load_source_artifact(self, payload: dict):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            self.write_json(path, payload)
            return module.load_source_shared_drift_artifact(path)

    def test_rejects_non_object_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            self.write_json(path, ["not", "an", "object"])

            with self.assertRaisesRegex(ValueError, "must be a JSON object"):
                module.load_source_shared_drift_artifact(path)

    def test_wrong_schema_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["schema"] = "wrong_schema"

        with self.assertRaisesRegex(ValueError, "wrong schema"):
            self.load_source_artifact(payload)

    def test_classification_other_than_shared_mechanism_disproved_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["classification"]["classification"] = "root_prior_decay"

        with self.assertRaisesRegex(ValueError, "shared_mechanism_disproved"):
            self.load_source_artifact(payload)

    def test_missing_required_row_fails_closed(self):
        payload = self.valid_source_artifact()
        del payload["rows"]["capture_available-003"]

        with self.assertRaisesRegex(ValueError, "capture_available-003"):
            self.load_source_artifact(payload)

    def test_missing_row_id_fails_closed(self):
        payload = self.valid_source_artifact()
        del payload["rows"]["capture_available-003"]["row_id"]

        with self.assertRaisesRegex(ValueError, r"row_id"):
            self.load_source_artifact(payload)

    def test_mismatched_row_id_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"]["row_id"] = "capture_available-999"

        with self.assertRaisesRegex(ValueError, r"row_id"):
            self.load_source_artifact(payload)

    def test_missing_selected_artifact_fails_closed(self):
        payload = self.valid_source_artifact()
        del payload["selected_artifact"]

        with self.assertRaisesRegex(ValueError, "selected_artifact"):
            self.load_source_artifact(payload)

    def test_missing_paired_summary_failure_paths_fails_closed(self):
        for key in ["paired_summary", "probe_mode_failure_paths"]:
            with self.subTest(key=key):
                payload = self.valid_source_artifact()
                if key == "paired_summary":
                    del payload["paired_summary"]
                else:
                    del payload["paired_summary"]["probe_mode_failure_paths"]

                with self.assertRaisesRegex(ValueError, "probe_mode_failure_paths|paired_summary"):
                    self.load_source_artifact(payload)

    def test_missing_policy_only_failure_path_entry_fails_closed(self):
        payload = self.valid_source_artifact()
        del payload["paired_summary"]["probe_mode_failure_paths"]["capture_available-003"]["policy_only"]

        with self.assertRaisesRegex(ValueError, r"policy_only"):
            self.load_source_artifact(payload)

    def test_missing_value_only_failure_path_entry_fails_closed(self):
        payload = self.valid_source_artifact()
        del payload["paired_summary"]["probe_mode_failure_paths"]["capture_available-003"]["value_only"]

        with self.assertRaisesRegex(ValueError, r"value_only"):
            self.load_source_artifact(payload)

    def test_invalid_failure_path_value_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["paired_summary"]["probe_mode_failure_paths"]["capture_available-003"]["value_only"] = "bogus"

        with self.assertRaisesRegex(ValueError, r"failure path"):
            self.load_source_artifact(payload)

    def test_missing_probe_mode_selected_moves_fails_closed(self):
        payload = self.valid_source_artifact()
        del payload["paired_summary"]["probe_mode_selected_moves"]

        with self.assertRaisesRegex(ValueError, r"probe_mode_selected_moves"):
            self.load_source_artifact(payload)

    def test_missing_probe_mode_selected_move_entry_fails_closed(self):
        payload = self.valid_source_artifact()
        del payload["paired_summary"]["probe_mode_selected_moves"]["capture_available-003"]["value_only"]

        with self.assertRaisesRegex(ValueError, r"probe_mode_selected_moves|value_only"):
            self.load_source_artifact(payload)

    def test_illegal_probe_mode_selected_move_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["paired_summary"]["probe_mode_selected_moves"]["capture_available-003"]["value_only"] = 8

        with self.assertRaisesRegex(ValueError, r"probe_mode_selected_moves|value_only"):
            self.load_source_artifact(payload)

    def test_mismatched_probe_mode_selected_move_summary_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["paired_summary"]["probe_mode_selected_moves"]["capture_available-003"]["value_only"] = 2

        with self.assertRaisesRegex(ValueError, r"probe_mode_selected_moves|value_only"):
            self.load_source_artifact(payload)

    def test_missing_shared_mechanism_supported_fails_closed(self):
        payload = self.valid_source_artifact()
        del payload["paired_summary"]["shared_mechanism_supported"]

        with self.assertRaisesRegex(ValueError, r"shared_mechanism_supported"):
            self.load_source_artifact(payload)

    def test_non_bool_shared_mechanism_supported_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["paired_summary"]["shared_mechanism_supported"] = "false"

        with self.assertRaisesRegex(ValueError, r"shared_mechanism_supported"):
            self.load_source_artifact(payload)

    def test_true_shared_mechanism_supported_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["paired_summary"]["shared_mechanism_supported"] = True

        with self.assertRaisesRegex(ValueError, r"shared_mechanism_supported"):
            self.load_source_artifact(payload)

    def test_unexpected_source_decision_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["decision"] = "stop_unresolved"

        with self.assertRaisesRegex(ValueError, r"write_row_split_followup_spec|decision"):
            self.load_source_artifact(payload)

    def test_missing_search_settings_key_fails_closed(self):
        payload = self.valid_source_artifact()
        del payload["settings"]["search_settings"]["tactical_root_bias"]

        with self.assertRaisesRegex(ValueError, "search_settings"):
            self.load_source_artifact(payload)

    def test_extra_search_settings_keys_are_preserved(self):
        payload = self.valid_source_artifact()
        payload["settings"]["search_settings"]["dirichlet_alpha"] = 0.03

        artifact = self.load_source_artifact(payload)

        self.assertEqual(payload["settings"]["search_settings"], artifact["settings"]["search_settings"])

    def test_missing_seed_or_invalid_simulation_count_fails_closed(self):
        invalid_cases = [
            ("missing_seed", lambda payload: payload["settings"].pop("seed"), "seed"),
            ("seed_not_int", lambda payload: payload["settings"].__setitem__("seed", "17"), "seed"),
            (
                "simulation_count_not_positive",
                lambda payload: payload["settings"].__setitem__("simulation_count", 0),
                "simulation_count",
            ),
        ]

        for case_name, mutate, expected_error in invalid_cases:
            with self.subTest(case=case_name):
                payload = self.valid_source_artifact()
                mutate(payload)

                with self.assertRaisesRegex(ValueError, expected_error):
                    self.load_source_artifact(payload)

    def test_infinite_setting_numeric_value_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["settings"]["search_settings"]["c_puct"] = float("inf")

        with self.assertRaisesRegex(ValueError, r"search_settings|c_puct"):
            self.load_source_artifact(payload)

    def test_stale_capture_002_full_search_shape_fails_closed(self):
        for case_name, root_start, snapshots in [
            ("missing_root_start", None, None),
            ("empty_snapshots", object(), []),
        ]:
            with self.subTest(case=case_name):
                payload = self.valid_source_artifact()
                trace = payload["rows"]["capture_available-002"]["probe_mode_traces"]["full_search"]
                if root_start is None:
                    trace["root_start"] = None
                if snapshots is not None:
                    trace["snapshots"] = snapshots

                with self.assertRaisesRegex(ValueError, "capture_available-002"):
                    self.load_source_artifact(payload)

    def test_capture_003_empty_full_search_snapshots_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"]["probe_mode_traces"]["full_search"]["snapshots"] = []
        payload["rows"]["capture_available-003"]["snapshots"] = []

        with self.assertRaisesRegex(ValueError, r"capture_available-003|full_search snapshots"):
            self.load_source_artifact(payload)

    def test_capture_003_missing_value_only_snapshots_without_declared_missing_fields_fails_closed(self):
        payload = self.valid_source_artifact()
        trace = payload["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]
        trace["snapshots"] = []

        with self.assertRaisesRegex(ValueError, "capture_available-003"):
            self.load_source_artifact(payload)

    def test_capture_003_missing_value_only_snapshots_with_declared_missing_fields_is_allowed(self):
        payload = self.valid_source_artifact()
        row = payload["rows"]["capture_available-003"]
        trace = row["probe_mode_traces"]["value_only"]
        trace["snapshots"] = []
        row["missing_fields"] = list(row.get("missing_fields") or []) + ["value_only.snapshots"]

        artifact = self.load_source_artifact(payload)

        self.assertEqual(
            "value_only.snapshots",
            artifact["rows"]["capture_available-003"]["missing_fields"][-1],
        )
        self.assertEqual(
            [],
            artifact["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["snapshots"],
        )

    def test_capture_003_missing_value_only_root_start_without_declared_missing_fields_fails_closed(self):
        payload = self.valid_source_artifact()
        trace = payload["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]
        trace["root_start"] = None

        with self.assertRaisesRegex(ValueError, r"value_only\.root_start|capture_available-003"):
            self.load_source_artifact(payload)

    def test_capture_003_missing_value_only_root_start_with_declared_missing_fields_is_allowed(self):
        payload = self.valid_source_artifact()
        row = payload["rows"]["capture_available-003"]
        trace = row["probe_mode_traces"]["value_only"]
        trace["root_start"] = None
        row["missing_fields"] = list(row.get("missing_fields") or []) + ["value_only.root_start"]

        artifact = self.load_source_artifact(payload)

        self.assertEqual(
            "value_only.root_start",
            artifact["rows"]["capture_available-003"]["missing_fields"][-1],
        )
        self.assertIsNone(
            artifact["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["root_start"]
        )

    def test_capture_003_declared_missing_value_only_root_start_does_not_allow_malformed_data(self):
        payload = self.valid_source_artifact()
        row = payload["rows"]["capture_available-003"]
        row["probe_mode_traces"]["value_only"]["root_start"] = "bad"
        row["missing_fields"] = list(row.get("missing_fields") or []) + ["value_only.root_start"]

        with self.assertRaisesRegex(ValueError, r"value_only\.root_start"):
            self.load_source_artifact(payload)

    def test_capture_003_stale_missing_value_only_flags_fail_closed_when_trace_data_is_present(self):
        payload = self.valid_source_artifact()
        row = payload["rows"]["capture_available-003"]
        row["missing_fields"] = ["value_only.root_start", "value_only.snapshots"]

        with self.assertRaisesRegex(ValueError, r"missing_fields|value_only\.(root_start|snapshots)"):
            self.load_source_artifact(payload)

    def test_capture_003_missing_value_only_snapshots_with_root_start_present_and_declared_is_allowed(self):
        payload = self.valid_source_artifact()
        row = payload["rows"]["capture_available-003"]
        trace = row["probe_mode_traces"]["value_only"]
        trace.pop("snapshots")
        row["missing_fields"] = list(row.get("missing_fields") or []) + ["value_only.snapshots"]

        artifact = self.load_source_artifact(payload)

        self.assertEqual(
            [],
            artifact["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["snapshots"],
        )

    def test_capture_003_upstream_empty_value_only_trace_without_declared_missing_fields_is_allowed(self):
        payload = self.valid_source_artifact()
        row = payload["rows"]["capture_available-003"]
        trace = row["probe_mode_traces"]["value_only"]
        trace["root_start"] = None
        trace["snapshots"] = []

        artifact = self.load_source_artifact(payload)

        self.assertEqual([], artifact["rows"]["capture_available-003"]["missing_fields"])
        self.assertIsNone(
            artifact["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["root_start"]
        )
        self.assertEqual(
            [],
            artifact["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["snapshots"],
        )

    def test_capture_003_missing_fields_rejects_non_string_entries(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"]["missing_fields"] = [{"field": "value_only.root_start"}]

        with self.assertRaisesRegex(ValueError, r"missing_fields.*strings"):
            self.load_source_artifact(payload)

    def test_capture_003_declared_missing_value_only_snapshots_do_not_allow_malformed_data(self):
        payload = self.valid_source_artifact()
        row = payload["rows"]["capture_available-003"]
        row["probe_mode_traces"]["value_only"]["snapshots"] = "bad"
        row["missing_fields"] = list(row.get("missing_fields") or []) + ["value_only.snapshots"]

        with self.assertRaisesRegex(ValueError, r"value_only\.snapshots"):
            self.load_source_artifact(payload)

    def test_capture_003_missing_row_level_root_start_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"].pop("root_start")

        with self.assertRaisesRegex(ValueError, r"row capture_available-003 root_start"):
            self.load_source_artifact(payload)

    def test_capture_003_missing_row_level_snapshots_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"].pop("snapshots")

        with self.assertRaisesRegex(ValueError, r"row capture_available-003 snapshots"):
            self.load_source_artifact(payload)

    def test_capture_003_missing_row_level_final_deltas_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"].pop("final_deltas")

        with self.assertRaisesRegex(ValueError, r"row capture_available-003 final_deltas"):
            self.load_source_artifact(payload)

    def test_capture_003_full_search_root_start_wrong_type_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"]["probe_mode_traces"]["full_search"]["root_start"] = []
        payload["rows"]["capture_available-003"]["root_start"] = []

        with self.assertRaisesRegex(ValueError, r"full_search root_start"):
            self.load_source_artifact(payload)

    def test_capture_003_full_search_snapshots_wrong_type_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"]["probe_mode_traces"]["full_search"]["snapshots"] = "bad"
        payload["rows"]["capture_available-003"]["snapshots"] = "bad"

        with self.assertRaisesRegex(ValueError, r"full_search snapshots"):
            self.load_source_artifact(payload)

    def test_full_search_final_deltas_missing_required_key_fails_closed(self):
        payload = self.valid_source_artifact()
        del payload["rows"]["capture_available-003"]["probe_mode_traces"]["full_search"]["final_deltas"]["selected_visits"]
        del payload["rows"]["capture_available-003"]["final_deltas"]["selected_visits"]

        with self.assertRaisesRegex(ValueError, r"final_deltas"):
            self.load_source_artifact(payload)

    def test_bool_selected_move_in_consumed_trace_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["selected_move"] = True

        with self.assertRaisesRegex(ValueError, r"selected_move"):
            self.load_source_artifact(payload)

    def test_missing_capture_002_policy_only_selected_move_fails_closed(self):
        payload = self.valid_source_artifact()
        del payload["rows"]["capture_available-002"]["probe_mode_traces"]["policy_only"]["selected_move"]

        with self.assertRaisesRegex(ValueError, r"selected_move|policy_only"):
            self.load_source_artifact(payload)

    def test_illegal_selected_move_in_consumed_trace_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["selected_move"] = 5

        with self.assertRaisesRegex(ValueError, r"selected_move"):
            self.load_source_artifact(payload)

    def test_root_start_missing_required_moves_fails_closed(self):
        payload = self.valid_source_artifact()
        del payload["rows"]["capture_available-003"]["probe_mode_traces"]["full_search"]["root_start"]["moves"]

        with self.assertRaisesRegex(ValueError, r"root_start|moves"):
            self.load_source_artifact(payload)

    def test_snapshot_entry_missing_required_selected_move_fails_closed(self):
        payload = self.valid_source_artifact()
        del payload["rows"]["capture_available-003"]["probe_mode_traces"]["full_search"]["snapshots"][0]["selected_move"]

        with self.assertRaisesRegex(ValueError, r"snapshots|selected_move"):
            self.load_source_artifact(payload)

    def test_move_entry_missing_required_selection_score_fails_closed(self):
        payload = self.valid_source_artifact()
        del payload["rows"]["capture_available-003"]["probe_mode_traces"]["full_search"]["root_start"]["moves"][0]["selection_score"]
        del payload["rows"]["capture_available-003"]["root_start"]["moves"][0]["selection_score"]

        with self.assertRaisesRegex(ValueError, r"moves|selection_score"):
            self.load_source_artifact(payload)

    def test_malformed_numeric_field_in_final_deltas_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["final_deltas"]["selected_visits"] = "bad"

        with self.assertRaisesRegex(ValueError, r"final_deltas|selected_visits"):
            self.load_source_artifact(payload)

    def test_null_consumed_final_deltas_field_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["final_deltas"]["selected_visits"] = None

        with self.assertRaisesRegex(ValueError, r"final_deltas|selected_visits"):
            self.load_source_artifact(payload)

    def test_non_numeric_root_start_simulation_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["root_start"]["simulation"] = "bad"

        with self.assertRaisesRegex(ValueError, r"root_start|simulation"):
            self.load_source_artifact(payload)

    def test_nan_consumed_numeric_field_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["root_start"]["moves"][0]["prior"] = float("nan")

        with self.assertRaisesRegex(ValueError, r"prior"):
            self.load_source_artifact(payload)

    def test_non_list_snapshot_visits_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["snapshots"][0]["visits"] = "bad"

        with self.assertRaisesRegex(ValueError, r"snapshots|visits"):
            self.load_source_artifact(payload)

    def test_non_numeric_entry_in_visits_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["snapshots"][0]["visits"][1] = "bad"

        with self.assertRaisesRegex(ValueError, r"visits"):
            self.load_source_artifact(payload)

    def test_too_short_visits_list_for_legal_moves_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["snapshots"][0]["visits"] = [1.0, 0.0]

        with self.assertRaisesRegex(ValueError, r"visits"):
            self.load_source_artifact(payload)

    def test_illegal_move_entry_move_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["root_start"]["moves"][0]["move"] = 5

        with self.assertRaisesRegex(ValueError, r"moves|move"):
            self.load_source_artifact(payload)

    def test_duplicate_move_entry_in_moves_fails_closed(self):
        payload = self.valid_source_artifact()
        moves = payload["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["root_start"]["moves"]
        moves[-1]["move"] = moves[0]["move"]

        with self.assertRaisesRegex(ValueError, r"moves|legal_moves"):
            self.load_source_artifact(payload)

    def test_missing_legal_move_entry_in_moves_fails_closed(self):
        payload = self.valid_source_artifact()
        moves = payload["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["root_start"]["moves"]
        payload["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["root_start"]["moves"] = moves[:-1]

        with self.assertRaisesRegex(ValueError, r"moves|legal_moves"):
            self.load_source_artifact(payload)

    def test_non_bool_used_fpu_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["root_start"]["moves"][0]["used_fpu"] = "bad"

        with self.assertRaisesRegex(ValueError, r"moves|used_fpu"):
            self.load_source_artifact(payload)

    def test_duplicate_legal_moves_entry_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"]["legal_moves"] = [0, 1, 1, 3, 4]

        with self.assertRaisesRegex(ValueError, r"legal_moves"):
            self.load_source_artifact(payload)

    def test_negative_legal_moves_entry_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"]["legal_moves"] = [0, 1, -1, 3, 4]

        with self.assertRaisesRegex(ValueError, r"legal_moves"):
            self.load_source_artifact(payload)

    def test_out_of_range_probe_mode_selected_move_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-003"]["legal_moves"] = [0, 1, 2, 3, 99]
        payload["rows"]["capture_available-003"]["probe_mode_traces"]["value_only"]["selected_move"] = 99
        payload["paired_summary"]["probe_mode_selected_moves"]["capture_available-003"]["value_only"] = 99

        with self.assertRaisesRegex(ValueError, r"selected_move|legal_moves"):
            self.load_source_artifact(payload)

    def test_full_search_alias_mismatch_fails_closed(self):
        payload = self.valid_source_artifact()
        row = payload["rows"]["capture_available-002"]
        row["full_search_selected_move"] = row["probe_mode_traces"]["full_search"]["selected_move"] + 1

        with self.assertRaisesRegex(ValueError, "full_search_selected_move"):
            self.load_source_artifact(payload)

    def test_invalid_reference_move_fails_closed(self):
        payload = self.valid_source_artifact()
        payload["rows"]["capture_available-002"]["reference_move"] = 5

        with self.assertRaisesRegex(ValueError, "reference_move"):
            self.load_source_artifact(payload)

    def test_invalid_full_search_selected_move_fails_closed(self):
        payload = self.valid_source_artifact()
        row = payload["rows"]["capture_available-002"]
        row["full_search_selected_move"] = 5

        with self.assertRaisesRegex(ValueError, "full_search_selected_move"):
            self.load_source_artifact(payload)

    def test_happy_path_preserves_selected_artifact_provenance_and_settings(self):
        payload = self.valid_source_artifact()

        artifact = self.load_source_artifact(payload)

        self.assertEqual(payload["selected_artifact"], artifact["selected_artifact"])
        self.assertEqual(payload["settings"], artifact["settings"])
        self.assertEqual(payload["paired_summary"], artifact["paired_summary"])
        self.assertEqual(payload["classification"], artifact["classification"])
        self.assertEqual(payload["decision"], artifact["decision"])
        self.assertEqual(
            payload["rows"]["capture_available-002"]["canonical_state"],
            artifact["rows"]["capture_available-002"]["canonical_state"],
        )
        self.assertEqual(
            payload["rows"]["capture_available-002"]["probe_mode_traces"]["full_search"]["root_start"],
            artifact["rows"]["capture_available-002"]["probe_mode_traces"]["full_search"]["root_start"],
        )

    def test_invalid_canonical_state_shape_fails_closed(self):
        invalid_cases = [
            ("malformed_json", "not-json", "canonical_state"),
            (
                "truncated_pits",
                '{"player_pits":[1,6],"opponent_pits":[5,5]}',
                "canonical_state",
            ),
        ]

        for case_name, canonical_state, expected_error in invalid_cases:
            with self.subTest(case=case_name):
                payload = self.valid_source_artifact()
                payload["rows"]["capture_available-003"]["canonical_state"] = canonical_state

                with self.assertRaisesRegex(ValueError, expected_error):
                    self.load_source_artifact(payload)


class Capture002003RowSplitFollowupTraceTest(Capture002003RowSplitFollowupSourceArtifactTest):
    def test_visit_share_from_trace_uses_full_visit_distribution(self):
        trace_point = self.valid_trace_point(
            selected_move=1,
            simulation=128.0,
            visits=[22.0, 156.0, 150.0, 30.0, 26.0],
            legal_moves=[0, 1, 2, 3, 4],
        )

        visit_share = module._visit_share_from_trace(trace_point, selected_move=1, reference_move=2)

        self.assertAlmostEqual(0.015625, visit_share)

    def test_build_lane_002_uses_final_snapshot_visits_not_final_deltas_for_visit_share(self):
        payload = self.valid_source_artifact()
        row = payload["rows"]["capture_available-002"]
        full_search = row["probe_mode_traces"]["full_search"]
        row["final_deltas"] = {"selected_visits": 999.0, "reference_visits": 1.0}
        full_search["final_deltas"] = copy.deepcopy(row["final_deltas"])

        artifact = self.load_source_artifact(payload)

        lane = module.build_lane_002(artifact)

        self.assertAlmostEqual(1.0 / 22.0, lane["derived_metrics"]["final_selected_minus_reference_visit_share"])

    def test_build_lane_003_uses_final_snapshot_visits_not_final_deltas_for_visit_share(self):
        payload = self.valid_source_artifact()
        row = payload["rows"]["capture_available-003"]
        value_only = row["probe_mode_traces"]["value_only"]
        full_search = row["probe_mode_traces"]["full_search"]
        row["final_deltas"] = {"selected_visits": 999.0, "reference_visits": 1.0}
        value_only["final_deltas"] = {"selected_visits": 999.0, "reference_visits": 1.0}
        full_search["final_deltas"] = copy.deepcopy(row["final_deltas"])

        artifact = self.load_source_artifact(payload)

        lane = module.build_lane_003(artifact)

        self.assertAlmostEqual(0.25, lane["derived_metrics"]["value_only_selected_minus_reference_visit_share"])
        self.assertAlmostEqual(1.0 / 19.0, lane["derived_metrics"]["full_search_selected_minus_reference_visit_share"])

    def test_build_lane_002_populates_trace_inputs_and_derived_metrics(self):
        payload = self.valid_source_artifact()
        row = payload["rows"]["capture_available-002"]
        full_search = row["probe_mode_traces"]["full_search"]
        row["final_deltas"] = {"selected_visits": 999.0, "reference_visits": 1.0}
        full_search["final_deltas"] = copy.deepcopy(row["final_deltas"])

        root_start_moves = {
            0: {"prior": 0.35, "q_value": 0.12, "selection_score": 0.20},
            1: {"prior": 0.15, "q_value": 0.10, "selection_score": 0.10},
            2: {"prior": 0.20, "q_value": 0.18, "selection_score": 0.75},
            3: {"prior": 0.18, "q_value": 0.11, "selection_score": 0.50},
            4: {"prior": 0.12, "q_value": 0.09, "selection_score": 0.60},
        }
        final_snapshot_moves = {
            0: {"prior": 0.35, "q_value": 0.42, "selection_score": 0.90},
            1: {"prior": 0.15, "q_value": 0.13, "selection_score": 0.12},
            2: {"prior": 0.20, "q_value": 0.38, "selection_score": 0.70},
            3: {"prior": 0.18, "q_value": 0.14, "selection_score": 0.40},
            4: {"prior": 0.12, "q_value": 0.11, "selection_score": 0.30},
        }

        for move_entry in full_search["root_start"]["moves"]:
            values = root_start_moves[move_entry["move"]]
            move_entry.update(values)

        for move_entry in full_search["snapshots"][0]["moves"]:
            values = root_start_moves[move_entry["move"]]
            move_entry.update(values)

        for move_entry in full_search["snapshots"][1]["moves"]:
            values = final_snapshot_moves[move_entry["move"]]
            move_entry.update(values)

        row["root_start"] = copy.deepcopy(full_search["root_start"])
        row["snapshots"] = copy.deepcopy(full_search["snapshots"])

        artifact = self.load_source_artifact(payload)

        lane = module.build_lane_002(artifact)

        self.assertEqual("capture_available-002", lane["row_id"])
        self.assertEqual("full_search_drift", lane["failure_path"])
        self.assertEqual(row["canonical_state"], lane["canonical_state"])
        self.assertEqual(row["legal_moves"], lane["legal_moves"])
        self.assertEqual(2, lane["reference_move"])
        self.assertEqual(
            {"policy_only": 2, "value_only": 2, "full_search": 0},
            lane["selected_moves_by_probe_mode"],
        )
        self.assertEqual(0, lane["trace_inputs"]["full_search"]["selected_move"])
        self.assertEqual([], lane["missing_fields"])
        self.assertEqual(
            {
                "reference_move": 2,
                "full_search_selected_move": 0,
                "root_start_reference_rank_by_selection_score": 1,
                "root_start_selected_rank_by_selection_score": 4,
                "root_start_reference_q": 0.18,
                "root_start_selected_q": 0.12,
                "root_start_reference_selection_score": 0.75,
                "root_start_selected_selection_score": 0.20,
                "first_selected_visit_overtake_snapshot": 16.0,
                "first_selected_q_overtake_snapshot": 16.0,
                "first_selected_selection_score_overtake_snapshot": 16.0,
                "final_selected_minus_reference_visit_share": 0.0454545455,
                "final_selected_minus_reference_q_margin": 0.04,
                "final_selected_minus_reference_prior_margin": 0.15,
                "missing_fields": [],
            },
            lane["derived_metrics"],
        )

    def test_build_lane_003_sets_value_only_snapshot_metrics_to_null_when_declared_missing(self):
        payload = self.valid_source_artifact()
        row = payload["rows"]["capture_available-003"]
        value_only = row["probe_mode_traces"]["value_only"]
        value_only["snapshots"] = []
        value_only["final_deltas"] = {"selected_visits": 11.0, "reference_visits": 5.0}
        row["missing_fields"] = ["value_only.snapshots"]

        policy_only = row["probe_mode_traces"]["policy_only"]
        for move_entry in policy_only["root_start"]["moves"]:
            if move_entry["move"] == 2:
                move_entry["prior"] = 0.41
            if move_entry["move"] == 1:
                move_entry["prior"] = 0.27

        artifact = self.load_source_artifact(payload)

        lane = module.build_lane_003(artifact)

        self.assertEqual("capture_available-003", lane["row_id"])
        self.assertEqual("diverged_before_full_search", lane["failure_path"])
        self.assertEqual(
            {"policy_only": 2, "value_only": 1, "full_search": 1},
            lane["selected_moves_by_probe_mode"],
        )
        self.assertEqual([], lane["trace_inputs"]["value_only"]["snapshots"])
        self.assertEqual(["value_only.snapshots"], lane["missing_fields"])
        self.assertEqual(2, lane["derived_metrics"]["reference_move"])
        self.assertEqual(1, lane["derived_metrics"]["value_only_selected_move"])
        self.assertEqual(1, lane["derived_metrics"]["full_search_selected_move"])
        self.assertEqual(2, lane["derived_metrics"]["policy_only_selected_move"])
        self.assertAlmostEqual(0.41, lane["derived_metrics"]["policy_reference_prior"])
        self.assertAlmostEqual(0.27, lane["derived_metrics"]["policy_value_selected_prior"])
        self.assertEqual(["value_only.snapshots"], lane["derived_metrics"]["missing_fields"])
        self.assertEqual(
            {
                "player_empty_pits": [2],
                "opponent_empty_pits": [5],
                "reference_pit_stones": 0,
            },
            lane["derived_metrics"]["rule_features"],
        )
        self.assertIsNone(lane["derived_metrics"]["value_only_selected_minus_reference_q_margin"])
        self.assertIsNone(lane["derived_metrics"]["value_only_selected_minus_reference_visit_share"])
        self.assertIsNone(lane["derived_metrics"]["value_only_first_selected_visit_overtake_snapshot"])
        self.assertIsNone(lane["derived_metrics"]["value_only_first_selected_q_overtake_snapshot"])
        self.assertIsNone(lane["derived_metrics"]["value_only_first_selected_selection_score_overtake_snapshot"])


class Capture002003RowSplitFollowupLane002ClassificationTest(Capture002003RowSplitFollowupSourceArtifactTest):
    FIXTURE_DIR = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "diagnostics"
        / "capture_002_003_row_split_followup"
    )

    def load_fixture(self, name: str) -> dict:
        return json.loads((self.FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))

    def assert_fixture_classification(self, name: str) -> None:
        fixture = self.load_fixture(name)

        classification = module.classify_lane_002(
            lane=fixture["lane"],
            thresholds=fixture.get("thresholds", module.THRESHOLDS),
        )

        self.assertEqual(fixture["expected_classification"], classification)
        self.assertEqual(fixture["expected_classification"]["classification"], classification["classification"])
        self.assertEqual(fixture["expected_classification"]["decision"], classification["decision"])

    def test_selection_score_overtake_fixture(self):
        self.assert_fixture_classification("002_selection_score_overtake")

    def test_early_fpu_pressure_fixture(self):
        self.assert_fixture_classification("002_early_fpu_pressure")

    def test_early_child_value_override_fixture(self):
        self.assert_fixture_classification("002_early_child_value_override")

    def test_q_margin_equal_to_threshold_prefers_meaningful_q_branch(self):
        self.assert_fixture_classification("002_q_margin_boundary_prefers_meaningful_q")

    def test_single_snapshot_uses_simulation_horizon_for_early_classification(self):
        self.assert_fixture_classification("002_single_snapshot_early_fpu_pressure")

    def test_build_lane_002_uses_root_start_as_earliest_overtake_point(self):
        payload = self.valid_source_artifact()
        row = payload["rows"]["capture_available-002"]
        full_search = row["probe_mode_traces"]["full_search"]

        root_start_moves = {
            0: {"prior": 0.35, "q_value": 0.22, "selection_score": 0.80},
            1: {"prior": 0.15, "q_value": 0.10, "selection_score": 0.10},
            2: {"prior": 0.20, "q_value": 0.18, "selection_score": 0.75},
            3: {"prior": 0.18, "q_value": 0.11, "selection_score": 0.50},
            4: {"prior": 0.12, "q_value": 0.09, "selection_score": 0.60},
        }

        snapshot_moves = {
            0: {"prior": 0.35, "q_value": 0.22, "selection_score": 0.80},
            1: {"prior": 0.15, "q_value": 0.10, "selection_score": 0.10},
            2: {"prior": 0.20, "q_value": 0.18, "selection_score": 0.75},
            3: {"prior": 0.18, "q_value": 0.11, "selection_score": 0.50},
            4: {"prior": 0.12, "q_value": 0.09, "selection_score": 0.60},
        }

        for move_entry in full_search["root_start"]["moves"]:
            move_entry.update(root_start_moves[move_entry["move"]])

        for snapshot in full_search["snapshots"]:
            for move_entry in snapshot["moves"]:
                move_entry.update(snapshot_moves[move_entry["move"]])

        row["root_start"] = copy.deepcopy(full_search["root_start"])
        row["snapshots"] = copy.deepcopy(full_search["snapshots"])

        artifact = self.load_source_artifact(payload)

        lane = module.build_lane_002(artifact)

        self.assertEqual(1.0, lane["derived_metrics"]["first_selected_q_overtake_snapshot"])
        self.assertEqual(1.0, lane["derived_metrics"]["first_selected_selection_score_overtake_snapshot"])

    def test_backup_accumulation_drift_fixture(self):
        self.assert_fixture_classification("002_backup_accumulation_drift")

    def test_prior_pressure_with_small_q_fixture(self):
        self.assert_fixture_classification("002_prior_pressure_with_small_q")

    def test_unresolved_fixture(self):
        self.assert_fixture_classification("002_unresolved")


class Capture002003RowSplitFollowupLane003ClassificationTest(Capture002003RowSplitFollowupSourceArtifactTest):
    FIXTURE_DIR = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "diagnostics"
        / "capture_002_003_row_split_followup"
    )

    def load_fixture(self, name: str) -> dict:
        return json.loads((self.FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))

    def assert_fixture_classification(self, name: str) -> None:
        fixture = self.load_fixture(name)

        classification = module.classify_lane_003(
            lane=fixture["lane"],
            thresholds=fixture.get("thresholds", module.THRESHOLDS),
        )

        self.assertEqual(fixture["expected_classification"], classification)
        self.assertEqual(fixture["expected_classification"]["classification"], classification["classification"])
        self.assertEqual(fixture["expected_classification"]["decision"], classification["decision"])

    def test_value_only_child_q_prefers_wrong_move_fixture(self):
        self.assert_fixture_classification("003_value_only_child_q_prefers_wrong_move")

    def test_value_only_visit_amplification_without_q_fixture(self):
        self.assert_fixture_classification("003_value_only_visit_amplification_without_q")

    def test_policy_value_conflict_fixture(self):
        self.assert_fixture_classification("003_policy_value_conflict")

    def test_policy_value_conflict_allows_visit_share_without_meaningful_q_when_policy_still_favors_reference(self):
        lane = {
            "row_id": "capture_available-003",
            "missing_fields": [],
            "derived_metrics": {
                "reference_move": 2,
                "value_only_selected_move": 1,
                "full_search_selected_move": 1,
                "policy_only_selected_move": 2,
                "value_only_selected_minus_reference_q_margin": 0.01,
                "value_only_selected_minus_reference_visit_share": 0.06,
                "full_search_selected_minus_reference_q_margin": 0.01,
                "full_search_selected_minus_reference_visit_share": 0.06,
                "policy_reference_prior": 0.46,
                "policy_value_selected_prior": 0.18,
                "rule_features": {
                    "player_empty_pits": [],
                    "opponent_empty_pits": [],
                    "reference_pit_stones": 5,
                },
                "value_only_first_selected_visit_overtake_snapshot": 18.0,
                "value_only_first_selected_q_overtake_snapshot": 6.0,
                "value_only_first_selected_selection_score_overtake_snapshot": 12.0,
                "missing_fields": [],
            },
        }

        classification = module.classify_lane_003(lane=lane, thresholds=module.THRESHOLDS)

        self.assertEqual("policy_value_conflict", classification["classification"])
        self.assertEqual("write_003_policy_value_conflict_spec", classification["decision"])

    def test_rule_feature_value_collision_fixture(self):
        self.assert_fixture_classification("003_rule_feature_value_collision")

    def test_reference_pit_stones_zero_alone_does_not_force_rule_feature_collision(self):
        lane = {
            "row_id": "capture_available-003",
            "missing_fields": [],
            "derived_metrics": {
                "reference_move": 2,
                "value_only_selected_move": 1,
                "full_search_selected_move": 1,
                "policy_only_selected_move": 1,
                "value_only_selected_minus_reference_q_margin": 0.05,
                "value_only_selected_minus_reference_visit_share": 0.01,
                "full_search_selected_minus_reference_q_margin": 0.05,
                "full_search_selected_minus_reference_visit_share": 0.04,
                "policy_reference_prior": 0.41,
                "policy_value_selected_prior": 0.27,
                "rule_features": {
                    "player_empty_pits": [2],
                    "opponent_empty_pits": [5],
                    "reference_pit_stones": 0,
                },
                "value_only_first_selected_visit_overtake_snapshot": 20.0,
                "value_only_first_selected_q_overtake_snapshot": 8.0,
                "value_only_first_selected_selection_score_overtake_snapshot": 10.0,
                "missing_fields": [],
            },
        }

        classification = module.classify_lane_003(lane=lane, thresholds=module.THRESHOLDS)

        self.assertEqual("unresolved", classification["classification"])
        self.assertEqual("stop_003_unresolved", classification["decision"])

    def test_insufficient_value_trace_fixture(self):
        self.assert_fixture_classification("003_insufficient_value_trace")

    def test_unresolved_fixture(self):
        self.assert_fixture_classification("003_unresolved")


class Capture002003RowSplitFollowupDecisionTest(unittest.TestCase):
    def test_returns_parallel_when_both_lane_decisions_are_actionable(self):
        decision = module.top_level_decision(
            lane_002={"decision": "write_002_selection_score_trace_spec"},
            lane_003={
                "decision": "write_003_policy_value_conflict_spec",
                "classification": "policy_value_conflict",
                "missing_fields": [],
            },
        )

        self.assertEqual("write_parallel_row_followup_specs", decision)

    def test_returns_single_actionable_lane_when_other_lane_stops(self):
        decision = module.top_level_decision(
            lane_002={"decision": "stop_002_unresolved"},
            lane_003={
                "decision": "write_003_rule_value_collision_spec",
                "classification": "rule_feature_value_collision",
                "missing_fields": [],
            },
        )

        self.assertEqual("write_003_rule_value_collision_spec", decision)

    def test_returns_lane_002_decision_when_lane_002_is_actionable_and_lane_003_stops(self):
        decision = module.top_level_decision(
            lane_002={"decision": "write_002_child_value_audit_spec"},
            lane_003={
                "decision": "stop_003_unresolved",
                "classification": "unresolved",
                "missing_fields": [],
            },
        )

        self.assertEqual("write_002_child_value_audit_spec", decision)

    def test_prefers_value_trace_capture_when_missing_trace_blocks_lane_003_interpretation(self):
        decision = module.top_level_decision(
            lane_002={"decision": "write_002_backup_accumulation_spec"},
            lane_003={
                "decision": "write_003_value_trace_capture_spec",
                "classification": "insufficient_value_trace",
                "missing_fields": ["value_only.root_start"],
            },
        )

        self.assertEqual("write_003_value_trace_capture_spec", decision)

    def test_prefers_value_trace_capture_for_upstream_empty_value_only_trace_without_missing_fields(self):
        decision = module.top_level_decision(
            lane_002={"decision": "write_002_backup_accumulation_spec"},
            lane_003={
                "decision": "write_003_value_trace_capture_spec",
                "classification": "insufficient_value_trace",
                "missing_fields": [],
                "derived_metrics": {
                    "missing_fields": [],
                    "value_only_selected_minus_reference_q_margin": None,
                    "value_only_selected_minus_reference_visit_share": None,
                },
            },
        )

        self.assertEqual("write_003_value_trace_capture_spec", decision)

    def test_returns_stop_when_both_lanes_stop(self):
        decision = module.top_level_decision(
            lane_002={"decision": "stop_002_unresolved"},
            lane_003={
                "decision": "stop_003_unresolved",
                "classification": "unresolved",
                "missing_fields": [],
            },
        )

        self.assertEqual("stop_row_split_unresolved", decision)


class Capture002003RowSplitFollowupPayloadCliTest(Capture002003RowSplitFollowupSourceArtifactTest):
    def build_classified_lanes(self) -> tuple[dict, dict, dict, str]:
        artifact = self.load_source_artifact(self.valid_source_artifact())
        lane_002 = module.build_lane_002(artifact)
        lane_002.update(module.classify_lane_002(lane=lane_002, thresholds=module.THRESHOLDS))
        lane_003 = module.build_lane_003(artifact)
        lane_003.update(module.classify_lane_003(lane=lane_003, thresholds=module.THRESHOLDS))
        decision = module.top_level_decision(lane_002=lane_002, lane_003=lane_003)
        return artifact, lane_002, lane_003, decision

    def test_build_payload_assembles_required_top_level_and_summary_fields(self):
        artifact, lane_002, lane_003, decision = self.build_classified_lanes()

        payload = module.build_payload(
            source_artifact=artifact,
            lane_002=lane_002,
            lane_003=lane_003,
            decision=decision,
        )

        self.assertEqual(module.SCHEMA, payload["schema"])
        self.assertEqual(artifact["artifact_path"], payload["source_shared_drift_artifact"])
        self.assertEqual(artifact["selected_artifact"], payload["selected_artifact"])
        self.assertEqual(module.THRESHOLDS, payload["thresholds"])
        self.assertEqual(artifact["settings"], payload["settings"])
        self.assertEqual(decision, payload["decision"])
        self.assertEqual({"capture_available-002", "capture_available-003"}, set(payload["lanes"].keys()))
        self.assertEqual("unresolved", payload["lanes"]["capture_available-002"]["classification"])
        self.assertEqual(
            "stop_002_unresolved",
            payload["lanes"]["capture_available-002"]["decision"],
        )
        self.assertEqual(
            "Lane 002 evidence does not isolate one supported mechanism.",
            payload["lanes"]["capture_available-002"]["evidence_summary"],
        )
        self.assertEqual(
            "value_only_visit_amplification_without_q",
            payload["lanes"]["capture_available-003"]["classification"],
        )
        self.assertEqual(
            "write_003_value_only_visit_trace_spec",
            payload["lanes"]["capture_available-003"]["decision"],
        )
        self.assertEqual(module.ROW_IDS, payload["summary"]["row_ids"])
        self.assertFalse(payload["summary"]["shared_mechanism_supported"])
        self.assertEqual(
            {
                "capture_available-002": "unresolved",
                "capture_available-003": "value_only_visit_amplification_without_q",
            },
            payload["summary"]["lane_classifications"],
        )
        self.assertEqual(
            {
                "capture_available-002": "stop_002_unresolved",
                "capture_available-003": "write_003_value_only_visit_trace_spec",
            },
            payload["summary"]["lane_decisions"],
        )
        self.assertEqual(decision, payload["summary"]["next_safe_branch"])
        self.assertEqual(
            "capture_available-002: Lane 002 evidence does not isolate one supported mechanism. capture_available-003: Value-only search amplifies visits toward the wrong move without a meaningful Q advantage over the reference move.",
            payload["summary"]["evidence_summary"],
        )

    def test_main_writes_payload_and_prints_compact_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.json"
            out_path = Path(tmp) / "out" / "payload.json"
            self.write_json(source_path, self.valid_source_artifact())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = module.main(
                    [
                        "--source-shared-drift-artifact",
                        str(source_path),
                        "--out",
                        str(out_path),
                    ]
                )

            self.assertEqual(0, exit_code)
            written_payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(module.SCHEMA, written_payload["schema"])
            self.assertEqual("write_003_value_only_visit_trace_spec", written_payload["decision"])
            self.assertEqual(str(source_path), written_payload["source_shared_drift_artifact"])
            self.assertEqual(
                {
                    "artifact_path": str(out_path),
                    "schema": module.SCHEMA,
                    "decision": "write_003_value_only_visit_trace_spec",
                },
                json.loads(stdout.getvalue()),
            )

    def test_main_writes_sorted_json_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.json"
            out_path = Path(tmp) / "out" / "payload.json"
            self.write_json(source_path, self.valid_source_artifact())

            module.main(
                [
                    "--source-shared-drift-artifact",
                    str(source_path),
                    "--out",
                    str(out_path),
                ]
            )

            payload = json.loads(out_path.read_text(encoding="utf-8"))

            self.assertEqual(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                out_path.read_text(encoding="utf-8"),
            )
    
