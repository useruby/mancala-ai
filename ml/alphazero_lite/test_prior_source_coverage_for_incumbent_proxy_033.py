import json
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite import prior_source_coverage_for_incumbent_proxy_033 as module


class PriorSourceCoverageForIncumbentProxy033Test(unittest.TestCase):
    def write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def minimal_artifacts(self) -> dict:
        return {
            "artifact_pinned_ablation": {
                "path": "/tmp/artifact_pinned_value_trust_ablation.json",
                "payload": {
                    "schema": "azlite_artifact_pinned_value_trust_ablation_v1",
                    "artifact_path": "/artifacts/current",
                    "probe_settings": {"simulations": 384, "c_puct": 1.25, "seed": 42},
                    "rows": {
                        "incumbent_proxy_disagreement-033": {
                            "reference_move": 4,
                            "configs": {
                                "full_default": {
                                    "selected_move": 0,
                                    "passes_reference": False,
                                }
                            },
                        },
                        "capture_available-002": {
                            "reference_move": 2,
                            "configs": {
                                "full_default": {
                                    "selected_move": 2,
                                    "passes_reference": True,
                                }
                            },
                        },
                        "capture_available-003": {
                            "reference_move": 2,
                            "configs": {
                                "full_default": {
                                    "selected_move": 2,
                                    "passes_reference": True,
                                }
                            },
                        },
                        "incumbent_proxy_disagreement-031": {
                            "reference_move": 4,
                            "configs": {
                                "full_default": {
                                    "selected_move": 4,
                                    "passes_reference": True,
                                }
                            },
                        },
                    },
                },
            },
            "candidate_search_interaction": {
                "path": "/tmp/candidate_search_interaction.json",
                "payload": {
                    "schema": "azlite_search_interaction_diagnostic_v1",
                    "rows": {},
                },
            },
            "rebalance_search_interaction": {
                "path": "/tmp/rebalance_search_interaction.json",
                "payload": {
                    "schema": "azlite_search_interaction_diagnostic_v1",
                    "rows": {},
                },
            },
            "replay_source": {"path": "/tmp/replay_source.jsonl", "memberships": {}},
        }

    def test_build_payload_emits_required_schema_and_row_roles(self):
        payload = module.build_payload(source_artifacts=self.minimal_artifacts())

        self.assertEqual(
            "azlite_prior_source_coverage_for_incumbent_proxy_033_v1", payload["schema"]
        )
        self.assertEqual("incumbent_proxy_disagreement-033", payload["target_row_id"])
        self.assertEqual(
            ["capture_available-002", "capture_available-003"], payload["guard_row_ids"]
        )
        self.assertEqual(
            ["incumbent_proxy_disagreement-031"], payload["context_row_ids"]
        )
        self.assertEqual(
            "target", payload["rows"]["incumbent_proxy_disagreement-033"]["row_role"]
        )
        self.assertEqual("guard", payload["rows"]["capture_available-002"]["row_role"])
        self.assertEqual("guard", payload["rows"]["capture_available-003"]["row_role"])
        self.assertEqual(
            "context", payload["rows"]["incumbent_proxy_disagreement-031"]["row_role"]
        )

    def test_decision_writes_candidate_spec_for_absent_target_when_guards_are_clean(
        self,
    ):
        artifacts = self.minimal_artifacts()
        artifacts["replay_source"]["memberships"] = {
            "incumbent_proxy_disagreement-033": {"present": False, "count": 0},
            "capture_available-002": {"present": True, "count": 2},
            "capture_available-003": {"present": True, "count": 2},
            "incumbent_proxy_disagreement-031": {"present": True, "count": 1},
        }

        payload = module.build_payload(source_artifacts=artifacts)

        self.assertEqual(
            "absent_from_source_construction",
            payload["summary"]["source_gap_classification"],
        )
        self.assertEqual(
            "write_targeted_source_coverage_candidate_spec",
            payload["summary"]["decision"],
        )
        self.assertTrue(payload["summary"]["future_candidate_justified"])

    def test_decision_writes_candidate_spec_for_misclassified_target_when_guards_are_clean(
        self,
    ):
        artifacts = self.minimal_artifacts()
        artifacts["replay_source"]["memberships"] = {
            "incumbent_proxy_disagreement-033": {
                "present": True,
                "count": 1,
                "classification": "misclassified",
            },
            "capture_available-002": {"present": True, "count": 2},
            "capture_available-003": {"present": True, "count": 2},
        }

        payload = module.build_payload(source_artifacts=artifacts)

        self.assertEqual(
            "misclassified_in_source_construction",
            payload["summary"]["source_gap_classification"],
        )
        self.assertEqual(
            "write_targeted_source_coverage_candidate_spec",
            payload["summary"]["decision"],
        )

    def test_underrepresentation_is_diagnostic_only_without_missing_or_misclassified_signal(
        self,
    ):
        artifacts = self.minimal_artifacts()
        artifacts["replay_source"]["memberships"] = {
            "incumbent_proxy_disagreement-033": {"present": True, "count": 1},
            "capture_available-002": {"present": True, "count": 4},
            "capture_available-003": {"present": True, "count": 4},
        }

        payload = module.build_payload(source_artifacts=artifacts)

        self.assertEqual(
            "underrepresented_in_source_construction",
            payload["summary"]["source_gap_classification"],
        )
        self.assertEqual(
            "close_033_source_coverage_variant", payload["summary"]["decision"]
        )
        self.assertFalse(payload["summary"]["future_candidate_justified"])

    def test_reference_or_teacher_drift_is_diagnostic_only_without_missing_or_misclassified_signal(
        self,
    ):
        artifacts = self.minimal_artifacts()
        artifacts["candidate_search_interaction"]["payload"]["rows"] = {
            "incumbent_proxy_disagreement-033": {
                "reference_move": 4,
                "teacher_value": 0.65,
            },
        }
        artifacts["rebalance_search_interaction"]["payload"]["rows"] = {
            "incumbent_proxy_disagreement-033": {
                "reference_move": 3,
                "teacher_value": 0.65,
            },
        }

        payload = module.build_payload(source_artifacts=artifacts)

        self.assertEqual(
            "reference_or_teacher_drift_detected",
            payload["summary"]["source_gap_classification"],
        )
        self.assertEqual(
            "close_033_source_coverage_variant", payload["summary"]["decision"]
        )

    def test_guard_reference_instability_blocks_candidate_spec(self):
        artifacts = self.minimal_artifacts()
        artifacts["replay_source"]["memberships"] = {
            "incumbent_proxy_disagreement-033": {"present": False, "count": 0},
            "capture_available-002": {"present": True, "count": 2},
            "capture_available-003": {"present": True, "count": 2},
        }
        artifacts["candidate_search_interaction"]["payload"]["rows"] = {
            "capture_available-002": {"reference_move": 2, "teacher_value": 0.5},
        }
        artifacts["rebalance_search_interaction"]["payload"]["rows"] = {
            "capture_available-002": {"reference_move": 1, "teacher_value": 0.5},
        }

        payload = module.build_payload(source_artifacts=artifacts)

        self.assertFalse(payload["summary"]["guards_clean"])
        self.assertEqual(
            "close_033_source_coverage_variant", payload["summary"]["decision"]
        )

    def test_missing_guard_replay_membership_blocks_candidate_spec(self):
        artifacts = self.minimal_artifacts()
        artifacts["replay_source"]["memberships"] = {
            "incumbent_proxy_disagreement-033": {"present": False, "count": 0},
            "capture_available-002": {"present": False, "count": 0},
            "capture_available-003": {"present": True, "count": 2},
            "incumbent_proxy_disagreement-031": {"present": True, "count": 1},
        }

        payload = module.build_payload(source_artifacts=artifacts)

        self.assertFalse(payload["summary"]["guards_clean"])
        self.assertEqual(
            "close_033_source_coverage_variant", payload["summary"]["decision"]
        )

    def test_misclassified_guard_replay_membership_blocks_candidate_spec(self):
        artifacts = self.minimal_artifacts()
        artifacts["replay_source"]["memberships"] = {
            "incumbent_proxy_disagreement-033": {"present": False, "count": 0},
            "capture_available-002": {
                "present": True,
                "count": 2,
                "classification": "misclassified",
            },
            "capture_available-003": {"present": True, "count": 2},
            "incumbent_proxy_disagreement-031": {"present": True, "count": 1},
        }

        payload = module.build_payload(source_artifacts=artifacts)

        self.assertFalse(payload["summary"]["guards_clean"])
        self.assertEqual(
            "close_033_source_coverage_variant", payload["summary"]["decision"]
        )

    def test_validate_schema_allows_positional_arguments_and_raises_on_mismatch(self):
        with self.assertRaisesRegex(ValueError, "unexpected schema"):
            module.validate_schema({"schema": "wrong"}, "expected", Path("source.json"))

    def test_replay_memberships_ignores_substring_mentions_in_unrelated_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay_source.jsonl"
            replay_path.write_text(
                '{"note":"incumbent_proxy_disagreement-033"}\n', encoding="utf-8"
            )

            memberships = module.replay_memberships(replay_path)

            self.assertEqual(
                {"present": False, "count": 0, "matched_by": [], "line_numbers": []},
                memberships["incumbent_proxy_disagreement-033"],
            )

    def test_replay_memberships_matches_canonical_states_without_row_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay_source.jsonl"
            replay_path.write_text(
                "\n".join(
                    [
                        json.dumps({"canonical_state": "guard-002-state"}),
                        json.dumps({"canonical_state": "guard-003-state"}),
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            memberships = module.replay_memberships(
                replay_path,
                canonical_states={
                    "incumbent_proxy_disagreement-033": "target-state",
                    "capture_available-002": "guard-002-state",
                    "capture_available-003": "guard-003-state",
                },
            )

            self.assertEqual(
                {
                    "present": True,
                    "count": 1,
                    "matched_by": ["canonical_state"],
                    "line_numbers": [1],
                },
                memberships["capture_available-002"],
            )
            self.assertEqual(
                {
                    "present": True,
                    "count": 1,
                    "matched_by": ["canonical_state"],
                    "line_numbers": [2],
                },
                memberships["capture_available-003"],
            )
            self.assertEqual(
                {"present": False, "count": 0, "matched_by": [], "line_numbers": []},
                memberships["incumbent_proxy_disagreement-033"],
            )

    def test_canonical_state_guard_presence_keeps_target_absence_actionable(self):
        artifacts = self.minimal_artifacts()
        artifacts["replay_source"]["memberships"] = {
            "incumbent_proxy_disagreement-033": {
                "present": False,
                "count": 0,
                "matched_by": [],
                "line_numbers": [],
            },
            "capture_available-002": {
                "present": True,
                "count": 1,
                "matched_by": ["canonical_state"],
                "line_numbers": [10],
            },
            "capture_available-003": {
                "present": True,
                "count": 1,
                "matched_by": ["canonical_state"],
                "line_numbers": [11],
            },
            "incumbent_proxy_disagreement-031": {
                "present": False,
                "count": 0,
                "matched_by": [],
                "line_numbers": [],
            },
        }
        artifacts["canonical_state_provenance"] = {
            "states": {
                "incumbent_proxy_disagreement-033": "target-state",
                "capture_available-002": "guard-002-state",
                "capture_available-003": "guard-003-state",
            },
            "paths": ["/tmp/final/selected_candidate_forensics.json"],
            "missing_row_ids": ["incumbent_proxy_disagreement-031"],
        }

        payload = module.build_payload(source_artifacts=artifacts)

        self.assertEqual(
            "write_targeted_source_coverage_candidate_spec",
            payload["summary"]["decision"],
        )
        self.assertEqual(
            "absent_from_source_construction",
            payload["summary"]["source_gap_classification"],
        )
        self.assertTrue(payload["summary"]["guards_clean"])
        self.assertEqual(
            "guard-002-state",
            payload["rows"]["capture_available-002"]["canonical_state"],
        )
        self.assertEqual(
            {
                "present": True,
                "count": 1,
                "matched_by": ["canonical_state"],
                "line_numbers": [10],
            },
            payload["rows"]["capture_available-002"]["replay_source_membership"],
        )

    def test_wrong_bucket_replay_match_marks_target_misclassified(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay_source.jsonl"
            replay_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "canonical_state": "target-state",
                                "bucket": "capture_available",
                            }
                        ),
                        json.dumps(
                            {
                                "canonical_state": "guard-002-state",
                                "bucket": "capture_available",
                            }
                        ),
                        json.dumps(
                            {
                                "canonical_state": "guard-003-state",
                                "bucket": "capture_available",
                            }
                        ),
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            artifacts = self.minimal_artifacts()
            artifacts["replay_source"]["memberships"] = module.replay_memberships(
                replay_path,
                canonical_states={
                    "incumbent_proxy_disagreement-033": "target-state",
                    "capture_available-002": "guard-002-state",
                    "capture_available-003": "guard-003-state",
                },
            )

            payload = module.build_payload(source_artifacts=artifacts)

            target_membership = payload["rows"]["incumbent_proxy_disagreement-033"][
                "replay_source_membership"
            ]
            self.assertEqual("misclassified", target_membership["classification"])
            self.assertEqual(
                [
                    {
                        "expected_bucket": "incumbent_proxy_disagreement",
                        "actual_bucket": "capture_available",
                        "line_number": 1,
                        "matched_by": "canonical_state",
                    }
                ],
                target_membership["classification_evidence"],
            )
            self.assertEqual(
                "misclassified_in_source_construction",
                payload["summary"]["source_gap_classification"],
            )
            self.assertEqual(
                "write_targeted_source_coverage_candidate_spec",
                payload["summary"]["decision"],
            )

    def test_canonical_state_conflict_is_recorded_per_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate_run = tmp_path / "candidate"
            rebalance_run = tmp_path / "rebalance"
            self.write_json(
                candidate_run / "final" / "selected_candidate_forensics.json",
                {
                    "systems": {
                        "current": {
                            "rows": [
                                {
                                    "id": "incumbent_proxy_disagreement-033",
                                    "canonical_state": "candidate-state",
                                }
                            ],
                        },
                    },
                },
            )
            self.write_json(
                rebalance_run / "final" / "selected_candidate_forensics.json",
                {
                    "systems": {
                        "current": {
                            "rows": [
                                {
                                    "id": "incumbent_proxy_disagreement-033",
                                    "canonical_state": "rebalance-state",
                                }
                            ],
                        },
                    },
                },
            )
            artifacts = self.minimal_artifacts()
            artifacts["canonical_state_provenance"] = (
                module.load_canonical_state_provenance(
                    candidate_payload={"rebalanced_run_dir": str(candidate_run)},
                    rebalance_payload={"rebalanced_run_dir": str(rebalance_run)},
                )
            )

            payload = module.build_payload(source_artifacts=artifacts)

            row = payload["rows"]["incumbent_proxy_disagreement-033"]
            self.assertEqual("candidate-state", row["canonical_state"])
            self.assertEqual(
                "candidate-state",
                row["canonical_state_provenance"]["candidate_canonical_state"],
            )
            self.assertEqual(
                "rebalance-state",
                row["canonical_state_provenance"]["rebalance_canonical_state"],
            )
            self.assertTrue(row["canonical_state_provenance"]["conflict"])
            self.assertIn(
                "canonical state conflict between candidate and rebalance forensics",
                row["notes"],
            )

    def test_replay_memberships_matches_both_conflicting_canonical_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay_source.jsonl"
            replay_path.write_text(
                "\n".join(
                    [
                        json.dumps({"canonical_state": "candidate-state"}),
                        json.dumps({"canonical_state": "rebalance-state"}),
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            memberships = module.replay_memberships(
                replay_path,
                canonical_states={
                    "incumbent_proxy_disagreement-033": [
                        "candidate-state",
                        "rebalance-state",
                    ],
                },
            )

            self.assertEqual(
                {
                    "present": True,
                    "count": 2,
                    "matched_by": ["canonical_state"],
                    "line_numbers": [1, 2],
                },
                memberships["incumbent_proxy_disagreement-033"],
            )

    def test_replay_memberships_raises_when_source_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                module.replay_memberships(Path(tmp) / "missing.jsonl")


class PriorSourceCoverageForIncumbentProxy033CliTest(unittest.TestCase):
    def write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_main_writes_diagnostic_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pinned_path = tmp_path / "artifact_pinned_value_trust_ablation.json"
            candidate_path = tmp_path / "candidate_search_interaction.json"
            rebalance_path = tmp_path / "rebalance_search_interaction.json"
            replay_path = tmp_path / "replay_source.jsonl"
            out_path = (
                tmp_path
                / "final"
                / "prior_source_coverage_for_incumbent_proxy_033.json"
            )

            self.write_json(
                pinned_path,
                {
                    "schema": "azlite_artifact_pinned_value_trust_ablation_v1",
                    "probe_settings": {"simulations": 384, "c_puct": 1.25, "seed": 42},
                    "rows": {
                        "incumbent_proxy_disagreement-033": {
                            "reference_move": 4,
                            "configs": {
                                "full_default": {
                                    "selected_move": 0,
                                    "passes_reference": False,
                                }
                            },
                        },
                        "capture_available-002": {
                            "reference_move": 2,
                            "configs": {
                                "full_default": {
                                    "selected_move": 2,
                                    "passes_reference": True,
                                }
                            },
                        },
                        "capture_available-003": {
                            "reference_move": 2,
                            "configs": {
                                "full_default": {
                                    "selected_move": 2,
                                    "passes_reference": True,
                                }
                            },
                        },
                        "incumbent_proxy_disagreement-031": {
                            "reference_move": 4,
                            "configs": {
                                "full_default": {
                                    "selected_move": 4,
                                    "passes_reference": True,
                                }
                            },
                        },
                    },
                },
            )
            self.write_json(
                candidate_path,
                {"schema": "azlite_search_interaction_diagnostic_v1", "rows": {}},
            )
            self.write_json(
                rebalance_path,
                {"schema": "azlite_search_interaction_diagnostic_v1", "rows": {}},
            )
            replay_path.write_text(
                '{"row_id":"capture_available-002"}\n{"row_id":"capture_available-003"}\n',
                encoding="utf-8",
            )

            exit_code = module.main(
                [
                    "--artifact-pinned-ablation",
                    str(pinned_path),
                    "--candidate-search-interaction",
                    str(candidate_path),
                    "--rebalance-search-interaction",
                    str(rebalance_path),
                    "--replay-source",
                    str(replay_path),
                    "--out",
                    str(out_path),
                ]
            )

            self.assertEqual(0, exit_code)
            written = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(
                "azlite_prior_source_coverage_for_incumbent_proxy_033_v1",
                written["schema"],
            )
            self.assertEqual(
                str(pinned_path),
                written["source_artifacts"]["artifact_pinned_ablation"],
            )
            self.assertEqual(
                {"present": False, "count": 0, "matched_by": [], "line_numbers": []},
                written["rows"]["incumbent_proxy_disagreement-033"][
                    "replay_source_membership"
                ],
            )
            self.assertEqual(
                {
                    "present": True,
                    "count": 1,
                    "matched_by": ["row_id"],
                    "line_numbers": [1],
                },
                written["rows"]["capture_available-002"]["replay_source_membership"],
            )
            self.assertEqual(
                {
                    "present": True,
                    "count": 1,
                    "matched_by": ["row_id"],
                    "line_numbers": [2],
                },
                written["rows"]["capture_available-003"]["replay_source_membership"],
            )
            self.assertEqual(
                "absent_from_source_construction",
                written["summary"]["source_gap_classification"],
            )
            self.assertEqual(
                "write_targeted_source_coverage_candidate_spec",
                written["summary"]["decision"],
            )
