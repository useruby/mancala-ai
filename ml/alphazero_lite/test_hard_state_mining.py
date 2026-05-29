import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.forensic_suite import canonical_state_key


PYTHON_BIN = sys.executable


class HardStateMiningTest(unittest.TestCase):
    def make_candidate(self, **overrides):
        candidate = {
            "state": {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            },
            "side_to_move": 0,
            "legal_moves": [0, 1, 2, 3, 4, 5],
            "selection_reason": "large_value_error",
            "source_artifact": "report.json",
            "source_run": {"kind": "forensic"},
            "consequence": "promotion_risk",
            "metrics": {"value_error": 0.7},
        }
        candidate.update(overrides)
        return candidate

    def sample_state(self, **overrides):
        state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        state.update(overrides)
        return state

    def test_normalize_candidate_requires_core_fields(self):
        from ml.alphazero_lite import hard_state_mining

        with self.assertRaisesRegex(ValueError, "candidate is missing required field"):
            hard_state_mining.normalize_candidate({"canonical_state": "x"})

    def test_normalize_candidate_returns_canonical_state_copy(self):
        from ml.alphazero_lite import hard_state_mining

        candidate = self.make_candidate(
            state={
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [1, 2, 3, 4, 5, 6],
                "player_store": 7,
                "opponent_store": 8,
                "current_player": 0,
            }
        )

        normalized = hard_state_mining.normalize_candidate(candidate)

        expected_state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [1, 2, 3, 4, 5, 6],
            "player_store": 7,
            "opponent_store": 8,
            "current_player": 0,
        }
        self.assertEqual(expected_state, normalized["state"])
        self.assertEqual(
            canonical_state_key(expected_state), normalized["canonical_state"]
        )

    def test_normalize_candidate_rejects_malformed_state(self):
        from ml.alphazero_lite import hard_state_mining

        with self.assertRaisesRegex(ValueError, "candidate state must be a dictionary"):
            hard_state_mining.normalize_candidate(self.make_candidate(state=[]))

        malformed_state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
        }
        with self.assertRaisesRegex(
            ValueError, "candidate state is missing required field: current_player"
        ):
            hard_state_mining.normalize_candidate(
                self.make_candidate(state=malformed_state)
            )

    def test_normalize_candidate_state_is_independent_from_input(self):
        from ml.alphazero_lite import hard_state_mining

        candidate = self.make_candidate()
        normalized = hard_state_mining.normalize_candidate(candidate)

        candidate["state"]["player_pits"][0] = 99
        candidate["state"]["opponent_pits"][1] = 88

        self.assertEqual([4, 4, 4, 4, 4, 4], normalized["state"]["player_pits"])
        self.assertEqual([4, 4, 4, 4, 4, 4], normalized["state"]["opponent_pits"])

    def test_deduplicate_candidates_merges_reasons_sources_and_metadata(self):
        from ml.alphazero_lite import hard_state_mining

        first = self.make_candidate(
            selection_reason="large_value_error",
            source_artifact="report-a.json",
            source_run={"kind": "forensic", "run": "a"},
            metrics={"value_error": 0.7, "entropy": 0.1},
        )
        second = self.make_candidate(
            state={
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            },
            selection_reason="student_teacher_disagreement",
            source_artifact="report-b.json",
            source_run={"kind": "arena", "run": "b"},
            metrics={"regret": 0.4, "entropy": 0.3, "best_second_gap": 0.2},
        )

        deduplicated = hard_state_mining.deduplicate_candidates([first, second])

        self.assertEqual(1, len(deduplicated))
        self.assertEqual(
            canonical_state_key(
                {
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                }
            ),
            deduplicated[0]["canonical_state"],
        )
        self.assertEqual(
            ["large_value_error", "student_teacher_disagreement"],
            deduplicated[0]["selection_reasons"],
        )
        self.assertEqual(
            ["report-a.json", "report-b.json"],
            deduplicated[0]["source_artifacts"],
        )
        self.assertEqual(
            [
                {"kind": "forensic", "run": "a"},
                {"kind": "arena", "run": "b"},
            ],
            deduplicated[0]["source_runs"],
        )
        self.assertEqual(
            {
                "max_regret": 0.4,
                "max_value_error": 0.7,
                "max_entropy": 0.3,
                "max_best_second_gap": 0.2,
            },
            deduplicated[0]["metadata"],
        )
        self.assertNotIn("metrics", deduplicated[0])

    def test_deduplicate_candidates_suppresses_duplicate_merged_values(self):
        from ml.alphazero_lite import hard_state_mining

        first = self.make_candidate(
            selection_reason="large_value_error",
            source_artifact="report-a.json",
            source_run={"kind": "forensic", "run": "a"},
        )
        second = self.make_candidate(
            state={
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            },
            selection_reason="large_value_error",
            source_artifact="report-a.json",
            source_run={"kind": "forensic", "run": "a"},
        )

        deduplicated = hard_state_mining.deduplicate_candidates([first, second])

        self.assertEqual(1, len(deduplicated))
        self.assertEqual(["large_value_error"], deduplicated[0]["selection_reasons"])
        self.assertEqual(["report-a.json"], deduplicated[0]["source_artifacts"])
        self.assertEqual(
            [{"kind": "forensic", "run": "a"}],
            deduplicated[0]["source_runs"],
        )

    def test_deduplicate_candidates_omits_stale_singular_provenance_fields(self):
        from ml.alphazero_lite import hard_state_mining

        deduplicated = hard_state_mining.deduplicate_candidates(
            [
                self.make_candidate(
                    selection_reason="large_value_error",
                    source_artifact="report-a.json",
                    source_run={"kind": "forensic", "run": "a"},
                ),
                self.make_candidate(
                    state={
                        "player_pits": [4, 4, 4, 4, 4, 4],
                        "opponent_pits": [4, 4, 4, 4, 4, 4],
                        "player_store": 0,
                        "opponent_store": 0,
                        "current_player": 0,
                    },
                    selection_reason="policy_instability",
                    source_artifact="report-b.json",
                    source_run={"kind": "self_play", "run": "b"},
                ),
            ]
        )

        self.assertEqual(1, len(deduplicated))
        self.assertNotIn("selection_reason", deduplicated[0])
        self.assertNotIn("source_artifact", deduplicated[0])
        self.assertNotIn("source_run", deduplicated[0])

    def test_deduplicate_candidates_merges_strongest_metrics_evidence(self):
        from ml.alphazero_lite import hard_state_mining

        first = self.make_candidate(metrics={"value_error": 0.7, "regret": 0.1})
        second = self.make_candidate(
            state={
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            },
            metrics={"value_error": 0.9, "regret": 0.6, "best_second_gap": 0.4},
        )

        deduplicated = hard_state_mining.deduplicate_candidates([first, second])

        self.assertEqual(1, len(deduplicated))
        self.assertEqual(
            {
                "max_regret": 0.6,
                "max_value_error": 0.9,
                "max_entropy": 0.0,
                "max_best_second_gap": 0.4,
            },
            deduplicated[0]["metadata"],
        )

    def test_score_rows_prefers_promotion_risk_over_duplicate_count(self):
        from ml.alphazero_lite import hard_state_mining

        rows = [
            {
                "canonical_state": "critical",
                "selection_reasons": ["challenger_loss_vs_current"],
                "source_artifacts": ["loss.json"],
                "source_runs": [{"kind": "arena"}],
                "consequence": "caused_loss",
                "metadata": {
                    "max_regret": 0.7,
                    "max_value_error": 0.0,
                    "max_entropy": 0.0,
                    "max_best_second_gap": 0.0,
                },
            },
            {
                "canonical_state": "frequent",
                "selection_reasons": ["high_search_entropy"],
                "source_artifacts": ["a.json", "b.json", "c.json", "d.json"],
                "source_runs": [{"kind": "forensic"}],
                "consequence": "high_entropy_decision",
                "metadata": {
                    "max_regret": 0.0,
                    "max_value_error": 0.0,
                    "max_entropy": 0.2,
                    "max_best_second_gap": 0.0,
                },
            },
        ]

        scored = hard_state_mining.score_rows(rows)

        self.assertEqual("critical", scored[0]["canonical_state"])
        self.assertGreater(scored[0]["priority_score"], scored[1]["priority_score"])
        self.assertIn("priority_score", scored[0])
        self.assertIn("priority_breakdown", scored[0])
        self.assertEqual(13.5, scored[0]["priority_score"])
        self.assertEqual(
            {
                "reason_score": 10.0,
                "regret_boost": 3.5,
                "value_error_boost": 0.0,
                "entropy_boost": 0.0,
                "gap_boost": 0.0,
                "multi_source_boost": 0.0,
            },
            scored[0]["priority_breakdown"],
        )

    def test_score_rows_uses_distinct_source_run_kinds_for_multi_source_boost(self):
        from ml.alphazero_lite import hard_state_mining

        scored = hard_state_mining.score_rows(
            [
                {
                    "canonical_state": "multi-source",
                    "selection_reasons": [
                        "large_value_error",
                        "student_teacher_disagreement",
                    ],
                    "source_artifacts": ["a.json", "b.json", "c.json"],
                    "source_runs": [
                        {"kind": "forensic", "run": "a"},
                        {"kind": "forensic", "run": "b"},
                        {"kind": "arena", "run": "c"},
                    ],
                    "consequence": "promotion_risk",
                    "metadata": {
                        "max_regret": 0.0,
                        "max_value_error": 0.0,
                        "max_entropy": 0.0,
                        "max_best_second_gap": 0.0,
                    },
                }
            ]
        )

        self.assertEqual(14.0, scored[0]["priority_score"])
        self.assertEqual(2.0, scored[0]["priority_breakdown"]["multi_source_boost"])

    def test_score_rows_breaks_ties_by_canonical_state(self):
        from ml.alphazero_lite import hard_state_mining

        scored = hard_state_mining.score_rows(
            [
                {
                    "canonical_state": "state-b",
                    "selection_reasons": ["high_search_entropy"],
                    "source_artifacts": ["b.json"],
                    "source_runs": [{"kind": "forensic"}],
                    "consequence": "high_entropy_decision",
                    "metadata": {
                        "max_regret": 0.0,
                        "max_value_error": 0.0,
                        "max_entropy": 0.0,
                        "max_best_second_gap": 0.0,
                    },
                },
                {
                    "canonical_state": "state-a",
                    "selection_reasons": ["high_search_entropy"],
                    "source_artifacts": ["a.json"],
                    "source_runs": [{"kind": "forensic"}],
                    "consequence": "high_entropy_decision",
                    "metadata": {
                        "max_regret": 0.0,
                        "max_value_error": 0.0,
                        "max_entropy": 0.0,
                        "max_best_second_gap": 0.0,
                    },
                },
            ]
        )

        self.assertEqual(
            ["state-a", "state-b"], [row["canonical_state"] for row in scored]
        )

    def test_score_rows_rejects_unknown_selection_reason(self):
        from ml.alphazero_lite import hard_state_mining

        with self.assertRaisesRegex(ValueError, "unknown selection reason"):
            hard_state_mining.score_rows(
                [
                    {
                        "canonical_state": "mystery",
                        "selection_reasons": ["policy_instability"],
                        "source_artifacts": ["mystery.json"],
                        "source_runs": [{"kind": "forensic"}],
                        "consequence": "promotion_risk",
                        "metadata": {
                            "max_regret": 0.0,
                            "max_value_error": 0.0,
                            "max_entropy": 0.0,
                            "max_best_second_gap": 0.0,
                        },
                    }
                ]
            )

    def test_normalize_and_deduplicate_outputs_are_detached_from_input_mutation(self):
        from ml.alphazero_lite import hard_state_mining

        candidate = self.make_candidate(
            legal_moves=[0, 1, 2],
            source_run={"kind": "forensic", "run": "a"},
        )

        normalized = hard_state_mining.normalize_candidate(candidate)
        deduplicated = hard_state_mining.deduplicate_candidates([candidate])

        candidate["legal_moves"].append(99)
        candidate["source_run"]["kind"] = "mutated"

        self.assertEqual([0, 1, 2], normalized["legal_moves"])
        self.assertEqual([0, 1, 2], deduplicated[0]["legal_moves"])
        self.assertEqual(
            {"kind": "forensic", "run": "a"}, deduplicated[0]["source_runs"][0]
        )

    def test_extract_candidates_covers_requested_source_classes_from_supported_report_shapes(
        self,
    ):
        from ml.alphazero_lite import hard_state_mining

        artifacts = [
            {
                "path": "reports/forensic.json",
                "schema": "azlite_forensic_suite_v1",
                "systems": {
                    "challenger": {
                        "artifact_path": "challenger.pt",
                        "rows": [
                            {
                                "id": "pos-1",
                                "bucket": "capture_available",
                                "state": self.sample_state(),
                                "side_to_move": 0,
                                "legal_moves": [0, 1],
                                "selected_move": 1,
                                "reference_move": 0,
                                "agrees_top1": False,
                                "regret": 0.4,
                                "teacher_value": 0.1,
                                "system_value": -0.2,
                                "value_error": 0.8,
                                "entropy": 0.9,
                                "best_second_gap": 0.7,
                            },
                            {
                                "id": "pos-2",
                                "bucket": "high_value_swing",
                                "state": self.sample_state(player_store=1),
                                "side_to_move": 0,
                                "legal_moves": [0, 1],
                                "selected_move": 0,
                                "reference_move": 0,
                                "agrees_top1": True,
                                "regret": 0.0,
                                "teacher_value": 0.2,
                                "system_value": -0.6,
                                "value_error": 0.8,
                                "entropy": 0.1,
                                "best_second_gap": 0.2,
                            },
                        ],
                    }
                },
            },
            {
                "path": "reports/arena_current.json",
                "kind": "arena_loss_vs_current",
                "rows": [
                    {
                        "state": self.sample_state(player_store=4),
                        "side_to_move": 0,
                        "legal_moves": [0, 1],
                        "selection_reason": "challenger_loss_vs_current",
                        "consequence": "caused_loss",
                        "metrics": {"regret": 0.5},
                    },
                ],
            },
            {
                "path": "reports/arena_mcts1200.json",
                "kind": "arena_loss_vs_mcts1200",
                "rows": [
                    {
                        "state": self.sample_state(player_store=5),
                        "side_to_move": 0,
                        "legal_moves": [0, 1],
                        "selection_reason": "challenger_loss_vs_mcts1200",
                        "consequence": "caused_loss",
                        "metrics": {"regret": 0.6},
                    },
                ],
            },
        ]

        rows = hard_state_mining.extract_candidates(artifacts)

        self.assertEqual(
            {
                "challenger_loss_vs_current",
                "challenger_loss_vs_mcts1200",
                "student_teacher_disagreement",
                "high_search_entropy",
                "large_best_second_gap",
                "large_value_error",
            },
            {row["selection_reason"] for row in rows},
        )
        forensic_rows = [
            row for row in rows if row["source_run"]["kind"] == "forensic_suite"
        ]
        self.assertTrue(forensic_rows)
        self.assertEqual("challenger", forensic_rows[0]["source_run"]["system"])
        self.assertEqual(
            "azlite_forensic_suite_v1", forensic_rows[0]["source_run"]["schema"]
        )

    def test_forensic_multi_signal_candidates_deduplicate_without_consequence_conflicts(
        self,
    ):
        from ml.alphazero_lite import hard_state_mining

        artifacts = [
            {
                "path": "reports/forensic.json",
                "schema": "azlite_forensic_suite_v1",
                "systems": {
                    "challenger": {
                        "artifact_path": "challenger.pt",
                        "rows": [
                            {
                                "id": "pos-1",
                                "bucket": "capture_available",
                                "state": self.sample_state(),
                                "side_to_move": 0,
                                "legal_moves": [0, 1],
                                "selected_move": 1,
                                "reference_move": 0,
                                "agrees_top1": False,
                                "regret": 0.4,
                                "teacher_value": 0.1,
                                "system_value": -0.2,
                                "value_error": 0.8,
                                "entropy": 0.9,
                                "best_second_gap": 0.7,
                            }
                        ],
                    }
                },
            }
        ]

        rows = hard_state_mining.extract_candidates(artifacts)
        deduplicated = hard_state_mining.deduplicate_candidates(rows)

        self.assertEqual(1, len(deduplicated))
        self.assertEqual(
            {
                "student_teacher_disagreement",
                "large_value_error",
                "high_search_entropy",
                "large_best_second_gap",
            },
            set(deduplicated[0]["selection_reasons"]),
        )

    def test_forensic_candidates_preserve_ply_from_tags_through_deduplication(self):
        from ml.alphazero_lite import hard_state_mining

        artifacts = [
            {
                "path": "reports/forensic.json",
                "schema": "azlite_forensic_suite_v1",
                "systems": {
                    "challenger": {
                        "artifact_path": "challenger.pt",
                        "rows": [
                            {
                                "id": "opening-1",
                                "bucket": "high_value_swing",
                                "tags": ["high_value_swing", "seed", "ply_2"],
                                "state": self.sample_state(player_store=1),
                                "side_to_move": 0,
                                "legal_moves": [0, 1],
                                "selected_move": 1,
                                "reference_move": 0,
                                "agrees_top1": False,
                                "regret": 0.4,
                                "teacher_value": 0.1,
                                "system_value": -0.2,
                                "value_error": 0.8,
                                "entropy": 0.9,
                                "best_second_gap": 0.7,
                            }
                        ],
                    }
                },
            }
        ]

        candidates = hard_state_mining.extract_candidates(artifacts)

        self.assertTrue(candidates)
        self.assertEqual({2}, {candidate["ply"] for candidate in candidates})

        deduplicated = hard_state_mining.deduplicate_candidates(candidates)

        self.assertEqual(1, len(deduplicated))
        self.assertEqual(2, deduplicated[0]["ply"])

    def test_forensic_candidates_fallback_to_opening_ply_when_tags_omit_it(self):
        from ml.alphazero_lite import hard_state_mining

        artifacts = [
            {
                "path": "reports/forensic.json",
                "schema": "azlite_forensic_suite_v1",
                "systems": {
                    "challenger": {
                        "artifact_path": "challenger.pt",
                        "rows": [
                            {
                                "id": "opening-0",
                                "bucket": "opening_plies_1_8",
                                "phase": "opening",
                                "tags": ["opening_plies_1_8", "seed"],
                                "state": self.sample_state(),
                                "side_to_move": 0,
                                "legal_moves": [0, 1],
                                "selected_move": 1,
                                "reference_move": 0,
                                "agrees_top1": False,
                                "regret": 0.4,
                                "teacher_value": 0.1,
                                "system_value": -0.2,
                                "value_error": 0.8,
                                "entropy": 0.9,
                                "best_second_gap": 0.7,
                            }
                        ],
                    }
                },
            }
        ]

        candidates = hard_state_mining.extract_candidates(artifacts)

        self.assertTrue(candidates)
        self.assertEqual({8}, {candidate["ply"] for candidate in candidates})

    def test_forensic_candidates_skip_unstable_reference_rows_for_disagreement_mining(
        self,
    ):
        from ml.alphazero_lite import hard_state_mining

        artifacts = [
            {
                "path": "reports/forensic.json",
                "schema": "azlite_forensic_suite_v1",
                "systems": {
                    "challenger": {
                        "artifact_path": "challenger.pt",
                        "rows": [
                            {
                                "id": "unstable-1",
                                "bucket": "capture_available",
                                "tags": ["capture_available", "seed", "ply_4"],
                                "state": self.sample_state(),
                                "side_to_move": 0,
                                "legal_moves": [0, 1],
                                "selected_move": 1,
                                "reference_move": None,
                                "reference_unstable": True,
                                "agrees_top1": None,
                                "regret": None,
                                "teacher_value": None,
                                "system_value": -0.2,
                                "value_error": 0.8,
                                "entropy": 0.9,
                                "best_second_gap": 0.7,
                            }
                        ],
                    }
                },
            }
        ]

        rows = hard_state_mining.extract_candidates(artifacts)

        self.assertNotIn(
            "student_teacher_disagreement", {row["selection_reason"] for row in rows}
        )
        self.assertEqual(
            {"large_value_error", "high_search_entropy", "large_best_second_gap"},
            {row["selection_reason"] for row in rows},
        )

    def test_normalize_candidate_rejects_conflicting_ply_and_move_index(self):
        from ml.alphazero_lite import hard_state_mining

        with self.assertRaisesRegex(
            ValueError, "candidate ply and move_index must match"
        ):
            hard_state_mining.normalize_candidate(
                self.make_candidate(ply=8, move_index=9)
            )

    def test_normalize_candidate_rejects_non_integer_move_index_before_mismatch_check(
        self,
    ):
        from ml.alphazero_lite import hard_state_mining

        with self.assertRaisesRegex(ValueError, "candidate ply must be an integer"):
            hard_state_mining.normalize_candidate(
                self.make_candidate(ply=8, move_index="8")
            )

    def test_cli_writes_jsonl_and_summary_report(self):
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="azlite-hard-mining-") as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "report.json"
            out_jsonl = tmp_path / "mined.jsonl"
            out_report = tmp_path / "summary.json"
            input_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_forensic_suite_v1",
                        "systems": {
                            "challenger": {
                                "artifact_path": "challenger.pt",
                                "rows": [
                                    {
                                        "id": "pos-1",
                                        "bucket": "capture_available",
                                        "state": self.sample_state(),
                                        "side_to_move": 0,
                                        "legal_moves": [0, 1],
                                        "selected_move": 1,
                                        "reference_move": 0,
                                        "agrees_top1": False,
                                        "regret": 0.4,
                                        "teacher_value": 0.1,
                                        "system_value": -0.2,
                                        "value_error": 0.8,
                                        "entropy": 0.9,
                                        "best_second_gap": 0.7,
                                    }
                                ],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    PYTHON_BIN,
                    "ml/alphazero_lite/mine_hard_states.py",
                    "--inputs",
                    str(input_path),
                    "--out-jsonl",
                    str(out_jsonl),
                    "--out-report",
                    str(out_report),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertTrue(out_jsonl.exists())
            self.assertTrue(out_report.exists())

            report = json.loads(out_report.read_text(encoding="utf-8"))
            self.assertIn("candidate_counts_by_source_class", report)
            self.assertIn("contribution_totals_by_reason", report)

    def test_cli_fails_clearly_when_no_supported_artifacts_are_found(self):
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="azlite-hard-mining-empty-") as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "unsupported.json"
            out_jsonl = tmp_path / "mined.jsonl"
            out_report = tmp_path / "summary.json"
            input_path.write_text(json.dumps({"schema": "unknown"}), encoding="utf-8")

            result = subprocess.run(
                [
                    PYTHON_BIN,
                    "ml/alphazero_lite/mine_hard_states.py",
                    "--inputs",
                    str(input_path),
                    "--out-jsonl",
                    str(out_jsonl),
                    "--out-report",
                    str(out_report),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn(
                "no supported hard-state mining artifacts found", result.stderr
            )
            self.assertNotIn("Traceback", result.stderr)

    def test_cli_rejects_summary_only_loss_artifact_without_traceback(self):
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="azlite-hard-mining-loss-") as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "loss.json"
            out_jsonl = tmp_path / "mined.jsonl"
            out_report = tmp_path / "summary.json"
            input_path.write_text(
                json.dumps(
                    {
                        "kind": "arena_loss_vs_current",
                        "games_played": 10,
                        "losses": 6,
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    PYTHON_BIN,
                    "ml/alphazero_lite/mine_hard_states.py",
                    "--inputs",
                    str(input_path),
                    "--out-jsonl",
                    str(out_jsonl),
                    "--out-report",
                    str(out_report),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("summary-only loss artifact is not supported", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_run_pipeline_summary_reports_all_requested_source_classes(self):
        from ml.alphazero_lite import hard_state_mining

        fixture_dir = (
            Path(__file__).resolve().parents[2] / "test/fixtures/ai/hard_state_mining"
        )
        rows, summary = hard_state_mining.run_pipeline([str(fixture_dir)])

        self.assertTrue(rows)
        self.assertEqual(
            {
                "challenger_loss_vs_current",
                "challenger_loss_vs_mcts1200",
                "student_teacher_disagreement",
                "high_search_entropy",
                "large_best_second_gap",
                "large_value_error",
            },
            set(summary["source_coverage"]),
        )
        self.assertEqual(
            {
                "challenger_loss_vs_current": 1,
                "challenger_loss_vs_mcts1200": 1,
                "student_teacher_disagreement": 1,
                "high_search_entropy": 1,
                "large_best_second_gap": 1,
                "large_value_error": 2,
            },
            summary["candidate_counts_by_source_class"],
        )

    def test_hard_state_mining_doc_mentions_cli_and_schema(self):
        doc_path = (
            Path(__file__).resolve().parents[2]
            / "docs/alphazero-lite-hard-state-mining.md"
        )
        doc = doc_path.read_text(encoding="utf-8")

        self.assertIn("# AlphaZero-lite Hard-State Mining", doc)
        self.assertIn("`ml/alphazero_lite/mine_hard_states.py`", doc)
        self.assertIn("`priority_score`", doc)
        self.assertIn("`selection_reasons`", doc)

    def test_extract_candidates_rejects_loss_reason_mismatch(self):
        from ml.alphazero_lite import hard_state_mining

        artifacts = [
            {
                "path": "reports/arena_current.json",
                "kind": "arena_loss_vs_current",
                "rows": [
                    {
                        "state": self.sample_state(),
                        "side_to_move": 0,
                        "legal_moves": [0, 1],
                        "selection_reason": "challenger_loss_vs_mcts1200",
                        "consequence": "caused_loss",
                        "metrics": {"regret": 0.5},
                    }
                ],
            }
        ]

        with self.assertRaisesRegex(
            ValueError, "selection_reason does not match artifact kind"
        ):
            hard_state_mining.extract_candidates(artifacts)

    def test_normalize_candidate_rejects_unknown_consequence(self):
        from ml.alphazero_lite import hard_state_mining

        with self.assertRaisesRegex(ValueError, "unknown consequence"):
            hard_state_mining.normalize_candidate(
                self.make_candidate(consequence="caused_los")
            )

    def test_normalize_candidate_rejects_invalid_state_shape(self):
        from ml.alphazero_lite import hard_state_mining

        invalid_state = {
            "player_pits": [4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }

        with self.assertRaisesRegex(
            ValueError, "player_pits must be a list of 6 integers"
        ):
            hard_state_mining.normalize_candidate(
                self.make_candidate(state=invalid_state)
            )

    def test_normalize_candidate_rejects_invalid_current_player_and_side_to_move(self):
        from ml.alphazero_lite import hard_state_mining

        with self.assertRaisesRegex(ValueError, "state.current_player must be 0 or 1"):
            hard_state_mining.normalize_candidate(
                self.make_candidate(
                    state={
                        "player_pits": [4, 4, 4, 4, 4, 4],
                        "opponent_pits": [4, 4, 4, 4, 4, 4],
                        "player_store": 0,
                        "opponent_store": 0,
                        "current_player": 2,
                    }
                )
            )

        with self.assertRaisesRegex(
            ValueError, "side_to_move must match state.current_player"
        ):
            hard_state_mining.normalize_candidate(self.make_candidate(side_to_move=1))

    def test_normalize_candidate_rejects_bool_side_to_move_and_invalid_legal_moves(
        self,
    ):
        from ml.alphazero_lite import hard_state_mining

        with self.assertRaisesRegex(ValueError, "side_to_move must be 0 or 1"):
            hard_state_mining.normalize_candidate(
                self.make_candidate(side_to_move=True)
            )

        with self.assertRaisesRegex(
            ValueError, "legal_moves must contain unique moves in range 0..5"
        ):
            hard_state_mining.normalize_candidate(
                self.make_candidate(legal_moves=[0, 0])
            )

    def test_normalize_candidate_rejects_non_dict_source_run_and_metrics(self):
        from ml.alphazero_lite import hard_state_mining

        with self.assertRaisesRegex(
            ValueError, "candidate source_run must be a dictionary"
        ):
            hard_state_mining.normalize_candidate(
                self.make_candidate(source_run="forensic")
            )

        with self.assertRaisesRegex(
            ValueError, "candidate metrics must be a dictionary"
        ):
            hard_state_mining.normalize_candidate(self.make_candidate(metrics="oops"))

    def test_deduplicate_candidates_rejects_non_numeric_metrics(self):
        from ml.alphazero_lite import hard_state_mining

        with self.assertRaisesRegex(
            ValueError, "candidate metric regret must be numeric"
        ):
            hard_state_mining.deduplicate_candidates(
                [self.make_candidate(metrics={"regret": "oops"})]
            )

    def test_cli_rejects_malformed_json_without_traceback(self):
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="azlite-hard-mining-json-") as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "broken.json"
            out_jsonl = tmp_path / "mined.jsonl"
            out_report = tmp_path / "summary.json"
            input_path.write_text("{not json", encoding="utf-8")

            result = subprocess.run(
                [
                    PYTHON_BIN,
                    "ml/alphazero_lite/mine_hard_states.py",
                    "--inputs",
                    str(input_path),
                    "--out-jsonl",
                    str(out_jsonl),
                    "--out-report",
                    str(out_report),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("invalid JSON", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_cli_rejects_missing_input_without_traceback(self):
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="azlite-hard-mining-missing-") as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "missing.json"
            out_jsonl = tmp_path / "mined.jsonl"
            out_report = tmp_path / "summary.json"

            result = subprocess.run(
                [
                    PYTHON_BIN,
                    "ml/alphazero_lite/mine_hard_states.py",
                    "--inputs",
                    str(input_path),
                    "--out-jsonl",
                    str(out_jsonl),
                    "--out-report",
                    str(out_report),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("unable to read hard-state mining artifact", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_cli_rejects_malformed_supported_row_without_traceback(self):
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="azlite-hard-mining-row-") as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "loss.json"
            out_jsonl = tmp_path / "mined.jsonl"
            out_report = tmp_path / "summary.json"
            input_path.write_text(
                json.dumps(
                    {
                        "kind": "arena_loss_vs_current",
                        "rows": [
                            {
                                "side_to_move": 0,
                                "legal_moves": [0, 1],
                                "selection_reason": "challenger_loss_vs_current",
                                "consequence": "caused_loss",
                                "metrics": {"regret": 0.5},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    PYTHON_BIN,
                    "ml/alphazero_lite/mine_hard_states.py",
                    "--inputs",
                    str(input_path),
                    "--out-jsonl",
                    str(out_jsonl),
                    "--out-report",
                    str(out_report),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("missing required row field", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_cli_rejects_malformed_forensic_row_without_traceback(self):
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(
            prefix="azlite-hard-mining-forensic-row-"
        ) as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "forensic.json"
            out_jsonl = tmp_path / "mined.jsonl"
            out_report = tmp_path / "summary.json"
            input_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_forensic_suite_v1",
                        "systems": {
                            "challenger": {
                                "artifact_path": "challenger.pt",
                                "rows": [
                                    {
                                        "agrees_top1": False,
                                        "side_to_move": 0,
                                        "legal_moves": [0, 1],
                                    }
                                ],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    PYTHON_BIN,
                    "ml/alphazero_lite/mine_hard_states.py",
                    "--inputs",
                    str(input_path),
                    "--out-jsonl",
                    str(out_jsonl),
                    "--out-report",
                    str(out_report),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("missing required row field", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_extract_candidates_rejects_non_dict_loss_row(self):
        from ml.alphazero_lite import hard_state_mining

        artifacts = [
            {
                "path": "reports/arena_current.json",
                "kind": "arena_loss_vs_current",
                "rows": ["not-a-dict"],
            }
        ]

        with self.assertRaisesRegex(ValueError, "artifact row must be a dictionary"):
            hard_state_mining.extract_candidates(artifacts)

    def test_extract_candidates_rejects_non_dict_forensic_row(self):
        from ml.alphazero_lite import hard_state_mining

        artifacts = [
            {
                "path": "reports/forensic.json",
                "schema": "azlite_forensic_suite_v1",
                "systems": {
                    "challenger": {
                        "rows": ["not-a-dict"],
                    }
                },
            }
        ]

        with self.assertRaisesRegex(ValueError, "forensic row must be a dictionary"):
            hard_state_mining.extract_candidates(artifacts)

    def test_load_artifacts_rejects_non_list_loss_rows(self):
        from ml.alphazero_lite import hard_state_mining

        with tempfile.TemporaryDirectory(prefix="azlite-hard-mining-bad-rows-") as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "loss.json"
            input_path.write_text(
                json.dumps(
                    {
                        "kind": "arena_loss_vs_current",
                        "rows": {"bad": "shape"},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "malformed hard-state mining artifact rows"
            ):
                hard_state_mining.load_artifacts([str(input_path)])

    def test_load_artifacts_rejects_non_dict_forensic_systems(self):
        from ml.alphazero_lite import hard_state_mining

        with tempfile.TemporaryDirectory(
            prefix="azlite-hard-mining-bad-systems-"
        ) as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "forensic.json"
            input_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_forensic_suite_v1",
                        "systems": ["challenger"],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "invalid forensic-suite artifact"):
                hard_state_mining.load_artifacts([str(input_path)])

    def test_load_artifacts_rejects_non_dict_forensic_system_payload(self):
        from ml.alphazero_lite import hard_state_mining

        with tempfile.TemporaryDirectory(
            prefix="azlite-hard-mining-bad-system-payload-"
        ) as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "forensic.json"
            input_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_forensic_suite_v1",
                        "systems": {
                            "challenger": "bad-payload",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "system 'challenger' must be an object"
            ):
                hard_state_mining.load_artifacts([str(input_path)])

    def test_load_artifacts_rejects_non_object_payload(self):
        from ml.alphazero_lite import hard_state_mining

        with tempfile.TemporaryDirectory(
            prefix="azlite-hard-mining-bad-payload-"
        ) as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "payload.json"
            input_path.write_text(
                json.dumps([{"schema": "azlite_forensic_suite_v1"}]), encoding="utf-8"
            )

            with self.assertRaisesRegex(
                ValueError, "hard-state mining artifact must be a JSON object"
            ):
                hard_state_mining.load_artifacts([str(input_path)])

    def test_load_artifacts_does_not_classify_kind_forensic_suite_without_schema(self):
        from ml.alphazero_lite import hard_state_mining

        with tempfile.TemporaryDirectory(prefix="azlite-hard-mining-kind-only-") as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "forensic-kind.json"
            input_path.write_text(
                json.dumps({"kind": "forensic_suite"}), encoding="utf-8"
            )

            with self.assertRaisesRegex(
                ValueError, "no supported hard-state mining artifacts found"
            ):
                hard_state_mining.load_artifacts([str(input_path)])

    def test_extract_candidates_treats_null_forensic_threshold_metrics_as_zero(self):
        from ml.alphazero_lite import hard_state_mining

        artifacts = [
            {
                "path": "reports/forensic.json",
                "schema": "azlite_forensic_suite_v1",
                "systems": {
                    "challenger": {
                        "rows": [
                            {
                                "state": self.sample_state(),
                                "side_to_move": 0,
                                "legal_moves": [0, 1],
                                "agrees_top1": False,
                                "value_error": None,
                            }
                        ]
                    }
                },
            }
        ]

        extracted = hard_state_mining.extract_candidates(artifacts)
        deduplicated = hard_state_mining.deduplicate_candidates(extracted)

        self.assertEqual(0.0, deduplicated[0]["metadata"]["max_value_error"])
