import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class BuildForensicReferencesTest(unittest.TestCase):
    def suite_rows(self):
        return [
            {
                "id": "capture_available-016",
                "state": {
                    "player_pits": [5, 1, 5, 5, 5, 0],
                    "opponent_pits": [1, 6, 0, 7, 6, 5],
                    "player_store": 1,
                    "opponent_store": 1,
                    "current_player": 0,
                },
                "side_to_move": 0,
                "legal_moves": [0, 1, 2, 3, 4],
                "phase": "opening",
                "bucket": "capture_available",
                "tags": ["capture_available", "generated", "ply_4"],
                "source": "generated",
            }
        ]

    def test_build_references_writes_one_row_per_canonical_state(self):
        from ml.alphazero_lite import build_forensic_references as module

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            suite_path = tmp_path / "suite.json"
            out_path = tmp_path / "reference_moves.json"
            suite_path.write_text(json.dumps(self.suite_rows()), encoding="utf-8")

            rows = module.build_reference_artifact(
                suite_path=suite_path,
                out_path=out_path,
                policy_simulations=1200,
                value_simulations=1800,
                seed=500,
                reference_runner=lambda *_args: {
                    "selected_move": 3,
                    "teacher_value": 0.4183,
                },
            )

            artifact = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual("azlite_forensic_references_v1", artifact["schema"])
            self.assertEqual(1, len(rows))
            self.assertEqual(1, len(artifact["rows"]))
            self.assertEqual("capture_available-016", rows[0]["id"])
            self.assertIn("canonical_state", rows[0])
            self.assertIn("reference_move", rows[0])
            self.assertEqual(rows, artifact["rows"])

    def test_reference_artifact_marks_explicit_instability_when_same_state_disagrees_across_seed_samples(
        self,
    ):
        from ml.alphazero_lite import build_forensic_references as module

        disagreement = module.finalize_reference_row(
            row_id="capture_available-016",
            canonical_state="state-key",
            state=self.suite_rows()[0]["state"],
            seed_samples=[
                {"seed": 2100, "reference_move": 3, "teacher_value": 0.4183},
                {"seed": 3101, "reference_move": 4, "teacher_value": 0.4367},
            ],
        )

        self.assertTrue(disagreement["reference_unstable"])
        self.assertCountEqual([3, 4], disagreement["observed_reference_moves"])

    def test_finalize_reference_rows_collapses_duplicate_canonical_states(self):
        from ml.alphazero_lite import build_forensic_references as module

        state = self.suite_rows()[0]["state"]
        rows = module.finalize_reference_rows(
            [
                {
                    "row_id": "capture_available-016",
                    "canonical_state": "state-key",
                    "state": state,
                    "seed_samples": [
                        {"seed": 2100, "reference_move": 3, "teacher_value": 0.4183},
                    ],
                },
                {
                    "row_id": "capture_available-099",
                    "canonical_state": "state-key",
                    "state": state,
                    "seed_samples": [
                        {"seed": 2100, "reference_move": 3, "teacher_value": 0.4183},
                    ],
                },
            ]
        )

        self.assertEqual(1, len(rows))
        self.assertEqual("state-key", rows[0]["canonical_state"])
        self.assertEqual("capture_available-016", rows[0]["id"])
        self.assertEqual(3, rows[0]["reference_move"])

    def test_build_reference_artifact_does_not_recompute_duplicate_canonical_states(
        self,
    ):
        from ml.alphazero_lite import build_forensic_references as module

        state = self.suite_rows()[0]["state"]
        reference_calls = []

        def fake_reference_runner(*args):
            reference_calls.append(args)
            return {"selected_move": 3, "teacher_value": 0.4183}

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "reference_moves.json"
            duplicate_suite = [
                SimpleNamespace(id="capture_available-016", state=state),
                SimpleNamespace(id="capture_available-099", state=dict(state)),
            ]

            with patch.object(module, "load_suite", return_value=duplicate_suite):
                rows = module.build_reference_artifact(
                    suite_path="ignored.json",
                    out_path=out_path,
                    policy_simulations=1200,
                    value_simulations=1800,
                    seed=500,
                    reference_runner=fake_reference_runner,
                )

        self.assertEqual(1, len(reference_calls))
        self.assertEqual(1, len(rows))
        self.assertEqual("capture_available-016", rows[0]["id"])

    def test_build_reference_artifact_rejects_empty_sample_seeds(self):
        from ml.alphazero_lite import build_forensic_references as module

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            suite_path = tmp_path / "suite.json"
            out_path = tmp_path / "reference_moves.json"
            suite_path.write_text(json.dumps(self.suite_rows()), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "sample_seeds must not be empty"):
                module.build_reference_artifact(
                    suite_path=suite_path,
                    out_path=out_path,
                    policy_simulations=1200,
                    value_simulations=1800,
                    seed=500,
                    sample_seeds=[],
                    reference_runner=lambda *_args: {
                        "selected_move": 3,
                        "teacher_value": 0.4183,
                    },
                )

    def test_build_reference_artifact_stable_multi_seed_rows_use_deterministic_teacher_value(
        self,
    ):
        from ml.alphazero_lite import build_forensic_references as module

        references_by_seed = {
            2100: {"selected_move": 3, "teacher_value": 0.4183},
            3101: {"selected_move": 3, "teacher_value": 0.4367},
        }

        def fake_reference_runner(
            _state, _policy_simulations, _value_simulations, sample_seed, _index
        ):
            return references_by_seed[sample_seed]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            suite_path = tmp_path / "suite.json"
            out_path = tmp_path / "reference_moves.json"
            suite_path.write_text(json.dumps(self.suite_rows()), encoding="utf-8")

            rows = module.build_reference_artifact(
                suite_path=suite_path,
                out_path=out_path,
                policy_simulations=1200,
                value_simulations=1800,
                seed=500,
                sample_seeds=[2100, 3101],
                reference_runner=fake_reference_runner,
            )

        self.assertEqual(3, rows[0]["reference_move"])
        self.assertEqual(0.4275, rows[0]["teacher_value"])
        self.assertFalse(rows[0]["reference_unstable"])
        self.assertEqual(
            [
                {"seed": 2100, "reference_move": 3, "teacher_value": 0.4183},
                {"seed": 3101, "reference_move": 3, "teacher_value": 0.4367},
            ],
            rows[0]["seed_samples"],
        )

    def test_build_reference_artifact_marks_multi_seed_move_disagreement_as_unstable(
        self,
    ):
        from ml.alphazero_lite import build_forensic_references as module

        references_by_seed = {
            2100: {"selected_move": 3, "teacher_value": 0.4183},
            3101: {"selected_move": 4, "teacher_value": 0.4367},
        }

        def fake_reference_runner(
            _state, _policy_simulations, _value_simulations, sample_seed, _index
        ):
            return references_by_seed[sample_seed]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            suite_path = tmp_path / "suite.json"
            out_path = tmp_path / "reference_moves.json"
            suite_path.write_text(json.dumps(self.suite_rows()), encoding="utf-8")

            rows = module.build_reference_artifact(
                suite_path=suite_path,
                out_path=out_path,
                policy_simulations=1200,
                value_simulations=1800,
                seed=500,
                sample_seeds=[2100, 3101],
                reference_runner=fake_reference_runner,
            )

        self.assertIsNone(rows[0]["reference_move"])
        self.assertIsNone(rows[0]["teacher_value"])
        self.assertTrue(rows[0]["reference_unstable"])
        self.assertEqual([3, 4], rows[0]["observed_reference_moves"])

    def test_build_reference_artifact_preserves_child_stats_for_stable_rows(self):
        from ml.alphazero_lite import build_forensic_references as module

        child_stats = [
            {"move": 3, "visits": 10, "win_rate": 0.7},
            {"move": 1, "visits": 8, "win_rate": 0.3},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            suite_path = tmp_path / "suite.json"
            out_path = tmp_path / "reference_moves.json"
            suite_path.write_text(json.dumps(self.suite_rows()), encoding="utf-8")

            rows = module.build_reference_artifact(
                suite_path=suite_path,
                out_path=out_path,
                policy_simulations=1200,
                value_simulations=1800,
                seed=500,
                reference_runner=lambda *_args: {
                    "selected_move": 3,
                    "teacher_value": 0.4183,
                    "child_stats": child_stats,
                },
            )

        self.assertEqual(child_stats, rows[0]["child_stats"])

    def test_capture_available_016_mismatch_stabilization_emits_one_shared_unstable_reference_row(
        self,
    ):
        from ml.alphazero_lite import build_forensic_references as module

        references_by_seed = {
            2100: {"selected_move": 3, "teacher_value": 0.4183},
            3101: {"selected_move": 4, "teacher_value": 0.4367},
        }

        def fake_reference_runner(
            _state, _policy_simulations, _value_simulations, sample_seed, _index
        ):
            return references_by_seed[sample_seed]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            suite_path = tmp_path / "suite.json"
            out_path = tmp_path / "reference_moves.json"
            suite_path.write_text(json.dumps(self.suite_rows()), encoding="utf-8")

            rows = module.build_reference_artifact(
                suite_path=suite_path,
                out_path=out_path,
                policy_simulations=1200,
                value_simulations=1800,
                seed=500,
                sample_seeds=[2100, 3101],
                reference_runner=fake_reference_runner,
            )

            artifact = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual(1, len(rows))
        self.assertEqual(1, len(artifact["rows"]))
        self.assertEqual("capture_available-016", rows[0]["id"])
        self.assertEqual(
            module.canonical_state_key(self.suite_rows()[0]["state"]),
            rows[0]["canonical_state"],
        )
        self.assertTrue(rows[0]["reference_unstable"])
        self.assertIsNone(rows[0]["reference_move"])
        self.assertIsNone(rows[0]["teacher_value"])
        self.assertEqual([3, 4], rows[0]["observed_reference_moves"])
        self.assertEqual(rows, artifact["rows"])

    def test_run_forensic_suite_reuses_shared_reference_artifact_across_report_seeds(
        self,
    ):
        import importlib

        from ml.alphazero_lite import build_forensic_references as reference_module

        with patch.dict(os.environ, {"AZLITE_FORENSIC_SUITE_STUB": "1"}):
            from ml.alphazero_lite import run_forensic_suite as suite_module

            importlib.reload(suite_module)

        state = self.suite_rows()[0]["state"]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            suite_path = tmp_path / "suite.json"
            reference_path = tmp_path / "reference_moves.json"
            baseline_out_path = tmp_path / "baseline_report.json"
            candidate_out_path = tmp_path / "candidate_report.json"
            current_artifact = tmp_path / "current-artifact"
            candidate_artifact = tmp_path / "candidate-artifact"
            suite_path.write_text(json.dumps(self.suite_rows()), encoding="utf-8")
            current_artifact.write_text("current", encoding="utf-8")
            candidate_artifact.write_text("candidate", encoding="utf-8")

            reference_module.build_reference_artifact(
                suite_path=suite_path,
                out_path=reference_path,
                policy_simulations=1200,
                value_simulations=1800,
                seed=500,
                sample_seeds=[2100, 3101],
                reference_runner=lambda *_args: {
                    "selected_move": 3,
                    "teacher_value": 0.4183,
                },
            )

            def fake_run_reference(
                _state, _policy_simulations, _value_simulations, seed, _index
            ):
                selected_move = 4 if seed >= 3000 else 1
                return {
                    "selected_move": selected_move,
                    "teacher_value": 0.99,
                    "child_stats": [
                        {"move": selected_move, "visits": 10, "win_rate": 0.9},
                        {"move": 3, "visits": 5, "win_rate": 0.4},
                    ],
                }

            def fake_evaluate_artifact_position(*, artifact_path, **_kwargs):
                return {
                    "selected_move": 3
                    if str(artifact_path).endswith("candidate-artifact")
                    else 1,
                    "value": 0.15,
                }

            def run_suite(seed, challenger_artifact_path, out_path):
                argv = [
                    "run_forensic_suite.py",
                    "--suite",
                    str(suite_path),
                    "--current-artifact",
                    str(current_artifact),
                    "--challenger-artifact",
                    str(challenger_artifact_path),
                    "--reference-artifact",
                    str(reference_path),
                    "--seed",
                    str(seed),
                    "--out",
                    str(out_path),
                ]

                with patch.object(sys, "argv", argv):
                    suite_module.main()

                return json.loads(out_path.read_text(encoding="utf-8"))

            with (
                patch.object(
                    suite_module,
                    "ArtifactEvaluator",
                    side_effect=lambda path: {"artifact": str(path)},
                ),
                patch.object(
                    suite_module,
                    "build_eval_search_options",
                    return_value={},
                ),
                patch.object(
                    suite_module,
                    "evaluate_artifact_position",
                    side_effect=fake_evaluate_artifact_position,
                ),
                patch.object(
                    suite_module,
                    "run_reference",
                    side_effect=fake_run_reference,
                ) as run_reference_mock,
            ):
                baseline_report = run_suite(2040, current_artifact, baseline_out_path)
                candidate_report = run_suite(
                    3041, candidate_artifact, candidate_out_path
                )

        baseline_row = baseline_report["systems"]["challenger"]["rows"][0]
        candidate_row = candidate_report["systems"]["challenger"]["rows"][0]

        self.assertEqual(3, baseline_row["reference_move"])
        self.assertEqual(3, candidate_row["reference_move"])
        self.assertIsNone(baseline_row["regret"])
        self.assertIsNone(candidate_row["regret"])
        self.assertIsNone(
            baseline_report["systems"]["challenger"]["overall"]["average_regret"]
        )
        self.assertIsNone(
            candidate_report["systems"]["challenger"]["overall"]["average_regret"]
        )
        self.assertIsNone(
            baseline_report["systems"]["challenger"]["overall"]["blunder_rate"]
        )
        self.assertIsNone(
            candidate_report["systems"]["challenger"]["overall"]["blunder_rate"]
        )
        self.assertIsNone(
            baseline_report["systems"]["challenger"]["buckets"]["capture_available"][
                "average_regret"
            ]
        )
        self.assertIsNone(
            candidate_report["systems"]["challenger"]["buckets"]["capture_available"][
                "average_regret"
            ]
        )
        self.assertIsNone(
            baseline_report["systems"]["challenger"]["buckets"]["capture_available"][
                "blunder_rate"
            ]
        )
        self.assertIsNone(
            candidate_report["systems"]["challenger"]["buckets"]["capture_available"][
                "blunder_rate"
            ]
        )
        self.assertIsNone(
            baseline_report["buckets"]["capture_available"]["systems"]["challenger"][
                "average_regret"
            ]
        )
        self.assertIsNone(
            candidate_report["buckets"]["capture_available"]["systems"]["challenger"][
                "average_regret"
            ]
        )
        self.assertIsNone(
            baseline_report["buckets"]["capture_available"]["systems"]["challenger"][
                "blunder_rate"
            ]
        )
        self.assertIsNone(
            candidate_report["buckets"]["capture_available"]["systems"]["challenger"][
                "blunder_rate"
            ]
        )
        self.assertEqual(0.4183, baseline_row["teacher_value"])
        self.assertEqual(0.4183, candidate_row["teacher_value"])
        self.assertEqual(
            reference_module.canonical_state_key(state), baseline_row["canonical_state"]
        )
        self.assertEqual(
            reference_module.canonical_state_key(state),
            candidate_row["canonical_state"],
        )
        self.assertEqual(
            [
                {"seed": 2100, "reference_move": 3, "teacher_value": 0.4183},
                {"seed": 3101, "reference_move": 3, "teacher_value": 0.4183},
            ],
            baseline_row["seed_samples"],
        )
        self.assertEqual(baseline_row["seed_samples"], candidate_row["seed_samples"])
        self.assertEqual("shared_artifact", baseline_report["reference"]["kind"])
        self.assertEqual(
            str(reference_path), baseline_report["reference"]["artifact_path"]
        )
        self.assertEqual(baseline_report["reference"], candidate_report["reference"])
        self.assertEqual(0, run_reference_mock.call_count)

    def test_run_forensic_suite_excludes_unstable_shared_references_from_top1_agreement(
        self,
    ):
        import importlib

        from ml.alphazero_lite import build_forensic_references as reference_module

        with patch.dict(os.environ, {"AZLITE_FORENSIC_SUITE_STUB": "1"}):
            from ml.alphazero_lite import run_forensic_suite as suite_module

            importlib.reload(suite_module)

        stable_state = self.suite_rows()[0]["state"]
        unstable_state = {
            "player_pits": [1, 5, 0, 6, 2, 7],
            "opponent_pits": [4, 3, 5, 0, 1, 6],
            "player_store": 3,
            "opponent_store": 2,
            "current_player": 0,
        }
        suite_rows = self.suite_rows() + [
            {
                "id": "sparse_endgame-101",
                "state": unstable_state,
                "side_to_move": 0,
                "legal_moves": [0, 1, 3, 4, 5],
                "phase": "endgame",
                "bucket": "sparse_endgame",
                "tags": ["sparse_endgame", "generated", "ply_18"],
                "source": "generated",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            suite_path = tmp_path / "suite.json"
            reference_path = tmp_path / "reference_moves.json"
            out_path = tmp_path / "report_out.json"
            current_artifact = tmp_path / "current-artifact"
            challenger_artifact = tmp_path / "challenger-artifact"
            suite_path.write_text(json.dumps(suite_rows), encoding="utf-8")
            current_artifact.write_text("current", encoding="utf-8")
            challenger_artifact.write_text("challenger", encoding="utf-8")

            reference_artifact = {
                "schema": "azlite_forensic_references_v1",
                "reference": {
                    "policy_simulations": 1200,
                    "value_simulations": 1800,
                    "sample_seeds": [2100, 3101],
                },
                "rows": [
                    {
                        "id": "capture_available-016",
                        "canonical_state": reference_module.canonical_state_key(
                            stable_state
                        ),
                        "state": stable_state,
                        "reference_move": 3,
                        "teacher_value": 0.4183,
                        "reference_unstable": False,
                        "observed_reference_moves": [3],
                        "seed_samples": [
                            {
                                "seed": 2100,
                                "reference_move": 3,
                                "teacher_value": 0.4183,
                            },
                            {
                                "seed": 3101,
                                "reference_move": 3,
                                "teacher_value": 0.4183,
                            },
                        ],
                    },
                    {
                        "id": "sparse_endgame-101",
                        "canonical_state": reference_module.canonical_state_key(
                            unstable_state
                        ),
                        "state": unstable_state,
                        "reference_move": None,
                        "teacher_value": None,
                        "reference_unstable": True,
                        "observed_reference_moves": [1, 4],
                        "seed_samples": [
                            {"seed": 2100, "reference_move": 1, "teacher_value": 0.2},
                            {"seed": 3101, "reference_move": 4, "teacher_value": 0.3},
                        ],
                    },
                ],
            }
            reference_path.write_text(json.dumps(reference_artifact), encoding="utf-8")

            def fake_evaluate_artifact_position(*, state, **_kwargs):
                if state == stable_state:
                    return {"selected_move": 3, "value": 0.15}
                return {"selected_move": 1, "value": 0.2}

            argv = [
                "run_forensic_suite.py",
                "--suite",
                str(suite_path),
                "--current-artifact",
                str(current_artifact),
                "--challenger-artifact",
                str(challenger_artifact),
                "--reference-artifact",
                str(reference_path),
                "--out",
                str(out_path),
            ]

            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    suite_module,
                    "ArtifactEvaluator",
                    side_effect=lambda path: {"artifact": str(path)},
                ),
                patch.object(
                    suite_module, "build_eval_search_options", return_value={}
                ),
                patch.object(
                    suite_module,
                    "evaluate_artifact_position",
                    side_effect=fake_evaluate_artifact_position,
                ),
            ):
                suite_module.main()

            report = json.loads(out_path.read_text(encoding="utf-8"))

        challenger_rows = report["systems"]["challenger"]["rows"]
        stable_row = next(
            row for row in challenger_rows if row["id"] == "capture_available-016"
        )
        unstable_row = next(
            row for row in challenger_rows if row["id"] == "sparse_endgame-101"
        )

        self.assertTrue(stable_row["agrees_top1"])
        self.assertIsNone(unstable_row["agrees_top1"])
        self.assertEqual(
            1.0, report["systems"]["challenger"]["overall"]["top1_agreement"]
        )
        self.assertEqual(
            1.0,
            report["systems"]["challenger"]["buckets"]["capture_available"][
                "top1_agreement"
            ],
        )
        self.assertIsNone(
            report["systems"]["challenger"]["buckets"]["sparse_endgame"][
                "top1_agreement"
            ]
        )
        self.assertEqual(
            1.0,
            report["buckets"]["capture_available"]["systems"]["challenger"][
                "top1_agreement"
            ],
        )
        self.assertIsNone(
            report["buckets"]["sparse_endgame"]["systems"]["challenger"][
                "top1_agreement"
            ]
        )

    def test_run_forensic_suite_rejects_report_shaped_shared_reference_artifact(self):
        import importlib

        from ml.alphazero_lite.build_forensic_references import canonical_state_key

        with patch.dict(os.environ, {"AZLITE_FORENSIC_SUITE_STUB": "1"}):
            from ml.alphazero_lite import run_forensic_suite as suite_module

            importlib.reload(suite_module)

            canonical_state = canonical_state_key(self.suite_rows()[0]["state"])

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                suite_path = tmp_path / "suite.json"
                reference_path = tmp_path / "report.json"
                out_path = tmp_path / "report_out.json"
                current_artifact = tmp_path / "current-artifact"
                challenger_artifact = tmp_path / "challenger-artifact"
                suite_path.write_text(json.dumps(self.suite_rows()), encoding="utf-8")
                current_artifact.write_text("current", encoding="utf-8")
                challenger_artifact.write_text("challenger", encoding="utf-8")
                reference_path.write_text(
                    json.dumps(
                        {
                            "schema": "azlite_forensic_suite_v1",
                            "rows": [
                                {
                                    "id": "capture_available-016",
                                    "state": self.suite_rows()[0]["state"],
                                    "canonical_state": canonical_state,
                                    "selected_move": 3,
                                    "teacher_value": 0.4183,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

                argv = [
                    "run_forensic_suite.py",
                    "--suite",
                    str(suite_path),
                    "--current-artifact",
                    str(current_artifact),
                    "--challenger-artifact",
                    str(challenger_artifact),
                    "--reference-artifact",
                    str(reference_path),
                    "--out",
                    str(out_path),
                ]

                with (
                    patch.object(sys, "argv", argv),
                    self.assertRaisesRegex(SystemExit, "shared reference artifact"),
                ):
                    suite_module.main()

    def test_run_forensic_suite_rejects_shared_reference_rows_missing_required_fields(
        self,
    ):
        import importlib

        with patch.dict(os.environ, {"AZLITE_FORENSIC_SUITE_STUB": "1"}):
            from ml.alphazero_lite import run_forensic_suite as suite_module

            importlib.reload(suite_module)

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                suite_path = tmp_path / "suite.json"
                reference_path = tmp_path / "reference_moves.json"
                out_path = tmp_path / "report_out.json"
                current_artifact = tmp_path / "current-artifact"
                challenger_artifact = tmp_path / "challenger-artifact"
                suite_path.write_text(json.dumps(self.suite_rows()), encoding="utf-8")
                current_artifact.write_text("current", encoding="utf-8")
                challenger_artifact.write_text("challenger", encoding="utf-8")
                reference_path.write_text(
                    json.dumps(
                        {
                            "schema": "azlite_forensic_references_v1",
                            "reference": {
                                "policy_simulations": 1200,
                                "value_simulations": 1800,
                                "sample_seeds": [2100],
                            },
                            "rows": [
                                {
                                    "id": "capture_available-016",
                                    "state": self.suite_rows()[0]["state"],
                                    "reference_move": 3,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

                argv = [
                    "run_forensic_suite.py",
                    "--suite",
                    str(suite_path),
                    "--current-artifact",
                    str(current_artifact),
                    "--challenger-artifact",
                    str(challenger_artifact),
                    "--reference-artifact",
                    str(reference_path),
                    "--out",
                    str(out_path),
                ]

                with (
                    patch.object(sys, "argv", argv),
                    self.assertRaisesRegex(SystemExit, "missing required fields"),
                ):
                    suite_module.main()

    def test_run_forensic_suite_rejects_duplicate_shared_reference_canonical_states(
        self,
    ):
        import importlib

        from ml.alphazero_lite.build_forensic_references import canonical_state_key

        with patch.dict(os.environ, {"AZLITE_FORENSIC_SUITE_STUB": "1"}):
            from ml.alphazero_lite import run_forensic_suite as suite_module

            importlib.reload(suite_module)

        state = self.suite_rows()[0]["state"]
        canonical_state = canonical_state_key(state)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            suite_path = tmp_path / "suite.json"
            reference_path = tmp_path / "reference_moves.json"
            out_path = tmp_path / "report_out.json"
            current_artifact = tmp_path / "current-artifact"
            challenger_artifact = tmp_path / "challenger-artifact"
            suite_path.write_text(json.dumps(self.suite_rows()), encoding="utf-8")
            current_artifact.write_text("current", encoding="utf-8")
            challenger_artifact.write_text("challenger", encoding="utf-8")
            reference_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_forensic_references_v1",
                        "reference": {
                            "policy_simulations": 1200,
                            "value_simulations": 1800,
                            "sample_seeds": [2100],
                        },
                        "rows": [
                            {
                                "id": "capture_available-016",
                                "canonical_state": canonical_state,
                                "state": state,
                                "reference_move": 3,
                                "teacher_value": 0.4183,
                                "reference_unstable": False,
                                "observed_reference_moves": [3],
                                "seed_samples": [
                                    {
                                        "seed": 2100,
                                        "reference_move": 3,
                                        "teacher_value": 0.4183,
                                    }
                                ],
                            },
                            {
                                "id": "capture_available-099",
                                "canonical_state": canonical_state,
                                "state": state,
                                "reference_move": 3,
                                "teacher_value": 0.4183,
                                "reference_unstable": False,
                                "observed_reference_moves": [3],
                                "seed_samples": [
                                    {
                                        "seed": 2100,
                                        "reference_move": 3,
                                        "teacher_value": 0.4183,
                                    }
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "run_forensic_suite.py",
                "--suite",
                str(suite_path),
                "--current-artifact",
                str(current_artifact),
                "--challenger-artifact",
                str(challenger_artifact),
                "--reference-artifact",
                str(reference_path),
                "--out",
                str(out_path),
            ]

            with (
                patch.dict(os.environ, {"AZLITE_FORENSIC_SUITE_STUB": "1"}),
                patch.object(sys, "argv", argv),
                self.assertRaisesRegex(SystemExit, "duplicate canonical_state"),
            ):
                suite_module.main()

    def test_run_forensic_suite_rejects_shared_reference_canonical_state_mismatch(self):
        import importlib

        from ml.alphazero_lite.build_forensic_references import canonical_state_key

        with patch.dict(os.environ, {"AZLITE_FORENSIC_SUITE_STUB": "1"}):
            from ml.alphazero_lite import run_forensic_suite as suite_module

            importlib.reload(suite_module)

        state = self.suite_rows()[0]["state"]
        mismatched_state = dict(state)
        mismatched_state["player_store"] = state["player_store"] + 1

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            suite_path = tmp_path / "suite.json"
            reference_path = tmp_path / "reference_moves.json"
            out_path = tmp_path / "report_out.json"
            current_artifact = tmp_path / "current-artifact"
            challenger_artifact = tmp_path / "challenger-artifact"
            suite_path.write_text(json.dumps(self.suite_rows()), encoding="utf-8")
            current_artifact.write_text("current", encoding="utf-8")
            challenger_artifact.write_text("challenger", encoding="utf-8")
            reference_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_forensic_references_v1",
                        "reference": {
                            "policy_simulations": 1200,
                            "value_simulations": 1800,
                            "sample_seeds": [2100],
                        },
                        "rows": [
                            {
                                "id": "capture_available-016",
                                "canonical_state": canonical_state_key(state),
                                "state": mismatched_state,
                                "reference_move": 3,
                                "teacher_value": 0.4183,
                                "reference_unstable": False,
                                "observed_reference_moves": [3],
                                "seed_samples": [
                                    {
                                        "seed": 2100,
                                        "reference_move": 3,
                                        "teacher_value": 0.4183,
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "run_forensic_suite.py",
                "--suite",
                str(suite_path),
                "--current-artifact",
                str(current_artifact),
                "--challenger-artifact",
                str(challenger_artifact),
                "--reference-artifact",
                str(reference_path),
                "--out",
                str(out_path),
            ]

            with (
                patch.dict(os.environ, {"AZLITE_FORENSIC_SUITE_STUB": "1"}),
                patch.object(sys, "argv", argv),
                self.assertRaisesRegex(
                    SystemExit, "canonical_state.*does not match state"
                ),
            ):
                suite_module.main()

    def test_run_forensic_suite_keeps_regret_summaries_for_rows_with_child_stats(self):
        import importlib

        from ml.alphazero_lite import build_forensic_references as reference_module

        with patch.dict(os.environ, {"AZLITE_FORENSIC_SUITE_STUB": "1"}):
            from ml.alphazero_lite import run_forensic_suite as suite_module

            importlib.reload(suite_module)

        state = self.suite_rows()[0]["state"]
        second_state = {
            "player_pits": [1, 5, 0, 6, 2, 7],
            "opponent_pits": [4, 3, 5, 0, 1, 6],
            "player_store": 3,
            "opponent_store": 2,
            "current_player": 0,
        }
        suite_rows = self.suite_rows() + [
            {
                "id": "sparse_endgame-101",
                "state": second_state,
                "side_to_move": 0,
                "legal_moves": [0, 1, 3, 4, 5],
                "phase": "endgame",
                "bucket": "sparse_endgame",
                "tags": ["sparse_endgame", "generated", "ply_18"],
                "source": "generated",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            suite_path = tmp_path / "suite.json"
            reference_path = tmp_path / "reference_moves.json"
            out_path = tmp_path / "report_out.json"
            current_artifact = tmp_path / "current-artifact"
            challenger_artifact = tmp_path / "challenger-artifact"
            suite_path.write_text(json.dumps(suite_rows), encoding="utf-8")
            current_artifact.write_text("current", encoding="utf-8")
            challenger_artifact.write_text("challenger", encoding="utf-8")

            reference_artifact = {
                "schema": "azlite_forensic_references_v1",
                "reference": {
                    "policy_simulations": 1200,
                    "value_simulations": 1800,
                    "sample_seeds": [2100],
                },
                "rows": [
                    {
                        "id": "capture_available-016",
                        "canonical_state": reference_module.canonical_state_key(state),
                        "state": state,
                        "reference_move": 3,
                        "teacher_value": 0.4183,
                        "reference_unstable": False,
                        "observed_reference_moves": [3],
                        "seed_samples": [
                            {"seed": 2100, "reference_move": 3, "teacher_value": 0.4183}
                        ],
                        "child_stats": [
                            {"move": 3, "visits": 10, "win_rate": 0.7},
                            {"move": 1, "visits": 8, "win_rate": 0.3},
                        ],
                    },
                    {
                        "id": "sparse_endgame-101",
                        "canonical_state": reference_module.canonical_state_key(
                            second_state
                        ),
                        "state": second_state,
                        "reference_move": 4,
                        "teacher_value": 0.2,
                        "reference_unstable": False,
                        "observed_reference_moves": [4],
                        "seed_samples": [
                            {"seed": 2100, "reference_move": 4, "teacher_value": 0.2}
                        ],
                    },
                ],
            }
            reference_path.write_text(json.dumps(reference_artifact), encoding="utf-8")

            def fake_evaluate_artifact_position(*, state, **_kwargs):
                if state == second_state:
                    return {"selected_move": 4, "value": 0.2}
                return {"selected_move": 1, "value": 0.15}

            argv = [
                "run_forensic_suite.py",
                "--suite",
                str(suite_path),
                "--current-artifact",
                str(current_artifact),
                "--challenger-artifact",
                str(challenger_artifact),
                "--reference-artifact",
                str(reference_path),
                "--out",
                str(out_path),
            ]

            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    suite_module,
                    "ArtifactEvaluator",
                    side_effect=lambda path: {"artifact": str(path)},
                ),
                patch.object(
                    suite_module, "build_eval_search_options", return_value={}
                ),
                patch.object(
                    suite_module,
                    "evaluate_artifact_position",
                    side_effect=fake_evaluate_artifact_position,
                ),
            ):
                suite_module.main()

            report = json.loads(out_path.read_text(encoding="utf-8"))

        challenger_overall = report["systems"]["challenger"]["overall"]
        challenger_buckets = report["systems"]["challenger"]["buckets"]
        matrix_buckets = report["buckets"]

        self.assertEqual(0.4, challenger_overall["average_regret"])
        self.assertEqual(0.5, challenger_overall["blunder_rate"])
        self.assertEqual(0.4, challenger_buckets["capture_available"]["average_regret"])
        self.assertEqual(1.0, challenger_buckets["capture_available"]["blunder_rate"])
        self.assertIsNone(challenger_buckets["sparse_endgame"]["average_regret"])
        self.assertIsNone(challenger_buckets["sparse_endgame"]["blunder_rate"])
        self.assertEqual(
            0.4,
            matrix_buckets["capture_available"]["systems"]["challenger"][
                "average_regret"
            ],
        )
        self.assertEqual(
            1.0,
            matrix_buckets["capture_available"]["systems"]["challenger"][
                "blunder_rate"
            ],
        )
        self.assertIsNone(
            matrix_buckets["sparse_endgame"]["systems"]["challenger"]["average_regret"]
        )
        self.assertIsNone(
            matrix_buckets["sparse_endgame"]["systems"]["challenger"]["blunder_rate"]
        )


if __name__ == "__main__":
    unittest.main()
