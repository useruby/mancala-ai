import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from ml.alphazero_lite.forensic_suite import (
    REQUIRED_BUCKETS,
    ForensicPosition,
    centered_value_from_probability,
    load_suite,
    summarize_bucket,
)
from ml.alphazero_lite import build_forensic_suite
from ml.alphazero_lite import run_forensic_suite


PYTHON_BIN = Path(sys.executable)


class ForensicSuiteTest(unittest.TestCase):
    def write_suite(self, rows: list[dict]) -> Path:
        temp_dir = tempfile.TemporaryDirectory(prefix="azlite-forensic-suite-")
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "suite.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        return path

    def test_load_suite_rejects_duplicate_canonical_states(self):
        duplicate_state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        path = self.write_suite(
            [
                {
                    "id": "opening-a",
                    "state": duplicate_state,
                    "side_to_move": 0,
                    "legal_moves": [0, 1, 2, 3, 4, 5],
                    "phase": "opening",
                    "bucket": "opening_plies_1_8",
                    "tags": ["opening"],
                    "source": "seed",
                },
                {
                    "id": "opening-b",
                    "state": duplicate_state,
                    "side_to_move": 0,
                    "legal_moves": [0, 1, 2, 3, 4, 5],
                    "phase": "opening",
                    "bucket": "opening_plies_1_8",
                    "tags": ["opening"],
                    "source": "seed",
                },
            ]
        )

        with self.assertRaisesRegex(ValueError, "duplicate canonical state"):
            load_suite(path)

    def test_load_suite_rejects_empty_legal_moves(self):
        path = self.write_suite(
            [
                {
                    "id": "missing-legal-moves",
                    "state": {
                        "player_pits": [0, 0, 1, 0, 0, 0],
                        "opponent_pits": [4, 4, 4, 4, 4, 4],
                        "player_store": 18,
                        "opponent_store": 12,
                        "current_player": 0,
                    },
                    "side_to_move": 0,
                    "legal_moves": [],
                    "phase": "late",
                    "bucket": "sparse_endgame",
                    "tags": ["late"],
                    "source": "seed",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "legal_moves"):
            load_suite(path)

    def test_load_suite_rejects_empty_tags(self):
        path = self.write_suite(
            [
                {
                    "id": "missing-tags",
                    "state": {
                        "player_pits": [0, 0, 1, 0, 0, 0],
                        "opponent_pits": [4, 4, 4, 4, 4, 4],
                        "player_store": 18,
                        "opponent_store": 12,
                        "current_player": 0,
                    },
                    "side_to_move": 0,
                    "legal_moves": [2],
                    "phase": "late",
                    "bucket": "sparse_endgame",
                    "tags": [],
                    "source": "seed",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "tags"):
            load_suite(path)

    def test_load_suite_rejects_out_of_range_side_to_move(self):
        path = self.write_suite(
            [
                {
                    "id": "bad-side-to-move",
                    "state": {
                        "player_pits": [4, 4, 4, 4, 4, 4],
                        "opponent_pits": [4, 4, 4, 4, 4, 4],
                        "player_store": 0,
                        "opponent_store": 0,
                        "current_player": 0,
                    },
                    "side_to_move": 2,
                    "legal_moves": [0, 1, 2, 3, 4, 5],
                    "phase": "opening",
                    "bucket": "opening_plies_1_8",
                    "tags": ["opening"],
                    "source": "seed",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "side_to_move"):
            load_suite(path)

    def test_load_suite_rejects_side_to_move_mismatch(self):
        path = self.write_suite(
            [
                {
                    "id": "mismatch-side-to-move",
                    "state": {
                        "player_pits": [4, 4, 4, 4, 4, 4],
                        "opponent_pits": [4, 4, 4, 4, 4, 4],
                        "player_store": 0,
                        "opponent_store": 0,
                        "current_player": 0,
                    },
                    "side_to_move": 1,
                    "legal_moves": [0, 1, 2, 3, 4, 5],
                    "phase": "opening",
                    "bucket": "opening_plies_1_8",
                    "tags": ["opening"],
                    "source": "seed",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "side_to_move"):
            load_suite(path)

    def test_load_suite_rejects_out_of_range_legal_move(self):
        path = self.write_suite(
            [
                {
                    "id": "bad-legal-range",
                    "state": {
                        "player_pits": [4, 4, 4, 4, 4, 4],
                        "opponent_pits": [4, 4, 4, 4, 4, 4],
                        "player_store": 0,
                        "opponent_store": 0,
                        "current_player": 0,
                    },
                    "side_to_move": 0,
                    "legal_moves": [0, 6],
                    "phase": "opening",
                    "bucket": "opening_plies_1_8",
                    "tags": ["opening"],
                    "source": "seed",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "legal_moves"):
            load_suite(path)

    def test_load_suite_rejects_duplicate_legal_moves(self):
        path = self.write_suite(
            [
                {
                    "id": "duplicate-legal-moves",
                    "state": {
                        "player_pits": [4, 4, 4, 4, 4, 4],
                        "opponent_pits": [4, 4, 4, 4, 4, 4],
                        "player_store": 0,
                        "opponent_store": 0,
                        "current_player": 0,
                    },
                    "side_to_move": 0,
                    "legal_moves": [0, 0, 1],
                    "phase": "opening",
                    "bucket": "opening_plies_1_8",
                    "tags": ["opening"],
                    "source": "seed",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "legal_moves"):
            load_suite(path)

    def test_load_suite_loads_valid_rows(self):
        path = self.write_suite(
            [
                {
                    "id": "opening-a",
                    "state": {
                        "player_pits": [4, 4, 4, 4, 4, 4],
                        "opponent_pits": [4, 4, 4, 4, 4, 4],
                        "player_store": 0,
                        "opponent_store": 0,
                        "current_player": 0,
                    },
                    "side_to_move": 0,
                    "legal_moves": [0, 1, 2, 3, 4, 5],
                    "phase": "opening",
                    "bucket": "opening_plies_1_8",
                    "tags": ["opening", "seed"],
                    "source": "seed",
                }
            ]
        )

        suite = load_suite(path)

        self.assertEqual(1, len(suite))
        self.assertIsInstance(suite[0], ForensicPosition)
        self.assertEqual("opening-a", suite[0].id)
        self.assertEqual((0, 1, 2, 3, 4, 5), suite[0].legal_moves)
        self.assertEqual(("opening", "seed"), suite[0].tags)
        self.assertEqual(
            '{"current_player":0,"opponent_pits":[4,4,4,4,4,4],"opponent_store":0,"player_pits":[4,4,4,4,4,4],"player_store":0}',
            suite[0].canonical_key,
        )

    def write_basic_artifact(self, directory: Path, *, policy_bias: list[float] | None = None, value_bias: float = 0.0) -> Path:
        artifact_dir = directory
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "metadata.json").write_text(
            json.dumps({"input_encoding": "kalah_v1", "architecture": {"model_type": "mlp_v1"}}),
            encoding="utf-8",
        )
        (artifact_dir / "weights.json").write_text(
            json.dumps(
                {
                    "w1": [[0.1, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0]] + [[0.0, 0.0, 0.0, 0.0]] * 13,
                    "b1": [0.0, 0.0, 0.0, 0.0],
                    "w2": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                    "b2": [0.0, 0.0, 0.0, 0.0],
                    "w_policy": [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    ],
                    "b_policy": policy_bias or [0.0, 0.0, 0.0, 0.0, -1.0, -1.0],
                    "w_value": [[0.25], [0.25], [0.25], [0.25]],
                    "b_value": [value_bias],
                }
            ),
            encoding="utf-8",
        )
        return artifact_dir

    def test_summarize_bucket_reports_agreement_regret_blunder_rate_and_value_calibration(self):
        summary = summarize_bucket(
            [
                {
                    "agrees_top1": True,
                    "regret": 0.0,
                    "value_error": 0.1,
                },
                {
                    "agrees_top1": False,
                    "regret": 0.25,
                    "value_error": 0.3,
                },
                {
                    "agrees_top1": False,
                    "regret": 0.5,
                    "value_error": None,
                },
            ]
        )

        self.assertEqual(3, summary["positions"])
        self.assertEqual(0.3333, summary["top1_agreement"])
        self.assertEqual(0.25, summary["average_regret"])
        self.assertEqual(0.6667, summary["blunder_rate"])
        self.assertEqual(0.2, summary["value_calibration_mae"])

    def test_centered_value_from_probability_handles_win_draw_loss_cases(self):
        self.assertEqual(1.0, centered_value_from_probability(1.0))
        self.assertEqual(0.0, centered_value_from_probability(0.5))
        self.assertEqual(-1.0, centered_value_from_probability(0.0))

    def test_stub_reference_clamps_teacher_value_range(self):
        summary = run_forensic_suite._stub_reference(index=0, policy_simulations=7, value_simulations=1800)

        self.assertGreaterEqual(summary["teacher_value"], -1.0)
        self.assertLessEqual(summary["teacher_value"], 1.0)

    def test_cli_writes_deterministic_numeric_report_with_teacher_reference(self):
        with tempfile.TemporaryDirectory(prefix="azlite-forensic-report-") as tmp:
            temp_root = Path(tmp)
            suite_path = self.write_suite(
                [
                    {
                        "id": "opening-a",
                        "state": {
                            "player_pits": [4, 4, 4, 4, 4, 4],
                            "opponent_pits": [4, 4, 4, 4, 4, 4],
                            "player_store": 0,
                            "opponent_store": 0,
                            "current_player": 0,
                        },
                        "side_to_move": 0,
                        "legal_moves": [0, 1, 2, 3, 4, 5],
                        "phase": "opening",
                        "bucket": "opening_plies_1_8",
                        "tags": ["opening", "seed"],
                        "source": "seed",
                    },
                    {
                        "id": "loss-a",
                        "state": {
                            "player_pits": [0, 1, 0, 0, 2, 0],
                            "opponent_pits": [0, 0, 3, 0, 0, 1],
                            "player_store": 17,
                            "opponent_store": 18,
                            "current_player": 0,
                        },
                        "side_to_move": 0,
                        "legal_moves": [1, 4],
                        "phase": "late",
                        "bucket": "incumbent_proxy_disagreement",
                        "tags": ["late", "generated"],
                        "source": "generated",
                    },
                ]
            )
            current_artifact = self.write_basic_artifact(temp_root / "current")
            challenger_artifact = self.write_basic_artifact(temp_root / "challenger")
            out_path = temp_root / "report.json"
            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "ml/alphazero_lite/run_forensic_suite.py",
                    "--suite",
                    str(suite_path),
                    "--current-artifact",
                    str(current_artifact),
                    "--challenger-artifact",
                    str(challenger_artifact),
                    "--mcts-simulations",
                    "7",
                    "--teacher-simulations",
                    "13",
                    "--artifact-simulations",
                    "5",
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "AZLITE_FORENSIC_SUITE_STUB": "1"},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertTrue(report["stub"])
            self.assertEqual("azlite_forensic_suite_v1", report["schema"])
            self.assertEqual(2, report["positions"])
            self.assertEqual(7, report["reference"]["policy_reference"]["simulations"])
            self.assertEqual(13, report["reference"]["value_reference"]["simulations"])
            self.assertIn("stub mode", result.stderr.lower())

            self.assertEqual(
                {
                    "positions": 2,
                    "top1_agreement": 0.5,
                    "average_regret": 0.2,
                    "blunder_rate": 0.5,
                    "value_calibration_mae": 0.2,
                },
                report["systems"]["current"]["overall"],
            )
            self.assertEqual(
                {
                    "positions": 2,
                    "top1_agreement": 0.5,
                    "average_regret": 0.4,
                    "blunder_rate": 0.5,
                    "value_calibration_mae": 0.5,
                },
                report["systems"]["challenger"]["overall"],
            )
            self.assertEqual(
                {
                    "positions": 1,
                    "top1_agreement": 1.0,
                    "average_regret": 0.0,
                    "blunder_rate": 0.0,
                    "value_calibration_mae": 0.0,
                },
                report["systems"]["current"]["buckets"]["opening_plies_1_8"],
            )
            self.assertEqual(
                {
                    "positions": 1,
                    "top1_agreement": 0.0,
                    "average_regret": 0.4,
                    "blunder_rate": 1.0,
                    "value_calibration_mae": 0.4,
                },
                report["systems"]["current"]["buckets"]["incumbent_proxy_disagreement"],
            )
            self.assertEqual(1, report["buckets"]["opening_plies_1_8"]["positions"])
            self.assertEqual(1.0, report["buckets"]["opening_plies_1_8"]["systems"]["current"]["top1_agreement"])
            self.assertEqual(0.8, report["buckets"]["opening_plies_1_8"]["systems"]["challenger"]["average_regret"])
            self.assertEqual(1.0, report["buckets"]["opening_plies_1_8"]["systems"]["challenger"]["blunder_rate"])
            self.assertEqual(0.6, report["buckets"]["opening_plies_1_8"]["systems"]["challenger"]["value_calibration_mae"])
            self.assertEqual(2, len(report["systems"]["current"]["rows"]))

    def test_cli_stub_mode_handles_more_than_two_positions(self):
        with tempfile.TemporaryDirectory(prefix="azlite-forensic-report-stub-") as tmp:
            temp_root = Path(tmp)
            suite_path = self.write_suite(
                [
                    {
                        "id": "opening-a",
                        "state": {
                            "player_pits": [4, 4, 4, 4, 4, 4],
                            "opponent_pits": [4, 4, 4, 4, 4, 4],
                            "player_store": 0,
                            "opponent_store": 0,
                            "current_player": 0,
                        },
                        "side_to_move": 0,
                        "legal_moves": [0, 1, 2, 3, 4, 5],
                        "phase": "opening",
                        "bucket": "opening_plies_1_8",
                        "tags": ["opening", "seed"],
                        "source": "seed",
                    },
                    {
                        "id": "loss-a",
                        "state": {
                            "player_pits": [0, 1, 0, 0, 2, 0],
                            "opponent_pits": [0, 0, 3, 0, 0, 1],
                            "player_store": 17,
                            "opponent_store": 18,
                            "current_player": 0,
                        },
                        "side_to_move": 0,
                        "legal_moves": [1, 4],
                        "phase": "late",
                        "bucket": "incumbent_proxy_disagreement",
                        "tags": ["late", "generated"],
                        "source": "generated",
                    },
                    {
                        "id": "capture-a",
                        "state": {
                            "player_pits": [0, 0, 1, 0, 4, 0],
                            "opponent_pits": [0, 2, 0, 3, 0, 1],
                            "player_store": 20,
                            "opponent_store": 12,
                            "current_player": 0,
                        },
                        "side_to_move": 0,
                        "legal_moves": [2, 4],
                        "phase": "late",
                        "bucket": "capture_available",
                        "tags": ["late", "generated"],
                        "source": "generated",
                    },
                ]
            )
            current_artifact = self.write_basic_artifact(temp_root / "current")
            challenger_artifact = self.write_basic_artifact(temp_root / "challenger")
            out_path = temp_root / "report.json"

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "ml/alphazero_lite/run_forensic_suite.py",
                    "--suite",
                    str(suite_path),
                    "--current-artifact",
                    str(current_artifact),
                    "--challenger-artifact",
                    str(challenger_artifact),
                    "--mcts-simulations",
                    "7",
                    "--teacher-simulations",
                    "13",
                    "--artifact-simulations",
                    "5",
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "AZLITE_FORENSIC_SUITE_STUB": "1"},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertTrue(report["stub"])
            self.assertEqual(3, report["positions"])
            self.assertEqual(3, len(report["systems"]["current"]["rows"]))
            self.assertEqual(3, len(report["systems"]["challenger"]["rows"]))
            legal_moves_by_id = {row.id: set(row.legal_moves) for row in load_suite(suite_path)}
            for system_name in ("current", "challenger"):
                for row in report["systems"][system_name]["rows"]:
                    self.assertIn(row["selected_move"], legal_moves_by_id[row["id"]])

    def test_main_preloads_artifact_evaluators_once_per_system(self):
        suite_path = self.write_suite(
            [
                {
                    "id": "opening-0",
                    "state": {
                        "player_pits": [4, 4, 4, 4, 4, 4],
                        "opponent_pits": [4, 4, 4, 4, 4, 4],
                        "player_store": 0,
                        "opponent_store": 0,
                        "current_player": 0,
                    },
                    "side_to_move": 0,
                    "legal_moves": [0, 1, 2, 3, 4, 5],
                    "phase": "opening",
                    "bucket": "opening_plies_1_8",
                    "tags": ["opening", "seed"],
                    "source": "seed",
                },
                {
                    "id": "opening-1",
                    "state": {
                        "player_pits": [0, 2, 0, 0, 8, 7],
                        "opponent_pits": [0, 9, 0, 7, 6, 5],
                        "player_store": 3,
                        "opponent_store": 1,
                        "current_player": 0,
                    },
                    "side_to_move": 0,
                    "legal_moves": [1, 4, 5],
                    "phase": "mid",
                    "bucket": "incumbent_proxy_disagreement",
                    "tags": ["incumbent_proxy_disagreement", "seed"],
                    "source": "seed",
                },
                {
                    "id": "opening-2",
                    "state": {
                        "player_pits": [1, 0, 2, 0, 8, 7],
                        "opponent_pits": [0, 9, 6, 6, 0, 5],
                        "player_store": 3,
                        "opponent_store": 1,
                        "current_player": 0,
                    },
                    "side_to_move": 0,
                    "legal_moves": [0, 2, 4, 5],
                    "phase": "mid",
                    "bucket": "incumbent_proxy_disagreement",
                    "tags": ["incumbent_proxy_disagreement", "generated"],
                    "source": "generated",
                },
            ]
        )

        with tempfile.TemporaryDirectory(prefix="azlite-forensic-preload-") as tmp:
            temp_root = Path(tmp)
            current_artifact = self.write_basic_artifact(temp_root / "current")
            challenger_artifact = self.write_basic_artifact(temp_root / "challenger")
            out_path = temp_root / "report.json"
            parse_args = Namespace(
                suite=str(suite_path),
                current_artifact=str(current_artifact),
                challenger_artifact=str(challenger_artifact),
                mcts_simulations=7,
                teacher_simulations=13,
                artifact_simulations=5,
                c_puct=1.25,
                seed=42,
                out=str(out_path),
            )
            created_evaluators: list[str] = []
            evaluate_calls: list[dict] = []

            class FakeEvaluator:
                def __init__(self, artifact_dir):
                    created_evaluators.append(str(artifact_dir))
                    self.artifact_dir = str(artifact_dir)

            def fake_evaluate(**kwargs):
                evaluate_calls.append(kwargs)
                return {
                    "selected_move": 0,
                    "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "value": 0.0,
                    "child_stats": [],
                    "visits": [],
                }

            with mock.patch("ml.alphazero_lite.run_forensic_suite.parse_args", return_value=parse_args), mock.patch(
                "ml.alphazero_lite.run_forensic_suite.ArtifactEvaluator",
                FakeEvaluator,
                create=True,
            ), mock.patch(
                "ml.alphazero_lite.run_forensic_suite.run_reference",
                return_value={
                    "selected_move": 0,
                    "child_stats": [{"move": 0, "win_rate": 1.0, "visits": 1}],
                    "teacher_value": 0.0,
                },
            ), mock.patch(
                "ml.alphazero_lite.run_forensic_suite.evaluate_artifact_position",
                side_effect=fake_evaluate,
            ):
                run_forensic_suite.main()

            self.assertEqual([str(current_artifact), str(challenger_artifact)], created_evaluators)
            self.assertEqual(6, len(evaluate_calls))
            self.assertTrue(all("evaluator" in call for call in evaluate_calls))
            self.assertEqual(2, len({id(call["evaluator"]) for call in evaluate_calls}))


class ForensicSuiteBuilderTest(unittest.TestCase):
    def test_choose_bucket_does_not_use_fake_loss_fallback_for_unclassified_state(self):
        state = {
            "player_pits": [0, 2, 0, 0, 8, 7],
            "opponent_pits": [0, 9, 0, 7, 6, 5],
            "player_store": 3,
            "opponent_store": 1,
            "current_player": 0,
        }

        self.assertIsNone(build_forensic_suite._choose_bucket(state, ply=9))

    def test_proxy_disagreement_bucket_uses_disagreement_heuristic(self):
        state = {
            "player_pits": [0, 2, 0, 0, 8, 7],
            "opponent_pits": [0, 9, 0, 7, 6, 5],
            "player_store": 3,
            "opponent_store": 1,
            "current_player": 0,
        }

        with mock.patch(
            "ml.alphazero_lite.build_forensic_suite._proxy_root_summary",
            side_effect=[
                {
                    "selected_move": 1,
                    "child_stats": [
                        {"move": 1, "win_rate": 0.8},
                        {"move": 4, "win_rate": 0.2},
                    ],
                },
                {"selected_move": 4, "child_stats": [{"move": 4, "win_rate": 0.2}]},
                {
                    "selected_move": 1,
                    "child_stats": [
                        {"move": 1, "win_rate": 0.75},
                        {"move": 4, "win_rate": 0.25},
                    ],
                },
            ],
        ):
            self.assertTrue(build_forensic_suite._is_proxy_disagreement_candidate(state, ply=9))

        with mock.patch(
            "ml.alphazero_lite.build_forensic_suite._proxy_root_summary",
            side_effect=[
                {"selected_move": 1, "child_stats": [{"move": 1, "win_rate": 0.8}]},
                {"selected_move": 1, "child_stats": [{"move": 1, "win_rate": 0.8}]},
            ],
        ):
            self.assertFalse(build_forensic_suite._is_proxy_disagreement_candidate(state, ply=9))

    def test_proxy_disagreement_scores_challenger_move_from_challenger_summary(self):
        state = {
            "player_pits": [0, 2, 0, 0, 8, 7],
            "opponent_pits": [0, 9, 0, 7, 6, 5],
            "player_store": 3,
            "opponent_store": 1,
            "current_player": 0,
        }

        with mock.patch(
            "ml.alphazero_lite.build_forensic_suite._proxy_root_summary",
            side_effect=[
                {
                    "selected_move": 1,
                    "child_stats": [
                        {"move": 1, "win_rate": 0.8},
                    ],
                },
                {
                    "selected_move": 4,
                    "child_stats": [
                        {"move": 4, "win_rate": 0.7},
                    ],
                },
            ],
        ):
            self.assertTrue(build_forensic_suite._is_proxy_disagreement_candidate(state, ply=9))

    def test_choose_bucket_uses_current_player_perspective_for_player_one_state(self):
        state = {
            "player_pits": [1, 1, 1, 1, 2, 4],
            "opponent_pits": [1, 1, 1, 1, 2, 1],
            "player_store": 10,
            "opponent_store": 10,
            "current_player": 1,
        }

        self.assertEqual("early_extra_turn", build_forensic_suite._choose_bucket(state, ply=9))

    def test_checked_in_suite_covers_all_required_buckets(self):
        suite = load_suite(Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"))
        buckets = {row.bucket for row in suite}
        ids = [row.id for row in suite]

        self.assertEqual(REQUIRED_BUCKETS, buckets)
        self.assertGreaterEqual(len(suite), 200)
        self.assertEqual(len(ids), len(set(ids)))

    def test_builder_writes_suite_with_non_empty_loss_bucket(self):
        with tempfile.TemporaryDirectory(prefix="azlite-build-forensic-") as tmp:
            output_path = Path(tmp) / "suite.json"
            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "ml/alphazero_lite/build_forensic_suite.py",
                    "--output",
                    str(output_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            suite = load_suite(output_path)
            self.assertTrue(any(row.bucket == "incumbent_proxy_disagreement" for row in suite))

    def test_builder_has_no_external_tmp_loss_source_dependency(self):
        self.assertFalse(hasattr(build_forensic_suite, "LOSS_SOURCE"))

        rows = build_forensic_suite.build_rows()

        self.assertTrue(rows)
        self.assertTrue(all(row["source"] in {"seed", "generated"} for row in rows))
        self.assertFalse(any("harvested" in row["tags"] for row in rows))

    def test_required_buckets_use_proxy_disagreement_bucket(self):
        self.assertIn("incumbent_proxy_disagreement", REQUIRED_BUCKETS)
        self.assertNotIn("challenger_loss_vs_current", REQUIRED_BUCKETS)

    def test_builder_uses_repo_owned_proxy_artifact(self):
        self.assertEqual(
            build_forensic_suite.ROOT_DIR / "ml/alphazero_lite/fixtures/incumbent_forensic_proxy_current",
            build_forensic_suite.PROXY_ARTIFACT,
        )

    def test_seed_rows_start_bucket_ids_at_001(self):
        rows = build_forensic_suite._seed_rows(build_forensic_suite._candidate_states())
        seed_rows = [row for row in rows if row["bucket"] != "opening_plies_1_8"]

        self.assertTrue(seed_rows)
        self.assertTrue(seed_rows[0]["id"].endswith("-001"))

    def test_build_rows_reuses_candidate_state_enumeration(self):
        call_count = 0
        candidates = build_forensic_suite._candidate_states()

        def fake_candidate_states():
            nonlocal call_count
            call_count += 1
            return candidates

        with mock.patch("ml.alphazero_lite.build_forensic_suite._candidate_states", side_effect=fake_candidate_states):
            build_forensic_suite.build_rows()

        self.assertEqual(1, call_count)

    def test_checked_in_fixture_matches_builder_output_exactly(self):
        fixture_path = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
        fixture_text = fixture_path.read_text(encoding="utf-8")
        built_text = build_forensic_suite.build_fixture_text()
        suite = load_suite(fixture_path)

        self.assertEqual(fixture_text.strip(), built_text.strip())
        self.assertGreaterEqual(len(suite), 200)
        self.assertEqual(REQUIRED_BUCKETS, {row.bucket for row in suite})


if __name__ == "__main__":
    unittest.main()
