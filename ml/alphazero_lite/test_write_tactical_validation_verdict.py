import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite import check_bucket_promotion_gate


PYTHON_BIN = Path(sys.executable)


def forensic_report(
    *,
    overall_average_regret: float = 0.08,
    overall_top1_agreement: float = 0.7,
    buckets: dict[str, dict[str, float | int | list[float]]] | None = None,
) -> dict:
    bucket_defaults = {
        bucket: {
            "positions": 20,
            "average_regret": 0.08,
            "top1_agreement": 0.7,
            "rows": [0.05] * 20,
        }
        for bucket in check_bucket_promotion_gate.REQUIRED_BUCKETS
    }
    bucket_defaults["opening_plies_1_8"]["positions"] = 40
    bucket_defaults["opening_plies_1_8"]["rows"] = [0.05] * 40

    for bucket, overrides in (buckets or {}).items():
        bucket_defaults[bucket].update(overrides)

    challenger_rows = []
    bucket_matrix = {}
    challenger_buckets = {}
    for bucket, metrics in bucket_defaults.items():
        row_values = [float(value) for value in metrics["rows"]]
        rows = [{"bucket": bucket, "regret": value} for value in row_values]
        challenger_rows.extend(rows)
        summary = {
            "positions": int(metrics["positions"]),
            "top1_agreement": float(metrics["top1_agreement"]),
            "average_regret": float(metrics["average_regret"]),
        }
        challenger_buckets[bucket] = summary
        bucket_matrix[bucket] = {
            "positions": int(metrics["positions"]),
            "systems": {"challenger": summary},
        }

    return {
        "schema": "azlite_forensic_suite_v1",
        "reference": {
            "kind": "shared_artifact",
            "artifact_path": "inputs/reference_moves.json",
            "shared_reference": {
                "policy_simulations": 1800,
                "value_simulations": 1800,
                "sample_seeds": [3041, 4041, 5041],
            },
        },
        "systems": {
            "challenger": {
                "overall": {
                    "positions": len(challenger_rows),
                    "top1_agreement": float(overall_top1_agreement),
                    "average_regret": float(overall_average_regret),
                },
                "buckets": challenger_buckets,
                "rows": challenger_rows,
            }
        },
        "buckets": bucket_matrix,
    }


def regression_report(*, passed: bool) -> dict:
    return {
        "passed": passed,
        "artifact_path": "candidate",
        "positions_path": "test/fixtures/ai/superhuman_regression_positions.json",
        "results": [],
    }


def arena_report(*, passed: bool) -> dict:
    return {
        "schema": "arena_v1",
        "wins": 66,
        "losses": 42,
        "draws": 12,
        "games_played": 120,
        "promotion_decision": {"passed": passed},
    }


class TacticalValidationVerdictTest(unittest.TestCase):
    def test_verdict_passes_when_regression_and_bucket_checks_pass(self):
        importlib.import_module("ml.alphazero_lite.write_tactical_validation_verdict")

        baseline = forensic_report()
        candidate = forensic_report()
        bucket_gate = check_bucket_promotion_gate.evaluate_gate(baseline, candidate)

        with tempfile.TemporaryDirectory(
            prefix="azlite-tactical-validation-verdict-"
        ) as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run" / "final"
            selected_artifact = run_dir / "selected-candidate"
            selected_artifact.mkdir(parents=True, exist_ok=True)
            (selected_artifact / "model.npz").write_text("stub", encoding="utf-8")

            baseline_path = run_dir / "baseline_candidate_forensics.json"
            candidate_path = run_dir / "selected_candidate_forensics.json"
            bucket_gate_path = run_dir / "bucket_gate.json"
            regression_path = run_dir / "candidate_regression_suite.json"
            arena_path = run_dir / "arena_seed_1041.json"
            out_path = run_dir / "tactical_validation_verdict.json"

            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            bucket_gate_path.write_text(json.dumps(bucket_gate), encoding="utf-8")
            regression_path.write_text(
                json.dumps(regression_report(passed=True)), encoding="utf-8"
            )
            arena_path.write_text(
                json.dumps(arena_report(passed=True)), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.write_tactical_validation_verdict",
                    "--run-dir",
                    str(run_dir),
                    "--selected-artifact",
                    str(selected_artifact),
                    "--baseline-forensics",
                    str(baseline_path),
                    "--candidate-forensics",
                    str(candidate_path),
                    "--bucket-gate",
                    str(bucket_gate_path),
                    "--regression-report",
                    str(regression_path),
                    "--arena-report",
                    str(arena_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            verdict = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual("azlite_tactical_validation_verdict_v1", verdict["schema"])
            self.assertTrue(verdict["passed"])
            self.assertEqual("pass", verdict["verdict"])
            self.assertEqual([], verdict["failure_reasons"])
            self.assertEqual([], verdict["extra_adverse_deltas"])
            self.assertTrue(verdict["rubric"]["strict_binary"])
            self.assertEqual(str(run_dir), verdict["run_dir"])
            self.assertEqual(str(selected_artifact), verdict["selected_artifact"])
            self.assertEqual(str(baseline_path), verdict["baseline_forensics_path"])
            self.assertEqual(str(candidate_path), verdict["candidate_forensics_path"])
            self.assertEqual(str(bucket_gate_path), verdict["bucket_gate_path"])
            self.assertEqual(str(regression_path), verdict["regression_report_path"])
            self.assertEqual(str(arena_path), verdict["arena_report_path"])

    def test_verdict_fails_on_extra_adverse_delta_not_covered_by_bucket_gate(self):
        importlib.import_module("ml.alphazero_lite.write_tactical_validation_verdict")

        baseline = forensic_report(
            buckets={
                "high_imbalance": {
                    "average_regret": 0.12,
                    "rows": [0.10] * 20,
                }
            }
        )
        candidate = forensic_report(
            buckets={
                "high_imbalance": {
                    "average_regret": 0.121,
                    "rows": ([0.10] * 19) + [0.25],
                }
            }
        )
        bucket_gate = check_bucket_promotion_gate.evaluate_gate(baseline, candidate)
        self.assertTrue(bucket_gate["passed"])

        with tempfile.TemporaryDirectory(
            prefix="azlite-tactical-validation-verdict-"
        ) as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run" / "final"
            selected_artifact = run_dir / "selected-candidate"
            selected_artifact.mkdir(parents=True, exist_ok=True)
            (selected_artifact / "model.npz").write_text("stub", encoding="utf-8")

            baseline_path = run_dir / "baseline_candidate_forensics.json"
            candidate_path = run_dir / "selected_candidate_forensics.json"
            bucket_gate_path = run_dir / "bucket_gate.json"
            regression_path = run_dir / "candidate_regression_suite.json"
            arena_path = run_dir / "arena_seed_1041.json"
            out_path = run_dir / "tactical_validation_verdict.json"

            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            bucket_gate_path.write_text(json.dumps(bucket_gate), encoding="utf-8")
            regression_path.write_text(
                json.dumps(regression_report(passed=True)), encoding="utf-8"
            )
            arena_path.write_text(
                json.dumps(arena_report(passed=True)), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.write_tactical_validation_verdict",
                    "--run-dir",
                    str(run_dir),
                    "--selected-artifact",
                    str(selected_artifact),
                    "--baseline-forensics",
                    str(baseline_path),
                    "--candidate-forensics",
                    str(candidate_path),
                    "--bucket-gate",
                    str(bucket_gate_path),
                    "--regression-report",
                    str(regression_path),
                    "--arena-report",
                    str(arena_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(1, result.returncode, msg=result.stderr)
            verdict = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertFalse(verdict["passed"])
            self.assertEqual("fail", verdict["verdict"])
            self.assertIn("extra_adverse_delta_detected", verdict["failure_reasons"])
            self.assertEqual(1, len(verdict["extra_adverse_deltas"]))
            self.assertEqual(
                {
                    "id": "high_imbalance.blunder_rate_0_20",
                    "comparison": "max",
                    "baseline_value": 0.0,
                    "candidate_value": 0.05,
                },
                verdict["extra_adverse_deltas"][0],
            )
            self.assertTrue(verdict["bucket_gate"]["passed"])
            self.assertTrue(verdict["regression_report"]["passed"])
            self.assertTrue(verdict["arena_report"]["promotion_decision"]["passed"])

    def test_cli_writes_failure_artifact_for_malformed_input(self):
        importlib.import_module("ml.alphazero_lite.write_tactical_validation_verdict")

        baseline = forensic_report()
        candidate = forensic_report()
        bucket_gate = check_bucket_promotion_gate.evaluate_gate(baseline, candidate)

        with tempfile.TemporaryDirectory(
            prefix="azlite-tactical-validation-verdict-"
        ) as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run" / "final"
            selected_artifact = run_dir / "selected-candidate"
            selected_artifact.mkdir(parents=True, exist_ok=True)
            (selected_artifact / "model.npz").write_text("stub", encoding="utf-8")

            baseline_path = run_dir / "baseline_candidate_forensics.json"
            candidate_path = run_dir / "selected_candidate_forensics.json"
            bucket_gate_path = run_dir / "bucket_gate.json"
            regression_path = run_dir / "candidate_regression_suite.json"
            arena_path = run_dir / "arena_seed_1041.json"
            out_path = run_dir / "tactical_validation_verdict.json"

            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text("{not-json", encoding="utf-8")
            bucket_gate_path.write_text(json.dumps(bucket_gate), encoding="utf-8")
            regression_path.write_text(
                json.dumps(regression_report(passed=True)), encoding="utf-8"
            )
            arena_path.write_text(
                json.dumps(arena_report(passed=True)), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.write_tactical_validation_verdict",
                    "--run-dir",
                    str(run_dir),
                    "--selected-artifact",
                    str(selected_artifact),
                    "--baseline-forensics",
                    str(baseline_path),
                    "--candidate-forensics",
                    str(candidate_path),
                    "--bucket-gate",
                    str(bucket_gate_path),
                    "--regression-report",
                    str(regression_path),
                    "--arena-report",
                    str(arena_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(1, result.returncode)
            self.assertTrue(out_path.exists())
            verdict = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual("azlite_tactical_validation_verdict_v1", verdict["schema"])
            self.assertFalse(verdict["passed"])
            self.assertEqual("fail", verdict["verdict"])
            self.assertEqual(["input_validation_failed"], verdict["failure_reasons"])
            self.assertEqual("invalid_input", verdict["error"]["code"])
            self.assertEqual(str(candidate_path), verdict["error"]["path"])

    def test_cli_rejects_malformed_bucket_gate_checks_as_input_validation_failure(self):
        importlib.import_module("ml.alphazero_lite.write_tactical_validation_verdict")

        baseline = forensic_report()
        candidate = forensic_report()
        malformed_bucket_gate = {
            "schema": "azlite_bucket_promotion_gate_v1",
            "passed": True,
            "checks": [{"id": "capture_available.average_regret", "passed": True}],
        }

        with tempfile.TemporaryDirectory(
            prefix="azlite-tactical-validation-verdict-"
        ) as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run" / "final"
            selected_artifact = run_dir / "selected-candidate"
            selected_artifact.mkdir(parents=True, exist_ok=True)
            (selected_artifact / "model.npz").write_text("stub", encoding="utf-8")

            baseline_path = run_dir / "baseline_candidate_forensics.json"
            candidate_path = run_dir / "selected_candidate_forensics.json"
            bucket_gate_path = run_dir / "bucket_gate.json"
            regression_path = run_dir / "candidate_regression_suite.json"
            arena_path = run_dir / "arena_seed_1041.json"
            out_path = run_dir / "tactical_validation_verdict.json"

            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            bucket_gate_path.write_text(
                json.dumps(malformed_bucket_gate), encoding="utf-8"
            )
            regression_path.write_text(
                json.dumps(regression_report(passed=True)), encoding="utf-8"
            )
            arena_path.write_text(
                json.dumps(arena_report(passed=True)), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.write_tactical_validation_verdict",
                    "--run-dir",
                    str(run_dir),
                    "--selected-artifact",
                    str(selected_artifact),
                    "--baseline-forensics",
                    str(baseline_path),
                    "--candidate-forensics",
                    str(candidate_path),
                    "--bucket-gate",
                    str(bucket_gate_path),
                    "--regression-report",
                    str(regression_path),
                    "--arena-report",
                    str(arena_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(1, result.returncode)
            self.assertTrue(out_path.exists())
            verdict = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertFalse(verdict["passed"])
            self.assertEqual(["input_validation_failed"], verdict["failure_reasons"])
            self.assertEqual("invalid_input", verdict["error"]["code"])
            self.assertEqual(str(bucket_gate_path), verdict["error"]["path"])
            self.assertIn("comparison", verdict["error"]["message"])

    def test_cli_attributes_malformed_bucket_gate_id_to_bucket_gate_file(self):
        importlib.import_module("ml.alphazero_lite.write_tactical_validation_verdict")

        baseline = forensic_report()
        candidate = forensic_report()
        malformed_bucket_gate = {
            "schema": "azlite_bucket_promotion_gate_v1",
            "passed": True,
            "checks": [
                {
                    "id": "badid",
                    "comparison": "max",
                    "passed": True,
                    "baseline_value": 0.1,
                    "candidate_value": 0.1,
                    "threshold": 0.0,
                }
            ],
        }

        with tempfile.TemporaryDirectory(
            prefix="azlite-tactical-validation-verdict-"
        ) as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run" / "final"
            selected_artifact = run_dir / "selected-candidate"
            selected_artifact.mkdir(parents=True, exist_ok=True)
            (selected_artifact / "model.npz").write_text("stub", encoding="utf-8")

            baseline_path = run_dir / "baseline_candidate_forensics.json"
            candidate_path = run_dir / "selected_candidate_forensics.json"
            bucket_gate_path = run_dir / "bucket_gate.json"
            regression_path = run_dir / "candidate_regression_suite.json"
            arena_path = run_dir / "arena_seed_1041.json"
            out_path = run_dir / "tactical_validation_verdict.json"

            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            bucket_gate_path.write_text(
                json.dumps(malformed_bucket_gate), encoding="utf-8"
            )
            regression_path.write_text(
                json.dumps(regression_report(passed=True)), encoding="utf-8"
            )
            arena_path.write_text(
                json.dumps(arena_report(passed=True)), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.write_tactical_validation_verdict",
                    "--run-dir",
                    str(run_dir),
                    "--selected-artifact",
                    str(selected_artifact),
                    "--baseline-forensics",
                    str(baseline_path),
                    "--candidate-forensics",
                    str(candidate_path),
                    "--bucket-gate",
                    str(bucket_gate_path),
                    "--regression-report",
                    str(regression_path),
                    "--arena-report",
                    str(arena_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(1, result.returncode)
            verdict = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(["input_validation_failed"], verdict["failure_reasons"])
            self.assertEqual("invalid_input", verdict["error"]["code"])
            self.assertEqual(str(bucket_gate_path), verdict["error"]["path"])
            self.assertIn("bucket_gate checks[0]", verdict["error"]["message"])

    def test_verdict_fails_on_non_derived_uncovered_adverse_delta(self):
        importlib.import_module("ml.alphazero_lite.write_tactical_validation_verdict")

        baseline = forensic_report(overall_top1_agreement=0.70)
        candidate = forensic_report(overall_top1_agreement=0.68)
        bucket_gate = check_bucket_promotion_gate.evaluate_gate(baseline, candidate)
        self.assertTrue(bucket_gate["passed"])

        bucket_gate["checks"] = [
            check
            for check in bucket_gate["checks"]
            if check["id"] != "overall.top1_agreement"
        ]

        with tempfile.TemporaryDirectory(
            prefix="azlite-tactical-validation-verdict-"
        ) as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run" / "final"
            selected_artifact = run_dir / "selected-candidate"
            selected_artifact.mkdir(parents=True, exist_ok=True)
            (selected_artifact / "model.npz").write_text("stub", encoding="utf-8")

            baseline_path = run_dir / "baseline_candidate_forensics.json"
            candidate_path = run_dir / "selected_candidate_forensics.json"
            bucket_gate_path = run_dir / "bucket_gate.json"
            regression_path = run_dir / "candidate_regression_suite.json"
            arena_path = run_dir / "arena_seed_1041.json"
            out_path = run_dir / "tactical_validation_verdict.json"

            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            bucket_gate_path.write_text(json.dumps(bucket_gate), encoding="utf-8")
            regression_path.write_text(
                json.dumps(regression_report(passed=True)), encoding="utf-8"
            )
            arena_path.write_text(
                json.dumps(arena_report(passed=True)), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.write_tactical_validation_verdict",
                    "--run-dir",
                    str(run_dir),
                    "--selected-artifact",
                    str(selected_artifact),
                    "--baseline-forensics",
                    str(baseline_path),
                    "--candidate-forensics",
                    str(candidate_path),
                    "--bucket-gate",
                    str(bucket_gate_path),
                    "--regression-report",
                    str(regression_path),
                    "--arena-report",
                    str(arena_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(1, result.returncode, msg=result.stderr)
            verdict = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertFalse(verdict["passed"])
            self.assertIn("extra_adverse_delta_detected", verdict["failure_reasons"])
            self.assertEqual(
                {
                    "id": "overall.top1_agreement",
                    "comparison": "min",
                    "baseline_value": 0.7,
                    "candidate_value": 0.68,
                },
                verdict["extra_adverse_deltas"][0],
            )

    def test_cli_attributes_metric_readiness_failure_to_candidate_forensics_file(self):
        importlib.import_module("ml.alphazero_lite.write_tactical_validation_verdict")

        baseline = forensic_report()
        candidate = forensic_report()
        bucket_gate = check_bucket_promotion_gate.evaluate_gate(baseline, baseline)

        candidate["systems"]["challenger"]["rows"] = [
            row
            for row in candidate["systems"]["challenger"]["rows"]
            if row["bucket"] != "capture_available"
        ]

        with tempfile.TemporaryDirectory(
            prefix="azlite-tactical-validation-verdict-"
        ) as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run" / "final"
            selected_artifact = run_dir / "selected-candidate"
            selected_artifact.mkdir(parents=True, exist_ok=True)
            (selected_artifact / "model.npz").write_text("stub", encoding="utf-8")

            baseline_path = run_dir / "baseline_candidate_forensics.json"
            candidate_path = run_dir / "selected_candidate_forensics.json"
            bucket_gate_path = run_dir / "bucket_gate.json"
            regression_path = run_dir / "candidate_regression_suite.json"
            arena_path = run_dir / "arena_seed_1041.json"
            out_path = run_dir / "tactical_validation_verdict.json"

            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            bucket_gate_path.write_text(json.dumps(bucket_gate), encoding="utf-8")
            regression_path.write_text(
                json.dumps(regression_report(passed=True)), encoding="utf-8"
            )
            arena_path.write_text(
                json.dumps(arena_report(passed=True)), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.write_tactical_validation_verdict",
                    "--run-dir",
                    str(run_dir),
                    "--selected-artifact",
                    str(selected_artifact),
                    "--baseline-forensics",
                    str(baseline_path),
                    "--candidate-forensics",
                    str(candidate_path),
                    "--bucket-gate",
                    str(bucket_gate_path),
                    "--regression-report",
                    str(regression_path),
                    "--arena-report",
                    str(arena_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(1, result.returncode)
            self.assertTrue(out_path.exists())
            verdict = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(["input_validation_failed"], verdict["failure_reasons"])
            self.assertEqual("invalid_input", verdict["error"]["code"])
            self.assertEqual(str(candidate_path), verdict["error"]["path"])
            self.assertIn(
                "missing challenger rows for capture_available",
                verdict["error"]["message"],
            )

    def test_verdict_dedupes_overlapping_extra_adverse_delta_ids(self):
        importlib.import_module("ml.alphazero_lite.write_tactical_validation_verdict")

        baseline = forensic_report(
            buckets={
                "capture_available": {
                    "average_regret": 0.10,
                    "rows": [0.05] * 20,
                }
            }
        )
        candidate = forensic_report(
            buckets={
                "capture_available": {
                    "average_regret": 0.11,
                    "rows": [0.25] + ([0.05] * 19),
                }
            }
        )
        bucket_gate = check_bucket_promotion_gate.evaluate_gate(baseline, candidate)
        self.assertFalse(bucket_gate["passed"])

        bucket_gate["checks"] = [
            check
            for check in bucket_gate["checks"]
            if check["id"] != "capture_available.blunder_rate_0_20"
        ]
        bucket_gate["passed"] = all(check["passed"] for check in bucket_gate["checks"])

        with tempfile.TemporaryDirectory(
            prefix="azlite-tactical-validation-verdict-"
        ) as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run" / "final"
            selected_artifact = run_dir / "selected-candidate"
            selected_artifact.mkdir(parents=True, exist_ok=True)
            (selected_artifact / "model.npz").write_text("stub", encoding="utf-8")

            baseline_path = run_dir / "baseline_candidate_forensics.json"
            candidate_path = run_dir / "selected_candidate_forensics.json"
            bucket_gate_path = run_dir / "bucket_gate.json"
            regression_path = run_dir / "candidate_regression_suite.json"
            arena_path = run_dir / "arena_seed_1041.json"
            out_path = run_dir / "tactical_validation_verdict.json"

            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            bucket_gate_path.write_text(json.dumps(bucket_gate), encoding="utf-8")
            regression_path.write_text(
                json.dumps(regression_report(passed=True)), encoding="utf-8"
            )
            arena_path.write_text(
                json.dumps(arena_report(passed=True)), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.write_tactical_validation_verdict",
                    "--run-dir",
                    str(run_dir),
                    "--selected-artifact",
                    str(selected_artifact),
                    "--baseline-forensics",
                    str(baseline_path),
                    "--candidate-forensics",
                    str(candidate_path),
                    "--bucket-gate",
                    str(bucket_gate_path),
                    "--regression-report",
                    str(regression_path),
                    "--arena-report",
                    str(arena_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(1, result.returncode)
            verdict = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertFalse(verdict["passed"])
            self.assertEqual(
                [
                    delta
                    for delta in verdict["extra_adverse_deltas"]
                    if delta["id"] == "capture_available.blunder_rate_0_20"
                ],
                [
                    {
                        "id": "capture_available.blunder_rate_0_20",
                        "comparison": "max",
                        "baseline_value": 0.0,
                        "candidate_value": 0.05,
                    }
                ],
            )

    def test_cli_attributes_non_capture_derived_blunder_metric_failure_to_candidate_forensics_file(
        self,
    ):
        importlib.import_module("ml.alphazero_lite.write_tactical_validation_verdict")

        baseline = forensic_report()
        candidate = forensic_report()
        bucket_gate = check_bucket_promotion_gate.evaluate_gate(baseline, baseline)

        candidate["systems"]["challenger"]["rows"] = [
            row
            for row in candidate["systems"]["challenger"]["rows"]
            if row["bucket"] != "high_imbalance"
        ]

        with tempfile.TemporaryDirectory(
            prefix="azlite-tactical-validation-verdict-"
        ) as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run" / "final"
            selected_artifact = run_dir / "selected-candidate"
            selected_artifact.mkdir(parents=True, exist_ok=True)
            (selected_artifact / "model.npz").write_text("stub", encoding="utf-8")

            baseline_path = run_dir / "baseline_candidate_forensics.json"
            candidate_path = run_dir / "selected_candidate_forensics.json"
            bucket_gate_path = run_dir / "bucket_gate.json"
            regression_path = run_dir / "candidate_regression_suite.json"
            arena_path = run_dir / "arena_seed_1041.json"
            out_path = run_dir / "tactical_validation_verdict.json"

            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            bucket_gate_path.write_text(json.dumps(bucket_gate), encoding="utf-8")
            regression_path.write_text(
                json.dumps(regression_report(passed=True)), encoding="utf-8"
            )
            arena_path.write_text(
                json.dumps(arena_report(passed=True)), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.write_tactical_validation_verdict",
                    "--run-dir",
                    str(run_dir),
                    "--selected-artifact",
                    str(selected_artifact),
                    "--baseline-forensics",
                    str(baseline_path),
                    "--candidate-forensics",
                    str(candidate_path),
                    "--bucket-gate",
                    str(bucket_gate_path),
                    "--regression-report",
                    str(regression_path),
                    "--arena-report",
                    str(arena_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(1, result.returncode)
            self.assertTrue(out_path.exists())
            verdict = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(["input_validation_failed"], verdict["failure_reasons"])
            self.assertEqual("invalid_input", verdict["error"]["code"])
            self.assertEqual(str(candidate_path), verdict["error"]["path"])
            self.assertIn(
                "missing challenger rows for high_imbalance",
                verdict["error"]["message"],
            )


if __name__ == "__main__":
    unittest.main()
