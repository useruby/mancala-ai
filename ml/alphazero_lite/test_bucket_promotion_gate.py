import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite import check_bucket_promotion_gate


PYTHON_BIN = Path(sys.executable)


def report(
    *,
    overall_average_regret: float = 0.08,
    overall_top1_agreement: float = 0.7,
    buckets: dict[str, dict[str, float | int | list[float]]] | None = None,
    reference: dict | None = None,
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
            "artifact_path": "/tmp/reference_moves.json",
            "shared_reference": {
                "policy_simulations": 1200,
                "value_simulations": 1200,
                "sample_seeds": [42],
            },
        }
        if reference is None
        else reference,
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


def repeated_rows(value: float, count: int) -> list[float]:
    return [value] * count


def set_bucket_row_regret(report_payload: dict, bucket: str, regret: object) -> None:
    for row in report_payload["systems"]["challenger"]["rows"]:
        if row["bucket"] == bucket:
            row["regret"] = regret
            return
    raise AssertionError(f"missing row for bucket {bucket}")


def remove_bucket_row_regret(report_payload: dict, bucket: str) -> None:
    for row in report_payload["systems"]["challenger"]["rows"]:
        if row["bucket"] == bucket:
            row.pop("regret", None)


class BucketPromotionGateTest(unittest.TestCase):
    def test_gate_passes_when_candidate_matches_exact_threshold_boundaries(self):
        baseline_capture_rows = ([0.10] * 40) + ([0.20] * 10)
        candidate_capture_rows = ([0.10] * 39) + ([0.20] * 11)

        baseline = report(
            overall_average_regret=0.08,
            overall_top1_agreement=0.72,
            buckets={
                "capture_available": {
                    "positions": 50,
                    "average_regret": 0.10,
                    "rows": baseline_capture_rows,
                },
                "high_imbalance": {
                    "average_regret": 0.12,
                    "rows": repeated_rows(0.1, 20),
                },
                "high_value_swing": {
                    "average_regret": 0.11,
                    "rows": repeated_rows(0.1, 20),
                },
                "sparse_endgame": {
                    "top1_agreement": 0.74,
                    "rows": repeated_rows(0.1, 20),
                },
                "opening_plies_1_8": {
                    "average_regret": 0.09,
                    "positions": 40,
                    "rows": repeated_rows(0.1, 40),
                },
            },
        )
        candidate = report(
            overall_average_regret=0.085,
            overall_top1_agreement=0.70,
            buckets={
                "capture_available": {
                    "positions": 50,
                    "average_regret": 0.105,
                    "rows": candidate_capture_rows,
                },
                "high_imbalance": {
                    "average_regret": 0.125,
                    "rows": repeated_rows(0.1, 20),
                },
                "high_value_swing": {
                    "average_regret": 0.115,
                    "rows": repeated_rows(0.1, 20),
                },
                "sparse_endgame": {
                    "top1_agreement": 0.71,
                    "rows": repeated_rows(0.1, 20),
                },
                "opening_plies_1_8": {
                    "average_regret": 0.095,
                    "positions": 40,
                    "rows": repeated_rows(0.1, 40),
                },
            },
        )

        result = check_bucket_promotion_gate.evaluate_gate(baseline, candidate)

        self.assertTrue(result["passed"])
        self.assertTrue(all(check["passed"] for check in result["checks"]))

    def test_gate_passes_when_candidate_stays_inside_thresholds(self):
        baseline = report(
            overall_average_regret=0.08,
            overall_top1_agreement=0.72,
            buckets={
                "capture_available": {
                    "average_regret": 0.10,
                    "rows": [0.05, 0.10, 0.15, 0.19, 0.10] + repeated_rows(0.05, 15),
                },
                "high_imbalance": {
                    "average_regret": 0.12,
                    "rows": repeated_rows(0.1, 20),
                },
                "high_value_swing": {
                    "average_regret": 0.11,
                    "rows": repeated_rows(0.1, 20),
                },
                "sparse_endgame": {
                    "top1_agreement": 0.74,
                    "rows": repeated_rows(0.1, 20),
                },
                "opening_plies_1_8": {
                    "average_regret": 0.09,
                    "positions": 40,
                    "rows": repeated_rows(0.1, 40),
                },
            },
        )
        candidate = report(
            overall_average_regret=0.085,
            overall_top1_agreement=0.70,
            buckets={
                "capture_available": {
                    "average_regret": 0.105,
                    "rows": [0.05, 0.10, 0.15, 0.19, 0.10] + repeated_rows(0.05, 15),
                },
                "high_imbalance": {
                    "average_regret": 0.125,
                    "rows": repeated_rows(0.1, 20),
                },
                "high_value_swing": {
                    "average_regret": 0.115,
                    "rows": repeated_rows(0.1, 20),
                },
                "sparse_endgame": {
                    "top1_agreement": 0.71,
                    "rows": repeated_rows(0.1, 20),
                },
                "opening_plies_1_8": {
                    "average_regret": 0.095,
                    "positions": 40,
                    "rows": repeated_rows(0.1, 40),
                },
            },
        )

        result = check_bucket_promotion_gate.evaluate_gate(baseline, candidate)

        self.assertEqual("azlite_bucket_promotion_gate_v1", result["schema"])
        self.assertTrue(result["passed"])
        self.assertEqual(8, len(result["checks"]))
        self.assertTrue(all(check["passed"] for check in result["checks"]))
        self.assertTrue(all("id" in check for check in result["checks"]))
        self.assertTrue(all("baseline_value" in check for check in result["checks"]))
        self.assertTrue(all("candidate_value" in check for check in result["checks"]))
        self.assertTrue(all("threshold" in check for check in result["checks"]))

    def test_gate_rejects_incomplete_bucket_row_coverage_for_blunder_metric(self):
        baseline = report(
            buckets={
                "capture_available": {
                    "positions": 20,
                    "rows": [0.05] * 20,
                }
            }
        )
        candidate = report(
            buckets={
                "capture_available": {
                    "positions": 20,
                    "rows": [0.05],
                }
            }
        )

        with self.assertRaisesRegex(
            ValueError, "capture_available rows must match positions"
        ):
            check_bucket_promotion_gate.evaluate_gate(baseline, candidate)

    def test_gate_rejects_missing_regret_in_bucket_rows_for_blunder_metric(self):
        baseline = report()
        candidate = report()
        remove_bucket_row_regret(candidate, "capture_available")

        with self.assertRaisesRegex(
            ValueError, "capture_available rows must include numeric regret"
        ):
            check_bucket_promotion_gate.evaluate_gate(baseline, candidate)

    def test_gate_rejects_mismatched_shared_reference_provenance(self):
        baseline = report(
            reference={
                "kind": "shared_artifact",
                "artifact_path": "/tmp/reference-a.json",
                "shared_reference": {
                    "policy_simulations": 1200,
                    "value_simulations": 1200,
                    "sample_seeds": [42, 99],
                },
            }
        )
        candidate = report(
            reference={
                "kind": "shared_artifact",
                "artifact_path": "/tmp/reference-b.json",
                "shared_reference": {
                    "policy_simulations": 1200,
                    "value_simulations": 1200,
                    "sample_seeds": [42],
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "shared reference provenance"):
            check_bucket_promotion_gate.evaluate_gate(baseline, candidate)

    def test_gate_rejects_missing_shared_reference_provenance(self):
        baseline = report(reference=None)
        candidate = report(reference={"kind": "classic_mcts"})

        with self.assertRaisesRegex(ValueError, "shared reference provenance"):
            check_bucket_promotion_gate.evaluate_gate(baseline, candidate)

    def test_gate_rejects_boolean_bucket_metric_values(self):
        baseline = report()
        candidate = report()
        candidate["buckets"]["capture_available"]["systems"]["challenger"][
            "average_regret"
        ] = True

        with self.assertRaisesRegex(
            ValueError, "capture_available missing average_regret"
        ):
            check_bucket_promotion_gate.evaluate_gate(baseline, candidate)

    def test_gate_rejects_boolean_overall_metric_values(self):
        baseline = report()
        candidate = report()
        candidate["systems"]["challenger"]["overall"]["top1_agreement"] = False

        with self.assertRaisesRegex(ValueError, "overall missing top1_agreement"):
            check_bucket_promotion_gate.evaluate_gate(baseline, candidate)

    def test_gate_excludes_reference_unstable_rows_from_strict_capture_metrics(self):
        baseline = report(
            buckets={
                "capture_available": {
                    "positions": 20,
                    "average_regret": 0.10,
                    "rows": [0.05] * 20,
                }
            }
        )
        candidate = report(
            buckets={
                "capture_available": {
                    "positions": 20,
                    "average_regret": 0.105,
                    "rows": ([0.05] * 18) + [0.25, 0.30],
                }
            }
        )
        unstable_marked = 0
        for row in candidate["systems"]["challenger"]["rows"]:
            if row["bucket"] != "capture_available":
                continue
            row["reference_unstable"] = unstable_marked < 2 and row["regret"] >= 0.20
            if row["reference_unstable"]:
                unstable_marked += 1

        result = check_bucket_promotion_gate.evaluate_gate(baseline, candidate)

        self.assertTrue(result["passed"])
        blunder_check = next(
            check
            for check in result["checks"]
            if check["id"] == "capture_available.blunder_rate_0_20"
        )
        self.assertEqual(0.0, blunder_check["candidate_value"])

    def test_gate_fails_capture_regret_regression(self):
        baseline = report(
            buckets={
                "capture_available": {
                    "average_regret": 0.10,
                    "rows": [0.05, 0.10, 0.15, 0.19, 0.10] + repeated_rows(0.05, 15),
                }
            }
        )
        candidate = report(
            buckets={
                "capture_available": {
                    "average_regret": 0.106,
                    "rows": [0.05, 0.10, 0.25, 0.30, 0.10] + repeated_rows(0.05, 15),
                }
            }
        )

        result = check_bucket_promotion_gate.evaluate_gate(baseline, candidate)

        self.assertFalse(result["passed"])
        failed = [check for check in result["checks"] if not check["passed"]]
        self.assertEqual(
            ["capture_available.average_regret", "capture_available.blunder_rate_0_20"],
            [check["id"] for check in failed],
        )

    def test_cli_writes_gate_report_and_returns_nonzero_on_failure(self):
        baseline = report(
            buckets={
                "capture_available": {
                    "average_regret": 0.10,
                    "rows": [0.05, 0.10, 0.15, 0.19, 0.10] + repeated_rows(0.05, 15),
                }
            }
        )
        candidate = report(
            buckets={
                "capture_available": {
                    "average_regret": 0.106,
                    "rows": [0.05, 0.10, 0.25, 0.30, 0.10] + repeated_rows(0.05, 15),
                }
            }
        )

        with tempfile.TemporaryDirectory(prefix="azlite-bucket-gate-") as tmp:
            tmp_path = Path(tmp)
            baseline_path = tmp_path / "baseline.json"
            candidate_path = tmp_path / "candidate.json"
            out_path = tmp_path / "nested" / "gate.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.check_bucket_promotion_gate",
                    "--baseline-forensics",
                    str(baseline_path),
                    "--candidate-forensics",
                    str(candidate_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            gate_report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertFalse(gate_report["passed"])
            self.assertEqual("azlite_bucket_promotion_gate_v1", gate_report["schema"])

    def test_cli_writes_structured_failure_report_for_malformed_forensics(self):
        baseline = report()
        malformed_candidate = {
            "schema": "azlite_forensic_suite_v1",
            "systems": {
                "challenger": {
                    "overall": {"average_regret": 0.08, "top1_agreement": 0.7},
                    "rows": repeated_rows(0.1, 120),
                }
            },
            "buckets": {},
        }

        with tempfile.TemporaryDirectory(prefix="azlite-bucket-gate-") as tmp:
            tmp_path = Path(tmp)
            baseline_path = tmp_path / "baseline.json"
            candidate_path = tmp_path / "candidate.json"
            out_path = tmp_path / "nested" / "gate.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(malformed_candidate), encoding="utf-8")

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.check_bucket_promotion_gate",
                    "--baseline-forensics",
                    str(baseline_path),
                    "--candidate-forensics",
                    str(candidate_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertTrue(out_path.exists())
            gate_report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual("azlite_bucket_promotion_gate_v1", gate_report["schema"])
            self.assertFalse(gate_report["passed"])
            self.assertEqual([], gate_report["checks"])
            self.assertEqual("invalid_forensics", gate_report["error"]["code"])
            self.assertIn("missing bucket", gate_report["error"]["message"])
            self.assertEqual(str(candidate_path), gate_report["error"]["path"])
            self.assertEqual("", result.stderr)

    def test_cli_writes_structured_failure_report_for_malformed_baseline_forensics(
        self,
    ):
        malformed_baseline = {
            "schema": "azlite_forensic_suite_v1",
            "systems": {
                "challenger": {
                    "overall": {"average_regret": 0.08, "top1_agreement": 0.7},
                    "rows": repeated_rows(0.1, 120),
                }
            },
            "buckets": {},
        }
        candidate = report()

        with tempfile.TemporaryDirectory(prefix="azlite-bucket-gate-") as tmp:
            tmp_path = Path(tmp)
            baseline_path = tmp_path / "baseline.json"
            candidate_path = tmp_path / "candidate.json"
            out_path = tmp_path / "nested" / "gate.json"
            baseline_path.write_text(json.dumps(malformed_baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.check_bucket_promotion_gate",
                    "--baseline-forensics",
                    str(baseline_path),
                    "--candidate-forensics",
                    str(candidate_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertTrue(out_path.exists())
            gate_report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual("azlite_bucket_promotion_gate_v1", gate_report["schema"])
            self.assertFalse(gate_report["passed"])
            self.assertEqual([], gate_report["checks"])
            self.assertEqual("invalid_forensics", gate_report["error"]["code"])
            self.assertIn("missing bucket", gate_report["error"]["message"])
            self.assertEqual(str(baseline_path), gate_report["error"]["path"])
            self.assertEqual("", result.stderr)

    def test_cli_attributes_malformed_baseline_row_regret_to_baseline_file(self):
        malformed_baseline = report()
        set_bucket_row_regret(malformed_baseline, "capture_available", "bad-regret")
        candidate = report()

        with tempfile.TemporaryDirectory(prefix="azlite-bucket-gate-") as tmp:
            tmp_path = Path(tmp)
            baseline_path = tmp_path / "baseline.json"
            candidate_path = tmp_path / "candidate.json"
            out_path = tmp_path / "nested" / "gate.json"
            baseline_path.write_text(json.dumps(malformed_baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.check_bucket_promotion_gate",
                    "--baseline-forensics",
                    str(baseline_path),
                    "--candidate-forensics",
                    str(candidate_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertTrue(out_path.exists())
            gate_report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual("azlite_bucket_promotion_gate_v1", gate_report["schema"])
            self.assertFalse(gate_report["passed"])
            self.assertEqual([], gate_report["checks"])
            self.assertEqual("invalid_forensics", gate_report["error"]["code"])
            self.assertIn("regret", gate_report["error"]["message"])
            self.assertEqual(str(baseline_path), gate_report["error"]["path"])
            self.assertEqual("", result.stderr)

    def test_cli_attributes_malformed_candidate_row_regret_to_candidate_file(self):
        baseline = report()
        malformed_candidate = report()
        set_bucket_row_regret(malformed_candidate, "capture_available", "bad-regret")

        with tempfile.TemporaryDirectory(prefix="azlite-bucket-gate-") as tmp:
            tmp_path = Path(tmp)
            baseline_path = tmp_path / "baseline.json"
            candidate_path = tmp_path / "candidate.json"
            out_path = tmp_path / "nested" / "gate.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(malformed_candidate), encoding="utf-8")

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.check_bucket_promotion_gate",
                    "--baseline-forensics",
                    str(baseline_path),
                    "--candidate-forensics",
                    str(candidate_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertTrue(out_path.exists())
            gate_report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual("azlite_bucket_promotion_gate_v1", gate_report["schema"])
            self.assertFalse(gate_report["passed"])
            self.assertEqual([], gate_report["checks"])
            self.assertEqual("invalid_forensics", gate_report["error"]["code"])
            self.assertIn("regret", gate_report["error"]["message"])
            self.assertEqual(str(candidate_path), gate_report["error"]["path"])
            self.assertEqual("", result.stderr)

    def test_cli_attributes_shared_reference_provenance_failure_to_baseline_file(self):
        malformed_baseline = report(reference={"kind": "classic_mcts"})
        candidate = report()

        with tempfile.TemporaryDirectory(prefix="azlite-bucket-gate-") as tmp:
            tmp_path = Path(tmp)
            baseline_path = tmp_path / "baseline.json"
            candidate_path = tmp_path / "candidate.json"
            out_path = tmp_path / "nested" / "gate.json"
            baseline_path.write_text(json.dumps(malformed_baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.check_bucket_promotion_gate",
                    "--baseline-forensics",
                    str(baseline_path),
                    "--candidate-forensics",
                    str(candidate_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            gate_report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual("invalid_forensics", gate_report["error"]["code"])
            self.assertIn(
                "shared reference provenance", gate_report["error"]["message"]
            )
            self.assertEqual(str(baseline_path), gate_report["error"]["path"])


if __name__ == "__main__":
    unittest.main()
