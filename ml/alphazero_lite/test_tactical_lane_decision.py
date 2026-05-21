import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PYTHON_BIN = Path(sys.executable)


def gate_report(*, passed: bool) -> dict:
    return {
        "schema": "gate_report_v1",
        "passed": passed,
        "checks": [],
    }


def exploratory_report(
    *,
    passed: bool,
    qualifying_seed_count: int = 2,
    required_qualifying_seed_count: int = 2,
) -> dict:
    return {
        "schema": "azlite_tactical_exploratory_summary_v1",
        "passed": passed,
        "qualifying_seed_count": qualifying_seed_count,
        "required_qualifying_seed_count": required_qualifying_seed_count,
    }


class TacticalLaneDecisionTest(unittest.TestCase):
    def test_decision_passes_only_when_bucket_and_promotion_pass(self):
        tactical_lane_decision = importlib.import_module(
            "ml.alphazero_lite.write_tactical_lane_decision"
        )

        result = tactical_lane_decision.build_decision(
            gate_report(passed=True),
            gate_report(passed=True),
            exploratory_report(passed=True, qualifying_seed_count=2),
        )

        self.assertEqual("azlite_tactical_lane_decision_v1", result["schema"])
        self.assertTrue(result["passed"])
        self.assertEqual([], result["failure_reasons"])
        self.assertTrue(result["bucket_gate"]["passed"])
        self.assertTrue(result["promotion_gate"]["passed"])
        self.assertTrue(result["exploratory_summary"]["passed"])

    def test_decision_fails_when_bucket_gate_fails(self):
        tactical_lane_decision = importlib.import_module(
            "ml.alphazero_lite.write_tactical_lane_decision"
        )

        result = tactical_lane_decision.build_decision(
            gate_report(passed=False),
            gate_report(passed=True),
            exploratory_report(passed=True, qualifying_seed_count=2),
        )

        self.assertFalse(result["passed"])
        self.assertEqual(["bucket_gate_failed"], result["failure_reasons"])

        result = tactical_lane_decision.build_decision(
            gate_report(passed=True),
            gate_report(passed=False),
            exploratory_report(passed=True, qualifying_seed_count=2),
        )

        self.assertFalse(result["passed"])
        self.assertEqual(["local_promotion_gate_failed"], result["failure_reasons"])

        result = tactical_lane_decision.build_decision(
            gate_report(passed=False),
            gate_report(passed=False),
            exploratory_report(passed=True, qualifying_seed_count=2),
        )

        self.assertFalse(result["passed"])
        self.assertEqual(
            ["bucket_gate_failed", "local_promotion_gate_failed"],
            result["failure_reasons"],
        )

    def test_decision_fails_without_two_qualifying_exploratory_seeds(self):
        tactical_lane_decision = importlib.import_module(
            "ml.alphazero_lite.write_tactical_lane_decision"
        )

        result = tactical_lane_decision.build_decision(
            gate_report(passed=True),
            gate_report(passed=True),
            exploratory_report(passed=True, qualifying_seed_count=1),
        )

        self.assertFalse(result["passed"])
        self.assertEqual(
            ["exploratory_seed_confirmation_failed"], result["failure_reasons"]
        )

    def test_decision_uses_required_qualifying_seed_count_from_exploratory_summary(
        self,
    ):
        tactical_lane_decision = importlib.import_module(
            "ml.alphazero_lite.write_tactical_lane_decision"
        )

        result = tactical_lane_decision.build_decision(
            gate_report(passed=True),
            gate_report(passed=True),
            exploratory_report(
                passed=True, qualifying_seed_count=2, required_qualifying_seed_count=3
            ),
        )

        self.assertFalse(result["passed"])
        self.assertEqual(
            ["exploratory_seed_confirmation_failed"], result["failure_reasons"]
        )

        result = tactical_lane_decision.build_decision(
            gate_report(passed=True),
            gate_report(passed=True),
            exploratory_report(
                passed=True, qualifying_seed_count=3, required_qualifying_seed_count=3
            ),
        )

        self.assertTrue(result["passed"])
        self.assertEqual([], result["failure_reasons"])

        result = tactical_lane_decision.build_decision(
            gate_report(passed=True),
            gate_report(passed=True),
            exploratory_report(passed=False, qualifying_seed_count=3),
        )

        self.assertFalse(result["passed"])
        self.assertEqual(
            ["exploratory_seed_confirmation_failed"], result["failure_reasons"]
        )

    def test_cli_writes_decision_report(self):
        importlib.import_module("ml.alphazero_lite.write_tactical_lane_decision")

        with tempfile.TemporaryDirectory(
            prefix="azlite-tactical-lane-decision-"
        ) as tmp:
            tmp_path = Path(tmp)
            bucket_path = tmp_path / "bucket_gate.json"
            promotion_path = tmp_path / "promotion_gate.json"
            out_path = tmp_path / "nested" / "decision.json"
            exploratory_path = tmp_path / "exploratory_summary.json"

            bucket_path.write_text(
                json.dumps(gate_report(passed=True)), encoding="utf-8"
            )
            promotion_path.write_text(
                json.dumps(gate_report(passed=False)), encoding="utf-8"
            )
            exploratory_path.write_text(
                json.dumps(exploratory_report(passed=True)), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.write_tactical_lane_decision",
                    "--bucket-gate",
                    str(bucket_path),
                    "--promotion-gate",
                    str(promotion_path),
                    "--exploratory-summary",
                    str(exploratory_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(1, result.returncode)
            decision = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual("azlite_tactical_lane_decision_v1", decision["schema"])
            self.assertFalse(decision["passed"])
            self.assertEqual(
                ["local_promotion_gate_failed"], decision["failure_reasons"]
            )
            self.assertEqual(gate_report(passed=True), decision["bucket_gate"])
            self.assertEqual(gate_report(passed=False), decision["promotion_gate"])
            self.assertEqual(
                exploratory_report(passed=True), decision["exploratory_summary"]
            )

    def test_cli_returns_zero_when_both_gates_pass(self):
        importlib.import_module("ml.alphazero_lite.write_tactical_lane_decision")

        with tempfile.TemporaryDirectory(
            prefix="azlite-tactical-lane-decision-"
        ) as tmp:
            tmp_path = Path(tmp)
            bucket_path = tmp_path / "bucket_gate.json"
            promotion_path = tmp_path / "promotion_gate.json"
            out_path = tmp_path / "nested" / "decision.json"
            exploratory_path = tmp_path / "exploratory_summary.json"

            bucket_path.write_text(
                json.dumps(gate_report(passed=True)), encoding="utf-8"
            )
            promotion_path.write_text(
                json.dumps(gate_report(passed=True)), encoding="utf-8"
            )
            exploratory_path.write_text(
                json.dumps(exploratory_report(passed=True)), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.write_tactical_lane_decision",
                    "--bucket-gate",
                    str(bucket_path),
                    "--promotion-gate",
                    str(promotion_path),
                    "--exploratory-summary",
                    str(exploratory_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode)
            decision = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertTrue(decision["passed"])
            self.assertEqual([], decision["failure_reasons"])
            self.assertEqual(gate_report(passed=True), decision["bucket_gate"])
            self.assertEqual(gate_report(passed=True), decision["promotion_gate"])
            self.assertEqual(
                exploratory_report(passed=True), decision["exploratory_summary"]
            )

    def test_cli_rejects_malformed_valid_json_shape_without_traceback(self):
        importlib.import_module("ml.alphazero_lite.write_tactical_lane_decision")

        with tempfile.TemporaryDirectory(
            prefix="azlite-tactical-lane-decision-"
        ) as tmp:
            tmp_path = Path(tmp)
            bucket_path = tmp_path / "bucket_gate.json"
            promotion_path = tmp_path / "promotion_gate.json"
            exploratory_path = tmp_path / "exploratory_summary.json"
            out_path = tmp_path / "nested" / "decision.json"

            bucket_path.write_text(json.dumps([]), encoding="utf-8")
            promotion_path.write_text(
                json.dumps(gate_report(passed=True)), encoding="utf-8"
            )
            exploratory_path.write_text(
                json.dumps(exploratory_report(passed=True)), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.write_tactical_lane_decision",
                    "--bucket-gate",
                    str(bucket_path),
                    "--promotion-gate",
                    str(promotion_path),
                    "--exploratory-summary",
                    str(exploratory_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(1, result.returncode)
            self.assertFalse(out_path.exists())
            self.assertNotEqual("", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_cli_rejects_non_boolean_passed_field_without_traceback(self):
        importlib.import_module("ml.alphazero_lite.write_tactical_lane_decision")

        with tempfile.TemporaryDirectory(
            prefix="azlite-tactical-lane-decision-"
        ) as tmp:
            tmp_path = Path(tmp)
            bucket_path = tmp_path / "bucket_gate.json"
            promotion_path = tmp_path / "promotion_gate.json"
            exploratory_path = tmp_path / "exploratory_summary.json"
            out_path = tmp_path / "nested" / "decision.json"

            bucket_path.write_text(json.dumps({"passed": "yes"}), encoding="utf-8")
            promotion_path.write_text(
                json.dumps(gate_report(passed=True)), encoding="utf-8"
            )
            exploratory_path.write_text(
                json.dumps(exploratory_report(passed=True)), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.write_tactical_lane_decision",
                    "--bucket-gate",
                    str(bucket_path),
                    "--promotion-gate",
                    str(promotion_path),
                    "--exploratory-summary",
                    str(exploratory_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(1, result.returncode)
            self.assertFalse(out_path.exists())
            self.assertNotEqual("", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_cli_rejects_exploratory_summary_without_two_qualifying_seeds(self):
        importlib.import_module("ml.alphazero_lite.write_tactical_lane_decision")

        with tempfile.TemporaryDirectory(
            prefix="azlite-tactical-lane-decision-"
        ) as tmp:
            tmp_path = Path(tmp)
            bucket_path = tmp_path / "bucket_gate.json"
            promotion_path = tmp_path / "promotion_gate.json"
            exploratory_path = tmp_path / "exploratory_summary.json"
            out_path = tmp_path / "nested" / "decision.json"

            bucket_path.write_text(
                json.dumps(gate_report(passed=True)), encoding="utf-8"
            )
            promotion_path.write_text(
                json.dumps(gate_report(passed=True)), encoding="utf-8"
            )
            exploratory_path.write_text(
                json.dumps(exploratory_report(passed=True, qualifying_seed_count=1)),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.write_tactical_lane_decision",
                    "--bucket-gate",
                    str(bucket_path),
                    "--promotion-gate",
                    str(promotion_path),
                    "--exploratory-summary",
                    str(exploratory_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(1, result.returncode)
            decision = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertFalse(decision["passed"])
            self.assertEqual(
                ["exploratory_seed_confirmation_failed"], decision["failure_reasons"]
            )


if __name__ == "__main__":
    unittest.main()
