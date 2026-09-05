from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from ml.alphazero_lite.run_kalah_v1_native_tablebase_preflight import (
    ROOT,
    classify,
    cumulative_count,
    portable_format_gate,
    rank,
    run_gate,
    transition_gate,
    unrank,
)


class KalahV1NativeTablebasePreflightTest(unittest.TestCase):
    def test_native_format_selftest(self) -> None:
        binary = Path(
            subprocess.check_output(
                ["bash", "native/kalah_v1_tablebase/build.sh"], cwd=ROOT, text=True
            ).strip()
        )
        subprocess.run([binary, "format-selftest"], check=True)

    def test_dense_rank_unrank_and_count_identity_through_ten(self) -> None:
        for tier in range(11):
            self.assertEqual(
                cumulative_count(tier), 2 * __import__("math").comb(tier + 12, 12)
            )
            for index in range(__import__("math").comb(tier + 11, 11)):
                self.assertEqual(rank(unrank(tier, index)), index)

    def test_native_probe_covers_cited_and_one_sided_positions(self) -> None:
        binary = Path(
            subprocess.check_output(
                ["bash", "native/kalah_v1_tablebase/build.sh"], cwd=ROOT, text=True
            ).strip()
        )
        with tempfile.TemporaryDirectory(
            prefix="kalah_v1_test_", dir="/tmp"
        ) as temporary:
            tablebase = Path(temporary) / "eight.kvtb"
            subprocess.run(
                [binary, "generate", "8", tablebase],
                check=True,
                capture_output=True,
                text=True,
            )
            requests = [
                {"pits": [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0], "player": 0},
                {"pits": [0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 1], "player": 1},
                {"pits": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "player": 1},
            ]
            result = subprocess.run(
                [binary, "probe", tablebase],
                input="".join(json.dumps(request) + "\n" for request in requests),
                check=True,
                capture_output=True,
                text=True,
            )
        answers = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual({"1": -4, "2": 0}, answers[0]["actions"])
        self.assertEqual({"0": -4, "5": -4}, answers[1]["actions"])
        self.assertEqual({}, answers[2]["actions"])

    def test_portable_reader_rejects_corruption_and_results_are_incomplete(
        self,
    ) -> None:
        binary = Path(
            subprocess.check_output(
                ["bash", "native/kalah_v1_tablebase/build.sh"], cwd=ROOT, text=True
            ).strip()
        )
        with tempfile.TemporaryDirectory(
            prefix="kalah_v1_format_", dir="/tmp"
        ) as temporary:
            tablebase = Path(temporary) / "portable.kvtb"
            subprocess.run([binary, "generate", "1", tablebase], check=True)
            result = portable_format_gate(binary, tablebase, Path(temporary))
            self.assertTrue(result["passed"])
            self.assertGreaterEqual(result["fixture_count"], 19)
            self.assertTrue(all(result["fixtures"].values()))
        summary = json.loads(
            (
                ROOT
                / "docs/data/alphazero-lite-kalah-v1-native-tablebase-preflight-summary.json"
            ).read_text()
        )
        self.assertEqual("canonical_tablebase_feasible", summary["classification"])
        self.assertTrue(summary["validation_complete"])
        full = json.loads((ROOT / summary["full_result"]).read_text())
        self.assertEqual(summary["classification"], full["classification"])
        self.assertEqual(summary["validation_complete"], full["validation_complete"])
        report = (
            ROOT / "docs/alphazero-lite-kalah-v1-native-tablebase-preflight-results.md"
        ).read_text()
        self.assertIn(f"`{summary['classification']}`", report)

    def test_bounded_native_python_transition_parity(self) -> None:
        binary = Path(
            subprocess.check_output(
                ["bash", "native/kalah_v1_tablebase/build.sh"], cwd=ROOT, text=True
            ).strip()
        )
        result = transition_gate(binary, 3)
        self.assertTrue(result["passed"])
        self.assertGreater(result["legal_actions"], 0)

    def test_projection_returning_false_is_a_budget_failure(self) -> None:
        classification, complete, reason = classify(
            {"projection_18": {"status": "failed", "error": {}}}, True
        )
        self.assertEqual("canonical_tablebase_budget_exceeded", classification)
        self.assertFalse(complete)
        self.assertIn("projection_18", reason)

    def test_gate_returning_false_is_recorded_as_failed(self) -> None:
        gates: dict[str, dict] = {}
        self.assertIsNone(run_gate(gates, "projection_18", lambda: {"passed": False}))
        self.assertEqual("failed", gates["projection_18"]["status"])

    def test_generator_cycle_stderr_is_classified(self) -> None:
        classification, complete, reason = classify(
            {
                "generation": {
                    "status": "failed",
                    "error": {"message": "command failed", "stderr": "cycle"},
                }
            },
            False,
        )
        self.assertEqual("canonical_tablebase_recurrence_blocked", classification)
        self.assertFalse(complete)
        self.assertEqual("native recurrence cycle", reason)


if __name__ == "__main__":
    unittest.main()
