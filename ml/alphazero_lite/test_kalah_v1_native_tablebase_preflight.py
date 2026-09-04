from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from ml.alphazero_lite.run_kalah_v1_native_tablebase_preflight import (
    ROOT,
    cumulative_count,
    rank,
    transition_gate,
    unrank,
)


class KalahV1NativeTablebasePreflightTest(unittest.TestCase):
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
            corrupt = Path(temporary) / "corrupt.kvtb"
            corrupt.write_bytes(tablebase.read_bytes() + b"unexpected")
            self.assertNotEqual(
                0,
                subprocess.run(
                    [binary, "probe", corrupt], input="", text=True
                ).returncode,
            )
        summary = json.loads(
            (
                ROOT
                / "docs/data/alphazero-lite-kalah-v1-native-tablebase-preflight-summary.json"
            ).read_text()
        )
        self.assertEqual(
            "canonical_tablebase_validation_incomplete", summary["classification"]
        )
        self.assertFalse(summary["validation_complete"])
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


if __name__ == "__main__":
    unittest.main()
