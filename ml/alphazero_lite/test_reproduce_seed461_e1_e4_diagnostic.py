"""Check public-matrix bootstrap reproduction against the completed run."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from ml.alphazero_lite.reproduce_seed461_e1_e4_diagnostic import reproduce


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed461-e1-e4-corrected-diagnostic"


class Seed461MatrixReproductionTests(unittest.TestCase):
    def test_registered_intervals_reproduce_from_paired_matrix(self) -> None:
        result = reproduce(DATA / "per-opening-score-matrix.json")
        published = json.loads((DATA / "results.json").read_text())
        self.assertEqual(result["means"]["E1"], published["scores"]["E1"])
        self.assertEqual(result["means"]["E4"], published["scores"]["E4"])
        self.assertEqual(
            result["means"]["paired_E1_minus_E4"],
            published["paired_E1_minus_E4"]["mean"],
        )
        self.assertEqual(result["intervals_95"]["E1"], published["intervals_95"]["E1"])
        self.assertEqual(result["intervals_95"]["E4"], published["intervals_95"]["E4"])
        self.assertEqual(
            result["intervals_95"]["paired_E1_minus_E4"],
            published["paired_E1_minus_E4"]["interval_95"],
        )


if __name__ == "__main__":
    unittest.main()
