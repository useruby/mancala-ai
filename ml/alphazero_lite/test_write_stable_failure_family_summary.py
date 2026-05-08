import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class WriteStableFailureFamilySummaryTest(unittest.TestCase):
    def test_build_summary_reports_capture_search_flips_and_high_imbalance_blunders(self):
        from ml.alphazero_lite import write_stable_failure_family_summary as module

        candidate_forensics = {
            "systems": {
                "challenger": {
                    "rows": [
                        {
                            "id": "capture_available-016",
                            "bucket": "capture_available",
                            "phase": "opening",
                            "legal_moves": [0, 1, 2, 3, 4],
                            "reference_move": 4,
                            "reference_unstable": False,
                            "selected_move": 0,
                            "regret": 0.0882,
                        },
                        {
                            "id": "high_imbalance-001",
                            "bucket": "high_imbalance",
                            "phase": "midgame",
                            "legal_moves": [0, 1, 2, 3, 5],
                            "reference_move": 2,
                            "reference_unstable": False,
                            "selected_move": 0,
                            "regret": 0.2112,
                        },
                        {
                            "id": "high_imbalance-002",
                            "bucket": "high_imbalance",
                            "phase": "midgame",
                            "legal_moves": [0, 1, 2, 3, 5],
                            "reference_move": 2,
                            "reference_unstable": False,
                            "selected_move": 2,
                            "regret": 0.0888,
                        },
                    ]
                }
            }
        }
        opening_family_report = {
            "rows": [
                {
                    "id": "capture_available-016",
                    "reference_move": 4,
                    "candidate_prior_summary": {"selected_move": 4, "reference_move": 4, "reference_mass": 0.55, "reference_margin": 0.25, "early_mass": 0.3, "value": 0.51},
                    "candidate_searched_summary": {"selected_move": 0, "reference_move": 4, "reference_mass": 0.15, "reference_margin": -0.53, "early_mass": 0.68, "value": 0.51},
                }
            ],
            "missing_references": [],
        }

        summary = module.build_summary(candidate_forensics=candidate_forensics, opening_family_report=opening_family_report)

        self.assertEqual(1, summary["capture_available"]["tracked_rows"])
        self.assertEqual(1, summary["capture_available"]["search_flipped_rows"])
        self.assertEqual(0.0882, summary["capture_available"]["average_regret"])
        self.assertEqual(0.0, summary["capture_available"]["blunder_rate_0_20"])
        self.assertEqual(2, summary["high_imbalance"]["stable_rows"])
        self.assertEqual(0.15, summary["high_imbalance"]["average_regret"])
        self.assertEqual(0.5, summary["high_imbalance"]["blunder_rate_0_20"])
        self.assertEqual(["high_imbalance-001"], summary["high_imbalance"]["blunder_ids"])

    def test_build_summary_treats_missing_regret_as_non_blunder(self):
        from ml.alphazero_lite import write_stable_failure_family_summary as module

        candidate_forensics = {
            "systems": {
                "challenger": {
                    "rows": [
                        {
                            "id": "high_imbalance-001",
                            "bucket": "high_imbalance",
                            "reference_move": 2,
                            "reference_unstable": False,
                            "regret": None,
                        }
                    ]
                }
            }
        }

        summary = module.build_summary(candidate_forensics=candidate_forensics, opening_family_report={"rows": []})

        self.assertEqual(1, summary["high_imbalance"]["stable_rows"])
        self.assertEqual(0.0, summary["high_imbalance"]["average_regret"])
        self.assertEqual(0.0, summary["high_imbalance"]["blunder_rate_0_20"])
        self.assertEqual([], summary["high_imbalance"]["blunder_ids"])

    def test_build_summary_rejects_invalid_regret_with_row_id(self):
        from ml.alphazero_lite import write_stable_failure_family_summary as module

        for regret in (True, "nan", "inf"):
            with self.subTest(regret=regret):
                candidate_forensics = {
                    "systems": {
                        "challenger": {
                            "rows": [
                                {
                                    "id": "high_imbalance-001",
                                    "bucket": "high_imbalance",
                                    "reference_move": 2,
                                    "reference_unstable": False,
                                    "regret": regret,
                                }
                            ]
                        }
                    }
                }

                with self.assertRaisesRegex(ValueError, "high_imbalance-001.*regret"):
                    module.build_summary(candidate_forensics=candidate_forensics, opening_family_report={"rows": []})

    def test_main_writes_summary_with_canonical_metric_name(self):
        from ml.alphazero_lite import write_stable_failure_family_summary as module

        candidate_forensics = {
            "systems": {
                "challenger": {
                    "rows": [
                        {
                            "id": "high_imbalance-001",
                            "bucket": "high_imbalance",
                            "phase": "midgame",
                            "legal_moves": [0, 1, 2, 3, 5],
                            "reference_move": 2,
                            "reference_unstable": False,
                            "regret": 0.2112,
                        }
                    ]
                }
            }
        }
        opening_family_report = {"rows": [], "missing_references": []}

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate_forensics_path = tmp_path / "candidate_forensics.json"
            opening_family_report_path = tmp_path / "opening_capture_family_report.json"
            out_path = tmp_path / "reports" / "stable_failure_family_summary.json"
            candidate_forensics_path.write_text(json.dumps(candidate_forensics), encoding="utf-8")
            opening_family_report_path.write_text(json.dumps(opening_family_report), encoding="utf-8")

            with mock.patch(
                "sys.argv",
                [
                    "write_stable_failure_family_summary",
                    "--candidate-forensics",
                    str(candidate_forensics_path),
                    "--opening-family-report",
                    str(opening_family_report_path),
                    "--out",
                    str(out_path),
                ],
            ):
                module.main()

            payload = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual("azlite_stable_failure_family_summary_v1", payload["schema"])
        self.assertIn("blunder_rate_0_20", payload["high_imbalance"])
        self.assertNotIn("blunder_rate", payload["high_imbalance"])


if __name__ == "__main__":
    unittest.main()
