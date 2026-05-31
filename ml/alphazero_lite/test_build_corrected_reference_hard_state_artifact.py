import json
import tempfile
import unittest
from pathlib import Path


class BuildCorrectedReferenceHardStateArtifactTest(unittest.TestCase):
    def test_builder_filters_to_corrected_failures_and_adds_mining_metrics(self):
        from ml.alphazero_lite.build_corrected_reference_hard_state_artifact import (
            build_corrected_reference_hard_state_artifact,
        )

        forensic_report = {
            "schema": "azlite_forensic_suite_v1",
            "systems": {
                "current": {
                    "artifact_path": "storage/ai/alphazero_lite/current",
                    "rows": [
                        {
                            "id": "capture_available-001",
                            "bucket": "capture_available",
                            "state": {
                                "player_pits": [4, 4, 4, 4, 0, 5],
                                "opponent_pits": [0, 1, 6, 6, 6, 6],
                                "player_store": 1,
                                "opponent_store": 1,
                                "current_player": 0,
                            },
                            "side_to_move": 0,
                            "legal_moves": [0, 1, 2, 3, 5],
                            "selected_move": 2,
                            "reference_move": 4,
                            "agrees_top1": False,
                            "regret": 0.4,
                            "teacher_value": 0.2,
                            "system_value": -0.3,
                            "value_error": 0.5,
                            "child_stats": [
                                {"move": 2, "visits": 30, "win_rate": 0.6},
                                {"move": 4, "visits": 10, "win_rate": 0.2},
                            ],
                        },
                        {
                            "id": "capture_available-002",
                            "bucket": "capture_available",
                            "state": {
                                "player_pits": [1, 0, 7, 6, 6, 5],
                                "opponent_pits": [5, 4, 4, 4, 4, 0],
                                "player_store": 1,
                                "opponent_store": 1,
                                "current_player": 1,
                            },
                            "side_to_move": 1,
                            "legal_moves": [0, 2, 3, 4, 5],
                            "selected_move": 2,
                            "reference_move": 2,
                            "agrees_top1": True,
                            "regret": 0.0,
                            "teacher_value": 0.7,
                            "system_value": 0.6,
                            "value_error": 0.1,
                            "child_stats": [
                                {"move": 2, "visits": 40, "win_rate": 0.8},
                                {"move": 4, "visits": 5, "win_rate": 0.5},
                            ],
                        },
                    ],
                }
            },
        }
        corrected_inventory = [
            {
                "row_id": "capture_available-001",
                "family": "capture_available",
                "failure_status": "fail_corrected_reference",
                "severity": "high",
                "recommended_use": "candidate_failure_family",
            },
            {
                "row_id": "capture_available-002",
                "family": "capture_available",
                "failure_status": "pass_corrected_reference",
                "severity": "none",
                "recommended_use": "keep_validation_gate",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            forensic_path = tmp_path / "forensic.json"
            inventory_path = tmp_path / "inventory.json"
            forensic_path.write_text(json.dumps(forensic_report), encoding="utf-8")
            inventory_path.write_text(json.dumps(corrected_inventory), encoding="utf-8")

            artifact, summary = build_corrected_reference_hard_state_artifact(
                forensic_report_path=forensic_path,
                corrected_inventory_path=inventory_path,
            )

        rows = artifact["systems"]["challenger"]["rows"]
        self.assertEqual(1, len(rows))
        self.assertEqual("capture_available-001", rows[0]["id"])
        self.assertEqual(
            "fail_corrected_reference", rows[0]["corrected_failure_status"]
        )
        self.assertEqual("capture_available", rows[0]["corrected_failure_family"])
        self.assertGreater(rows[0]["entropy"], 0.0)
        self.assertGreater(rows[0]["best_second_gap"], 0.0)
        self.assertEqual(1, summary["row_count"])
        self.assertEqual({"capture_available": 1}, summary["family_counts"])


if __name__ == "__main__":
    unittest.main()
