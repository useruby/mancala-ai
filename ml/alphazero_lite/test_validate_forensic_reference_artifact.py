import json
import tempfile
import unittest
from pathlib import Path


class ValidateForensicReferenceArtifactTest(unittest.TestCase):
    def setUp(self):
        self.suite_rows = [
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

    def test_validator_accepts_matching_reference_artifact(self):
        from ml.alphazero_lite.forensic_suite import canonical_state_key
        from ml.alphazero_lite.validate_forensic_reference_artifact import (
            validate_reference_artifact,
        )

        state = self.suite_rows[0]["state"]
        canonical_state = canonical_state_key(state)
        artifact = {
            "schema": "azlite_forensic_references_v1",
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
                        {"seed": 2040, "reference_move": 3, "teacher_value": 0.4183}
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            suite_path = tmp_path / "suite.json"
            artifact_path = tmp_path / "references.json"
            suite_path.write_text(json.dumps(self.suite_rows), encoding="utf-8")
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

            summary = validate_reference_artifact(
                suite_path=suite_path, reference_artifact_path=artifact_path
            )

        self.assertTrue(summary["valid"])
        self.assertEqual(0, summary["error_count"])

    def test_validator_rejects_missing_and_illegal_reference_rows(self):
        from ml.alphazero_lite.forensic_suite import canonical_state_key
        from ml.alphazero_lite.validate_forensic_reference_artifact import (
            validate_reference_artifact,
        )

        state = self.suite_rows[0]["state"]
        canonical_state = canonical_state_key(state)
        artifact = {
            "schema": "azlite_forensic_references_v1",
            "rows": [
                {
                    "id": "capture_available-016",
                    "canonical_state": canonical_state,
                    "state": state,
                    "reference_move": 5,
                    "teacher_value": 0.4183,
                    "reference_unstable": False,
                    "observed_reference_moves": [5],
                    "seed_samples": [
                        {"seed": 2040, "reference_move": 5, "teacher_value": 0.4183}
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            suite_path = tmp_path / "suite.json"
            artifact_path = tmp_path / "references.json"
            suite_path.write_text(json.dumps(self.suite_rows), encoding="utf-8")
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

            summary = validate_reference_artifact(
                suite_path=suite_path, reference_artifact_path=artifact_path
            )

        self.assertFalse(summary["valid"])
        self.assertIn("capture_available-016", summary["illegal_reference_row_ids"])


if __name__ == "__main__":
    unittest.main()
