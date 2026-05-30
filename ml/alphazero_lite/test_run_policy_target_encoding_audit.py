import unittest
from unittest.mock import patch

from ml.alphazero_lite.run_policy_target_encoding_audit import (
    dataset_policy_target_consistency,
)


class PolicyTargetEncodingAuditTest(unittest.TestCase):
    def test_reference_source_rates_are_computed(self):
        reference_rows = {
            "row-extra-turn": {
                "reference_move": 5,
                "state": {
                    "current_player": 0,
                    "player_pits": [1, 0, 0, 0, 0, 1],
                    "opponent_pits": [1, 1, 1, 1, 1, 1],
                    "player_store": 0,
                    "opponent_store": 0,
                },
                "child_stats": [
                    {"move": 0, "visits": 1},
                    {"move": 5, "visits": 9},
                ],
            },
            "row-capture": {
                "reference_move": 2,
                "state": {
                    "current_player": 0,
                    "player_pits": [1, 0, 1, 0, 0, 0],
                    "opponent_pits": [0, 0, 5, 1, 0, 0],
                    "player_store": 0,
                    "opponent_store": 0,
                },
                "child_stats": [
                    {"move": 0, "visits": 1},
                    {"move": 2, "visits": 9},
                ],
            },
        }

        with patch(
            "ml.alphazero_lite.run_policy_target_encoding_audit.DATASET_PATHS", []
        ):
            summary = dataset_policy_target_consistency(reference_rows)

        reference_source = next(
            row
            for row in summary["sources"]
            if row["source"].endswith("train_only_forensic_references.json")
        )
        self.assertEqual(reference_source["rows"], 2)
        self.assertEqual(reference_source["extra_turn_top_move_rate"], 0.5)
        self.assertEqual(reference_source["no_extra_turn_capture_top_move_rate"], 0.5)
        self.assertEqual(
            reference_source[
                "rows_with_extra_turn_over_no_extra_turn_capture_conflict"
            ],
            0,
        )


if __name__ == "__main__":
    unittest.main()
