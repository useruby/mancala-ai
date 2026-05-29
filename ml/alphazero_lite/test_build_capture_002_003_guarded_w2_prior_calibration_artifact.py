from __future__ import annotations

import unittest


class BuildCapture002003GuardedW2PriorCalibrationArtifactTest(unittest.TestCase):
    def test_build_rows_repeats_002_three_times_and_003_once(self) -> None:
        from ml.alphazero_lite import (
            build_capture_002_003_guarded_w2_prior_calibration_artifact as module,
        )

        reference_artifact = {
            "reference": {
                "policy_simulations": 1200,
                "value_simulations": 1800,
                "sample_seeds": [42],
            },
            "rows": [
                {
                    "id": "capture_available-002",
                    "canonical_state": "canon-002",
                    "state": {
                        "player_pits": [5, 4, 4, 0, 5, 0],
                        "opponent_pits": [1, 0, 7, 7, 6, 6],
                        "player_store": 2,
                        "opponent_store": 1,
                        "current_player": 0,
                    },
                    "reference_move": 4,
                    "teacher_value": 0.5606,
                    "child_stats": [
                        {"move": 0, "visits": 159},
                        {"move": 1, "visits": 154},
                        {"move": 2, "visits": 387},
                        {"move": 4, "visits": 500},
                    ],
                },
                {
                    "id": "capture_available-003",
                    "canonical_state": "canon-003",
                    "state": {
                        "player_pits": [5, 5, 5, 0, 5, 0],
                        "opponent_pits": [1, 6, 6, 0, 6, 6],
                        "player_store": 2,
                        "opponent_store": 1,
                        "current_player": 0,
                    },
                    "reference_move": 1,
                    "teacher_value": 0.6278,
                    "child_stats": [
                        {"move": 0, "visits": 137},
                        {"move": 1, "visits": 475},
                        {"move": 2, "visits": 283},
                        {"move": 4, "visits": 305},
                    ],
                },
            ],
        }

        rows, summary = module.build_rows(
            reference_artifact=reference_artifact,
            reference_artifact_path="/tmp/reference.json",
            input_encoding="kalah_v3",
            policy_target_mode="sharpened",
            value_target_mode="sharpened",
            state_encoder=lambda raw_state, input_encoding: {
                "encoded": raw_state,
                "input_encoding": input_encoding,
            },
        )

        self.assertEqual(4, len(rows))
        self.assertEqual(
            [
                "capture_available-002",
                "capture_available-002",
                "capture_available-002",
                "capture_available-003",
            ],
            [row["source_runs"][0]["id"] for row in rows],
        )
        self.assertEqual(
            [1, 2, 3, 1], [row["source_runs"][0]["copy_index"] for row in rows]
        )
        self.assertEqual(
            {"capture_available-002": 3, "capture_available-003": 1},
            summary["multiplicity_by_row"],
        )
        self.assertEqual(
            {"capture_available-002": 3, "capture_available-003": 1},
            summary["row_counts"],
        )
        self.assertEqual(
            {module.REPLAY_ROLE: 4},
            summary["replay_role_counts"],
        )


if __name__ == "__main__":
    unittest.main()
