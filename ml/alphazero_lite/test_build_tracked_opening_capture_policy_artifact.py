from __future__ import annotations

import unittest


class BuildTrackedOpeningCapturePolicyArtifactTest(unittest.TestCase):
    def test_derive_policy_normalizes_child_visits(self) -> None:
        from ml.alphazero_lite import build_tracked_opening_capture_policy_artifact as module

        policy = module.derive_policy(
            [
                {"move": 1, "visits": 3},
                {"move": 4, "visits": 1},
            ]
        )

        self.assertEqual(0.75, policy[1])
        self.assertEqual(0.25, policy[4])
        self.assertAlmostEqual(1.0, sum(policy))

    def test_build_artifact_rows_uses_reference_moves_not_existing_opening_replay_moves(self) -> None:
        from ml.alphazero_lite import build_tracked_opening_capture_policy_artifact as module

        candidate_forensics = {
            "systems": {
                "challenger": {
                    "rows": [
                        {
                            "id": "capture_available-005",
                            "canonical_state": "canon-a",
                            "state": {
                                "player_pits": [5, 5, 1, 0, 6, 6],
                                "opponent_pits": [5, 5, 4, 4, 4, 0],
                                "player_store": 2,
                                "opponent_store": 1,
                                "current_player": 0,
                            },
                            "side_to_move": 0,
                            "legal_moves": [0, 1, 2, 4, 5],
                            "bucket": "capture_available",
                            "regret": 0.1,
                            "value_error": 0.2,
                        }
                    ]
                }
            }
        }
        reference_artifact = {
            "reference": {
                "policy_simulations": 1200,
                "value_simulations": 1800,
                "sample_seeds": [42],
            },
            "rows": [
                {
                    "id": "capture_available-005",
                    "reference_move": 4,
                    "teacher_value": 0.75,
                    "child_stats": [
                        {"move": 1, "visits": 10, "win_rate": 0.1},
                        {"move": 4, "visits": 30, "win_rate": 0.9},
                    ],
                }
            ],
        }
        move_summary = [
            {
                "id": "capture_available-005",
                "reference_move": 4,
                "baseline_prior": 1,
                "baseline_search": 1,
                "w2_prior": 1,
                "w2_search": 0,
            }
        ]
        rows, summary = module.build_artifact_rows(
            candidate_forensics=candidate_forensics,
            reference_artifact=reference_artifact,
            move_selection_summary=move_summary,
            reference_artifact_path="/tmp/ref.json",
            input_encoding="kalah_v3",
            policy_target_mode="sharpened",
            value_target_mode="sharpened",
            state_encoder=lambda raw_state, input_encoding: {
                "encoded": raw_state,
                "input_encoding": input_encoding,
            },
        )

        self.assertEqual(1, len(rows))
        self.assertEqual(4, rows[0]["teacher_selected_move"])
        self.assertEqual(
            module.NO_EXTRA_TURN_REPLAY_ROLE,
            rows[0]["replay_role"],
        )
        self.assertFalse(rows[0]["reference_move_extra_turn_available"])
        self.assertEqual({4: 1}, summary["teacher_selected_move_distribution"])
        self.assertEqual(
            {module.NO_EXTRA_TURN_REPLAY_ROLE: 1},
            summary["replay_role_counts"],
        )
        self.assertEqual(
            {module.NO_EXTRA_TURN_REPLAY_ROLE: ["capture_available-005"]},
            summary["row_ids_by_replay_role"],
        )
        self.assertEqual(
            {module.NO_EXTRA_TURN_REPLAY_ROLE: {4: 1}},
            summary["teacher_selected_move_distribution_by_replay_role"],
        )
        self.assertEqual(
            {"capture_available-005": False},
            summary["reference_extra_turn_by_row"],
        )

    def test_replay_role_for_reference_move_distinguishes_extra_turn_polarity(self) -> None:
        from ml.alphazero_lite import build_tracked_opening_capture_policy_artifact as module

        no_extra_turn_state = {
            "player_pits": [5, 4, 4, 0, 5, 0],
            "opponent_pits": [1, 0, 7, 7, 6, 6],
            "player_store": 2,
            "opponent_store": 1,
            "current_player": 0,
        }
        extra_turn_state = {
            "player_pits": [5, 5, 5, 0, 5, 0],
            "opponent_pits": [1, 6, 6, 0, 6, 6],
            "player_store": 2,
            "opponent_store": 1,
            "current_player": 0,
        }

        self.assertEqual(
            module.NO_EXTRA_TURN_REPLAY_ROLE,
            module.replay_role_for_reference_move(
                raw_state=no_extra_turn_state,
                reference_move=4,
            ),
        )
        self.assertEqual(
            module.EXTRA_TURN_REPLAY_ROLE,
            module.replay_role_for_reference_move(
                raw_state=extra_turn_state,
                reference_move=1,
            ),
        )


if __name__ == "__main__":
    unittest.main()
