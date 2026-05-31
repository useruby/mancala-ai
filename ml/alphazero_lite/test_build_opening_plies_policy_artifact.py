from __future__ import annotations

import unittest


class BuildOpeningPliesPolicyArtifactTest(unittest.TestCase):
    def test_build_artifact_rows_uses_opening_failure_rows(self) -> None:
        from ml.alphazero_lite import build_opening_plies_policy_artifact as module

        candidate_forensics = {
            "reference": {
                "artifact_path": "/tmp/ref.json",
            },
            "systems": {
                "current": {
                    "rows": [
                        {
                            "id": "opening_plies_1_8-004",
                            "canonical_state": "canon-opening-004",
                            "state": {
                                "player_pits": [4, 0, 5, 5, 5, 5],
                                "opponent_pits": [4, 4, 4, 4, 4, 4],
                                "player_store": 0,
                                "opponent_store": 0,
                                "current_player": 1,
                            },
                            "side_to_move": 1,
                            "legal_moves": [0, 1, 2, 3, 4, 5],
                            "bucket": "opening_plies_1_8",
                            "reference_move": 4,
                            "selected_move": 2,
                            "teacher_value": 0.42,
                            "regret": 0.05,
                            "value_error": 0.33,
                            "teacher_child_stats": [
                                {"move": 2, "visits": 300, "win_rate": 0.80},
                                {"move": 4, "visits": 500, "win_rate": 0.86},
                            ],
                        },
                        {
                            "id": "opening_plies_1_8-001",
                            "canonical_state": "canon-opening-001",
                            "state": {
                                "player_pits": [4, 4, 4, 4, 4, 4],
                                "opponent_pits": [4, 4, 4, 4, 4, 4],
                                "player_store": 0,
                                "opponent_store": 0,
                                "current_player": 0,
                            },
                            "side_to_move": 0,
                            "legal_moves": [0, 1, 2, 3, 4, 5],
                            "bucket": "opening_plies_1_8",
                            "reference_move": 2,
                            "selected_move": 2,
                            "teacher_value": 0.52,
                            "regret": 0.0,
                            "value_error": 0.1,
                            "teacher_child_stats": [
                                {"move": 2, "visits": 700, "win_rate": 0.76},
                            ],
                        },
                    ]
                }
            },
        }
        reference_artifact = {
            "reference": {
                "policy_simulations": 1200,
                "value_simulations": 1800,
                "sample_seeds": [2040],
            },
            "rows": [
                {
                    "id": "opening_plies_1_8-004",
                    "reference_move": 4,
                    "teacher_value": 0.42,
                    "child_stats": [
                        {"move": 2, "visits": 300, "win_rate": 0.80},
                        {"move": 4, "visits": 500, "win_rate": 0.86},
                    ],
                }
            ],
        }

        rows, summary = module.build_artifact_rows(
            candidate_forensics=candidate_forensics,
            reference_artifact=reference_artifact,
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
        self.assertEqual("opening_plies_1_8-004", rows[0]["source_runs"][0]["id"])
        self.assertEqual(4, rows[0]["teacher_selected_move"])
        self.assertEqual("preservation", rows[0]["bucket_group"])
        self.assertEqual(
            module.OPENING_NO_EXTRA_TURN_REPLAY_ROLE,
            rows[0]["replay_role"],
        )
        self.assertFalse(rows[0]["reference_move_extra_turn_available"])
        self.assertEqual(["opening_plies_1_8-004"], summary["row_ids"])

    def test_opening_replay_role_distinguishes_no_extra_turn_rows(self) -> None:
        from ml.alphazero_lite import build_opening_plies_policy_artifact as module

        replay_role = module.opening_replay_role(
            raw_state={
                "player_pits": [4, 0, 5, 5, 5, 5],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 1,
            },
            reference_move=4,
        )

        self.assertEqual(module.OPENING_NO_EXTRA_TURN_REPLAY_ROLE, replay_role)

    def test_opening_replay_role_distinguishes_extra_turn_rows(self) -> None:
        from ml.alphazero_lite import build_opening_plies_policy_artifact as module

        replay_role = module.opening_replay_role(
            raw_state={
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            },
            reference_move=2,
        )

        self.assertEqual(module.OPENING_EXTRA_TURN_REPLAY_ROLE, replay_role)


if __name__ == "__main__":
    unittest.main()
