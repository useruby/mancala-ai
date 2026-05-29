from __future__ import annotations

import unittest


class BuildCapture002003RuleCollisionGuardArtifactTest(unittest.TestCase):
    def test_build_rows_splits_guard_rows_by_extra_turn_polarity(self) -> None:
        from ml.alphazero_lite import (
            build_capture_002_003_rule_collision_guard_artifact as module,
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
        rule_collision_diagnostic = {
            "rows": {
                "capture_available-002": {
                    "search_runs": {
                        "tracked": {"searched_selected_move": 2},
                        "broader": {"searched_selected_move": 2},
                    }
                },
                "capture_available-003": {
                    "search_runs": {
                        "tracked": {"searched_selected_move": 1},
                        "broader": {"searched_selected_move": 1},
                    }
                },
            }
        }

        rows, summary = module.build_rows(
            reference_artifact=reference_artifact,
            rule_collision_diagnostic=rule_collision_diagnostic,
            reference_artifact_path="/tmp/reference.json",
            rule_collision_diagnostic_path="/tmp/diagnostic.json",
            input_encoding="kalah_v3",
            policy_target_mode="sharpened",
            value_target_mode="sharpened",
            state_encoder=lambda raw_state, input_encoding: {
                "encoded": raw_state,
                "input_encoding": input_encoding,
            },
        )

        self.assertEqual(2, len(rows))
        self.assertEqual(
            module.RULE_COLLISION_NO_EXTRA_TURN_GUARD_ROLE,
            rows[0]["replay_role"],
        )
        self.assertEqual(
            module.RULE_COLLISION_EXTRA_TURN_GUARD_ROLE,
            rows[1]["replay_role"],
        )
        self.assertEqual(
            {
                module.RULE_COLLISION_EXTRA_TURN_GUARD_ROLE: 1,
                module.RULE_COLLISION_NO_EXTRA_TURN_GUARD_ROLE: 1,
            },
            summary["replay_role_counts"],
        )
        self.assertEqual(
            {
                module.RULE_COLLISION_EXTRA_TURN_GUARD_ROLE: ["capture_available-003"],
                module.RULE_COLLISION_NO_EXTRA_TURN_GUARD_ROLE: ["capture_available-002"],
            },
            summary["row_ids_by_replay_role"],
        )
        self.assertEqual(
            {"capture_available-002": False, "capture_available-003": True},
            summary["reference_extra_turn_by_row"],
        )


if __name__ == "__main__":
    unittest.main()
