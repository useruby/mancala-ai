from __future__ import annotations

from unittest import TestCase

from ml.alphazero_lite import build_train_only_forensic_suite_from_selfplay as module


class BuildTrainOnlyForensicSuiteFromSelfPlayTest(TestCase):
    def test_decode_state_recovers_integer_board(self) -> None:
        state = module.decode_state(
            [
                4 / 48,
                3 / 48,
                2 / 48,
                1 / 48,
                0.0,
                5 / 48,
                6 / 48,
                4 / 48,
                3 / 48,
                2 / 48,
                1 / 48,
                0.0,
                7 / 48,
                8 / 48,
                1.0,
            ]
        )
        self.assertEqual([4, 3, 2, 1, 0, 5], state["player_pits"])
        self.assertEqual([6, 4, 3, 2, 1, 0], state["opponent_pits"])
        self.assertEqual(7, state["player_store"])
        self.assertEqual(8, state["opponent_store"])
        self.assertEqual(1, state["current_player"])

    def test_build_suite_rows_rejects_missing_buckets(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required buckets"):
            module.build_suite_rows(
                [
                    {
                        "state": {
                            "player_pits": [4, 4, 4, 4, 4, 4],
                            "opponent_pits": [4, 4, 4, 4, 4, 4],
                            "player_store": 0,
                            "opponent_store": 0,
                            "current_player": 0,
                        },
                        "legal_moves": [0, 1, 2, 3, 4, 5],
                        "bucket": "opening_plies_1_8",
                        "phase": "opening",
                        "ply": 0,
                        "source": "test",
                        "canonical_state": "x",
                    }
                ],
                target_size=1,
                min_per_bucket=1,
            )
