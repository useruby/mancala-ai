import unittest

from ml.alphazero_lite import build_superhuman_strength_dataset as dataset_builder


class BuildSuperhumanStrengthDatasetTest(unittest.TestCase):
    def test_build_dataset_rows_replays_selected_positions(self):
        games = [
            {
                "id": 1,
                "winner": 0,
                "move_history": [
                    {"seat": "player", "pit": "2"},
                    {"seat": "player", "pit": "5"},
                    {"seat": "bot", "pit": 1},
                ],
            }
        ]

        selections = [
            {"game_id": 1, "move_number": 1, "label": "opening_player"},
            {"game_id": 1, "move_number": 3, "label": "first_bot_reply"},
        ]

        rows = dataset_builder.build_dataset_rows(games, selections=selections)

        self.assertEqual(2, len(rows))

        opening_row = rows[0]
        self.assertEqual(2, opening_row["move_index"])
        self.assertEqual(0, opening_row["player"])
        self.assertEqual(1.0, opening_row["value"])
        self.assertEqual("superhuman_strength_opening_player", opening_row["source"])
        self.assertAlmostEqual(1.0, sum(opening_row["policy"]))
        self.assertGreater(opening_row["policy"][2], 0.9)

        bot_row = rows[1]
        self.assertEqual(1, bot_row["move_index"])
        self.assertEqual(1, bot_row["player"])
        self.assertEqual(-1.0, bot_row["value"])
        self.assertEqual("superhuman_strength_first_bot_reply", bot_row["source"])
        self.assertAlmostEqual(1.0, sum(bot_row["policy"]))
        self.assertGreater(bot_row["policy"][1], 0.9)
        self.assertEqual("sharpened", bot_row["policy_target_mode"])
        self.assertEqual("sharpened", bot_row["value_target_mode"])

    def test_build_dataset_rows_rejects_missing_game_id(self):
        with self.assertRaisesRegex(ValueError, "missing game_id 99"):
            dataset_builder.build_dataset_rows([], selections=[{"game_id": 99, "move_number": 1, "label": "missing"}])

    def test_build_dataset_rows_rejects_out_of_range_move_number(self):
        games = [{"id": 1, "winner": 0, "move_history": [{"seat": "player", "pit": "2"}]}]

        with self.assertRaisesRegex(ValueError, "move_number out of range"):
            dataset_builder.build_dataset_rows(games, selections=[{"game_id": 1, "move_number": 2, "label": "bad_move"}])


if __name__ == "__main__":
    unittest.main()
