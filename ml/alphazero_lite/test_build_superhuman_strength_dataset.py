import unittest
from pathlib import Path
import json

from ml.alphazero_lite import build_superhuman_strength_dataset as dataset_builder
from ml.alphazero_lite.input_encodings import feature_count_for


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

    def test_default_selections_build_deterministic_curated_rows_with_kalah_v3_encoding(self):
        fixture_path = Path(__file__).with_name("fixtures") / "superhuman_strength_games_2026_04_06.json"
        games = json.loads(fixture_path.read_text(encoding="utf-8"))

        rows = dataset_builder.build_dataset_rows(games)
        expected_contract = [
            ("superhuman_strength_g11_m18", 4, 0, 1.0, 0.97),
            ("superhuman_strength_g11_m29", 0, 1, -1.0, 0.97),
            ("superhuman_strength_g11_m32", 2, 1, -1.0, 0.97),
            ("superhuman_strength_g11_m38", 4, 0, 1.0, 0.97),
            ("superhuman_strength_g11_m40", 0, 0, 1.0, 1.0),
            ("superhuman_strength_g12_m19", 4, 0, 1.0, 0.97),
            ("superhuman_strength_g12_m31", 1, 1, -1.0, 0.97),
            ("superhuman_strength_g12_m34", 5, 1, -1.0, 0.97),
            ("superhuman_strength_g12_m37", 2, 0, 1.0, 0.97),
            ("superhuman_strength_g12_m39", 1, 0, 1.0, 0.97),
            ("superhuman_strength_g12_m44", 4, 0, 1.0, 1.0),
            ("superhuman_strength_g12_m45", 2, 1, -1.0, 0.97),
        ]

        self.assertEqual(len(dataset_builder.DEFAULT_SELECTIONS), len(rows))
        self.assertEqual(
            [f"superhuman_strength_{selection['label']}" for selection in dataset_builder.DEFAULT_SELECTIONS],
            [row["source"] for row in rows],
        )
        self.assertEqual(
            expected_contract,
            [
                (row["source"], row["move_index"], row["player"], row["value"], row["policy"][row["move_index"]])
                for row in rows
            ],
        )

        for row in rows:
            self.assertEqual("sharpened", row["policy_target_mode"])
            self.assertEqual("sharpened", row["value_target_mode"])
            self.assertAlmostEqual(1.0, sum(row["policy"]))
            self.assertEqual(6, len(row["policy"]))
            self.assertEqual(feature_count_for("kalah_v3"), len(row["state"]))

        first_rows = dataset_builder.build_dataset_rows(games)
        self.assertEqual(rows, first_rows)

    def test_search_control_config_uses_deterministic_superhuman_search_settings(self):
        config_path = Path(__file__).with_name("configs") / "aggressive_v3_superhuman_search_control.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected_programs = {
            "arena_confirm_report": "ml/alphazero_lite/arena.py",
            "mcts1200_baseline_report": "ml/alphazero_lite/mcts1200_baseline.py",
            "current_mcts1200_baseline_report": "ml/alphazero_lite/mcts1200_baseline.py",
        }

        self.assertEqual("aggressive-v3-superhuman-search-control", config["run_id"])
        self.assertEqual(
            ["arena_confirm_report", "mcts1200_baseline_report", "current_mcts1200_baseline_report"],
            [step["name"] for step in config["steps"]],
        )

        for step in config["steps"]:
            command = step["command"]
            self.assertEqual(expected_programs[step["name"]], command[1])
            self.assertIn("--fpu-mode", command)
            self.assertIn("parent_q", command)
            self.assertIn("--reuse-subtree", command)
            self.assertIn("--normalize-values", command)
            self.assertIn("--root-policy-mode", command)
            root_policy_index = command.index("--root-policy-mode")
            self.assertEqual("deterministic", command[root_policy_index + 1])
            self.assertIn("--tactical-root-bias", command)
            tactical_bias_index = command.index("--tactical-root-bias")
            self.assertEqual("0.1", command[tactical_bias_index + 1])


if __name__ == "__main__":
    unittest.main()
