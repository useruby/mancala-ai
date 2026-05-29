import unittest

from ml.alphazero_lite import capture_002_003_rule_collision_diagnostic as module


ROW_002_STATE = {
    "player_pits": [5, 4, 4, 0, 5, 0],
    "opponent_pits": [1, 0, 7, 7, 6, 6],
    "player_store": 2,
    "opponent_store": 1,
    "current_player": 0,
}

ROW_003_STATE = {
    "player_pits": [5, 5, 5, 0, 5, 0],
    "opponent_pits": [1, 6, 6, 0, 6, 6],
    "player_store": 2,
    "opponent_store": 1,
    "current_player": 0,
}


class Capture002003RuleCollisionDiagnosticTest(unittest.TestCase):
    def test_simulate_move_rule_features_for_row_002_reference_move(self):
        features = module.simulate_move_rule_features(state=ROW_002_STATE, move=4)

        self.assertEqual(4, features["move"])
        self.assertFalse(features["capture_legal"])
        self.assertFalse(features["extra_turn_available"])
        self.assertEqual(1, features["store_gain"])
        self.assertEqual(2, features["landing_pit"])
        self.assertFalse(features["lands_in_store"])
        self.assertIsNone(features["captured_opposite_pit"])

    def test_simulate_move_rule_features_for_row_002_wrong_selected_move(self):
        features = module.simulate_move_rule_features(state=ROW_002_STATE, move=2)

        self.assertFalse(features["capture_legal"])
        self.assertTrue(features["extra_turn_available"])
        self.assertEqual(1, features["store_gain"])
        self.assertIsNone(features["landing_pit"])
        self.assertTrue(features["lands_in_store"])

    def test_simulate_move_rule_features_for_row_003_reference_move(self):
        features = module.simulate_move_rule_features(state=ROW_003_STATE, move=1)

        self.assertFalse(features["capture_legal"])
        self.assertTrue(features["extra_turn_available"])
        self.assertEqual(1, features["store_gain"])
        self.assertIsNone(features["landing_pit"])
        self.assertTrue(features["lands_in_store"])

    def test_infer_diagnosis_prefers_extra_turn_overvaluation(self):
        rows = {
            "capture_available-002": {
                "reference_move": 4,
                "search_runs": {
                    "tracked": {
                        "searched_selected_move": 2,
                        "selected_minus_reference_q_margin": 0.0168,
                    },
                    "broader": {
                        "searched_selected_move": 2,
                        "selected_minus_reference_q_margin": -0.0416,
                    },
                },
                "move_rule_features": {
                    "reference_move": {
                        "capture_legal": False,
                        "extra_turn_available": False,
                    },
                    "tracked_selected_move": {
                        "capture_legal": False,
                        "extra_turn_available": True,
                    },
                    "broader_selected_move": {
                        "capture_legal": False,
                        "extra_turn_available": True,
                    },
                },
            },
            "capture_available-003": {
                "reference_move": 1,
                "search_runs": {
                    "tracked": {"searched_selected_move": 1},
                    "broader": {"searched_selected_move": 1},
                },
                "move_rule_features": {
                    "reference_move": {
                        "capture_legal": False,
                        "extra_turn_available": True,
                    },
                    "tracked_selected_move": {
                        "capture_legal": False,
                        "extra_turn_available": True,
                    },
                    "broader_selected_move": {
                        "capture_legal": False,
                        "extra_turn_available": True,
                    },
                },
            },
        }

        diagnosis = module.infer_diagnosis(rows=rows)

        self.assertEqual("extra_turn_overvaluation", diagnosis["best_explanation"])
        self.assertEqual(
            "rule_conditioned_policy_artifact_redesign",
            diagnosis["recommendation"],
        )


if __name__ == "__main__":
    unittest.main()
