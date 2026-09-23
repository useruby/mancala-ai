import unittest
from pathlib import Path

import numpy as np

from ml.alphazero_lite import arena
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.runtime_search_policy import load_runtime_search_policy


class CurrentArtifactRuntimeTest(unittest.TestCase):
    artifact_dir = Path(__file__).resolve().parents[2] / "model-artifact/current"

    def test_current_artifact_loads_and_selects_legal_move(self):
        artifact_dir = self.artifact_dir
        evaluator = arena.ArtifactEvaluator(artifact_dir)

        states = [
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            },
            {
                "player_pits": [0, 1, 0, 0, 0, 2],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            },
        ]

        for state in states:
            game = KalahGame.from_state(state)
            legal_moves = game.possible_moves()
            policy, _value = evaluator.evaluate(game)
            self.assertEqual((6,), policy.shape)
            self.assertAlmostEqual(1.0, float(policy[legal_moves].sum()), places=6)
            for move in range(6):
                if move not in legal_moves:
                    self.assertEqual(0.0, float(policy[move]))

            summary = arena.evaluate_artifact_position(
                artifact_path=artifact_dir,
                evaluator=evaluator,
                state=state,
                simulations=0,
                seed=42,
                c_puct=1.25,
                search_options=arena.build_eval_search_options(
                    root_policy_mode="deterministic"
                ),
            )
            self.assertIn(summary["selected_move"], legal_moves)

    def test_current_incumbent_policy_uses_exact_root16_without_leaf_solving(self):
        policy = load_runtime_search_policy(self.artifact_dir)
        self.assertIsNotNone(policy)
        assert policy is not None
        self.assertEqual("puct_exact_root_hybrid", policy["mode"])
        self.assertEqual(16, policy["exact_root_threshold"])
        self.assertEqual("disabled", policy["exact_leaf_solve"])

        sparse_endgame_023 = {
            "player_pits": [0, 6, 0, 0, 1, 2],
            "opponent_pits": [0, 0, 3, 0, 0, 4],
            "player_store": 13,
            "opponent_store": 19,
            "current_player": 0,
        }
        result = arena.evaluate_artifact_position(
            artifact_path=self.artifact_dir,
            state=sparse_endgame_023,
            simulations=32,
            seed=42,
            c_puct=1.25,
            search_options=arena.build_eval_search_options(
                root_policy_mode="deterministic"
            ),
        )
        self.assertEqual(4, result["selected_move"])
        self.assertEqual(
            {"1": -12, "4": 0, "5": -6}, result["exact_root_decision"]["action_margins"]
        )
        self.assertEqual(0, result["exact_root_decision"]["puct_simulations_executed"])

    def test_current_policy_keeps_puct_unchanged_above_exact_root16(self):
        source = (
            Path(__file__).resolve().parents[2]
            / ".tmp/seed48-nextgen-s455-default-value/runs/seed48-nextgen-s455-default-value-iter1"
        )
        state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        options = arena.build_eval_search_options(root_policy_mode="deterministic")
        source_result = arena.evaluate_artifact_position(
            artifact_path=source,
            state=state,
            simulations=32,
            seed=42,
            c_puct=1.25,
            search_options=options,
            exact_root_solve_threshold=16,
        )
        current_result = arena.evaluate_artifact_position(
            artifact_path=self.artifact_dir,
            state=state,
            simulations=32,
            seed=42,
            c_puct=1.25,
            search_options=options,
        )
        self.assertEqual(
            source_result["selected_move"], current_result["selected_move"]
        )
        self.assertEqual(source_result["visits"], current_result["visits"])
        self.assertEqual(source_result["child_stats"], current_result["child_stats"])
        source_outputs = arena.ArtifactEvaluator(source).evaluate(
            KalahGame.from_state(state)
        )
        current_outputs = arena.ArtifactEvaluator(self.artifact_dir).evaluate(
            KalahGame.from_state(state)
        )
        self.assertTrue(np.array_equal(source_outputs[0], current_outputs[0]))
        self.assertEqual(source_outputs[1], current_outputs[1])


if __name__ == "__main__":
    unittest.main()
