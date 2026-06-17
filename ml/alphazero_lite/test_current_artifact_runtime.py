import unittest
from pathlib import Path

from ml.alphazero_lite import arena
from ml.alphazero_lite.kalah_rules import KalahGame


class CurrentArtifactRuntimeTest(unittest.TestCase):
    def test_current_artifact_loads_and_selects_legal_move(self):
        artifact_dir = Path(__file__).resolve().parents[2] / "model-artifact/current"
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


if __name__ == "__main__":
    unittest.main()
