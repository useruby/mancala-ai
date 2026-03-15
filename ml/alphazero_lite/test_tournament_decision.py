import unittest

from ml.alphazero_lite.tournament_decision import pick_best_topk, summarize_tournament


class TournamentDecisionTest(unittest.TestCase):
    def test_pick_best_topk_prefers_higher_mcts_then_arena(self):
        candidates = [
            {"topk_index": 1, "mcts_screen_score": 0.25, "arena_score": 1.0},
            {"topk_index": 2, "mcts_screen_score": 0.5, "arena_score": 0.25},
            {"topk_index": 3, "mcts_screen_score": 0.5, "arena_score": 0.75},
        ]

        best = pick_best_topk(candidates)
        self.assertEqual(3, best["topk_index"])

    def test_summarize_tournament_uses_median_mcts_and_best_arena(self):
        winners = [
            {"seed": 41, "mcts_confirm_score": 0.5, "arena_score": 0.25},
            {"seed": 42, "mcts_confirm_score": 0.5, "arena_score": 1.0},
            {"seed": 43, "mcts_confirm_score": 0.25, "arena_score": 0.0},
        ]

        summary = summarize_tournament(winners, min_mcts_score=0.45, min_arena_score=0.55)

        self.assertEqual(0.5, summary["median_mcts_score"])
        self.assertEqual(1.0, summary["best_arena_score"])
        self.assertTrue(summary["pass_mcts_median"])
        self.assertTrue(summary["pass_arena_best_of_3"])
        self.assertTrue(summary["passed"])


if __name__ == "__main__":
    unittest.main()
