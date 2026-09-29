import unittest

from ml.alphazero_lite.arena_alignment_audit import (
    ArenaAlignmentAuditError,
    classify,
    recompute_score,
    semantic_diff,
)


class ArenaAlignmentAuditTest(unittest.TestCase):
    def test_recomputes_score_and_seat_swapped_pairs(self):
        rows = [
            {
                "game_index": 0,
                "opening_index": 4,
                "game_within_opening": 0,
                "challenger_player": 0,
                "winner": "challenger",
            },
            {
                "game_index": 1,
                "opening_index": 4,
                "game_within_opening": 1,
                "challenger_player": 1,
                "winner": "draw",
            },
        ]
        result = recompute_score(rows)
        self.assertEqual(1, result["wins"])
        self.assertEqual(1, result["draws"])
        self.assertEqual(0.75, result["score"])
        self.assertEqual({0.75: 1}, result["opening_pair_score_distribution"])

    def test_rejects_missing_seat_counterpart(self):
        with self.assertRaisesRegex(ArenaAlignmentAuditError, "invalid_seat_pair"):
            recompute_score(
                [
                    {
                        "game_index": 0,
                        "opening_index": 4,
                        "game_within_opening": 0,
                        "challenger_player": 0,
                        "winner": "challenger",
                    },
                    {
                        "game_index": 1,
                        "opening_index": 4,
                        "game_within_opening": 1,
                        "challenger_player": 0,
                        "winner": "current",
                    },
                ]
            )

    def test_contract_mismatch_is_a_blocker(self):
        diff = semantic_diff(
            {"exact_root_solver": "python"}, {"exact_root_solver": "native"}
        )
        self.assertEqual(
            "diagnostic_canonical_search_contract_mismatch",
            classify(True, diff, True),
        )

    def test_identity_mismatch_takes_priority(self):
        self.assertEqual(
            "diagnostic_canonical_candidate_identity_mismatch",
            classify(False, {}, True),
        )


if __name__ == "__main__":
    unittest.main()
