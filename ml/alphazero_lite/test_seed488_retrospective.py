"""Focused accounting tests for seed488 retrospective analysis."""

import unittest

from ml.alphazero_lite.seed488_retrospective import analyze


class Seed488RetrospectiveTests(unittest.TestCase):
    def test_complete_accounting(self):
        from pathlib import Path

        result = analyze(Path.cwd())
        self.assertEqual(
            (
                result["opening_count"],
                result["game_count"],
                result["ledger_game_count"],
            ),
            (512, 1024, 1024),
        )
        self.assertEqual(sum(result["outcomes"].values()), 1024)
        self.assertAlmostEqual(result["score"], 0.4892578125)
        self.assertEqual(sum(result["pair_score_distribution"].values()), 512)

    def test_telemetry_and_first_action_accounting(self):
        from pathlib import Path

        result = analyze(Path.cwd())
        self.assertEqual(result["handoff"]["games_with_any"], 953)
        self.assertEqual(result["paired_first_action"]["compared_openings"], 512)
        self.assertEqual(
            result["paired_first_action"]["challenger_agreements"]
            + result["paired_first_action"]["challenger_disagreements"],
            512,
        )


if __name__ == "__main__":
    unittest.main()
