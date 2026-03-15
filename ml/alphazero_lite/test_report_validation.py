import unittest


from ml.alphazero_lite.report_validation import ArenaReportValidationError, validate_arena_report


class ArenaReportValidationTest(unittest.TestCase):
    def valid_report(self):
        return {
            "schema": "arena_v1",
            "games_played": 60,
            "wins": 36,
            "losses": 20,
            "draws": 4,
            "promotion_decision": {"passed": True},
        }

    def test_validate_arena_report_returns_computed_summary(self):
        result = validate_arena_report(report=self.valid_report(), min_score=0.55)

        self.assertTrue(result["passed"])
        self.assertEqual(60, result["games_played"])
        self.assertAlmostEqual((36 + (4 * 0.5)) / 60.0, result["score"])

    def test_validate_arena_report_rejects_decision_mismatch(self):
        report = self.valid_report()
        report["promotion_decision"] = {"passed": False}

        with self.assertRaises(ArenaReportValidationError) as error:
            validate_arena_report(report=report, min_score=0.55)

        self.assertEqual("ARENA_VALIDATION::DECISION_MISMATCH", error.exception.code)

    def test_validate_arena_report_rejects_invalid_counts(self):
        report = self.valid_report()
        report["games_played"] = 0

        with self.assertRaises(ArenaReportValidationError) as error:
            validate_arena_report(report=report, min_score=0.55)

        self.assertEqual("ARENA_VALIDATION::COUNTS", error.exception.code)
