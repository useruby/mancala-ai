import unittest


from ml.alphazero_lite.report_validation import (
    ArenaReportValidationError,
    validate_arena_report,
)


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
        self.assertIn("score threshold", str(error.exception))
        self.assertIn("not confidence gate", str(error.exception))

    def test_validate_arena_report_rejects_invalid_counts(self):
        report = self.valid_report()
        report["games_played"] = 0

        with self.assertRaises(ArenaReportValidationError) as error:
            validate_arena_report(report=report, min_score=0.55)

        self.assertEqual("ARENA_VALIDATION::COUNTS", error.exception.code)

    def test_validate_arena_report_confidence_gate_is_opt_in(self):
        report = {
            "schema": "arena_v1",
            "games_played": 4,
            "wins": 3,
            "losses": 1,
            "draws": 0,
            "promotion_decision": {"passed": True},
        }

        result = validate_arena_report(report=report, min_score=0.55)

        self.assertTrue(result["passed"])
        self.assertIn("confidence_lower_bound", result)

    def test_validate_arena_report_confidence_gate_can_fail_promotion(self):
        report = {
            "schema": "arena_v1",
            "games_played": 4,
            "wins": 3,
            "losses": 1,
            "draws": 0,
            "promotion_decision": {"passed": True},
        }

        result = validate_arena_report(
            report=report, min_score=0.55, min_confidence_lower_bound=0.6
        )

        self.assertFalse(result["passed"])
        self.assertLess(result["confidence_lower_bound"], 0.6)

    def test_validate_arena_report_rejects_boolean_integer_fields(self):
        report = self.valid_report()
        report["wins"] = True

        with self.assertRaises(ArenaReportValidationError) as error:
            validate_arena_report(report=report, min_score=0.55)

        self.assertEqual("ARENA_VALIDATION::SCHEMA", error.exception.code)

    def test_validate_arena_report_rejects_string_integer_fields(self):
        report = self.valid_report()
        report["draws"] = "4"

        with self.assertRaises(ArenaReportValidationError) as error:
            validate_arena_report(report=report, min_score=0.55)

        self.assertEqual("ARENA_VALIDATION::SCHEMA", error.exception.code)
