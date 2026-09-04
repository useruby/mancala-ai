import unittest

from ml.alphazero_lite.run_exact_solver_feasibility_preflight import (
    generate_feasibility_corpus,
    run_exactness_validation,
    run_rules_parity,
)


class ExactSolverFeasibilityPreflightTest(unittest.TestCase):
    def test_rules_parity_on_golden_and_reachable_sample(self):
        report = run_rules_parity(sample_count=100, seed=271)

        self.assertTrue(report["passed"])
        self.assertEqual(100, report["reachable_states"])
        self.assertGreater(report["legal_moves_checked"], 100)

    def test_exactness_validation_on_tiny_positions(self):
        report = run_exactness_validation(seed=271)

        self.assertTrue(report["passed"])
        self.assertEqual(12, report["tiny_positions"])

    def test_corpus_is_fresh_balanced_and_training_ineligible(self):
        corpus = generate_feasibility_corpus(seed=271)

        self.assertEqual(96, len(corpus))
        self.assertTrue(all(not row["training_eligible"] for row in corpus))
        self.assertEqual(32, sum(17 <= row["stones_remaining"] <= 24 for row in corpus))
        self.assertEqual(32, sum(25 <= row["stones_remaining"] <= 32 for row in corpus))
        self.assertEqual(32, sum(33 <= row["stones_remaining"] <= 40 for row in corpus))
