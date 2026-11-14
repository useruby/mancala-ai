import unittest

from ml.alphazero_lite.strength_gap_audit import (
    exact_regret,
    paired_summary,
    validate_oracle,
    validate_pair_keys,
)


class StrengthGapAuditTests(unittest.TestCase):
    def test_exact_regret_and_wdl_downgrade(self):
        self.assertEqual(exact_regret({0: 3, 1: -2}, 1), (5, True))
        self.assertEqual(exact_regret({0: 3, 1: 1}, 1), (2, False))

    def test_incomplete_oracle_is_rejected(self):
        with self.assertRaises(ValueError):
            exact_regret({0: 3}, 1)
        with self.assertRaises(ValueError):
            validate_oracle({0: 3}, {0, 1}, True)
        with self.assertRaises(ValueError):
            validate_oracle({0: 3}, {0}, False)

    def test_opening_pairing_clusters_rows(self):
        result = paired_summary([(1, 2.0), (1, 4.0), (2, 8.0)], seed=1, draws=100)
        self.assertEqual(result["n_openings"], 2)
        self.assertEqual(result["mean"], 5.5)

    def test_pairing_and_evidence_corruption_are_rejected(self):
        one = {"root": {"opening_index": 4}}
        validate_pair_keys(one, {"root": {"opening_index": 4}})
        with self.assertRaises(ValueError):
            validate_pair_keys(one, {"other": {"opening_index": 4}})
        with self.assertRaises(ValueError):
            validate_pair_keys(one, {"root": {"opening_index": 5}})

    def test_evidence_corruption_and_pairing_rejected_by_analyzer(self):
        from ml.alphazero_lite.strength_gap_audit import analyze

        aggregate, roots, _ = analyze()
        self.assertEqual(len(roots["seed557"]), 768)
        self.assertEqual(len(roots["seed558"]), 256)
        self.assertEqual(len(roots["seed577"]), 423)
        self.assertEqual(aggregate["cohort_558"]["changed_decisions"], 9)
        self.assertAlmostEqual(
            aggregate["cohort_577"]["regret_contrast"]["mean"], -0.07092198581560284
        )


if __name__ == "__main__":
    unittest.main()
