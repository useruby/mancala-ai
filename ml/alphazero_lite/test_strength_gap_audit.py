import unittest
import tempfile
from pathlib import Path

from ml.alphazero_lite.strength_gap_audit import (
    exact_regret,
    paired_roots,
    paired_summary,
    validate_oracle,
    validate_pair_keys,
    validate_trusted_digest,
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
        linear = paired_summary(
            [(1, 1.0), (2, 2.0)], seed=11, draws=100, quantile="linear"
        )
        self.assertEqual(linear["quantile_convention"], "linear")

    def test_paired_roots_reject_missing_and_duplicate_policy_records(self):
        complete = [
            {"state_hash": "r", "opening_index": 1, "policy": "A", "metric": 2},
            {"state_hash": "r", "opening_index": 1, "policy": "B", "metric": 1},
        ]
        self.assertEqual(
            paired_roots(complete, "A", "B", "metric", 1, expected_roots={"r"})["mean"],
            1,
        )
        with self.assertRaises(ValueError):
            paired_roots(complete[:1], "A", "B", "metric", 1)
        with self.assertRaises(ValueError):
            paired_roots(complete + [complete[0]], "A", "B", "metric", 1)
        with self.assertRaises(ValueError):
            paired_roots(
                complete, "A", "B", "metric", 1, expected_roots={"r", "absent"}
            )

    def test_trusted_input_hash_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text("original")
            with self.assertRaises(ValueError):
                validate_trusted_digest(path, "0" * 64)

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
        self.assertEqual(aggregate["cohort_577"]["decision_changes"], 32)
        self.assertEqual(sum(row["changed_decision"] for row in roots["seed577"]), 32)
        self.assertIn("by_active_pit_stones", aggregate["cohort_577"])


if __name__ == "__main__":
    unittest.main()
