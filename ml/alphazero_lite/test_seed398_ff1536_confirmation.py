"""Regression checks for the frozen FF1536 continuation plan."""

from __future__ import annotations

import unittest
from ml.alphazero_lite import seed398_ff1536_analysis
from ml.alphazero_lite.seed398_ff1536_confirmation import (
    OUT,
    REFERENCES,
    _json,
    _json_lines,
    build_reuse_plan,
    frozen_actions,
)
from ml.alphazero_lite.verify_seed398_ff1536_confirmation import verify


class FF1536ConfirmationTests(unittest.TestCase):
    def test_frozen_mapping_and_reuse_accounting(self) -> None:
        actions = frozen_actions()
        plan = build_reuse_plan()
        self.assertEqual(len(actions), 64)
        self.assertEqual(len(plan), 128)
        self.assertEqual(sum(row["source"] == "reused" for row in plan), 120)
        self.assertEqual(sum(row["source"] == "new" for row in plan), 8)
        self.assertTrue(
            all(
                row["source_ledger_sha256"] and row["source_case_identity"]
                for row in plan
                if row["source"] == "reused"
            )
        )

    def test_only_registered_eight_cases_require_games(self) -> None:
        plan = build_reuse_plan()
        selected = [row for row in plan if row["source"] == "new"]
        cases = sorted(
            (row["opening_index"], row["forced_action"])
            for row in selected
            if row["reference"] == "seed455"
        )
        self.assertEqual(cases, [(164, 2), (217, 0), (260, 1), (372, 4)])
        self.assertEqual(
            sum(row["reference"] == "original_O0_E4" for row in selected), 4
        )

    def test_analysis_rejects_missing_and_duplicate_new_outcomes(self) -> None:
        registration = _json(OUT / "registration.json")
        new_rows = _json_lines(OUT / "new-outcomes.jsonl")
        baselines = {name: _json_lines(path) for name, path in REFERENCES.items()}
        with self.assertRaisesRegex(ValueError, "missing_new_case"):
            seed398_ff1536_analysis.calculate(
                registration,
                new_rows[:-1],
                baselines["seed455"],
                baselines["original_O0_E4"],
            )
        with self.assertRaisesRegex(ValueError, "duplicate_new_case"):
            seed398_ff1536_analysis.calculate(
                registration,
                [*new_rows, new_rows[0]],
                baselines["seed455"],
                baselines["original_O0_E4"],
            )

    def test_read_only_verifier_preserves_publication_bytes(self) -> None:
        published = (
            "registration.json",
            "new-outcomes.jsonl",
            "analysis.json",
            "provenance-matrix.json",
            "results.md",
            "publication-binding.json",
        )
        before = {name: (OUT / name).read_bytes() for name in published}
        result = verify()
        after = {name: (OUT / name).read_bytes() for name in published}
        self.assertEqual(before, after)
        self.assertEqual(result["provenance_cases"], 128)

    def test_verifier_rejects_missing_published_trajectory(self) -> None:
        outcomes = OUT / "new-outcomes.jsonl"
        original = outcomes.read_bytes()
        try:
            outcomes.write_bytes(b"\n".join(original.splitlines()[:-1]) + b"\n")
            with self.assertRaisesRegex(ValueError, "new_outcome_count"):
                verify()
        finally:
            outcomes.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
