"""Focused tests for the equal-budget exact-teacher distribution ablation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import exact_teacher_labeling as exact
from ml.alphazero_lite.run_exact_teacher_distribution_ablation import (
    gate_pass,
    lane_integrity,
    select_lanes,
    verify_leakage,
    write_lane_files,
)


def rows(prefix: str, count: int) -> list[dict]:
    return [
        {
            "source_id": f"{prefix}-{index:04d}",
            "canonical_state": f"{prefix}-canonical-{index:04d}",
            "teacher": "native_hybrid_exact",
            "teacher_version": "pr281_production_v1",
        }
        for index in reversed(range(count))
    ]


class ExactTeacherDistributionSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.opening = rows("opening", 3000)
        self.midgame = rows("midgame", 3000)

    def test_selection_is_deterministic_and_equal_budget(self) -> None:
        left = select_lanes(self.opening, self.midgame)
        right = select_lanes(list(reversed(self.opening)), list(reversed(self.midgame)))
        self.assertEqual(left, right)
        self.assertEqual(2940, len(left["opening_100"]))
        self.assertEqual(2940, len(left["midgame_100"]))
        self.assertEqual(2940, len(left["mixed_50_50"]))

    def test_mixed_composition_and_nested_subsets(self) -> None:
        selected = select_lanes(self.opening, self.midgame)
        mixed_keys = {row["canonical_state"] for row in selected["mixed_50_50"]}
        opening_keys = {row["canonical_state"] for row in selected["opening_100"]}
        midgame_keys = {row["canonical_state"] for row in selected["midgame_100"]}
        self.assertEqual(1470, len(mixed_keys & opening_keys))
        self.assertEqual(1470, len(mixed_keys & midgame_keys))

    def test_duplicate_canonical_state_is_rejected(self) -> None:
        duplicate = list(self.opening)
        duplicate[1] = dict(
            duplicate[1], canonical_state=duplicate[0]["canonical_state"]
        )
        with self.assertRaisesRegex(ValueError, "duplicate canonical"):
            select_lanes(duplicate, self.midgame)

    def test_holdout_and_suite_leakage_are_rejected(self) -> None:
        selected = select_lanes(self.opening, self.midgame)
        with patch(
            "ml.alphazero_lite.run_exact_teacher_distribution_ablation.suite_canonical_keys",
            return_value={selected["opening_100"][0]["canonical_state"]},
        ):
            with self.assertRaisesRegex(ValueError, "evaluation suite"):
                verify_leakage(selected, {"opening": [], "midgame": []}, Path("suite"))
        with patch(
            "ml.alphazero_lite.run_exact_teacher_distribution_ablation.suite_canonical_keys",
            return_value=set(),
        ):
            with self.assertRaisesRegex(ValueError, "opening holdout"):
                verify_leakage(
                    selected,
                    {"opening": [selected["opening_100"][0]], "midgame": []},
                    Path("suite"),
                )

    def test_lane_sha_is_reproducible_and_integrity_records_composition(self) -> None:
        selected = select_lanes(self.opening, self.midgame)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = root / "suite.jsonl"
            suite.write_text("", encoding="utf-8")
            first = write_lane_files(root / "first", selected)
            second = write_lane_files(root / "second", selected)
            self.assertEqual(
                exact.sha256_file(first["mixed_50_50"]),
                exact.sha256_file(second["mixed_50_50"]),
            )
            with patch(
                "ml.alphazero_lite.run_exact_teacher_distribution_ablation.suite_canonical_keys",
                return_value=set(),
            ):
                integrity = lane_integrity(
                    selected, None, None, {"opening": [], "midgame": []}, suite
                )
        mixed = integrity["lanes"]["mixed_50_50"]
        self.assertEqual({"opening": 1470, "midgame": 1470}, mixed["composition"])
        self.assertEqual(0, mixed["exact_duplicate_count"])
        self.assertEqual(2940, mixed["canonical_unique_count"])

    def test_gate_uses_frozen_pr282_ds_rule_not_paired_effect(self) -> None:
        budgets = {
            "standard": {"ds": 0.01, "paired_candidate_effect": -0.9},
            "equal_768": {"ds": 0.0, "paired_candidate_effect": -0.9},
            "equal_high": {"ds": 0.0, "paired_candidate_effect": -0.9},
            "1200_vs_256": {"ds": 0.0, "paired_candidate_effect": -0.9},
        }
        passed, checks = gate_pass(budgets)
        self.assertTrue(passed)
        self.assertTrue(all(checks.values()))


if __name__ == "__main__":
    unittest.main()
