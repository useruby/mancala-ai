"""Focused tests for the pre-registered mixed scale confirmation."""

import unittest

from ml.alphazero_lite.run_exact_teacher_distribution_scale_confirmation import (
    compact_arena,
    paired_effect_deltas,
    select_scaled_mixed,
)


def rows(prefix: str, count: int) -> list[dict]:
    return [
        {"source_id": f"{prefix}-{index}", "canonical_state": f"{prefix}-{index}"}
        for index in reversed(range(count))
    ]


class ExactTeacherDistributionScaleConfirmationTest(unittest.TestCase):
    def test_selection_is_deterministic_and_has_preregistered_composition(self) -> None:
        opening = rows("opening", 3000)
        midgame = rows("midgame", 3000)
        selected = select_scaled_mixed(opening, midgame)
        self.assertEqual(
            selected,
            select_scaled_mixed(list(reversed(opening)), list(reversed(midgame))),
        )
        self.assertEqual(5880, len(selected))
        self.assertEqual(
            2940, sum(row["canonical_state"].startswith("opening") for row in selected)
        )
        self.assertEqual(
            2940, sum(row["canonical_state"].startswith("midgame") for row in selected)
        )

    def test_overlapping_sources_are_rejected(self) -> None:
        opening = rows("opening", 2940)
        midgame = rows("midgame", 2940)
        midgame[0]["canonical_state"] = opening[0]["canonical_state"]
        with self.assertRaisesRegex(ValueError, "duplicate canonical"):
            select_scaled_mixed(opening, midgame)

    def test_paired_effects_are_compared_with_pr283_mixed_lane(self) -> None:
        budgets = ("standard", "equal_768", "equal_high", "1200_vs_256")
        scaled = {
            "seed42": {
                "arena": {
                    budget: {"paired_candidate_effect": 0.2} for budget in budgets
                }
            }
        }
        baseline = {
            "lane_results": {
                "mixed_50_50": {
                    "seeds": {
                        "seed42": {
                            "arena": {
                                budget: {"paired_candidate_effect": 0.1}
                                for budget in budgets
                            }
                        }
                    }
                }
            }
        }
        self.assertEqual(
            0.1, paired_effect_deltas(scaled, baseline)["seed42"]["standard"]
        )

    def test_summary_arena_is_compact(self) -> None:
        arena = {
            "standard": {
                "paired_candidate_effect": 0.1,
                "ds": 0.2,
                "disadvantaged_seat_score": 0.3,
                "opening_bootstrap_ci": {"lower_95": 0.0, "upper_95": 0.2},
                "best_10_openings": [{"large": "evidence"}],
            }
        }
        self.assertNotIn("best_10_openings", compact_arena(arena)["standard"])


if __name__ == "__main__":
    unittest.main()
