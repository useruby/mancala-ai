from __future__ import annotations

import unittest

from ml.alphazero_lite.run_value_delta_blend_suite_completion import (
    BUDGETS,
    classify,
    deduplicate_lanes,
)


def completed_candidate(delta_384: float = 0.0) -> dict:
    by_budget = {
        budget: {
            "candidate_minus_current_ds": delta_384 if budget == "384:256" else 0.0,
            "raw_candidate_ds": 0.0,
            "raw_current_ds": 0.0,
        }
        for budget in BUDGETS
    }
    return {
        "role": "candidate_lane",
        "completed": True,
        "per_opening_artifact": "/tmp/per-opening",
        "by_budget": by_budget,
    }


class ValueDeltaBlendSuiteCompletionTest(unittest.TestCase):
    def test_global025_is_deduplicated_with_alpha025(self):
        lanes, aliases = deduplicate_lanes(["alpha025", "global025"])
        self.assertEqual(lanes, ["blend_alpha_025"])
        self.assertEqual(aliases, {"global025": "blend_alpha_025"})

    def test_passing_probe_without_medium_is_incomplete(self):
        summary = {
            "probe_passing_candidates": ["blend_alpha_025"],
            "stages": {
                "medium": {"status": "stopped_by_probe_gates"},
                "fixed_large": {},
                "heldout": {},
            },
            "gate": {},
        }
        self.assertEqual(classify(summary), "value_delta_blend_suite_incomplete")

    def test_status_only_medium_cannot_be_strength_neutral(self):
        summary = {
            "probe_passing_candidates": ["blend_alpha_025"],
            "stages": {
                "medium": {"status": "completed", "lanes": {}},
                "fixed_large": {},
                "heldout": {},
            },
            "gate": {},
        }
        self.assertEqual(classify(summary), "value_delta_blend_suite_incomplete")

    def test_safe_but_neutral_requires_completed_candidate_results(self):
        candidate = completed_candidate(0.02)
        summary = {
            "probe_passing_candidates": ["blend_alpha_025"],
            "stages": {
                "medium": {
                    "status": "completed",
                    "lanes": {"blend_alpha_025": candidate},
                },
                "fixed_large": {},
                "heldout": {},
            },
            "gate": {},
        }
        self.assertEqual(classify(summary), "blended_value_safe_but_strength_neutral")
