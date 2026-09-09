"""Focused contracts for PR #289's midgame 2x2 construction."""

from __future__ import annotations

import copy
import unittest

from ml.alphazero_lite.run_uniform1200_midgame_2x2 import (
    LANES,
    control_like_mid_rows,
    phase,
    phase_selection,
    sharpen,
    transform_rows,
    unsharpen,
)


def row(index: int, move_index: int, policy: list[float] | None = None) -> dict:
    # A valid kalah_v3 encoding with a legal first action.
    return {
        "state": [1 / 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [0] * 12,
        "move_index": move_index,
        "policy": policy or [1.0, 0, 0, 0, 0, 0],
        "value": index / 100,
        "row_id": index,
    }


class Uniform1200Midgame2x2Test(unittest.TestCase):
    def test_phase_bucket_semantics(self) -> None:
        self.assertEqual("early", phase(row(0, 9)))
        self.assertEqual("mid", phase(row(0, 10)))
        self.assertEqual("mid", phase(row(0, 29)))
        self.assertEqual("late", phase(row(0, 30)))

    def test_phase_selection_uses_median_with_registered_support(self) -> None:
        audit = {"seeds": {}}
        for seed in (44, 45, 46):
            audit["seeds"][str(seed)] = {
                "comparisons": {
                    "rowmatched_uniform1200": {
                        "matched_targets": {
                            "aggregate": {
                                "phase=early": {"states": 200, "tv": 0.2},
                                "phase=mid": {"states": 201, "tv": 0.4},
                                "phase=late": {"states": 202, "tv": 0.1},
                            }
                        }
                    }
                }
            }
        self.assertEqual("mid", phase_selection(audit)["selected_phase"])

    def test_unsharpen_preserves_zeros_and_round_trips(self) -> None:
        policy = [0.64, 0.36, 0.0, 0.0, 0.0, 0.0]
        restored = unsharpen(policy)
        self.assertAlmostEqual(1.0, sum(restored))
        self.assertEqual(0.0, restored[2])
        for actual, expected in zip(sharpen(restored), policy, strict=True):
            self.assertAlmostEqual(expected, actual, places=12)

    def test_only_selected_phase_policy_changes_and_value_is_identical(self) -> None:
        rows = [row(1, 9), row(2, 10, [0.64, 0.36, 0, 0, 0, 0]), row(3, 30)]
        transformed, _ = transform_rows(
            rows, "mid", exposure="uniform_exposure", target="unsharpened"
        )
        self.assertEqual(rows[0], transformed[0])
        self.assertEqual(rows[2], transformed[2])
        self.assertEqual(rows[1]["value"], transformed[1]["value"])
        self.assertNotEqual(rows[1]["policy"], transformed[1]["policy"])
        self.assertIn("experiment_target_provenance", transformed[1])

    def test_control_like_sampling_is_deterministic_and_preserves_rows(self) -> None:
        uniform = [row(index, 10) for index in range(8)]
        control = [row(index, 10) for index in range(4)]
        first, report = control_like_mid_rows(uniform, control)
        second, _ = control_like_mid_rows(
            list(reversed(uniform)), list(reversed(control))
        )
        self.assertEqual(8, len(first))
        self.assertEqual(
            [item["row_id"] for item in first], [item["row_id"] for item in second]
        )
        self.assertEqual(0.0, report["original_tv"])
        self.assertEqual(0.0, report["control_like_tv"])

    def test_factorial_lane_contract_and_seed_isolation(self) -> None:
        self.assertEqual(4, len(LANES))
        first = [row(1, 10), row(2, 9)]
        second = copy.deepcopy(first)
        transformed, _ = transform_rows(
            first, "mid", exposure="uniform_exposure", target="sharpened"
        )
        self.assertEqual(second, transformed)


if __name__ == "__main__":
    unittest.main()
