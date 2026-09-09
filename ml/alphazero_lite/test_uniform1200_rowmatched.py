"""Focused contracts for the uniform1200 row-count replay ablation."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.run_uniform1200_rowmatched import (
    LANE,
    check_distribution,
    distribution,
    exposure,
    proportional_stratified_sample,
    row_key,
    stable_row_digest,
)


def row(index: int, phase: int, player: int, winner: int) -> dict:
    return {
        "state": [index],
        "move_index": phase,
        "player": player,
        "winner": winner,
        "policy": [0.5, 0.5],
    }


class Uniform1200RowmatchedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            row(index, index % 36, index % 2, (index // 3) % 2) for index in range(300)
        ]

    def test_sampling_is_deterministic_sha_ordered_and_non_mutating(self) -> None:
        original = copy.deepcopy(self.rows)
        first = proportional_stratified_sample(self.rows, 211)
        self.assertEqual(
            first, proportional_stratified_sample(list(reversed(self.rows)), 211)
        )
        self.assertEqual(self.rows, original)
        self.assertEqual(len(first), 211)
        self.assertEqual(
            sorted(stable_row_digest(item) for item in first),
            sorted(
                stable_row_digest(item)
                for item in proportional_stratified_sample(self.rows, 211)
            ),
        )

    def test_sampling_preserves_proportions_within_registered_tolerance(self) -> None:
        sample = proportional_stratified_sample(self.rows, 211)
        comparison = check_distribution(distribution(self.rows), distribution(sample))
        self.assertTrue(comparison["passed"])

    def test_invalid_requested_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            proportional_stratified_sample(self.rows, 301)

    def test_lane_is_the_only_new_training_lane(self) -> None:
        self.assertEqual("uniform1200_rowmatched", LANE)

    def test_effective_exposure_is_rows_times_integer_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dynamic, selected, controls = (
                root / name for name in ("dynamic", "selected", "controls")
            )
            for path, count in ((dynamic, 7), (selected, 2), (controls, 3)):
                path.write_text(
                    "".join(
                        json.dumps(row(index, 0, 0, 0)) + "\n" for index in range(count)
                    )
                )
            report = exposure(
                [
                    ("dynamic", dynamic, 1),
                    ("selected", selected, 1),
                    ("controls", controls, 2),
                ]
            )
        self.assertEqual(15, report["total_effective_sampled_index_count"])
        self.assertEqual(6, report["sources"][2]["effective_sampled_index_count"])
        self.assertEqual(7 / 15, report["sources"][0]["fraction"])

    def test_sampling_is_seed_isolated(self) -> None:
        sample = proportional_stratified_sample(self.rows, 200)
        self.assertEqual(200, len(sample))
        self.assertEqual(
            {"early", "mid", "late"}, {row_key(item)[0] for item in sample}
        )


if __name__ == "__main__":
    unittest.main()
