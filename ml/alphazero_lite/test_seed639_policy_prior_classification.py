"""Synthetic regression tests for Seed 639 policy comparison contracts."""

from __future__ import annotations

import unittest

import numpy as np

from ml.alphazero_lite import seed639_policy_prior_classification as diagnostic


class Seed639PolicyPriorClassificationTests(unittest.TestCase):
    def test_legal_masking_zeroes_illegal_actions_and_normalizes(self):
        result = diagnostic.compare_legal_priors(
            [0.1, 0.2, 0.3, 0.1, 0.1, 0.2],
            [0.2, 0.1, 0.2, 0.2, 0.1, 0.2],
            [0, 2, 5],
        )
        np.testing.assert_allclose(
            result["B512_legal_probabilities"], [1 / 6, 0, 1 / 2, 0, 0, 1 / 3]
        )
        self.assertEqual(result["B512_legal_probabilities"][1], 0)
        self.assertEqual(sum(result["S615_legal_probabilities"]), 1.0)
        self.assertAlmostEqual(result["total_variation"], 1 / 6)

    def test_first_legal_index_wins_a_top_action_tie(self):
        result = diagnostic.compare_legal_priors(
            [0.5, 0.5, 0, 0, 0, 0],
            [0.5, 0.5, 0, 0, 0, 0],
            [0, 1],
        )
        self.assertEqual(result["B512_top_action"], 0)
        self.assertFalse(result["top_action_disagreement"])

    def test_invalid_probabilities_and_legal_actions_fail_closed(self):
        invalid = (
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.3],
            [0.2, 0.2, 0.2, 0.2, 0.2, -0.1],
            [float("nan"), 0.2, 0.2, 0.2, 0.2, 0.2],
        )
        for vector in invalid:
            with self.subTest(vector=vector):
                with self.assertRaisesRegex(
                    ValueError, "invalid_raw_probability_vector"
                ):
                    diagnostic.compare_legal_priors(vector, [1 / 6] * 6, [0, 1])
        with self.assertRaisesRegex(ValueError, "invalid_legal_action_set"):
            diagnostic.compare_legal_priors([1 / 6] * 6, [1 / 6] * 6, [6])

    def test_subset_caps_and_hash_tie_break_are_deterministic(self):
        rows = []
        for index in range(80):
            rows.append(
                {
                    "canonical_identity": f"state-{index:03d}",
                    "game_id": f"game-{index:03d}",
                    "opening_identity": f"opening-{index // 5:03d}",
                    "total_variation": 0.75 if index < 70 else 0.25,
                }
            )
        first = diagnostic.select_diagnostic_subset(rows)
        second = diagnostic.select_diagnostic_subset(rows)
        self.assertEqual(first, second)
        self.assertTrue(first["available"])
        self.assertEqual(first["selected_count"], 64)
        self.assertEqual(first["distinct_continuation_games"], 64)
        self.assertTrue(
            all(count <= 4 for count in first["states_per_opening_cluster"].values())
        )
        ranked = sorted(
            rows,
            key=lambda row: (
                -row["total_variation"],
                diagnostic.sha256_bytes(row["canonical_identity"].encode()),
                row["canonical_identity"],
            ),
        )
        expected = []
        opening_counts = {}
        for row in ranked:
            opening = row["opening_identity"]
            if opening_counts.get(opening, 0) >= 4:
                continue
            expected.append(row)
            opening_counts[opening] = opening_counts.get(opening, 0) + 1
            if len(expected) == 64:
                break
        self.assertEqual(
            first["selected_state_ids"],
            [row["canonical_identity"] for row in expected],
        )

    def test_subset_unavailable_below_distinct_game_gate(self):
        rows = [
            {
                "canonical_identity": f"state-{index}",
                "game_id": f"game-{index % 31}",
                "opening_identity": f"opening-{index}",
                "total_variation": 0.5,
            }
            for index in range(64)
        ]
        result = diagnostic.select_diagnostic_subset(rows)
        self.assertFalse(result["available"])
        self.assertEqual(result["selected_state_ids"], [])

    def test_summary_uses_equal_fixed_bin_weights(self):
        rows = []
        for stones, disagreement in zip((17, 18, 19, 20, 21), (0, 1, 0, 1, 0)):
            for index in range(102):
                rows.append(
                    {
                        "active_pit_stones": stones,
                        "total_variation": float(disagreement),
                        "top_action_disagreement": bool(disagreement),
                        "B512_entropy_bits": float(disagreement),
                        "S615_entropy_bits": float(disagreement),
                        "entropy_difference_S615_minus_B512_bits": float(disagreement),
                    }
                )
        rows.extend(
            [
                {
                    "active_pit_stones": 17,
                    "total_variation": 0.0,
                    "top_action_disagreement": False,
                    "B512_entropy_bits": 0.0,
                    "S615_entropy_bits": 0.0,
                    "entropy_difference_S615_minus_B512_bits": 0.0,
                },
                {
                    "active_pit_stones": 18,
                    "total_variation": 1.0,
                    "top_action_disagreement": True,
                    "B512_entropy_bits": 1.0,
                    "S615_entropy_bits": 1.0,
                    "entropy_difference_S615_minus_B512_bits": 1.0,
                },
            ]
        )
        summary = diagnostic.summarize_population(rows)
        self.assertEqual(summary["population_size"], 512)
        self.assertAlmostEqual(
            summary["fixed_one_fifth_bin_standardized"]["top_action_disagreement_rate"],
            0.4,
        )
        self.assertEqual(len(summary["overall"]["total_variation_quantiles"]), 8)


if __name__ == "__main__":
    unittest.main()
