"""Tamper-negative tests for the independent Seed 621 reconstruction."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest

from ml.alphazero_lite.verify_seed621_oracle_cohort_audit import (
    assert_inventory_matches,
)


def expected_inventory() -> list[dict]:
    return [
        {
            "state_id": "state-a",
            "state": {"current_player": 0},
            "legal_moves": [1, 2],
            "exact_action_margins_to_move_perspective": {"1": 3, "2": 0},
            "cohorts": ["seed557"],
            "source_clusters": [
                {
                    "opening_index": 17,
                    "source_path": "games-0017.jsonl",
                    "source_line": 1,
                }
            ],
            "eligible_by_exact_membership_rule": True,
            "exclusion_reasons": [],
            "contamination_status": "unresolved_nonmembership_not_proven",
            "B512_legal_probabilities": [0.0, 0.8, 0.2, 0.0, 0.0, 0.0],
            "S615_legal_probabilities": [0.0, 0.3, 0.7, 0.0, 0.0, 0.0],
            "top_prior_agreement": False,
            "B512_top_action": 1,
            "S615_top_action": 2,
            "B512_preferred_action_probability": 0.8,
            "S615_preferred_action_probability": 0.7,
            "B512_mass_on_S615_preferred_action": 0.2,
            "S615_mass_on_B512_preferred_action": 0.3,
            "total_variation": 0.5,
            "jensen_shannon_bits": 0.1187091007693073,
            "B512_entropy_bits": 0.7219280948873623,
            "S615_entropy_bits": 0.8812908992306927,
            "legal_action_count": 2,
        }
    ]


class Seed621IndependentVerifierNegativeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = expected_inventory()
        self.published = copy.deepcopy(self.expected)

    def assert_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "reconstructed_inventory_mismatch"):
            assert_inventory_matches(self.published, self.expected)

    def test_rejects_altered_oracle_action_margin(self) -> None:
        self.published[0]["exact_action_margins_to_move_perspective"]["1"] = 4
        self.assert_rejected()

    def test_rejects_incorrect_membership_exclusion_classification(self) -> None:
        self.published[0]["eligible_by_exact_membership_rule"] = False
        self.published[0]["exclusion_reasons"] = ["S615_training_state_overlap"]
        self.published[0]["contamination_status"] = "excluded_exact_membership"
        self.assert_rejected()

    def test_rejects_prior_class_change_even_if_aggregates_are_recomputed(self) -> None:
        self.published[0]["top_prior_agreement"] = True
        self.published[0]["B512_top_action"] = 2
        self.published[0]["total_variation"] = 0.0
        # Model a tampered publisher updating its summary and inventory digest
        # consistently. The source-derived record is still the authority.
        report = {
            "counts": {"agreement": 1, "disagreement": 0},
            "inventory_sha256": hashlib.sha256(
                (json.dumps(self.published[0], sort_keys=True) + "\n").encode()
            ).hexdigest(),
        }
        self.assertEqual(report["counts"]["agreement"], 1)
        self.assertEqual(
            report["inventory_sha256"],
            hashlib.sha256(
                (json.dumps(self.published[0], sort_keys=True) + "\n").encode()
            ).hexdigest(),
        )
        self.assert_rejected()

    def test_rejects_missing_inventory_state(self) -> None:
        self.published.clear()
        with self.assertRaisesRegex(ValueError, "missing_substituted_or_reordered"):
            assert_inventory_matches(self.published, self.expected)

    def test_rejects_substituted_inventory_state(self) -> None:
        self.published[0]["state_id"] = "state-b"
        with self.assertRaisesRegex(ValueError, "missing_substituted_or_reordered"):
            assert_inventory_matches(self.published, self.expected)


if __name__ == "__main__":
    unittest.main()
