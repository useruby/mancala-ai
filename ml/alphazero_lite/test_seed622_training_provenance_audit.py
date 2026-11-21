"""Synthetic negative checks for Seed 622's fail-closed exposure contract."""

from __future__ import annotations

import unittest

from ml.alphazero_lite.seed622_training_provenance_audit import classify_exposure
from ml.alphazero_lite.verify_seed622_training_provenance_audit import (
    validate_complete_consumption,
    validate_source_provenance,
    validate_state_identity,
)


class Seed622ExposureTests(unittest.TestCase):
    def test_nonmatch_alone_remains_unresolved(self) -> None:
        self.assertEqual(classify_exposure({}), "unresolved_insufficient_provenance")

    def test_canonical_training_identity_excludes(self) -> None:
        self.assertEqual(
            classify_exposure({"S615_canonical": True}),
            "excluded_confirmed_exposure",
        )

    def test_float32_collision_excludes_even_when_canonical_differs(self) -> None:
        self.assertEqual(
            classify_exposure({"B512_float32": True}),
            "excluded_confirmed_exposure",
        )

    def test_protected_population_overlap_excludes_from_candidate_pool(self) -> None:
        self.assertEqual(
            classify_exposure({"protected_holdout_float32": True}),
            "excluded_confirmed_exposure",
        )

    def test_independent_verifier_rejects_corrupted_source_game_provenance(
        self,
    ) -> None:
        source = {
            "cohorts": ["seed559"],
            "source_clusters": [{"source_path": "game-a", "opening_index": 9}],
            "top_prior_agreement": False,
        }
        valid = {
            "cohorts": ["seed559"],
            "source_clusters": [{"source_path": "game-a", "opening_index": 9}],
            "prior_class": "disagreement",
        }
        validate_source_provenance(valid, source)
        valid["source_clusters"][0]["opening_index"] = 10
        with self.assertRaisesRegex(ValueError, "corrupted_source_provenance"):
            validate_source_provenance(valid, source)

    def test_independent_verifier_rejects_omitted_training_membership_row(self) -> None:
        validate_complete_consumption({0, 1, 2}, 3)
        with self.assertRaisesRegex(
            ValueError, "unconsumed_registered_membership_rows"
        ):
            validate_complete_consumption({0, 2}, 3)

    def test_independent_verifier_rejects_identity_mismatch(self) -> None:
        state = {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        with self.assertRaisesRegex(ValueError, "state_identity_mismatch"):
            validate_state_identity(state, "wrong-canonical", "wrong-float32")


if __name__ == "__main__":
    unittest.main()
