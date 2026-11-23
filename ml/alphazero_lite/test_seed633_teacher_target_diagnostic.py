"""Synthetic validation tests for Seed 633 diagnostic verification."""

from __future__ import annotations

import unittest

from ml.alphazero_lite import verify_seed633_teacher_target_diagnostic as verifier
from ml.alphazero_lite.kalah_rules import KalahGame


class Seed633VerifierNegativeTests(unittest.TestCase):
    def test_rejects_substituted_numpy_model_artifact(self) -> None:
        expected = {
            "model_npz_sha256": "a" * 64,
            "weights_json_sha256": "b" * 64,
            "metadata_sha256": "c" * 64,
            "checkpoint_sha256": "a" * 64,
        }
        with self.assertRaisesRegex(ValueError, "artifact_identity_mismatch"):
            verifier.validate_artifact_identity(
                "B512", "d" * 64, "b" * 64, "c" * 64, expected
            )

    def test_rejects_altered_teacher_target_in_frozen_cohort(self) -> None:
        expected = [{"canonical_identity": "state-1", "target": [1.0, 0, 0, 0, 0, 0]}]
        observed = [{"canonical_identity": "state-1", "target": [0.5, 0.5, 0, 0, 0, 0]}]
        with self.assertRaisesRegex(ValueError, "cohort_membership_or_target_mismatch"):
            verifier.validate_cohort_rows(observed, expected)

    def test_rejects_invalid_or_incomplete_legal_mask(self) -> None:
        state = KalahGame.from_state(
            {
                "player_pits": [4] * 6,
                "opponent_pits": [4] * 6,
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        ).to_state()
        legal = [0, 1, 2, 3, 4, 5]
        with self.assertRaisesRegex(ValueError, "legal_mask_invalid"):
            verifier.validate_mask(state, legal[:-1])

    def test_rejects_changed_cohort_membership(self) -> None:
        expected = [
            {"canonical_identity": "state-1", "membership_class": "B512_only"},
            {"canonical_identity": "state-2", "membership_class": "S615_only"},
        ]
        observed = expected[:1]
        with self.assertRaisesRegex(ValueError, "cohort_membership_or_target_mismatch"):
            verifier.validate_cohort_rows(observed, expected)

    def test_rejects_teacher_target_with_illegal_action_mass(self) -> None:
        with self.assertRaisesRegex(ValueError, "teacher_target_illegal_mass"):
            verifier.validate_target([0.5, 0.5, 0, 0, 0, 0], [0])


if __name__ == "__main__":
    unittest.main()
