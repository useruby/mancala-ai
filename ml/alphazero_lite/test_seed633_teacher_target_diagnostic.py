"""Synthetic validation tests for Seed 633 diagnostic verification."""

from __future__ import annotations

import unittest
from pathlib import Path

from ml.alphazero_lite import verify_seed633_teacher_target_diagnostic as verifier
from ml.alphazero_lite.kalah_rules import KalahGame


class Seed633VerifierNegativeTests(unittest.TestCase):
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

    def _artifact_fixture(self):
        root = Path("/repository")
        relative_directory = "docs/data/seed615/evaluation/runtime-artifacts/B512"
        checkpoint = verifier.EXPECTED_CHECKPOINTS["B512"]
        seed_record = {
            "artifact_directory": relative_directory,
            "checkpoint_sha256": checkpoint,
            "model_npz_sha256": checkpoint,
            "weights_json_sha256": "w" * 64,
            "metadata_sha256": "m" * 64,
        }
        evaluation_record = {
            "directory": relative_directory,
            "model_npz_sha256": checkpoint,
            "weights_json_sha256": "w" * 64,
            "metadata_sha256": "m" * 64,
        }
        actual_hashes = {
            "model_npz_sha256": checkpoint,
            "weights_json_sha256": "w" * 64,
            "metadata_sha256": "m" * 64,
        }
        metadata = {
            "framework": "numpy",
            "artifacts": {
                "weights_file": "weights.json",
                "weights_sha256": checkpoint,
                "weights_json_sha256": "w" * 64,
            },
        }
        return root, seed_record, evaluation_record, actual_hashes, metadata, checkpoint

    def test_rejects_model_npz_substitution_even_when_seed_freeze_is_updated(self):
        root, seed_record, evaluation_record, actual_hashes, metadata, checkpoint = (
            self._artifact_fixture()
        )
        seed_record["model_npz_sha256"] = "x" * 64
        actual_hashes["model_npz_sha256"] = "x" * 64
        with self.assertRaisesRegex(ValueError, "artifact_identity_mismatch"):
            verifier.validate_numpy_artifact_binding(
                "B512",
                seed_record,
                evaluation_record,
                root / evaluation_record["directory"],
                root,
                actual_hashes,
                metadata,
                checkpoint,
            )

    def test_accepts_artifacts_bound_to_seed615_evaluation_freeze(self) -> None:
        root, seed_record, evaluation_record, hashes, metadata, checkpoint = (
            self._artifact_fixture()
        )
        verifier.validate_numpy_artifact_binding(
            "B512",
            seed_record,
            evaluation_record,
            root / evaluation_record["directory"],
            root,
            hashes,
            metadata,
            checkpoint,
        )

    def test_rejects_weights_substitution_even_when_seed_freeze_is_updated(self):
        root, seed_record, evaluation_record, actual_hashes, metadata, checkpoint = (
            self._artifact_fixture()
        )
        seed_record["weights_json_sha256"] = "x" * 64
        actual_hashes["weights_json_sha256"] = "x" * 64
        metadata["artifacts"]["weights_json_sha256"] = "x" * 64
        with self.assertRaisesRegex(ValueError, "artifact_identity_mismatch"):
            verifier.validate_numpy_artifact_binding(
                "B512",
                seed_record,
                evaluation_record,
                root / evaluation_record["directory"],
                root,
                actual_hashes,
                metadata,
                checkpoint,
            )

    def test_rejects_metadata_substitution_even_when_seed_freeze_is_updated(self):
        root, seed_record, evaluation_record, actual_hashes, metadata, checkpoint = (
            self._artifact_fixture()
        )
        seed_record["metadata_sha256"] = "x" * 64
        actual_hashes["metadata_sha256"] = "x" * 64
        with self.assertRaisesRegex(ValueError, "artifact_identity_mismatch"):
            verifier.validate_numpy_artifact_binding(
                "B512",
                seed_record,
                evaluation_record,
                root / evaluation_record["directory"],
                root,
                actual_hashes,
                metadata,
                checkpoint,
            )

    def test_rejects_substituted_artifact_directory(self):
        root, seed_record, evaluation_record, actual_hashes, metadata, checkpoint = (
            self._artifact_fixture()
        )
        seed_record["artifact_directory"] = "docs/data/replacement"
        with self.assertRaisesRegex(ValueError, "artifact_directory_identity_mismatch"):
            verifier.validate_numpy_artifact_binding(
                "B512",
                seed_record,
                evaluation_record,
                root / "docs/data/replacement",
                root,
                actual_hashes,
                metadata,
                checkpoint,
            )

    def test_rejects_weights_json_digest_mismatch_in_metadata(self):
        root, seed_record, evaluation_record, actual_hashes, metadata, checkpoint = (
            self._artifact_fixture()
        )
        metadata["artifacts"]["weights_json_sha256"] = "x" * 64
        with self.assertRaisesRegex(ValueError, "artifact_metadata_binding_mismatch"):
            verifier.validate_numpy_artifact_binding(
                "B512",
                seed_record,
                evaluation_record,
                root / evaluation_record["directory"],
                root,
                actual_hashes,
                metadata,
                checkpoint,
            )

    def test_rejects_substituted_self_play_source_even_if_seed_freeze_changes(self):
        with self.assertRaisesRegex(ValueError, "self_play_source_identity_mismatch"):
            verifier.validate_self_play_source_identity(
                "x" * 64,
                "x" * 64,
                verifier.EXPECTED_SELF_PLAY_SHA256,
            )


if __name__ == "__main__":
    unittest.main()
