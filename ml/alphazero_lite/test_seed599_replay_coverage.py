"""Synthetic mutation coverage for Seed 599's preparation invariants."""

from __future__ import annotations

import copy
import unittest

from ml.alphazero_lite import seed599_replay_coverage as seed599
from ml.alphazero_lite.kalah_rules import KalahGame


def _record(stones: int = 48) -> dict:
    state = {
        "player_pits": [stones, 0, 0, 0, 0, 0],
        "opponent_pits": [0, 0, 0, 0, 0, 0],
        "player_store": 0,
        "opponent_store": 48 - stones,
        "current_player": 0,
    }
    legal = KalahGame.from_state(state).possible_moves()
    visits = [0] * 6
    visits[legal[0]] = 128
    target = [0.0] * 6
    target[legal[0]] = 1.0
    return {
        "canonical_state": state,
        "legal_moves": legal,
        "stone_count": stones,
        "move_index": 10,
        "target": target,
        "teacher_evidence": {
            "teacher_source": "artifact_puct",
            "target_visits": visits,
            "target_temperature": 0.1,
        },
    }


class Seed599PreparationTests(unittest.TestCase):
    def test_phase_boundaries_are_exact(self) -> None:
        self.assertEqual(seed599.phase(33), "above_32")
        self.assertEqual(seed599.phase(32), "17_to_32")
        self.assertEqual(seed599.phase(17), "17_to_32")
        self.assertEqual(seed599.phase(16), "at_most_16")

    def test_target_reconstruction_rejects_mutated_teacher_visits(self) -> None:
        record = _record()
        seed599.validate_target_record(record)
        mutated = copy.deepcopy(record)
        mutated["teacher_evidence"]["target_visits"] = [0, 0, 0, 0, 0, 128]
        with self.assertRaisesRegex(
            ValueError, "puct_target_visits_include_illegal_move"
        ):
            seed599.validate_target_record(mutated)

    def test_target_reconstruction_rejects_illegal_policy_mass(self) -> None:
        record = _record()
        record["target"] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        with self.assertRaises(ValueError):
            seed599.validate_target_record(record)

    def test_target_reconstruction_rejects_legal_index_mutation(self) -> None:
        record = _record()
        record["legal_moves"] = []
        with self.assertRaisesRegex(ValueError, "legal_action_index_mismatch"):
            seed599.validate_target_record(record)

    def test_membership_rejects_phase_count_mutation(self) -> None:
        with self.assertRaisesRegex(ValueError, "treatment_phase_count_mismatch"):
            seed599.validate_membership([], [])

    def test_conflicting_duplicate_state_targets_are_counted(self) -> None:
        first = {
            "canonical_identity": "same",
            "target_sha256": "one",
            "source": "#580",
            "game_id": "g",
            "row_id": "r",
        }
        second = {**first, "target_sha256": "two", "source": "#585"}
        _unique, conflicts = seed599.deduplicate([first, second])
        self.assertEqual(conflicts, 1)

    def test_cross_group_duplicate_is_attributed_to_heldout_occurrence(self) -> None:
        base = {
            "canonical_identity": "same-canonical",
            "float32_identity": "same-float32",
            "target_sha256": "same-target",
            "source": "#580",
            "game_id": "game-a",
            "row_id": "row-a",
            "phase": "above_32",
        }
        training_occurrence = {**base, "opening_id": "training-opening"}
        heldout_occurrence = {
            **base,
            "opening_id": "heldout-opening",
            "game_id": "game-b",
            "row_id": "row-b",
        }

        training, heldout = seed599.split_provenance_records(
            [training_occurrence, heldout_occurrence],
            {("#580", "heldout-opening")},
        )

        self.assertEqual(training, {})
        self.assertEqual(len(heldout), 1)
        self.assertEqual(heldout[0]["opening_id"], "heldout-opening")
        self.assertEqual(heldout[0]["game_id"], "game-b")

    def test_float32_duplicate_across_groups_is_excluded_from_training(self) -> None:
        heldout = {
            "canonical_identity": "heldout-canonical",
            "float32_identity": "shared-float32",
            "target_sha256": "target-a",
            "source": "#580",
            "game_id": "game-heldout",
            "row_id": "row-heldout",
            "opening_id": "heldout-opening",
            "phase": "above_32",
        }
        training = {
            **heldout,
            "canonical_identity": "training-canonical",
            "target_sha256": "target-b",
            "game_id": "game-training",
            "row_id": "row-training",
            "opening_id": "training-opening",
        }

        training_by_identity, heldout_rows = seed599.split_provenance_records(
            [training, heldout], {("#580", "heldout-opening")}
        )

        self.assertEqual(training_by_identity, {})
        self.assertEqual(
            [row["opening_id"] for row in heldout_rows], ["heldout-opening"]
        )

    def test_split_validation_rejects_training_overlap_under_both_identities(
        self,
    ) -> None:
        heldout = {
            "canonical_identity": "heldout-canonical",
            "float32_identity": "heldout-float32",
        }
        with self.assertRaisesRegex(ValueError, "heldout_treatment_canonical_overlap"):
            seed599.validate_training_heldout_disjointness([{**heldout}], [], [heldout])
        with self.assertRaisesRegex(ValueError, "heldout_control_float32_overlap"):
            seed599.validate_training_heldout_disjointness(
                [], [{**heldout, "canonical_identity": "other-canonical"}], [heldout]
            )


if __name__ == "__main__":
    unittest.main()
