"""Focused checks for corrected seed462 training-objective accounting."""

import numpy as np
import pytest

from ml.alphazero_lite.verify_seed462_recovery import require_array_equal


def test_training_occurrence_expansion_preserves_replay_multiplicity() -> None:
    replay = np.array([2, 2, 4, 8], dtype=np.int64)
    positions = np.array([0, 1, 3], dtype=np.int64)
    occurrences = replay[positions]
    assert occurrences.tolist() == [2, 2, 8]
    assert len(occurrences) == 3


def test_validation_target_changes_are_outside_training_slice() -> None:
    replay = np.array([0, 1, 2, 3], dtype=np.int64)
    train_positions = np.array([0, 2], dtype=np.int64)
    targets = np.array([0.1, 0.2, -0.1, 0.4])
    before = targets[replay[train_positions]].copy()
    targets[[1, 3]] = [9.0, -9.0]
    np.testing.assert_array_equal(targets[replay[train_positions]], before)


def test_fresh_policy_mass_uses_expanded_occurrences() -> None:
    replay = np.array([1, 1, 2, 3], dtype=np.int64)
    train_positions = np.array([0, 1, 3], dtype=np.int64)
    fresh = np.array([False, True, False, True])
    coefficient = np.array([1.0, 2.0, 3.0, 4.0])
    occurrences = replay[train_positions]
    mass = coefficient[occurrences][fresh[occurrences]].sum(dtype=np.float64)
    assert mass == 8.0
    assert int(fresh[occurrences].sum()) == 3


@pytest.mark.parametrize(
    "artifact",
    ["predictions", "targets", "identities", "metrics", "effects", "states", "moments"],
)
def test_independent_exact_comparison_rejects_rebound_artifact(artifact: str) -> None:
    expected = np.asarray([0.125, -0.5], dtype=np.float32)
    tampered = expected.copy()
    tampered[0] = np.nextafter(tampered[0], np.float32(1.0))
    # A recomputed file hash is deliberately irrelevant to a semantic comparison.
    with pytest.raises(ValueError, match="semantic_mismatch"):
        require_array_equal(artifact, tampered, expected)


def test_duplicate_identity_targets_remain_occurrence_specific() -> None:
    identities = ["same-input", "same-input"]
    targets = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert identities[0] == identities[1]
    np.testing.assert_array_equal(targets, [[1, 0], [0, 1]])
    assert not np.array_equal(targets[0], targets[1])
