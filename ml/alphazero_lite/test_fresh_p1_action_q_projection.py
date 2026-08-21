from __future__ import annotations

import numpy as np

from ml.alphazero_lite.fresh_p1_action_q_projection import (
    action_q_projected_teacher,
    magnitude_matched_teacher,
)


def test_projection_preserves_policy_and_never_moves_filtered_actions() -> None:
    p0 = np.array([0.5, 0.3, 0.2, 0.0])
    teacher = np.array([0.2, 0.5, 0.3, 0.0])
    projected, supported = action_q_projected_teacher(
        p0,
        teacher,
        np.array([-1.0, 1.0, 0.5, 0.0]),
        np.array([-0.5, 0.4, 0.2, 0.0]),
        np.array([5, 5, 0, 0]),
        np.array([5, 5, 5, 0]),
        np.array([True, True, True, False]),
    )
    assert np.allclose(projected, np.array([0.3, 0.5, 0.2, 0.0]))
    assert not supported[2]
    assert projected[2] == p0[2]
    assert projected[3] == 0.0
    assert np.isclose(projected.sum(), 1.0)


def test_projection_balances_supported_transfers_and_matches_control_magnitude() -> (
    None
):
    p0 = np.array([0.5, 0.3, 0.2])
    teacher = np.array([0.2, 0.5, 0.3])
    projected, supported = action_q_projected_teacher(
        p0,
        teacher,
        np.array([-1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, -1.0]),
        np.ones(3),
        np.ones(3),
        np.ones(3, dtype=bool),
    )
    control = magnitude_matched_teacher(p0, teacher, projected)
    assert supported.tolist() == [True, True, False]
    assert np.allclose(projected, np.array([0.3, 0.5, 0.2]))
    assert np.isclose(np.abs(control - p0).sum(), np.abs(projected - p0).sum())
    assert np.all(control >= 0)
    assert np.isclose(control.sum(), 1.0)
