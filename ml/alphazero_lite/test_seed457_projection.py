"""Synthetic coverage for seed457 joint projection arithmetic."""

import numpy as np
import pytest

from ml.alphazero_lite.seed457_projection import project, projected_proposal


def test_inactive_projection_and_zero_normal() -> None:
    z, report = project(
        [np.array([1.0, 2.0])], [[np.array([-1.0, 0.0])], [np.zeros(2)]]
    )
    np.testing.assert_array_equal(z[0], [1.0, 2.0])
    assert report["zero_normals"] == [1]


def test_duplicate_rank_deficient_and_opposing_constraints() -> None:
    z, _ = project(
        [np.array([2.0, 1.0])],
        [[np.array([1.0, 0.0])], [np.array([2.0, 0.0])], [np.array([-1.0, 0.0])]],
    )
    np.testing.assert_allclose(z[0], [0.0, 1.0], atol=1e-12)


def test_joint_solution_is_nearest_not_sequential() -> None:
    z, report = project(
        [np.array([1.0, 1.0])],
        [[np.array([1.0, 0.0])], [np.array([1.0, 1.0])]],
    )
    np.testing.assert_allclose(z[0], [0.0, 0.0], atol=1e-12)
    assert np.all(np.asarray(report["slacks"]) <= 1e-10)


def test_invalid_constraint_fails_closed() -> None:
    with pytest.raises(ValueError, match="nonfinite_constraint"):
        project([np.array([1.0])], [[np.array([np.nan])]])


def test_zero_normals_preserve_float64_until_state_writeback() -> None:
    pre = [np.array([1.0], dtype=np.float32)]
    raw = [np.array([1.0 + 1e-8], dtype=np.float64)]
    result, _ = projected_proposal(pre, raw, [[np.zeros(1, dtype=np.float32)]])
    expected = np.asarray(
        pre[0].astype(np.float64)
        + raw[0].astype(np.float64)
        - pre[0].astype(np.float64),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result[0], expected)
