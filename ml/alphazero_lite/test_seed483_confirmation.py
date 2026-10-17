"""Tests for the corrected, prospective seed483 confirmation contract."""

from __future__ import annotations

import numpy as np
import pytest

from ml.alphazero_lite.seed483_confirmation import (
    BOOTSTRAP_REPLICATES,
    cluster_bootstrap,
    confirmation_decision,
    empirical_inverse_cdf,
)


def test_duplicate_game_draws_are_preserved() -> None:
    calls: list[np.ndarray] = []

    def statistic(rows: np.ndarray) -> float:
        calls.append(rows.copy())
        return float(len(rows))

    result = cluster_bootstrap(["a", "b"], statistic)
    assert len(result) == BOOTSTRAP_REPLICATES
    assert all(len(rows) == 2 for rows in calls)
    assert any(len(set(rows.tolist())) < len(rows) for rows in calls)


def test_cluster_bootstrap_reproducibility_and_protocol_is_fixed() -> None:
    first = cluster_bootstrap(["a", "b", "c"], lambda rows: float(rows.sum()))
    second = cluster_bootstrap(["a", "b", "c"], lambda rows: float(rows.sum()))
    np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError, match="bootstrap_protocol_mismatch"):
        cluster_bootstrap(["a"], lambda rows: 0.0, replicates=2)


def test_empirical_inverse_cdf_order_statistic() -> None:
    assert empirical_inverse_cdf([9, 1, 5, 3], 0.5) == 3
    assert empirical_inverse_cdf([9, 1, 5, 3], 0.0) == 1
    assert empirical_inverse_cdf([9, 1, 5, 3], 1.0) == 9


def _ce(improvement: float = 0.001, upper: float = -0.0001) -> dict:
    return {
        "F_initializer_exposure": (improvement, upper),
        "F_M_exposure": (improvement, upper),
        "F_initializer_equal_input": (improvement, upper),
        "F_M_equal_input": (improvement, upper),
    }


def test_all_confirmation_gate_boundaries() -> None:
    value = {"F_initializer_exposure": 0.01, "F_initializer_equal_input": 0.01}
    assert (
        confirmation_decision(_ce(), value)
        == "advance_to_registered_strength_evaluation"
    )
    for key in _ce():
        below_point = _ce()
        below_point[key] = (0.000999, -0.0001)
        assert confirmation_decision(below_point, value) == "stop_fresh_confirmation"
        nonnegative_bound = _ce()
        nonnegative_bound[key] = (0.001, 0.0)
        assert (
            confirmation_decision(nonnegative_bound, value) == "stop_fresh_confirmation"
        )
    for key in value:
        above_value_bound = dict(value)
        above_value_bound[key] = 0.010001
        assert (
            confirmation_decision(_ce(), above_value_bound) == "stop_fresh_confirmation"
        )


def test_missing_or_extra_contrasts_fail_closed() -> None:
    with pytest.raises(ValueError, match="confirmation_contrast_inventory_mismatch"):
        confirmation_decision({}, {})
