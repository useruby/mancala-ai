"""Synthetic tests for seed452 signed attribution arithmetic."""

import numpy as np
import pytest

from ml.alphazero_lite.seed452_arithmetic import classify, decompose, totals, weights


def test_linear_quadratic_and_group_decomposition() -> None:
    # For f(x)=x^2, at x=2 and delta=0.25: d=1.0625, s=1, r=0.0625.
    layout = [
        {"start": 0, "stop": 1, "group": "shared_trunk"},
        {"start": 1, "stop": 2, "group": "policy_head"},
    ]
    result = decompose(4.0, 5.0625, np.array([4.0, 0.0]), np.array([0.25, 0.0]), layout)
    assert result["d"] == pytest.approx(1.0625)
    assert result["s"] == pytest.approx(1.0)
    assert result["r"] == pytest.approx(0.0625)
    assert result["group_s"]["shared_trunk"] == pytest.approx(1.0)


def test_weightings_retain_duplicate_exposures() -> None:
    result = weights(["a", "a", "b"])
    assert result["exposure_weighted"] == pytest.approx([1 / 3] * 3)
    assert result["equal_exact_input"] == pytest.approx([0.25, 0.25, 0.5])


def test_zero_movement_and_signed_cancellation() -> None:
    layout = [{"start": 0, "stop": 1, "group": "policy_head"}]
    zero = decompose(2, 2, np.array([7.0]), np.array([0.0]), layout)
    assert zero == {"d": 0.0, "s": 0.0, "r": 0.0, "group_s": {"policy_head": 0.0}}
    summed = totals([{"d": 1.0, "s": 2.0, "r": -1.0}, {"d": -1.0, "s": -2.0, "r": 1.0}])
    assert summed == {"d": 0.0, "s": 0.0, "r": 0.0}


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"D": 1.0, "S": 0.75, "R": 0.25}, "recorded_direction_harms_fresh_policy"),
        (
            {"D": 1.0, "S": 0.74, "R": 0.75},
            "fresh_policy_finite_step_residual_dominant",
        ),
        ({"D": 0.0, "S": 0.0, "R": 0.0}, "mixed_or_no_fresh_policy_regression"),
    ],
)
def test_classifier_boundaries(values: dict[str, float], expected: str) -> None:
    assert classify({"exposure_weighted": values})["overall"] == expected


def test_displacement_shape_is_checked() -> None:
    with pytest.raises(ValueError, match="shape"):
        decompose(0, 1, np.ones(2), np.ones(3), [])


def test_float32_state_displacement_is_taken_after_cast() -> None:
    before = np.asarray([1.0], dtype=np.float32)
    after = np.asarray([np.nextafter(before[0], np.float32(2.0))], dtype=np.float32)
    delta = after.astype(np.float64) - before.astype(np.float64)
    assert delta[0] == pytest.approx(float(after[0]) - float(before[0]))
    result = decompose(
        0.0,
        1.0,
        np.array([1.0]),
        delta,
        [{"start": 0, "stop": 1, "group": "value_head"}],
    )
    assert result["s"] == pytest.approx(float(delta[0]))


def test_weighting_dependent_classification() -> None:
    output = classify(
        {
            "exposure_weighted": {"D": 1.0, "S": 0.8, "R": 0.2},
            "equal_exact_input": {"D": 1.0, "S": 0.1, "R": 0.9},
        }
    )
    assert output["overall"] == "weighting_dependent_fresh_attribution"
