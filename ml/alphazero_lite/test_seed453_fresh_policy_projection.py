"""Synthetic checks for seed453 projection and selection arithmetic."""

import numpy as np
import torch

from ml.alphazero_lite.seed453_fresh_policy_projection import (
    choose_largest_feasible,
    archive_tensor,
    project_direction,
    project_proposal,
    realized_dot,
    trial_displacement,
)
from ml.alphazero_lite.verify_seed453_fresh_policy_projection import (
    require_array,
    require_close,
    require_value,
)


def test_projection_removes_harmful_component_and_leaves_safe_direction() -> None:
    gradient = [np.array([1.0, 0.0], dtype=np.float32)]
    projected, report = project_direction(
        [np.array([2.0, 3.0], dtype=np.float32)], gradient
    )
    np.testing.assert_array_equal(projected[0], [0.0, 3.0])
    assert report["raw_dot"] == 2.0
    assert report["projected_dot"] == 0.0
    safe, _ = project_direction([np.array([-2.0, 3.0])], gradient)
    np.testing.assert_array_equal(safe[0], [-2.0, 3.0])


def test_zero_gradient_and_unused_parameters() -> None:
    raw = [np.array([1.0, 2.0]), np.array([3.0])]
    projected, report = project_direction(raw, [np.zeros(2), np.zeros(1)])
    assert report["gradient_norm"] == 0
    for actual, expected in zip(projected, raw, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_realized_float32_rounding_dot_and_trial_arithmetic() -> None:
    pre = [np.array([1.0], dtype=np.float32)]
    proposal = [np.array([0.0], dtype=np.float32)]
    half = trial_displacement(pre, proposal, 0.5)
    assert half[0].dtype == np.float32
    assert realized_dot(pre, half, [np.array([1.0])]) == -0.5


def test_proposal_projection_uses_float64_displacement_before_float32_writeback() -> (
    None
):
    pre = [np.array([1.0000001, -2.0], dtype=np.float32)]
    raw = [np.array([1.25, -1.5], dtype=np.float32)]
    gradient = [np.array([2.0, 0.0], dtype=np.float32)]
    result, report = project_proposal(pre, raw, gradient)
    assert report["raw_dot"] > 0
    assert report["projected_dot"] == 0.0
    assert result[0].dtype == np.float32
    assert report["projected_proposal_dot"] <= 1e-7


def test_projection_does_not_mutate_input_or_adam_moment_arrays() -> None:
    pre = [np.array([0.0, 1.0], dtype=np.float32)]
    raw = [np.array([2.0, 3.0], dtype=np.float32)]
    moment = np.array([0.25, -0.5], dtype=np.float32)
    before_moment = moment.copy()
    before_pre = pre[0].copy()
    project_proposal(pre, raw, [np.array([1.0, 0.0], dtype=np.float32)])
    np.testing.assert_array_equal(moment, before_moment)
    np.testing.assert_array_equal(pre[0], before_pre)


def test_archived_tensor_is_a_defensive_copy() -> None:
    tensor = torch.tensor([1.0, 2.0])
    archived = archive_tensor(tensor)
    tensor[0] = 9.0
    assert archived.tolist() == [1.0, 2.0]


def test_archive_comparison_rejects_gradient_proposal_and_moment_tampering() -> None:
    expected = np.array([1.0, 2.0], dtype=np.float32)
    for label, tampered in (
        ("gradient", np.array([1.0, 2.01], dtype=np.float32)),
        ("proposal", np.array([1.01, 2.0], dtype=np.float32)),
        ("moment", np.array([1.0, 2.1], dtype=np.float32)),
    ):
        try:
            require_array(tampered, expected, label)
        except ValueError as error:
            assert str(error) == f"semantic_tensor_mismatch:{label}"
        else:
            raise AssertionError(f"tamper_not_rejected:{label}")


def test_semantic_tampering_of_guard_scales_metrics_and_decision_is_rejected() -> None:
    cases = (
        (
            require_value,
            ([{"identity": "frozen"}], [{"identity": "changed"}], "guard"),
        ),
        (require_value, (0.5, 0.25, "scale")),
        (require_close, (0.001, 0.00101, "metric")),
        (
            require_value,
            (
                "close_fresh_policy_projection_branch",
                "advance_to_separately_preregistered_strength_experiment",
                "decision",
            ),
        ),
    )
    for checker, (actual, expected, label) in cases:
        try:
            checker(actual, expected, label)
        except ValueError as error:
            assert label in str(error)
        else:
            raise AssertionError(f"semantic_tamper_not_rejected:{label}")


def test_feasibility_selection_does_not_assume_monotonicity() -> None:
    trials = [
        {"scale": 1.0, "feasible": False},
        {"scale": 0.5, "feasible": True},
        {"scale": 0.25, "feasible": False},
        {"scale": 0.125, "feasible": True},
    ]
    assert choose_largest_feasible(trials) == 0.5
