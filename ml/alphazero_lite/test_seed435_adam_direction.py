"""Unit tests for seed435's matched global first-moment update."""

from __future__ import annotations

import pytest
import torch

from ml.alphazero_lite.seed435_adam_direction import (
    first_moment_direction,
    global_norm,
    matched_step,
    parameters,
)


def test_bias_corrected_first_moment_recurrence() -> None:
    beta = 0.9
    first = torch.tensor([2.0, -1.0])
    second = torch.tensor([-1.0, 3.0])
    moment = beta * torch.zeros_like(first) + (1 - beta) * first
    moment = beta * moment + (1 - beta) * second
    expected = (beta * (1 - beta) * first + (1 - beta) * second) / (1 - beta**2)
    assert torch.allclose(first_moment_direction([moment], 2)[0], expected)


def test_global_norm_matches_requested_delta() -> None:
    model = torch.nn.Linear(2, 1, bias=True)
    direction = [torch.tensor([[1.0, 2.0]]), torch.tensor([2.0])]
    before = [parameter.detach().clone() for _, parameter in parameters(model)]
    _, actual = matched_step(model, direction, radius=0.125, step=1)
    delta = [
        parameter.detach() - old
        for (_, parameter), old in zip(parameters(model), before, strict=True)
    ]
    assert actual == pytest.approx(0.125, abs=2e-6)
    assert global_norm(delta) == pytest.approx(0.125, abs=2e-6)


def test_positive_radius_rejects_zero_direction() -> None:
    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match="positive_radius_with_zero_direction"):
        matched_step(model, [torch.zeros_like(p) for _, p in parameters(model)], 0.1, 1)


def test_zero_radius_zero_direction_is_noop() -> None:
    model = torch.nn.Linear(2, 1)
    before = [parameter.detach().clone() for _, parameter in parameters(model)]
    assert matched_step(
        model, [torch.zeros_like(p) for _, p in parameters(model)], 0.0, 1
    ) == (0.0, 0.0)
    assert all(
        torch.equal(old, parameter)
        for old, (_, parameter) in zip(before, parameters(model), strict=True)
    )


def test_named_parameter_layout_is_stable() -> None:
    model = torch.nn.Linear(2, 1)
    assert [name for name, _ in parameters(model)] == ["weight", "bias"]


def test_decision_uses_only_final_fixed_metrics() -> None:
    from ml.alphazero_lite.seed435_adam_direction import decide

    passing = {
        "initializer": {
            "full_training_objective": 2.0,
            "exposure_weighted": {"policy_ce": 1.0, "value_mse": 1.0},
            "equal_input": {"policy_ce": 1.0, "value_mse": 1.0},
        },
        "A": {
            "full_training_objective": 1.0,
            "exposure_weighted": {"policy_ce": 1.0, "value_mse": 1.0},
            "equal_input": {"policy_ce": 1.0, "value_mse": 1.0},
        },
        "B": {
            "full_training_objective": 1.5,
            "exposure_weighted": {"policy_ce": 0.98, "value_mse": 1.0},
            "equal_input": {"policy_ce": 0.98, "value_mse": 1.0},
        },
    }
    assert (
        decide(passing)["classification"]
        == "advance_to_separately_preregistered_strength_experiment"
    )
    passing["B"]["full_training_objective"] = 2.1
    assert decide(passing)["classification"] == "close_optimizer_direction_branch"
