"""Focused arithmetic and decision tests for seed447."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.alphazero_lite.seed447_joint_output_cap import (
    VALUE_CAP,
    VALUE_TOLERANCE,
    archive_tensor,
    cap_activation,
    choose_largest_feasible,
    value_movement,
)


def test_value_movement_zero_and_cap_boundary() -> None:
    assert value_movement(np.array([0.0]), np.array([0.0]), np.array([1.0])) == 0.0
    # Exactly 0.01 movement has squared movement exactly 0.0001.
    assert value_movement(
        np.array([0.0]), np.array([0.01]), np.array([1.0])
    ) == pytest.approx(VALUE_CAP)


def test_duplicate_exposure_weights_are_not_equal_input_weights() -> None:
    movement = value_movement(
        np.zeros(3), np.array([0.0, 0.0, 0.02]), np.full(3, 1 / 3)
    )
    assert movement == pytest.approx(0.0004 / 3)


def test_cap_boundary_includes_registered_absolute_tolerance() -> None:
    trial = {
        "scale": 1.0,
        "batch_kl": 0.005,
        "guard_kl": 0.005,
        "batch_value_movement": VALUE_CAP + VALUE_TOLERANCE,
        "guard_value_movement": VALUE_CAP,
    }
    assert (
        choose_largest_feasible([trial], cap=VALUE_CAP, tolerance=VALUE_TOLERANCE)
        == 1.0
    )
    trial["guard_value_movement"] = VALUE_CAP + 2 * VALUE_TOLERANCE
    assert (
        choose_largest_feasible([trial], cap=VALUE_CAP, tolerance=VALUE_TOLERANCE)
        is None
    )


def test_largest_feasible_checks_all_scales_not_monotonicity() -> None:
    trials = [
        {
            "scale": 1.0,
            "batch_kl": 0.006,
            "guard_kl": 0.0,
            "batch_value_movement": 0.0,
            "guard_value_movement": 0.0,
        },
        {
            "scale": 0.5,
            "batch_kl": 0.001,
            "guard_kl": 0.001,
            "batch_value_movement": 0.0,
            "guard_value_movement": 0.0,
        },
        {
            "scale": 0.25,
            "batch_kl": 0.006,
            "guard_kl": 0.001,
            "batch_value_movement": 0.0,
            "guard_value_movement": 0.0,
        },
    ]
    assert (
        choose_largest_feasible(trials, cap=VALUE_CAP, tolerance=VALUE_TOLERANCE) == 0.5
    )


def test_cap_activation_is_proposal_local() -> None:
    assert cap_activation(1.0, 0.5)
    assert not cap_activation(0.5, 0.5)
    assert not cap_activation(None, None)
    assert not cap_activation(0.5, None)


def test_rejection_restores_parameters_but_advances_adam_moments() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.001)
    before = parameter.detach().clone()
    parameter.grad = torch.tensor([1.0])
    optimizer.step()
    proposal = parameter.detach().clone()
    moment = optimizer.state[parameter]["exp_avg"].detach().clone()
    with torch.no_grad():
        parameter.copy_(before)
    assert not torch.equal(proposal, before)
    assert torch.equal(parameter, before)
    assert optimizer.state[parameter]["step"].item() == 1
    assert torch.equal(optimizer.state[parameter]["exp_avg"], moment)
    assert moment.item() != 0


def test_archive_tensor_is_detached_immutable_copy() -> None:
    tensor = torch.tensor([1.0, 2.0])
    snapshot = archive_tensor(tensor)
    tensor.add_(10)
    assert np.array_equal(snapshot, np.array([1.0, 2.0], dtype=np.float32))
