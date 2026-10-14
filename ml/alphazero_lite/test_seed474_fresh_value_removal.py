"""Real-computation and receipt-rebound tests for seed474."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.alphazero_lite.seed474_fresh_value_removal import compare_control


def test_zero_value_coefficient_and_reset_moments_with_parameter_continuity() -> None:
    torch.manual_seed(474)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0000625)
    x = torch.randn(8, 3)
    prior = [p.detach().clone() for p in model.parameters()]
    observed_gradients = []
    for index in range(3):
        optimizer.state.clear()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        policy_ce = prediction.square().mean()
        value_loss = (prediction - 1000).square().mean()
        objective = policy_ce + 0.0 * value_loss
        objective.backward()
        for parameter in model.parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        observed_gradients.append([p.grad.detach().clone() for p in model.parameters()])
        optimizer.step()
        assert all(optimizer.state[p]["step"].item() == 1 for p in model.parameters())
        if index:
            assert not torch.equal(prior[0], next(model.parameters()).detach())
        prior = [p.detach().clone() for p in model.parameters()]
    assert all(g is not None for step in observed_gradients for g in step)


def test_exact_control_comparison_checks_each_post_state(tmp_path) -> None:
    reference = {
        f"{step:02d}_post_00": np.asarray([step], dtype=np.float32)
        for step in range(16)
    }
    candidate = {key: value.copy() for key, value in reference.items()}
    left, right = tmp_path / "left.npz", tmp_path / "right.npz"
    np.savez_compressed(left, **reference)
    np.savez_compressed(right, **candidate)
    compare_control(left, right)
    candidate["15_post_00"][0] += 1
    np.savez_compressed(right, **candidate)
    with pytest.raises(ValueError, match="control_reproduction_mismatch:16:0"):
        compare_control(left, right)
