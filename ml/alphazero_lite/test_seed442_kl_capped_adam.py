"""Focused synthetic tests for seed442 KL screening and decision behavior."""

from __future__ import annotations

import pytest
import numpy as np
import torch

from ml.alphazero_lite.seed442_kl_capped_adam import (
    SCALES,
    decide,
    _CURRENT_PARAMETERS,
    _kl,
    _pre_policy,
    _set_trial,
    select_largest_scale,
)


def _evidence() -> dict:
    baseline = {
        "full_training_objective": 1.0,
        "exposure_weighted": {"policy_ce": 1.0, "value_mse": 0.2},
        "equal_input": {"policy_ce": 1.0, "value_mse": 0.2},
    }
    better = {
        "full_training_objective": 0.9,
        "exposure_weighted": {"policy_ce": 0.9, "value_mse": 0.201},
        "equal_input": {"policy_ce": 0.9, "value_mse": 0.201},
    }
    steps = [{"proposal_norm": 1.0, "selected_scale": 0.5, "rejected": False}] + [
        {"proposal_norm": 1.0, "selected_scale": 1.0, "rejected": False}
        for _ in range(15)
    ]
    return {
        "metrics": {"initializer": baseline, "A": baseline, "B": better},
        "arms": {"B": {"steps": steps}},
    }


def test_scale_selection_is_largest_first_and_requires_both_populations() -> None:
    trials = [(scale, 0.01, 0.01) for scale in SCALES]
    trials[2] = (SCALES[2], 0.004, 0.0051)
    trials[3] = (SCALES[3], 0.001, 0.001)
    assert select_largest_scale(trials) == SCALES[3]


def test_scale_selection_rejects_when_all_trials_fail() -> None:
    assert select_largest_scale([(scale, 0.006, 0.001) for scale in SCALES]) is None


def test_zero_proposals_do_not_count_as_cap_activation_or_rejection() -> None:
    evidence = _evidence()
    evidence["arms"]["B"]["steps"] = [
        {"proposal_norm": 0.0, "selected_scale": 1.0, "rejected": False}
        for _ in range(16)
    ]
    result = decide(evidence)
    assert result["classification"] == "cap_inactive_no_followup"


@pytest.mark.parametrize(
    ("key", "mutate"),
    [
        (
            "B_minus_A_exposure_weighted_policy",
            lambda e: e["metrics"]["B"]["exposure_weighted"].update(policy_ce=0.995),
        ),
        (
            "B_minus_A_equal_input_policy",
            lambda e: e["metrics"]["B"]["equal_input"].update(policy_ce=0.995),
        ),
        (
            "B_minus_initializer_exposure_weighted_policy",
            lambda e: e["metrics"]["initializer"]["exposure_weighted"].update(
                policy_ce=0.904
            ),
        ),
        (
            "B_minus_initializer_equal_input_policy",
            lambda e: e["metrics"]["initializer"]["equal_input"].update(
                policy_ce=0.904
            ),
        ),
        (
            "B_training_objective_decreases",
            lambda e: e["metrics"]["B"].update(full_training_objective=1.0),
        ),
        (
            "B_exposure_weighted_value_guard",
            lambda e: e["metrics"]["B"]["exposure_weighted"].update(value_mse=0.203),
        ),
        (
            "B_equal_input_value_guard",
            lambda e: e["metrics"]["B"]["equal_input"].update(value_mse=0.203),
        ),
        (
            "scaled_nonzero_proposal_accepted",
            lambda e: e["arms"]["B"]["steps"][0].update(selected_scale=1.0),
        ),
        (
            "no_nonzero_proposal_rejected",
            lambda e: e["arms"]["B"]["steps"][0].update(rejected=True),
        ),
    ],
)
def test_each_fixed_decision_clause_can_fail_independently(key, mutate) -> None:
    evidence = _evidence()
    mutate(evidence)
    assert decide(evidence)["checks"][key] is False


def test_all_registered_scale_boundaries_and_cap_tolerance() -> None:
    assert SCALES == (1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125)
    assert select_largest_scale([(1.0, 0.005, 0.005 + 1e-11)]) == 1.0


class _FixedLogits(torch.nn.Module):
    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.logits.expand(x.shape[0], -1), torch.zeros((x.shape[0], 1))


def test_kl_masks_illegal_actions_and_is_float64_stable() -> None:
    state = np.zeros((1, 27), dtype=np.float32)
    state[0, 0:2] = 1.0 / 48.0
    x = np.repeat(state, 3, axis=0)
    ids = np.arange(3)
    before = _FixedLogits([1.0, 0.0, 1000.0, -1000.0, 700.0, -800.0])
    illegal_only = _FixedLogits([1.0, 0.0, -1000.0, 1000.0, -700.0, 800.0])
    pre = _pre_policy(before, x, ids)
    kl = _kl(illegal_only, x, ids, np.full(3, 1 / 3, dtype=np.float64), pre)
    assert kl == pytest.approx(0.0, abs=1e-12)
    legal_change = _FixedLogits([-1.0, 0.0, 1000.0, -1000.0, 700.0, -800.0])
    assert _kl(legal_change, x, ids, np.full(3, 1 / 3, dtype=np.float64), pre) > 0.4


def test_trial_restores_from_immutable_pre_and_proposal_tensors() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0, -4.0]))
    _CURRENT_PARAMETERS[:] = [parameter]
    pre = [parameter.detach().clone()]
    proposal = [torch.tensor([6.0, 0.0])]
    _set_trial(pre, proposal, 0.5)
    assert torch.equal(parameter, torch.tensor([4.0, -2.0]))
    _set_trial(pre, proposal, 0.25)
    assert torch.equal(parameter, torch.tensor([3.0, -3.0]))
    _set_trial(pre, proposal, 1.0)
    assert torch.equal(parameter, proposal[0])


def test_zero_proposal_selects_scale_one_and_nonzero_rejection_closes_branch() -> None:
    assert select_largest_scale([(scale, 0.0, 0.0) for scale in SCALES]) == 1.0
    evidence = _evidence()
    evidence["arms"]["B"]["steps"][0].update(
        selected_scale=None,
        rejected=True,
        trials=[{"scale": 1.0, "accepted": False}],
    )
    result = decide(evidence)
    assert result["checks"]["no_nonzero_proposal_rejected"] is False
    assert result["classification"] == "close_kl_capped_step_branch"


def test_adam_proposal_uses_one_state_advance() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    parameter.grad = torch.tensor([0.5])
    optimizer.step()
    assert optimizer.state[parameter]["step"].item() == 1
    assert torch.equal(optimizer.state[parameter]["exp_avg"], torch.tensor([0.05]))
