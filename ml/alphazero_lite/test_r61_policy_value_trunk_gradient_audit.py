import copy

import pytest
import torch

from ml.alphazero_lite.run_r61_policy_value_trunk_gradient_audit import (
    FINE_GROUPS,
    IDENTITY_TOLERANCE,
    SHARED_TRUNK,
    VALUE_WEIGHT,
    cloned_adam_step,
    component_metrics,
    contribution,
    divergence_step,
    gradients,
    parameter_groups,
    snapshot,
)
from ml.alphazero_lite.train import PolicyValueNet


def model() -> PolicyValueNet:
    return PolicyValueNet((8, 3), "residual_v3", 21)


def test_parameter_groups_keep_each_residual_block_separate() -> None:
    network = model()
    groups = parameter_groups(network)
    assert tuple(groups) == FINE_GROUPS
    assert SHARED_TRUNK == (
        "trunk_input",
        "residual_block_0",
        "residual_block_1",
        "residual_block_2",
    )
    assert set().union(*map(set, groups.values())) == {
        name
        for name, parameter in network.named_parameters()
        if parameter.requires_grad
    }


def test_policy_value_gradient_identity_and_common_clip_are_additive() -> None:
    network = model()
    x = torch.rand(3, 21)
    logits, values = network(x)
    policy_loss = logits.square().mean()
    value_loss = values.square().mean()
    total_loss = policy_loss + VALUE_WEIGHT * value_loss
    policy = gradients(policy_loss, network)
    raw_value = gradients(value_loss, network)
    weighted_value = {name: VALUE_WEIGHT * value for name, value in raw_value.items()}
    total = gradients(total_loss, network)
    for name in total:
        assert torch.allclose(
            total[name], policy[name] + weighted_value[name], atol=IDENTITY_TOLERANCE
        )
    groups = parameter_groups(network)
    metrics = component_metrics(
        policy, weighted_value, total, total, total, groups, 0.25
    )
    assert metrics["shared_trunk"]["policy_postclip_norm"] == pytest.approx(
        0.25 * metrics["shared_trunk"]["policy_preclip_norm"]
    )


def test_cloned_counterfactual_does_not_mutate_historical_state() -> None:
    network = model()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    before = snapshot(network)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    x = torch.rand(2, 21).numpy()
    policy = torch.tensor([[1.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0]]).numpy()
    value = torch.zeros(2, 1).numpy()
    clone = cloned_adam_step(
        before, optimizer_before, x, policy, value, source="no_value_to_trunk"
    )
    assert isinstance(clone, PolicyValueNet)
    assert all(
        torch.equal(value, snapshot(network)[name]) for name, value in before.items()
    )
    assert optimizer.state_dict() == optimizer_before


def test_contribution_reports_step_and_absolute_effect_weighting() -> None:
    rows = [
        {
            "groups": {
                "shared_trunk": {
                    "policy_anchor_effect": -2.0,
                    "value_anchor_effect": 1.0,
                }
            }
        },
        {
            "groups": {
                "shared_trunk": {
                    "policy_anchor_effect": 1.0,
                    "value_anchor_effect": -3.0,
                }
            }
        },
    ]
    result = contribution(rows, "shared_trunk", "anchor_effect")
    assert result["policy_harmful"] == 2.0
    assert result["value_harmful"] == 3.0
    assert result["policy_absolute_effect_weighted_harmful"] == pytest.approx(4 / 3)
    assert result["value_absolute_effect_weighted_harmful"] == pytest.approx(9 / 4)


def test_divergence_step_uses_anchor_margin_correctness_sign() -> None:
    t61 = [{"anchor_margin_after": 1.0}, {"anchor_margin_after": -1.0}]
    t63 = [{"anchor_margin_after": 2.0}, {"anchor_margin_after": 3.0}]
    assert divergence_step(t61, t63) == 2
