import copy

import pytest
import torch

from ml.alphazero_lite.run_r61_parameter_subspace_drift_audit import (
    FINE_GROUPS,
    ROLLED_GROUPS,
    anchor_probe,
    compare_progress,
    make_hybrid,
    margin_metrics,
    parameter_groups,
    state_snapshot,
)
from ml.alphazero_lite.train import PolicyValueNet


def model() -> PolicyValueNet:
    return PolicyValueNet((8, 3), "residual_v3", 21)


def test_residual_v3_parameters_are_an_exact_fine_and_rolled_partition() -> None:
    network = model()
    groups = parameter_groups(network)
    all_names = {
        name
        for name, parameter in network.named_parameters()
        if parameter.requires_grad
    }
    assert set().union(*map(set, groups.values())) == all_names
    assert sum(len(names) for names in groups.values()) == len(all_names)
    assert set(ROLLED_GROUPS["shared_trunk"]) == {"trunk_input", "trunk_residual"}
    assert set(ROLLED_GROUPS["policy_path"]) == {"policy_hidden", "policy_readout"}
    assert set(ROLLED_GROUPS["value_path"]) == {"value_hidden", "value_readout"}
    assert tuple(groups) == FINE_GROUPS


def test_whole_group_swap_preserves_shapes_and_value_path_cannot_change_logits() -> (
    None
):
    base, donor = model(), model()
    groups = parameter_groups(base)
    x = torch.rand(1, 21)
    with torch.no_grad():
        native, _ = base(x)
    hybrid = make_hybrid(base, donor, groups, ("value_path",))
    with torch.no_grad():
        swapped, _ = hybrid(x)
    assert torch.equal(native, swapped)
    for name, parameter in hybrid.named_parameters():
        assert parameter.shape == dict(base.named_parameters())[name].shape


def test_shared_and_policy_swaps_copy_complete_named_groups() -> None:
    base, donor = model(), model()
    groups = parameter_groups(base)
    for swap in ("shared_trunk", "policy_path", "policy_hidden", "policy_readout"):
        hybrid = make_hybrid(base, donor, groups, (swap,))
        selected = {
            name for fine in ROLLED_GROUPS.get(swap, (swap,)) for name in groups[fine]
        }
        for name, parameter in hybrid.named_parameters():
            expected = (
                dict(donor.named_parameters())[name]
                if name in selected
                else dict(base.named_parameters())[name]
            )
            assert torch.equal(parameter, expected)


def test_margin_drift_and_probe_are_observational() -> None:
    network = model()
    groups, g0 = parameter_groups(network), state_snapshot(network)
    x, legal = torch.rand(1, 21), [0, 1, 3, 4]
    metric = margin_metrics(network, x, legal)
    assert metric["anchor_margin"] == pytest.approx(
        metric["logits"][0]
        - max(metric["logits"][1], metric["logits"][3], metric["logits"][4])
    )
    before = state_snapshot(network)
    grads_before = {
        name: parameter.grad for name, parameter in network.named_parameters()
    }
    probe = anchor_probe(
        network,
        x,
        legal,
        g0,
        {name: torch.zeros_like(value) for name, value in g0.items()},
        groups,
    )
    assert set(probe) == set(FINE_GROUPS) | set(ROLLED_GROUPS)
    assert all(
        torch.equal(before[name], value)
        for name, value in state_snapshot(network).items()
    )
    assert all(
        parameter.grad is grads_before[name]
        for name, parameter in network.named_parameters()
    )
    comparison = compare_progress(before, copy.deepcopy(before), g0, groups)
    assert all(values["euclidean_distance"] == 0.0 for values in comparison.values())
