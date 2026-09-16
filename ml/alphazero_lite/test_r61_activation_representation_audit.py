import pytest
import torch

from ml.alphazero_lite.run_r61_activation_representation_audit import (
    PATCH_STAGES,
    block_output,
    continue_policy_from_stage,
    pair_metrics,
    residual_v3_activations,
    rescue,
    reverse_break,
    select_earliest_causal_stage,
    validate_activation_helpers,
)
from ml.alphazero_lite.run_r61_parameter_subspace_drift_audit import (
    make_hybrid,
    parameter_groups,
)
from ml.alphazero_lite.train import PolicyValueNet


def model() -> PolicyValueNet:
    return PolicyValueNet((8, 3), "residual_v3", 21)


def test_activation_stages_reproduce_normal_forward_and_self_patch() -> None:
    network, x = model(), torch.rand(3, 21)
    validate_activation_helpers(network, x)
    stages = residual_v3_activations(network, x)
    assert torch.equal(stages["A3"], network.trunk_features(x))
    logits, _ = network(x)
    for stage in PATCH_STAGES:
        assert torch.equal(
            continue_policy_from_stage(network, stage, stages[stage]), logits
        )


def test_a3_patch_equals_shared_trunk_hybrid_in_both_directions() -> None:
    t61, t63, x = model(), model(), torch.rand(2, 21)
    groups = parameter_groups(t61)
    for recipient, donor in ((t61, t63), (t63, t61)):
        hybrid = make_hybrid(recipient, donor, groups, ("shared_trunk",))
        patched = continue_policy_from_stage(
            recipient, "A3", residual_v3_activations(donor, x)["A3"]
        )
        assert torch.equal(patched, hybrid(x)[0])


def test_support_and_block_delta_inputs_are_mechanical() -> None:
    network, x = model(), torch.rand(1, 21)
    stages = residual_v3_activations(network, x)
    assert torch.equal(block_output(network, 0, stages["A0"]), stages["A1"])
    metrics = pair_metrics(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0]))
    assert metrics["support_disagreement"] == pytest.approx(0.5)


def test_causal_definitions_and_stage_selection_are_deterministic() -> None:
    failed, successful = (
        {"top_is_outcome_optimal": False},
        {"top_is_outcome_optimal": True},
    )
    assert rescue(failed, successful)
    assert reverse_break(successful, failed)
    rows = {
        stage: {
            "anchor_rescue": stage == "A2",
            "anchor_reverse_harm": stage == "A2",
            "rescue_rate": 0.6,
            "mean_transfer_fraction": 0.6,
            "control_degradation": 0.19,
        }
        for stage in PATCH_STAGES
    }
    assert select_earliest_causal_stage(rows) == "A2"
