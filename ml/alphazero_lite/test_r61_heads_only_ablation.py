import pytest
import torch

from ml.alphazero_lite.run_r61_heads_only_ablation import (
    REPLAY,
    SCOPES,
    TRAINING_SEEDS,
    assert_design,
    assert_trunk_equal,
    classify,
    scope_manifest,
    trunk_state,
)
from ml.alphazero_lite.train import PolicyValueNet


def cell(
    seed: str,
    scope: str,
    *,
    top: bool = True,
    mass: float = 0.6,
    regression: bool = False,
) -> dict:
    return {
        "training_seed": seed,
        "scope": scope,
        "anchor": {"top_is_outcome_optimal": top, "optimal_mass": mass},
        "trunk_immutable": True,
        "material_regression": regression,
    }


def test_design_is_exact_r61_seed_scope_matrix() -> None:
    assert_design(
        [(REPLAY, seed, scope) for seed in TRAINING_SEEDS for scope in SCOPES]
    )
    with pytest.raises(ValueError):
        assert_design([("R61", "T61", "heads_only")] * 4)


def test_heads_only_scope_membership_and_counts_are_mechanical() -> None:
    manifest = scope_manifest(PolicyValueNet((96, 3), "residual_v3", 27), "heads_only")
    assert manifest["trainable_parameter_count"] > 0
    assert manifest["frozen_parameter_count"] > manifest["trainable_parameter_count"]
    for name, enabled in manifest["parameters"].items():
        assert enabled == name.startswith(
            (
                "policy_hidden_layer.",
                "policy_head.",
                "value_hidden_layer.",
                "value_head.",
            )
        )


def test_trunk_equality_detects_any_frozen_tensor_change() -> None:
    model = PolicyValueNet((96, 3), "residual_v3", 27)
    baseline = trunk_state(model)
    assert_trunk_equal(baseline, model)
    with torch.no_grad():
        model.input_layer.weight.add_(1)
    with pytest.raises(RuntimeError, match="heads_only_scope_leaked_trunk_updates"):
        assert_trunk_equal(baseline, model)


def test_heads_can_update_while_trunk_remains_frozen() -> None:
    model = PolicyValueNet((96, 3), "residual_v3", 27)
    manifest = scope_manifest(model, "heads_only")
    baseline = trunk_state(model)
    before = model.policy_head.weight.detach().clone()
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.001,
    )
    logits, values = model(torch.randn(8, 27))
    (logits.square().mean() + values.square().mean()).backward()
    optimizer.step()
    assert not torch.equal(before, model.policy_head.weight)
    assert manifest["trainable_parameter_count"] > 0
    assert_trunk_equal(baseline, model)


def test_hard_classification_paths() -> None:
    controls = [cell("T61", "all", top=False, mass=0.27), cell("T63", "all")]
    good = controls + [cell("T61", "heads_only", mass=0.5), cell("T63", "heads_only")]
    assert (
        classify(good, {"optimal_mass": 0.4})[0]
        == "heads_only_stabilizes_shared_trunk_failure"
    )
    underfit = controls + [
        cell("T61", "heads_only", mass=0.5, regression=True),
        cell("T63", "heads_only"),
    ]
    assert (
        classify(underfit, {"optimal_mass": 0.4})[0]
        == "heads_only_stabilizes_but_underfits"
    )
    no_rescue = controls + [
        cell("T61", "heads_only", top=False, mass=0.27),
        cell("T63", "heads_only"),
    ]
    assert (
        classify(no_rescue, {"optimal_mass": 0.4})[0] == "heads_only_no_anchor_rescue"
    )
    leaked = controls + [
        {**cell("T61", "heads_only", mass=0.5), "trunk_immutable": False},
        cell("T63", "heads_only"),
    ]
    assert (
        classify(leaked, {"optimal_mass": 0.4})[0]
        == "heads_only_scope_leaked_trunk_updates"
    )
