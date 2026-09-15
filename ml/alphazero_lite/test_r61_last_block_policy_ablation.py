import pytest
import torch

from ml.alphazero_lite.run_r61_last_block_policy_ablation import (
    INHERITED_FULL,
    INHERITED_HEADS,
    REPLAY,
    SCOPES,
    TRAINING_SEEDS,
    assert_design,
    assert_frozen_equal,
    classify,
    drift_cosines,
    final_block_prefix,
    parameter_state,
    scope_manifest,
    trainable_prefixes,
    verify_inherited_checkpoints,
)
from ml.alphazero_lite.train import PolicyValueNet


def cell(seed: str, scope: str, *, top: bool = True, mass: float = 0.6) -> dict:
    return {
        "training_seed": seed,
        "scope": scope,
        "anchor": {"top_is_outcome_optimal": top, "optimal_mass": mass},
        "frozen_parameter_invariant": True,
        "material_regression": False,
        "critical_forensic_regression": False,
        "arena_material_regression": False,
        "value_limited": False,
    }


def test_design_is_exact_r61_seed_scope_matrix() -> None:
    assert TRAINING_SEEDS == {"T61": 61, "T63": 63}
    assert_design(
        [(REPLAY, seed, scope) for seed in TRAINING_SEEDS for scope in SCOPES]
    )
    with pytest.raises(ValueError):
        assert_design([("R61", "T61", "last_block_policy")] * 6)


def test_last_block_scope_membership_is_mechanical() -> None:
    model = PolicyValueNet((96, 3), "residual_v3", 27)
    manifest = scope_manifest(model)
    assert final_block_prefix(model) == "residual_layers.2."
    assert manifest["trainable_parameter_count"] > 0
    for name, enabled in manifest["parameters"].items():
        assert enabled == name.startswith(trainable_prefixes(model))


def test_inherited_checkpoint_sha_manifest_is_complete() -> None:
    for controls in (INHERITED_FULL, INHERITED_HEADS):
        assert set(controls) == {"T61", "T63"}
        assert all(len(expected_sha) == 64 for _, expected_sha in controls.values())
    with pytest.raises(RuntimeError, match="last_block_policy_ablation_inconclusive"):
        verify_inherited_checkpoints({"T61": ("missing.npz", "0" * 64)}, "all")


def test_frozen_parameters_remain_bit_identical_and_last_block_moves() -> None:
    model = PolicyValueNet((96, 3), "residual_v3", 27)
    scope_manifest(model)
    frozen = parameter_state(
        model,
        (
            "input_layer.",
            "residual_layers.0.",
            "residual_layers.1.",
            "value_hidden_layer.",
            "value_head.",
        ),
    )
    before = model.residual_layers[-1][0].weight.detach().clone()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=0.001
    )
    logits, values = model(torch.randn(8, 27))
    (logits.square().mean() + values.square().mean()).backward()
    optimizer.step()
    assert not torch.equal(before, model.residual_layers[-1][0].weight)
    assert_frozen_equal(frozen, model)
    with torch.no_grad():
        model.value_head.weight.add_(1)
    with pytest.raises(RuntimeError, match="last_block_policy_scope_leak"):
        assert_frozen_equal(frozen, model)


def test_drift_cosines_are_descriptive_for_final_block_only(tmp_path) -> None:
    paths = []
    for index in range(5):
        model = PolicyValueNet((96, 3), "residual_v3", 27)
        with torch.no_grad():
            model.residual_layers[-1][0].weight.add_(index)
        path = tmp_path / f"model-{index}.npz"
        import numpy as np

        from ml.alphazero_lite.train import checkpoint_from_model

        np.savez(path, **checkpoint_from_model(model))
        paths.append(path)
    result = drift_cosines(
        paths[0], {"T61": paths[1], "T63": paths[2]}, {"T61": paths[3], "T63": paths[4]}
    )
    assert set(result) == {"lastblock61", "lastblock63"}
    assert set(result["lastblock61"]) == {"full61", "full63"}


def test_hard_success_rule_paths() -> None:
    controls = [cell("T61", "all", top=False, mass=0.27), cell("T63", "all")]
    heads = [
        cell("T61", "heads_only", top=False, mass=0.25),
        cell("T63", "heads_only", top=False, mass=0.22),
    ]
    good = (
        controls
        + heads
        + [cell("T61", "last_block_policy", mass=0.5), cell("T63", "last_block_policy")]
    )
    assert classify(good)[0] == "last_block_policy_stabilizes_shared_trunk_failure"
    sensitive = (
        controls
        + heads
        + [
            cell("T61", "last_block_policy", mass=0.5),
            cell("T63", "last_block_policy", top=False),
        ]
    )
    assert classify(sensitive)[0] == "last_block_policy_seed_sensitive"
    no_rescue = (
        controls
        + heads
        + [
            cell("T61", "last_block_policy", top=False, mass=0.30),
            cell("T63", "last_block_policy"),
        ]
    )
    assert classify(no_rescue)[0] == "last_block_policy_no_anchor_rescue"
