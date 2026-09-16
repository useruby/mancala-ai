import copy

import pytest
import torch

from ml.alphazero_lite.run_r61_step82_parameter_adam_decomposition import (
    CELL_SPECS,
    INPUT_NAMES,
    STEPS,
    factorial,
    transplant_input_adam_state,
    transplant_input_parameters,
    transplant_optimizer_state,
    transplant_parameters,
    validate_optimizer_compatibility,
)
from ml.alphazero_lite.train import PolicyValueNet


def snapshots() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    left, right = (
        PolicyValueNet((8, 3), "residual_v3", 21),
        PolicyValueNet((8, 3), "residual_v3", 21),
    )
    return (
        {name: value.detach().clone() for name, value in left.named_parameters()},
        {name: value.detach().clone() for name, value in right.named_parameters()},
    )


def adam_state(model: PolicyValueNet) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    return copy.deepcopy(optimizer.state_dict())


def test_full_parameter_transplant_is_deep_and_complete() -> None:
    left, right = snapshots()
    result = transplant_parameters(left, right)
    assert all(torch.equal(result[name], right[name]) for name in right)
    result["input_layer.bias"].add_(1)
    assert not torch.equal(result["input_layer.bias"], right["input_layer.bias"])
    assert tuple(CELL_SPECS) == ("PP61_A61", "P61_A63", "P63_A61", "P63_A63")


def test_input_parameter_transplant_is_complete_and_reverse_safe() -> None:
    left, right = snapshots()
    result = transplant_input_parameters(left, right)
    reverse = transplant_input_parameters(right, left)
    for name in left:
        expected = right[name] if name in INPUT_NAMES else left[name]
        assert torch.equal(result[name], expected)
    assert all(
        torch.equal(reverse[name], left[name] if name in INPUT_NAMES else right[name])
        for name in right
    )


def test_optimizer_transplants_are_deep_and_keep_parameter_mapping() -> None:
    model = PolicyValueNet((8, 3), "residual_v3", 21)
    parameters = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    source = adam_state(model)
    result = transplant_optimizer_state(source, source, parameters)
    validate_optimizer_compatibility(parameters, result)
    first_slot = next(iter(result["state"]))
    assert torch.equal(
        result["state"][first_slot]["exp_avg"], source["state"][first_slot]["exp_avg"]
    )
    result["state"][first_slot]["exp_avg"].add_(1)
    assert not torch.equal(
        result["state"][first_slot]["exp_avg"], source["state"][first_slot]["exp_avg"]
    )


def test_input_adam_transplant_replaces_only_input_moments() -> None:
    left_model, right_model = (
        PolicyValueNet((8, 3), "residual_v3", 21),
        PolicyValueNet((8, 3), "residual_v3", 21),
    )
    left, right = adam_state(left_model), adam_state(right_model)
    parameters = {
        name: value.detach().clone() for name, value in left_model.named_parameters()
    }
    result = transplant_input_adam_state(left, right, parameters)
    left_slots = [slot for group in left["param_groups"] for slot in group["params"]]
    right_slots = [slot for group in right["param_groups"] for slot in group["params"]]
    for name, left_slot, right_slot in zip(
        parameters, left_slots, right_slots, strict=True
    ):
        expected = (
            right["state"][right_slot]
            if name in INPUT_NAMES
            else left["state"][left_slot]
        )
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(result["state"][left_slot][key], expected[key])
        assert result["state"][left_slot]["step"] == expected["step"]


def test_factorial_matches_registered_parameter_and_adam_equations() -> None:
    result = factorial(
        {"PP61_A61": 1.0, "P61_A63": 3.0, "P63_A61": 5.0, "P63_A63": 11.0}
    )
    assert result["parameter_main_effect"] == 6.0
    assert result["adam_main_effect"] == 4.0
    assert result["interaction"] == 4.0
    assert result["abs_parameter_main_effect"] == 6.0


def test_window_and_primary_sequence_are_fixed() -> None:
    assert STEPS == (80, 81, 82, 83, 84)
    assert len(STEPS) == 5


def test_optimizer_mapping_rejects_incompatible_shapes() -> None:
    model = PolicyValueNet((8, 3), "residual_v3", 21)
    state = adam_state(model)
    parameters = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    parameters["input_layer.bias"] = torch.zeros(9)
    with pytest.raises(RuntimeError, match="optimizer_parameter_mapping_invalid"):
        validate_optimizer_compatibility(parameters, state)
