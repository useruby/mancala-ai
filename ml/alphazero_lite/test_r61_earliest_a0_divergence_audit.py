import copy

import pytest
import torch

from ml.alphazero_lite.run_r61_earliest_a0_divergence_audit import (
    MATERIAL_FRACTION,
    PERSISTENT_NEXT_STEPS,
    PERSISTENT_REQUIRED,
    adam_update,
    classify_step,
    effect_curve,
    factorial,
    first_material_divergence,
    frozen_window,
    overlap,
    order_test_eligible,
    tensor_snapshot,
)
from ml.alphazero_lite.train import PolicyValueNet


def trace(values: list[float]) -> list[dict]:
    return [
        {"optimizer_step": step, "cluster_specific_a0_effect": value}
        for step, value in enumerate(values, 1)
    ]


def cells(a: float, b: float, c: float, d: float) -> dict:
    return {
        name: {"cluster_specific_a0_effect": value}
        for name, value in zip("ABCD", (a, b, c, d), strict=True)
    }


def test_effect_curve_reconstructs_cumulative_difference() -> None:
    # Pad to the fixed endpoint; no rounded final values are inputs.
    curve = effect_curve(trace([1.0] + [0.0] * 107), trace([0.25] + [0.0] * 107))
    assert curve[0] == {"step": 1, "c61": 1.0, "c63": 0.25, "difference": 0.75}
    assert curve[-1]["difference"] == pytest.approx(0.75)


def test_material_threshold_and_persistence_select_earliest_deterministically() -> None:
    # D is briefly material at 1 but does not persist; the signed plateau at 3 does.
    differences = [0.3, 0.0, 0.0, 0.3, 0.3, 0.3, 0.3, 0.3] + [0.3] * 100
    curve = [
        {"step": step, "c61": value, "c63": 0.0, "difference": value}
        for step, value in enumerate(differences, 1)
    ]
    step, threshold = first_material_divergence(curve)
    assert threshold == MATERIAL_FRACTION * abs(curve[-1]["difference"])
    assert step == 4
    assert PERSISTENT_NEXT_STEPS == 5
    assert PERSISTENT_REQUIRED == 4


def test_frozen_window_is_single_contiguous_window_of_at_most_nine_steps() -> None:
    assert frozen_window(5) == list(range(1, 10))
    assert frozen_window(108) == list(range(104, 109))
    assert len(frozen_window(50)) == 9


def test_factorial_decomposition_matches_registered_formula() -> None:
    assert factorial(cells(1.0, 3.0, 5.0, 11.0)) == {
        "batch_effect": 4.0,
        "state_effect": 6.0,
        "interaction": 4.0,
    }


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (cells(-0.1, 0.2, 0.0, 0.1), "state_dependent_harm"),
        (cells(-0.1, 0.2, -0.2, 0.1), "content_stable_harm"),
        (cells(0.0, 0.01, 0.0, 0.0), "protective_t63_batch"),
        (cells(-0.1, -0.2, 0.0, 0.0), "state_dependent_harm"),
        (cells(0.1, 0.105, 0.0, 0.0), "both_batches_protective"),
    ],
)
def test_step_classification_uses_registered_precedence(
    values: dict, expected: str
) -> None:
    assert classify_step(values) == expected


def test_overlap_uses_compact_and_canonical_rows_without_mutating_metadata() -> None:
    metadata = [
        {"canonical_state_hash": "a", "source": "x", "policy_target": [1.0, 0.0]},
        {"canonical_state_hash": "b", "source": "y", "policy_target": [0.0, 1.0]},
        {"canonical_state_hash": "a", "source": "x", "policy_target": [0.5, 0.5]},
    ]
    before = copy.deepcopy(metadata)
    value = overlap([0, 1], [0, 2], metadata)
    assert value["compact_row_jaccard"] == pytest.approx(1 / 3)
    assert value["canonical_state_jaccard"] == pytest.approx(1 / 2)
    assert value["shared_state_policy_target_tv"] == pytest.approx(0.0)
    assert metadata == before


def test_order_micro_test_is_eligible_only_at_fifty_percent_canonical_overlap() -> None:
    metadata = [{"canonical_state_hash": value} for value in "abcd"]
    rows = [
        {"t61_batch": {"batch_indexes": [0, 1]}, "t63_batch": {"batch_indexes": [0, 2]}}
    ]
    assert order_test_eligible(rows, metadata) is False
    rows[0]["t63_batch"]["batch_indexes"] = [0, 1]
    assert order_test_eligible(rows, metadata) is True


def test_adam_update_does_not_continue_or_mutate_source_state() -> None:
    model = PolicyValueNet((8, 3), "residual_v3", 21)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before, state = tensor_snapshot(model), copy.deepcopy(optimizer.state_dict())
    x = torch.rand(2, 21).numpy()
    policy = torch.tensor([[1.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0]]).numpy()
    value = torch.zeros(2, 1).numpy()
    after, _ = adam_update(before, state, x, policy, value)
    assert tensor_snapshot(model).keys() == before.keys()
    assert all(
        torch.equal(tensor_snapshot(model)[name], value)
        for name, value in before.items()
    )
    assert optimizer.state_dict() == state
    assert not torch.equal(
        tensor_snapshot(after)["input_layer.weight"], before["input_layer.weight"]
    )
