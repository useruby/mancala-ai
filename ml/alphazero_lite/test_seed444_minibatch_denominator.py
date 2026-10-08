"""Focused arithmetic and decision tests for seed444."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.alphazero_lite.seed444_minibatch_denominator import (
    SCREEN,
    _flatten_gradients,
    _flat_vector,
)
from ml.alphazero_lite.verify_seed444_minibatch_denominator import (
    _validate_coefficients,
    _validate_component_vectors,
    _validate_decision_field,
    _validate_denominators,
    _validate_layout,
    _validate_split_and_order,
)


def _weighted(batch_gradients: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(batch_gradients * weights[:, None], axis=0)


def _decision(norm: float, error: float) -> str:
    if norm <= 1e-12:
        return "degenerate_reference_gradient"
    if error >= SCREEN:
        return "material_minibatch_denominator_mismatch"
    return "denominator_mismatch_below_screen_threshold"


def test_unequal_coefficient_mass_changes_policy_aggregation() -> None:
    per_batch = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(_weighted(per_batch, np.asarray([0.5, 0.5])), [0.5, 0.5])
    assert np.allclose(_weighted(per_batch, np.asarray([0.9, 0.1])), [0.9, 0.1])


def test_all_zero_policy_batch_contributes_zero_but_value_remains() -> None:
    policy = np.zeros(2)
    value = np.asarray([3.0, 4.0])
    assert np.allclose(policy + 0.3 * value, [0.9, 1.2])


def test_constant_coefficients_make_denominator_weights_equal_exposure() -> None:
    n = np.asarray([512.0, 512.0, 17.0])
    s = 2.0 * n
    assert np.allclose(s / s.sum(), n / n.sum())


def test_replay_copies_are_exposures_but_compact_duplicates_are_not_collapsed() -> None:
    compact_rows = np.asarray([8, 8, 9])
    coefficients = np.asarray([1.0, 1.0, 2.0])
    assert len(compact_rows) == 3
    assert coefficients.sum() == 4.0
    assert len(np.unique(compact_rows)) == 2


def test_partial_batch_changes_equal_update_weighting() -> None:
    sizes = np.asarray([512.0, 512.0, 3.0])
    assert not np.allclose(sizes / sizes.sum(), np.full(3, 1 / 3))
    assert np.allclose(np.full(3, 1 / 3), np.ones(3) / 3)


def test_value_coefficient_is_applied_once() -> None:
    policy, value = np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0])
    assert np.array_equal(policy + 0.3 * value, [1.9, 3.2])


def test_group_squared_norms_reconcile_to_full_vector() -> None:
    difference = np.arange(7, dtype=np.float64)
    pieces = [difference[:2], difference[2:5], difference[5:]]
    assert np.isclose(
        sum(float(part @ part) for part in pieces), float(difference @ difference)
    )


def test_zero_norm_and_inclusive_threshold_classification() -> None:
    assert _decision(0.0, 1.0) == "degenerate_reference_gradient"
    assert _decision(1.0, SCREEN) == "material_minibatch_denominator_mismatch"
    assert (
        _decision(1.0, np.nextafter(SCREEN, 0.0))
        == "denominator_mismatch_below_screen_threshold"
    )


def test_policy_unused_value_parameters_keep_shaped_zero_components() -> None:
    policy_parameter = torch.nn.Parameter(torch.ones(2, 3))
    value_parameter = torch.nn.Parameter(torch.ones(4))
    gradients = torch.autograd.grad(
        policy_parameter.square().sum(),
        (policy_parameter, value_parameter),
        allow_unused=True,
    )
    components = _flatten_gradients((policy_parameter, value_parameter), gradients)
    assert components[0].shape == (2, 3)
    assert components[1].shape == (4,)
    assert np.array_equal(components[1], np.zeros(4))


def test_value_unused_policy_parameters_keep_shaped_zero_components() -> None:
    policy_parameter = torch.nn.Parameter(torch.ones(2, 3))
    value_parameter = torch.nn.Parameter(torch.ones(4))
    gradients = torch.autograd.grad(
        value_parameter.square().sum(),
        (policy_parameter, value_parameter),
        allow_unused=True,
    )
    components = _flatten_gradients((policy_parameter, value_parameter), gradients)
    assert components[0].shape == (2, 3)
    assert np.array_equal(components[0], np.zeros((2, 3)))
    assert np.array_equal(components[1], np.full(4, 2.0))


def test_flat_component_vector_reconstructs_registered_parameter_order() -> None:
    components = (np.arange(6).reshape(2, 3), np.asarray([6.0, 7.0]))
    assert np.array_equal(_flat_vector(components), np.arange(8, dtype=np.float64))


def test_semantic_tamper_altered_coefficient_rejected() -> None:
    with pytest.raises(ValueError, match="coefficient_semantics_mismatch"):
        _validate_coefficients(
            np.asarray([1.0, 1.25], dtype=np.float32), np.ones(2, dtype=np.float32)
        )


def test_semantic_tamper_split_and_permutation_rejected() -> None:
    with pytest.raises(ValueError, match="split_semantics_mismatch"):
        _validate_split_and_order(
            np.asarray([1, 2]), np.asarray([1, 3]), np.asarray([0, 1])
        )
    with pytest.raises(ValueError, match="permutation_semantics_mismatch"):
        _validate_split_and_order(
            np.asarray([1, 2]), np.asarray([1, 2]), np.asarray([0, 0])
        )


def test_semantic_tamper_batch_denominator_rejected() -> None:
    with pytest.raises(ValueError, match="denominator_ratios_mismatch"):
        _validate_denominators(
            np.asarray([2.0, 2.0]), np.asarray([1.0, 3.0]), [1.0, 1.0]
        )


def test_semantic_tamper_parameter_layout_rejected() -> None:
    layout = [{"name": "weight", "shape": [2, 2]}]
    with pytest.raises(ValueError, match="registered_layout_mismatch"):
        _validate_layout([("weight", [4])], layout)


def test_semantic_tamper_component_vectors_rejected() -> None:
    zeros = np.zeros(2)
    with pytest.raises(ValueError, match="global_component_vector_mismatch"):
        _validate_component_vectors(zeros, zeros, zeros, np.ones(2), zeros)


def test_semantic_tamper_decision_field_rejected() -> None:
    evidence = {
        "decision": {"classification": "material_minibatch_denominator_mismatch"}
    }
    with pytest.raises(ValueError, match="corrected_decision_mismatch"):
        _validate_decision_field(
            evidence, "denominator_mismatch_below_screen_threshold"
        )
