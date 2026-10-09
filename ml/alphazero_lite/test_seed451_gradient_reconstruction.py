"""Synthetic tests for independent seed451 input and gradient reconstruction."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.alphazero_lite.verify_seed451_gradient_reconstruction import (
    _compare_gradient,
    _derive_classification,
    _expand_replay_rows,
    _identity_weights,
    _flat,
    _validate_classification,
    _validate_denominators,
    _validate_archive_names,
    _validate_geometry,
    _validate_group_accounting,
    _validate_loader_materialization,
    _validate_source_mapping,
)


def test_replay_copies_are_distinct_from_compact_duplicate_rows():
    # Compact rows 0 and 1 can encode repeated inputs; replay weight 3 repeats each
    # compact row three times, while row 2 is a distinct single-copy source row.
    expanded = _expand_replay_rows([2, 1], [3, 1])
    assert expanded.tolist() == [0, 1, 0, 1, 0, 1, 2]
    input_identity_by_compact = ["same", "same", "other"]
    assert [input_identity_by_compact[i] for i in expanded].count("same") == 6
    assert len(set(input_identity_by_compact)) == 2


def test_source_order_mapping_and_invalid_multiplicity_are_rejected():
    assert _expand_replay_rows([2, 1], [1, 4]).tolist() == [0, 1, 2, 2, 2, 2]
    with pytest.raises(ValueError, match="invalid_source_multiplicity"):
        _expand_replay_rows([1], [0])


def test_equal_input_weights_keep_exposures_and_normalize_identity_mass():
    weights = _identity_weights(["a", "a", "b"])
    assert weights == pytest.approx([0.25, 0.25, 0.5])
    assert weights.sum() == pytest.approx(1.0)


def test_source_cohorts_are_row_partitioned_and_identity_overlap_is_reportable():
    row_sources = np.asarray(["fresh", "history", "history", "fresh"])
    fresh_rows = np.flatnonzero(row_sources == "fresh")
    history_rows = np.flatnonzero(row_sources != "fresh")
    row_identities = ["shared", "shared", "history-only", "fresh-only"]
    assert fresh_rows.tolist() == [0, 3]
    assert history_rows.tolist() == [1, 2]
    assert {row_identities[i] for i in fresh_rows} & {
        row_identities[i] for i in history_rows
    } == {"shared"}


def test_unused_parameters_are_covered_by_explicit_zero_components():
    used = torch.nn.Parameter(torch.tensor([2.0]))
    unused = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    gradient = torch.autograd.grad(
        used.square().sum(), (used, unused), allow_unused=True
    )
    flattened = _flat(gradient, (used, unused))
    assert flattened.tolist() == [4.0, 0.0, 0.0]


def test_chunked_linear_gradient_reconciles_and_value_coefficient_is_once():
    # For a linear scalar model, summing per-chunk gradients equals the full gradient.
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    features = torch.tensor([[1.0], [2.0], [3.0]])
    policy_component = []
    value_component = []
    for start in (0, 2):
        batch = features[start : start + 2]
        policy_loss = (batch * parameter).sum()
        value_loss = (batch * parameter - 1.0).square().sum()
        policy_component.append(
            torch.autograd.grad(policy_loss, parameter, retain_graph=True)[0]
        )
        value_component.append(torch.autograd.grad(value_loss, parameter)[0])
    policy_gradient = sum(policy_component)
    value_gradient = sum(value_component) / len(features)
    complete_direct = torch.autograd.grad(
        (features * parameter).sum() / 1.0
        + 0.3 * (features * parameter - 1.0).square().mean(),
        parameter,
    )[0]
    reconstructed = policy_gradient + 0.3 * value_gradient
    assert float(reconstructed) == pytest.approx(float(complete_direct))
    assert float(reconstructed) != pytest.approx(
        float(policy_gradient + 0.3 * 0.3 * value_gradient)
    )


def test_unequal_source_coefficient_masses_reconcile_policy_mixture():
    fresh_gradient = np.asarray([2.0, 1.0])
    historical_gradient = np.asarray([-1.0, 3.0])
    rho_f, rho_h = 2.0 / 5.0, 3.0 / 5.0
    combined = rho_f * fresh_gradient + rho_h * historical_gradient
    direct = np.asarray([0.2, 2.2])
    assert combined == pytest.approx(direct)


def _five_source_fixture():
    sources = (
        "fresh",
        "generic_bootstrap",
        "random_teacher",
        "opening_disagreement",
        "stability",
    )
    ledger = [
        {
            "compact_row": i,
            "source": source,
            "input_hex": np.asarray([i], dtype="<f4").tobytes().hex(),
            "policy_coefficient": 1.0,
        }
        for i, source in enumerate(sources)
    ]
    registration = {"replays": [{"weight": 1} for _ in sources]}
    x = np.arange(5, dtype=np.float32).reshape(-1, 1)
    policy = np.zeros((5, 6), dtype=np.float32)
    value = np.zeros((5, 1), dtype=np.float32)
    coefficients = np.ones(5, dtype=np.float32)
    replay = np.arange(5, dtype=np.int64)
    return sources, ledger, registration, x, policy, value, coefficients, replay


def test_loader_mapping_rejects_altered_targets_and_source_assignments():
    sources, ledger, registration, x, policy, value, coefficients, replay = (
        _five_source_fixture()
    )
    rebuilt = _validate_loader_materialization(
        x,
        policy,
        policy.copy(),
        value,
        value.copy(),
        coefficients,
        replay,
        ledger,
        registration,
    )
    assert rebuilt.tolist() == list(sources)
    with pytest.raises(ValueError, match="policy_target_mapping"):
        _validate_loader_materialization(
            x,
            policy,
            np.ones_like(policy),
            value,
            value.copy(),
            coefficients,
            replay,
            ledger,
            registration,
        )
    with pytest.raises(ValueError, match="value_target_mapping"):
        _validate_loader_materialization(
            x,
            policy,
            policy.copy(),
            value,
            np.ones_like(value),
            coefficients,
            replay,
            ledger,
            registration,
        )
    changed = [dict(row) for row in ledger]
    changed[0]["source"] = "generic_bootstrap"
    with pytest.raises(ValueError, match="source_order_mapping"):
        _validate_source_mapping(changed, tuple(sources), [1] * len(sources))


def test_denominator_and_gradient_component_tampering_are_rejected():
    record = {
        "rho_F": 0.4,
        "rho_H": 0.6,
        "policy_mass": 10.0,
        "policy_denominator": 10.0,
        "value_denominator": 8,
        "train_exposures": 8,
    }
    _validate_denominators(record, 4.0, 6.0, 8)
    record["policy_denominator"] = 9.0
    with pytest.raises(ValueError, match="independent_metric_mismatch"):
        _validate_denominators(record, 4.0, 6.0, 8)
    vector = np.asarray([0.25, -0.5])
    assert _compare_gradient("fixture", vector, vector.copy()) == 0.0
    with pytest.raises(ValueError, match="independent_gradient_mismatch"):
        _compare_gradient("fixture", vector + 0.1, vector)


def test_group_total_geometry_and_classification_tampering_are_rejected():
    unseen = np.asarray([1.0, 0.0])
    gradient = np.asarray([1.0, 1.0])
    geometry = {
        "dot": 1.0,
        "norm": float(np.sqrt(2.0)),
        "cosine": float(1 / np.sqrt(2.0)),
        "unit_direction_loss_derivative": float(-1 / np.sqrt(2.0)),
    }
    _validate_geometry(unseen, gradient, geometry, "fixture")
    geometry["cosine"] = 0.0
    with pytest.raises(ValueError, match="independent_metric_mismatch"):
        _validate_geometry(unseen, gradient, geometry, "fixture")
    fields = {
        "G_PF": np.asarray([1.0, 0.0]),
        "G_PH": np.asarray([0.0, 1.0]),
        "G_V": np.zeros(2),
        "G_T": np.asarray([0.5, 0.5]),
    }
    group_totals = {"fresh": 0.5, "historical": 0.5, "value": 0.0, "joint": 1.0}
    _validate_group_accounting(
        np.ones(2), fields, [0, 1], 0.5, 0.5, group_totals, "fixture"
    )
    group_totals["joint"] += 0.2
    with pytest.raises(ValueError, match="independent_metric_mismatch"):
        _validate_group_accounting(
            np.ones(2), fields, [0, 1], 0.5, 0.5, group_totals, "fixture"
        )
    cosines = {
        "initializer": {"G_PF": 0.2, "G_PH": -0.2, "G_T": 0.0},
        "T_final": {"G_PF": 0.6, "G_PH": 0.3, "G_T": 0.7},
    }
    assert _derive_classification(cosines) == "mixed_or_checkpoint_dependent_alignment"
    with pytest.raises(ValueError, match="classification_mismatch"):
        _validate_classification(cosines, "historical_policy_opposition_present")
    with pytest.raises(ValueError, match="archived_gradient_coverage"):
        _validate_archive_names(["gradient", "gradient"], {"gradient"})
    with pytest.raises(ValueError, match="archived_gradient_coverage"):
        _validate_archive_names(["misnamed"], {"gradient"})
