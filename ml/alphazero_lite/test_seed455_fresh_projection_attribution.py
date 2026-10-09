"""Focused tests for seed455's verifier-facing arithmetic and evidence checks."""

from __future__ import annotations

import numpy as np
import pytest

from ml.alphazero_lite import seed455_analysis as analysis
from ml.alphazero_lite import verify_seed455_fresh_projection_attribution as verifier


def test_equal_input_weights_keep_every_exposure() -> None:
    weights = analysis.row_weights(["a", "a", "b"], "equal_input")
    assert np.array_equal(weights, np.asarray([0.25, 0.25, 0.5]))
    assert weights.sum() == 1


def test_decomposition_has_zero_value_head_contribution() -> None:
    layout = [
        {"start": 0, "stop": 2, "group": "shared_trunk"},
        {"start": 2, "stop": 3, "group": "policy_head"},
        {"start": 3, "stop": 4, "group": "value_head"},
    ]
    result = analysis.decompose(
        1.0,
        1.2,
        np.asarray([1.0, 0.0, 2.0, 0.0]),
        np.asarray([0.1, 0.0, 0.2, 3.0]),
        layout,
    )
    assert result["s"] == pytest.approx(0.5)
    assert result["r"] == pytest.approx(-0.3)
    assert result["group_s"]["value_head"] == 0.0


def test_positive_remainder_and_zero_gradient_classification() -> None:
    assert analysis.decompose(
        1.0,
        1.1,
        np.zeros(2),
        np.ones(2),
        [{"start": 0, "stop": 2, "group": "shared_trunk"}],
    )["r"] == pytest.approx(0.1)
    assert (
        analysis.classify(
            {"D": 1e-5, "S": 1e-7, "R": 9.9e-6}, {"D": 1e-5, "S": 1e-7, "R": 9.9e-6}
        )
        == "fresh_unseen_remainder_dominant"
    )


@pytest.mark.parametrize(
    "exposure,equal,expected",
    [
        (
            {"D": 1e-5, "S": 6e-6, "R": 4e-6},
            {"D": 1e-5, "S": 6e-6, "R": 4e-6},
            "fresh_unseen_direction_dominant",
        ),
        (
            {"D": 1e-5, "S": 4e-6, "R": 6e-6},
            {"D": 1e-5, "S": 4e-6, "R": 6e-6},
            "fresh_unseen_remainder_dominant",
        ),
        (
            {"D": 1e-5, "S": 6e-6, "R": 4e-6},
            {"D": 1e-5, "S": 4e-6, "R": 6e-6},
            "mixed_or_weighting_dependent_attribution",
        ),
    ],
)
def test_fixed_classification_rule(exposure, equal, expected) -> None:
    assert analysis.classify(exposure, equal) == expected


def test_bad_group_coverage_is_rejected() -> None:
    with pytest.raises(ValueError, match="group_slope_reconciliation_failed"):
        analysis.decompose(
            0.0,
            1.0,
            np.ones(2),
            np.ones(2),
            [{"start": 0, "stop": 1, "group": "shared_trunk"}],
        )


def test_verifier_rejects_cohort_membership_weighting_and_target_tampering() -> None:
    rows = np.asarray([2, 3, 4], dtype=np.int64)
    identities = ["same", "same", "other"]
    targets = np.arange(15, dtype=np.float32).reshape(5, 3)
    archive = {
        "cohort_guard_compact_rows": rows.copy(),
        "cohort_guard_identities": np.asarray(identities, dtype="U"),
        "cohort_guard_targets": targets[rows].copy(),
    }
    verifier._check_cohort_archive(archive, "guard", rows, identities, targets)
    assert np.array_equal(
        verifier._cohort_weights(identities, "equal_input"),
        np.asarray([0.25, 0.25, 0.5]),
    )
    archive["cohort_guard_compact_rows"] = np.asarray([2, 4, 3])
    with pytest.raises(ValueError, match="cohort_order"):
        verifier._check_cohort_archive(archive, "guard", rows, identities, targets)
    archive["cohort_guard_compact_rows"] = rows
    archive["cohort_guard_targets"][0, 0] += 1
    with pytest.raises(ValueError, match="cohort_targets"):
        verifier._check_cohort_archive(archive, "guard", rows, identities, targets)


def test_verifier_rejects_prediction_and_gradient_mutations() -> None:
    expected = np.asarray([[1.0, 2.0]], dtype=np.float32)
    verifier._check_prediction(expected.copy(), expected, "baseline")
    with pytest.raises(ValueError, match="prediction"):
        verifier._check_prediction(expected + 1, expected, "tampered")
    verifier._check_gradient(expected.ravel().copy(), expected.ravel(), "baseline")
    with pytest.raises(ValueError, match="gradient"):
        verifier._check_gradient(expected.ravel() + 1, expected.ravel(), "tampered")


def test_verifier_rejects_state_link_mutation() -> None:
    prior = [np.asarray([1.0], dtype=np.float32)]
    verifier._check_trajectory_link(prior, [prior[0].copy()], "baseline")
    with pytest.raises(ValueError, match="trajectory_link"):
        verifier._check_trajectory_link(
            prior, [np.asarray([2.0], dtype=np.float32)], "tampered"
        )


def test_verifier_rejects_group_contribution_and_total_tampering() -> None:
    row = {
        "d": 0.3,
        "s": 0.2,
        "r": 0.1,
        "group_s": {"shared_trunk": 0.15, "policy_head": 0.05, "value_head": 0.0},
    }
    group_terms = {"shared_trunk": 0.15, "policy_head": 0.05, "value_head": 0.0}
    verifier._check_attribution(row, 0.3, 0.2, group_terms, "baseline")
    bad_group = {**group_terms, "shared_trunk": 0.16}
    with pytest.raises(ValueError, match="group_contribution"):
        verifier._check_attribution(row, 0.3, 0.2, bad_group, "tampered")
    record = {"totals": {"D": 0.3, "S": 0.2, "R": 0.1}}
    verifier._check_totals(record, record["totals"], "baseline")
    with pytest.raises(ValueError, match="total"):
        verifier._check_totals(record, {"D": 0.4, "S": 0.2, "R": 0.1}, "tampered")


def test_rebound_outer_inventory_does_not_bypass_semantic_checks(tmp_path) -> None:
    payload = tmp_path / "evidence.bin"
    payload.write_bytes(b"semantically wrong")
    digest = verifier.sha(payload)
    verifier._check_inventory(tmp_path, {"evidence.bin": digest})
    row = {"d": 0.2, "s": 0.1, "r": 0.1, "group_s": {"shared_trunk": 0.1}}
    with pytest.raises(ValueError, match="D:"):
        verifier._check_attribution(row, 0.3, 0.1, {"shared_trunk": 0.1}, "rebound")


def test_verifier_rejects_classification_tampering() -> None:
    exposure = {"D": 1e-5, "S": 7e-6, "R": 3e-6}
    equal = {"D": 1e-5, "S": 7e-6, "R": 3e-6}
    assert (
        verifier._classification(exposure, equal) == "fresh_unseen_direction_dominant"
    )
    verifier._check_classification("fresh_unseen_direction_dominant", exposure, equal)
    with pytest.raises(ValueError, match="classification"):
        verifier._check_classification(
            "fresh_unseen_remainder_dominant", exposure, equal
        )
