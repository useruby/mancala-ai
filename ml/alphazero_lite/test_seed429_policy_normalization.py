"""Focused tests for seed429's compact-row coefficient transform."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import json

from ml.alphazero_lite.seed429_policy_normalization import (
    normalize_policy_coefficients,
    reconstruct_frozen_census,
)

ROOT = Path(__file__).resolve().parents[2]


def test_mass_conservation_and_equal_identity_mass_with_unequal_repetitions() -> None:
    # Identity x appears three times and y once; their starting masses differ.
    q = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result, census = normalize_policy_coefficients(
        q,
        ["x", "x", "x", "y"],
        np.asarray([40, 40, 40, 40]),
        ["fresh"] * 4,
        np.ones(4, dtype=bool),
    )
    np.testing.assert_allclose(result[:3].sum(), 5.0, rtol=0, atol=1e-6)
    assert result[3] == pytest.approx(5.0)
    assert result.astype(np.float64).sum() == pytest.approx(q.sum())
    group = census["groups"]["fresh/>32"]
    assert group["positive_identities"] == 2
    assert group["effective_concentration"] == pytest.approx(0.52)


def test_scopes_by_source_bucket_and_leaves_validation_low_stone_and_zero_rows() -> (
    None
):
    q = np.asarray([1, 2, 3, 0, 5, 7], dtype=np.float32)
    identities = ["a", "a", "a", "z", "a", "a"]
    stones = np.asarray([40, 40, 16, 40, 40, 40])
    sources = ["s1", "s1", "s1", "s1", "s2", "s1"]
    training = np.asarray([True, True, True, True, True, False])
    result, report = normalize_policy_coefficients(
        q, identities, stones, sources, training
    )
    assert result[0] == q[0]  # one positive identity in this source/bucket
    assert result[2] == q[2]  # <=16 is unchanged
    assert result[3] == q[3]  # zero coefficient remains zero
    assert result[4] == q[4]  # source boundary
    assert result[5] == q[5]  # validation row
    assert report["rows_changed"] == 0
    assert report["normalization_is_noop"]


@pytest.mark.parametrize(
    "coefficients", [[1.0, -1.0], [1.0, float("nan")], [1.0, float("inf")]]
)
def test_rejects_malformed_coefficients(coefficients: list[float]) -> None:
    with pytest.raises(ValueError, match="malformed_policy_coefficients"):
        normalize_policy_coefficients(
            np.asarray(coefficients),
            ["a", "b"],
            np.asarray([40, 40]),
            ["source", "source"],
            np.asarray([True, True]),
        )


def test_published_census_reconstructs_exact_float32_vectors_and_copy_counts() -> None:
    actual = reconstruct_frozen_census(ROOT)
    published = json.loads(
        (
            ROOT
            / "docs/data/seed429-canonical-policy-normalization/coefficient-census.json"
        ).read_text(encoding="utf-8")
    )
    for key in (
        "control_coefficients_sha256",
        "treatment_coefficients_sha256",
        "rows_changed",
        "source_row_count",
        "training_compact_row_count",
        "validation_compact_row_count",
    ):
        assert actual[key] == published[key]
    assert actual["normalization_is_noop"] is False
    assert actual["validation_coefficients_unchanged"] is True
    assert actual["low_stone_coefficients_unchanged"] is True
    assert actual["expanded_replay_position_count"] == sum(
        row["compact_rows"] * row["weight"] for row in actual["replay_sources"]
    )
    assert actual["expanded_replay_position_count"] > actual["source_row_count"]
    for group in actual["groups"].values():
        assert abs(group["mass_delta"]) < 0.002
        assert group["positive_identity_rows_repeated_beyond_first"] >= 0
        assert group["maximum_positive_compact_row_multiplicity"] >= 1
