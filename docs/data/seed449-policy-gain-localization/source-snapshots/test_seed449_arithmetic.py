"""Synthetic tests for seed449 arithmetic and descriptive classification."""

import pytest

from ml.alphazero_lite.seed449_arithmetic import analyze, classify, policy_ce


def fixture_rows(deltas: list[tuple[str, float]]) -> list[dict]:
    return [
        {
            "input_identity": key,
            "canonical_identity": key,
            "compact_row": i,
            "source_ref": {"source": "fixture", "row": i},
            "losses": {"I": 1.0, "M": 1.0 + value},
        }
        for i, (key, value) in enumerate(deltas)
    ]


def test_ce_uses_only_legal_actions_and_rejects_illegal_target_mass():
    assert policy_ce(
        [0.0] * 6, [0.5, 0.5, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]
    ) == pytest.approx(0.6931471805599453)
    with pytest.raises(ValueError, match="target_mass_on_illegal"):
        policy_ce([0.0] * 6, [0.5, 0, 0.5, 0, 0, 0], [1, 1, 0, 0, 0, 0])


def test_duplicate_exposures_different_targets_are_row_averaged_and_covariance_reconciles():
    result = analyze(fixture_rows([("a", -0.2), ("a", -0.4), ("b", 0.2)]), "I", "M")
    assert result["exposure_count"] == 3
    assert result["identity_count"] == 2
    assert result["exposure_mean_change"] == pytest.approx(-0.13333333333333333)
    assert result["equal_identity_mean_change"] == pytest.approx(-0.05)
    assert result["difference"] == pytest.approx(
        result["covariance_over_mean_exposure"]
    )


def test_gross_signed_mass_retains_cancellation_and_zero_fraction():
    result = analyze(fixture_rows([("a", -1.0), ("b", 1.0), ("c", 0.0)]), "I", "M")
    assert result["gross_improvement_mass"] == -1.0
    assert result["gross_worsening_mass"] == 1.0
    assert result["net_signed_mass"] == 0.0
    assert result["strata"]["1"]["unchanged_fraction"] == pytest.approx(1 / 3)


def test_empty_repeated_stratum_is_mixed():
    result = analyze(fixture_rows([("a", -0.1), ("b", -0.2)]), "I", "M")
    assert classify(result) == "mixed_policy_response"


def test_broad_small_boundary_is_strict_at_minus_point_zero_zero_five():
    rows = fixture_rows([("a", -0.004), ("b", -0.002), ("c", -0.003), ("c", -0.003)])
    assert classify(analyze(rows, "I", "M")) == "broad_small_policy_gain"
    rows = fixture_rows([("a", -0.005), ("b", -0.005), ("b", -0.005)])
    assert classify(analyze(rows, "I", "M")) == "mixed_policy_response"


def test_frequency_concentrated_thresholds_are_inclusive_and_exclusive_as_frozen():
    at_boundary = {
        "exposure_count": 3,
        "exposure_mean_change": -0.005,
        "equal_identity_mean_change": -0.004,
        "population_covariance": -0.001,
        "strata": {
            "1": {
                "exposure_weighted_mean": 0.0,
                "exposure_count": 1,
                "exposure_signed_contribution": 0.0,
            },
            "2-4": {"exposure_count": 2, "exposure_signed_contribution": -0.005},
            ">=5": {"exposure_count": 0, "exposure_signed_contribution": 0.0},
        },
    }
    assert classify(at_boundary) == "frequency_concentrated_policy_gain"
    at_equal_boundary = {**at_boundary, "equal_identity_mean_change": -0.005}
    assert classify(at_equal_boundary) == "mixed_policy_response"
