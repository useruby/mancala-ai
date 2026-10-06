"""Regression coverage for global-denominator source attribution."""

from __future__ import annotations

from ml.alphazero_lite.seed428_source_attribution import decompose


def _row(identity: str, source: str, diff: float, weight: float = 1.0) -> dict:
    return {
        "membership": {
            "subset": "unseen",
            "active_stones": 40,
            "canonical_identity": identity,
            "source": source,
        },
        "losses": {"policy_weight": weight},
        "initializer_policy_loss": 1.0,
        "e4_policy_loss": 1.0 + diff,
    }


def test_global_identity_denominator_cross_source_and_weighted_copies() -> None:
    rows = [
        _row("shared", "fresh", 1.0, 2.0),
        _row("shared", "fresh", 1.0, 1.0),
        _row("shared", "generic_bootstrap", -0.5, 1.0),
        _row("only-fresh", "fresh", 0.0, 1.0),
    ]
    result = decompose(rows)["unseen/>32"]
    source = result["source_contributions"]
    assert result["weighted_positions"] == 4
    assert result["canonical_identities"] == 2
    assert result["identities_in_multiple_sources"] == 1
    assert source["fresh"]["weighted_positions"] == 3
    assert source["fresh"]["canonical_identities"] == 2
    assert source["fresh"]["exposure_contribution"] == 0.6
    assert source["generic_bootstrap"]["exposure_contribution"] == -0.1
    # Shared identity keeps its combined weight denominator of four; the
    # source numerators are divided by that denominator before global averaging.
    assert source["fresh"]["equal_identity_contribution"] == 0.375
    assert source["generic_bootstrap"]["equal_identity_contribution"] == -0.0625
    assert sum(item["exposure_contribution"] for item in source.values()) == 0.5
    assert (
        sum(item["equal_identity_contribution"] for item in source.values()) == 0.3125
    )


def test_empty_cohorts_are_zero_and_finite() -> None:
    result = decompose([])
    for cohort in result.values():
        assert cohort["weighted_positions"] == 0
        assert cohort["canonical_identities"] == 0
        assert all(
            row["exposure_contribution"] == 0.0
            and row["equal_identity_contribution"] == 0.0
            for row in cohort["source_contributions"].values()
        )
