"""Semantic tamper checks for seed449's independently reconstructed row evidence."""

import copy

import pytest

from ml.alphazero_lite.verify_seed449_policy_gain_localization import (
    _ce,
    _independent_summary,
    _validate_classification,
    _validate_identity_ledger,
    _validate_rows,
    _validate_summary,
)


def _rows():
    return [
        {
            "compact_row": 10,
            "source_ref": {"source": "s", "source_row": 1},
            "input_identity": "x",
            "canonical_identity": "cx",
            "legal_mask": [1, 1, 0, 0, 0, 0],
            "policy_target": [1.0, 0, 0, 0, 0, 0],
            "losses": {"initializer": 1.0, "C": 0.9, "T": 0.8},
        },
        {
            "compact_row": 11,
            "source_ref": {"source": "s", "source_row": 2},
            "input_identity": "x",
            "canonical_identity": "cx",
            "legal_mask": [1, 1, 0, 0, 0, 0],
            "policy_target": [0.0, 1.0, 0, 0, 0, 0],
            "losses": {"initializer": 1.0, "C": 1.1, "T": 1.2},
        },
        {
            "compact_row": 12,
            "source_ref": {"source": "s", "source_row": 3},
            "input_identity": "y",
            "canonical_identity": "cy",
            "legal_mask": [1, 1, 0, 0, 0, 0],
            "policy_target": [0.5, 0.5, 0, 0, 0, 0],
            "losses": {"initializer": 1.0, "C": 0.9, "T": 0.8},
        },
    ]


def test_independent_summary_retains_rows_with_different_targets():
    summary = _independent_summary(_rows(), "initializer", "T")
    duplicate = next(
        row for row in summary["identities"] if row["input_identity"] == "x"
    )
    assert duplicate["exposure_count"] == 2
    assert duplicate["row_changes"] == pytest.approx([-0.2, 0.2])
    assert duplicate["mean_change"] == 0


@pytest.mark.parametrize(
    "mask,target",
    [
        ([0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]),
        ([1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]),
    ],
)
def test_independent_loss_reconstruction_rejects_mask_and_target_tampering(
    mask, target
):
    with pytest.raises(ValueError):
        _ce(
            __import__("numpy").zeros(6, dtype="float32"),
            __import__("numpy").array(target),
            __import__("numpy").array(mask),
        )


def test_identity_count_loss_and_coverage_changes_change_reconstruction():
    baseline = _independent_summary(_rows(), "initializer", "T")
    for mutate in (
        lambda rows: rows[1].update(input_identity="z"),
        lambda rows: rows[0]["losses"].update(T=8.0),
        lambda rows: rows.pop(),
    ):
        changed = copy.deepcopy(_rows())
        mutate(changed)
        assert _independent_summary(changed, "initializer", "T") != baseline


@pytest.mark.parametrize(
    "field",
    [
        "legal_mask",
        "policy_target",
        "input_identity",
        "exposure_count",
        "initial_mean_loss",
        "comparison_mean_loss",
        "mean_change",
        "exposure_mean_change",
        "equal_identity_mean_change",
        "difference",
        "population_covariance",
        "gross_improvement_mass",
        "gross_worsening_mass",
        "exposure_signed_contribution",
        "equal_signed_contribution",
        "classification",
    ],
)
def test_semantic_tampering_is_rejected(field):
    rows = _rows()
    summary = _independent_summary(rows, "initializer", "T")
    if field in {"legal_mask", "policy_target", "input_identity"}:
        tampered_rows = copy.deepcopy(rows)
        tampered_rows[0][field] = "tampered"
        with pytest.raises(ValueError, match="row_semantics"):
            _validate_rows(rows, tampered_rows)
        return
    if field in {
        "exposure_count",
        "initial_mean_loss",
        "comparison_mean_loss",
        "mean_change",
        "exposure_mean_change",
        "equal_identity_mean_change",
        "difference",
        "population_covariance",
        "gross_improvement_mass",
        "gross_worsening_mass",
        "exposure_signed_contribution",
        "equal_signed_contribution",
    }:
        key = "exposure_count" if field == "exposure_count" else field
        tampered = copy.deepcopy(summary)
        if key in tampered["strata"]["1"]:
            tampered["strata"]["1"][key] = -99
        elif key in tampered["identities"][0]:
            tampered["identities"][0][key] = -99
        elif key in tampered:
            tampered[key] = -99
        else:
            # Fields such as contributions are checked recursively in strata.
            tampered["strata"]["1"]["exposure_signed_contribution"] = -99
        if field in {"initial_mean_loss", "comparison_mean_loss", "mean_change"}:
            with pytest.raises(ValueError, match="identity_ledger_semantics"):
                _validate_identity_ledger(tampered["identities"], summary["identities"])
        else:
            with pytest.raises(ValueError, match="semantic_field"):
                _validate_summary("test", tampered, summary)
        return
    with pytest.raises(ValueError, match="classification_invalid"):
        _validate_classification(
            "tampered_classification", {"T_minus_initializer": summary}
        )
