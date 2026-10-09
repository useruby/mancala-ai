"""Synthetic semantic tests for seed451 objective and cohort arithmetic."""

from __future__ import annotations

import numpy as np
import pytest
import json
import hashlib
import shutil
from pathlib import Path

from ml.alphazero_lite.seed451_source_policy_alignment import (
    _cos,
    _equal_identity_weights,
    _norm,
    classify_fresh_alignment,
)
from ml.alphazero_lite.verify_seed451_source_policy_alignment import verify

ROOT = Path(__file__).resolve().parents[2]


def test_equal_exact_input_weights_preserve_duplicate_exposures():
    weights = _equal_identity_weights(["a", "a", "b"])
    assert weights == pytest.approx([0.25, 0.25, 0.5])
    assert weights.sum() == pytest.approx(1)


def test_unequal_policy_masses_and_value_coefficient_once():
    fresh = np.array([2.0, 0.0])
    historical = np.array([0.0, 6.0])
    rho_f, rho_h = 0.25, 0.75
    policy = rho_f * fresh + rho_h * historical
    value = np.array([4.0, 8.0])
    total = policy + 0.3 * value
    assert total == pytest.approx([1.7, 6.9])


def test_source_partition_keeps_overlapping_identity_cohorts_distinct():
    fresh = {"shared", "fresh-only"}
    historical = {"shared", "history-only"}
    assert fresh & historical == {"shared"}


def test_zero_vectors_have_undefined_cosine():
    zero = np.zeros(3)
    assert _norm(zero) == 0
    assert _cos(zero, np.ones(3)) is None


def _metrics(pf: float, ph: float, total: float) -> dict:
    return {
        "G_PF": {"cosine": pf},
        "G_PH": {"cosine": ph},
        "G_T": {"cosine": total},
    }


def test_classifier_opposition_precedes_positive_joint_alignment():
    rows = [_metrics(0.1, -0.1, 0.8), _metrics(0.2, -0.3, 0.7)]
    assert classify_fresh_alignment(rows) == "historical_policy_opposition_present"


@pytest.mark.parametrize(
    "metrics,expected",
    [
        (
            [_metrics(0.1, 0.0, 0.1), _metrics(0.1, 0.0, 0.1)],
            "fresh_first_order_alignment_positive",
        ),
        (
            [_metrics(0.1, 0.0, 0.099), _metrics(0.1, 0.0, 0.2)],
            "mixed_or_checkpoint_dependent_alignment",
        ),
        (
            [_metrics(0.1, None, 0.5), _metrics(0.1, None, 0.5)],
            "fresh_first_order_alignment_positive",
        ),
        (
            [_metrics(None, None, None), _metrics(None, None, None)],
            "mixed_or_checkpoint_dependent_alignment",
        ),
    ],
)
def test_classifier_boundaries_and_undefined_cosines(metrics, expected):
    assert classify_fresh_alignment(metrics) == expected


def _mutable_publication(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    data = root / "docs/data"
    data.mkdir(parents=True)
    for path in (ROOT / "docs/data").iterdir():
        if path.name != "seed451-source-policy-alignment":
            (data / path.name).symlink_to(path, target_is_directory=True)
    out = data / "seed451-source-policy-alignment"
    shutil.copytree(ROOT / "docs/data/seed451-source-policy-alignment", out)
    (root / "ml").symlink_to(ROOT / "ml", target_is_directory=True)
    return root


def _rebind_receipt(out: Path, filename: str) -> None:
    receipt_path = out / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["files_sha256"][filename] = hashlib.sha256(
        (out / filename).read_bytes()
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize(
    "tamper", ["mapping", "denominator", "group", "classification"]
)
def test_semantic_result_tampering_is_rejected(tmp_path, tamper):
    root = _mutable_publication(tmp_path)
    out = root / "docs/data/seed451-source-policy-alignment"
    path = out / "results.json"
    result = json.loads(path.read_text())
    checkpoint = result["checkpoints"]["initializer"]
    if tamper == "mapping":
        result["parameter_groups"]["shared_trunk"]["indices"].pop()
    elif tamper == "denominator":
        checkpoint["rho_F"] += 0.05
    elif tamper == "group":
        checkpoint["cohorts"]["U_F_equal_exact_input"]["group_accounting"][
            "shared_trunk"
        ]["joint"] += 1
    else:
        result["classification"] = "fresh_first_order_alignment_positive"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _rebind_receipt(out, "results.json")
    with pytest.raises(ValueError):
        verify(root)


def test_semantic_gradient_tampering_is_rejected(tmp_path):
    root = _mutable_publication(tmp_path)
    out = root / "docs/data/seed451-source-policy-alignment"
    archive_path = out / "gradient-vectors.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    arrays["initializer__G_PF"][0] += 1
    np.savez_compressed(archive_path, **arrays)
    _rebind_receipt(out, "gradient-vectors.npz")
    with pytest.raises(ValueError):
        verify(root)
