"""Synthetic algebra and semantic-tamper tests for seed445."""

from __future__ import annotations

import numpy as np
import os
import json
import pytest
import shutil
import subprocess
import sys
import torch
from pathlib import Path

from ml.alphazero_lite.seed445_finite_step_attribution import (
    classify,
    decompose,
    shaped_gradient_components,
    weight_vectors,
)
from ml.alphazero_lite.verify_seed445_finite_step_attribution import (
    _array_close,
    _close,
    _linked,
)


def _groups() -> dict[str, np.ndarray]:
    return {
        "shared_trunk": np.array([0, 1]),
        "policy_head": np.array([2]),
        "value_head": np.array([3]),
    }


def test_linear_loss_has_zero_residual_and_group_dots_reconcile() -> None:
    x0 = np.array([1.0, -2.0, 0.5, 3.0])
    delta = np.array([0.2, 0.4, -0.3, 0.1])
    grad = np.array([2.0, -1.0, 4.0, 0.5])
    before = float(grad @ x0)
    after = float(grad @ (x0 + delta))
    got = decompose(before, after, grad, delta, _groups())
    assert got["r"] == pytest.approx(0.0, abs=1e-14)
    assert sum(got["group_s"].values()) == pytest.approx(got["s"])


def test_quadratic_loss_residual_is_signed_and_not_grouped() -> None:
    x0 = np.array([1.0, 2.0])
    delta = np.array([0.5, -0.25])
    hessian = np.diag([-2.0, -4.0])
    grad = hessian @ x0
    before = 0.5 * float(x0 @ hessian @ x0)
    end = x0 + delta
    after = 0.5 * float(end @ hessian @ end)
    got = decompose(
        before,
        after,
        grad,
        delta,
        {"shared_trunk": np.array([0]), "value_head": np.array([1])},
    )
    assert got["r"] == pytest.approx(0.5 * float(delta @ hessian @ delta))
    assert got["r"] < 0
    assert set(got) == {"d", "s", "r", "group_s"}


def test_unused_parameters_keep_shaped_float64_zero_slots() -> None:
    params = (torch.nn.Parameter(torch.ones(2, 3)), torch.nn.Parameter(torch.ones(4)))
    components = shaped_gradient_components(
        params, (None, torch.arange(4, dtype=torch.float32))
    )
    assert components[0].shape == (2, 3)
    assert components[0].dtype == np.float64
    assert np.count_nonzero(components[0]) == 0
    assert np.array_equal(components[1], np.arange(4, dtype=np.float64))


def test_duplicate_exposures_are_retained_with_equal_identity_mass() -> None:
    weights = weight_vectors(["a", "a", "b"])
    assert weights["exposure_weighted"].tolist() == pytest.approx([1 / 3] * 3)
    assert weights["equal_input"].tolist() == pytest.approx([0.25, 0.25, 0.5])
    assert weights["equal_input"].sum() == pytest.approx(1)


def test_all_classifier_branches_use_signed_contributions() -> None:
    first = {w: {"D": 1.0, "S": 0.75, "R": 0.25} for w in ("a", "b")}
    residual = {w: {"D": 1.0, "S": 0.1, "R": 0.9} for w in ("a", "b")}
    mixed = {"a": {"D": 1.0, "S": 0.8, "R": 0.2}, "b": {"D": 1.0, "S": 0.2, "R": 0.8}}
    assert classify(first) == "first_order_value_harm_dominant"
    assert classify(residual) == "finite_step_value_residual_dominant"
    assert classify(mixed) == "mixed_or_no_common_value_dominance"
    assert (
        classify({w: {"D": -1.0, "S": 10.0, "R": -11.0} for w in ("a", "b")})
        == "mixed_or_no_common_value_dominance"
    )


def test_broken_trajectory_and_altered_evidence_fields_are_rejected() -> None:
    state = [np.array([1.0, 2.0], dtype=np.float32)]
    _linked(state, [state[0].copy()], "good")
    with pytest.raises(ValueError, match="link_invalid"):
        _linked(state, [np.array([1.0, 3.0], dtype=np.float32)], "link")
    with pytest.raises(ValueError, match="prediction_mismatch"):
        _array_close(np.array([1.0]), np.array([1.1]), "prediction")
    with pytest.raises(ValueError, match="gradient_mismatch"):
        _array_close(np.array([1.0]), np.array([0.9]), "gradient")
    with pytest.raises(ValueError, match="ledger_mismatch"):
        _close(1.0, 1.01, 1e-8, 1e-8, "ledger")


def test_group_keys_and_zero_unused_gradient_contribute_no_residual() -> None:
    params = (torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(2)))
    grad = np.concatenate(
        [
            a.reshape(-1)
            for a in shaped_gradient_components(params, (torch.ones(1), None))
        ]
    )
    delta = np.array([0.2, 3.0, -2.0])
    parts = decompose(
        2.0,
        2.2,
        grad,
        delta,
        {"shared_trunk": np.array([0]), "value_head": np.array([1, 2])},
    )
    assert parts["s"] == pytest.approx(0.2)
    assert parts["group_s"]["value_head"] == 0
    assert parts["r"] == pytest.approx(0)


def test_read_only_verifier_in_physical_source_only_checkout() -> None:
    source_root = Path(__file__).resolve().parents[2]
    published = source_root / "docs/data/seed445-finite-step-attribution"
    if not (published / "receipt.json").is_file():
        pytest.skip("requires completed seed445 publication")
    scratch = source_root / ".tmp/seed445-physical-copy-check"
    checkout, elsewhere = scratch / "checkout", scratch / "elsewhere"
    if scratch.exists():
        shutil.rmtree(scratch)
    (checkout / "ml").mkdir(parents=True)
    elsewhere.mkdir(parents=True)
    shutil.copytree(source_root / "ml/alphazero_lite", checkout / "ml/alphazero_lite")
    data = checkout / "docs/data"
    data.mkdir(parents=True)
    for name in (
        "seed416-policy-target-softening",
        "seed426-canonical-overlap",
        "seed442-kl-capped-adam-screen",
        "seed443-seed442-trajectory",
        "seed444-minibatch-denominator-audit",
        "seed445-finite-step-attribution",
    ):
        shutil.copytree(source_root / "docs/data" / name, data / name)
    init_rel = (
        "seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    init_dest = data / init_rel
    init_dest.parent.mkdir(parents=True)
    shutil.copy2(source_root / "docs/data" / init_rel, init_dest)
    assert not any(path.is_symlink() for path in checkout.rglob("*"))
    before = {
        str(path.relative_to(checkout)): path.read_bytes()
        for path in (data / "seed442-kl-capped-adam-screen").rglob("*")
        if path.is_file()
    }
    env = os.environ.copy()
    env["PYTHONPATH"] = str(checkout)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed445_finite_step_attribution",
            "--root",
            str(checkout),
        ],
        cwd=elsewhere,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert json.loads(proc.stdout)["status"] == "valid"
    after = {
        str(path.relative_to(checkout)): path.read_bytes()
        for path in (data / "seed442-kl-capped-adam-screen").rglob("*")
        if path.is_file()
    }
    assert before == after
