"""Tests for seed480 value-MSE correction semantics and verifier tampering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ml.alphazero_lite.verify_seed480_value_mse_correction import verify
from ml.alphazero_lite.seed479_population_ablation import (
    INPUT_PATHS,
    SOURCE_PATHS as SEED479_SOURCES,
)
from ml.alphazero_lite.seed480_value_mse_correction import (
    SOURCE_PATHS as SEED480_SOURCES,
)


def test_policy_squared_error_can_move_opposite_to_value_mse() -> None:
    # A synthetic two-output model: policy gets worse while its value becomes exact.
    policy_target = np.asarray([[1.0, 0.0]])
    initial_policy = np.asarray([[0.99, 0.01]])
    endpoint_policy = np.asarray([[0.5, 0.5]])
    value_target = np.asarray([0.75])
    initial_value = np.asarray([0.0])
    endpoint_value = np.asarray([0.75])
    p0 = np.square(initial_policy - policy_target).sum()
    p1 = np.square(endpoint_policy - policy_target).sum()
    v0 = np.square(initial_value - value_target).sum()
    v1 = np.square(endpoint_value - value_target).sum()
    assert p1 > p0
    assert v1 < v0


def test_unchanged_published_baseline_verifies() -> None:
    root = Path(__file__).resolve().parents[2]
    verify(root)


def _rebind(path: Path) -> None:
    receipt_path = path.parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _copy_verification_checkout(source: Path, destination: Path) -> None:
    import shutil

    shutil.copytree(
        source / "ml/alphazero_lite",
        destination / "ml/alphazero_lite",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for package_file in (source / "ml").glob("*.py"):
        target = destination / "ml" / package_file.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package_file, target)
    paths = set(SEED479_SOURCES) | set(SEED480_SOURCES)
    paths.update(str(path) for path in INPUT_PATHS.values())
    paths.update(
        str(path.relative_to(source))
        for directory in (
            source / "docs/data/seed479-population-ablation",
            source / "docs/data/seed480-value-mse-correction",
        )
        for path in directory.iterdir()
        if path.is_file()
    )
    paths.update(
        [
            "docs/data/seed466-replay-input-bundle/manifest.json",
            "docs/data/seed466-replay-input-bundle/training.npz",
            "docs/data/seed466-replay-input-bundle/evaluation.npz",
        ]
    )
    for relative in paths:
        src, dst = source / relative, destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


@pytest.mark.parametrize(
    "mutation",
    [
        "prediction",
        "swapped_policy_value_metric",
        "target",
        "weight",
        "cohort",
        "effect",
        "threshold",
        "gate",
        "decision",
        "historical",
    ],
)
def test_receipt_rebound_tampering_rejected(tmp_path: Path, mutation: str) -> None:
    root = Path(__file__).resolve().parents[2]
    _copy_verification_checkout(root, tmp_path)
    out = tmp_path / "docs/data/seed480-value-mse-correction"
    evidence_path = out / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    if mutation == "prediction":
        evidence["occurrences"]["pooled"][0]["prediction"]["F"] += 0.1
    elif mutation == "swapped_policy_value_metric":
        evidence["historical_policy_metric"]["label"] = "value_mse"
    elif mutation == "target":
        evidence["occurrences"]["pooled"][0]["value_target"] += 0.1
    elif mutation == "weight":
        evidence["occurrences"]["pooled"][0]["weights"]["exposure"] += 0.1
    elif mutation == "cohort":
        evidence["occurrences"]["pooled"].pop()
    elif mutation == "effect":
        evidence["effects"]["pooled"]["F_minus_initializer"]["exposure"] += 0.1
    elif mutation == "threshold":
        registration_path = out / "registration.json"
        registration = json.loads(registration_path.read_text())
        registration["gate_definition"] = registration["gate_definition"].replace(
            "0.01", "0.02"
        )
        registration_path.write_text(
            json.dumps(registration, indent=2, sort_keys=True) + "\n"
        )
        _rebind(registration_path)
    elif mutation == "gate":
        evidence["gates"]["F_full_training_policy_CE_decreases"] = not evidence[
            "gates"
        ]["F_full_training_policy_CE_decreases"]
    elif mutation == "decision":
        evidence["decision"] = "close_full_population_direction_branch"
    else:
        evidence["historical_policy_metric"]["metrics"]["pooled"]["F"]["exposure"] += (
            0.1
        )
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    _rebind(evidence_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        verify(tmp_path)
