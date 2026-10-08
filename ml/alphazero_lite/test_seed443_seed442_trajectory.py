"""Regression coverage for seed443 semantic validation primitives."""

from __future__ import annotations

import numpy as np
import pytest
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ml.alphazero_lite.verify_seed443_seed442_trajectory import (
    _validate_link,
    _validate_trials,
)


def test_parameter_state_link_rejects_wrong_initializer_and_broken_adjacency() -> None:
    initialized = [np.array([1.0, 2.0], dtype=np.float32)]
    _validate_link(initialized, initialized, "initial")
    with pytest.raises(ValueError, match="initial_invalid"):
        _validate_link(initialized, [np.array([1.0, 3.0], dtype=np.float32)], "initial")
    with pytest.raises(ValueError, match="adjacent_invalid"):
        _validate_link(initialized, [], "adjacent")


def test_trials_require_complete_rejection_coverage() -> None:
    scales = [1.0, 0.5, 0.25]
    failed = [{"scale": s, "batch_kl": 0.1, "guard_kl": 0.1} for s in scales]
    _validate_trials(failed, scales, 0.005, 1e-10, None, True, "trial")
    with pytest.raises(ValueError, match="trial_rejection_coverage_invalid"):
        _validate_trials(failed[:-1], scales, 0.005, 1e-10, None, True, "trial")


def test_first_passing_trial_must_be_selected() -> None:
    trials = [
        {"scale": 1.0, "batch_kl": 0.01, "guard_kl": 0.01},
        {"scale": 0.5, "batch_kl": 0.004, "guard_kl": 0.004},
    ]
    _validate_trials(trials, [1.0, 0.5], 0.005, 1e-10, 0.5, False, "trial")
    with pytest.raises(ValueError, match="trial_first_pass_invalid"):
        _validate_trials(trials, [1.0, 0.5], 0.005, 1e-10, 0.25, False, "trial")


def test_numpy_view_aliasing_requires_copy_for_historical_optimizer_states() -> None:
    live_state = np.array([0.0, 1.0], dtype=np.float32)
    raw_view = live_state
    snapshot = live_state.copy()
    live_state += 10.0
    assert np.array_equal(raw_view, [10.0, 11.0])
    assert np.array_equal(snapshot, [0.0, 1.0])


def test_cli_runs_from_unrelated_cwd_in_physical_source_only_copy() -> None:
    source_root = Path(__file__).resolve().parents[2]
    scratch = source_root / ".tmp/seed443-portability-check"
    checkout, elsewhere = scratch / "checkout", scratch / "elsewhere"
    if scratch.exists():
        shutil.rmtree(scratch)
    checkout.mkdir(parents=True)
    elsewhere.mkdir()
    (checkout / "ml").mkdir()
    shutil.copytree(source_root / "ml/alphazero_lite", checkout / "ml/alphazero_lite")
    data = checkout / "docs/data"
    data.mkdir(parents=True)
    for name in (
        "seed416-policy-target-softening",
        "seed426-canonical-overlap",
        "seed438-seed437-gradient-correction",
        "seed435-adam-direction-screen",
        "seed440-recorded-update-attribution",
        "seed441-seed440-publication-completion",
        "seed442-kl-capped-adam-screen",
        "seed443-seed442-trajectory",
    ):
        shutil.copytree(source_root / "docs/data" / name, data / name)
    init_rel = (
        "seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    init_dest = data / init_rel
    init_dest.parent.mkdir(parents=True)
    shutil.copy2(source_root / "docs/data" / init_rel, init_dest)
    assert not any(path.is_symlink() for path in checkout.rglob("*"))
    evidence = checkout / "docs/data/seed442-kl-capped-adam-screen"

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    before = {
        str(p.relative_to(checkout)): digest(p)
        for p in evidence.rglob("*")
        if p.is_file()
    }
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(checkout)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed443_seed442_trajectory",
            "--root",
            str(checkout),
        ],
        cwd=elsewhere,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert json.loads(result.stdout)["semantic"]["status"] == "valid"
    after = {
        str(p.relative_to(checkout)): digest(p)
        for p in evidence.rglob("*")
        if p.is_file()
    }
    assert before == after
