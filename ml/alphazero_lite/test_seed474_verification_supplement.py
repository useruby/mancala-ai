"""Temporary-copy baseline and receipt-rebound tests for the supplement verifier."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from ml.alphazero_lite.seed474_verification_supplement import verify

ROOT = Path(__file__).resolve().parents[2]
PUBLICATION = ROOT / "docs/data/seed474-fresh-value-removal"


@pytest.fixture(scope="module")
def staged_publication(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Use a temporary publication view; original evidence stays read-only."""
    staged = tmp_path_factory.mktemp("seed474-publication")
    for path in PUBLICATION.iterdir():
        if path.is_file():
            (staged / path.name).symlink_to(path)
    verify(ROOT, staged)
    return staged


def _private_file(path: Path) -> None:
    if path.is_symlink():
        target = path.resolve()
        path.unlink()
        shutil.copy2(target, path)


def _rebind(staged: Path, changed: Path) -> None:
    receipt_path = staged / "receipt.json"
    _private_file(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    receipt[changed.name] = hashlib.sha256(changed.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _change_npz(staged: Path, name: str, key: str) -> Path:
    path = staged / name
    _private_file(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {item: archive[item].copy() for item in archive.files}
    arrays[key].flat[0] += np.asarray(0.125, dtype=arrays[key].dtype)
    np.savez_compressed(path, **arrays)
    return path


def _change_json(staged: Path, name: str, change) -> Path:
    path = staged / name
    _private_file(path)
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


@pytest.mark.parametrize(
    "case",
    (
        "training_metric",
        "training_J",
        "effect",
        "objective_accounting",
        "prediction",
        "target",
        "weight",
        "gradient",
        "moment",
        "second_moment",
        "counter",
        "state",
        "pre_state",
        "control_reproduction",
        "gate",
    ),
)
def test_receipt_rebound_semantic_tampering_rejected_in_temporary_copy(
    staged_publication: Path, case: str
) -> None:
    if case in {
        "training_metric",
        "training_J",
        "effect",
        "objective_accounting",
        "gate",
    }:

        def mutate(report: dict) -> None:
            if case == "training_metric":
                report["arms"]["P"]["training_policy_ce"]["total"] += 0.125
            elif case == "training_J":
                report["arms"]["P"]["training_J"]["total"] += 0.125
            elif case == "effect":
                report["effects"]["fresh"]["equal_input"]["P_minus_C"]["policy_ce"] += (
                    0.125
                )
            elif case == "objective_accounting":
                report["arms"]["P"]["training_J"]["value_numerator"] += 0.125
            else:
                report["gates"]["training_policy_ce_decreases"] = False

        changed = _change_json(staged_publication, "endpoint-evidence.json", mutate)
    elif case in {"prediction", "target", "weight"}:
        key = {
            "prediction": "prediction",
            "target": "policy_target",
            "weight": "equal_input_weight",
        }[case]
        changed = _change_npz(staged_publication, "P-fresh-predictions.npz", key)
    elif case == "gradient":
        changed = _change_npz(staged_publication, "P-steps.npz", "00_gradient_00")
    elif case == "moment":
        changed = _change_npz(staged_publication, "P-steps.npz", "00_exp_avg_00")
    elif case == "second_moment":
        changed = _change_npz(staged_publication, "P-steps.npz", "00_exp_avg_sq_00")
    elif case == "state":
        changed = _change_npz(staged_publication, "P-steps.npz", "15_post_00")
    elif case == "pre_state":
        changed = _change_npz(staged_publication, "P-steps.npz", "01_pre_00")
    elif case == "counter":
        changed = _change_json(
            staged_publication,
            "P-trajectory.json",
            lambda data: data["steps"][0].__setitem__("optimizer_step", 2),
        )
    else:
        changed = _change_npz(staged_publication, "C-steps.npz", "15_post_00")
        gate_path = staged_publication / "validity-gate.json"
        _private_file(gate_path)
        gate = json.loads(gate_path.read_text())
        gate["C_steps_sha256"] = hashlib.sha256(changed.read_bytes()).hexdigest()
        gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
        _rebind(staged_publication, gate_path)
    _rebind(staged_publication, changed)
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in staged_publication.iterdir()
        if path.is_file()
    }
    with pytest.raises(ValueError):
        verify(ROOT, staged_publication)
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in staged_publication.iterdir()
        if path.is_file()
    }
    assert before == after
    # Restore the staged directory to symlinked original bytes for the next case.
    for path in staged_publication.iterdir():
        if path.is_file() and not path.is_symlink():
            original = PUBLICATION / path.name
            path.unlink()
            path.symlink_to(original)
