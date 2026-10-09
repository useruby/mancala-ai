"""Verifier-path semantic and tamper tests for seed454."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ml.alphazero_lite import seed454_analysis
from ml.alphazero_lite.verify_seed454_full_guard_compliance import OUT, verify

ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def copied_checkout():
    """Build a physical minimal source/evidence checkout for real verifier calls."""
    scratch = Path(tempfile.mkdtemp(prefix="seed454-physical-copy-", dir=ROOT / ".tmp"))
    root = scratch / "checkout"
    root.mkdir()
    freeze = json.loads((ROOT / OUT / "audit-registration.json").read_text())
    paths = set(freeze["input_sha256"]) | set(freeze["source_sha256"])
    paths |= {
        str(OUT / "audit-registration.json"),
        str(OUT / "audit.json"),
        str(OUT / "analysis.md"),
        str(OUT / "receipt.json"),
        str(OUT / "supersession-receipt.json"),
    }
    seed453_receipt = json.loads(
        (ROOT / "docs/data/seed453-fresh-policy-projection/receipt.json").read_text()
    )
    paths |= set(seed453_receipt["files_sha256"])
    paths |= {
        str(OUT / "source-snapshots" / Path(name).name)
        for name in freeze["source_sha256"]
    }
    paths |= {
        "docs/data/seed453-fresh-policy-projection/receipt.json",
        "docs/data/seed442-kl-capped-adam-screen/guard-cohort.json",
        "docs/data/seed416-policy-target-softening/registration-v3.json",
    }
    for relative in sorted(paths):
        source = ROOT / relative
        if not source.is_file():
            continue
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    yield root
    shutil.rmtree(scratch)


@pytest.fixture(autouse=True)
def restore_manifests(copied_checkout: Path):
    """Keep the shared test checkout pristine between semantic tamper cases."""
    paths = (
        copied_checkout / OUT / "audit-registration.json",
        copied_checkout / OUT / "receipt.json",
        copied_checkout / "docs/data/seed453-fresh-policy-projection/receipt.json",
    )
    saved = [path.read_bytes() for path in paths]
    yield
    for path, content in zip(paths, saved, strict=True):
        path.write_bytes(content)


def test_nonmonotone_feasibility_selects_largest_tested_scale() -> None:
    trials = [
        {
            "scale": scale,
            "batch_kl": 0.0,
            "batch_value_movement": 0.0,
            "executed_guard_kl": 0.0 if feasible else 1.0,
            "executed_guard_value_movement": 0.0,
            "full_guard_kl": 0.0 if feasible else 1.0,
            "full_guard_value_movement": 0.0,
            "realized_fresh_dot": 0.0,
        }
        for scale, feasible in ((1.0, False), (0.5, True), (0.25, False), (0.125, True))
    ]
    assert seed454_analysis.largest_feasible(trials, "C", requested=True) == 0.5


def test_compliant_accepted_state_can_have_selection_mismatch() -> None:
    accepted = 0.25
    feasible_scales = [0.5, 0.25]
    assert accepted in feasible_scales
    assert max(feasible_scales) != accepted
    assert seed454_analysis.classify(0, 1) == "full_guard_selection_mismatch"


def _rebind(root: Path, changed_paths: list[str]) -> None:
    seed453_receipt_path = (
        root / "docs/data/seed453-fresh-policy-projection/receipt.json"
    )
    if any(
        path.startswith("docs/data/seed453-fresh-policy-projection/")
        for path in changed_paths
    ):
        seed453_receipt = json.loads(seed453_receipt_path.read_text())
        for relative in changed_paths:
            if relative in seed453_receipt["files_sha256"]:
                seed453_receipt["files_sha256"][relative] = _sha(root / relative)
        seed453_receipt_path.write_text(
            json.dumps(seed453_receipt, indent=2, sort_keys=True) + "\n"
        )
        receipt_relative = "docs/data/seed453-fresh-policy-projection/receipt.json"
        if receipt_relative not in changed_paths:
            changed_paths.append(receipt_relative)
    freeze_path = root / OUT / "audit-registration.json"
    freeze = json.loads(freeze_path.read_text())
    for relative in changed_paths:
        freeze["input_sha256"][relative] = _sha(root / relative)
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    receipt_path = root / OUT / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for relative in receipt["files_sha256"]:
        receipt["files_sha256"][relative] = _sha(root / relative)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _assert_semantic_rejection(root: Path, changed_paths: list[str]) -> None:
    _rebind(root, changed_paths)
    with pytest.raises(ValueError):
        verify(root)


def test_real_verifier_accepts_physical_copy(copied_checkout: Path) -> None:
    result = verify(copied_checkout)
    assert result["status"] == "valid"
    assert result["trials"] == 256


def test_real_cli_is_portable_and_read_only_with_absolute_derivatives_blocked(
    copied_checkout: Path,
) -> None:
    """Run the copied module CLI outside the checkout and deny historical absolutes."""
    registration_path = (
        copied_checkout
        / "docs/data/seed416-policy-target-softening/registration-v3.json"
    )
    registration = json.loads(registration_path.read_text())
    blocked = sorted(
        {row["A"]["derivative"] for row in registration["derivatives"].values()}
    )
    sitecustomize = "\n".join(
        (
            "import sys",
            f"blocked = set({blocked!r})",
            "def reject(event, args):",
            "    if event == 'open' and args and args[0] in blocked:",
            "        raise PermissionError('historical_absolute_derivative_forbidden')",
            "sys.addaudithook(reject)",
            "",
        )
    )
    (copied_checkout / "sitecustomize.py").write_text(sitecustomize)
    (copied_checkout / ".tmp").mkdir(exist_ok=True)
    publication = copied_checkout / OUT
    before = {
        path.relative_to(publication): (_sha(path), path.stat().st_mtime_ns)
        for path in publication.rglob("*")
        if path.is_file()
    }
    env = dict(os.environ)
    env["PYTHONPATH"] = str(copied_checkout)
    env["TMPDIR"] = str(copied_checkout / ".tmp")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed454_full_guard_compliance",
            "--root",
            str(copied_checkout),
        ],
        cwd=Path.home(),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"status": "valid"' in completed.stdout
    after = {
        path.relative_to(publication): (_sha(path), path.stat().st_mtime_ns)
        for path in publication.rglob("*")
        if path.is_file()
    }
    assert before == after


def test_real_verifier_rejects_altered_guard_membership(copied_checkout: Path) -> None:
    relative = "docs/data/seed453-fresh-policy-projection/fresh-guard.json"
    path = copied_checkout / relative
    original = path.read_bytes()
    try:
        guard = json.loads(original)
        guard[0]["exact_input_identity"] = "0" * len(guard[0]["exact_input_identity"])
        path.write_text(json.dumps(guard))
        _assert_semantic_rejection(copied_checkout, [relative])
    finally:
        path.write_bytes(original)


def test_real_verifier_rejects_altered_trial_tensor_with_rebound_hashes(
    copied_checkout: Path,
) -> None:
    relative = "docs/data/seed453-fresh-policy-projection/step-tensors-C.npz"
    path = copied_checkout / relative
    original = path.read_bytes()
    try:
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in source.files}
        key = "C_00_00_trial_1"
        arrays[key].flat[0] += np.float32(1e-4)
        np.savez_compressed(path, **arrays)
        _assert_semantic_rejection(copied_checkout, [relative])
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("field", ["full_guard_kl", "classification"])
def test_real_verifier_rejects_tampered_audit_semantics_with_rebound_receipt(
    copied_checkout: Path, field: str
) -> None:
    relative = str(OUT / "audit.json")
    path = copied_checkout / relative
    original = path.read_bytes()
    try:
        report = json.loads(original)
        if field == "full_guard_kl":
            report["trial_measurements"][0][field] += 0.01
        else:
            report[field] = "full_guard_constraint_violation"
        path.write_text(json.dumps(report))
        _assert_semantic_rejection(copied_checkout, [])
    finally:
        path.write_bytes(original)


def test_real_verifier_rejects_selected_scale_change_with_rebound_receipt(
    copied_checkout: Path,
) -> None:
    relative = str(OUT / "audit.json")
    path = copied_checkout / relative
    original = path.read_bytes()
    try:
        report = json.loads(original)
        report["selected_scale_comparisons"][0]["requested_rule_selected_scale"] = 0.5
        path.write_text(json.dumps(report))
        _assert_semantic_rejection(copied_checkout, [])
    finally:
        path.write_bytes(original)
