"""Verifier-path semantic and tamper tests for seed454."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from ml.alphazero_lite import seed454_analysis
from ml.alphazero_lite.verify_seed454_full_guard_compliance import OUT, verify

ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def copied_checkout(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a physical minimal source/evidence checkout for real verifier calls."""
    root = tmp_path_factory.mktemp("seed454-copy") / "checkout"
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
    return root


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
        {"scale": scale, "feasible": feasible}
        for scale, feasible in ((1.0, False), (0.5, True), (0.25, False), (0.125, True))
    ]
    assert seed454_analysis.choose_largest_feasible(trials) == 0.5


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
            report[field] = "archived_trajectory_matches_full_guard_rule"
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
