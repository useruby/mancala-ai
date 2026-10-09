"""Integration tamper tests for the published seed455 semantic verifier."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from ml.alphazero_lite import verify_seed455_fresh_projection_attribution as verifier

ROOT = Path(__file__).resolve().parents[2]
PUBLICATION = Path("docs/data/seed455-fresh-projection-attribution")
IDENTITIES = {
    "freeze-v7.json": "ab4651ca6a9aa12356356c17e0d2913fad7fb8168fc1420eb328b8544fc48c33",
    "results.json": "9365838ff5386f9a696795846e25f2bcaff4ac5c21d9990ce06e3e3925f54c52",
    "receipt.json": "296f772fc8a5e6e4de7d24501da567691023dc13c8c8cb7572ff942dba9f2566",
}


def _copy_source_and_evidence(destination: Path) -> None:
    publication = ROOT / PUBLICATION
    frozen = json.loads((publication / "freeze-v7.json").read_text())
    (destination / ".tmp").mkdir(parents=True)
    shutil.copytree(ROOT / "ml", destination / "ml")
    required = set(frozen["inputs"]) | set(frozen["sources"])
    required.update(
        path.relative_to(ROOT).as_posix()
        for path in publication.rglob("*")
        if path.is_file()
    )
    for relative in sorted(required):
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _historical_inventory() -> dict[str, tuple[str, int]]:
    publication = ROOT / PUBLICATION
    frozen = json.loads((publication / "freeze-v7.json").read_text())
    paths = set(frozen["inputs"]) | set(frozen["sources"])
    paths.update(
        path.relative_to(ROOT).as_posix()
        for path in publication.rglob("*")
        if path.is_file()
    )
    return {
        relative: (
            verifier.sha(ROOT / relative),
            (ROOT / relative).stat().st_mtime_ns,
        )
        for relative in sorted(paths)
    }


def _rebind_receipt(root: Path) -> None:
    out = root / PUBLICATION
    receipt_path = out / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for relative in list(receipt["files_sha256"]):
        path = root / relative
        receipt["files_sha256"][relative] = verifier.sha(path)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _tamper(root: Path, category: str) -> str:
    out = root / PUBLICATION
    results_path = out / "results.json"
    results = json.loads(results_path.read_text())
    if category in {
        "cohort identity",
        "equal-input accounting",
        "archived policy targets",
    }:
        evidence_path = out / "predictions-targets.npz"
        with np.load(evidence_path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        key = {
            "cohort identity": "cohort_fresh_unseen_equal_input_identities",
            "equal-input accounting": "cohort_fresh_unseen_equal_input_compact_rows",
            "archived policy targets": "ordered_targets",
        }[category]
        if category == "cohort identity":
            arrays[key][0] = "tampered-identity"
        elif category == "archived policy targets":
            arrays[key].flat[0] += 1
        if category != "equal-input accounting":
            np.savez_compressed(evidence_path, **arrays)
        if category == "equal-input accounting":
            # Substitute the exposure-weighted reconstruction into the
            # equal-input record while preserving its cohort labels.
            exposure = results["cohorts"]["B"]["fresh_unseen_exposures"]["ledger"]
            equal = results["cohorts"]["B"]["fresh_unseen_equal_input"]["ledger"]
            for target, source in zip(equal, exposure, strict=True):
                for field in ("pre_ce", "post_ce", "d", "s", "r", "group_s"):
                    target[field] = source[field]
            results_path.write_text(
                json.dumps(results, indent=2, sort_keys=True) + "\n"
            )
            return "pre_ce:"
        return (
            "cohort_identity"
            if category == "cohort identity"
            else "targets_reconstructed"
        )
    if category == "archived parameter-state continuity":
        relative = Path("docs/data/seed453-fresh-policy-projection/step-tensors-B.npz")
        frozen_path = out / "freeze-v7.json"
        frozen = json.loads(frozen_path.read_text())
        path = root / relative
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays["B_00_00_pre"].flat[0] += 1
        np.savez_compressed(path, **arrays)
        frozen["inputs"][relative.as_posix()] = verifier.sha(path)
        frozen_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
        return "trajectory_link:"
    if category == "saved gradients":
        evidence_path = out / "gradients.npz"
        with np.load(evidence_path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays[next(iter(arrays))].flat[0] += 1
        np.savez_compressed(evidence_path, **arrays)
        return "gradient:"
    if category == "saved prediction logits":
        evidence_path = out / "predictions-targets.npz"
        with np.load(evidence_path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        key = next(
            key for key in arrays if key.startswith("B_") and key.endswith("_00")
        )
        arrays[key].flat[0] += 1
        np.savez_compressed(evidence_path, **arrays)
        return "prediction:"
    if category == "parameter-group slope contributions":
        results["cohorts"]["B"]["fresh_unseen_equal_input"]["ledger"][0]["group_s"][
            "shared_trunk"
        ] += 1
        results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        return "group_contribution:"
    if category == "per-step/block totals":
        results["blocks"]["B"]["fresh_unseen_equal_input"]["all_16"]["D"] += 1
        results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        return "block:"
    if category == "classification":
        results["classification"] = "fresh_unseen_remainder_dominant"
        results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        return "classification"
    raise AssertionError(category)


@pytest.mark.parametrize(
    "category",
    [
        "cohort identity",
        "equal-input accounting",
        "archived policy targets",
        "archived parameter-state continuity",
        "saved gradients",
        "saved prediction logits",
        "parameter-group slope contributions",
        "per-step/block totals",
        "classification",
    ],
)
def test_full_verifier_rejects_tampered_publication(
    tmp_path: Path, category: str
) -> None:
    original_inventory = _historical_inventory()
    root = tmp_path / "copy"
    root.mkdir()
    _copy_source_and_evidence(root)
    expected = _tamper(root, category)
    _rebind_receipt(root)
    with pytest.raises(ValueError, match=expected):
        verifier.verify(root)
    assert _historical_inventory() == original_inventory


def test_unchanged_physical_evidence_copy_is_valid_and_pinned(tmp_path: Path) -> None:
    original_inventory = _historical_inventory()
    for name, digest in IDENTITIES.items():
        assert verifier.sha(ROOT / PUBLICATION / name) == digest
    root = tmp_path / "copy"
    root.mkdir()
    _copy_source_and_evidence(root)
    result = verifier.verify(root)
    assert result == {
        "status": "valid",
        "classification": "fresh_unseen_direction_dominant",
        "row_count": 96,
    }
    assert _historical_inventory() == original_inventory
