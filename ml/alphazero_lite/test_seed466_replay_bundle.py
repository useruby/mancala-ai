"""Entry-point and immutability tests for seed466 replay bundles."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from ml.alphazero_lite.seed466_replay_bundle import (
    EXPECTED_MANIFEST_ID,
    ROOT,
    read_bundle,
)
from ml.alphazero_lite.verify_seed466_replay_bundle import verify

BUNDLE = ROOT / "docs/data/seed466-replay-input-bundle"
SCRATCH = ROOT / ".tmp/seed466-tests"


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _copy_bundle(name: str) -> Path:
    destination = SCRATCH / name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(BUNDLE, destination)
    return destination


def _refresh_npz_records(path: Path, manifest: dict[str, object], name: str) -> None:
    file_name = f"{name}.npz"
    raw = (path / file_name).read_bytes()
    manifest["files"][file_name] = _hash(raw)  # type: ignore[index]
    with np.load(path / file_name, allow_pickle=False) as data:
        manifest["arrays"][name] = {  # type: ignore[index]
            key: {
                "dtype": str(data[key].dtype),
                "shape": list(data[key].shape),
                "sha256": _hash(data[key].tobytes(order="C")),
            }
            for key in data.files
        }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def test_reader_and_verifier_reconcile_frozen_bundle_read_only() -> None:
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in BUNDLE.iterdir()
        if path.is_file()
    }
    loaded = read_bundle(BUNDLE)
    verify(BUNDLE, ROOT)
    training = loaded["training"]
    evaluation = loaded["evaluation"]
    assert len(training["train_positions"]) == 134502
    assert (
        int(
            training["fresh_source_mask"][
                training["expanded_replay_indices"][training["train_positions"]]
            ].sum()
        )
        == 60015
    )
    assert len(training["validation_positions"]) == 14946
    assert len(evaluation["value_targets"]) == 2607
    assert len(set(evaluation["input_identity"].tolist())) == 1242
    assert np.count_nonzero(evaluation["source_ids"] == 0) == 859
    assert (
        len(set(evaluation["input_identity"][evaluation["source_ids"] == 0].tolist()))
        == 851
    )
    for path, (content, mtime) in before.items():
        assert path.read_bytes() == content
        assert path.stat().st_mtime_ns == mtime


def test_reader_rejects_missing_member_and_wrong_manifest_identity() -> None:
    copied = _copy_bundle("missing")
    (copied / "evaluation.npz").unlink()
    with pytest.raises(FileNotFoundError):
        read_bundle(copied)
    with pytest.raises(FileNotFoundError):
        verify(copied, ROOT)
    with pytest.raises(ValueError, match="manifest_identity_mismatch"):
        read_bundle(BUNDLE, "different-commitment")


def test_independent_verifier_rejects_array_with_rebound_routine_hashes() -> None:
    copied = _copy_bundle("rebound-array")
    with np.load(copied / "training.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["policy_targets"][0, 0] += np.float32(0.125)
    np.savez(copied / "training.npz", **arrays)
    manifest = json.loads((copied / "manifest.json").read_text())
    _refresh_npz_records(copied, manifest, "training")
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in copied.iterdir()
        if path.is_file()
    }
    with pytest.raises(
        ValueError, match="independent_reconstruction_mismatch:policy_targets"
    ):
        verify(copied, ROOT)
    for path, (content, mtime) in before.items():
        assert path.read_bytes() == content
        assert path.stat().st_mtime_ns == mtime


def test_verifier_rejects_source_commitment_change_without_writing() -> None:
    copied = _copy_bundle("changed-source-binding")
    manifest = json.loads((copied / "manifest.json").read_text())
    key = next(iter(manifest["source_closure"]))
    manifest["source_closure"][key] = "0" * 64
    (copied / "manifest.json").write_text(json.dumps(manifest))
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in copied.iterdir()
        if path.is_file()
    }
    with pytest.raises(ValueError, match="source_identity_mismatch"):
        verify(copied, ROOT)
    for path, (content, mtime) in before.items():
        assert path.read_bytes() == content
        assert path.stat().st_mtime_ns == mtime


def test_independent_reconstruction_rejects_rebound_recipe_mutations() -> None:
    mutations = (
        ("training", "policy_coefficients", "coefficient"),
        ("training", "effective_value_mode_ids", "mode_precedence"),
        ("training", "multiplicities", "multiplicity"),
        ("training", "train_positions", "split"),
        ("training", "first16_minibatch_positions", "permutation"),
        ("training", "expanded_replay_indices", "expanded_indices"),
        ("evaluation", "policy_targets", "duplicate_occurrence_target"),
    )
    for name, array_name, label in mutations:
        copied = _copy_bundle(label)
        with np.load(copied / f"{name}.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if label == "duplicate_occurrence_target":
            identity = arrays["input_identity"]
            duplicate = next(
                i
                for i, value in enumerate(identity)
                if np.count_nonzero(identity == value) > 1
            )
            arrays[array_name][duplicate, 0] += np.float32(0.125)
        elif label == "split":
            arrays[array_name][0], arrays[array_name][1] = (
                arrays[array_name][1],
                arrays[array_name][0],
            )
        elif label == "permutation":
            arrays[array_name][0], arrays[array_name][1] = (
                arrays[array_name][1],
                arrays[array_name][0],
            )
        elif label == "expanded_indices":
            arrays[array_name][0] = (arrays[array_name][0] + 1) % len(arrays["inputs"])
        elif label == "coefficient":
            arrays[array_name][0] = np.float32(0.5)
        elif label == "mode_precedence":
            arrays[array_name][0] = np.uint8(1 - arrays[array_name][0])
        else:
            arrays[array_name][0] += np.uint8(1)
        np.savez(copied / f"{name}.npz", **arrays)
        manifest = json.loads((copied / "manifest.json").read_text())
        _refresh_npz_records(copied, manifest, name)
        with pytest.raises(
            ValueError, match="independent_(reconstruction|evaluation)_mismatch"
        ):
            verify(copied, ROOT)


def test_verifier_rejects_rebound_incorrect_counts_and_masses() -> None:
    count_copy = _copy_bundle("incorrect-count")
    count_manifest = json.loads((count_copy / "manifest.json").read_text())
    count_manifest["counts"]["expanded_training"] += 1
    (count_copy / "manifest.json").write_text(json.dumps(count_manifest))
    with pytest.raises(ValueError, match="training_count_reconciliation"):
        verify(count_copy, ROOT)

    identity_copy = _copy_bundle("incorrect-evaluation-identity-count")
    identity_manifest = json.loads((identity_copy / "manifest.json").read_text())
    identity_manifest["counts"]["evaluation_fresh_identities"] += 1
    (identity_copy / "manifest.json").write_text(json.dumps(identity_manifest))
    with pytest.raises(ValueError, match="fresh_evaluation_identity_reconciliation"):
        verify(identity_copy, ROOT)

    mass_copy = _copy_bundle("incorrect-mass")
    mass_manifest = json.loads((mass_copy / "manifest.json").read_text())
    mass_manifest["policy_masses"]["training_fresh"] += 1.0
    (mass_copy / "manifest.json").write_text(json.dumps(mass_manifest))
    with pytest.raises(ValueError, match="policy_mass_reconciliation"):
        verify(mass_copy, ROOT)


def test_manifest_records_occurrence_level_duplicates_and_recipe() -> None:
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    assert manifest["identity"] == EXPECTED_MANIFEST_ID
    assert manifest["recipe"]["sources"] == [
        "fresh",
        "generic_bootstrap",
        "random_teacher",
        "opening_disagreement",
        "stability",
    ]
    assert manifest["recipe"]["weights"] == [1, 4, 1, 8, 4]
    assert manifest["counts"]["expanded_occurrences_including_validation"] == 149448
    assert manifest["counts"]["expanded_training"] == 134502
    assert manifest["counts"]["fresh_expanded"] == 60015
    assert (
        manifest["counts"]["evaluation_pooled_occurrences"]
        > manifest["counts"]["evaluation_pooled_identities"]
    )
    assert (
        manifest["counts"]["evaluation_fresh_occurrences"]
        > manifest["counts"]["evaluation_fresh_identities"]
    )
