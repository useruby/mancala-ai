"""Tamper tests for the read-only seed441 receipt hash verifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.alphazero_lite.verify_seed441_receipt_hashes import verify


def _write(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _fixture(root: Path) -> None:
    d441 = root / "docs/data/seed441-seed440-publication-completion"
    d440 = root / "docs/data/seed440-recorded-update-attribution"
    d435 = root / "docs/data/seed435-adam-direction-screen"
    d441.mkdir(parents=True)
    d440.mkdir(parents=True)
    d435.mkdir(parents=True)
    historical = {}
    for name, field in (
        ("protocol.json", "protocol_sha256"),
        ("evidence-bindings.json", "evidence_bindings_sha256"),
        ("step-ledger.json", "step_ledger_sha256"),
        ("reconstructed-path-hashes.npz", "path_hash_archive_sha256"),
    ):
        historical[field] = _write(d440 / name, name.encode())
    binding_files = {}
    for name in ("initializer.npz", "A-final.npz", "B-final.npz"):
        relative = f"docs/data/seed435-adam-direction-screen/{name}"
        binding_files[name] = {
            "path": relative,
            "sha256": _write(root / relative, name.encode()),
        }
    for name in ("u-gradient.npz", "membership.jsonl.gz"):
        relative = f"docs/data/seed440-recorded-update-attribution/{name}"
        binding_files[name] = {
            "path": relative,
            "sha256": _write(root / relative, name.encode()),
        }
    bindings_path = d440 / "evidence-bindings.json"
    bindings_path.write_text(
        json.dumps(
            {
                "seed435": {
                    key: val
                    for key, val in binding_files.items()
                    if key.endswith(".npz")
                    and key in ("initializer.npz", "A-final.npz", "B-final.npz")
                },
                "seed439_U_gradient_archive": binding_files["u-gradient.npz"],
                "authoritative_membership": binding_files["membership.jsonl.gz"],
            }
        )
    )
    historical["evidence_bindings_sha256"] = hashlib.sha256(
        bindings_path.read_bytes()
    ).hexdigest()
    supplemental_files = {}
    for name in (
        "protocol.json",
        "completion.json",
        "predictions.npz",
        "weighting-gradients.npz",
        "row-identities.json",
        "parameter-layout.json",
    ):
        supplemental_files[name] = _write(d441 / name, name.encode())
    source_hashes = {}
    for relative in (
        "ml/alphazero_lite/seed441_seed440_publication.py",
        "ml/alphazero_lite/verify_seed441_seed440_publication.py",
        "ml/alphazero_lite/test_seed441_seed440_publication.py",
    ):
        source_hashes[relative] = _write(root / relative, relative.encode())
    dependencies = {}
    for relative in (
        "docs/data/seed416-policy-target-softening/registration-v3.json",
        "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz",
        "docs/data/seed426-canonical-overlap/sources/fresh.jsonl.gz",
    ):
        dependencies[relative] = _write(root / relative, relative.encode())
    (d441 / "protocol.json").write_text(
        json.dumps({"source_sha256": source_hashes, "dependency_sha256": dependencies})
    )
    # Rebind receipt hashes after the protocol bytes are finalized.
    supplemental_files["protocol.json"] = hashlib.sha256(
        (d441 / "protocol.json").read_bytes()
    ).hexdigest()
    receipt = {
        "historical_seed440": historical,
        "supplemental_files": supplemental_files,
        "source_sha256": source_hashes,
    }
    (d441 / "receipt.json").write_text(json.dumps(receipt))


def test_seed441_hash_verifier_accepts_complete_receipt(tmp_path: Path) -> None:
    _fixture(tmp_path)
    assert verify(tmp_path)["status"] == "valid"


@pytest.mark.parametrize("tamper", ["altered", "missing"])
def test_seed441_hash_verifier_rejects_altered_or_missing_archive(
    tmp_path: Path, tamper: str
) -> None:
    _fixture(tmp_path)
    archive = (
        tmp_path
        / "docs/data/seed440-recorded-update-attribution/reconstructed-path-hashes.npz"
    )
    if tamper == "altered":
        archive.write_bytes(b"tampered")
    else:
        archive.unlink()
    with pytest.raises(
        (ValueError, FileNotFoundError),
        match="seed441_receipt_hash_mismatch|No such file",
    ):
        verify(tmp_path)
