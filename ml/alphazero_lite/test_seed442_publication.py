"""Receipt, tamper, and copied-source portability tests for seed442."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ml.alphazero_lite import seed442_kl_capped_adam as seed442
from ml.alphazero_lite.verify_seed442_publication import _verify_receipt


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registered_training_objective_applies_value_coefficient_once() -> None:
    import inspect

    source = inspect.getsource(seed442.run)
    assert source.count("loss = policy_loss + 0.3 * value_loss") == 1
    assert '"value_coefficient": 0.3' in inspect.getsource(seed442.register)


@pytest.mark.parametrize(
    "artifact_name",
    [
        "predictions.npz",
        "step-tensors.npz",
        "optimizer-state-reconstruction.npz",
        "evidence.json",
    ],
)
@pytest.mark.parametrize("tamper", ["altered", "missing"])
def test_publication_receipt_rejects_missing_and_altered_evidence(
    tmp_path: Path, artifact_name: str, tamper: str
) -> None:
    root = tmp_path
    out = root / "docs/data/seed442-kl-capped-adam-screen"
    out.mkdir(parents=True)
    evidence = out / artifact_name
    evidence.write_bytes(b"unchanged")
    registration = out / "registration.json"
    registration.write_bytes(b"registration")
    source = root / "ml/alphazero_lite/verify_seed442_publication.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    receipt = {
        "registration_sha256": _digest(registration),
        "evidence_files_sha256": {str(evidence.relative_to(root)): _digest(evidence)},
        "source_sha256": {str(source.relative_to(root)): _digest(source)},
    }
    (out / "receipt.json").write_text(json.dumps(receipt))
    assert _verify_receipt(root)["evidence_files_sha256"]
    if tamper == "altered":
        evidence.write_bytes(b"tampered")
    else:
        evidence.unlink()
    with pytest.raises(ValueError, match="evidence_file_hash_mismatch"):
        _verify_receipt(root)


def test_copied_source_only_checkout_is_read_only_and_cwd_independent() -> None:
    source_root = Path(__file__).resolve().parents[2]
    scratch = source_root / ".tmp/seed442-portability-check"
    checkout = scratch / "checkout"
    unrelated_cwd = scratch / "elsewhere"
    if scratch.exists():
        shutil.rmtree(scratch)
    checkout.mkdir(parents=True)
    unrelated_cwd.mkdir(parents=True)
    (checkout / "ml").mkdir()
    shutil.copytree(source_root / "ml/alphazero_lite", checkout / "ml/alphazero_lite")
    data_dirs = (
        "seed416-policy-target-softening",
        "seed426-canonical-overlap",
        "seed438-seed437-gradient-correction",
        "seed435-adam-direction-screen",
        "seed440-recorded-update-attribution",
        "seed441-seed440-publication-completion",
        "seed442-kl-capped-adam-screen",
    )
    destination = checkout / "docs/data"
    destination.mkdir(parents=True)
    for name in data_dirs:
        shutil.copytree(source_root / "docs/data" / name, destination / name)
    initializer = (
        "seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    init_dest = (
        destination
        / "seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    init_dest.parent.mkdir(parents=True)
    shutil.copy2(source_root / "docs/data" / initializer, init_dest)
    if any(path.is_symlink() for path in checkout.rglob("*")):
        raise AssertionError("copied_checkout_contains_symlink")
    evidence = checkout / "docs/data/seed442-kl-capped-adam-screen"
    before = {
        str(p.relative_to(checkout)): _digest(p)
        for p in evidence.rglob("*")
        if p.is_file()
    }
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(checkout)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed442_publication",
            "--root",
            str(checkout),
        ],
        cwd=unrelated_cwd,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert json.loads(result.stdout)["status"] == "valid"
    after = {
        str(p.relative_to(checkout)): _digest(p)
        for p in evidence.rglob("*")
        if p.is_file()
    }
    assert before == after
