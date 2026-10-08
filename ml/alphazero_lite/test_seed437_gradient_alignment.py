"""Synthetic checks for seed437 weighting and gradient geometry."""

import numpy as np
import pytest
import torch
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from ml.alphazero_lite import train
from ml.alphazero_lite.run_seed437_training_unseen_gradient_alignment import (
    flat_grad,
    parameter_group_indexes,
)
from ml.alphazero_lite.seed437_gradient_alignment import (
    equal_input_weights,
    geometry,
    normalized_exposure_weights,
    sum_vectors,
)
from ml.alphazero_lite.verify_seed437_training_unseen_gradient_alignment import (
    verify,
)

ROOT = Path(__file__).resolve().parents[2]
PUBLICATION = ROOT / "docs/data/seed437-training-unseen-gradient-alignment"


def test_equal_input_weighting_preserves_within_identity_exposure_mean() -> None:
    weights = equal_input_weights(["a", "a", "b"])
    assert weights.sum() == pytest.approx(1.0)
    assert weights.tolist() == pytest.approx([0.25, 0.25, 0.5])
    assert normalized_exposure_weights(3).tolist() == pytest.approx(
        [1 / 3, 1 / 3, 1 / 3]
    )
    expanded = ["a", "a", "a", "b"]
    multiplicity_weights = equal_input_weights(expanded)
    assert multiplicity_weights.tolist() == pytest.approx([1 / 6, 1 / 6, 1 / 6, 1 / 2])


def test_geometry_and_zero_norm_handling() -> None:
    result = geometry(np.array([3.0, 4.0]), np.array([-3.0, -4.0]))
    assert result["dot"] == -25.0
    assert result["cosine"] == -1.0
    assert result["unit_descent_ce_change"] == 5.0
    zero = geometry(np.zeros(2), np.ones(2))
    assert zero["cosine"] is None
    assert zero["unit_descent_ce_change"] is None


def test_partition_sum_and_reject_altered_shape() -> None:
    pieces = {"left": np.array([1.0, 2.0]), "right": np.array([3.0, 4.0])}
    assert sum_vectors(pieces, 2).tolist() == [4.0, 6.0]
    with pytest.raises(ValueError, match="gradient_component_invalid"):
        sum_vectors({"bad": np.array([1.0])}, 2)


def test_chunk_size_invariance(monkeypatch: pytest.MonkeyPatch) -> None:
    class TinyPolicyValue(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layer = torch.nn.Linear(2, 4)
            self.policy_head = torch.nn.Linear(4, 6)
            self.value_head = torch.nn.Linear(4, 1)

        def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = torch.tanh(self.input_layer(value))
            return self.policy_head(hidden), self.value_head(hidden)

    monkeypatch.setattr(
        train,
        "legal_mask_matrix_for_encoded_states",
        lambda states: np.ones((len(states), 6), dtype=np.float32),
    )
    model = TinyPolicyValue()
    x = np.arange(22, dtype=np.float32).reshape(11, 2) / 10
    p = np.full((11, 6), 1 / 6, dtype=np.float32)
    v = np.linspace(-0.5, 0.5, 11, dtype=np.float32).reshape(-1, 1)
    ids = np.arange(11)
    weights = np.full(11, 1 / 11, dtype=np.float64)
    small = flat_grad(model, ids, weights, x, p, v, "policy", 2)
    large = flat_grad(model, ids, weights, x, p, v, "policy", 7)
    assert np.allclose(small, large, atol=1e-7, rtol=1e-6)
    groups = parameter_group_indexes(model, tuple(model.parameters()))
    assert sum(map(len, groups.values())) == sum(p.numel() for p in model.parameters())


def _relocated_root(destination: Path) -> Path:
    destination.mkdir()
    publication = destination / "docs/data/seed437-training-unseen-gradient-alignment"
    shutil.copytree(
        PUBLICATION,
        publication,
    )
    for relative in (
        "docs/data/seed416-policy-target-softening",
        "docs/data/seed429-canonical-policy-normalization",
        "docs/data/seed435-adam-direction-screen",
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(ROOT / relative, target_is_directory=True)
    source = ROOT / "docs/data/seed426-canonical-overlap"
    target = destination / "docs/data/seed426-canonical-overlap"
    target.mkdir(parents=True)
    for entry in source.iterdir():
        item = target / entry.name
        if entry.name == "seed427-validation-membership.jsonl.gz":
            shutil.copy2(entry, item)
        else:
            item.symlink_to(entry, target_is_directory=entry.is_dir())
    ml_target = destination / "ml"
    ml_target.symlink_to(ROOT / "ml", target_is_directory=True)
    manifest_path = publication / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for source in manifest["execution_sources"]:
        manifest["execution_sources"][source] = hashlib.sha256(
            (ROOT / source).read_bytes()
        ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result_path = publication / "results.json"
    result = json.loads(result_path.read_text())
    result["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return destination


def _evidence_hashes(root: Path) -> dict[str, str]:
    manifest = json.loads(
        (
            root / "docs/data/seed437-training-unseen-gradient-alignment/manifest.json"
        ).read_text()
    )
    paths = [
        manifest["registration"]["path"],
        manifest["split"]["path"],
        manifest["membership"]["path"],
        manifest["reconstruction_evidence"]["row_accounting_path"],
        *manifest["execution_sources"],
    ]
    paths.extend(source["snapshot"] for source in manifest["sources"])
    paths.extend(value["path"] for value in manifest["checkpoints"].values())
    paths.append(
        "docs/data/seed437-training-unseen-gradient-alignment/"
        + json.loads(
            (
                root
                / "docs/data/seed437-training-unseen-gradient-alignment/results.json"
            ).read_text()
        )["vector_archive"]["path"]
    )
    return {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths
    }


def test_relocated_cli_and_tamper_rejection(tmp_path: Path) -> None:
    relocated = _relocated_root(tmp_path / "relocated")
    before = _evidence_hashes(relocated)
    command = [
        sys.executable,
        "-m",
        "ml.alphazero_lite.verify_seed437_training_unseen_gradient_alignment",
        "--root",
        str(relocated),
    ]
    completed = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout)["status"] == "valid"
    assert _evidence_hashes(relocated) == before

    result_path = (
        relocated / "docs/data/seed437-training-unseen-gradient-alignment/results.json"
    )
    original = result_path.read_bytes()
    with pytest.raises(ValueError, match="classification_mismatch"):
        altered = original.replace(
            b'"classification": "no_persistent_training_objective_opposition"',
            b'"classification": "persistent_training_objective_opposition"',
            1,
        )
        result_path.write_bytes(altered)
        verify(relocated)
    result_path.write_bytes(original)

    archive_path = result_path.parent / "gradient-vectors.npz"
    archive_original = archive_path.read_bytes()
    with pytest.raises(ValueError, match="vector_digest_mismatch"):
        result_data = json.loads(original)
        arrays = {
            key: value.copy()
            for key, value in np.load(archive_path, allow_pickle=False).items()
        }
        checkpoint = next(iter(result_data["checkpoints"].values()))
        evidence = next(iter(checkpoint["vector_evidence"].values()))
        arrays[evidence["archive_key"]][0] += 0.123456789
        np.savez_compressed(archive_path, **arrays)
        result_data["vector_archive"]["sha256"] = hashlib.sha256(
            archive_path.read_bytes()
        ).hexdigest()
        result_path.write_text(json.dumps(result_data, indent=2, sort_keys=True) + "\n")
        verify(relocated)
    result_path.write_bytes(original)
    archive_path.write_bytes(archive_original)

    membership_path = (
        relocated
        / "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz"
    )
    membership = membership_path.read_bytes()
    with pytest.raises(ValueError, match="membership_hash_mismatch"):
        membership_path.write_bytes(bytes([membership[0] ^ 1]) + membership[1:])
        verify(relocated)
    membership_path.write_bytes(membership)
