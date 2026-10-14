"""Real-computation tests for seed479 update arithmetic and verifier binding."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ml.alphazero_lite import seed479_population_ablation as protocol
from ml.alphazero_lite import verify_seed479_population_ablation as verifier
from ml.alphazero_lite.seed479_population_ablation import INPUT_PATHS, OUT, SOURCE_PATHS


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = torch.nn.Linear(3, 4, bias=True)
        self.unused = torch.nn.Parameter(torch.tensor([2.0]))

    def forward(self, x):
        return self.policy(x)


def _gradient(model, x, target, mask, coefficient, denominator):
    logits = model(x).masked_fill(mask == 0, -1e9)
    ce = -(target * F.log_softmax(logits, 1)).sum(1)
    return torch.autograd.grad(
        (ce * coefficient / denominator).sum(),
        tuple(model.parameters()),
        allow_unused=True,
    )


def _vector(model, gradients):
    return np.concatenate(
        [
            (torch.zeros_like(parameter) if gradient is None else gradient)
            .detach()
            .numpy()
            .reshape(-1)
            .astype(np.float64)
            for parameter, gradient in zip(model.parameters(), gradients, strict=True)
        ]
    )


def test_occurrence_multiplicity_own_targets_legal_masks_and_unused_parameter():
    torch.manual_seed(479)
    model = TinyPolicy()
    x = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    mask = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]
    )
    coefficient = torch.tensor([1.0, 2.0, 1.0])
    gradient = _vector(model, _gradient(model, x, target, mask, coefficient, 4.0))
    changed_targets = target.clone()
    changed_targets[1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    changed = _vector(
        model, _gradient(model, x, changed_targets, mask, coefficient, 4.0)
    )
    assert not np.array_equal(gradient, changed)
    assert gradient[-1] == 0.0
    # Same occurrences and targets, different coefficient multiplicity changes the raw gradient.
    without_multiplicity = _vector(
        model, _gradient(model, x, target, mask, torch.ones(3), 3.0)
    )
    assert not np.allclose(gradient, without_multiplicity)


def test_minibatch_and_full_population_use_their_own_masses():
    torch.manual_seed(17)
    model = TinyPolicy()
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    y = torch.eye(4)[:3]
    mask = torch.ones_like(y)
    c = torch.tensor([1.0, 2.0, 3.0])
    batch = _vector(model, _gradient(model, x[:2], y[:2], mask[:2], c[:2], 3.0))
    full = _vector(model, _gradient(model, x, y, mask, c, 6.0))
    assert not np.allclose(batch, full)


def test_fixed_norm_zero_gradient_and_float32_writeback():
    raw = np.array([3.0, 4.0], dtype=np.float64)
    proposal = protocol.normalized_proposal(raw)
    assert proposal.dtype == np.float32
    assert np.linalg.norm(proposal.astype(np.float64)) == pytest.approx(
        protocol.STEP_NORM, abs=2e-9
    )
    assert np.array_equal(
        protocol.normalized_proposal(np.zeros(2)), np.zeros(2, np.float32)
    )
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    before = parameter.detach().clone()
    displacement = protocol.apply_proposal([parameter], proposal)
    assert parameter.dtype == torch.float32
    assert np.array_equal(
        displacement,
        parameter.detach().numpy().astype(np.float64)
        - before.numpy().astype(np.float64),
    )
    assert not np.array_equal(
        displacement, proposal.astype(np.float64)
    )  # rounding is explicitly observable


def test_sequential_state_continuity_is_float32_addition():
    parameter = torch.nn.Parameter(torch.tensor([0.25, -0.5], dtype=torch.float32))
    first = np.array([0.001, -0.002], dtype=np.float32)
    second = np.array([-0.003, 0.004], dtype=np.float32)
    protocol.apply_proposal([parameter], first)
    after_first = parameter.detach().clone()
    protocol.apply_proposal([parameter], second)
    expected = after_first + torch.from_numpy(second)
    assert torch.equal(parameter.detach(), expected)


def test_verifier_readonly_preserves_publication_on_failure(tmp_path: Path):
    # Receipt rejection happens before numerical reconstruction; the entry point is read-only.
    out = tmp_path / protocol.OUT
    out.mkdir(parents=True)
    (out / "receipt.json").write_text(json.dumps({"status": "bad"}))
    path = out / "receipt.json"
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    with pytest.raises(ValueError):
        verifier.verify(tmp_path)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def _publication_copy(tmp_path: Path) -> Path:
    source = protocol.ROOT
    for relative in (*SOURCE_PATHS, *INPUT_PATHS.values()):
        src, dst = source / relative, tmp_path / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for folder in (
        Path("docs/data/seed466-replay-input-bundle"),
        Path("docs/data/seed429-canonical-policy-normalization/training-checkpoints"),
        Path("docs/data/seed474-fresh-value-removal"),
        Path("docs/data/seed475-policy-population-attribution"),
        Path("docs/data/seed478-finite-step-evaluation"),
    ):
        shutil.copytree(source / folder, tmp_path / folder, dirs_exist_ok=True)
    shutil.copytree(source / OUT, tmp_path / OUT)
    return tmp_path


def _rebind_receipt(root: Path) -> None:
    out = root / OUT
    receipt = json.loads((out / "receipt.json").read_text())
    for name in receipt:
        if name != "status":
            receipt[name] = protocol.sha(out / name)
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


@pytest.mark.skipif(
    not (protocol.ROOT / OUT / "receipt.json").is_file(),
    reason="publication not yet executed",
)
def test_receipt_rebound_baseline_passes(tmp_path: Path):
    verifier.verify(_publication_copy(tmp_path))


@pytest.mark.parametrize(
    "mutation",
    [
        "population",
        "denominator",
        "gradient",
        "proposal",
        "state",
        "prediction",
        "weights",
        "effects",
        "gates",
    ],
)
@pytest.mark.skipif(
    not (protocol.ROOT / OUT / "receipt.json").is_file(),
    reason="publication not yet executed",
)
def test_receipt_rebound_entrypoint_rejects_corruption(tmp_path: Path, mutation: str):
    copied = _publication_copy(tmp_path)
    out = copied / OUT
    evidence_path = out / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    archive_path = out / "trajectory-evidence.npz"
    if mutation == "population":
        registration_path = out / "registration.json"
        registration = json.loads(registration_path.read_text())
        registration["population"]["fresh_training_occurrences"] += 1
        registration_path.write_text(
            json.dumps(registration, indent=2, sort_keys=True) + "\n"
        )
    elif mutation == "denominator":
        evidence["steps"][0]["arms"]["F"]["objective_denominator"] += 1
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    elif mutation == "effects":
        evidence["effects"]["F"]["fresh"]["ce"]["exposure"] += 0.1
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    elif mutation == "gates":
        evidence["gates"]["F_full_training_policy_CE_decreases"] = not evidence[
            "gates"
        ]["F_full_training_policy_CE_decreases"]
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    else:
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if mutation == "gradient":
            arrays["F_step_01_gradient"].flat[0] += 1e-3
        elif mutation == "proposal":
            arrays["F_step_01_proposal"].flat[0] += 1e-3
        elif mutation == "state":
            arrays["F_step_01_pre_parameter_00"].flat[0] += 1e-3
        elif mutation == "prediction":
            arrays["F_step_01_fresh_predictions"].flat[0] += 1e-3
        elif mutation == "weights":
            arrays["fresh_equal_input_weights"].flat[0] += 1e-3
        np.savez_compressed(archive_path, **arrays)
    _rebind_receipt(copied)
    with pytest.raises(ValueError):
        verifier.verify(copied)
