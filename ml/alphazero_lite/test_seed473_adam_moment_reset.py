"""Real-computation and semantic tests for seed473."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import hashlib
import json
import os
from pathlib import Path

from ml.alphazero_lite import train
from ml.alphazero_lite.seed473_adam_moment_reset import (
    compare_control,
    require_control_gate,
)
from ml.alphazero_lite.verify_seed473_adam_moment_reset import verify

ROOT = Path(__file__).resolve().parents[2]
PUBLICATION = ROOT / "docs/data/seed473-adam-moment-reset"


def _update(
    model: torch.nn.Module, optimizer: torch.optim.Adam, x: torch.Tensor, reset: bool
) -> None:
    if reset:
        optimizer.state.clear()
    optimizer.zero_grad(set_to_none=True)
    logits, value = model(x)
    loss = logits.square().mean() + 0.3 * value.square().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def test_real_carried_and_reset_updates_preserve_parameters_and_reset_state() -> None:
    torch.manual_seed(473)
    carried = train.PolicyValueNet((96, 3), "residual_v3", 27)
    reset = train.PolicyValueNet((96, 3), "residual_v3", 27)
    reset.load_state_dict(carried.state_dict())
    oc = torch.optim.Adam(carried.parameters(), lr=0.0000625)
    or_ = torch.optim.Adam(reset.parameters(), lr=0.0000625)
    batch = torch.randn(512, 27)
    initial = [p.detach().clone() for p in reset.parameters()]
    unused = torch.nn.Parameter(torch.ones(2))
    for index in range(3):
        _update(carried, oc, batch, False)
        _update(reset, or_, batch, True)
        assert all(or_.state[p]["step"].item() == 1 for p in reset.parameters())
        assert all(
            oc.state[p]["step"].item() == index + 1 for p in carried.parameters()
        )
        assert unused.grad is None
    assert any(
        not torch.equal(a, b) for a, b in zip(initial, reset.parameters(), strict=True)
    )
    assert all(torch.isfinite(p).all() for p in reset.parameters())


def test_control_comparison_accepts_match_and_rejects_changed_state(tmp_path) -> None:
    reference = {}
    candidate = {}
    for step in range(16):
        for index in range(2):
            value = np.asarray([step, index], dtype=np.float32)
            reference[f"{step:02d}_post_{index:02d}"] = value
            candidate[f"{step:02d}_post_{index:02d}"] = value.copy()
    ref_path, candidate_path = tmp_path / "reference.npz", tmp_path / "candidate.npz"
    np.savez_compressed(ref_path, **reference)
    np.savez_compressed(candidate_path, **candidate)
    compare_control(ref_path, candidate_path)
    candidate["07_post_01"][0] += 1
    np.savez_compressed(candidate_path, **candidate)
    with pytest.raises(ValueError, match="control_reproduction_mismatch:8:1"):
        compare_control(ref_path, candidate_path)


def test_failed_control_gate_blocks_reset_arm(tmp_path) -> None:
    gate = tmp_path / "validity-gate.json"
    with pytest.raises(ValueError, match="C_validity_gate_required_before_R"):
        require_control_gate(gate)
    gate.write_text('{"passed": false}\n')
    with pytest.raises(ValueError, match="C_reproduction_gate_failed"):
        require_control_gate(gate)
    gate.write_text('{"passed": true}\n')
    require_control_gate(gate)


@pytest.mark.parametrize(
    ("file_name", "kind", "item"),
    [
        ("R-steps.npz", "array", "00_exp_avg_00"),
        ("R-steps.npz", "array", "00_gradient_00"),
        ("R-steps.npz", "array", "01_pre_00"),
        ("R-steps.npz", "array", "15_post_00"),
        ("C-steps.npz", "array", "00_post_00"),
        ("R-trajectory.json", "trajectory", "optimizer_step"),
        ("R-fresh-predictions.npz", "array", "prediction"),
        ("endpoint-evidence.json", "evidence", "effect"),
        ("endpoint-evidence.json", "evidence", "gate"),
        ("endpoint-evidence.json", "evidence", "classification"),
    ],
)
def test_receipt_rebound_semantic_tampering_is_rejected(
    file_name: str, kind: str, item: str
) -> None:
    if not (PUBLICATION / "receipt.json").is_file():
        pytest.skip("seed473 publication has not been executed")
    changed: list[tuple[Path, bytes, int]] = []

    def remember(path: Path) -> bytes:
        content = path.read_bytes()
        changed.append((path, content, path.stat().st_mtime_ns))
        return content

    target = PUBLICATION / file_name
    original = remember(target)
    if kind == "array":
        with np.load(target, allow_pickle=False) as archive:
            values = {key: archive[key].copy() for key in archive.files}
        values[item].flat[0] += np.float32(0.125)
        np.savez_compressed(target, **values)
    elif kind == "trajectory":
        trajectory = json.loads(original)
        trajectory["steps"][0][item] += 1
        target.write_text(json.dumps(trajectory, indent=2, sort_keys=True) + "\n")
    else:
        evidence = json.loads(original)
        if item == "effect":
            evidence["effects"]["fresh"]["equal_input"]["R_minus_C"]["policy_ce"] += 0.1
        elif item == "gate":
            evidence["gates"]["training_J_decreases"] = not evidence["gates"][
                "training_J_decreases"
            ]
        else:
            evidence["classification"] = "tampered"
        target.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    receipt_path = PUBLICATION / "receipt.json"
    receipt = json.loads(remember(receipt_path))
    receipt[file_name] = hashlib.sha256(target.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    try:
        with pytest.raises(ValueError):
            verify(ROOT)
    finally:
        for path, content, mtime in reversed(changed):
            path.write_bytes(content)
            os.utime(path, ns=(mtime, mtime))
