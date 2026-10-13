"""Focused seed471 protocol tests."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import numpy as np

from ml.alphazero_lite import train
from ml.alphazero_lite.verify_seed471_fresh_only_adam_ablation import (
    decision_from_gates,
    reconstruct_gates,
    summarize_occurrences,
)

ROOT = Path(__file__).resolve().parents[2]
PUBLICATION = ROOT / "docs/data/seed471-fresh-only-adam-ablation"


def test_real_model_sequential_adam_state_and_clipping() -> None:
    torch.manual_seed(471)
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    initial = [p.detach().clone() for p in model.parameters()]
    initial_buffers = [b.detach().clone() for b in model.buffers()]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0000625, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    x = torch.randn(512, 27)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits, value = model(x)
        loss = logits.square().mean() + 0.3 * value.square().mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(norm)
        optimizer.step()
    assert any(
        not torch.equal(a, b) for a, b in zip(initial, model.parameters(), strict=True)
    )
    assert all(optimizer.state[p]["step"].item() == 2 for p in model.parameters())
    assert all(
        torch.equal(a, b) for a, b in zip(initial_buffers, model.buffers(), strict=True)
    )


def test_occurrence_metrics_keep_duplicate_targets_and_weighting() -> None:
    summary = summarize_occurrences(
        np.asarray([1.0, 5.0, 3.0]),
        np.asarray([2.0, 8.0, 4.0]),
        ["same", "same", "other"],
    )
    assert summary["exposure_weighted"]["policy_ce"] == 3.0
    assert summary["equal_input"]["policy_ce"] == 3.0
    assert summary["exposure_weighted"]["value_mse"] == pytest.approx(14 / 3)
    assert summary["equal_input"]["value_mse"] == 4.5


def _gate_fixture() -> dict:
    cohort = {
        weight: {"policy_ce": 1.0, "value_mse": 1.0}
        for weight in ("exposure_weighted", "equal_input")
    }
    return {
        "initializer": {
            "fresh": copy.deepcopy(cohort),
            "pooled": copy.deepcopy(cohort),
            "training_J": {"total": 1.0},
        },
        "C": {
            "fresh": copy.deepcopy(cohort),
            "pooled": copy.deepcopy(cohort),
            "training_J": {"total": 0.9},
        },
        "T": {
            "fresh": copy.deepcopy(cohort),
            "pooled": copy.deepcopy(cohort),
            "training_J": {"total": 0.8},
        },
    }


def test_each_gate_boundary_and_decision_precedence() -> None:
    arms = _gate_fixture()
    for weighting in ("exposure_weighted", "equal_input"):
        arms["T"]["fresh"][weighting]["policy_ce"] = 1.0 - 0.001
        arms["C"]["fresh"][weighting]["policy_ce"] = 1.0
        assert reconstruct_gates(arms)[f"fresh_ce_vs_initializer_{weighting}"]
        assert reconstruct_gates(arms)[f"fresh_ce_vs_C_{weighting}"]
        arms["T"]["fresh"][weighting]["policy_ce"] = 1.0 - 0.001 + 1e-7
        assert not reconstruct_gates(arms)[f"fresh_ce_vs_initializer_{weighting}"]
        arms["T"]["fresh"][weighting]["policy_ce"] = 1.0 - 0.001
        arms["T"]["pooled"][weighting]["policy_ce"] = 1.0 + 0.005
        arms["T"]["pooled"][weighting]["value_mse"] = 1.0 + 0.01
        assert reconstruct_gates(arms)[f"pooled_ce_vs_initializer_{weighting}"]
        assert reconstruct_gates(arms)[f"pooled_mse_vs_initializer_{weighting}"]
        arms["T"]["pooled"][weighting]["policy_ce"] = 1.0 + 0.005 + 1e-7
        assert not reconstruct_gates(arms)[f"pooled_ce_vs_initializer_{weighting}"]
        arms["T"]["pooled"][weighting]["policy_ce"] = 1.0
        arms["T"]["pooled"][weighting]["value_mse"] = 1.0 + 0.01 + 1e-7
        assert not reconstruct_gates(arms)[f"pooled_mse_vs_initializer_{weighting}"]
        arms["T"]["pooled"][weighting]["value_mse"] = 1.0
    gates = reconstruct_gates(arms)
    assert all(gates.values())
    assert decision_from_gates(gates) == "advance_to_fresh_confirmation"
    gates["training_J_decreases"] = False
    assert decision_from_gates(gates) == "close_small_step_fresh_only_branch"


def test_training_objective_population_aggregation() -> None:
    ce = np.asarray([2.0, 7.0, 4.0, 9.0])
    coefficient = np.asarray([1.0, 4.0, 1.0, 1.0])
    fresh = np.asarray([1.0, 0.0, 1.0, 1.0])
    huber = np.asarray([0.5, 10.0, 1.5, 2.0])
    policy_denominator = np.sum(coefficient * fresh)
    value_denominator = 4
    objective = (
        np.sum(ce * coefficient * fresh) / policy_denominator
        + 0.3 * np.sum(huber * fresh) / value_denominator
    )
    assert policy_denominator == 3
    assert value_denominator == 4
    assert objective == pytest.approx((2 + 4 + 9) / 3 + 0.3 * (0.5 + 1.5 + 2) / 4)


def test_receipt_rebound_input_tamper_fails(tmp_path) -> None:
    from ml.alphazero_lite.verify_seed471_fresh_only_adam_ablation import verify

    (tmp_path / "docs/data/seed471-fresh-only-adam-ablation").mkdir(parents=True)
    out = tmp_path / "docs/data/seed471-fresh-only-adam-ablation"
    (out / "registration.json").write_text("{}")
    (out / "receipt.json").write_text('{"status":"post_execution_bound_receipt"}')
    with pytest.raises((KeyError, FileNotFoundError, ValueError)):
        verify(tmp_path)


@pytest.mark.parametrize(
    "target",
    [
        "C-fresh-prediction",
        "C-fresh-target",
        "C-fresh-weight",
        "training-J",
        "effect",
        "gate",
        "classification",
        "trajectory",
        "gradient",
        "moment",
        "state-continuity",
        "learning-rate",
        "authenticated-input",
    ],
)
def test_receipt_rebound_publication_tampering_is_rejected(target: str) -> None:
    if not (PUBLICATION / "receipt.json").is_file():
        pytest.skip("post-execution seed471 publication not present")
    from ml.alphazero_lite.verify_seed471_fresh_only_adam_ablation import verify

    changed: list[tuple[Path, bytes, int]] = []

    def remember(path: Path) -> bytes:
        data = path.read_bytes()
        changed.append((path, data, path.stat().st_mtime_ns))
        return data

    receipt_path = PUBLICATION / "receipt.json"
    receipt_bytes = remember(receipt_path)
    receipt = json.loads(receipt_bytes)
    if target.startswith("C-fresh-"):
        path = PUBLICATION / "C-fresh-predictions.npz"
        remember(path)
        with np.load(path, allow_pickle=False) as z:
            arrays = {key: z[key].copy() for key in z.files}
        key = {
            "C-fresh-prediction": "prediction",
            "C-fresh-target": "policy_target",
            "C-fresh-weight": "equal_input_weight",
        }[target]
        arrays[key].flat[0] += (
            np.float32(0.125) if arrays[key].dtype != np.float64 else 0.125
        )
        np.savez_compressed(path, **arrays)
    elif target in {"training-J", "effect", "gate", "classification"}:
        path = PUBLICATION / "endpoint-evidence.json"
        remember(path)
        evidence = json.loads(path.read_text())
        if target == "training-J":
            evidence["arms"]["T"]["training_J"]["total"] += 0.1
        elif target == "effect":
            evidence["effects"]["fresh"]["equal_input"]["T_minus_C"]["policy_ce"] += 0.1
        elif target == "gate":
            evidence["gates"]["training_J_decreases"] = not evidence["gates"][
                "training_J_decreases"
            ]
        else:
            evidence["classification"] = "tampered_classification"
        path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    elif target == "learning-rate":
        path = PUBLICATION / "registration.json"
        remember(path)
        registration = json.loads(path.read_text())
        registration["arms"]["T"]["lr"] = 0.001
        path.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n")
    elif target == "authenticated-input":
        path = ROOT / "docs/data/seed466-replay-input-bundle/training.npz"
        remember(path)
        path.write_bytes(path.read_bytes() + b"tamper")
        registration_path = PUBLICATION / "registration.json"
        remember(registration_path)
        registration = json.loads(registration_path.read_text())
        registration["inputs_sha256"]["training"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        registration_path.write_text(
            json.dumps(registration, indent=2, sort_keys=True) + "\n"
        )
        receipt["registration.json"] = hashlib.sha256(
            registration_path.read_bytes()
        ).hexdigest()
        path = registration_path
    elif target in {"gradient", "moment", "state-continuity"}:
        path = PUBLICATION / "C-steps.npz"
        remember(path)
        with np.load(path, allow_pickle=False) as z:
            arrays = {key: z[key].copy() for key in z.files}
        key = {
            "gradient": "00_gradient_00",
            "moment": "00_exp_avg_00",
            "state-continuity": "01_pre_00",
        }[target]
        arrays[key].flat[0] += np.float32(0.125)
        np.savez_compressed(path, **arrays)
    else:
        path = PUBLICATION / "C-trajectory.json"
        remember(path)
        trajectory = json.loads(path.read_text())
        trajectory["steps"][0]["clipped"] = not trajectory["steps"][0]["clipped"]
        path.write_text(json.dumps(trajectory, indent=2, sort_keys=True) + "\n")
    receipt[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    try:
        with pytest.raises(ValueError):
            verify(ROOT)
    finally:
        for restore_path, contents, mtime in reversed(changed):
            restore_path.write_bytes(contents)
            os.utime(restore_path, ns=(mtime, mtime))
