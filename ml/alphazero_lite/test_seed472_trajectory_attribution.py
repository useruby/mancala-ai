"""Synthetic arithmetic and receipt-rebound semantic tests for seed472."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.alphazero_lite.verify_seed472_trajectory_attribution import verify

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed472-trajectory-attribution"


def _classify(a: float, r: float, delta: float) -> str:
    if delta > 1e-6 and r > 0 and r > abs(a):
        return "remainder_dominant"
    if delta > 1e-6 and a > 0 and a >= abs(r):
        return "direction_dominant"
    return "mixed_or_inconclusive"


def _actual_scalar_change(
    theta: float, step: float, slope: float, curvature: float
) -> tuple[float, float, float]:
    parameter = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    objective = slope * parameter + curvature * parameter.square()
    gradient = torch.autograd.grad(objective, parameter)[0].item()
    after = parameter.detach() + step
    observed = (slope * after + curvature * after.square() - objective.detach()).item()
    first = gradient * step
    return observed, first, observed - first


def test_real_computation_synthetic_classification_cases() -> None:
    observed, first, remainder = _actual_scalar_change(0.1, 0.01, 1.0, 0.0)
    assert _classify(first, remainder, observed) == "direction_dominant"
    observed, first, remainder = _actual_scalar_change(0.0, 0.1, 0.0, 1.0)
    assert _classify(first, remainder, observed) == "remainder_dominant"
    observed, first, remainder = _actual_scalar_change(0.0, 0.0, 0.0, 1.0)
    assert _classify(first, remainder, observed) == "mixed_or_inconclusive"


def test_real_computation_weighting_dependent_and_telescoping() -> None:
    # Repeated identity has occurrence-specific targets, so the two gradients differ.
    theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    occurrence_target = torch.tensor([0.0, 2.0, 0.5], dtype=torch.float64)
    identity = ["duplicate", "duplicate", "single"]
    exposure = torch.full((3,), 1 / 3, dtype=torch.float64)
    equal = torch.tensor([0.25, 0.25, 0.5], dtype=torch.float64)
    per_occurrence = (theta - occurrence_target).square()
    gradients = []
    for weights in (exposure, equal):
        gradients.append(
            torch.autograd.grad(
                (per_occurrence * weights).sum(), theta, retain_graph=True
            )[0].item()
        )
    assert identity[0] == identity[1] and occurrence_target[0] != occurrence_target[1]
    assert gradients[0] != gradients[1]
    # Separate local nonlinear examples produce opposite labels under their respective weights.
    weighted_labels = [_classify(2e-7, 2e-6, 2.2e-6), _classify(2e-6, 1e-7, 2.1e-6)]
    shared = (
        weighted_labels[0]
        if weighted_labels[0] == weighted_labels[1]
        else "weighting_dependent"
    )
    assert shared == "weighting_dependent"
    losses = np.asarray([0.1, -0.03, 0.2, 0.05], dtype=np.float64)
    step_changes = np.diff(losses)
    assert np.sum(step_changes, dtype=np.float64) == pytest.approx(
        losses[-1] - losses[0], abs=1e-15
    )


def test_zero_displacement_and_parameter_buffer_preservation() -> None:
    class Small(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)
            self.register_buffer("marker", torch.tensor([7.0]))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.linear(value)

    model = Small()
    before = [p.detach().clone() for p in model.parameters()]
    buffer = model.marker.clone()
    displacement = [torch.zeros_like(p) for p in model.parameters()]
    assert all(torch.count_nonzero(d) == 0 for d in displacement)
    assert all(
        torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True)
    )
    assert torch.equal(buffer, model.marker)


def test_valid_published_baseline_verifies() -> None:
    if not (OUT / "receipt.json").exists():
        pytest.skip("seed472 evidence is generated after frozen source tests")
    verify(ROOT)


def _rebind(path: Path, receipt_path: Path, receipt: dict[str, str]) -> None:
    receipt[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize(
    "target",
    [
        "gradient",
        "moment",
        "pre_state",
        "post_state",
        "prediction",
        "target",
        "weight",
        "displacement",
        "first_order",
        "remainder",
        "total",
        "counterfactual_normalization",
        "record_coverage",
        "classification",
    ],
)
def test_actual_verifier_rejects_receipt_rebound_semantic_tampering(
    target: str,
) -> None:
    if not (OUT / "receipt.json").exists():
        pytest.skip("seed472 evidence is generated after frozen source tests")
    receipt_path = OUT / "receipt.json"
    report_path = OUT / "diagnostic.json"
    evidence_path = OUT / "evidence.npz"
    touched = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (receipt_path, report_path, evidence_path)
    }
    receipt = json.loads(touched[receipt_path][0])
    try:
        if target in {
            "gradient",
            "moment",
            "pre_state",
            "post_state",
            "prediction",
            "target",
            "weight",
            "displacement",
            "counterfactual_normalization",
        }:
            with np.load(evidence_path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            keys = {
                "gradient": "step00_J_gradient_00",
                "moment": "step00_carried_exp_avg_00",
                "pre_state": "step00_pre_00",
                "post_state": "step00_post_00",
                "prediction": "step00_pre_prediction",
                "target": "occurrence_target",
                "weight": "weight_equal_input",
                "displacement": "step00_realized_displacement_00",
                "counterfactual_normalization": "step00_reset_adam_scale",
            }
            key = keys[target]
            arrays[key].flat[0] += np.asarray(0.125, dtype=arrays[key].dtype)
            np.savez_compressed(evidence_path, **arrays)
            changed_path = evidence_path
        else:
            report = json.loads(touched[report_path][0])
            if target == "first_order":
                report["steps"][0]["first_order"] += 0.125
            elif target == "remainder":
                report["steps"][0]["remainder"] += 0.125
            elif target == "total":
                report["totals"]["equal_input"]["A_signed"] += 0.125
            elif target == "record_coverage":
                report["steps"].pop()
            else:
                report["classification"] = "remainder_dominant"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            changed_path = report_path
        _rebind(changed_path, receipt_path, receipt)
        with pytest.raises(ValueError):
            verify(ROOT)
    finally:
        for path, (contents, mtime) in touched.items():
            path.write_bytes(contents)
            os.utime(path, ns=(mtime, mtime))
