"""Synthetic arithmetic and receipt-rebound tests for seed476."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ml.alphazero_lite.verify_seed476_direction_transport import _same


def _transport(
    gradient: torch.Tensor, pre: torch.Tensor, target_norm: float
) -> torch.Tensor:
    norm = torch.linalg.vector_norm(gradient.double())
    raw = (
        torch.zeros_like(gradient)
        if norm == 0
        else (gradient.double() * (target_norm / norm)).float()
    )
    return (pre + raw).double() - pre.double()


def test_real_autograd_minibatch_opposition_and_coordinate_scaling_reversal():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    full_loss = parameter[0] + parameter[1]
    minibatch_loss = parameter[0] - parameter[1]
    full = torch.autograd.grad(full_loss, parameter)[0]
    mini = torch.autograd.grad(minibatch_loss, parameter)[0]
    unseen = torch.tensor([1.0, 0.0])
    assert torch.dot(unseen, -full) < 0
    assert torch.dot(unseen, -mini) < 0
    opposed_full = torch.tensor([1.0, 0.0])
    opposed_minibatch = torch.tensor([-1.0, 0.0])
    assert torch.dot(unseen, -opposed_full) < 0
    assert torch.dot(unseen, -opposed_minibatch) > 0
    # Diagonal coordinate scaling can rotate an otherwise descending proposal.
    gradient = torch.tensor([1.0, -0.2])
    unseen = torch.tensor([1.0, 2.0])
    adam = -gradient * torch.tensor([0.1, 4.0])
    assert torch.dot(unseen, -gradient) < 0
    assert torch.dot(unseen, adam) > 0


def test_norm_matching_float32_writeback_duplicates_and_zero():
    pre = torch.tensor([1.0000001, -2.0], dtype=torch.float32)
    direction = torch.tensor([3.0, 4.0], dtype=torch.float32)
    realized = _transport(direction, pre, 0.5)
    assert np.isclose(torch.linalg.vector_norm(realized).item(), 0.5, atol=2e-7)
    rounded = (pre + (direction.double() * (0.5 / 5)).float()).double() - pre.double()
    assert torch.equal(realized, rounded)
    assert torch.count_nonzero(_transport(torch.zeros_like(direction), pre, 0.5)) == 0

    # Repeated inputs remain separate occurrences and can carry distinct targets.
    logits = torch.nn.Parameter(torch.zeros((2, 2)))
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    losses = -(targets * torch.log_softmax(logits, dim=1)).sum(1)
    grad = torch.autograd.grad((losses * torch.tensor([1.0, 3.0])).sum() / 4, logits)[0]
    assert not torch.equal(targets[0], targets[1])
    assert torch.equal(grad[:, 0], torch.tensor([-0.125, 0.375]))


@pytest.mark.parametrize(
    "path",
    [
        ("population", "training_occurrences"),
        ("steps", 0, "direction_norms", "full_negative"),
        ("steps", 0, "unseen", "exposure", "directions", "recorded_adam", "dot"),
        ("steps", 0, "unseen", "exposure", "category"),
        ("summary", "exposure", "both_descent"),
    ],
)
def test_receipt_rebound_semantic_mutations_rejected(path):
    baseline = {
        "population": {"training_occurrences": 60015},
        "steps": [
            {
                "direction_norms": {"full_negative": 0.2},
                "unseen": {
                    "exposure": {
                        "directions": {"recorded_adam": {"dot": -1.0}},
                        "category": "both_descent",
                    }
                },
            }
        ],
        "summary": {"exposure": {"both_descent": 16}},
    }
    changed = json.loads(json.dumps(baseline))
    node = changed
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = "corrupt" if isinstance(node[path[-1]], str) else 123456
    with pytest.raises(ValueError):
        _same(changed, baseline, "evidence")


def test_optimizer_parameter_and_buffer_preservation():
    module = torch.nn.BatchNorm1d(2)
    module.eval()
    parameters = [p.detach().clone() for p in module.parameters()]
    buffers = [b.detach().clone() for b in module.buffers()]
    with torch.inference_mode():
        module(torch.ones((4, 2)))
    assert all(
        torch.equal(a, b) for a, b in zip(parameters, module.parameters(), strict=True)
    )
    assert all(
        torch.equal(a, b) for a, b in zip(buffers, module.buffers(), strict=True)
    )
