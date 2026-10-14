"""Focused real-autograd and receipt-rebound tests for seed475."""

from __future__ import annotations


import numpy as np
import pytest
import torch
import torch.nn.functional as F
import json

from ml.alphazero_lite.checkpoint_phase_selection import (
    active_pit_stones_from_encoded_state,
)
from ml.alphazero_lite.seed475_policy_population_attribution import BUCKETS
from ml.alphazero_lite.verify_seed475_policy_population_attribution import _close


def _gradient(model, x, target, coeff, mask, selected, denominator):
    logits, _ = model(x)
    losses = -(target * F.log_softmax(logits.masked_fill(mask == 0, -1e9), dim=1)).sum(
        1
    )
    objective = (losses * coeff * selected / denominator).sum()
    return torch.autograd.grad(objective, tuple(model.parameters()), allow_unused=True)


def _vector(grads, model):
    parts = []
    for parameter, gradient in zip(model.parameters(), grads, strict=True):
        value = torch.zeros_like(parameter) if gradient is None else gradient
        parts.append(value.detach().reshape(-1).double())
    return torch.cat(parts)


def test_shared_denominator_additivity_and_multiplicity_and_zero_unused():
    torch.manual_seed(475)

    class TinyPolicyValue(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = torch.nn.Sequential(
                torch.nn.Linear(3, 6), torch.nn.Linear(6, 6)
            )
            self.unused = torch.nn.Parameter(torch.tensor([3.0]))

        def forward(self, inputs):
            return self.policy(inputs), torch.zeros((len(inputs), 1))

    model = TinyPolicyValue()
    x = torch.tensor(
        [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    # Duplicate input rows deliberately carry different targets.
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    target6 = torch.zeros((4, 6))
    target6[:, :2] = target
    mask = torch.ones_like(target6)
    coeff = torch.tensor([1.0, 2.0, 3.0, 4.0])
    buckets = torch.tensor([1, 2, 2, 2])
    denominator = float(coeff.sum())
    full = _vector(
        _gradient(model, x, target6, coeff, mask, torch.ones(4), denominator), model
    )
    pieces = []
    for bucket in range(3):
        pieces.append(
            _vector(
                _gradient(
                    model,
                    x,
                    target6,
                    coeff,
                    mask,
                    (buckets == bucket).float(),
                    denominator,
                ),
                model,
            )
        )
    assert torch.allclose(full, sum(pieces), atol=2e-7, rtol=2e-6)
    assert torch.count_nonzero(pieces[0]) == 0  # no occurrences, zero gradient
    assert not torch.allclose(target6[0], target6[1])
    # A module disconnected from the objective is represented by exact zero parameters.
    assert model.unused.grad is None
    assert BUCKETS == ("<=16", "17-32", ">32")


def test_bucket_boundaries_and_input_encoding():
    for count, expected in ((16, 0), (17, 1), (32, 1), (33, 2)):
        encoded = np.zeros(27, dtype=np.float32)
        encoded[0] = count / 48
        decoded = active_pit_stones_from_encoded_state(encoded)
        bucket = 0 if decoded <= 16 else 1 if decoded <= 32 else 2
        assert bucket == expected


def test_zero_norm_cosine_convention():
    a = np.zeros(5, dtype=np.float64)
    b = np.ones(5, dtype=np.float64)
    cosine = (
        float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        if np.linalg.norm(a) and np.linalg.norm(b)
        else 0.0
    )
    assert cosine == 0.0


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("population", "occurrences"), 9),
        (("population", "coefficients"), 11.0),
        (("population", "stone_counts"), [16, 17, 32, 33, 33]),
        (("population", "bucket_counts"), [1, 2, 2]),
        (("objective", "bucket_masses"), [1.0, 2.0, 6.0]),
        (("objective", "denominator"), 9.0),
        (("gradients", "full"), [1.0, 2.1]),
        (("contributions", "over32"), -0.4),
        (("classification", "result"), "full_population_opposition"),
    ],
)
def test_receipt_rebound_semantic_mutations_rejected(path, replacement):
    reconstructed = {
        "population": {
            "occurrences": 10,
            "coefficients": 10.0,
            "stone_counts": [16, 17, 32, 33],
            "bucket_counts": [1, 2, 1],
        },
        "objective": {"bucket_masses": [1.0, 2.0, 7.0], "denominator": 10.0},
        "gradients": {"full": [1.0, 2.0]},
        "contributions": {"over32": -0.5},
        "classification": {"result": "full_population_descent"},
    }
    altered = json.loads(json.dumps(reconstructed))
    cursor = altered
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    # This is the verifier's semantic comparison after any receipt/hash rebound.
    with pytest.raises(ValueError, match="value_mismatch"):
        _close(altered, reconstructed, "evidence")


def test_checkpoint_parameters_and_buffers_are_preserved_during_read_only_forward():
    from ml.alphazero_lite import train

    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    model.eval()
    parameters = [parameter.detach().clone() for parameter in model.parameters()]
    buffers = [buffer.detach().clone() for buffer in model.buffers()]
    with torch.inference_mode():
        model(torch.zeros((3, 27)))
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameters, model.parameters(), strict=True)
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(buffers, model.buffers(), strict=True)
    )
