"""Synthetic numerical checks for seed468 transport calculations."""

import numpy as np
import torch

from ml.alphazero_lite.seed468_transport import (
    adam_displacement,
    classify,
    geometry,
    gradients,
    proposals,
    weights,
)


def test_weighting_retains_occurrences_and_duplicate_targets() -> None:
    identities = ["a", "a", "b"]
    exposure = weights(identities, False)
    equal = weights(identities, True)
    assert np.allclose(exposure, [1 / 3] * 3)
    assert np.allclose(equal, [1 / 4, 1 / 4, 1 / 2])
    target_grads = np.asarray([[1.0, 0], [3.0, 0], [0, 2.0]])
    assert not np.array_equal(exposure @ target_grads, equal @ target_grads)


def test_unused_parameters_are_explicit_zero() -> None:
    used = torch.nn.Parameter(torch.tensor([2.0]))
    unused = torch.nn.Parameter(torch.tensor([4.0, 5.0]))
    got = torch.autograd.grad((used**2).sum(), (used, unused), allow_unused=True)
    complete = [
        torch.zeros_like(p) if g is None else g
        for p, g in zip((used, unused), got, strict=True)
    ]
    assert np.array_equal(complete[1].numpy(), [0, 0])
    assert np.array_equal(
        gradients(torch.nn.Linear(1, 1), torch.tensor(0.0, requires_grad=True)),
        np.zeros(2, np.float32),
    )


def test_clipping_and_adam_against_pytorch_isolated_copy() -> None:
    gradient = np.asarray([3.0, 4.0], dtype=np.float32)
    delta, raw_norm, clipped = adam_displacement(gradient)
    assert raw_norm == 5.0 and clipped
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([parameter], lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    parameter.grad = torch.tensor([0.6, 0.8])
    optimizer.step()
    assert np.allclose(delta, parameter.detach().numpy(), rtol=1e-6, atol=1e-8)
    zero, _, clipped_zero = adam_displacement(np.zeros(2, dtype=np.float32))
    assert not clipped_zero and np.array_equal(zero, [0, 0])


def test_matched_norm_negative_gradient_and_geometry_zero_cases() -> None:
    proposal = proposals(np.asarray([0.2, -0.3], dtype=np.float32))
    assert np.isclose(
        np.linalg.norm(proposal["adam"]), np.linalg.norm(proposal["negative_gradient"])
    )
    assert geometry(np.zeros(2), proposal["adam"])["cosine"] is None
    assert geometry(np.asarray([1.0, 0.0]), np.asarray([-1.0, 0.0]))["cosine"] == -1.0
    assert geometry(np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0]))["cosine"] == 1.0


def _records(p: float, j: float, pa: float, ja: float) -> list[dict]:
    result = []
    for objective, neg, adam in (("P", p, pa), ("J", j, ja)):
        for direction, cosine in (("negative_gradient", neg), ("adam", adam)):
            for weighting in ("exposure_weighted", "equal_input"):
                for batch in range(16):
                    result.append(
                        {
                            "objective": objective,
                            "direction": direction,
                            "weighting": weighting,
                            "cosine": cosine,
                            "batch": batch,
                        }
                    )
    return result


def test_classification_precedence_and_ambiguity() -> None:
    assert classify(_records(1, -1, -1, -1)) == "minibatch_policy_transport_failure"
    assert classify(_records(-1, 1, 1, 1)) == "fresh_value_transport_failure"
    assert classify(_records(-1, -1, 1, -1)) == "adam_direction_transport_failure"
    assert classify(_records(-1, -1, -1, -1)) == "first_order_descent_consistent"
    assert classify(_records(0, 0, 0, 0)) == "mixed_or_inconclusive_transport"
