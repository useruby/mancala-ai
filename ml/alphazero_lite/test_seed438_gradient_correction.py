"""Synthetic regression checks for the append-only seed438 correction."""

import numpy as np
import pytest
import torch

from ml.alphazero_lite import train
from ml.alphazero_lite.seed438_gradient_correction import flat_grad


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layer = torch.nn.Linear(2, 3)
        self.policy_head = torch.nn.Linear(3, 6)
        self.value_head = torch.nn.Linear(3, 1)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(self.input_layer(value))
        return self.policy_head(hidden), self.value_head(hidden)


def test_value_gradient_matches_analytical_reference_with_multiplicity_and_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        train,
        "legal_mask_matrix_for_encoded_states",
        lambda states: np.ones((len(states), 6), dtype=np.float32),
    )
    model = TinyModel()
    x = np.array([[0.1, 0.3], [0.2, -0.1], [0.1, 0.3]], dtype=np.float32)
    p = np.full((3, 6), 1 / 6, dtype=np.float32)
    v = np.array([[0.2], [-0.4], [0.2]], dtype=np.float32)
    ids = np.array([0, 1, 2])
    weights = np.array([1 / 4, 1 / 2, 1 / 4], dtype=np.float64)
    actual = flat_grad(model, ids, weights, x, p, v, "value", 2)
    params = tuple(model.parameters())
    _, prediction = model(torch.from_numpy(x[ids]))
    losses = train.compute_value_loss_vector(
        prediction, torch.from_numpy(v[ids]), value_loss="huber", huber_delta=1.0
    )
    reference = torch.autograd.grad(
        (losses * torch.from_numpy(weights).float()).sum(), params, allow_unused=True
    )
    expected = np.concatenate(
        [
            np.zeros(parameter.numel())
            if gradient is None
            else gradient.detach().numpy().ravel()
            for parameter, gradient in zip(params, reference, strict=True)
        ]
    )
    assert np.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert np.allclose(actual, flat_grad(model, ids, weights, x, p, v, "value", 8))
    corrected = 0.3 * actual
    mislabeled = 0.09 * actual
    assert np.allclose(corrected, (10 / 3) * mislabeled)
    assert not np.allclose(corrected, mislabeled)


def test_unknown_objective_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown_objective"):
        flat_grad(
            TinyModel(),
            np.array([0]),
            np.ones(1),
            np.zeros((1, 2), np.float32),
            np.zeros((1, 6), np.float32),
            np.zeros((1, 1), np.float32),
            "other",
            1,
        )
