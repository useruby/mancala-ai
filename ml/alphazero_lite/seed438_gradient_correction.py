"""Append-only seed437 gradient correction calculations."""

from __future__ import annotations

import numpy as np
import torch

from ml.alphazero_lite import train


def flat_grad(
    model: torch.nn.Module,
    ids: np.ndarray,
    weights: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
    objective: str,
    chunk: int,
) -> np.ndarray:
    """Compute unweighted P or V gradients; caller applies coefficients once."""
    if objective not in {"policy", "value"}:
        raise ValueError(f"unknown_objective:{objective}")
    if chunk <= 0:
        raise ValueError("invalid_chunk_size")
    params = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    total = np.zeros(sum(parameter.numel() for parameter in params), dtype=np.float64)
    for start in range(0, len(ids), chunk):
        selected = ids[start : start + chunk]
        row_weights = weights[start : start + chunk]
        xb, pb, vb = (torch.from_numpy(array[selected]) for array in (x, p, v))
        mask = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[selected]))
        logits, prediction = model(xb)
        if objective == "value":
            losses = train.compute_value_loss_vector(
                prediction, vb, value_loss="huber", huber_delta=1.0
            )
        else:
            losses = train.compute_policy_cross_entropy(
                logits.masked_fill(mask <= 0, -1e9), pb
            )
        loss = (losses * torch.from_numpy(row_weights).to(losses.dtype)).sum()
        gradients = torch.autograd.grad(loss, params, allow_unused=True)
        total += np.concatenate(
            [
                np.zeros(parameter.numel(), dtype=np.float64)
                if gradient is None
                else gradient.detach().cpu().numpy().astype(np.float64).ravel()
                for parameter, gradient in zip(params, gradients, strict=True)
            ]
        )
    return total
