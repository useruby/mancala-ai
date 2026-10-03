"""Full-parameter replay-mixture versus fresh-value gradient accounting."""

from __future__ import annotations

from typing import Any

import torch

from ml.alphazero_lite.policy_value_gradient_audit import parameter_groups


def parameter_layout(
    model: torch.nn.Module,
) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    """Return exhaustive groups, including the complete shared trunk."""
    groups = parameter_groups(model)
    groups["shared_trunk"] = tuple(
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and name.startswith(("input_layer.", "residual_layers."))
    )
    groups["all_parameters"] = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    return groups


def flat_gradient(
    loss: torch.Tensor, parameters: tuple[torch.nn.Parameter, ...]
) -> tuple[torch.Tensor, int]:
    """Differentiate every requested parameter; represent unused gradients as zero."""
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    missing = sum(gradient is None for gradient in grads)
    return (
        torch.cat(
            [
                (torch.zeros_like(parameter) if gradient is None else gradient)
                .detach()
                .double()
                .reshape(-1)
                .cpu()
                for parameter, gradient in zip(parameters, grads)
            ]
        ),
        missing,
    )


def geometry(gradient: torch.Tensor, fresh_value: torch.Tensor) -> dict[str, Any]:
    """Compute directional geometry; undefined zero directions remain null."""
    g, v = gradient.double(), fresh_value.double()
    gn, vn = float(torch.linalg.vector_norm(g)), float(torch.linalg.vector_norm(v))
    dot = float(torch.dot(g, v))
    return {
        "dot": dot,
        "cosine": dot / (gn * vn) if gn and vn else None,
        "gradient_norm": gn,
        "fresh_value_norm": vn,
    }


def dot_accounting(
    contributions: dict[str, torch.Tensor], fresh_value: torch.Tensor
) -> dict[str, float]:
    """Return each objective/source dot, whose sum is dot(sum(contributions), VF)."""
    return {
        name: float(torch.dot(vector.double(), fresh_value.double()))
        for name, vector in contributions.items()
    }


def verify_sums(
    contributions: dict[str, torch.Tensor],
    mixture: torch.Tensor,
    fresh_value: torch.Tensor,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> None:
    """Raise unless vector addition and scalar cross-dot accounting agree."""
    reconstructed = sum(contributions.values(), torch.zeros_like(mixture))
    if not torch.allclose(reconstructed, mixture, atol=atol, rtol=rtol):
        raise ValueError("mixture_gradient_sum_mismatch")
    parts = dot_accounting(contributions, fresh_value)
    total_dot = float(torch.dot(mixture.double(), fresh_value.double()))
    if abs(sum(parts.values()) - total_dot) > atol + rtol * abs(total_dot):
        raise ValueError("cross_dot_sum_mismatch")
