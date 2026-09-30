"""Reusable, observational metrics for policy/value shared-trunk gradients."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

EPS = 1e-12
TRUNK_PREFIXES = ("input_layer.", "residual_layers.")
GROUP_PREFIXES = {
    "input_projection": ("input_layer.",),
    "residual_block_0": ("residual_layers.0.",),
    "residual_block_1": ("residual_layers.1.",),
    "residual_block_2": ("residual_layers.2.",),
    "policy_head": ("policy_hidden_layer.", "policy_head."),
    "value_head": ("value_hidden_layer.", "value_head."),
}


def parameter_groups(
    model: torch.nn.Module,
) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    """Partition every trainable residual_v3 parameter exactly once."""
    groups: dict[str, list[torch.nn.Parameter]] = {key: [] for key in GROUP_PREFIXES}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        matches = [
            group
            for group, prefixes in GROUP_PREFIXES.items()
            if name.startswith(prefixes)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"unclassified trainable residual_v3 parameter: {name}")
        groups[matches[0]].append(parameter)
    if not all(groups.values()):
        raise RuntimeError("unexpected empty residual_v3 parameter group")
    return {key: tuple(value) for key, value in groups.items()}


def gradients(
    loss: torch.Tensor, parameters: tuple[torch.nn.Parameter, ...]
) -> tuple[torch.Tensor, ...]:
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    if any(value is None for value in values):
        raise RuntimeError("unexpected_unused_shared_trunk_gradient")
    return tuple(value.detach() for value in values if value is not None)


def gradient_metrics(
    policy: tuple[torch.Tensor, ...],
    value: tuple[torch.Tensor, ...],
    raw_value: tuple[torch.Tensor, ...],
) -> dict[str, float | bool | None]:
    """Return safe directional and scale measurements for one parameter group."""
    p = torch.cat([item.reshape(-1).float().cpu() for item in policy])
    v = torch.cat([item.reshape(-1).float().cpu() for item in value])
    raw_v = torch.cat([item.reshape(-1).float().cpu() for item in raw_value])
    pn, vn = float(torch.linalg.vector_norm(p)), float(torch.linalg.vector_norm(v))
    raw_vn = float(torch.linalg.vector_norm(raw_v))
    total = p + v
    total_norm = float(torch.linalg.vector_norm(total))
    denominator = pn * vn
    cosine = None if denominator == 0.0 else float(torch.dot(p, v) / denominator)
    dot = float(torch.dot(p, v))
    sum_norm = pn + vn
    cancellation = 0.0 if sum_norm == 0.0 else 1.0 - total_norm / sum_norm
    # The opposite projection is only meaningful when the components conflict.
    policy_opposite = 0.0
    value_opposite = 0.0
    if dot < 0.0:
        policy_opposite = -dot / max(vn, EPS)
        value_opposite = -dot / max(pn, EPS)
    return {
        "cosine": cosine,
        "dot": dot,
        "conflict": bool(cosine is not None and cosine < 0.0),
        "strong_conflict": bool(cosine is not None and cosine < -0.25),
        "policy_norm": pn,
        "value_weighted_norm": vn,
        "value_raw_norm": raw_vn,
        "combined_norm": total_norm,
        "value_policy_norm_ratio": vn / max(pn, EPS),
        "cancellation_ratio": cancellation,
        "policy_opposite_projection": policy_opposite,
        "value_opposite_projection": value_opposite,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Summarize deterministic step observations; empty slices remain explicit."""
    if not rows:
        return {"steps": 0, "eligible_steps": 0}
    usable = [row for row in rows if row["cosine"] is not None]
    cosine = np.asarray([row["cosine"] for row in usable], dtype=float)
    result: dict[str, float | int] = {
        "steps": len(rows),
        "eligible_steps": len(usable),
        "mean_cosine": float(np.mean(cosine)) if len(cosine) else 0.0,
        "median_cosine": float(np.median(cosine)) if len(cosine) else 0.0,
        "p10_cosine": float(np.percentile(cosine, 10)) if len(cosine) else 0.0,
        "p25_cosine": float(np.percentile(cosine, 25)) if len(cosine) else 0.0,
        "p75_cosine": float(np.percentile(cosine, 75)) if len(cosine) else 0.0,
        "p90_cosine": float(np.percentile(cosine, 90)) if len(cosine) else 0.0,
        "conflict_fraction": float(np.mean([row["conflict"] for row in usable]))
        if usable
        else 0.0,
        "strong_conflict_fraction": float(
            np.mean([row["strong_conflict"] for row in usable])
        )
        if usable
        else 0.0,
    }
    for key in (
        "policy_norm",
        "value_weighted_norm",
        "value_raw_norm",
        "value_policy_norm_ratio",
        "cancellation_ratio",
    ):
        result[f"mean_{key}"] = float(np.mean([row[key] for row in rows]))
    return result
