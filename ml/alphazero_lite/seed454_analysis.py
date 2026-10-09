"""Pure numerical analysis for the retrospective seed454 audit."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

SCALES = tuple(2.0**-index for index in range(8))
KL_LIMIT = 0.005 + 1e-10
VALUE_LIMIT = 0.0001 + 1e-12
DOT_LIMIT = 1e-7


def trial_state(
    pre: list[np.ndarray], proposal: list[np.ndarray], scale: float
) -> list[np.ndarray]:
    """Build an archived trial using seed442/447 float32 interpolation."""
    return [
        b.copy() if scale == 1.0 else np.asarray(a + (b - a) * scale, dtype=np.float32)
        for a, b in zip(pre, proposal, strict=True)
    ]


def set_state(parameters: list[torch.Tensor], values: list[np.ndarray]) -> None:
    """Load a parameter vector without retaining any tensor alias."""
    with torch.no_grad():
        for parameter, value in zip(parameters, values, strict=True):
            parameter.copy_(torch.from_numpy(value.copy()))


def measurements(
    model: torch.nn.Module,
    inputs: np.ndarray,
    rows: np.ndarray,
    reference_log_policy: torch.Tensor,
    reference_values: np.ndarray,
) -> tuple[float, float]:
    """Return uniform legal-action KL and value MSE movement on ordered rows."""
    from ml.alphazero_lite import train

    kl_total = 0.0
    value_differences: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(rows), 512):
            selected = rows[offset : offset + 512]
            logits, prediction = model(torch.from_numpy(inputs[selected]))
            legal = torch.from_numpy(
                train.legal_mask_matrix_for_encoded_states(inputs[selected])
            ).bool()
            logp = torch.log_softmax(
                logits.double().masked_fill(~legal, -torch.inf), dim=1
            )
            pre_log = reference_log_policy[selected]
            pre_prob = pre_log.exp().masked_fill(~legal, 0.0)
            row_kl = (pre_prob * (pre_log - logp).masked_fill(~legal, 0.0)).sum(dim=1)
            kl_total += float(row_kl.sum().item()) / len(rows)
            value_differences.append(
                prediction.reshape(-1).cpu().numpy().astype(np.float64)
                - reference_values[offset : offset + len(selected)].astype(np.float64)
            )
    movement = float(
        np.mean(np.square(np.concatenate(value_differences)), dtype=np.float64)
    )
    return kl_total, movement


def feasibility(trial: dict[str, Any], arm: str, *, requested: bool) -> bool:
    """Apply seed454 caps; requested guard metrics must be supplied by caller."""
    kl_key = "full_guard_kl" if requested else "executed_guard_kl"
    value_key = (
        "full_guard_value_movement" if requested else "executed_guard_value_movement"
    )
    return bool(
        trial["batch_kl"] <= KL_LIMIT
        and trial[kl_key] <= KL_LIMIT
        and trial["batch_value_movement"] <= VALUE_LIMIT
        and trial[value_key] <= VALUE_LIMIT
        and (arm != "B" or trial["realized_fresh_dot"] <= DOT_LIMIT)
    )


def largest_feasible(
    trials: list[dict[str, Any]], arm: str, *, requested: bool
) -> float | None:
    """Select the numerically largest feasible tested scale, without monotonicity."""
    feasible = [
        float(row["scale"])
        for row in trials
        if feasibility(row, arm, requested=requested)
    ]
    return max(feasible) if feasible else None


def classify(accepted_violations: int, selection_mismatches: int) -> str:
    """Return the fixed seed454 audit label hierarchy."""
    if accepted_violations:
        return "full_guard_constraint_violation"
    if selection_mismatches:
        return "full_guard_selection_mismatch"
    return "archived_trajectory_matches_full_guard_rule"
