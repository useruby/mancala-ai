"""Pure arithmetic for seed455 archived-path attribution."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


GROUPS = ("shared_trunk", "policy_head", "value_head")


def row_weights(identities: list[str], weighting: str) -> np.ndarray:
    """Weight every exposure or give each exact identity equal mass."""
    if not identities:
        raise ValueError("empty_cohort")
    if weighting == "exposure_weighted":
        return np.full(len(identities), 1.0 / len(identities), dtype=np.float64)
    if weighting != "equal_input":
        raise ValueError(f"unknown_weighting:{weighting}")
    counts = Counter(identities)
    return np.asarray(
        [1.0 / (len(counts) * counts[identity]) for identity in identities],
        dtype=np.float64,
    )


def decompose(
    before: float,
    after: float,
    gradient: np.ndarray,
    delta: np.ndarray,
    layout: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate exact finite loss change and first-order displacement terms."""
    if gradient.ndim != 1 or delta.ndim != 1 or gradient.shape != delta.shape:
        raise ValueError("gradient_displacement_shape_mismatch")
    d = float(np.float64(after) - np.float64(before))
    s = float(np.dot(gradient.astype(np.float64), delta.astype(np.float64)))
    group_s = {group: 0.0 for group in GROUPS}
    covered = np.zeros(len(gradient), dtype=bool)
    for entry in layout:
        start, stop = int(entry["start"]), int(entry["stop"])
        group = entry["group"]
        if group not in group_s or start < 0 or stop > len(gradient) or start >= stop:
            raise ValueError("invalid_parameter_layout")
        if covered[start:stop].any():
            raise ValueError("overlapping_parameter_layout")
        covered[start:stop] = True
        group_s[group] += float(np.dot(gradient[start:stop], delta[start:stop]))
    if not covered.all() or not np.isclose(
        sum(group_s.values()), s, atol=2e-10, rtol=2e-10
    ):
        raise ValueError("group_slope_reconciliation_failed")
    return {"d": d, "s": s, "r": float(d - s), "group_s": group_s}


def totals(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Sum attribution terms in float64."""
    return {
        key: float(np.sum(np.asarray([row[key] for row in rows], dtype=np.float64)))
        for key in ("d", "s", "r")
    }


def classify(exposure: dict[str, float], equal_input: dict[str, float]) -> str:
    """Apply seed455's fixed both-weightings rule to all-step totals."""
    values = (exposure, equal_input)
    if all(v["D"] > 2e-6 and v["S"] >= 0.5 * v["D"] for v in values):
        return "fresh_unseen_direction_dominant"
    if all(v["D"] > 2e-6 and v["R"] >= 0.5 * v["D"] for v in values):
        return "fresh_unseen_remainder_dominant"
    return "mixed_or_weighting_dependent_attribution"
