"""Pure arithmetic for seed452 finite-step policy attribution."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def weights(identities: list[str]) -> dict[str, np.ndarray]:
    """Return exposure and equal-exact-input weights without merging rows."""
    if not identities:
        raise ValueError("empty_population")
    counts = Counter(identities)
    return {
        "exposure_weighted": np.full(len(identities), 1 / len(identities), np.float64),
        "equal_exact_input": np.asarray(
            [1 / (len(counts) * counts[item]) for item in identities], dtype=np.float64
        ),
    }


def decompose(
    before: float,
    after: float,
    gradient: np.ndarray,
    delta: np.ndarray,
    layout: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate signed loss change, full first-order term and group dots."""
    if gradient.ndim != 1 or delta.ndim != 1 or gradient.shape != delta.shape:
        raise ValueError("gradient_displacement_shape_mismatch")
    d = float(np.float64(after) - np.float64(before))
    s = float(np.dot(gradient.astype(np.float64), delta.astype(np.float64)))
    groups: dict[str, float] = {}
    for entry in layout:
        group = entry["group"]
        start, stop = int(entry["start"]), int(entry["stop"])
        groups[group] = groups.get(group, 0.0) + float(
            np.dot(
                gradient[start:stop].astype(np.float64),
                delta[start:stop].astype(np.float64),
            )
        )
    if not np.isclose(sum(groups.values()), s, atol=2e-10, rtol=2e-10):
        raise ValueError("group_dot_reconciliation_failed")
    return {"d": d, "s": s, "r": float(d - s), "group_s": groups}


def totals(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Sum signed row terms in float64."""
    return {
        key: float(
            np.sum(
                np.asarray([row[key] for row in rows], dtype=np.float64),
                dtype=np.float64,
            )
        )
        for key in ("d", "s", "r")
    }


def classify(totals_by_weighting: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Apply the frozen fresh-policy decision independently per weighting."""
    labels = {}
    for weighting, values in totals_by_weighting.items():
        d, s, r = values["D"], values["S"], values["R"]
        if d > 0 and s >= 0.75 * d:
            label = "recorded_direction_harms_fresh_policy"
        elif d > 0 and r >= 0.75 * d:
            label = "fresh_policy_finite_step_residual_dominant"
        else:
            label = "mixed_or_no_fresh_policy_regression"
        labels[weighting] = label
    unique = set(labels.values())
    overall = (
        next(iter(unique))
        if len(unique) == 1
        else "weighting_dependent_fresh_attribution"
    )
    return {"by_weighting": labels, "overall": overall}
