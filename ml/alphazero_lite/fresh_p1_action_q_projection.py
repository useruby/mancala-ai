"""Action-level robust-Q teacher projection primitives."""

from __future__ import annotations

import numpy as np


def action_q_projected_teacher(
    p0: np.ndarray,
    teacher: np.ndarray,
    q384: np.ndarray,
    q1200: np.ndarray,
    visits384: np.ndarray,
    visits1200: np.ndarray,
    legal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only bidirectionally visited teacher transfers with robust Q sign."""
    delta = teacher - p0
    qbar384 = float(np.dot(p0, q384))
    qbar1200 = float(np.dot(p0, q1200))
    supported = (
        legal.astype(bool)
        & (visits384 > 0)
        & (visits1200 > 0)
        & (delta != 0)
        & (delta * (q384 - qbar384) > 0)
        & (delta * (q1200 - qbar1200) > 0)
    )
    allowed = np.where(supported, delta, 0.0)
    positive = float(np.maximum(allowed, 0.0).sum())
    negative = float(np.maximum(-allowed, 0.0).sum())
    transferable = min(positive, negative)
    projected = p0.copy()
    if transferable:
        projected += np.where(allowed > 0, allowed * transferable / positive, 0.0)
        projected += np.where(allowed < 0, allowed * transferable / negative, 0.0)
    projected[~legal.astype(bool)] = 0.0
    if not np.isclose(projected.sum(), 1.0, atol=1e-10):
        raise ValueError("projected teacher is not normalized")
    if np.any(projected < -1e-12):
        raise ValueError("projected teacher contains negative probability")
    return projected, supported


def magnitude_matched_teacher(
    p0: np.ndarray, teacher: np.ndarray, projected: np.ndarray
) -> np.ndarray:
    """Retain the original teacher direction at the projected L1 magnitude."""
    original_l1 = float(np.abs(teacher - p0).sum())
    projected_l1 = float(np.abs(projected - p0).sum())
    gamma = 0.0 if original_l1 == 0 else projected_l1 / original_l1
    if not -1e-10 <= gamma <= 1.0 + 1e-10:
        raise ValueError("projected magnitude exceeds original teacher movement")
    return p0 + min(1.0, max(0.0, gamma)) * (teacher - p0)
