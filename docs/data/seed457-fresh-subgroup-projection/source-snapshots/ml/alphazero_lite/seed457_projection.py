"""Deterministic nearest-point projection onto a small intersection of halfspaces."""

from __future__ import annotations

from itertools import combinations

import numpy as np

PRIMAL_TOLERANCE = 1e-10
DUAL_TOLERANCE = 1e-10
COMPLEMENTARITY_TOLERANCE = 1e-10
STATIONARITY_FACTOR = 1e-9


def _dot(left: list[np.ndarray], right: list[np.ndarray]) -> float:
    return float(
        sum(np.sum(a * b, dtype=np.float64) for a, b in zip(left, right, strict=True))
    )


def project(
    displacement: list[np.ndarray], gradients: list[list[np.ndarray]]
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Project ``d`` to ``g_j z <= 0`` and return independently checkable KKT data.

    The active-set search enumerates all subsets (at most eight constraints). A
    least-squares Gram solve handles linearly dependent active normals; a
    candidate is accepted only after all KKT conditions pass.
    """
    if not displacement or not gradients or len(gradients) > 8:
        raise ValueError("invalid_constraint_count")
    shapes = [np.asarray(item).shape for item in displacement]
    d = [np.asarray(item, dtype=np.float64).copy() for item in displacement]
    if any(not np.isfinite(item).all() for item in d):
        raise ValueError("nonfinite_displacement")
    normalized: list[list[np.ndarray]] = []
    active_source: list[int] = []
    for index, gradient in enumerate(gradients):
        if len(gradient) != len(d):
            raise ValueError("constraint_parameter_count_mismatch")
        g = [np.asarray(item, dtype=np.float64).copy() for item in gradient]
        if any(a.shape != s for a, s in zip(g, shapes, strict=True)):
            raise ValueError("constraint_shape_mismatch")
        if any(not np.isfinite(item).all() for item in g):
            raise ValueError("nonfinite_constraint")
        norm = np.sqrt(_dot(g, g))
        if norm != 0.0:
            normalized.append([item / norm for item in g])
            active_source.append(index)

    if not normalized:
        return [item.copy() for item in d], {
            "source_indices": [],
            "dual": [],
            "slacks": [],
            "zero_normals": list(range(len(gradients))),
        }

    rhs = np.asarray([_dot(g, d) for g in normalized], dtype=np.float64)
    gram = np.asarray([[_dot(a, b) for b in normalized] for a in normalized])
    best: tuple[float, list[np.ndarray], np.ndarray] | None = None
    count = len(normalized)
    for size in range(count + 1):
        for subset in combinations(range(count), size):
            multipliers = np.zeros(count, dtype=np.float64)
            if subset:
                ids = np.asarray(subset, dtype=np.int64)
                solution, *_ = np.linalg.lstsq(
                    gram[np.ix_(ids, ids)], rhs[ids], rcond=1e-13
                )
                multipliers[ids] = solution
            if np.any(multipliers < -DUAL_TOLERANCE):
                continue
            z = [item.copy() for item in d]
            for coefficient, gradient in zip(multipliers, normalized, strict=True):
                if coefficient:
                    z = [a - coefficient * b for a, b in zip(z, gradient, strict=True)]
            slacks = np.asarray([_dot(g, z) for g in normalized])
            if np.max(slacks, initial=0.0) > PRIMAL_TOLERANCE:
                continue
            if (
                np.max(np.abs(multipliers * slacks), initial=0.0)
                > COMPLEMENTARITY_TOLERANCE
            ):
                continue
            stationarity = [
                a - b
                for a, b in zip(
                    [item.copy() for item in d],
                    [item.copy() for item in d],
                    strict=True,
                )
            ]
            correction = [np.zeros_like(item) for item in d]
            for coefficient, gradient in zip(multipliers, normalized, strict=True):
                correction = [
                    a + coefficient * b
                    for a, b in zip(correction, gradient, strict=True)
                ]
            stationarity = [a - b + c for a, b, c in zip(z, d, correction, strict=True)]
            stat_norm = np.sqrt(_dot(stationarity, stationarity))
            if stat_norm > STATIONARITY_FACTOR * max(1.0, np.sqrt(_dot(d, d))):
                continue
            objective = _dot(
                [a - b for a, b in zip(z, d, strict=True)],
                [a - b for a, b in zip(z, d, strict=True)],
            )
            if best is None or objective < best[0] - 1e-14:
                best = (objective, z, multipliers)
    if best is None:
        raise ValueError("joint_projection_solver_failure")
    _objective, z, dual = best
    report = {
        "source_indices": active_source,
        "dual": dual.tolist(),
        "slacks": [_dot(g, z) for g in normalized],
        "zero_normals": sorted(set(range(len(gradients))) - set(active_source)),
    }
    return z, report


def projected_proposal(
    pre: list[np.ndarray], raw: list[np.ndarray], gradients: list[list[np.ndarray]]
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Project float64(raw)-float64(pre), then write float32(pre+z)."""
    if len(pre) != len(raw):
        raise ValueError("proposal_parameter_count_mismatch")
    displacement = [
        b.astype(np.float64) - a.astype(np.float64)
        for a, b in zip(pre, raw, strict=True)
    ]
    projected, report = project(displacement, gradients)
    result = [
        np.asarray(a.astype(np.float64) + z, dtype=np.float32)
        for a, z in zip(pre, projected, strict=True)
    ]
    report["projected_displacement"] = [a.copy() for a in projected]
    return result, report
