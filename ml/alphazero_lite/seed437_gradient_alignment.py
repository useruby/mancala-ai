"""Numerical primitives for frozen training/unseen gradient alignment (seed437)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def equal_input_weights(input_identities: list[str]) -> np.ndarray:
    """Give every input identity equal total mass, retaining row multiplicity."""
    if not input_identities:
        raise ValueError("empty_validation_population")
    counts: dict[str, int] = {}
    for identity in input_identities:
        counts[identity] = counts.get(identity, 0) + 1
    if any(not identity for identity in input_identities):
        raise ValueError("empty_input_identity")
    return np.asarray(
        [1.0 / (len(counts) * counts[identity]) for identity in input_identities],
        dtype=np.float64,
    )


def normalized_exposure_weights(count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("empty_validation_population")
    return np.full(count, 1.0 / count, dtype=np.float64)


def dot(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("gradient_shape_mismatch")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("nonfinite_gradient")
    return float(np.dot(left.astype(np.float64), right.astype(np.float64)))


def geometry(left: np.ndarray, right: np.ndarray) -> dict[str, float | None]:
    """Return dot, cosine and unit-descent CE derivative with safe zero norms."""
    product = dot(left, right)
    left_norm = float(np.linalg.norm(left.astype(np.float64)))
    right_norm = float(np.linalg.norm(right.astype(np.float64)))
    denominator = left_norm * right_norm
    return {
        "dot": product,
        "cosine": product / denominator if denominator else None,
        "left_norm": left_norm,
        "right_norm": right_norm,
        "unit_descent_ce_change": -product / left_norm if left_norm else None,
    }


def sum_vectors(vectors: Mapping[str, np.ndarray], expected_size: int) -> np.ndarray:
    """Sum additive partitions and reject missing/shape-invalid components."""
    if not vectors:
        raise ValueError("empty_gradient_decomposition")
    result = np.zeros(expected_size, dtype=np.float64)
    for vector in vectors.values():
        if vector.shape != (expected_size,) or not np.isfinite(vector).all():
            raise ValueError("gradient_component_invalid")
        result += vector.astype(np.float64)
    return result
