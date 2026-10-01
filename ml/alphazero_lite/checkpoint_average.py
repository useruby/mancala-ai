"""Deterministic arithmetic averaging for compatible NumPy checkpoints."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def average_checkpoints(
    checkpoints: Sequence[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Average floating arrays with float64 accumulation; preserve other arrays."""
    if len(checkpoints) < 2:
        raise ValueError("at_least_two_checkpoints_required")
    keys = set(checkpoints[0])
    if any(set(checkpoint) != keys for checkpoint in checkpoints[1:]):
        raise ValueError("checkpoint_keys_mismatch")

    result: dict[str, np.ndarray] = {}
    for key in sorted(keys):
        arrays = [np.asarray(checkpoint[key]) for checkpoint in checkpoints]
        reference = arrays[0]
        for array in arrays:
            if array.shape != reference.shape:
                raise ValueError(f"checkpoint_shape_mismatch:{key}")
            if array.dtype != reference.dtype:
                raise ValueError(f"checkpoint_dtype_mismatch:{key}")
            if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                raise ValueError(f"checkpoint_nonfinite:{key}")
        if np.issubdtype(reference.dtype, np.floating):
            accumulator = np.zeros(reference.shape, dtype=np.float64)
            for array in arrays:
                accumulator += array.astype(np.float64)
            averaged = (accumulator / len(arrays)).astype(reference.dtype)
            if not np.isfinite(averaged).all():
                raise ValueError(f"average_nonfinite:{key}")
            result[key] = averaged
        else:
            if any(not np.array_equal(array, reference) for array in arrays[1:]):
                raise ValueError(f"checkpoint_nonfloating_mismatch:{key}")
            result[key] = reference.copy()
    return result


def load_npz(path: str) -> dict[str, np.ndarray]:
    """Load a checkpoint without retaining an open NPZ handle."""
    with np.load(path, allow_pickle=False) as checkpoint:
        return {key: checkpoint[key].copy() for key in checkpoint.files}
