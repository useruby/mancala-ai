"""Non-invasive helpers for historical-worktree reproducibility audits."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

H307 = "e83c263193b0c08f928eb22883859ae719e577b6"
H308 = "6084fd9a446afd5a78ba40d2185d6a805b05d1b7"
MATERIAL_PARAMETER_DIFFERENCE = 1e-7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_commit(worktree: Path, expected: str) -> bool:
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=worktree, text=True
    ).strip()
    return actual == expected


def tensor_difference(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("tensor shape and dtype must match")
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    safe = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-30)
    return {
        "dtype": str(left.dtype),
        "shape": list(left.shape),
        "byte_identical": bool(np.array_equal(left, right)),
        "max_abs": float(difference.max(initial=0.0)),
        "rms": float(np.sqrt(np.mean(difference**2))),
        "max_relative": float((difference / safe).max(initial=0.0)),
        "material": bool(difference.max(initial=0.0) > MATERIAL_PARAMETER_DIFFERENCE),
    }


def first_differing_named_tensor(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> dict[str, Any] | None:
    for name in left:
        comparison = tensor_difference(left[name], right[name])
        if comparison["byte_identical"]:
            continue
        delta = np.abs(left[name].astype(np.float64) - right[name].astype(np.float64))
        index = tuple(
            int(value) for value in np.unravel_index(delta.argmax(), delta.shape)
        )
        return {
            "name": name,
            **comparison,
            "first_max_difference_index": list(index),
            "left_value": float(left[name][index]),
            "right_value": float(right[name][index]),
            "absolute_difference": float(delta[index]),
        }
    return None


def runtime_fingerprint(
    artifacts: dict[str, str], training_config: dict[str, Any]
) -> dict[str, Any]:
    """Return JSON-safe runtime evidence without altering torch configuration."""
    package_freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    )
    relevant_environment = {
        key: os.environ.get(key)
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "PYTHONHASHSEED",
        )
    }
    return {
        "python_executable": sys.executable,
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_build": torch.__config__.show(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "environment": relevant_environment,
        "package_freeze_sha256": hashlib.sha256(package_freeze.encode()).hexdigest(),
        "artifacts": artifacts,
        "training_config": training_config,
    }


def write_fingerprint(path: Path, fingerprint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
