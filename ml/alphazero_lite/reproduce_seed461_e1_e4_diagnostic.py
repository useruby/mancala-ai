"""Reproduce the registered opening-cluster intervals from public matrix data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def reproduce(matrix_path: Path) -> dict[str, Any]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    rows = matrix["opening_scores"]
    if len(rows) != 512 or [row["opening_index"] for row in rows] != list(range(512)):
        raise ValueError("opening_matrix_shape_or_order_invalid")
    e1 = np.asarray([row["E1_score"] for row in rows], dtype=np.float64)
    e4 = np.asarray([row["E4_score"] for row in rows], dtype=np.float64)
    paired = np.asarray([row["paired_E1_minus_E4"] for row in rows], dtype=np.float64)
    if (
        not np.isfinite(e1).all()
        or not np.isfinite(e4).all()
        or not np.isfinite(paired).all()
        or not np.array_equal(paired, e1 - e4)
    ):
        raise ValueError("opening_matrix_scores_invalid")
    rng = np.random.default_rng(397)
    indexes = rng.integers(0, 512, size=(10000, 512))
    samples = {
        "E1": e1[indexes].mean(axis=1),
        "E4": e4[indexes].mean(axis=1),
        "paired_E1_minus_E4": paired[indexes].mean(axis=1),
    }
    means = {
        "E1": float(e1.mean()),
        "E4": float(e4.mean()),
        "paired_E1_minus_E4": float(paired.mean()),
    }
    intervals = {
        name: [float(value) for value in np.quantile(draws, [0.025, 0.975])]
        for name, draws in samples.items()
    }
    result = {
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 397,
        "cluster_count": 512,
        "means": means,
        "intervals_95": intervals,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    args = parser.parse_args()
    print(json.dumps(reproduce(args.matrix), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
