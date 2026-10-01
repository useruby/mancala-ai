"""Recompute paired estimates and shared-opening bootstrap from a score matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    matrix_data = json.loads(args.matrix.read_text())
    rows = matrix_data["rows"]
    orders = sorted(
        {
            int(column.removeprefix("order_").rsplit("_", 1)[0])
            for column in rows[0]
            if column.startswith("order_") and column.endswith("_A")
        }
    )
    if len(rows) != 256 or not orders:
        raise ValueError("score_matrix_shape_mismatch")
    effects = np.asarray(
        [
            [row[f"order_{order}_B"] - row[f"order_{order}_A"] for row in rows]
            for order in orders
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(args.seed)
    sampled = rng.integers(0, len(rows), size=(args.resamples, len(rows)))
    draws = effects[:, sampled].mean(axis=2).mean(axis=0)
    low, high = np.percentile(draws, [2.5, 97.5])
    for idx, order in enumerate(orders):
        a = np.mean([row[f"order_{order}_A"] for row in rows])
        b = np.mean([row[f"order_{order}_B"] for row in rows])
        print(f"order={order} A={a:.8f} B={b:.8f} B_minus_A={effects[idx].mean():+.8f}")
    print(f"mean_effect={effects.mean():+.8f}")
    print(f"bootstrap_95_percentile_interval=[{low:+.8f}, {high:+.8f}]")


if __name__ == "__main__":
    main()
