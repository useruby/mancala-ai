"""Recompute registered cross-order E4 estimates from its score matrix."""

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
    data = json.loads(args.matrix.read_text())
    rows = data["rows"]
    if len(rows) != 256:
        raise ValueError("score_matrix_shape_mismatch")
    treatment = np.asarray([row["P"] for row in rows], dtype=np.float64)
    baselines = np.stack(
        [
            np.asarray([row[f"A{i}"] for row in rows], dtype=np.float64)
            for i in range(1, 6)
        ]
    )
    primary_opening = treatment - baselines.mean(axis=0)
    effects = {
        f"P-A{i}": float(np.mean(treatment - baselines[i - 1])) for i in range(1, 6)
    }
    rng = np.random.default_rng(args.seed)
    sampled = rng.integers(0, len(rows), size=(args.resamples, len(rows)))
    primary_draws = primary_opening[sampled].mean(axis=1)
    absolute_draws = treatment[sampled].mean(axis=1)
    primary_ci = np.percentile(primary_draws, [2.5, 97.5])
    absolute_ci = np.percentile(absolute_draws, [2.5, 97.5])
    print(f"P_score={treatment.mean():.8f}")
    for index, score in enumerate(baselines.mean(axis=1), 1):
        print(
            f"A{index}_score={score:.8f} P_minus_A{index}={effects[f'P-A{index}']:+.8f}"
        )
    print(f"primary_effect={primary_opening.mean():+.8f}")
    print(
        f"primary_95_percentile_interval=[{primary_ci[0]:+.8f}, {primary_ci[1]:+.8f}]"
    )
    print(
        f"P_absolute_95_percentile_interval=[{absolute_ci[0]:.8f}, {absolute_ci[1]:.8f}]"
    )
    print(
        f"worst_comparison={min(effects, key=effects.get)} effect={min(effects.values()):+.8f}"
    )


if __name__ == "__main__":
    main()
