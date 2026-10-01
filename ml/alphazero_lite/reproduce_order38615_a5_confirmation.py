"""Independently reproduce registered score and interval estimates from matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def reproduce(path: Path) -> dict:
    matrix = json.loads(path.read_text())
    scores = {}
    for seed in (391, 392):
        row = matrix["suites"][str(seed)]
        values = np.asarray(row["opening_scores"], dtype=np.float64)
        if len(values) != 512 or not np.isfinite(values).all():
            raise ValueError(f"invalid_opening_scores:{seed}")
        scores[seed] = values
    result = {"suites": {}}
    for seed in (391, 392):
        values = scores[seed]
        rng = np.random.default_rng(seed)
        draws = values[rng.integers(0, 512, size=(10000, 512))].mean(axis=1)
        result["suites"][str(seed)] = {
            "score": float(values.mean()),
            "interval_95_percentile": [
                float(x) for x in np.quantile(draws, [0.025, 0.975])
            ],
        }
    rng = np.random.default_rng(391)
    first = rng.integers(0, 512, size=(10000, 512))
    second = rng.integers(0, 512, size=(10000, 512))
    pooled_draws = (
        scores[391][first].mean(axis=1) + scores[392][second].mean(axis=1)
    ) / 2
    result["pooled_score"] = float((scores[391].mean() + scores[392].mean()) / 2)
    result["pooled_interval_95_percentile"] = [
        float(x) for x in np.quantile(pooled_draws, [0.025, 0.975])
    ]
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    args = parser.parse_args()
    reproduce(args.matrix)


if __name__ == "__main__":
    main()
