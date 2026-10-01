"""Recompute frozen diagnostic cluster-bootstrap summaries from its matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def reproduce(path: Path) -> dict[str, Any]:
    matrix = json.loads(path.read_text())
    suites = matrix["opening_scores"]
    values = {
        seed: np.asarray(suites[seed], dtype=np.float64) for seed in ("393", "394")
    }
    if any(
        row.shape != (512,) or not np.isfinite(row).all() for row in values.values()
    ):
        raise ValueError("invalid_opening_score_matrix")
    summary: dict[str, Any] = {"suites": {}}
    for seed, scores in values.items():
        rng = np.random.default_rng(int(seed))
        draws = scores[rng.integers(0, 512, size=(10000, 512))].mean(axis=1)
        summary["suites"][seed] = {
            "score": float(scores.mean()),
            "interval_95_percentile": [
                float(x) for x in np.quantile(draws, [0.025, 0.975])
            ],
        }
    rng = np.random.default_rng(393)
    first = rng.integers(0, 512, size=(10000, 512))
    second = rng.integers(0, 512, size=(10000, 512))
    pooled = (
        values["393"][first].mean(axis=1) + values["394"][second].mean(axis=1)
    ) / 2
    summary["pooled_score"] = float((values["393"].mean() + values["394"].mean()) / 2)
    summary["pooled_interval_95_percentile"] = [
        float(x) for x in np.quantile(pooled, [0.025, 0.975])
    ]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    args = parser.parse_args()
    reproduce(args.matrix)


if __name__ == "__main__":
    main()
