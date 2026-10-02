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
    seed_names = tuple(sorted(suites, key=int))
    if len(seed_names) != 2:
        raise ValueError("expected_two_opening_suites")
    values = {seed: np.asarray(suites[seed], dtype=np.float64) for seed in seed_names}
    if any(
        row.shape != (512,) or not np.isfinite(row).all() for row in values.values()
    ):
        raise ValueError("invalid_opening_score_matrix")
    bootstrap = matrix.get("bootstrap", {})
    resamples = int(bootstrap.get("resamples", 10000))
    per_suite_seeds = bootstrap.get(
        "per_suite_seeds", {seed: int(seed) for seed in seed_names}
    )
    summary: dict[str, Any] = {"suites": {}}
    for seed, scores in values.items():
        rng = np.random.default_rng(int(per_suite_seeds[seed]))
        draws = scores[rng.integers(0, 512, size=(resamples, 512))].mean(axis=1)
        summary["suites"][seed] = {
            "score": float(scores.mean()),
            "bootstrap_seed": int(per_suite_seeds[seed]),
            "resamples": resamples,
            "interval_95_percentile": [
                float(x) for x in np.quantile(draws, [0.025, 0.975])
            ],
        }
    pooled_seed = int(bootstrap.get("pooled_seed", int(seed_names[0])))
    rng = np.random.default_rng(pooled_seed)
    first = rng.integers(0, 512, size=(resamples, 512))
    second = rng.integers(0, 512, size=(resamples, 512))
    pooled = (
        values[seed_names[0]][first].mean(axis=1)
        + values[seed_names[1]][second].mean(axis=1)
    ) / 2
    summary["pooled_score"] = float(
        (values[seed_names[0]].mean() + values[seed_names[1]].mean()) / 2
    )
    summary["pooled_bootstrap_seed"] = pooled_seed
    summary["pooled_resamples"] = resamples
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
