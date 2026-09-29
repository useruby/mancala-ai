"""Compute a deterministic opening-clustered arena score comparison."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def score(row: dict) -> float:
    winner = row["winner"]
    if winner == "challenger":
        return 1.0
    if winner == "draw":
        return 0.5
    if winner == "current":
        return 0.0
    raise ValueError(f"unknown winner: {winner!r}")


def opening_scores(path: Path) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        grouped[int(row["opening_index"])].append(score(row))
    if not grouped:
        raise ValueError(f"empty arena game report: {path}")
    return {opening: float(np.mean(values)) for opening, values in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    left = opening_scores(args.left)
    right = opening_scores(args.right)
    if left.keys() != right.keys():
        raise ValueError("paired arena opening identities differ")
    deltas = np.asarray([left[key] - right[key] for key in sorted(left)], dtype=float)
    rng = np.random.default_rng(args.seed)
    draws = deltas[rng.integers(0, len(deltas), size=(args.samples, len(deltas)))].mean(
        axis=1
    )
    result = {
        "schema": "azlite_paired_arena_opening_bootstrap_v1",
        "left_minus_right": float(deltas.mean()),
        "confidence_interval_95": {
            "lower": float(np.quantile(draws, 0.025)),
            "upper": float(np.quantile(draws, 0.975)),
        },
        "opening_count": len(deltas),
        "samples": args.samples,
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
