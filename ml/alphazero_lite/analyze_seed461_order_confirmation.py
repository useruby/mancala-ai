"""Fixed paired opening-cluster analysis for seed461 order confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARENA = ROOT / ".tmp/seed461-order-confirmation/arena"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(name: str) -> np.ndarray:
    rows = [
        json.loads(line)
        for line in (ARENA / f"{name}-games.jsonl").read_text().splitlines()
    ]
    by_opening: dict[int, list[dict]] = {}
    for row in rows:
        by_opening.setdefault(row["opening_index"], []).append(row)
    if len(by_opening) != 256 or any(
        len(value) != 2 or {r["challenger_player"] for r in value} != {0, 1}
        for value in by_opening.values()
    ):
        raise RuntimeError(f"seat_pairing_failure:{name}")
    return np.asarray(
        [
            sum(
                1.0
                if r["winner"] == "challenger"
                else 0.5
                if r["winner"] == "draw"
                else 0.0
                for r in by_opening[index]
            )
            / 2
            for index in range(256)
        ]
    )


def interval(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    samples = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(
        axis=1
    )
    return [float(np.percentile(samples, 0.1)), float(np.percentile(samples, 99.9))]


def main() -> None:
    all_scores = {
        f"{trajectory}-{epoch}": scores(f"{trajectory}-{epoch}")
        for trajectory in ("O0", "T1", "T2", "T3", "T4", "T5")
        for epoch in ("E1", "E4")
    }
    rng = np.random.default_rng(383)
    primary = {}
    for treatment in ("T1", "T2", "T3", "T4", "T5"):
        difference = all_scores[f"{treatment}-E4"] - all_scores["O0-E4"]
        primary[treatment] = {
            "estimate": float(difference.mean()),
            "interval": interval(difference, rng),
        }
    result = {
        "schema": "seed461-order-confirmation-analysis-v1",
        "games": 6144,
        "pairing": "two opposite challenger seats per opening",
        "bootstrap": {
            "samples": 10000,
            "seed": 383,
            "interval": "Bonferroni-adjusted 99% (99.8% per contrast)",
        },
        "arena_reports": {name: sha(ARENA / f"{name}.json") for name in all_scores},
        "primary_e4_minus_o0": primary,
        "secondary_e4_minus_e1": {
            trajectory: float(
                (all_scores[f"{trajectory}-E4"] - all_scores[f"{trajectory}-E1"]).mean()
            )
            for trajectory in ("O0", "T1", "T2", "T3", "T4", "T5")
        },
    }
    (
        ROOT / "docs/data/seed461-batch-order-sensitivity-confirmation-results.json"
    ).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
