"""Pure bootstrap calculations for public seed416 verification."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def bootstrap_paired(
    rows: list[dict[str, Any]], *, samples: int, seed: int
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    seats: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        key = (row["lane"], str(row["opening_id"]))
        grouped[key].append(float(row["opponent_score"]))
        seats[key].add(int(row["game"]["challenger_player"]))
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for (lane, opening), values in grouped.items():
        if len(values) != 2 or seats[(lane, opening)] != {0, 1}:
            raise ValueError(f"opening_seat_pair_incomplete:{lane}:{opening}")
        scores[lane][opening] = sum(values) / 2
    if set(scores) != {"A", "B"} or set(scores["A"]) != set(scores["B"]):
        raise ValueError("lane_opening_matrix_mismatch")
    openings = sorted(scores["A"])
    if len(openings) != 512:
        raise ValueError("opening_count_mismatch")
    delta = np.asarray([scores["B"][k] - scores["A"][k] for k in openings])
    b = np.asarray([scores["B"][k] for k in openings])
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(openings), size=(samples, len(openings)))
    delta_ci = [
        float(x) for x in np.quantile(delta[indexes].mean(axis=1), [0.025, 0.975])
    ]
    b_ci = [float(x) for x in np.quantile(b[indexes].mean(axis=1), [0.025, 0.975])]
    mean_delta, mean_b = float(delta.mean()), float(b.mean())
    advance = (
        mean_delta >= 0.03 and delta_ci[0] > 0 and mean_b >= 0.53 and b_ci[0] > 0.50
    )
    return {
        "schema": "seed416-policy-target-softening-analysis-v1",
        "opening_clusters": len(openings),
        "samples": samples,
        "seed": seed,
        "primary_mean_B_minus_A": mean_delta,
        "primary_95_percentile_interval": delta_ci,
        "B_opponent_score_mean": mean_b,
        "B_opponent_score_95_percentile_interval": b_ci,
        "decision": "advance_to_separate_confirmation"
        if advance
        else "retain_baseline",
        "inference": "conditional on this dataset and training seed",
        "per_opening": {
            k: {
                "A": scores["A"][k],
                "B": scores["B"][k],
                "B_minus_A": scores["B"][k] - scores["A"][k],
            }
            for k in openings
        },
    }
