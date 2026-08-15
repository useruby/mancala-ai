"""Unambiguous opening-suite statistics for AlphaZero-lite evaluations.

``seat_asymmetry_ds`` is P0 challenger score minus P1 challenger score.  It is
a seat/budget asymmetry diagnostic, not a candidate-strength statistic.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def score_from_game(record: dict[str, Any]) -> float:
    """Return challenger score from an arena game record."""
    winner = record.get("winner")
    if winner == "challenger":
        return 1.0
    if winner == "draw":
        return 0.5
    if winner == "current":
        return 0.0
    if "score" in record:
        return float(record["score"])
    raise ValueError("game record has neither winner nor score")


def seat_asymmetry_ds(p0_challenger_score: float, p1_challenger_score: float) -> float:
    """Return P0 challenger score minus P1 challenger score (diagnostic only)."""
    return float(p0_challenger_score - p1_challenger_score)


def candidate_challenger_score(candidate_scores: Iterable[float]) -> float:
    values = list(candidate_scores)
    if not values:
        raise ValueError("candidate scores are required")
    return float(np.mean(values))


def current_control_challenger_score(current_control_scores: Iterable[float]) -> float:
    values = list(current_control_scores)
    if not values:
        raise ValueError("current-control scores are required")
    return float(np.mean(values))


def candidate_minus_current_control_delta(
    candidate_score: float, current_control_score: float
) -> float:
    return float(candidate_score - current_control_score)


def _key(record: dict[str, Any]) -> tuple[int, int]:
    try:
        return int(record.get("opening_index", record.get("game_index", 0))), int(
            record["challenger_player"]
        )
    except KeyError as exc:
        raise ValueError(
            "paired metrics require opening_index and challenger_player"
        ) from exc


def paired_opening_candidate_effect(
    candidate_records: Iterable[dict[str, Any]],
    current_control_records: Iterable[dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Compute the paired candidate effect, clustered by unique opening index.

    Multiple games for an opening/seat are averaged before pairing.  Therefore
    worker partitioning and record ordering cannot affect the statistic.
    """
    candidate_by_key: dict[tuple[int, int], list[float]] = defaultdict(list)
    control_by_key: dict[tuple[int, int], list[float]] = defaultdict(list)
    for record in candidate_records:
        candidate_by_key[_key(record)].append(score_from_game(record))
    for record in current_control_records:
        control_by_key[_key(record)].append(score_from_game(record))
    if set(candidate_by_key) != set(control_by_key):
        missing_candidate = sorted(set(control_by_key) - set(candidate_by_key))
        missing_control = sorted(set(candidate_by_key) - set(control_by_key))
        raise ValueError(
            f"unmatched candidate/current-control games: candidate={missing_candidate}, "
            f"control={missing_control}"
        )
    openings = sorted({opening for opening, _seat in candidate_by_key})
    effects: list[float] = []
    p0_effects: list[float] = []
    p1_effects: list[float] = []
    for opening in openings:
        by_seat = []
        for seat, target in ((0, p0_effects), (1, p1_effects)):
            key = (opening, seat)
            if key not in candidate_by_key:
                raise ValueError(f"opening {opening} is missing challenger seat {seat}")
            delta = candidate_minus_current_control_delta(
                candidate_challenger_score(candidate_by_key[key]),
                current_control_challenger_score(control_by_key[key]),
            )
            by_seat.append(delta)
            target.append(delta)
        effects.append(float(np.mean(by_seat)))
    if not effects:
        raise ValueError("at least one paired opening is required")
    data = np.asarray(effects, dtype=float)
    draws = data[
        np.random.default_rng(bootstrap_seed).integers(
            0, len(data), size=(bootstrap_samples, len(data))
        )
    ].mean(axis=1)
    return {
        "paired_candidate_effect": float(data.mean()),
        "opening_bootstrap_ci": {
            "lower_95": float(np.quantile(draws, 0.025)),
            "upper_95": float(np.quantile(draws, 0.975)),
            "samples": bootstrap_samples,
            "unique_openings": len(openings),
        },
        "p0_effect": float(np.mean(p0_effects)),
        "p1_effect": float(np.mean(p1_effects)),
        "orientation_decomposition": {
            "candidate_challenger_score": candidate_challenger_score(
                value for values in candidate_by_key.values() for value in values
            ),
            "current_control_challenger_score": current_control_challenger_score(
                value for values in control_by_key.values() for value in values
            ),
        },
        "per_opening_effect": dict(zip(openings, effects, strict=True)),
    }
