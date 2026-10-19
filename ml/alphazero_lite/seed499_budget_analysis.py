"""Matched-opening analysis for the exploratory seed499 search-budget contrast."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPLICATES = 20_000
BOOTSTRAP_SEED = 499
OPENINGS = 512


def _score(row: dict[str, Any]) -> float:
    winner = row.get("winner")
    if winner in ("F", "challenger"):
        return 1.0
    if winner in ("draw",):
        return 0.5
    if winner in ("initializer", "current"):
        return 0.0
    raise ValueError("unknown_winner")


def paired_scores(rows: list[dict[str, Any]], label: str) -> list[float]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["opening_index"]), []).append(row)
    if set(grouped) != set(range(OPENINGS)):
        raise ValueError(f"{label}_opening_coverage")
    scores = []
    for opening in range(OPENINGS):
        pair = grouped[opening]
        if len(pair) != 2 or {int(row["F_player"]) for row in pair} != {
            0,
            1,
        }:
            raise ValueError(f"{label}_seat_pair")
        if len({row["opening_state_hash"] for row in pair}) != 1:
            raise ValueError(f"{label}_opening_identity")
        scores.append(sum(_score(row) for row in pair) / 2)
    return scores


def _bootstrap(values: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, OPENINGS, size=(REPLICATES, OPENINGS))
    means = values[indices].mean(axis=1)
    return {
        "replicates": REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "method": "percentile bootstrap resampling 512 matched opening clusters",
        "one_sided_95_lower": float(np.quantile(means, 0.05)),
        "two_sided_95_interval": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }


def analyze(
    low_budget_rows: list[dict[str, Any]], historical_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare new 128-simulation games with historical matched seed488 games."""
    low = paired_scores(low_budget_rows, "128")
    historical = paired_scores(historical_rows, "historical_1200")
    if [row["opening_state_hash"] for row in low_budget_rows] == []:
        raise ValueError("missing_opening_identity")
    low_by_opening = {
        int(row["opening_index"]): row["opening_state_hash"] for row in low_budget_rows
    }
    high_by_opening = {
        int(row["opening_index"]): row["opening_state_hash"] for row in historical_rows
    }
    if low_by_opening != high_by_opening:
        raise ValueError("matched_opening_identity_mismatch")
    differences = np.asarray(low, dtype=np.float64) - np.asarray(
        historical, dtype=np.float64
    )
    low_games = [_score(row) for row in low_budget_rows]
    counts = Counter(row["winner"] for row in low_budget_rows)
    standalone = _bootstrap(np.asarray(low, dtype=np.float64))
    contrast = _bootstrap(differences)
    normal_ms = [float(row["normal_puct_time_ms"]) for row in low_budget_rows]
    exact_ms = [float(row["exact_root_solve_time_ms"]) for row in low_budget_rows]
    capped = sum(int(row["game_length"]) == 200 for row in low_budget_rows)
    return {
        "schema": "seed499-budget-contrast-analysis-v1",
        "new_game_count": len(low_budget_rows),
        "historical_game_count": len(historical_rows),
        "standalone_128": {
            "score": float(np.mean(low_games)),
            "wins": counts.get("F", counts.get("challenger", 0)),
            "draws": counts.get("draw", 0),
            "losses": counts.get("initializer", counts.get("current", 0)),
            "capped_games": capped,
            "recorded_search_time_ms": {
                "normal_puct_total": float(sum(normal_ms)),
                "exact_root_total": float(sum(exact_ms)),
                "combined_total": float(sum(normal_ms) + sum(exact_ms)),
            },
            "opening_cluster_bootstrap": standalone,
        },
        "paired_budget_contrast": {
            "estimand": "mean across 512 openings of (128-simulation paired F score minus historical 1,200-simulation paired F score)",
            "mean_difference": float(np.mean(differences)),
            **contrast,
            "supports_budget_sensitivity": bool(contrast["one_sided_95_lower"] > 0),
            "historical_comparison_only": True,
        },
        "interpretation": (
            "A positive budget contrast does not establish strength above the initializer."
        ),
        "seed461_overlap_limitation": (
            "Non-overlap with the unresolved seed461 continuation population cannot be certified."
        ),
        "promotion": False,
        "fresh_confirmation": False,
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]
