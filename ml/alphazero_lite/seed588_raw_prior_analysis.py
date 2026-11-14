"""Frozen paired metrics and opening-cluster bootstrap for #588."""

from __future__ import annotations

import random
from typing import Any

BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 58_820_261_114
QUANTILES = (0.025, 0.975)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    low = int(location)
    high = min(low + 1, len(ordered) - 1)
    fraction = location - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _top_action(priors: list[float], legal: list[int]) -> int:
    return min(legal, key=lambda action: (-float(priors[action]), action))


def _wdl(margin: int) -> int:
    return (margin > 0) - (margin < 0)


def root_metrics(
    *,
    p_priors: list[float],
    t_priors: list[float],
    legal: list[int],
    margins: dict[int, int],
) -> dict[str, float]:
    """Calculate paired arm outcomes and deltas for one root."""
    best = max(margins[action] for action in legal)
    optimal = {action for action in legal if margins[action] == best}
    p_action = _top_action(p_priors, legal)
    t_action = _top_action(t_priors, legal)
    best_wdl = max(_wdl(margins[action]) for action in legal)
    p_mass = sum(float(p_priors[action]) for action in optimal)
    t_mass = sum(float(t_priors[action]) for action in optimal)
    p_regret = float(best - margins[p_action])
    t_regret = float(best - margins[t_action])
    p_selected = float(p_action in optimal)
    t_selected = float(t_action in optimal)
    p_downgrade = float(_wdl(margins[p_action]) < best_wdl)
    t_downgrade = float(_wdl(margins[t_action]) < best_wdl)
    return {
        "p_optimal_mass": p_mass,
        "t512_optimal_mass": t_mass,
        "optimal_mass_delta": t_mass - p_mass,
        "p_top_prior_regret": p_regret,
        "t512_top_prior_regret": t_regret,
        "top_prior_regret_delta": t_regret - p_regret,
        "p_optimal_selection": p_selected,
        "t512_optimal_selection": t_selected,
        "optimal_selection_delta": t_selected - p_selected,
        "p_wdl_downgrade": p_downgrade,
        "t512_wdl_downgrade": t_downgrade,
        "wdl_downgrade_delta": t_downgrade - p_downgrade,
    }


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize complete per-opening paired results deterministically."""
    if len(rows) != 512 or len({int(row["opening_index"]) for row in rows}) != 512:
        raise ValueError("analysis_requires_512_unique_openings")
    metrics = [row["metrics"] for row in rows]
    fields = (
        "optimal_mass_delta",
        "top_prior_regret_delta",
        "optimal_selection_delta",
        "wdl_downgrade_delta",
    )
    point = {field: sum(float(row[field]) for row in metrics) / 512 for field in fields}
    arm_fields = (
        "p_optimal_mass",
        "t512_optimal_mass",
        "p_top_prior_regret",
        "t512_top_prior_regret",
        "p_optimal_selection",
        "t512_optimal_selection",
        "p_wdl_downgrade",
        "t512_wdl_downgrade",
    )
    arm_means = {
        field: sum(float(row[field]) for row in metrics) / 512 for field in arm_fields
    }
    rng = random.Random(BOOTSTRAP_SEED)
    bootstrap = {field: [] for field in fields}
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [metrics[rng.randrange(512)] for _ in range(512)]
        for field in fields:
            bootstrap[field].append(sum(float(row[field]) for row in sample) / 512)
    intervals = {
        field: [_quantile(bootstrap[field], q) for q in QUANTILES] for field in fields
    }
    p_regret = sum(float(row["p_top_prior_regret"]) for row in metrics) / 512
    t_regret = sum(float(row["t512_top_prior_regret"]) for row in metrics) / 512
    primary = point["optimal_mass_delta"]
    lower = intervals["optimal_mass_delta"][0]
    keep = primary > 0.0 and lower > 0.0 and t_regret <= p_regret
    return {
        "roots": 512,
        "paired_point_estimates": point,
        "arm_point_estimates": arm_means,
        "paired_bootstrap_95_linear": intervals,
        "mean_p_top_prior_exact_regret": p_regret,
        "mean_t512_top_prior_exact_regret": t_regret,
        "bootstrap": {
            "cluster": "opening_index",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "rng": "random.Random; randrange with replacement",
            "quantiles": list(QUANTILES),
            "interpolation": "linear",
        },
        "decision": "may_propose_separately_frozen_search_diagnostic"
        if keep
        else "do_not_propose_search_followup",
        "model_promotion": False,
        "generalization_claim": False,
    }
