"""Frozen-statistics PUCT prior-perturbation algebra."""

from __future__ import annotations

from typing import Any

import numpy as np


def legal_policy(policy: np.ndarray, legal_moves: list[int]) -> np.ndarray:
    """Mask and normalize a policy exactly as PUCT expansion does."""
    masked = np.zeros(6, dtype=np.float32)
    masked[legal_moves] = np.asarray(policy, dtype=np.float32)[legal_moves]
    total = float(masked.sum())
    if total <= 0.0:
        masked[legal_moves] = 1.0 / len(legal_moves)
    else:
        masked /= total
    return masked


def decision_sensitivity(
    decision: dict[str, Any], candidate_policy: np.ndarray, *, c_puct: float
) -> dict[str, Any]:
    """Score one actual PUCT decision while holding Q and visits fixed."""
    children = {int(entry["move"]): entry for entry in decision["children"]}
    selected = int(decision["chosen_move"])
    legal_moves = [int(move) for move in decision["legal_moves"]]
    parent_visits = max(1, int(decision["parent_visit_count"]))
    candidate = legal_policy(candidate_policy, legal_moves)
    deltas = {
        move: float(
            c_puct
            * (float(candidate[move]) - float(children[move]["prior"]))
            * np.sqrt(parent_visits)
            / (1 + int(children[move]["visit_count"]))
        )
        for move in legal_moves
    }
    selected_score = float(children[selected]["selection_score"])
    competitors = []
    for move in legal_moves:
        if move == selected:
            continue
        margin = selected_score - float(children[move]["selection_score"])
        pressure = max(0.0, deltas[move] - deltas[selected]) / max(margin, 1e-12)
        flip_excess = (deltas[move] - deltas[selected]) - margin
        competitors.append(
            {
                "move": move,
                "margin": float(margin),
                "pressure_ratio": float(pressure),
                "flip_excess": float(flip_excess),
            }
        )
    if not competitors:
        return {
            "selected_move": selected,
            "max_pressure_ratio": 0.0,
            "max_flip_excess": 0.0,
            "counterfactual_flip": False,
            "counterfactual_move": selected,
            "winner_runner_up_margin": float("inf"),
            "delta_u": deltas,
        }
    best_pressure = max(competitors, key=lambda item: item["pressure_ratio"])
    best_excess = max(competitors, key=lambda item: item["flip_excess"])
    return {
        "selected_move": selected,
        "max_pressure_ratio": float(best_pressure["pressure_ratio"]),
        "max_flip_excess": float(best_excess["flip_excess"]),
        "counterfactual_flip": bool(best_excess["flip_excess"] > 0.0),
        "counterfactual_move": int(best_excess["move"]),
        "winner_runner_up_margin": float(min(item["margin"] for item in competitors)),
        "delta_u": deltas,
    }
