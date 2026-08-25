"""Pure helpers for the fixed root-Q directional trust-region protocol."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np
import torch


LAMBDAS = (1.0, 0.5, 0.25, 0.125, 0.0)


def q_direction_error(left: np.ndarray, right: np.ndarray) -> float:
    """Centered Q-direction error with deterministic degenerate behavior."""
    left_centered = np.asarray(left, dtype=np.float64) - np.mean(left)
    right_centered = np.asarray(right, dtype=np.float64) - np.mean(right)
    left_norm = float(np.linalg.norm(left_centered))
    right_norm = float(np.linalg.norm(right_centered))
    if left_norm <= 1e-12 and right_norm <= 1e-12:
        return 0.0
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 1.0
    cosine = np.clip(
        np.dot(left_centered, right_centered) / (left_norm * right_norm), -1.0, 1.0
    )
    error = float(1.0 - cosine)
    return 0.0 if abs(error) <= 1e-15 else error


def common_visited_q(
    candidate: dict, reference: dict
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return Q vectors only for actions visited in both root searches."""
    candidate_by_move = {
        int(item["move"]): item
        for item in candidate["child_stats"]
        if item["visits"] > 0
    }
    reference_by_move = {
        int(item["move"]): item
        for item in reference["child_stats"]
        if item["visits"] > 0
    }
    moves = sorted(set(candidate_by_move) & set(reference_by_move))
    return (
        np.asarray([candidate_by_move[move]["q_value"] for move in moves]),
        np.asarray([reference_by_move[move]["q_value"] for move in moves]),
        moves,
    )


def root_q_diagnostics(candidate: dict, reference: dict) -> dict[str, float | int]:
    candidate_q, reference_q, moves = common_visited_q(candidate, reference)
    candidate_visits = {
        int(item["move"]): float(item["visits"]) for item in candidate["child_stats"]
    }
    reference_visits = {
        int(item["move"]): float(item["visits"]) for item in reference["child_stats"]
    }
    visit_moves = sorted(set(candidate_visits) | set(reference_visits))
    candidate_distribution = np.asarray(
        [candidate_visits.get(move, 0.0) for move in visit_moves]
    )
    reference_distribution = np.asarray(
        [reference_visits.get(move, 0.0) for move in visit_moves]
    )
    candidate_distribution /= candidate_distribution.sum()
    reference_distribution /= reference_distribution.sum()
    midpoint = (candidate_distribution + reference_distribution) / 2.0
    safe_midpoint = np.maximum(midpoint, 1e-12)
    visit_js = 0.5 * np.sum(
        candidate_distribution
        * np.log(np.maximum(candidate_distribution, 1e-12) / safe_midpoint)
    )
    visit_js += 0.5 * np.sum(
        reference_distribution
        * np.log(np.maximum(reference_distribution, 1e-12) / safe_midpoint)
    )
    if len(moves) < 2:
        best_disagreement = 0
        top_two_disagreement = 0
        rank_disagreement = 0.0
    else:
        candidate_order = sorted(
            range(len(moves)), key=lambda i: (-candidate_q[i], moves[i])
        )
        reference_order = sorted(
            range(len(moves)), key=lambda i: (-reference_q[i], moves[i])
        )
        best_disagreement = int(candidate_order[0] != reference_order[0])
        top_two_disagreement = int(candidate_order[:2] != reference_order[:2])
        candidate_rank = {index: rank for rank, index in enumerate(candidate_order)}
        reference_rank = {index: rank for rank, index in enumerate(reference_order)}
        rank_disagreement = float(
            np.mean(
                [
                    candidate_rank[index] != reference_rank[index]
                    for index in range(len(moves))
                ]
            )
        )
    centered_l1 = (
        float(
            np.abs(
                (candidate_q - candidate_q.mean()) - (reference_q - reference_q.mean())
            ).sum()
        )
        if len(moves)
        else 0.0
    )
    return {
        "q_direction_error": q_direction_error(candidate_q, reference_q)
        if len(moves)
        else 0.0,
        "best_q_action_disagreement": best_disagreement,
        "top_two_q_order_disagreement": top_two_disagreement,
        "per_action_q_rank_disagreement": rank_disagreement,
        "centered_q_l1": centered_l1,
        "root_move_disagreement": int(
            candidate["selected_move"] != reference["selected_move"]
        ),
        "visit_js": float(visit_js),
        "common_visited_actions": len(moves),
    }


def select_guard_indexes(
    population: list[dict], sensitivity: Iterable[float], count: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Freeze high-sensitivity and disjoint deterministic-random guard sets."""
    values = list(sensitivity)
    ranked = sorted(
        range(len(population)), key=lambda i: (-values[i], population[i]["state_hash"])
    )
    primary = np.asarray(ranked[:count], dtype=np.int64)
    primary_set = set(primary.tolist())
    remaining = [index for index in range(len(population)) if index not in primary_set]
    random_order = sorted(
        remaining,
        key=lambda i: hashlib.sha256(
            f"{seed}:{population[i]['state_hash']}".encode()
        ).hexdigest(),
    )
    secondary = np.asarray(random_order[:count], dtype=np.int64)
    manifest = {
        "schema": "azlite_root_q_trust_region_guard_manifest_v1",
        "seed": seed,
        "primary_count": count,
        "secondary_count": count,
        "primary_indexes": primary.tolist(),
        "secondary_indexes": secondary.tolist(),
        "primary_hashes": [population[i]["state_hash"] for i in primary],
        "secondary_hashes": [population[i]["state_hash"] for i in secondary],
    }
    manifest["sha256_excluding_this_field"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return primary, secondary, manifest


def choose_lambda(errors: dict[float, float], epsilon_q: float) -> float:
    """Return the preregistered largest feasible trial scale."""
    for scale in LAMBDAS:
        if errors[scale] <= epsilon_q + 1e-12:
            return scale
    return 0.0


def adam_proposal_and_restore(
    parameters: list[torch.nn.Parameter], optimizer: torch.optim.Optimizer
) -> list[torch.Tensor]:
    """Advance Adam once and return its proposal while restoring parameters."""
    current = [parameter.detach().clone() for parameter in parameters]
    optimizer.step()
    delta = [
        parameter.detach().clone() - before
        for parameter, before in zip(parameters, current)
    ]
    with torch.no_grad():
        for parameter, before in zip(parameters, current):
            parameter.copy_(before)
    return delta


def apply_delta(
    parameters: list[torch.nn.Parameter], delta: list[torch.Tensor], scale: float
) -> None:
    with torch.no_grad():
        for parameter, update in zip(parameters, delta):
            parameter.add_(update, alpha=float(scale))
