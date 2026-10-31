"""Exact root-action handoff for positions inside the tablebase domain."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.kalah_rules import KalahGame


class RootEvaluator(Protocol):
    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]: ...


class ExactRootCoverageGap(RuntimeError):
    """An eligible root could not be evaluated exactly for every legal action."""


@dataclass(frozen=True)
class ExactRootDecision:
    selected_move: int
    legal_moves: list[int]
    action_margins: dict[int, int]
    optimal_moves: list[int]
    network_priors: list[float]
    active_pit_stones: int
    root_player: int
    root_latency_ms: float
    solver_calls: int

    @property
    def wdl(self) -> dict[int, str]:
        return {
            move: "win" if margin > 0 else "draw" if margin == 0 else "loss"
            for move, margin in self.action_margins.items()
        }


def exact_root_profile_fields(
    threshold: int | None, *, solver_identity: str | None = None
) -> dict[str, object]:
    """Semantic profile fields.  Leaf-value replacement remains explicitly disabled."""
    return {
        "exact_root_solve_enabled": threshold is not None,
        "exact_root_solve_threshold": threshold,
        "exact_root_solver": (
            None
            if threshold is None
            else solver_identity
            or f"{EndgameTablebase.__module__}.{EndgameTablebase.__qualname__}"
        ),
        "exact_root_solver_max_solved_seeds": (
            None if threshold is None else EndgameTablebase.MAX_SOLVED_SEEDS
        ),
        "exact_root_objective": "final_score_margin",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
        "exact_leaf_solve_mode": "disabled",
    }


def exact_root_decision(
    game: KalahGame,
    evaluator: RootEvaluator,
    *,
    tablebase: Any,
    threshold: int | None,
) -> ExactRootDecision | None:
    """Return an exact root decision, or ``None`` when the root is ineligible.

    The evaluator is deliberately not called for ineligible roots, preserving the
    normal PUCT execution path exactly.  It is called once for an eligible root
    solely to resolve game-theoretically equivalent actions.
    """
    if threshold is None:
        return None
    if threshold < 0 or threshold > EndgameTablebase.MAX_SOLVED_SEEDS:
        raise ValueError(
            "exact_root_solve_threshold must be within the tablebase complete domain"
        )
    active_pit_stones = int(sum(game.pits))
    if active_pit_stones > threshold:
        return None

    legal_moves = [int(move) for move in game.possible_moves()]
    if not legal_moves:
        raise ExactRootCoverageGap(
            "exact_root_coverage_gap: eligible root has no legal moves"
        )

    started = time.perf_counter()
    root_player = int(game.current_player)
    try:
        native_action_margins = getattr(tablebase, "root_action_margins", None)
        if callable(native_action_margins):
            raw_action_margins: Any = native_action_margins(game, root_player)
            action_margins = {
                int(move): int(margin) for move, margin in raw_action_margins.items()
            }
        else:
            action_margins = {}
            for move in legal_moves:
                child = game.clone()
                if not child.move(child.pit_index(move)):
                    raise ExactRootCoverageGap(
                        f"exact_root_coverage_gap: could not apply legal root move {move}"
                    )
                margin = tablebase.final_margin(child, root_player)
                if margin is None:
                    raise ExactRootCoverageGap(
                        f"exact_root_coverage_gap: missing final margin for move {move}"
                    )
                action_margins[move] = int(margin)
    except ExactRootCoverageGap:
        raise
    except Exception as exc:
        raise ExactRootCoverageGap("exact_root_coverage_gap") from exc
    if set(action_margins) != set(legal_moves):
        raise ExactRootCoverageGap(
            "exact_root_coverage_gap: incomplete action coverage"
        )

    priors, _value = evaluator.evaluate(game)
    legal_prior = {move: float(priors[move]) for move in legal_moves}
    best_margin = max(action_margins.values())
    optimal_moves = sorted(
        move for move, margin in action_margins.items() if margin == best_margin
    )
    selected_move = max(optimal_moves, key=lambda move: (legal_prior[move], -move))
    return ExactRootDecision(
        selected_move=int(selected_move),
        legal_moves=legal_moves,
        action_margins=action_margins,
        optimal_moves=optimal_moves,
        network_priors=[float(value) for value in priors.tolist()],
        active_pit_stones=active_pit_stones,
        root_player=root_player,
        root_latency_ms=(time.perf_counter() - started) * 1000.0,
        solver_calls=1
        if callable(getattr(tablebase, "root_action_margins", None))
        else len(legal_moves),
    )
