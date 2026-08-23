"""Independent post-search root-action verification for deterministic PUCT."""

from __future__ import annotations

import hashlib
import random
from typing import TYPE_CHECKING, Any

from ml.alphazero_lite.kalah_rules import KalahGame

if TYPE_CHECKING:
    from ml.alphazero_lite.self_play import Evaluator


VERIFICATION_SEED_NAMESPACE = "azlite-independent-root-action-verification-v1"


def verification_seed(
    root_hash: str, action: int, budget: int, *, mode: str = "full"
) -> int:
    """Derive an RNG seed that is independent of the parent search stream."""
    suffix = "" if mode == "full" else f":{mode}"
    material = f"{VERIFICATION_SEED_NAMESPACE}:{root_hash}:{action}:{budget}{suffix}"
    return int(hashlib.sha256(material.encode("ascii")).hexdigest()[:16], 16)


def root_perspective_value(
    child_value: float, *, root_player: int, child_player: int
) -> float:
    """Convert a child-root value by player identity, including extra turns."""
    return float(child_value) if child_player == root_player else -float(child_value)


def verify_root_actions(
    game: KalahGame,
    *,
    evaluator: "Evaluator",
    root_hash: str,
    budget: int,
    ablation_mode: str = "full",
) -> dict[int, dict[str, Any]]:
    """Score every legal root action using a wholly fresh post-action PUCT tree."""
    root_player = game.current_player
    results: dict[int, dict[str, Any]] = {}
    for action in game.possible_moves():
        child_game = game.clone()
        if not child_game.move(child_game.pit_index(action)):
            raise ValueError(f"legal root action {action} could not be applied")
        if child_game.over():
            exact_value = (
                0.0
                if child_game.winner is None
                else 1.0
                if child_game.winner == child_game.current_player
                else -1.0
            )
        else:
            exact_value = None
        if exact_value is not None:
            # terminal_value is already from the post-action current player's view.
            child_value = float(exact_value)
            simulations = 0
            terminal = True
        else:
            # Importing PUCT lazily keeps the perspective-only fixture independent.
            from ml.alphazero_lite.self_play import PUCT

            search = PUCT(
                evaluator=evaluator,
                simulations=budget,
                c_puct=1.25,
                rng=random.Random(
                    verification_seed(root_hash, action, budget, mode=ablation_mode)
                ),
                fpu_mode="zero",
                reuse_subtree=False,
                normalize_values=False,
                root_policy_mode="deterministic",
                root_temperature=0.0,
                tactical_root_bias=0.0,
                ablation_mode=ablation_mode,
            )
            search.run(child_game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
            child_value = float(search.root_summary()["root_q_value"])
            simulations = int(budget)
            terminal = False
        results[int(action)] = {
            "verification_q": root_perspective_value(
                child_value,
                root_player=root_player,
                child_player=child_game.current_player,
            ),
            "child_root_q": child_value,
            "child_player": int(child_game.current_player),
            "terminal": terminal,
            "simulations": simulations,
            "seed": verification_seed(root_hash, action, budget, mode=ablation_mode),
            "ablation_mode": ablation_mode,
        }
    return results


def select_verified_move(
    verification: dict[int, dict[str, Any]], *, main_summary: dict[str, Any]
) -> int:
    """Apply the preregistered verifier-Q then normal-root tie order."""
    main = {int(row["move"]): row for row in main_summary["child_stats"]}
    selection = main_summary["selection_breakdown"]["moves"]
    priors = {int(row["move"]): float(row["prior"]) for row in selection}
    return max(
        verification,
        key=lambda action: (
            float(verification[action]["verification_q"]),
            int(main[action]["visits"]),
            float(main[action]["q_value"]),
            priors[action],
            -int(action),
        ),
    )
