"""Lagged-parent root-Q PUCT evaluation without changing stored tree statistics."""

from __future__ import annotations

import random
import hashlib
import json
from collections import Counter
from typing import Any

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import Evaluator, Node, PUCT


def _state_hash(game: KalahGame) -> str:
    state = json.dumps(
        game.to_state(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(state.encode("utf-8")).hexdigest()


def _root_snapshot(root: Node) -> dict[int, dict[str, float | int]]:
    return {
        int(move): {"visits": int(child.visit_count), "q_value": float(child.q_value)}
        for move, child in root.children.items()
    }


def run_shadow_root_q_search(
    game: KalahGame,
    *,
    main_evaluator: Evaluator,
    shadow_evaluator: Evaluator,
    simulations: int,
    c_puct: float,
    seed: int,
    fpu_mode: str = "zero",
    reuse_subtree: bool = False,
    normalize_values: bool = False,
    root_policy_mode: str = "deterministic",
    tactical_root_bias: float = 0.0,
    root_temperature: float = 0.0,
    shadow_q_weight: float = 1.0,
    record_selection_q_direction: bool = False,
    record_selection_trace: bool = False,
) -> tuple[np.ndarray, Node, dict[str, Any]]:
    """Run a parent shadow then a main root-only, pre-simulation-Q search.

    Snapshot ``t`` is captured before shadow simulation ``t`` and is the sole
    reference available to main simulation ``t``. Thus it contains only shadow
    outcomes 1 through t-1; no main information can reach the shadow tree.
    """
    if not 0.0 <= shadow_q_weight <= 1.0:
        raise ValueError("shadow_q_weight must be in [0.0, 1.0]")

    snapshots: dict[int, dict[int, dict[str, float | int]]] = {}
    root_hash = _state_hash(game)

    def capture(simulation: int, root: Node) -> None:
        snapshots[simulation] = _root_snapshot(root)

    common = dict(
        simulations=simulations,
        c_puct=c_puct,
        fpu_mode=fpu_mode,
        reuse_subtree=reuse_subtree,
        normalize_values=normalize_values,
        root_policy_mode=root_policy_mode,
        tactical_root_bias=tactical_root_bias,
        root_temperature=root_temperature,
    )
    shadow = PUCT(
        evaluator=shadow_evaluator,
        rng=random.Random(seed),
        pre_simulation_hook=capture,
        **common,
    )
    shadow.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    uses: list[dict[str, float | int]] = []
    skips: Counter[str] = Counter()

    def override(
        simulation: int, state_hash: str, move: int, raw_q: float, visits: int
    ) -> float | None:
        if state_hash != root_hash:
            return None
        reference = snapshots.get(simulation, {}).get(move)
        if visits <= 0:
            skips["main_unvisited"] += 1
            return None
        if reference is None or int(reference["visits"]) <= 0:
            skips["shadow_unvisited"] += 1
            return None
        parent_q = float(reference["q_value"])
        blended_q = (1.0 - shadow_q_weight) * raw_q + shadow_q_weight * parent_q
        uses.append(
            {
                "simulation": simulation,
                "move": move,
                "candidate_q": raw_q,
                "parent_q": parent_q,
                "blended_q": blended_q,
            }
        )
        return blended_q

    trace: list[dict[str, Any]] | None = [] if record_selection_trace else None
    main = PUCT(
        evaluator=main_evaluator,
        rng=random.Random(seed),
        selection_q_override=override,
        selection_trace=trace,
        **common,
    )
    visits, root = main.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    shadow_summary, main_summary = shadow.root_summary(), main.root_summary()
    shadow_q = {row["move"]: row["q_value"] for row in shadow_summary["child_stats"]}
    main_q = {row["move"]: row["q_value"] for row in main_summary["child_stats"]}
    actions = sorted(set(shadow_q) | set(main_q))
    selected = main_summary["selected_move"]
    return (
        visits,
        root,
        {
            "shadow_summary": shadow_summary,
            "main_summary": main_summary,
            "shadow_q_weight": shadow_q_weight,
            "shadow_pre_simulation_snapshots": len(snapshots),
            "no_future_information": all(
                int(use["simulation"]) in snapshots for use in uses
            ),
            "selection_q_uses": len(uses),
            "selection_q_direction_telemetry": uses
            if record_selection_q_direction
            else [],
            "selection_trace": trace if trace is not None else [],
            "selection_q_skips": dict(skips),
            "selected_child_q_synchronized": any(
                int(use["move"]) == selected
                and float(use["candidate_q"]) == float(use["parent_q"])
                for use in uses
            ),
            "shadow_main_root_q_l1": float(
                sum(abs(shadow_q.get(a, 0.0) - main_q.get(a, 0.0)) for a in actions)
            ),
            "root_q_rank_disagreement": sorted(
                actions, key=lambda a: (-shadow_q.get(a, 0.0), a)
            )
            != sorted(actions, key=lambda a: (-main_q.get(a, 0.0), a)),
            "best_q_action_disagreement": max(
                actions, key=lambda a: (shadow_q.get(a, 0.0), -a)
            )
            != max(actions, key=lambda a: (main_q.get(a, 0.0), -a)),
        },
    )
