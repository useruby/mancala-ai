"""Audit Python and native exact-root action margins on a fixed corpus."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.exact_root_decision import exact_root_decision
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.native_exact_root_tablebase import NativeExactRootTablebase
from ml.alphazero_lite.runtime_search_policy import load_runtime_search_policy


class FixedPriorEvaluator:
    def evaluate(self, game: KalahGame):
        del game
        # Deliberately includes ties, resolved by lowest move index.
        return np.asarray([0.30, 0.30, 0.15, 0.10, 0.10, 0.05]), 0.0


def corpus() -> list[dict]:
    """Deterministic synthetic legal-root corpus spanning every 1--16 tier."""
    states = []
    rng = random.Random(380)
    for active in range(1, 17):
        for player in (0, 1):
            pits = [0] * 12
            # Keep each side nonempty so both turns exercise legal root actions.
            pits[player * 6] = 1
            remaining = active - 1
            while remaining:
                index = rng.randrange(12)
                pits[index] += 1
                remaining -= 1
            states.append(
                {
                    "player_pits": pits[:6],
                    "opponent_pits": pits[6:],
                    "player_store": (48 - active) // 2,
                    "opponent_store": 48 - active - ((48 - active) // 2),
                    "current_player": player,
                }
            )
    return states


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=Path("model-artifact/current"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    policy = load_runtime_search_policy(args.artifact)
    if policy is None:
        raise RuntimeError("arena_runtime_search_policy_mismatch")
    python_tablebase = EndgameTablebase()
    evaluator = FixedPriorEvaluator()
    rows = []
    mismatches = {"margins": [], "optimal_moves": [], "selected_move": []}
    with NativeExactRootTablebase(
        policy["native_probe"]["resolved_path"],
        policy["tablebase"]["resolved_path"],
        warm_on_start=True,
    ) as native_tablebase:
        for state in corpus():
            game = KalahGame.from_state(state)
            python = exact_root_decision(
                game, evaluator, tablebase=python_tablebase, threshold=16
            )
            native = exact_root_decision(
                game, evaluator, tablebase=native_tablebase, threshold=16
            )
            assert python is not None and native is not None
            row = {
                "state": state,
                "active_pit_stones": sum(game.pits),
                "root_player": game.current_player,
                "python": {
                    "action_margins": python.action_margins,
                    "optimal_moves": python.optimal_moves,
                    "selected_move": python.selected_move,
                },
                "native": {
                    "action_margins": native.action_margins,
                    "optimal_moves": native.optimal_moves,
                    "selected_move": native.selected_move,
                },
            }
            rows.append(row)
            for name, left, right in (
                ("margins", python.action_margins, native.action_margins),
                ("optimal_moves", python.optimal_moves, native.optimal_moves),
                ("selected_move", python.selected_move, native.selected_move),
            ):
                if left != right:
                    mismatches[name].append(row)
    result = {
        "schema": "exact_root_backend_parity_audit_v1",
        "states_tested": len(rows),
        "runtime_policy": policy,
        "mismatch_counts": {name: len(values) for name, values in mismatches.items()},
        "classification": "exact_root_backend_semantic_mismatch"
        if any(mismatches.values())
        else "exact_root_backend_semantic_parity_confirmed",
        "states": rows,
        "mismatches": mismatches,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 1 if any(mismatches.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
