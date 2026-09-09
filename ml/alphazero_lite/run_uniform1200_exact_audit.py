#!/usr/bin/env python3
"""Post-training exact audit for the uniform-1200 confirmation replay only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.exact_kalah_solver import (
    ExactKalahSolver,
    ExactState,
    SearchTimeout,
)
from ml.alphazero_lite.run_exact_teacher_training_ablation import (
    decode_kalah_v3_base_state,
)


def sample(rows: list[dict], opening: bool) -> list[dict]:
    unique = {
        json.dumps(r["state"], separators=(",", ":")): r
        for r in rows
        if (int(r["move_index"]) < 8) == opening
    }
    return [
        unique[key]
        for key in sorted(unique, key=lambda x: hashlib.sha256(x.encode()).hexdigest())[
            :128
        ]
    ]


def audit(rows: list[dict], opening: bool, timeout: float) -> dict:
    solver = ExactKalahSolver(cache_enabled=True)
    solved = top = single_n = single_ok = 0
    mass: list[float] = []
    ce: list[float] = []
    try:
        for row in sample(rows, opening):
            state = ExactState.from_game_state(decode_kalah_v3_base_state(row["state"]))
            try:
                margins = solver.action_margins(state, time_limit_seconds=timeout)
            except SearchTimeout:
                continue
            optimal_value = (
                max(margins.values())
                if state.current_player == 0
                else min(margins.values())
            )
            optimal = {
                move for move, value in margins.items() if value == optimal_value
            }
            policy = np.asarray(row["policy"], dtype=float)
            policy /= policy.sum()
            best = int(np.argmax(policy))
            solved += 1
            top += best in optimal
            optimal_mass = float(policy[list(optimal)].sum())
            mass.append(optimal_mass)
            ce.append(float(-np.log(max(optimal_mass / len(optimal), 1e-12))))
            if len(optimal) == 1:
                single_n += 1
                single_ok += best in optimal
    finally:
        solver.close()
    return {
        "sampled_unique_states": len(sample(rows, opening)),
        "solved": solved,
        "top_action_in_exact_optimal_set": top / solved if solved else None,
        "exact_optimal_probability_mass": float(np.mean(mass)) if mass else None,
        "single_optimum_agreement": single_ok / single_n if single_n else None,
        "exact_policy_cross_entropy": float(np.mean(ce)) if ce else None,
        "value_mae": None,
        "labels_used_for_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1.0)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.replay.read_text().splitlines() if line]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "opening": audit(rows, True, args.timeout),
                "midgame": audit(rows, False, args.timeout),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
