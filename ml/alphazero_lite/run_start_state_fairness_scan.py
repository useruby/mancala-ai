#!/usr/bin/env python3
"""Scan candidate start states for first-player fairness diagnostics."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint
from ml.alphazero_lite.self_play import (
    CheckpointEvaluator,
    PUCT,
    RANDOM_SYMMETRIC_TOTAL24_START_STATE_MODE,
    SearchOptions,
    build_eval_search_options,
    sample_random_symmetric_distribution,
    search_seed_for_classic_mcts,
    state_hash,
)
from ml.alphazero_lite.kalah_rules import KalahGame


DEFAULT_ARTIFACT_PATH = "storage/ai/alphazero_lite/current"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[RANDOM_SYMMETRIC_TOTAL24_START_STATE_MODE],
        default=RANDOM_SYMMETRIC_TOTAL24_START_STATE_MODE,
    )
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument(
        "--teacher", choices=["classic_mcts", "puct"], default="classic_mcts"
    )
    parser.add_argument("--simulations", type=int, default=1200)
    parser.add_argument("--seeds", default="11,23,37")
    parser.add_argument(
        "--artifact-path",
        default=DEFAULT_ARTIFACT_PATH,
        help="Artifact directory containing weights.json or checkpoint.npz for PUCT scans",
    )
    parser.add_argument("--input-encoding", default="kalah_v3")
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--fpu-mode", default="zero")
    parser.add_argument("--normalize-values", action="store_true")
    parser.add_argument("--reuse-subtree", action="store_true")
    parser.add_argument("--root-policy-mode", default="deterministic")
    parser.add_argument("--tactical-root-bias", type=float, default=0.1)
    parser.add_argument("--max-abs-margin", type=float, default=0.1)
    parser.add_argument("--max-seed-std", type=float, default=0.08)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def parse_seed_list(raw_seeds: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw_seeds.split(",") if value.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")
    return seeds


def mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def population_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / float(len(values))) ** 0.5


def current_player_value_to_player0(value: float, *, current_player: int) -> float:
    return float(value) if int(current_player) == 0 else -float(value)


def resolve_checkpoint_path(artifact_path: Path, *, temp_dir: Path) -> Path:
    checkpoint_candidates = [
        artifact_path / "checkpoint.npz",
        artifact_path / "model.npz",
    ]
    for candidate in checkpoint_candidates:
        if candidate.exists():
            return candidate
    weights_path = artifact_path / "weights.json"
    if weights_path.exists():
        return materialize_weights_json_checkpoint(
            weights_path=weights_path,
            out_path=temp_dir / "materialized_current_checkpoint.npz",
        )
    raise SystemExit(
        f"artifact path must contain checkpoint.npz, model.npz, or weights.json: {artifact_path}"
    )


def build_state(distribution: list[int], *, current_player: int) -> dict[str, object]:
    return {
        "player_pits": [int(value) for value in distribution],
        "opponent_pits": [int(value) for value in distribution],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": int(current_player),
    }


def estimate_classic_mcts(
    state: dict[str, object], *, simulations: int, seed: int
) -> dict[str, object]:
    game = KalahGame.from_state(state)
    root = ClassicMCTS(game, simulations=int(simulations), seed=int(seed)).search_root()
    child_stats = []
    for move, child in sorted(root.children.items()):
        child_stats.append(
            {
                "move": int(move),
                "visits": int(child.visits),
                "q_value": (2.0 * float(child.wins / child.visits)) - 1.0
                if child.visits
                else 0.0,
            }
        )
    selected_move = None
    if child_stats:
        selected_move = max(
            child_stats,
            key=lambda entry: (
                int(entry["visits"]),
                float(entry["q_value"]),
                -int(entry["move"]),
            ),
        )["move"]
    value = (2.0 * float(root.wins / root.visits)) - 1.0 if root.visits else 0.0
    return {
        "selected_move": selected_move,
        "value_current_player": float(value),
        "child_stats": child_stats,
    }


def estimate_puct(
    state: dict[str, object],
    *,
    evaluator: CheckpointEvaluator,
    simulations: int,
    seed: int,
    c_puct: float,
    search_options: SearchOptions,
) -> dict[str, object]:
    game = KalahGame.from_state(state)
    search = PUCT(
        evaluator=evaluator,
        simulations=int(simulations),
        c_puct=float(c_puct),
        rng=__import__("random").Random(int(seed)),
        fpu_mode=str(search_options["fpu_mode"]),
        reuse_subtree=bool(search_options["reuse_subtree"]),
        normalize_values=bool(search_options["normalize_values"]),
        root_policy_mode=str(search_options["root_policy_mode"]),
        tactical_root_bias=float(search_options["tactical_root_bias"]),
        value_trust_schedule=search_options.get("value_trust_schedule"),
    )
    legal_moves = game.possible_moves()
    _visits, root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    summary = search.root_summary()
    selected_move = (
        None if not legal_moves else search.select_root_move(root, legal_moves)
    )
    return {
        "selected_move": None if selected_move is None else int(selected_move),
        "value_current_player": float(root.q_value),
        "child_stats": list(summary.get("child_stats") or []),
    }


def current_player_has_legal_moves(
    state: dict[str, object], *, current_player: int
) -> bool:
    local_state = dict(state)
    local_state["current_player"] = int(current_player)
    return bool(KalahGame.from_state(local_state).possible_moves())


def scan_distribution(
    distribution: list[int],
    *,
    teacher: str,
    simulations: int,
    seeds: list[int],
    evaluator: CheckpointEvaluator | None,
    c_puct: float,
    search_options: SearchOptions,
) -> list[dict[str, object]]:
    per_player_results: dict[int, dict[str, object]] = {}
    for current_player in (0, 1):
        state = build_state(distribution, current_player=current_player)
        seed_results = []
        for seed in seeds:
            position_seed = search_seed_for_classic_mcts(seed, 0, current_player)
            if teacher == "classic_mcts":
                estimate = estimate_classic_mcts(
                    state,
                    simulations=simulations,
                    seed=position_seed,
                )
            else:
                if evaluator is None:
                    raise ValueError("PUCT scan requires an evaluator")
                estimate = estimate_puct(
                    state,
                    evaluator=evaluator,
                    simulations=simulations,
                    seed=position_seed,
                    c_puct=c_puct,
                    search_options=search_options,
                )
            seed_results.append(
                {
                    "seed": int(seed),
                    "selected_move": estimate["selected_move"],
                    "value_current_player": float(estimate["value_current_player"]),
                }
            )
        values_current = [
            float(entry["value_current_player"]) for entry in seed_results
        ]
        values_player0 = [
            current_player_value_to_player0(value, current_player=current_player)
            for value in values_current
        ]
        seed_std = population_std(values_current)
        per_player_results[current_player] = {
            "state": state,
            "seed_results": seed_results,
            "child_stats": estimate.get("child_stats") if seed_results else [],
            "mean_value_current_player": mean(values_current),
            "mean_value_player0": mean(values_player0),
            "seed_std": seed_std,
            "selected_move": seed_results[0]["selected_move"] if seed_results else None,
        }

    first_player_margin_estimate = 0.5 * (
        float(per_player_results[0]["mean_value_player0"])
        - float(per_player_results[1]["mean_value_player0"])
    )
    stability = max(
        float(per_player_results[0]["seed_std"]),
        float(per_player_results[1]["seed_std"]),
    )
    rows = []
    for current_player in (0, 1):
        result = per_player_results[current_player]
        state = result["state"]
        rows.append(
            {
                "player_pits": list(state["player_pits"]),
                "opponent_pits": list(state["opponent_pits"]),
                "current_player": int(current_player),
                "total_stones": int(
                    sum(state["player_pits"]) + sum(state["opponent_pits"])
                ),
                "start_state_hash": state_hash(state),
                "teacher": teacher,
                "simulations": int(simulations),
                "selected_move": result["selected_move"],
                "estimated_value_current_player": float(
                    result["mean_value_current_player"]
                ),
                "estimated_value_player0": float(result["mean_value_player0"]),
                "first_player_margin_estimate": float(first_player_margin_estimate),
                "stability_across_seeds": {
                    "value_std": float(result["seed_std"]),
                    "paired_margin_std_upper_bound": float(stability),
                    "seed_results": result["seed_results"],
                },
                "legal_non_terminal_state": bool(
                    KalahGame.from_state(state).possible_moves()
                ),
                "side0_has_legal_moves": bool(
                    current_player_has_legal_moves(state, current_player=0)
                ),
                "side1_has_legal_moves": bool(
                    current_player_has_legal_moves(state, current_player=1)
                ),
                "child_stats": result["child_stats"],
            }
        )
    return rows


def accepted_for_pool(
    row: dict[str, object], *, max_abs_margin: float, max_seed_std: float
) -> bool:
    if not bool(row.get("legal_non_terminal_state", False)):
        return False
    if not bool(row.get("side0_has_legal_moves", False)):
        return False
    if not bool(row.get("side1_has_legal_moves", False)):
        return False
    if abs(float(row.get("first_player_margin_estimate", 0.0))) > float(max_abs_margin):
        return False
    stability = row.get("stability_across_seeds") or {}
    return float(stability.get("value_std", 0.0)) <= float(max_seed_std)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_list(args.seeds)
    search_options = build_eval_search_options(
        fpu_mode=args.fpu_mode,
        reuse_subtree=bool(args.reuse_subtree),
        normalize_values=bool(args.normalize_values),
        root_policy_mode=args.root_policy_mode,
        tactical_root_bias=float(args.tactical_root_bias),
    )

    with tempfile.TemporaryDirectory(prefix="azlite-start-state-scan-") as tmp:
        evaluator = None
        if args.teacher == "puct":
            checkpoint_path = resolve_checkpoint_path(
                Path(args.artifact_path), temp_dir=Path(tmp)
            )
            evaluator = CheckpointEvaluator(
                checkpoint_path,
                input_encoding=args.input_encoding,
            )

        rows: list[dict[str, object]] = []
        import random

        rng = random.Random(0)
        for _sample_index in range(int(args.samples)):
            distribution = sample_random_symmetric_distribution(rng)
            sampled_rows = scan_distribution(
                distribution,
                teacher=args.teacher,
                simulations=int(args.simulations),
                seeds=seeds,
                evaluator=evaluator,
                c_puct=float(args.c_puct),
                search_options=search_options,
            )
            for row in sampled_rows:
                row["accepted_for_pool"] = accepted_for_pool(
                    row,
                    max_abs_margin=float(args.max_abs_margin),
                    max_seed_std=float(args.max_seed_std),
                )
                rows.append(row)

        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    accepted_count = sum(1 for row in rows if bool(row.get("accepted_for_pool")))
    print(f"wrote {len(rows)} fairness rows to {out_path} ({accepted_count} accepted)")


if __name__ == "__main__":
    main()
