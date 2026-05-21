#!/usr/bin/env python3
"""Parallel MCTS1200 baseline driver with deterministic per-game MCTS seeding."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


EARLY_DEFAULT_SEARCH_OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "visit_count",
    "tactical_root_bias": 0.0,
}
EARLY_SUPPORTED_ROOT_POLICY_MODES = ("deterministic", "visit_count")


def add_early_search_option_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fpu-mode", default=EARLY_DEFAULT_SEARCH_OPTIONS["fpu_mode"])
    parser.add_argument("--reuse-subtree", action="store_true")
    parser.add_argument("--normalize-values", action="store_true")
    parser.add_argument(
        "--root-policy-mode",
        choices=EARLY_SUPPORTED_ROOT_POLICY_MODES,
        default=EARLY_DEFAULT_SEARCH_OPTIONS["root_policy_mode"],
    )
    parser.add_argument(
        "--tactical-root-bias",
        type=float,
        default=EARLY_DEFAULT_SEARCH_OPTIONS["tactical_root_bias"],
    )


def build_stub_search_options(
    args: argparse.Namespace,
) -> dict[str, str | bool | float]:
    return {
        "fpu_mode": args.fpu_mode,
        "reuse_subtree": bool(args.reuse_subtree),
        "normalize_values": bool(args.normalize_values),
        "root_policy_mode": args.root_policy_mode,
        "tactical_root_bias": float(args.tactical_root_bias),
    }


def build_stub_eval_search_options(**kwargs) -> dict[str, str | bool | float]:
    return dict(kwargs)


def build_stub_search_profile(
    *,
    kind: str,
    player_mode: str,
    simulations: int,
    c_puct: float,
    search_options: dict[str, str | bool | float],
    extra_fields: dict[str, str | int | float | bool] | None = None,
) -> dict:
    del c_puct
    profile = {
        "version": "v1",
        "kind": str(kind),
        "player_mode": str(player_mode),
        "classic_mcts_simulations": int(simulations),
    }
    if extra_fields:
        profile.update(extra_fields)
    profile["hash"] = "mcts1200-baseline-stub-profile"
    return profile


if os.environ.get("AZLITE_MCTS1200_BASELINE_STUB") != "1":
    from ml.alphazero_lite.classic_mcts import MCTS
    from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
    from ml.alphazero_lite.kalah_rules import KalahGame
    from ml.alphazero_lite.self_play import (
        add_search_option_args,
        build_eval_search_options,
        build_search_profile,
        search_options_from_args,
    )
else:
    add_search_option_args = add_early_search_option_args
    build_eval_search_options = build_stub_eval_search_options
    build_search_profile = build_stub_search_profile
    search_options_from_args = build_stub_search_options
    MCTS = None
    EndgameTablebase = None
    KalahGame = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenger-path", required=True)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--az-base-simulations", type=int, default=640)
    parser.add_argument("--mcts-simulations", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dynamic-budget-enabled", action="store_true")
    parser.add_argument("--dynamic-budget-probe-simulations", type=int, default=0)
    parser.add_argument("--dynamic-budget-min-simulations", type=int)
    parser.add_argument("--dynamic-budget-max-simulations", type=int)
    parser.add_argument("--dynamic-budget-entropy-weight", type=float, default=0.8)
    parser.add_argument(
        "--dynamic-budget-low-margin-threshold", type=float, default=0.2
    )
    parser.add_argument("--dynamic-budget-low-margin-weight", type=float, default=1.5)
    parser.add_argument("--dynamic-budget-variance-weight", type=float, default=1.5)
    parser.add_argument("--exact-solve-enabled", action="store_true")
    parser.add_argument("--exact-solve-stone-threshold", type=int)
    add_search_option_args(parser)
    args = parser.parse_args()
    if args.exact_solve_enabled and args.exact_solve_stone_threshold is None:
        parser.error("--exact-solve-enabled requires --exact-solve-stone-threshold")
    if args.exact_solve_stone_threshold is not None and not args.exact_solve_enabled:
        parser.error("--exact-solve-stone-threshold requires --exact-solve-enabled")
    if (
        args.exact_solve_stone_threshold is not None
        and int(args.exact_solve_stone_threshold) < 0
    ):
        parser.error("--exact-solve-stone-threshold must be non-negative")
    if args.exact_solve_stone_threshold is not None and EndgameTablebase is not None:
        args.exact_solve_stone_threshold = min(
            int(args.exact_solve_stone_threshold), EndgameTablebase.MAX_SOLVED_SEEDS
        )
    if args.dynamic_budget_enabled:
        min_simulations = (
            args.mcts_simulations
            if args.dynamic_budget_min_simulations is None
            else int(args.dynamic_budget_min_simulations)
        )
        max_simulations = (
            args.mcts_simulations
            if args.dynamic_budget_max_simulations is None
            else int(args.dynamic_budget_max_simulations)
        )
        probe_simulations = int(args.dynamic_budget_probe_simulations)
        float_knobs = {
            "--dynamic-budget-entropy-weight": float(
                args.dynamic_budget_entropy_weight
            ),
            "--dynamic-budget-low-margin-threshold": float(
                args.dynamic_budget_low_margin_threshold
            ),
            "--dynamic-budget-low-margin-weight": float(
                args.dynamic_budget_low_margin_weight
            ),
            "--dynamic-budget-variance-weight": float(
                args.dynamic_budget_variance_weight
            ),
        }
        if min_simulations < 1:
            parser.error(
                "--dynamic-budget-min-simulations must be >= 1 when dynamic budget is enabled"
            )
        if max_simulations < min_simulations:
            parser.error(
                "--dynamic-budget-max-simulations must be >= --dynamic-budget-min-simulations"
            )
        if probe_simulations < 1:
            parser.error(
                "--dynamic-budget-probe-simulations must be >= 1 when dynamic budget is enabled"
            )
        if probe_simulations >= max_simulations:
            parser.error(
                "--dynamic-budget-probe-simulations must be < --dynamic-budget-max-simulations"
            )
        for name, value in float_knobs.items():
            if not math.isfinite(value):
                parser.error(f"{name} must be finite")
        if not (0.0 <= float_knobs["--dynamic-budget-low-margin-threshold"] <= 1.0):
            parser.error(
                "--dynamic-budget-low-margin-threshold must be between 0 and 1"
            )
        if float_knobs["--dynamic-budget-entropy-weight"] < 0.0:
            parser.error("--dynamic-budget-entropy-weight must be >= 0")
        if float_knobs["--dynamic-budget-low-margin-weight"] < 0.0:
            parser.error("--dynamic-budget-low-margin-weight must be >= 0")
        if float_knobs["--dynamic-budget-variance-weight"] < 0.0:
            parser.error("--dynamic-budget-variance-weight must be >= 0")
    return args


def partition_counts(total: int, workers: int) -> list[int]:
    workers = max(1, workers)
    base = total // workers
    remainder = total % workers
    return [base + (1 if index < remainder else 0) for index in range(workers)]


def partition_starts(counts: list[int]) -> list[int]:
    starts: list[int] = []
    cursor = 0
    for count in counts:
        starts.append(cursor)
        cursor += count
    return starts


def initial_game() -> KalahGame:
    return KalahGame.from_state(
        {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )


def run_stub_worker(
    *, start_index: int, games: int, az_base_simulations: int, mcts_simulations: int
) -> dict:
    az_wins = 0
    mcts_wins = 0
    draws = 0

    for local_index in range(games):
        game_index = start_index + local_index
        outcome = game_index % 5
        if outcome in (0, 4):
            az_wins += 1
        elif outcome == 1:
            draws += 1
        else:
            mcts_wins += 1

    fixed_simulations = float(mcts_simulations)
    dynamic_simulations = max(1.0, round(float(az_base_simulations) * 1.15, 2))
    return {
        "start_index": start_index,
        "games": games,
        "dynamic": {
            "az_wins": az_wins,
            "mcts_wins": mcts_wins,
            "draws": draws,
            "budget_sample_count": games,
            "budget_total_final_simulations": float(games * dynamic_simulations),
            "budget_total_root_latency_ms": float(games * 6.5),
        },
        "fixed": {
            "az_wins": az_wins,
            "mcts_wins": mcts_wins,
            "draws": draws,
            "budget_sample_count": games,
            "budget_total_final_simulations": float(games * fixed_simulations),
            "budget_total_root_latency_ms": float(games * 6.3),
        },
    }


def empty_mode_result(*, games: int, start_index: int) -> dict:
    return {
        "start_index": start_index,
        "games": games,
        "az_wins": 0,
        "mcts_wins": 0,
        "draws": 0,
        "budget_sample_count": 0,
        "budget_total_final_simulations": 0.0,
        "budget_total_root_latency_ms": 0.0,
    }


def head_to_head_role_for(*, game_index: int, player: int) -> str:
    dynamic_player = game_index % 2
    return "dynamic" if player == dynamic_player else "fixed"


def build_classic_mcts(
    *,
    game: KalahGame,
    game_index: int,
    seed: int,
    mcts_simulations: int,
    dynamic_budget_enabled: bool,
    dynamic_budget_probe_simulations: int,
    dynamic_budget_min_simulations: int | None,
    dynamic_budget_max_simulations: int | None,
    dynamic_budget_entropy_weight: float,
    dynamic_budget_low_margin_threshold: float,
    dynamic_budget_low_margin_weight: float,
    dynamic_budget_variance_weight: float,
    endgame_tablebase: EndgameTablebase | None,
    exact_solve_enabled: bool,
    exact_solve_stone_threshold: int | None,
) -> MCTS:
    return MCTS(
        game,
        simulations=mcts_simulations,
        seed=seed + (game_index * 37) + game.current_player,
        dynamic_budget_enabled=dynamic_budget_enabled,
        dynamic_budget_probe_simulations=dynamic_budget_probe_simulations,
        dynamic_budget_min_simulations=dynamic_budget_min_simulations,
        dynamic_budget_max_simulations=dynamic_budget_max_simulations,
        dynamic_budget_entropy_weight=dynamic_budget_entropy_weight,
        dynamic_budget_low_margin_threshold=dynamic_budget_low_margin_threshold,
        dynamic_budget_low_margin_weight=dynamic_budget_low_margin_weight,
        dynamic_budget_variance_weight=dynamic_budget_variance_weight,
        endgame_tablebase=endgame_tablebase,
        exact_solve_enabled=exact_solve_enabled,
        exact_solve_stone_threshold=exact_solve_stone_threshold,
    )


def run_worker(
    *,
    challenger_path: str,
    games: int,
    start_index: int,
    seed: int,
    az_base_simulations: int,
    mcts_simulations: int,
    search_options: dict,
    dynamic_budget_enabled: bool = False,
    dynamic_budget_probe_simulations: int = 0,
    dynamic_budget_min_simulations: int | None = None,
    dynamic_budget_max_simulations: int | None = None,
    dynamic_budget_entropy_weight: float = 0.8,
    dynamic_budget_low_margin_threshold: float = 0.2,
    dynamic_budget_low_margin_weight: float = 1.5,
    dynamic_budget_variance_weight: float = 1.5,
    exact_solve_enabled: bool = False,
    exact_solve_stone_threshold: int | None = None,
) -> dict:
    if os.environ.get("AZLITE_MCTS1200_BASELINE_STUB") == "1":
        return run_stub_worker(
            start_index=start_index,
            games=games,
            az_base_simulations=az_base_simulations,
            mcts_simulations=mcts_simulations,
        )

    del challenger_path, search_options

    tablebase = EndgameTablebase() if exact_solve_enabled else None
    dynamic_result = empty_mode_result(games=games, start_index=start_index)
    fixed_result = empty_mode_result(games=games, start_index=start_index)

    for local_index in range(games):
        game_index = start_index + local_index
        game = initial_game()

        while not game.over():
            role = head_to_head_role_for(
                game_index=game_index, player=game.current_player
            )
            is_dynamic = role == "dynamic"
            search = build_classic_mcts(
                game=game,
                game_index=game_index,
                seed=seed,
                mcts_simulations=mcts_simulations,
                dynamic_budget_enabled=is_dynamic and dynamic_budget_enabled,
                dynamic_budget_probe_simulations=dynamic_budget_probe_simulations
                if is_dynamic
                else 0,
                dynamic_budget_min_simulations=dynamic_budget_min_simulations
                if is_dynamic
                else mcts_simulations,
                dynamic_budget_max_simulations=dynamic_budget_max_simulations
                if is_dynamic
                else mcts_simulations,
                dynamic_budget_entropy_weight=dynamic_budget_entropy_weight,
                dynamic_budget_low_margin_threshold=dynamic_budget_low_margin_threshold,
                dynamic_budget_low_margin_weight=dynamic_budget_low_margin_weight,
                dynamic_budget_variance_weight=dynamic_budget_variance_weight,
                endgame_tablebase=tablebase,
                exact_solve_enabled=exact_solve_enabled,
                exact_solve_stone_threshold=exact_solve_stone_threshold,
            )
            if hasattr(search, "root_summary"):
                summary = search.root_summary()
                budget = summary.get("budget", {})
                relative_move = summary.get("selected_move")
            else:
                budget = {}
                relative_move = search.choose_move()

            target = dynamic_result if is_dynamic else fixed_result
            target["budget_sample_count"] += 1
            target["budget_total_final_simulations"] += float(
                budget.get("final_simulations", 0.0)
            )
            target["budget_total_root_latency_ms"] += float(
                budget.get("root_latency_ms", 0.0)
            )

            if relative_move is None:
                break
            if not game.move(game.pit_index(relative_move)):
                break

        dynamic_player = game_index % 2
        fixed_player = 1 - dynamic_player
        dynamic_score = game.captured_seeds[dynamic_player]
        fixed_score = game.captured_seeds[fixed_player]
        if dynamic_score > fixed_score:
            dynamic_result["az_wins"] += 1
            fixed_result["mcts_wins"] += 1
        elif dynamic_score < fixed_score:
            dynamic_result["mcts_wins"] += 1
            fixed_result["az_wins"] += 1
        else:
            dynamic_result["draws"] += 1
            fixed_result["draws"] += 1

    return {
        "start_index": start_index,
        "games": games,
        "dynamic": dynamic_result,
        "fixed": fixed_result,
    }


def summarize_mode_results(results: list[dict], *, mode: str) -> dict:
    mode_results = []
    for parent_result in results:
        mode_result = dict(
            parent_result.get(mode, empty_mode_result(games=0, start_index=0))
        )
        if "games" not in mode_result:
            inferred_games = int(parent_result.get("games", 0))
            if inferred_games <= 0:
                inferred_games = (
                    int(mode_result.get("az_wins", 0))
                    + int(mode_result.get("mcts_wins", 0))
                    + int(mode_result.get("draws", 0))
                )
            mode_result["games"] = inferred_games
        if "start_index" not in mode_result:
            mode_result["start_index"] = int(parent_result.get("start_index", 0))
        mode_results.append(mode_result)
    games = sum(int(result.get("games", 0)) for result in mode_results)
    az_wins = sum(int(result.get("az_wins", 0)) for result in mode_results)
    mcts_wins = sum(int(result.get("mcts_wins", 0)) for result in mode_results)
    draws = sum(int(result.get("draws", 0)) for result in mode_results)
    final_simulation_samples = sum(
        int(result.get("budget_sample_count", 0)) for result in mode_results
    )
    latency_samples = sum(
        int(result.get("budget_sample_count", 0)) for result in mode_results
    )
    total_final_simulations = sum(
        float(result.get("budget_total_final_simulations", 0.0))
        for result in mode_results
    )
    total_root_latency_ms = sum(
        float(result.get("budget_total_root_latency_ms", 0.0))
        for result in mode_results
    )
    if final_simulation_samples <= 0 and latency_samples <= 0:
        for result in mode_results:
            budget_summary = result.get("budget_summary")
            if not isinstance(budget_summary, dict):
                continue
            weight = int(result.get("games", 0))
            if weight <= 0:
                weight = 1
            if budget_summary.get("mean_final_simulations") is not None:
                total_final_simulations += (
                    float(budget_summary["mean_final_simulations"]) * weight
                )
                final_simulation_samples += weight
            if budget_summary.get("mean_root_latency_ms") is not None:
                total_root_latency_ms += (
                    float(budget_summary["mean_root_latency_ms"]) * weight
                )
                latency_samples += weight

    score = None if games <= 0 else (az_wins + (0.5 * draws)) / float(games)
    return {
        "games": games,
        "az_wins": az_wins,
        "mcts_wins": mcts_wins,
        "draws": draws,
        "score": score,
        "budget_summary": {
            "mean_final_simulations": round(
                total_final_simulations / final_simulation_samples, 2
            )
            if final_simulation_samples > 0
            else None,
            "mean_root_latency_ms": round(total_root_latency_ms / latency_samples, 2)
            if latency_samples > 0
            else None,
        },
    }


def seat_bias_neutralized(results: list[dict]) -> bool:
    dynamic_starts = 0
    dynamic_replies = 0
    for result in results:
        start_index = int(result.get("start_index", 0))
        game_count = int(result.get("games", 0))
        for game_index in range(start_index, start_index + game_count):
            if game_index % 2 == 0:
                dynamic_starts += 1
            else:
                dynamic_replies += 1
    return dynamic_starts == dynamic_replies


def build_report(
    *,
    games: int,
    az_base_simulations: int,
    mcts_simulations: int,
    search_options: dict,
    dynamic_budget_enabled: bool = False,
    dynamic_budget_probe_simulations: int = 0,
    dynamic_budget_min_simulations: int | None = None,
    dynamic_budget_max_simulations: int | None = None,
    dynamic_budget_entropy_weight: float = 0.8,
    dynamic_budget_low_margin_threshold: float = 0.2,
    dynamic_budget_low_margin_weight: float = 1.5,
    dynamic_budget_variance_weight: float = 1.5,
    results: list[dict],
    exact_solve_enabled: bool,
    exact_solve_stone_threshold: int | None,
) -> dict:
    comparison_enabled = bool(dynamic_budget_enabled)
    effective_dynamic_budget_min_simulations = (
        int(mcts_simulations)
        if (not dynamic_budget_enabled or dynamic_budget_min_simulations is None)
        else int(dynamic_budget_min_simulations)
    )
    effective_dynamic_budget_max_simulations = (
        int(mcts_simulations)
        if (not dynamic_budget_enabled or dynamic_budget_max_simulations is None)
        else int(dynamic_budget_max_simulations)
    )
    classic_mcts_dynamic_budget_config = {
        "enabled": comparison_enabled,
        "probe_simulations": int(dynamic_budget_probe_simulations)
        if comparison_enabled
        else 0,
        "min_simulations": effective_dynamic_budget_min_simulations,
        "max_simulations": effective_dynamic_budget_max_simulations,
        "entropy_weight": float(dynamic_budget_entropy_weight)
        if comparison_enabled
        else 0.0,
        "low_margin_threshold": float(dynamic_budget_low_margin_threshold)
        if comparison_enabled
        else 0.0,
        "low_margin_weight": float(dynamic_budget_low_margin_weight)
        if comparison_enabled
        else 0.0,
        "variance_weight": float(dynamic_budget_variance_weight)
        if comparison_enabled
        else 0.0,
    }
    dynamic_summary = summarize_mode_results(results, mode="dynamic")
    fixed_summary = summarize_mode_results(results, mode="fixed")
    search_profile = build_search_profile(
        kind="mcts1200_baseline_eval",
        player_mode="classic_mcts",
        simulations=mcts_simulations,
        c_puct=1.25,
        search_options=search_options,
        extra_fields={
            "az_base_simulations": int(az_base_simulations),
            "mcts_simulations": int(mcts_simulations),
            "simulation_budget_policy": "fixed_vs_dynamic_classic_mcts"
            if comparison_enabled
            else "fixed_classic_mcts",
            "simulation_budget_min": effective_dynamic_budget_min_simulations,
            "simulation_budget_max": effective_dynamic_budget_max_simulations,
            "simulation_budget_multipliers": "dynamic:adaptive,fixed:constant"
            if comparison_enabled
            else "fixed:constant",
            "dynamic_budget_enabled": classic_mcts_dynamic_budget_config["enabled"],
            "dynamic_budget_probe_simulations": classic_mcts_dynamic_budget_config[
                "probe_simulations"
            ],
            "dynamic_budget_min_simulations": classic_mcts_dynamic_budget_config[
                "min_simulations"
            ],
            "dynamic_budget_max_simulations": classic_mcts_dynamic_budget_config[
                "max_simulations"
            ],
            "dynamic_budget_entropy_weight": classic_mcts_dynamic_budget_config[
                "entropy_weight"
            ],
            "dynamic_budget_low_margin_threshold": classic_mcts_dynamic_budget_config[
                "low_margin_threshold"
            ],
            "dynamic_budget_low_margin_weight": classic_mcts_dynamic_budget_config[
                "low_margin_weight"
            ],
            "dynamic_budget_variance_weight": classic_mcts_dynamic_budget_config[
                "variance_weight"
            ],
            "exact_solve_enabled": bool(exact_solve_enabled),
            "exact_solve_stone_threshold": None
            if exact_solve_stone_threshold is None
            else int(exact_solve_stone_threshold),
        },
    )
    classic_mcts_mode = "dynamic" if bool(dynamic_budget_enabled) else "fixed"
    budget_summary_source = (
        "classic_mcts_dynamic_runtime"
        if bool(dynamic_budget_enabled)
        else "classic_mcts_fixed_runtime"
    )
    report = {
        "schema": "azlite_vs_mcts_v1",
        "classic_mcts_mode": classic_mcts_mode,
        "games": games,
        "az_base_simulations": az_base_simulations,
        "mcts_simulations": mcts_simulations,
        "search_option_notes": "PUCT-oriented search flags are recorded in search_profile for provenance but ignored by ClassicMCTS execution",
        "search_profile": search_profile,
        "search_profile_hash": search_profile["hash"],
        "classic_mcts_dynamic_budget_config": classic_mcts_dynamic_budget_config,
        "az_wins": dynamic_summary["az_wins"]
        if bool(dynamic_budget_enabled)
        else fixed_summary["az_wins"],
        "mcts_wins": dynamic_summary["mcts_wins"]
        if bool(dynamic_budget_enabled)
        else fixed_summary["mcts_wins"],
        "draws": dynamic_summary["draws"]
        if bool(dynamic_budget_enabled)
        else fixed_summary["draws"],
        "score": round(
            float(
                dynamic_summary["score"]
                if bool(dynamic_budget_enabled)
                else fixed_summary["score"]
            ),
            4,
        ),
        "budget_summary": {
            "source": budget_summary_source,
            "mean_final_simulations": dynamic_summary["budget_summary"][
                "mean_final_simulations"
            ]
            if bool(dynamic_budget_enabled)
            else fixed_summary["budget_summary"]["mean_final_simulations"],
            "mean_root_latency_ms": dynamic_summary["budget_summary"][
                "mean_root_latency_ms"
            ]
            if bool(dynamic_budget_enabled)
            else fixed_summary["budget_summary"]["mean_root_latency_ms"],
        },
    }
    if comparison_enabled:
        runtime_target_ms = fixed_summary["budget_summary"]["mean_root_latency_ms"]
        dynamic_latency_ms = dynamic_summary["budget_summary"]["mean_root_latency_ms"]
        report["comparison_mode"] = "classic_dynamic_vs_fixed"
        report["dynamic_budget_comparison"] = {
            "comparison_mode": "classic_dynamic_vs_fixed",
            "runtime_target_ms": runtime_target_ms,
            "runtime_target_matched": (
                runtime_target_ms is not None
                and dynamic_latency_ms is not None
                and dynamic_latency_ms >= runtime_target_ms
            ),
            "seat_bias_neutralized": seat_bias_neutralized(results),
            "dynamic_mean_final_simulations": dynamic_summary["budget_summary"][
                "mean_final_simulations"
            ],
            "dynamic_mean_root_latency_ms": dynamic_latency_ms,
            "fixed_mean_final_simulations": fixed_summary["budget_summary"][
                "mean_final_simulations"
            ],
            "fixed_mean_root_latency_ms": runtime_target_ms,
            "dynamic_score": None
            if dynamic_summary["score"] is None
            else round(float(dynamic_summary["score"]), 4),
            "fixed_score": None
            if fixed_summary["score"] is None
            else round(float(fixed_summary["score"]), 4),
        }
    return report


def main() -> None:
    args = parse_args()
    if args.games <= 0:
        raise SystemExit("--games must be > 0")

    challenger_path = Path(args.challenger_path).resolve()
    if (
        os.environ.get("AZLITE_MCTS1200_BASELINE_STUB") != "1"
        and not challenger_path.exists()
    ):
        raise SystemExit(f"challenger path not found: {challenger_path}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    search_options = build_eval_search_options(**search_options_from_args(args))

    counts = partition_counts(args.games, args.workers)
    starts = partition_starts(counts)

    results: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, args.workers)
    ) as pool:
        futures = [
            pool.submit(
                run_worker,
                challenger_path=str(challenger_path),
                games=count,
                start_index=start_index,
                seed=args.seed,
                az_base_simulations=args.az_base_simulations,
                mcts_simulations=args.mcts_simulations,
                search_options=search_options,
                dynamic_budget_enabled=bool(args.dynamic_budget_enabled),
                dynamic_budget_probe_simulations=int(
                    args.dynamic_budget_probe_simulations
                ),
                dynamic_budget_min_simulations=args.dynamic_budget_min_simulations,
                dynamic_budget_max_simulations=args.dynamic_budget_max_simulations,
                dynamic_budget_entropy_weight=float(args.dynamic_budget_entropy_weight),
                dynamic_budget_low_margin_threshold=float(
                    args.dynamic_budget_low_margin_threshold
                ),
                dynamic_budget_low_margin_weight=float(
                    args.dynamic_budget_low_margin_weight
                ),
                dynamic_budget_variance_weight=float(
                    args.dynamic_budget_variance_weight
                ),
                exact_solve_enabled=bool(args.exact_solve_enabled),
                exact_solve_stone_threshold=args.exact_solve_stone_threshold,
            )
            for start_index, count in zip(starts, counts, strict=True)
            if count > 0
        ]
        results = [future.result() for future in futures]

    results.sort(key=lambda item: int(item["start_index"]))
    report = build_report(
        games=args.games,
        az_base_simulations=args.az_base_simulations,
        mcts_simulations=args.mcts_simulations,
        search_options=search_options,
        dynamic_budget_enabled=bool(args.dynamic_budget_enabled),
        dynamic_budget_probe_simulations=int(args.dynamic_budget_probe_simulations),
        dynamic_budget_min_simulations=args.dynamic_budget_min_simulations,
        dynamic_budget_max_simulations=args.dynamic_budget_max_simulations,
        dynamic_budget_entropy_weight=float(args.dynamic_budget_entropy_weight),
        dynamic_budget_low_margin_threshold=float(
            args.dynamic_budget_low_margin_threshold
        ),
        dynamic_budget_low_margin_weight=float(args.dynamic_budget_low_margin_weight),
        dynamic_budget_variance_weight=float(args.dynamic_budget_variance_weight),
        results=results,
        exact_solve_enabled=bool(args.exact_solve_enabled),
        exact_solve_stone_threshold=args.exact_solve_stone_threshold,
    )
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote mcts1200 baseline report to {out_path}")


if __name__ == "__main__":
    main()
