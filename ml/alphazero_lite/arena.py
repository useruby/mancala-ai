#!/usr/bin/env python3
"""Arena evaluator for candidate-vs-current AlphaZero-lite checkpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import statistics
import time
from pathlib import Path

ARENA_STUB_MODE = os.environ.get("AZLITE_ARENA_STUB") == "1"

try:
    from ml.alphazero_lite.report_validation import wilson_interval_95
except ModuleNotFoundError:
    from report_validation import wilson_interval_95

if not ARENA_STUB_MODE:
    import numpy as np

    try:
        from ml.alphazero_lite.input_encodings import DEFAULT_INPUT_ENCODING
        from ml.alphazero_lite.kalah_rules import KalahGame
        from ml.alphazero_lite.opening_cache import (
            load_opening_cache,
            state_qualifies_for_opening_cache,
        )
        from ml.alphazero_lite.search_ablation import build_mode_config, neutral_value
        from ml.alphazero_lite.self_play import (
            ClassicMCTS,
            PUCT,
            DEFAULT_EVAL_SEARCH_OPTIONS,
            DEFAULT_SEARCH_OPTIONS,
            add_search_option_args,
            build_eval_search_options,
            build_search_profile,
            build_search_options,
            encode_state,
            search_options_from_args,
            value_from_classic_mcts_root,
            visits_from_classic_mcts_root,
        )
    except ModuleNotFoundError:
        from input_encodings import DEFAULT_INPUT_ENCODING
        from kalah_rules import KalahGame
        from opening_cache import load_opening_cache, state_qualifies_for_opening_cache
        from search_ablation import build_mode_config, neutral_value
        from self_play import (
            ClassicMCTS,
            PUCT,
            DEFAULT_EVAL_SEARCH_OPTIONS,
            DEFAULT_SEARCH_OPTIONS,
            add_search_option_args,
            build_eval_search_options,
            build_search_profile,
            build_search_options,
            encode_state,
            search_options_from_args,
            value_from_classic_mcts_root,
            visits_from_classic_mcts_root,
        )


EARLY_DEFAULT_SEARCH_OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "visit_count",
    "tactical_root_bias": 0.0,
}
EARLY_DEFAULT_EVAL_SEARCH_OPTIONS = {
    **EARLY_DEFAULT_SEARCH_OPTIONS,
    "root_policy_mode": "deterministic",
    "tactical_root_bias": 0.1,
}
EARLY_SUPPORTED_ROOT_POLICY_MODES = ("deterministic", "visit_count")

if ARENA_STUB_MODE:
    DEFAULT_SEARCH_OPTIONS = EARLY_DEFAULT_SEARCH_OPTIONS
    DEFAULT_EVAL_SEARCH_OPTIONS = EARLY_DEFAULT_EVAL_SEARCH_OPTIONS


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
    parser.add_argument("--value-trust-enabled", action="store_true")
    parser.add_argument("--value-trust-opening", type=float, default=1.0)
    parser.add_argument("--value-trust-midgame", type=float, default=1.0)
    parser.add_argument("--value-trust-late", type=float, default=1.0)


def parse_stub_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--challenger-simulations", type=int, default=384)
    parser.add_argument("--current-simulations", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--opening-cache", default=None)
    parser.add_argument("--opening-cache-training-summary", default=None)
    parser.add_argument("--out", required=True)
    add_early_search_option_args(parser)
    parser.set_defaults(
        root_policy_mode=EARLY_DEFAULT_EVAL_SEARCH_OPTIONS["root_policy_mode"],
        tactical_root_bias=EARLY_DEFAULT_EVAL_SEARCH_OPTIONS["tactical_root_bias"],
    )
    return parser.parse_args(argv)


def search_options_from_stub_args(
    args: argparse.Namespace,
) -> dict[str, str | bool | float]:
    search_options: dict[str, str | bool | float | dict[str, float | bool]] = {
        "fpu_mode": args.fpu_mode,
        "reuse_subtree": bool(args.reuse_subtree),
        "normalize_values": bool(args.normalize_values),
        "root_policy_mode": args.root_policy_mode,
        "tactical_root_bias": float(args.tactical_root_bias),
    }
    if bool(args.value_trust_enabled):
        search_options["value_trust_schedule"] = {
            "enabled": True,
            "opening": float(args.value_trust_opening),
            "midgame": float(args.value_trust_midgame),
            "late": float(args.value_trust_late),
        }
    return search_options


def load_stub_training_summary(path: str | None) -> dict | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stub_opening_cache_summary(training_summary: dict | None) -> dict:
    upstream = training_summary if isinstance(training_summary, dict) else {}
    return {
        "runtime_hit_rate": None,
        "training_hit_rate": upstream.get("training_hit_rate"),
        "opening_bucket_quality_delta": None,
        "latency_delta_ms": None,
    }


def stub_value_trust_summary(
    search_options: dict[str, str | bool | float | dict[str, float | bool]],
) -> dict | None:
    schedule = search_options.get("value_trust_schedule")
    if not isinstance(schedule, dict):
        return None
    return {
        "enabled": bool(schedule["enabled"]),
        "phase_bucket": "opening",
        "effective_multiplier": float(schedule["opening"]),
        "schedule": {
            "opening": float(schedule["opening"]),
            "midgame": float(schedule["midgame"]),
            "late": float(schedule["late"]),
        },
    }


def build_stub_search_profile(
    args: argparse.Namespace, search_options: dict[str, str | bool | float]
) -> dict:
    return {
        "kind": "arena_eval",
        "player_mode": "puct",
        "simulations": max(
            int(args.challenger_simulations), int(args.current_simulations)
        ),
        "c_puct": float(args.c_puct),
        "search_options": search_options,
        "challenger_simulations": int(args.challenger_simulations),
        "current_simulations": int(args.current_simulations),
        "hash": "arena-stub-profile",
    }


def attach_score_confidence_interval(report: dict, *, threshold: float) -> None:
    games = int(report["games_played"])
    score = float(report["score"])
    report["games"] = games
    report["confidence_interval_95"] = wilson_interval_95(
        score=score, sample_size=games
    )
    report["threshold"] = float(threshold)
    report["threshold_margin"] = score - float(threshold)
    lower = float(report["confidence_interval_95"]["lower"])
    upper = float(report["confidence_interval_95"]["upper"])
    report["unstable_decision"] = lower <= float(threshold) <= upper


def run_stub_main(argv: list[str] | None = None) -> int:
    args = parse_stub_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    search_options = search_options_from_stub_args(args)
    search_profile = build_stub_search_profile(args, search_options)
    training_summary = load_stub_training_summary(args.opening_cache_training_summary)
    wins = int(args.games * 0.6)
    losses = 0
    draws = int(args.games) - wins
    score = (wins + (0.5 * draws)) / int(args.games)
    report = {
        "schema": "arena_v1",
        "games_played": int(args.games),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": score,
        "promotion_decision": {"passed": bool(score >= float(args.min_score))},
        "notes": {
            "challenger_path": str(Path(args.challenger)),
            "current_path": str(Path(args.current)),
            "challenger_simulations": int(args.challenger_simulations),
            "current_simulations": int(args.current_simulations),
            "seed": int(args.seed),
            "workers_requested": 1,
            "workers_used": 1,
            "worker_game_counts": [int(args.games)],
            "search_options": search_options,
            "search_profile": search_profile,
            "search_profile_hash": search_profile["hash"],
            "move_time_mean_ms": 120.0,
            "move_time_p95_ms": 160.0,
        },
        "budget_summary": {
            "mean_final_simulations": float(
                max(int(args.challenger_simulations), int(args.current_simulations))
            ),
            "p95_root_latency_ms": 160.0,
            "trigger_counts": {"fixed_budget": int(args.games)},
        },
        "hard_suite_buckets": {
            "opening": {"games": wins, "score": 1.0 if wins > 0 else None},
            "midgame": {"games": 0, "score": None},
            "late": {"games": draws, "score": 0.5 if draws > 0 else None},
        },
        "opening_cache_summary": stub_opening_cache_summary(training_summary),
    }
    attach_score_confidence_interval(report, threshold=float(args.min_score))
    value_trust_summary = stub_value_trust_summary(search_options)
    if value_trust_summary is not None:
        report["value_trust_summary"] = value_trust_summary
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote arena report to {out_path}")
    print(
        f"score={report['score']:.4f} passed={report['promotion_decision']['passed']}"
    )
    return 0


PITS_PER_PLAYER = 6

HARD_SUITE_BUCKET_LABELS = {
    "early": "opening",
    "mid": "midgame",
    "late": "late",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--challenger-simulations", type=int, default=384)
    parser.add_argument("--current-simulations", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--opening-cache", default=None)
    parser.add_argument("--opening-cache-training-summary", default=None)
    parser.add_argument("--out", required=True)
    add_search_option_args(parser)
    parser.set_defaults(
        root_policy_mode=DEFAULT_EVAL_SEARCH_OPTIONS["root_policy_mode"],
        tactical_root_bias=DEFAULT_EVAL_SEARCH_OPTIONS["tactical_root_bias"],
    )
    return parser.parse_args()


def partition_counts(total: int, workers: int) -> list[int]:
    workers = max(1, workers)
    base = total // workers
    remainder = total % workers
    return [base + (1 if i < remainder else 0) for i in range(workers)]


class ArtifactEvaluator:
    def __init__(self, artifact_dir: Path):
        metadata_path = artifact_dir / "metadata.json"
        weights_path = artifact_dir / "weights.json"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"missing weights.json in artifact directory: {artifact_dir}"
            )

        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        weights = json.loads(weights_path.read_text(encoding="utf-8"))
        architecture = metadata.get("architecture", {})
        self.model_type = architecture.get("model_type", "mlp_v1")
        self.input_encoding = metadata.get("input_encoding", DEFAULT_INPUT_ENCODING)
        self.residual_blocks = self._extract_residual_blocks(weights)
        self.hidden_layers = (
            [] if self.residual_blocks else self._extract_hidden_layers(weights)
        )
        self.w_input = (
            np.asarray(weights["w_input"], dtype=np.float32)
            if "w_input" in weights
            else None
        )
        self.b_input = (
            np.asarray(weights["b_input"], dtype=np.float32)
            if "b_input" in weights
            else None
        )
        self.w_policy = np.asarray(weights["w_policy"], dtype=np.float32)
        self.b_policy = np.asarray(weights["b_policy"], dtype=np.float32)
        self.w_value = np.asarray(weights["w_value"], dtype=np.float32)
        self.b_value = np.asarray(weights["b_value"], dtype=np.float32)
        self.w_policy_hidden = None
        self.b_policy_hidden = None
        self.w_value_hidden = None
        self.b_value_hidden = None
        if self.model_type == "residual_v3":
            required_keys = [
                "w_policy_hidden",
                "b_policy_hidden",
                "w_value_hidden",
                "b_value_hidden",
            ]
            missing_keys = [key for key in required_keys if key not in weights]
            if missing_keys:
                missing = ", ".join(missing_keys)
                raise ValueError(
                    f"residual_v3 artifact is missing specialized head weights: {missing}"
                )
            self.w_policy_hidden = np.asarray(
                weights["w_policy_hidden"], dtype=np.float32
            )
            self.b_policy_hidden = np.asarray(
                weights["b_policy_hidden"], dtype=np.float32
            )
            self.w_value_hidden = np.asarray(
                weights["w_value_hidden"], dtype=np.float32
            )
            self.b_value_hidden = np.asarray(
                weights["b_value_hidden"], dtype=np.float32
            )

    def _extract_residual_blocks(
        self, weights: dict
    ) -> list[tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]:
        if "w_input" not in weights or "b_input" not in weights:
            return []

        blocks = []
        index = 1
        while f"w_residual_{index}_1" in weights and f"b_residual_{index}_1" in weights:
            blocks.append(
                (
                    (
                        np.asarray(weights[f"w_residual_{index}_1"], dtype=np.float32),
                        np.asarray(weights[f"b_residual_{index}_1"], dtype=np.float32),
                    ),
                    (
                        np.asarray(weights[f"w_residual_{index}_2"], dtype=np.float32),
                        np.asarray(weights[f"b_residual_{index}_2"], dtype=np.float32),
                    ),
                )
            )
            index += 1
        return blocks

    def _extract_hidden_layers(
        self, weights: dict
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        indexed_layers: list[tuple[np.ndarray, np.ndarray]] = []
        index = 1
        while f"w_hidden_{index}" in weights and f"b_hidden_{index}" in weights:
            w = np.asarray(weights[f"w_hidden_{index}"], dtype=np.float32)
            b = np.asarray(weights[f"b_hidden_{index}"], dtype=np.float32)
            indexed_layers.append((w, b))
            index += 1

        if indexed_layers:
            return indexed_layers

        return [
            (
                np.asarray(weights["w1"], dtype=np.float32),
                np.asarray(weights["b1"], dtype=np.float32),
            ),
            (
                np.asarray(weights["w2"], dtype=np.float32),
                np.asarray(weights["b2"], dtype=np.float32),
            ),
        ]

    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        x = np.asarray(
            encode_state(game.to_state(), input_encoding=self.input_encoding),
            dtype=np.float32,
        )

        if self.residual_blocks:
            assert self.w_input is not None
            assert self.b_input is not None
            hidden = np.maximum(0.0, (x @ self.w_input) + self.b_input)
            for (w1, b1), (w2, b2) in self.residual_blocks:
                residual = hidden
                hidden = np.maximum(0.0, (hidden @ w1) + b1)
                hidden = np.maximum(0.0, (hidden @ w2) + b2 + residual)
        else:
            hidden = x
            for w, b in self.hidden_layers:
                hidden = np.maximum(0.0, (hidden @ w) + b)

        policy_hidden = hidden
        value_hidden = hidden
        if self.model_type == "residual_v3":
            assert self.w_policy_hidden is not None
            assert self.b_policy_hidden is not None
            assert self.w_value_hidden is not None
            assert self.b_value_hidden is not None
            policy_hidden = np.maximum(
                0.0, (hidden @ self.w_policy_hidden) + self.b_policy_hidden
            )
            value_hidden = np.maximum(
                0.0, (hidden @ self.w_value_hidden) + self.b_value_hidden
            )

        logits = (policy_hidden @ self.w_policy) + self.b_policy
        logits = logits - np.max(logits)
        exp_values = np.exp(logits)
        priors = exp_values / np.sum(exp_values)

        legal_moves = game.possible_moves()
        masked = np.zeros(PITS_PER_PLAYER, dtype=np.float32)
        if legal_moves:
            masked[legal_moves] = priors[legal_moves]
            total = float(np.sum(masked))
            if total <= 0:
                masked[legal_moves] = 1.0 / len(legal_moves)
            else:
                masked /= total

        value = float(
            np.tanh((value_hidden @ self.w_value + self.b_value).reshape(-1)[0])
        )
        return masked, value


def choose_best_move(visits: np.ndarray, legal_moves: list[int]) -> int:
    legal_visits = {move: visits[move] for move in legal_moves}
    best_visit = max(legal_visits.values())
    best_moves = [move for move, visit in legal_visits.items() if visit == best_visit]
    return min(best_moves)


def evaluate_artifact_position(
    *,
    artifact_path: str | Path | None = None,
    evaluator: ArtifactEvaluator | None = None,
    state: dict,
    simulations: int,
    seed: int,
    c_puct: float,
    search_options: dict,
    ablation_mode: str = "full",
) -> dict:
    game = KalahGame.from_state(state)
    normalized_mode = build_mode_config(ablation_mode)

    if normalized_mode["use_classic"]:
        classic_kwargs = {}
        if "value_trust_schedule" in search_options:
            classic_kwargs["value_trust_schedule"] = search_options[
                "value_trust_schedule"
            ]
        search = ClassicMCTS(
            game.clone(), simulations=simulations, seed=seed, **classic_kwargs
        )
        root = search.search_root()
        visits = np.asarray(visits_from_classic_mcts_root(root), dtype=np.float32)
        legal_moves = game.possible_moves()
        summary = search.root_summary()
        selected_move = summary["selected_move"]
        total_visits = float(np.sum(visits[legal_moves])) if legal_moves else 0.0
        policy = np.zeros(PITS_PER_PLAYER, dtype=np.float32)
        if total_visits > 0:
            policy[legal_moves] = visits[legal_moves] / total_visits
        child_stats = []
        for move in legal_moves:
            child = root.children.get(move)
            child_stats.append(
                {
                    "move": int(move),
                    "visits": int(0 if child is None else child.visits),
                    "q_value": 0.0
                    if child is None or child.visits <= 0
                    else float((2.0 * (child.wins / float(child.visits))) - 1.0),
                }
            )

        result = {
            "selected_move": None if selected_move is None else int(selected_move),
            "legal_moves": [int(move) for move in legal_moves],
            "policy": [float(prior) for prior in policy.tolist()],
            "value": float(value_from_classic_mcts_root(root)),
            "child_stats": child_stats,
            "visits": [float(visit) for visit in visits.tolist()],
            "budget": summary.get(
                "budget",
                {
                    "dynamic_budget_enabled": False,
                    "baseline_simulations": int(simulations),
                    "chosen_simulations": int(simulations),
                    "probe_simulations": int(simulations),
                    "final_simulations": int(simulations),
                    "phase_bucket": "fixed",
                    "entropy": 0.0,
                    "top_move_margin": 0.0,
                    "child_value_variance": 0.0,
                    "trigger": "fixed_budget",
                    "root_latency_ms": 0.0,
                },
            ),
        }
        if (
            "value_trust_schedule" in search_options
            and summary.get("value_trust") is not None
        ):
            result["value_trust"] = summary.get("value_trust")
        return result

    if evaluator is None:
        if artifact_path is None:
            raise ValueError("artifact_path is required when evaluator is not provided")
        evaluator = ArtifactEvaluator(Path(artifact_path))

    root_value: float | None = None

    class RootValueEvaluator:
        def evaluate(self, position_game):
            nonlocal root_value
            policy, value = evaluator.evaluate(position_game)
            if root_value is None:
                root_value = (
                    float(value) if normalized_mode["use_value"] else neutral_value()
                )
            return policy, value

    puct_kwargs = {}
    if "value_trust_schedule" in search_options:
        puct_kwargs["value_trust_schedule"] = search_options["value_trust_schedule"]
    search = PUCT(
        evaluator=RootValueEvaluator(),
        simulations=simulations,
        c_puct=c_puct,
        rng=random.Random(seed),
        fpu_mode=str(search_options["fpu_mode"]),
        reuse_subtree=bool(search_options["reuse_subtree"]),
        normalize_values=bool(search_options["normalize_values"]),
        root_policy_mode=str(search_options["root_policy_mode"]),
        tactical_root_bias=float(search_options["tactical_root_bias"]),
        ablation_mode=str(normalized_mode["name"]),
        **puct_kwargs,
    )
    visits, root = search.run(game)
    legal_moves = game.possible_moves()
    if legal_moves and root is not None:
        selected_move = search.select_root_move(root, legal_moves)
    elif legal_moves:
        selected_move = choose_best_move(
            np.asarray(visits, dtype=np.float32), legal_moves
        )
    else:
        selected_move = None
    root_summary = search.root_summary() if hasattr(search, "root_summary") else None
    root_value_trust = None
    root_selection_breakdown = None
    root_visit_snapshots = None
    if isinstance(root_summary, dict):
        candidate_value_trust = root_summary.get("value_trust")
        if (
            isinstance(candidate_value_trust, dict)
            and "value_trust_schedule" in search_options
        ):
            root_value_trust = candidate_value_trust
        candidate_selection_breakdown = root_summary.get("selection_breakdown")
        if isinstance(candidate_selection_breakdown, dict):
            root_selection_breakdown = candidate_selection_breakdown
        candidate_visit_snapshots = root_summary.get("visit_snapshots")
        if isinstance(candidate_visit_snapshots, list):
            root_visit_snapshots = candidate_visit_snapshots
    policy = np.zeros(PITS_PER_PLAYER, dtype=np.float32)
    child_stats = []
    for move in legal_moves:
        child = root.children.get(move) if root is not None else None
        if child is not None:
            policy[move] = float(child.prior)
        child_stats.append(
            {
                "move": int(move),
                "visits": int(child.visit_count if child is not None else visits[move]),
                "q_value": float(
                    child.q_value
                    if child is not None and child.visit_count > 0
                    else 0.0
                ),
            }
        )

    result = {
        "selected_move": None if selected_move is None else int(selected_move),
        "legal_moves": [int(move) for move in legal_moves],
        "policy": [float(prior) for prior in policy.tolist()],
        "value": 0.0 if root_value is None else float(root_value),
        "child_stats": child_stats,
        "visits": [float(visit) for visit in visits.tolist()],
    }
    if root_value_trust is not None:
        result["value_trust"] = root_value_trust
    if root_selection_breakdown is not None:
        result["selection_breakdown"] = root_selection_breakdown
    if root_visit_snapshots is not None:
        result["visit_snapshots"] = root_visit_snapshots
    return result


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile_value))


def empty_hard_suite_buckets() -> dict[str, dict]:
    return {
        "opening": {"games": 0, "score": None},
        "midgame": {"games": 0, "score": None},
        "late": {"games": 0, "score": None},
    }


def phase_bucket_for_game(game: KalahGame) -> str:
    pits = getattr(game, "pits", None)
    if pits is None:
        return "early"
    seeds_remaining = sum(pits)
    if seeds_remaining <= 12:
        return "late"
    if seeds_remaining <= 24:
        return "mid"
    return "early"


def record_hard_suite_bucket(buckets: dict[str, dict], game: KalahGame) -> None:
    phase_bucket = phase_bucket_for_game(game)
    label = HARD_SUITE_BUCKET_LABELS[phase_bucket]
    buckets[label]["games"] += 1


def record_completed_game_bucket(
    buckets: dict[str, dict], seen_phase_buckets: set[str]
) -> None:
    if not seen_phase_buckets:
        label = HARD_SUITE_BUCKET_LABELS["early"]
    elif "late" in seen_phase_buckets:
        label = HARD_SUITE_BUCKET_LABELS["late"]
    elif "mid" in seen_phase_buckets:
        label = HARD_SUITE_BUCKET_LABELS["mid"]
    else:
        label = HARD_SUITE_BUCKET_LABELS["early"]
    buckets[label]["games"] += 1


def merge_hard_suite_buckets(bucket_summaries: list[dict | None]) -> dict[str, dict]:
    merged = empty_hard_suite_buckets()
    for summary in bucket_summaries:
        if not isinstance(summary, dict):
            continue
        for label in merged:
            row = (
                summary.get(label, {})
                if isinstance(summary.get(label, {}), dict)
                else {}
            )
            merged[label]["games"] += int(row.get("games", 0))
    return merged


def budget_summary_for(report: dict) -> dict:
    def parse_budget_number(budget: dict, key: str) -> float | None:
        value = budget.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    budgets = []
    for position in report.get("positions", []):
        summary = position.get("challenger_summary", {})
        budget = summary.get("budget") if isinstance(summary, dict) else None
        if isinstance(budget, dict) and budget:
            budgets.append(budget)

    if budgets:
        final_simulations = [
            value
            for budget in budgets
            if (value := parse_budget_number(budget, "final_simulations")) is not None
        ]
        root_latencies = [
            value
            for budget in budgets
            if (value := parse_budget_number(budget, "root_latency_ms")) is not None
        ]
        trigger_counts: dict[str, int] = {}
        for budget in budgets:
            trigger = budget.get("trigger")
            if trigger:
                trigger_counts[str(trigger)] = trigger_counts.get(str(trigger), 0) + 1

        summary = {
            "mean_final_simulations": round(statistics.fmean(final_simulations), 2)
            if final_simulations
            else None,
            "p95_root_latency_ms": round(percentile(root_latencies, 95), 2)
            if root_latencies
            else None,
            "trigger_counts": trigger_counts,
        }
        return summary

    notes = report.get("notes", {}) if isinstance(report.get("notes"), dict) else {}
    return {
        "mean_final_simulations": float(notes.get("challenger_simulations"))
        if notes.get("challenger_simulations") is not None
        else None,
        "p95_root_latency_ms": float(notes.get("move_time_p95_ms", 0.0)),
        "trigger_counts": (
            {"fixed_budget": int(report.get("games_played", 0))}
            if report.get("games_played") is not None
            else {}
        ),
    }


def attach_budget_summary(report: dict) -> dict:
    summary = budget_summary_for(report)
    report["budget_summary"] = summary
    if "hard_suite_buckets" not in report:
        report["hard_suite_buckets"] = empty_hard_suite_buckets()
    return report


def configured_value_trust_summary(search_options: dict) -> dict | None:
    schedule = (
        search_options.get("value_trust_schedule")
        if isinstance(search_options, dict)
        else None
    )
    if not isinstance(schedule, dict):
        return None
    return {
        "enabled": bool(schedule.get("enabled", False)),
        "phase_bucket": None,
        "effective_multiplier": None,
        "schedule": {
            "opening": float(schedule.get("opening", 1.0)),
            "midgame": float(schedule.get("midgame", 1.0)),
            "late": float(schedule.get("late", 1.0)),
        },
        "source": "configured_schedule",
    }


def aggregate_worker_reports(
    *,
    games: int,
    min_score: float,
    challenger_path: Path,
    current_path: Path,
    challenger_simulations: int,
    current_simulations: int,
    seed: int,
    workers: int,
    search_options: dict,
    results: list[dict],
    training_summary: dict | None = None,
) -> dict:
    wins = sum(int(result["wins"]) for result in results)
    losses = sum(int(result["losses"]) for result in results)
    draws = sum(int(result["draws"]) for result in results)
    move_durations_ms = [
        duration for result in results for duration in result["move_durations_ms"]
    ]

    score = (wins + (0.5 * draws)) / float(games)
    fallback_search_profile = build_search_profile(
        kind="arena_eval",
        player_mode="puct",
        simulations=max(int(challenger_simulations), int(current_simulations)),
        c_puct=1.25,
        search_options=search_options,
        extra_fields={
            "challenger_simulations": int(challenger_simulations),
            "current_simulations": int(current_simulations),
        },
    )
    emitted_search_profile = (
        results[0]["search_profile"] if results else fallback_search_profile
    )
    emitted_search_profile_hash = (
        results[0]["search_profile_hash"]
        if results
        else fallback_search_profile["hash"]
    )
    emitted_value_trust_summary = next(
        (
            result.get("value_trust_summary")
            for result in results
            if isinstance(result.get("value_trust_summary"), dict)
        ),
        None,
    )
    if emitted_value_trust_summary is None:
        emitted_value_trust_summary = configured_value_trust_summary(search_options)

    report = {
        "schema": "arena_v1",
        "games_played": int(games),
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "promotion_decision": {"passed": bool(score >= min_score)},
        "notes": {
            "challenger_path": str(challenger_path),
            "current_path": str(current_path),
            "challenger_simulations": int(challenger_simulations),
            "current_simulations": int(current_simulations),
            "seed": int(seed),
            "workers_requested": int(workers),
            "workers_used": len(results),
            "worker_game_counts": [
                int(result["wins"] + result["losses"] + result["draws"])
                for result in results
            ],
            "search_options": search_options,
            "search_profile": emitted_search_profile,
            "search_profile_hash": emitted_search_profile_hash,
            "move_time_mean_ms": round(statistics.fmean(move_durations_ms), 2)
            if move_durations_ms
            else 0.0,
            "move_time_p95_ms": round(percentile(move_durations_ms, 95), 2),
        },
        "hard_suite_buckets": merge_hard_suite_buckets(
            [result.get("hard_suite_buckets") for result in results]
        ),
    }
    report["opening_cache_summary"] = opening_cache_summary_for(
        results=results, training_summary=training_summary
    )
    if emitted_value_trust_summary is not None:
        report["value_trust_summary"] = emitted_value_trust_summary
    attach_budget_summary(report)
    report["score"] = score
    attach_score_confidence_interval(report, threshold=float(min_score))
    return report


def opening_cache_summary_for(
    *, results: list[dict], training_summary: dict | None = None
) -> dict:
    hits = sum(int(result.get("opening_cache_hits", 0)) for result in results)
    misses = sum(int(result.get("opening_cache_misses", 0)) for result in results)
    hit_quality_sum = sum(
        float(result.get("opening_cache_hit_quality_sum", 0.0)) for result in results
    )
    miss_quality_sum = sum(
        float(result.get("opening_cache_miss_quality_sum", 0.0)) for result in results
    )
    hit_latencies = [
        value
        for result in results
        for value in result.get("opening_cache_hit_latency_ms", [])
    ]
    miss_latencies = [
        value
        for result in results
        for value in result.get("opening_cache_miss_latency_ms", [])
    ]

    runtime_total = hits + misses
    runtime_hit_rate = None if runtime_total <= 0 else round(hits / runtime_total, 4)
    hit_quality_mean = None if hits <= 0 else hit_quality_sum / hits
    miss_quality_mean = None if misses <= 0 else miss_quality_sum / misses
    quality_delta = (
        None
        if hit_quality_mean is None or miss_quality_mean is None
        else round(hit_quality_mean - miss_quality_mean, 4)
    )
    hit_latency_mean = None if not hit_latencies else statistics.fmean(hit_latencies)
    miss_latency_mean = None if not miss_latencies else statistics.fmean(miss_latencies)
    latency_delta_ms = (
        None
        if hit_latency_mean is None or miss_latency_mean is None
        else round(hit_latency_mean - miss_latency_mean, 2)
    )

    upstream = training_summary if isinstance(training_summary, dict) else {}
    return {
        "runtime_hit_rate": runtime_hit_rate,
        "training_hit_rate": upstream.get("training_hit_rate"),
        "opening_bucket_quality_delta": quality_delta,
        "latency_delta_ms": latency_delta_ms,
    }


def load_training_summary(path: str | None) -> dict | None:
    if not path:
        return None

    summary_path = Path(path)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_opening_cache_artifact(path: str | None):
    if not path:
        return None

    cache_path = Path(path)
    return load_opening_cache(json.loads(cache_path.read_text(encoding="utf-8")))


def run_arena_worker(
    *,
    worker_id: int,
    start_index: int,
    games: int,
    challenger_path: str,
    current_path: str,
    challenger_simulations: int,
    current_simulations: int,
    seed: int,
    c_puct: float,
    max_moves: int,
    fpu_mode: str = DEFAULT_SEARCH_OPTIONS["fpu_mode"],
    reuse_subtree: bool = DEFAULT_SEARCH_OPTIONS["reuse_subtree"],
    normalize_values: bool = DEFAULT_SEARCH_OPTIONS["normalize_values"],
    root_policy_mode: str = DEFAULT_EVAL_SEARCH_OPTIONS["root_policy_mode"],
    tactical_root_bias: float = DEFAULT_EVAL_SEARCH_OPTIONS["tactical_root_bias"],
    value_trust_schedule: dict | None = None,
    opening_cache=None,
    opening_cache_path: str | None = None,
) -> dict:
    rng = np.random.default_rng(seed + worker_id)
    challenger = ArtifactEvaluator(Path(challenger_path))
    current = ArtifactEvaluator(Path(current_path))
    if opening_cache is None and opening_cache_path:
        opening_cache = load_opening_cache_artifact(opening_cache_path)

    normalized_search_options = build_search_options(
        fpu_mode=fpu_mode,
        reuse_subtree=reuse_subtree,
        normalize_values=normalize_values,
        root_policy_mode=root_policy_mode,
        tactical_root_bias=tactical_root_bias,
        value_trust_schedule=value_trust_schedule,
    )
    search_profile = build_search_profile(
        kind="arena_eval",
        player_mode="puct",
        simulations=max(int(challenger_simulations), int(current_simulations)),
        c_puct=c_puct,
        search_options=normalized_search_options,
        extra_fields={
            "challenger_simulations": int(challenger_simulations),
            "current_simulations": int(current_simulations),
        },
    )

    wins = 0
    losses = 0
    draws = 0
    move_durations_ms: list[float] = []
    hard_suite_buckets = empty_hard_suite_buckets()
    opening_cache_hits = 0
    opening_cache_misses = 0
    opening_cache_hit_quality_sum = 0.0
    opening_cache_miss_quality_sum = 0.0
    opening_cache_hit_latency_ms: list[float] = []
    opening_cache_miss_latency_ms: list[float] = []
    value_trust_summary = None

    for local_index in range(games):
        game_index = start_index + local_index
        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        challenger_player = 0 if game_index % 2 == 0 else 1
        reusable_roots = {
            0: None,
            1: None,
        }
        challenger_phase_buckets_seen: set[str] = set()
        ply = 0

        for _ in range(max_moves):
            if game.over():
                break

            legal_moves = game.possible_moves()
            if not legal_moves:
                break

            if game.current_player == challenger_player:
                challenger_phase_buckets_seen.add(phase_bucket_for_game(game))

            if game.current_player == challenger_player:
                evaluator = challenger
                sims = challenger_simulations
            else:
                evaluator = current
                sims = current_simulations
            acting_player = game.current_player

            cached_move: int | None = None
            lookup_duration_ms = 0.0
            if opening_cache is not None and acting_player == challenger_player:
                state = game.to_state()
                if state_qualifies_for_opening_cache(
                    state,
                    ply=ply,
                    opening_gate=getattr(opening_cache, "opening_gate", None),
                ):
                    lookup_started = time.perf_counter()
                    cached_entry = opening_cache.lookup(state, ply=ply)
                    lookup_duration_ms = (time.perf_counter() - lookup_started) * 1000.0
                    if cached_entry is not None:
                        selected_move = cached_entry.get("selected_move")
                        try:
                            normalized_selected_move = int(selected_move)
                        except (TypeError, ValueError):
                            cached_entry = None
                        else:
                            if normalized_selected_move in legal_moves:
                                cached_move = normalized_selected_move
                                opening_cache_hits += 1
                                opening_cache_hit_quality_sum += float(
                                    cached_entry.get("value", 0.0)
                                )
                                opening_cache_hit_latency_ms.append(lookup_duration_ms)
                                move_durations_ms.append(lookup_duration_ms)
                            else:
                                cached_entry = None

                    if cached_move is None:
                        opening_cache_misses += 1

            if cached_move is None:
                puct_kwargs = {}
                if "value_trust_schedule" in normalized_search_options:
                    puct_kwargs["value_trust_schedule"] = normalized_search_options[
                        "value_trust_schedule"
                    ]
                search = PUCT(
                    evaluator=evaluator,
                    simulations=sims,
                    c_puct=c_puct,
                    rng=random.Random(int(rng.integers(0, 2**31 - 1))),
                    root=reusable_roots[acting_player],
                    fpu_mode=str(normalized_search_options["fpu_mode"]),
                    reuse_subtree=bool(normalized_search_options["reuse_subtree"]),
                    normalize_values=bool(
                        normalized_search_options["normalize_values"]
                    ),
                    root_policy_mode=str(normalized_search_options["root_policy_mode"]),
                    tactical_root_bias=float(
                        normalized_search_options["tactical_root_bias"]
                    ),
                    **puct_kwargs,
                )
                started = time.perf_counter()
                visits, root = search.run(game)
                search_duration_ms = (time.perf_counter() - started) * 1000.0
                total_duration_ms = search_duration_ms + lookup_duration_ms
                move_durations_ms.append(total_duration_ms)

                if opening_cache is not None and acting_player == challenger_player:
                    opening_cache_miss_quality_sum += float(
                        getattr(root, "q_value", 0.0) if root is not None else 0.0
                    )
                    opening_cache_miss_latency_ms.append(total_duration_ms)

                if hasattr(search, "select_root_move") and root is not None:
                    if (
                        value_trust_summary is None
                        and "value_trust_schedule" in normalized_search_options
                        and hasattr(search, "root_summary")
                    ):
                        root_summary = search.root_summary()
                        if isinstance(root_summary, dict):
                            candidate_value_trust = root_summary.get("value_trust")
                            if isinstance(candidate_value_trust, dict):
                                value_trust_summary = candidate_value_trust
                    move = search.select_root_move(root, legal_moves)
                else:
                    move = choose_best_move(visits, legal_moves)
            else:
                root = None
                move = cached_move
            if not game.move(game.pit_index(move)):
                break
            if (
                reuse_subtree
                and root is not None
                and game.current_player == acting_player
            ):
                reusable_roots[acting_player] = root.child_for_action(move)
            else:
                reusable_roots[acting_player] = None
            if game.current_player != acting_player:
                reusable_roots[game.current_player] = None
            ply += 1

        challenger_store = game.captured_seeds[challenger_player]
        current_store = game.captured_seeds[1 - challenger_player]
        if challenger_store > current_store:
            wins += 1
        elif challenger_store < current_store:
            losses += 1
        else:
            draws += 1
        record_completed_game_bucket(hard_suite_buckets, challenger_phase_buckets_seen)

    return {
        "worker_id": worker_id,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "move_durations_ms": move_durations_ms,
        "hard_suite_buckets": hard_suite_buckets,
        "opening_cache_hits": opening_cache_hits,
        "opening_cache_misses": opening_cache_misses,
        "opening_cache_hit_quality_sum": opening_cache_hit_quality_sum,
        "opening_cache_miss_quality_sum": opening_cache_miss_quality_sum,
        "opening_cache_hit_latency_ms": opening_cache_hit_latency_ms,
        "opening_cache_miss_latency_ms": opening_cache_miss_latency_ms,
        "search_options": normalized_search_options,
        "search_profile": search_profile,
        "search_profile_hash": search_profile["hash"],
        "value_trust_summary": value_trust_summary,
    }


def main() -> None:
    args = parse_args()
    training_summary = load_training_summary(args.opening_cache_training_summary)

    if os.environ.get("AZLITE_ARENA_STUB") == "1":
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        search_options = build_eval_search_options(**search_options_from_args(args))
        search_profile = build_search_profile(
            kind="arena_eval",
            player_mode="puct",
            simulations=max(
                int(args.challenger_simulations), int(args.current_simulations)
            ),
            c_puct=args.c_puct,
            search_options=search_options,
            extra_fields={
                "challenger_simulations": int(args.challenger_simulations),
                "current_simulations": int(args.current_simulations),
            },
        )
        wins = int(args.games * 0.6)
        losses = 0
        draws = int(args.games) - wins
        hard_suite_buckets = {
            "opening": {"games": wins, "score": 1.0 if wins > 0 else None},
            "midgame": {"games": 0, "score": None},
            "late": {"games": draws, "score": 0.5 if draws > 0 else None},
        }
        report = {
            "schema": "arena_v1",
            "games_played": int(args.games),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "score": (wins + (0.5 * draws)) / int(args.games),
            "promotion_decision": {
                "passed": bool(
                    ((wins + (0.5 * draws)) / int(args.games)) >= args.min_score
                )
            },
            "notes": {
                "challenger_path": str(Path(args.challenger)),
                "current_path": str(Path(args.current)),
                "challenger_simulations": int(args.challenger_simulations),
                "current_simulations": int(args.current_simulations),
                "seed": int(args.seed),
                "workers_requested": 1,
                "workers_used": 1,
                "worker_game_counts": [int(args.games)],
                "search_options": search_options,
                "search_profile": search_profile,
                "search_profile_hash": search_profile["hash"],
                "move_time_mean_ms": 120.0,
                "move_time_p95_ms": 160.0,
            },
            "hard_suite_buckets": hard_suite_buckets,
        }
        attach_score_confidence_interval(report, threshold=float(args.min_score))
        if "value_trust_schedule" in search_options:
            report["value_trust_summary"] = {
                "enabled": bool(search_options["value_trust_schedule"]["enabled"]),
                "phase_bucket": "opening",
                "effective_multiplier": float(
                    search_options["value_trust_schedule"]["opening"]
                )
                if bool(search_options["value_trust_schedule"]["enabled"])
                else 1.0,
                "schedule": {
                    "opening": float(search_options["value_trust_schedule"]["opening"]),
                    "midgame": float(search_options["value_trust_schedule"]["midgame"]),
                    "late": float(search_options["value_trust_schedule"]["late"]),
                },
            }
        report["opening_cache_summary"] = opening_cache_summary_for(
            results=[], training_summary=training_summary
        )
        attach_budget_summary(report)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote arena report to {out_path}")
        print(
            f"score={report['score']:.4f} passed={report['promotion_decision']['passed']}"
        )
        return

    challenger_path = Path(args.challenger)
    current_path = Path(args.current)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workers = max(1, args.workers)
    search_options = build_eval_search_options(**search_options_from_args(args))
    counts = partition_counts(args.games, workers)
    starts: list[int] = []
    cursor = 0
    for count in counts:
        starts.append(cursor)
        cursor += count

    futures = []
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for worker_id, (start_index, count) in enumerate(
            zip(starts, counts, strict=True)
        ):
            if count <= 0:
                continue
            futures.append(
                pool.submit(
                    run_arena_worker,
                    worker_id=worker_id,
                    start_index=start_index,
                    games=count,
                    challenger_path=str(challenger_path),
                    current_path=str(current_path),
                    challenger_simulations=args.challenger_simulations,
                    current_simulations=args.current_simulations,
                    seed=args.seed,
                    c_puct=args.c_puct,
                    max_moves=args.max_moves,
                    opening_cache_path=args.opening_cache,
                    fpu_mode=str(search_options["fpu_mode"]),
                    reuse_subtree=bool(search_options["reuse_subtree"]),
                    normalize_values=bool(search_options["normalize_values"]),
                    root_policy_mode=str(search_options["root_policy_mode"]),
                    tactical_root_bias=float(search_options["tactical_root_bias"]),
                    value_trust_schedule=search_options.get("value_trust_schedule"),
                )
            )
        results = [future.result() for future in futures]

    report = aggregate_worker_reports(
        games=args.games,
        min_score=args.min_score,
        challenger_path=challenger_path,
        current_path=current_path,
        challenger_simulations=args.challenger_simulations,
        current_simulations=args.current_simulations,
        seed=args.seed,
        workers=args.workers,
        search_options=search_options,
        results=results,
        training_summary=training_summary,
    )

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote arena report to {out_path}")
    print(
        f"score={report['score']:.4f} passed={report['promotion_decision']['passed']}"
    )


if __name__ == "__main__":
    if ARENA_STUB_MODE:
        raise SystemExit(run_stub_main())
    main()
