#!/usr/bin/env python3
"""Arena evaluator for candidate-vs-current AlphaZero-lite checkpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

ARENA_STUB_MODE = os.environ.get("AZLITE_ARENA_STUB") == "1"

try:
    from ml.alphazero_lite.report_validation import wilson_interval_95
except ModuleNotFoundError:
    from report_validation import wilson_interval_95

try:
    from ml.alphazero_lite.value_transforms import (
        parse_value_transform_json,
        value_transform_summary_for_phase,
    )
except ModuleNotFoundError:
    from value_transforms import (
        parse_value_transform_json,
        value_transform_summary_for_phase,
    )

if not ARENA_STUB_MODE:
    import numpy as np

    try:
        from ml.alphazero_lite.root_prior_transforms import (
            ARENA_TRANSFORM_NAMES,
            build_root_prior_override,
            merge_root_prior_telemetry_summaries,
            summarize_root_prior_telemetry,
        )
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
        from root_prior_transforms import (
            ARENA_TRANSFORM_NAMES,
            build_root_prior_override,
            merge_root_prior_telemetry_summaries,
            summarize_root_prior_telemetry,
        )
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
    "root_temperature": 0.0,
}
EARLY_DEFAULT_EVAL_SEARCH_OPTIONS = {
    **EARLY_DEFAULT_SEARCH_OPTIONS,
    "root_policy_mode": "deterministic",
    "tactical_root_bias": 0.0,
}
EARLY_SUPPORTED_ROOT_POLICY_MODES = ("deterministic", "visit_count")

if ARENA_STUB_MODE:
    DEFAULT_SEARCH_OPTIONS = EARLY_DEFAULT_SEARCH_OPTIONS
    DEFAULT_EVAL_SEARCH_OPTIONS = EARLY_DEFAULT_EVAL_SEARCH_OPTIONS
    ARENA_TRANSFORM_NAMES = ()


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
    parser.add_argument("--value-transform-json", default=None)


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
    parser.add_argument("--random-opening-plies", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--root-prior-transform",
        choices=sorted(ARENA_TRANSFORM_NAMES),
        default=None,
    )
    parser.add_argument(
        "--challenger-root-prior-transform",
        choices=sorted(ARENA_TRANSFORM_NAMES),
        default=None,
    )
    parser.add_argument(
        "--current-root-prior-transform",
        choices=sorted(ARENA_TRANSFORM_NAMES),
        default=None,
    )
    parser.add_argument(
        "--challenger-starts",
        type=int,
        default=None,
        choices=(0, 1),
    )
    parser.add_argument(
        "--game-jsonl",
        default=None,
    )
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
    value_transform = parse_value_transform_json(
        getattr(args, "value_transform_json", None)
    )
    if value_transform is not None:
        search_options["value_transform"] = value_transform
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
    profile = {
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
    if args.root_prior_transform:
        profile["root_prior_transform"] = str(args.root_prior_transform)
    return profile


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
    if args.game_jsonl:
        game_jsonl_path = Path(args.game_jsonl)
        game_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(game_jsonl_path, "w", encoding="utf-8") as gjf:
            for i in range(args.games):
                cp = (
                    int(args.challenger_starts)
                    if args.challenger_starts is not None
                    else (i % 2)
                )
                w = "challenger" if cp == 0 else "current"
                margin = 5 if cp == 0 else -5
                gjf.write(
                    json.dumps(
                        {
                            "game_index": i,
                            "challenger_player": cp,
                            "first_move_challenger": 1 if i % 2 == 0 else 2,
                            "first_move_current": 2 if i % 2 == 0 else 1,
                            "margin": margin,
                            "game_length": 35,
                            "winner": w,
                            "trajectory": "1,2,3,4,5",
                        }
                    )
                    + "\n"
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
    parser.add_argument("--random-opening-plies", type=int, default=0)
    parser.add_argument("--opening-seed", type=int, default=None)
    parser.add_argument("--opening-samples", type=int, default=0)
    parser.add_argument("--opening-plies", type=int, default=None)
    parser.add_argument("--opening-prefixes-jsonl", default=None)
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument(
        "--challenger-starts",
        type=int,
        default=None,
        choices=(0, 1),
        help="Force challenger to always start as player 0 or 1 (diagnostic seat-split). Default: alternate by game index.",
    )
    parser.add_argument(
        "--game-jsonl",
        default=None,
        help="Optional path to write per-game diagnostic lines (JSONL). Does not affect normal arena output.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--root-prior-transform",
        choices=sorted(ARENA_TRANSFORM_NAMES),
        default=None,
    )
    parser.add_argument(
        "--challenger-root-prior-transform",
        choices=sorted(ARENA_TRANSFORM_NAMES),
        default=None,
    )
    parser.add_argument(
        "--current-root-prior-transform",
        choices=sorted(ARENA_TRANSFORM_NAMES),
        default=None,
    )
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


def apply_random_opening_prefix(
    game: KalahGame, *, pair_seed: int, opening_plies: int
) -> int:
    if opening_plies <= 0:
        return 0

    rng = np.random.default_rng(pair_seed)
    applied = 0
    for _ in range(opening_plies):
        if game.over():
            break
        legal_moves = game.possible_moves()
        if not legal_moves:
            break
        move = int(rng.choice(legal_moves))
        if not game.move(game.pit_index(move)):
            break
        applied += 1
    return applied


def generate_random_opening_moves(
    *,
    game: KalahGame,
    opening_seed: int,
    opening_plies: int,
) -> list[int]:
    if opening_plies <= 0:
        return []

    rng = np.random.default_rng(opening_seed)
    moves: list[int] = []
    for _ in range(opening_plies):
        if game.over():
            break
        legal_moves = game.possible_moves()
        if not legal_moves:
            break
        move = int(rng.choice(legal_moves))
        relative_move = game.pit_index(move)
        if not game.move(relative_move):
            break
        moves.append(relative_move)
    return moves


def apply_opening_moves(game: KalahGame, moves: list[int]) -> int:
    applied = 0
    for move in moves:
        if game.over():
            break
        legal_moves = game.possible_moves()
        if not legal_moves:
            break
        if move not in legal_moves:
            break
        if not game.move(move):
            break
        applied += 1
    return applied


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
        self.is_v4 = self.model_type == "residual_v4_move_factorized"
        self.move_factorized = self.is_v4 or architecture.get("move_factorized", False)

        if self.move_factorized:
            self.w_policy_move = [
                np.asarray(weights[f"w_policy_move_{i}"], dtype=np.float32)
                for i in range(PITS_PER_PLAYER)
            ]
            self.b_policy_move = [
                np.asarray(weights[f"b_policy_move_{i}"], dtype=np.float32)
                for i in range(PITS_PER_PLAYER)
            ]
            self.w_policy = None
            self.b_policy = None
        else:
            self.w_policy = np.asarray(weights["w_policy"], dtype=np.float32)
            self.b_policy = np.asarray(weights["b_policy"], dtype=np.float32)
            self.w_policy_move = None
            self.b_policy_move = None
        self.w_value = np.asarray(weights["w_value"], dtype=np.float32)
        self.b_value = np.asarray(weights["b_value"], dtype=np.float32)
        self.w_policy_hidden = None
        self.b_policy_hidden = None
        self.w_value_hidden = None
        self.b_value_hidden = None
        if self.model_type in ("residual_v3", "residual_v4_move_factorized"):
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
                    f"{self.model_type} artifact is missing specialized head weights: {missing}"
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
        if self.model_type in ("residual_v3", "residual_v4_move_factorized"):
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

        if self.move_factorized:
            assert self.w_policy_move is not None
            assert self.b_policy_move is not None
            move_logits = []
            for w_move, b_move in zip(self.w_policy_move, self.b_policy_move):
                move_logits.append(
                    float(
                        (policy_hidden @ w_move).reshape(-1)[0] + b_move.reshape(-1)[0]
                    )
                )
            logits = np.array(move_logits, dtype=np.float32)
        else:
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


TEACHER_CLASSIC_MCTS = "classic_mcts"
TEACHER_ARTIFACT_PUCT = "artifact_puct"
SUPPORTED_TEACHERS = frozenset({TEACHER_CLASSIC_MCTS, TEACHER_ARTIFACT_PUCT})


def teacher_bucket_for_row(row: dict) -> str | None:
    bucket = row.get("bucket") or row.get("teacher_bucket")
    if bucket in ("classic_teacher",):
        return TEACHER_CLASSIC_MCTS
    if bucket in ("puct_teacher", "artifact_puct"):
        return TEACHER_ARTIFACT_PUCT
    teacher_id = row.get("teacher_id")
    if teacher_id == "classic_mcts":
        return TEACHER_CLASSIC_MCTS
    if teacher_id in ("artifact_puct",):
        return TEACHER_ARTIFACT_PUCT
    return None


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
    root_prior_override=None,
    root_prior_transform: str | None = None,
    teacher: str | None = None,
) -> dict:
    game = KalahGame.from_state(state)
    normalized_mode = build_mode_config(ablation_mode)

    if teacher is not None:
        if teacher not in SUPPORTED_TEACHERS:
            raise ValueError(
                f"unsupported teacher: {teacher!r}; "
                f"expected one of {sorted(SUPPORTED_TEACHERS)}"
            )
        if teacher == TEACHER_CLASSIC_MCTS:
            normalized_mode["use_classic"] = True
        elif teacher == TEACHER_ARTIFACT_PUCT:
            normalized_mode["use_classic"] = False

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
    if root_prior_override is not None and root_prior_transform is not None:
        raise ValueError(
            "root_prior_override and root_prior_transform cannot both be provided"
        )
    if root_prior_transform is not None:
        root_prior_override = build_root_prior_override(root_prior_transform)

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
        root_prior_override=root_prior_override,
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
        root_prior_telemetry = root_summary.get("root_prior_telemetry")
    else:
        root_prior_telemetry = None
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
    if isinstance(root_prior_telemetry, dict):
        result["root_prior_telemetry"] = root_prior_telemetry
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


def configured_value_transform_summary(search_options: dict) -> dict | None:
    value_transform = (
        search_options.get("value_transform")
        if isinstance(search_options, dict)
        else None
    )
    if not isinstance(value_transform, dict):
        return None
    return value_transform_summary_for_phase(
        value_transform,
        phase_bucket=None,
        identity_name=str(value_transform.get("name") or "identity_ref"),
    )


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
    emitted_value_transform_summary = next(
        (
            result.get("value_transform_summary")
            for result in results
            if isinstance(result.get("value_transform_summary"), dict)
        ),
        None,
    )
    if emitted_value_transform_summary is None:
        emitted_value_transform_summary = configured_value_transform_summary(
            search_options
        )
    emitted_root_prior_telemetry = {
        "challenger": merge_root_prior_telemetry_summaries(
            [
                (result.get("root_prior_telemetry") or {}).get("challenger")
                for result in results
                if isinstance(result.get("root_prior_telemetry"), dict)
            ]
        ),
        "current": merge_root_prior_telemetry_summaries(
            [
                (result.get("root_prior_telemetry") or {}).get("current")
                for result in results
                if isinstance(result.get("root_prior_telemetry"), dict)
            ]
        ),
    }

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
            "value_transform_summary": emitted_value_transform_summary,
            "value_transform_hash": None
            if emitted_value_transform_summary is None
            else emitted_value_transform_summary.get("hash"),
            "root_prior_transform": next(
                (
                    result.get("root_prior_transform")
                    for result in results
                    if result.get("root_prior_transform") is not None
                ),
                None,
            ),
            "challenger_root_prior_transform": next(
                (
                    result.get("challenger_root_prior_transform")
                    for result in results
                    if result.get("challenger_root_prior_transform") is not None
                ),
                None,
            ),
            "current_root_prior_transform": next(
                (
                    result.get("current_root_prior_transform")
                    for result in results
                    if result.get("current_root_prior_transform") is not None
                ),
                None,
            ),
            "random_opening_plies": next(
                (
                    int(result.get("random_opening_plies", 0))
                    for result in results
                    if int(result.get("random_opening_plies", 0)) > 0
                ),
                0,
            ),
            "move_time_mean_ms": round(statistics.fmean(move_durations_ms), 2)
            if move_durations_ms
            else 0.0,
            "move_time_p95_ms": round(percentile(move_durations_ms, 95), 2),
        },
        "hard_suite_buckets": merge_hard_suite_buckets(
            [result.get("hard_suite_buckets") for result in results]
        ),
        "root_prior_telemetry": emitted_root_prior_telemetry,
    }
    report["opening_cache_summary"] = opening_cache_summary_for(
        results=results, training_summary=training_summary
    )
    if emitted_value_trust_summary is not None:
        report["value_trust_summary"] = emitted_value_trust_summary
    if emitted_value_transform_summary is not None:
        report["value_transform_summary"] = emitted_value_transform_summary
    attach_budget_summary(report)
    report["score"] = score
    attach_score_confidence_interval(report, threshold=float(min_score))
    return report


def worker_game_jsonl_path(base_path: str, worker_id: int) -> str:
    path = Path(base_path)
    return str(path.with_name(f"{path.stem}.worker_{worker_id}{path.suffix}"))


def merge_worker_game_jsonl_files(
    *, out_path: str, worker_ids: list[int], cleanup: bool = True
) -> None:
    merged_entries: list[dict] = []
    worker_paths = [
        Path(worker_game_jsonl_path(out_path, worker_id)) for worker_id in worker_ids
    ]
    for worker_path in worker_paths:
        if not worker_path.exists():
            continue
        with worker_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                merged_entries.append(json.loads(line))

    merged_entries.sort(key=lambda entry: int(entry.get("game_index", 0)))
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in merged_entries:
            handle.write(json.dumps(entry) + "\n")

    if cleanup:
        for worker_path in worker_paths:
            if worker_path.exists():
                worker_path.unlink()


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
    tactical_root_bias: float = 0.1,
    root_temperature: float = DEFAULT_EVAL_SEARCH_OPTIONS["root_temperature"],
    value_trust_schedule: dict | None = None,
    root_prior_transform: str | None = None,
    challenger_root_prior_transform: str | None = None,
    current_root_prior_transform: str | None = None,
    random_opening_plies: int = 0,
    opening_seed: int | None = None,
    opening_samples: int = 0,
    opening_prefixes_jsonl: str | None = None,
    games_per_opening: int = 2,
    challenger_starts: int | None = None,
    game_jsonl_path: str | None = None,
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
        root_temperature=root_temperature,
        value_trust_schedule=value_trust_schedule,
    )
    effective_challenger_root_prior_transform = (
        challenger_root_prior_transform
        if challenger_root_prior_transform is not None
        else root_prior_transform
    )
    effective_current_root_prior_transform = current_root_prior_transform
    search_profile = build_search_profile(
        kind="arena_eval",
        player_mode="puct",
        simulations=max(int(challenger_simulations), int(current_simulations)),
        c_puct=c_puct,
        search_options=normalized_search_options,
        extra_fields={
            "challenger_simulations": int(challenger_simulations),
            "current_simulations": int(current_simulations),
            **(
                {"root_prior_transform": str(effective_challenger_root_prior_transform)}
                if effective_challenger_root_prior_transform
                else {}
            ),
            **(
                {
                    "challenger_root_prior_transform": str(
                        effective_challenger_root_prior_transform
                    )
                }
                if effective_challenger_root_prior_transform
                else {}
            ),
            **(
                {
                    "current_root_prior_transform": str(
                        effective_current_root_prior_transform
                    )
                }
                if effective_current_root_prior_transform
                else {}
            ),
        },
    )
    challenger_root_prior_override = (
        None
        if effective_challenger_root_prior_transform is None
        else build_root_prior_override(effective_challenger_root_prior_transform)
    )
    current_root_prior_override = (
        None
        if effective_current_root_prior_transform is None
        else build_root_prior_override(effective_current_root_prior_transform)
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
    value_transform_summary = None
    challenger_root_prior_telemetry_entries: list[dict] = []
    current_root_prior_telemetry_entries: list[dict] = []
    game_entries: list[dict] = []
    trajectory_hashes: list[str] = []
    opening_prefix_plies_applied: list[int] = []
    effective_opening_seed: int | None = None
    if opening_seed is not None:
        effective_opening_seed = int(opening_seed)
    opening_prefix_cache: list[list[int]] | None = None
    if opening_prefixes_jsonl is not None:
        opening_prefix_cache = []
        with open(opening_prefixes_jsonl, "r", encoding="utf-8") as pf:
            for line in pf:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                opening_prefix_cache.append([int(m) for m in entry["prefix_moves"]])
    elif int(opening_samples) > 0 and int(random_opening_plies) > 0:
        effective_opening_seed = (
            int(opening_seed) if opening_seed is not None else int(seed)
        )
        opening_prefix_cache = []
        for sample_idx in range(int(opening_samples)):
            sample_game = KalahGame.from_state(
                {
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                }
            )
            prefix_moves = generate_random_opening_moves(
                game=sample_game,
                opening_seed=int(effective_opening_seed) + sample_idx,
                opening_plies=int(random_opening_plies),
            )
            opening_prefix_cache.append(prefix_moves)
        illegal_prefix_count = sum(
            1
            for moves in opening_prefix_cache
            if len(moves) < int(random_opening_plies)
        )
        if illegal_prefix_count > 0:
            opening_prefix_cache = None

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
        if challenger_starts is not None:
            challenger_player = challenger_starts
        else:
            challenger_player = 0 if game_index % 2 == 0 else 1

        opening_prefix_moves: list[int] = []
        if opening_prefix_cache is not None:
            gpo = max(1, int(games_per_opening))
            sample_idx = game_index // gpo
            if sample_idx < len(opening_prefix_cache):
                opening_prefix_moves = list(opening_prefix_cache[sample_idx])
                applied = apply_opening_moves(game, opening_prefix_moves)
                opening_prefix_plies_applied.append(applied)
            else:
                opening_prefix_plies_applied.append(0)
        else:
            applied = apply_random_opening_prefix(
                game,
                pair_seed=int(seed) + (game_index // 2),
                opening_plies=int(random_opening_plies),
            )
            opening_prefix_plies_applied.append(applied)
        reusable_roots = {
            0: None,
            1: None,
        }
        challenger_phase_buckets_seen: set[str] = set()
        ply = 0
        first_move_challenger: int | None = None
        first_move_current: int | None = None
        game_moves: list[int] = []

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
                if "value_transform" in normalized_search_options:
                    puct_kwargs["value_transform"] = normalized_search_options[
                        "value_transform"
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
                    root_temperature=float(
                        normalized_search_options.get("root_temperature", 0.0)
                    ),
                    root_prior_override=(
                        challenger_root_prior_override
                        if acting_player == challenger_player
                        else current_root_prior_override
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
                    root_summary = (
                        search.root_summary()
                        if hasattr(search, "root_summary")
                        else None
                    )
                    if (
                        value_trust_summary is None
                        and "value_trust_schedule" in normalized_search_options
                        and isinstance(root_summary, dict)
                    ):
                        candidate_value_trust = root_summary.get("value_trust")
                        if isinstance(candidate_value_trust, dict):
                            value_trust_summary = candidate_value_trust
                    if value_transform_summary is None and isinstance(
                        root_summary, dict
                    ):
                        candidate_value_transform = root_summary.get("value_transform")
                        if isinstance(candidate_value_transform, dict):
                            value_transform_summary = candidate_value_transform
                    if isinstance(root_summary, dict):
                        root_prior_telemetry = root_summary.get("root_prior_telemetry")
                        if isinstance(root_prior_telemetry, dict):
                            if acting_player == challenger_player:
                                challenger_root_prior_telemetry_entries.append(
                                    root_prior_telemetry
                                )
                            else:
                                current_root_prior_telemetry_entries.append(
                                    root_prior_telemetry
                                )
                    move = search.select_root_move(root, legal_moves)
                else:
                    move = choose_best_move(visits, legal_moves)
            else:
                root = None
                move = cached_move
            relative_move = game.pit_index(move)
            game_moves.append(relative_move)
            if acting_player == challenger_player and first_move_challenger is None:
                first_move_challenger = move
            elif acting_player != challenger_player and first_move_current is None:
                first_move_current = move
            if not game.move(relative_move):
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
        margin = challenger_store - current_store
        if challenger_store > current_store:
            wins += 1
            winner = "challenger"
        elif challenger_store < current_store:
            losses += 1
            winner = "current"
        else:
            draws += 1
            winner = "draw"
        record_completed_game_bucket(hard_suite_buckets, challenger_phase_buckets_seen)
        trajectory_str = ",".join(str(m) for m in game_moves)
        entry_data: dict = {
            "game_index": game_index,
            "challenger_player": challenger_player,
            "first_move_challenger": first_move_challenger,
            "first_move_current": first_move_current,
            "margin": margin,
            "game_length": ply,
            "winner": winner,
            "trajectory": trajectory_str,
        }
        if opening_prefix_moves:
            entry_data["opening_prefix_moves"] = [int(m) for m in opening_prefix_moves]
        game_entries.append(entry_data)
        trajectory_hashes.append(trajectory_str)

    if game_jsonl_path:
        with open(game_jsonl_path, "w", encoding="utf-8") as gjf:
            for entry in game_entries:
                gjf.write(json.dumps(entry) + "\n")

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
        "value_transform_summary": value_transform_summary,
        "root_prior_transform": root_prior_transform,
        "challenger_root_prior_transform": effective_challenger_root_prior_transform,
        "current_root_prior_transform": effective_current_root_prior_transform,
        "random_opening_plies": int(random_opening_plies),
        "opening_prefix_plies_applied": opening_prefix_plies_applied,
        "root_prior_telemetry": {
            "challenger": summarize_root_prior_telemetry(
                challenger_root_prior_telemetry_entries
            ),
            "current": summarize_root_prior_telemetry(
                current_root_prior_telemetry_entries
            ),
        },
    }


def main() -> None:
    args = parse_args()
    training_summary = load_training_summary(args.opening_cache_training_summary)

    if args.root_prior_transform is not None:
        if args.challenger_root_prior_transform is not None:
            raise SystemExit(
                "--root-prior-transform cannot be combined with --challenger-root-prior-transform"
            )
        args.challenger_root_prior_transform = args.root_prior_transform

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
                **(
                    {"root_prior_transform": str(args.challenger_root_prior_transform)}
                    if args.challenger_root_prior_transform
                    else {}
                ),
                **(
                    {
                        "challenger_root_prior_transform": str(
                            args.challenger_root_prior_transform
                        )
                    }
                    if args.challenger_root_prior_transform
                    else {}
                ),
                **(
                    {
                        "current_root_prior_transform": str(
                            args.current_root_prior_transform
                        )
                    }
                    if args.current_root_prior_transform
                    else {}
                ),
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
                "root_prior_transform": args.root_prior_transform,
                "challenger_root_prior_transform": args.challenger_root_prior_transform,
                "current_root_prior_transform": args.current_root_prior_transform,
                "random_opening_plies": int(args.random_opening_plies),
                "move_time_mean_ms": 120.0,
                "move_time_p95_ms": 160.0,
            },
            "hard_suite_buckets": hard_suite_buckets,
            "root_prior_telemetry": {
                "challenger": summarize_root_prior_telemetry([]),
                "current": summarize_root_prior_telemetry([]),
            },
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
        if "value_transform" in search_options:
            report["value_transform_summary"] = value_transform_summary_for_phase(
                search_options["value_transform"],
                phase_bucket="opening",
                identity_name=str(
                    search_options["value_transform"].get("name") or "identity_ref"
                ),
            )
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
    worker_game_paths: dict[int, str] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for worker_id, (start_index, count) in enumerate(
            zip(starts, counts, strict=True)
        ):
            if count <= 0:
                continue
            worker_game_jsonl = None
            if getattr(args, "game_jsonl", None):
                worker_game_jsonl = worker_game_jsonl_path(args.game_jsonl, worker_id)
                worker_game_paths[worker_id] = worker_game_jsonl
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
                    root_temperature=float(search_options.get("root_temperature", 0.0)),
                    value_trust_schedule=search_options.get("value_trust_schedule"),
                    root_prior_transform=args.root_prior_transform,
                    challenger_root_prior_transform=args.challenger_root_prior_transform,
                    current_root_prior_transform=args.current_root_prior_transform,
                    random_opening_plies=getattr(args, "opening_plies", None)
                    if getattr(args, "opening_plies", None) is not None
                    else args.random_opening_plies,
                    opening_seed=getattr(args, "opening_seed", None),
                    opening_samples=getattr(args, "opening_samples", 0),
                    games_per_opening=getattr(args, "games_per_opening", 2),
                    challenger_starts=getattr(args, "challenger_starts", None),
                    game_jsonl_path=worker_game_jsonl,
                    opening_prefixes_jsonl=getattr(
                        args, "opening_prefixes_jsonl", None
                    ),
                )
            )
        results = [future.result() for future in futures]

    if getattr(args, "game_jsonl", None):
        merge_worker_game_jsonl_files(
            out_path=args.game_jsonl,
            worker_ids=sorted(worker_game_paths),
        )

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
