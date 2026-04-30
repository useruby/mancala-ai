#!/usr/bin/env python3
"""Generate net-guided self-play trajectories for Kalah v1 encoding."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.input_encodings import DEFAULT_INPUT_ENCODING, SUPPORTED_INPUT_ENCODINGS, feature_count_for
from ml.alphazero_lite.eval_cache import EvalCache
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.opening_cache import load_opening_cache
from ml.alphazero_lite.opponent_pool import load_opponent_checkpoints, sample_opponent_checkpoint
from ml.alphazero_lite.search_ablation import build_mode_config, flat_legal_priors, neutral_value


PITS_PER_PLAYER = 6
SearchOptionValue = str | bool | float | dict[str, bool | float]
SearchOptions = dict[str, SearchOptionValue]
DEFAULT_SEARCH_OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "visit_count",
    "tactical_root_bias": 0.0,
}
SUPPORTED_ROOT_POLICY_MODES = frozenset({"visit_count", "deterministic"})
DEFAULT_EVAL_SEARCH_OPTIONS = {
    **DEFAULT_SEARCH_OPTIONS,
    "root_policy_mode": "deterministic",
    "tactical_root_bias": 0.1,
}
DEFAULT_POLICY_TARGET_MODE = "default"
SUPPORTED_POLICY_TARGET_MODES = frozenset({DEFAULT_POLICY_TARGET_MODE, "sharpened"})
DEFAULT_VALUE_TARGET_MODE = "default"
PHASE_AWARE_VALUE_TARGET_MODE = "phase_aware_sharpened"
HYBRID_VALUE_TARGET_MODE = "hybrid"
SUPPORTED_VALUE_TARGET_MODES = frozenset(
    {DEFAULT_VALUE_TARGET_MODE, "sharpened", PHASE_AWARE_VALUE_TARGET_MODE, HYBRID_VALUE_TARGET_MODE}
)
SUPPORTED_PLAYER_MODES = frozenset({"puct", "classic_mcts"})
DEFAULT_PLAYER_MODE = "puct"
VALUE_TARGET_BUCKETS = (
    (10, "early", 1.5),
    (30, "mid", 2.0),
    (math.inf, "late", 3.0),
)
HYBRID_VALUE_TARGET_WEIGHTS = {
    "early": {"outcome": 0.25, "search": 0.75},
    "mid": {"outcome": 0.5, "search": 0.5},
    "late": {"outcome": 0.75, "search": 0.25},
}


def non_negative_int(raw_value: str) -> int:
    value = int(raw_value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    total = np.sum(exp_values)
    if total <= 0:
        return np.full_like(logits, 1.0 / logits.shape[0], dtype=np.float32)
    return (exp_values / total).astype(np.float32)


def encode_state(state: dict, *, input_encoding: str = DEFAULT_INPUT_ENCODING) -> list[float]:
    if input_encoding == "kalah_v3":
        features = encode_kalah_v3(state)
    else:
        features = encode_base_state(state)

    total_features = feature_count_for(input_encoding)
    if len(features) != total_features:
        raise ValueError(f"expected {total_features} features, got {len(features)}")

    return features


def encode_base_state(state: dict) -> list[float]:
    pits_divisor = 48.0
    stores_divisor = 48.0

    features: list[float] = []
    features.extend(float(x) / pits_divisor for x in state["player_pits"])
    features.extend(float(x) / pits_divisor for x in state["opponent_pits"])
    features.append(float(state["player_store"]) / stores_divisor)
    features.append(float(state["opponent_store"]) / stores_divisor)
    features.append(float(state["current_player"]))
    return features


def encode_kalah_v3(state: dict) -> list[float]:
    features = encode_base_state(state)
    features.extend(_encode_kalah_v3_side(state, side=0, pits_key="player_pits"))
    features.extend(_encode_kalah_v3_side(state, side=1, pits_key="opponent_pits"))
    return features


def _encode_kalah_v3_side(state: dict, *, side: int, pits_key: str) -> list[float]:
    pits = [int(seeds) for seeds in state[pits_key]]
    remaining_stones = sum(pits)
    return [
        1.0 if _has_immediate_extra_turn(state, side=side) else 0.0,
        1.0 if _has_immediate_capture(state, side=side) else 0.0,
        pits.count(0) / PITS_PER_PLAYER,
        sum(1 for seeds in pits if seeds >= 7) / PITS_PER_PLAYER,
        sum(1 for move, seeds in enumerate(pits) if seeds == (PITS_PER_PLAYER - move)) / PITS_PER_PLAYER,
        remaining_stones / 48.0,
    ]


def _game_for_side(state: dict, *, side: int) -> KalahGame:
    return KalahGame(
        pits=[int(seeds) for seeds in state["player_pits"]] + [int(seeds) for seeds in state["opponent_pits"]],
        captured_seeds=[int(state["player_store"]), int(state["opponent_store"])],
        current_player=side,
    )


def _has_immediate_extra_turn(state: dict, *, side: int) -> bool:
    game = _game_for_side(state, side=side)
    return any(_is_immediate_extra_turn(game, move) for move in game.possible_moves())


def _has_immediate_capture(state: dict, *, side: int) -> bool:
    game = _game_for_side(state, side=side)
    return any(_is_immediate_capture(game, move) for move in game.possible_moves())


def _is_immediate_extra_turn(game: KalahGame, move: int) -> bool:
    simulated = game.clone()
    player = simulated.current_player
    if not simulated.move(simulated.pit_index(move)):
        return False
    return simulated.current_player == player and not simulated.over()


def _is_immediate_capture(game: KalahGame, move: int) -> bool:
    absolute_index = game.pit_index(move)
    seeds = game.pits[absolute_index]
    store_before = game.captured_seeds[game.current_player]
    simulated = game.clone()
    if not simulated.move(simulated.pit_index(move)):
        return False

    store_gain = simulated.captured_seeds[game.current_player] - store_before
    return store_gain > _own_store_passes(move, seeds)


def _own_store_passes(move: int, seeds: int) -> int:
    distance_to_store = PITS_PER_PLAYER - move
    if seeds < distance_to_store:
        return 0
    return 1 + ((seeds - distance_to_store) // ((PITS_PER_PLAYER * 2) + 1))


def encode_kalah_v1(state: dict) -> list[float]:
    return encode_state(state, input_encoding="kalah_v1")


def build_search_options(
    *,
    fpu_mode: str = DEFAULT_SEARCH_OPTIONS["fpu_mode"],
    reuse_subtree: bool = DEFAULT_SEARCH_OPTIONS["reuse_subtree"],
    normalize_values: bool = DEFAULT_SEARCH_OPTIONS["normalize_values"],
    root_policy_mode: str = DEFAULT_SEARCH_OPTIONS["root_policy_mode"],
    tactical_root_bias: float = DEFAULT_SEARCH_OPTIONS["tactical_root_bias"],
    value_trust_schedule: dict | None = None,
) -> SearchOptions:
    root_policy_mode = normalize_root_policy_mode(root_policy_mode)
    options: SearchOptions = {
        "fpu_mode": fpu_mode,
        "reuse_subtree": bool(reuse_subtree),
        "normalize_values": bool(normalize_values),
        "root_policy_mode": root_policy_mode,
        "tactical_root_bias": float(tactical_root_bias),
    }
    normalized_value_trust_schedule = normalize_value_trust_schedule(value_trust_schedule)
    if normalized_value_trust_schedule is not None:
        options["value_trust_schedule"] = normalized_value_trust_schedule
    return options


def normalize_value_trust_schedule(value_trust_schedule: dict | None) -> dict[str, bool | float] | None:
    if value_trust_schedule is None:
        return None

    enabled = value_trust_schedule.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("value_trust_schedule.enabled must be a boolean")

    def normalize_phase_multiplier(phase_name: str) -> float:
        raw_value = value_trust_schedule.get(phase_name, 1.0)
        if isinstance(raw_value, bool):
            raise ValueError(f"value_trust_schedule.{phase_name} must be a finite number > 0")
        try:
            normalized_value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"value_trust_schedule.{phase_name} must be a finite number > 0") from exc
        if not math.isfinite(normalized_value) or normalized_value <= 0.0:
            raise ValueError(f"value_trust_schedule.{phase_name} must be a finite number > 0")
        return normalized_value

    return {
        "enabled": enabled,
        "opening": normalize_phase_multiplier("opening"),
        "midgame": normalize_phase_multiplier("midgame"),
        "late": normalize_phase_multiplier("late"),
    }


def normalize_root_policy_mode(root_policy_mode: str) -> str:
    normalized = str(root_policy_mode)
    if normalized not in SUPPORTED_ROOT_POLICY_MODES:
        raise ValueError(f"unsupported root_policy_mode: {root_policy_mode}")
    return normalized


def build_eval_search_options(
    *,
    fpu_mode: str = DEFAULT_EVAL_SEARCH_OPTIONS["fpu_mode"],
    reuse_subtree: bool = DEFAULT_EVAL_SEARCH_OPTIONS["reuse_subtree"],
    normalize_values: bool = DEFAULT_EVAL_SEARCH_OPTIONS["normalize_values"],
    root_policy_mode: str = DEFAULT_EVAL_SEARCH_OPTIONS["root_policy_mode"],
    tactical_root_bias: float = DEFAULT_EVAL_SEARCH_OPTIONS["tactical_root_bias"],
    value_trust_schedule: dict | None = None,
) -> SearchOptions:
    return build_search_options(
        fpu_mode=fpu_mode,
        reuse_subtree=reuse_subtree,
        normalize_values=normalize_values,
        root_policy_mode=root_policy_mode,
        tactical_root_bias=tactical_root_bias,
        value_trust_schedule=value_trust_schedule,
    )


def build_search_profile(
    *,
    kind: str,
    player_mode: str,
    simulations: int,
    c_puct: float,
    search_options: SearchOptions,
    extra_fields: dict[str, str | int | float | bool] | None = None,
) -> dict[str, str | int | float | bool | dict[str, bool | float] | SearchOptions]:
    normalized_options = build_search_options(
        fpu_mode=str(search_options["fpu_mode"]),
        reuse_subtree=bool(search_options["reuse_subtree"]),
        normalize_values=bool(search_options["normalize_values"]),
        root_policy_mode=str(search_options["root_policy_mode"]),
        tactical_root_bias=float(search_options["tactical_root_bias"]),
        value_trust_schedule=search_options.get("value_trust_schedule"),
    )
    profile = {
        "version": "v1",
        "kind": str(kind),
        "player_mode": str(player_mode),
    }
    if str(player_mode) == "classic_mcts":
        profile["classic_mcts_simulations"] = int(simulations)
        if "value_trust_schedule" in normalized_options:
            profile["value_trust_schedule"] = normalized_options["value_trust_schedule"]
    else:
        profile.update(
            {
                "simulations": int(simulations),
                "c_puct": float(c_puct),
                "search_options": normalized_options,
            }
        )
    if extra_fields:
        profile.update(extra_fields)
    encoded_profile = json.dumps(profile, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    profile_hash = hashlib.sha256(encoded_profile.encode("utf-8")).hexdigest()
    return {
        **profile,
        "hash": profile_hash,
    }


def opponent_pool_fingerprint(checkpoints: list[str]) -> str:
    encoded = json.dumps([str(path) for path in checkpoints], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def cache_hit_rate(cache_hits: int, cache_misses: int) -> float:
    total = int(cache_hits) + int(cache_misses)
    if total <= 0:
        return 0.0
    return float(cache_hits) / float(total)


def cache_metrics_for(evaluators: list[Evaluator | None]) -> dict[str, bool | int | float]:
    cache_hits = 0
    cache_misses = 0
    cache_enabled = False
    for evaluator in evaluators:
        stats = getattr(evaluator, "cache_stats", None)
        if not isinstance(stats, dict):
            continue
        cache_enabled = cache_enabled or bool(stats.get("enabled", False))
        cache_hits += int(stats.get("hits", 0))
        cache_misses += int(stats.get("misses", 0))
    return {
        "cache_enabled": cache_enabled,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": cache_hit_rate(cache_hits, cache_misses),
    }


def format_metrics_line(
    *,
    prefix: str,
    extra_fields: dict[str, str | bool | int] | None = None,
    cache_hits: int,
    cache_misses: int,
) -> str:
    fields: list[str] = []
    for key, value in (extra_fields or {}).items():
        if isinstance(value, bool):
            fields.append(f"{key}={str(value).lower()}")
        else:
            fields.append(f"{key}={value}")
    fields.extend(
        [
            f"cache_hits={int(cache_hits)}",
            f"cache_misses={int(cache_misses)}",
            f"cache_hit_rate={cache_hit_rate(cache_hits, cache_misses):.6f}",
        ]
    )
    return f"{prefix} {' '.join(fields)}"


def normalize_policy_target_mode(policy_target_mode: str) -> str:
    normalized = str(policy_target_mode)
    if normalized not in SUPPORTED_POLICY_TARGET_MODES:
        raise ValueError(f"unsupported policy_target_mode: {policy_target_mode}")
    return normalized


def normalize_value_target_mode(value_target_mode: str) -> str:
    normalized = str(value_target_mode)
    if normalized not in SUPPORTED_VALUE_TARGET_MODES:
        raise ValueError(f"unsupported value_target_mode: {value_target_mode}")
    return normalized


def build_policy_target(
    visits: np.ndarray,
    *,
    legal_moves: list[int],
    temperature: float,
    mode: str = DEFAULT_POLICY_TARGET_MODE,
) -> list[float]:
    normalized_mode = normalize_policy_target_mode(mode)
    default_target = np.asarray(
        policy_from_visits(visits, legal_moves=legal_moves, temperature=temperature),
        dtype=np.float32,
    )
    return build_policy_target_from_distribution(default_target, mode=normalized_mode)


def build_policy_target_from_distribution(
    policy: list[float] | np.ndarray,
    *,
    mode: str = DEFAULT_POLICY_TARGET_MODE,
) -> list[float]:
    normalized_mode = normalize_policy_target_mode(mode)
    default_target = np.asarray(policy, dtype=np.float32)
    if normalized_mode != "sharpened":
        return default_target.tolist()

    sharpened = np.square(default_target)
    total = float(np.sum(sharpened))
    if total <= 0:
        return default_target.tolist()
    return (sharpened / total).tolist()


def value_target_bucket_for_move_index(move_index: int) -> str:
    normalized_move_index = max(0, int(move_index))
    for upper_bound, bucket_name, _strength in VALUE_TARGET_BUCKETS:
        if normalized_move_index < upper_bound:
            return bucket_name
    return "late"


def _phase_aware_value_target_strength(move_index: int) -> float:
    bucket_name = value_target_bucket_for_move_index(move_index)
    for _upper_bound, candidate_bucket_name, strength in VALUE_TARGET_BUCKETS:
        if candidate_bucket_name == bucket_name:
            return strength
    return 3.0


def _hybrid_value_target_weights(move_index: int) -> dict[str, float]:
    return HYBRID_VALUE_TARGET_WEIGHTS[value_target_bucket_for_move_index(move_index)]


def _sharpen_value_magnitude(value: float, *, strength: float) -> float:
    bounded_value = max(0.0, min(1.0, float(value)))
    if bounded_value <= 0.0 or bounded_value >= 1.0:
        return bounded_value
    return 1.0 - math.pow(1.0 - bounded_value, float(strength))


def _blend_hybrid_value_target(*, outcome_value: float, search_value: float, move_index: int) -> float:
    outcome_sign = 0.0 if outcome_value == 0 else math.copysign(1.0, outcome_value)
    if outcome_sign == 0.0:
        return 0.0

    weights = _hybrid_value_target_weights(move_index)
    outcome_magnitude = abs(max(-1.0, min(1.0, float(outcome_value))))
    search_magnitude = abs(max(-1.0, min(1.0, float(search_value))))
    blended_magnitude = (weights["outcome"] * outcome_magnitude) + (weights["search"] * search_magnitude)
    return math.copysign(max(-1.0, min(1.0, blended_magnitude)), outcome_sign)


def build_value_target(
    value: float,
    *,
    mode: str = DEFAULT_VALUE_TARGET_MODE,
    move_index: int = 0,
) -> float:
    normalized_mode = normalize_value_target_mode(mode)
    bounded_value = max(-1.0, min(1.0, float(value)))
    if normalized_mode != "sharpened":
        if normalized_mode != PHASE_AWARE_VALUE_TARGET_MODE:
            return bounded_value
        strength = _phase_aware_value_target_strength(move_index)
    else:
        strength = 2.0

    magnitude = _sharpen_value_magnitude(abs(bounded_value), strength=strength)
    return math.copysign(magnitude, bounded_value)


def canonical_value_target(
    *,
    outcome_value: float,
    search_value: float,
    move_index: int = 0,
    mode: str = DEFAULT_VALUE_TARGET_MODE,
) -> float:
    normalized_mode = normalize_value_target_mode(mode)
    if normalized_mode == HYBRID_VALUE_TARGET_MODE:
        return _blend_hybrid_value_target(
            outcome_value=outcome_value,
            search_value=search_value,
            move_index=move_index,
        )

    if normalized_mode not in {"sharpened", PHASE_AWARE_VALUE_TARGET_MODE}:
        return build_value_target(outcome_value, mode=normalized_mode)

    outcome_sign = 0.0 if outcome_value == 0 else math.copysign(1.0, outcome_value)
    if outcome_sign == 0.0:
        return 0.0

    # When search_value is effectively zero (e.g. classic MCTS at a 50/50 position),
    # there is no confidence signal to sharpen — fall back to the raw outcome value.
    if abs(search_value) < 1e-9:
        return float(outcome_value)

    sharpened_magnitude = build_value_target(abs(search_value), mode=normalized_mode, move_index=move_index)
    return outcome_sign * abs(sharpened_magnitude)


def derive_self_play_value_target(
    *,
    outcome_value: float,
    search_value: float,
    move_index: int = 0,
    mode: str = DEFAULT_VALUE_TARGET_MODE,
) -> float:
    return canonical_value_target(
        outcome_value=outcome_value,
        search_value=search_value,
        move_index=move_index,
        mode=mode,
    )


def add_search_option_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fpu-mode", default=DEFAULT_SEARCH_OPTIONS["fpu_mode"])
    parser.add_argument("--reuse-subtree", action="store_true")
    parser.add_argument("--normalize-values", action="store_true")
    parser.add_argument("--root-policy-mode", choices=sorted(SUPPORTED_ROOT_POLICY_MODES), default=DEFAULT_SEARCH_OPTIONS["root_policy_mode"])
    parser.add_argument("--tactical-root-bias", type=float, default=DEFAULT_SEARCH_OPTIONS["tactical_root_bias"])
    parser.add_argument("--value-trust-enabled", action="store_true")
    parser.add_argument("--value-trust-opening", type=float, default=None)
    parser.add_argument("--value-trust-midgame", type=float, default=None)
    parser.add_argument("--value-trust-late", type=float, default=None)


def value_trust_schedule_from_args(args: argparse.Namespace) -> dict[str, bool | float] | None:
    opening = getattr(args, "value_trust_opening", None)
    midgame = getattr(args, "value_trust_midgame", None)
    late = getattr(args, "value_trust_late", None)
    enabled = bool(getattr(args, "value_trust_enabled", False))
    if not enabled and opening is None and midgame is None and late is None:
        return None

    return {
        "enabled": enabled,
        "opening": 1.0 if opening is None else float(opening),
        "midgame": 1.0 if midgame is None else float(midgame),
        "late": 1.0 if late is None else float(late),
    }


def search_options_from_args(args: argparse.Namespace) -> SearchOptions:
    return build_search_options(
        fpu_mode=args.fpu_mode,
        reuse_subtree=args.reuse_subtree,
        normalize_values=args.normalize_values,
        root_policy_mode=args.root_policy_mode,
        tactical_root_bias=args.tactical_root_bias,
        value_trust_schedule=value_trust_schedule_from_args(args),
    )


class Evaluator:
    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        raise NotImplementedError


class HeuristicEvaluator(Evaluator):
    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        legal_moves = game.possible_moves()
        priors = np.zeros(PITS_PER_PLAYER, dtype=np.float32)
        if legal_moves:
            offset = game.current_player * PITS_PER_PLAYER
            weights = np.array([game.pits[offset + move] + 1 for move in legal_moves], dtype=np.float32)
            weights /= np.sum(weights)
            for idx, move in enumerate(legal_moves):
                priors[move] = weights[idx]

        store_delta = game.captured_seeds[game.current_player] - game.captured_seeds[1 - game.current_player]
        value = float(np.tanh(store_delta / 12.0))
        return priors, value


class CheckpointEvaluator(Evaluator):
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        input_encoding: str = DEFAULT_INPUT_ENCODING,
        cache_size: int = 0,
    ):
        checkpoint_path = checkpoint_path.resolve()
        npz = np.load(checkpoint_path)
        self.residual_blocks = self._extract_residual_blocks(npz)
        self.hidden_layers = [] if self.residual_blocks else self._extract_hidden_layers(npz)
        self.w_input = npz["w_input"] if "w_input" in npz else None
        self.b_input = npz["b_input"] if "b_input" in npz else None
        self.w_policy = npz["w_policy"]
        self.b_policy = npz["b_policy"]
        self.w_value = npz["w_value"]
        self.b_value = npz["b_value"]
        self.w_policy_hidden = None
        self.b_policy_hidden = None
        self.w_value_hidden = None
        self.b_value_hidden = None
        self.uses_specialized_heads = False
        specialized_head_keys = ["w_policy_hidden", "b_policy_hidden", "w_value_hidden", "b_value_hidden"]
        present_specialized_head_keys = [key for key in specialized_head_keys if key in npz]
        if present_specialized_head_keys:
            missing_specialized_head_keys = [key for key in specialized_head_keys if key not in npz]
            if missing_specialized_head_keys:
                missing = ", ".join(missing_specialized_head_keys)
                raise ValueError(f"checkpoint is missing specialized head weights: {missing}")
            self.w_policy_hidden = npz["w_policy_hidden"]
            self.b_policy_hidden = npz["b_policy_hidden"]
            self.w_value_hidden = npz["w_value_hidden"]
            self.b_value_hidden = npz["b_value_hidden"]
            self.uses_specialized_heads = True
            self._validate_specialized_head_shapes()
        self.input_encoding = input_encoding
        self.checkpoint_identity = self._checkpoint_identity_for(checkpoint_path)
        self.cache = EvalCache(cache_size) if cache_size > 0 else None

    @property
    def cache_stats(self) -> dict[str, int | bool]:
        if self.cache is None:
            return {"enabled": False, "hits": 0, "misses": 0, "size": 0}
        return {
            "enabled": True,
            "hits": self.cache.hits,
            "misses": self.cache.misses,
            "size": self.cache.size,
        }

    def _checkpoint_identity_for(self, checkpoint_path: Path) -> str:
        stat = checkpoint_path.stat()
        return f"{checkpoint_path}:{stat.st_mtime_ns}:{stat.st_size}"

    def _cache_key_for(self, game: KalahGame) -> str:
        canonical_state = json.dumps(game.to_state(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return "|".join((self.checkpoint_identity, self.input_encoding, canonical_state))

    def _clone_cached_result(self, result: tuple[np.ndarray, float]) -> tuple[np.ndarray, float]:
        policy, value = result
        return policy.copy(), value

    def _validate_specialized_head_shapes(self) -> None:
        assert self.w_policy_hidden is not None
        assert self.b_policy_hidden is not None
        assert self.w_value_hidden is not None
        assert self.b_value_hidden is not None

        trunk_size = self.w_input.shape[1] if self.w_input is not None else self.hidden_layers[-1][0].shape[1]
        policy_hidden_size = self.b_policy_hidden.shape[0]
        value_hidden_size = self.b_value_hidden.shape[0]

        if self.w_policy_hidden.shape != (trunk_size, policy_hidden_size):
            raise ValueError(
                f"checkpoint w_policy_hidden must have shape ({trunk_size}, {policy_hidden_size})"
            )
        if self.w_value_hidden.shape != (trunk_size, value_hidden_size):
            raise ValueError(
                f"checkpoint w_value_hidden must have shape ({trunk_size}, {value_hidden_size})"
            )
        if self.w_policy.shape[0] != policy_hidden_size:
            raise ValueError(
                f"checkpoint w_policy must have {policy_hidden_size} input rows when specialized heads are present"
            )
        if self.w_value.shape[0] != value_hidden_size:
            raise ValueError(
                f"checkpoint w_value must have {value_hidden_size} input rows when specialized heads are present"
            )

    def _extract_residual_blocks(self, npz: np.lib.npyio.NpzFile) -> list[tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]:
        if "w_input" not in npz or "b_input" not in npz:
            return []

        blocks = []
        index = 1
        while f"w_residual_{index}_1" in npz and f"b_residual_{index}_1" in npz:
            blocks.append(
                (
                    (npz[f"w_residual_{index}_1"], npz[f"b_residual_{index}_1"]),
                    (npz[f"w_residual_{index}_2"], npz[f"b_residual_{index}_2"]),
                )
            )
            index += 1
        return blocks

    def _extract_hidden_layers(self, npz: np.lib.npyio.NpzFile) -> list[tuple[np.ndarray, np.ndarray]]:
        indexed_layers: list[tuple[np.ndarray, np.ndarray]] = []
        index = 1
        while f"w_hidden_{index}" in npz and f"b_hidden_{index}" in npz:
            indexed_layers.append((npz[f"w_hidden_{index}"], npz[f"b_hidden_{index}"]))
            index += 1

        if indexed_layers:
            return indexed_layers

        return [(npz["w1"], npz["b1"]), (npz["w2"], npz["b2"])]

    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        if self.cache is not None:
            cache_key = self._cache_key_for(game)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return self._clone_cached_result(cached)

        x = np.asarray(encode_state(game.to_state(), input_encoding=self.input_encoding), dtype=np.float32)

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
        if self.uses_specialized_heads:
            assert self.b_policy_hidden is not None
            assert self.w_value_hidden is not None
            assert self.b_value_hidden is not None
            policy_hidden = np.maximum(0.0, (hidden @ self.w_policy_hidden) + self.b_policy_hidden)
            value_hidden = np.maximum(0.0, (hidden @ self.w_value_hidden) + self.b_value_hidden)

        policy_logits = (policy_hidden @ self.w_policy) + self.b_policy
        value_logit = float(((value_hidden @ self.w_value) + self.b_value).reshape(-1)[0])
        result = (softmax(policy_logits), float(np.tanh(value_logit)))
        if self.cache is not None:
            self.cache.put(cache_key, result)
            return self._clone_cached_result(result)
        return result


@dataclass
class Node:
    game: KalahGame
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)
    expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def child_for_action(self, action: int) -> Node | None:
        return self.children.get(action)


class PUCT:
    def __init__(
        self,
        evaluator: Evaluator,
        simulations: int,
        c_puct: float,
        rng: random.Random,
        root: Node | None = None,
        *,
        fpu_mode: str = DEFAULT_SEARCH_OPTIONS["fpu_mode"],
        reuse_subtree: bool = DEFAULT_SEARCH_OPTIONS["reuse_subtree"],
        normalize_values: bool = DEFAULT_SEARCH_OPTIONS["normalize_values"],
        root_policy_mode: str = DEFAULT_SEARCH_OPTIONS["root_policy_mode"],
        tactical_root_bias: float = DEFAULT_SEARCH_OPTIONS["tactical_root_bias"],
        value_trust_schedule: dict | None = None,
        ablation_mode: str | None = None,
    ):
        self.evaluator = evaluator
        self.simulations = simulations
        self.c_puct = c_puct
        self.rng = rng
        self.root = root
        self.fpu_mode = fpu_mode
        self.reuse_subtree = reuse_subtree
        self.normalize_values = normalize_values
        self.root_policy_mode = normalize_root_policy_mode(root_policy_mode)
        self.tactical_root_bias = float(tactical_root_bias)
        self.value_trust_schedule = normalize_value_trust_schedule(value_trust_schedule)
        self.ablation_mode = build_mode_config(ablation_mode or "full")
        self._last_root: Node | None = None
        self.search_options = build_search_options(
            fpu_mode=self.fpu_mode,
            reuse_subtree=self.reuse_subtree,
            normalize_values=self.normalize_values,
            root_policy_mode=self.root_policy_mode,
            tactical_root_bias=self.tactical_root_bias,
            value_trust_schedule=self.value_trust_schedule,
        )

    def run(
        self,
        root_game: KalahGame,
        *,
        dirichlet_alpha: float | None = None,
        dirichlet_epsilon: float = 0.25,
    ) -> tuple[np.ndarray, Node]:
        root = self._root_for(root_game)
        self._expand(
            root,
            apply_dirichlet=dirichlet_alpha is not None,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            is_root=True,
        )

        for _ in range(self.simulations):
            value = self._search(root)
            root.visit_count += 1
            root.value_sum += value

        visits = np.zeros(PITS_PER_PLAYER, dtype=np.float32)
        for move, child in root.children.items():
            visits[move] = child.visit_count
        self._last_root = root
        return visits, root

    def root_summary(self) -> dict:
        if self._last_root is None:
            raise ValueError("root_summary requires run() to be called first")

        child_stats = []
        for action, child in sorted(self._last_root.children.items()):
            child_stats.append(
                {
                    "move": int(action),
                    "visits": int(child.visit_count),
                    "q_value": float(child.q_value) if child.visit_count else 0.0,
                }
            )
        legal_moves = sorted(self._last_root.children)
        selected_move = None if not legal_moves else int(self.select_root_move(self._last_root, legal_moves))
        return {
            "selected_move": selected_move,
            "child_stats": child_stats,
            "value_trust": self._value_trust_summary_for(self._last_root.game),
        }

    def _value_trust_summary_for(self, game: KalahGame) -> dict:
        phase_key = self._value_trust_phase_key_for(game)
        schedule = self.value_trust_schedule or {
            "enabled": False,
            "opening": 1.0,
            "midgame": 1.0,
            "late": 1.0,
        }
        return {
            "enabled": bool(schedule["enabled"]),
            "phase_bucket": phase_key,
            "effective_multiplier": float(self._effective_value_trust_multiplier_for(game)),
            "schedule": {
                "opening": float(schedule["opening"]),
                "midgame": float(schedule["midgame"]),
                "late": float(schedule["late"]),
            },
        }

    def _value_trust_phase_key_for(self, game: KalahGame) -> str:
        seeds_remaining = sum(game.pits)
        if seeds_remaining <= 12:
            return "late"
        if seeds_remaining <= 24:
            return "midgame"
        return "opening"

    def _effective_value_trust_multiplier_for(self, game: KalahGame) -> float:
        if self.value_trust_schedule is None or not self.value_trust_schedule["enabled"]:
            return 1.0
        return float(self.value_trust_schedule[self._value_trust_phase_key_for(game)])

    def select_root_move(self, root: Node, legal_moves: list[int]) -> int:
        if not legal_moves:
            raise ValueError("select_root_move requires at least one legal move")

        if self.root_policy_mode == "deterministic":
            return max(
                legal_moves,
                key=lambda move: (
                    root.children[move].visit_count,
                    root.children[move].q_value,
                    root.children[move].prior,
                    -move,
                ),
            )

        return max(legal_moves, key=lambda move: (root.children[move].visit_count, -move))

    def _root_for(self, root_game: KalahGame) -> Node:
        if self.reuse_subtree and self.root is not None and self._same_state(self.root.game, root_game):
            return self.root
        return Node(game=root_game.clone())

    def _same_state(self, left: KalahGame, right: KalahGame) -> bool:
        return left.to_state() == right.to_state()

    def _search(self, node: Node) -> float:
        terminal = terminal_value(node.game)
        if terminal is not None:
            return terminal

        if not node.expanded:
            _, value = self._expand(node, apply_dirichlet=False, dirichlet_alpha=None, dirichlet_epsilon=0.0, is_root=False)
            return value

        if not node.children:
            return 0.0

        child = self._select_child(node)

        value = self._search(child)
        if child.game.current_player != node.game.current_player:
            value = -value
        child.visit_count += 1
        child.value_sum += value
        return value

    def _select_child(self, node: Node) -> Node:
        total_visits = sum(child.visit_count for child in node.children.values())
        total_visits = max(1, total_visits)
        items = list(node.children.items())
        q_values = [self._child_q_value(node, child) for _move, child in items]
        if self.normalize_values:
            q_values = self._normalize_child_values(q_values)
        value_trust_multiplier = self._effective_value_trust_multiplier_for(node.game)

        best_child = None
        best_score = -float("inf")

        for (move, child), q_value in zip(items, q_values):
            u_score = self.c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visit_count)
            score = (q_value * value_trust_multiplier) + u_score
            if score > best_score:
                best_score = score
                best_child = child

        assert best_child is not None
        return best_child

    def _child_q_value(self, parent: Node, child: Node) -> float:
        if child.visit_count > 0:
            return child.q_value
        if self.fpu_mode in {"parent_q", "parent_value"}:
            return parent.q_value
        return 0.0

    def _normalize_child_values(self, values: list[float]) -> list[float]:
        if not values:
            return []

        lower = min(values)
        upper = max(values)
        if upper <= lower:
            return list(values)

        return [self._normalize_value(value, lower, upper) for value in values]

    def _normalize_value(self, value: float, lower: float, upper: float) -> float:
        if upper <= lower:
            return value
        return (value - lower) / (upper - lower)

    def _expand(
        self,
        node: Node,
        *,
        apply_dirichlet: bool,
        dirichlet_alpha: float | None,
        dirichlet_epsilon: float,
        is_root: bool,
    ) -> tuple[np.ndarray, float]:
        priors, value = self.evaluator.evaluate(node.game)
        legal_moves = node.game.possible_moves()

        if not self.ablation_mode["use_policy"]:
            priors = flat_legal_priors(legal_moves)
        if not self.ablation_mode["use_value"]:
            value = neutral_value()

        masked = np.zeros(PITS_PER_PLAYER, dtype=np.float32)
        if legal_moves:
            masked[legal_moves] = priors[legal_moves]

        total = float(np.sum(masked))
        if total <= 0 and legal_moves:
            uniform = 1.0 / len(legal_moves)
            for move in legal_moves:
                masked[move] = uniform
        elif total > 0:
            masked /= total

        if apply_dirichlet and legal_moves and dirichlet_alpha is not None and dirichlet_alpha > 0:
            noise = np.random.default_rng(self.rng.randint(0, 2**31 - 1)).dirichlet([dirichlet_alpha] * len(legal_moves))
            for index, move in enumerate(legal_moves):
                masked[move] = (1.0 - dirichlet_epsilon) * masked[move] + dirichlet_epsilon * float(noise[index])

        if is_root:
            masked = self.apply_tactical_root_bias(node.game, masked)

        for move in legal_moves:
            if move not in node.children:
                child_game = node.game.clone()
                child_game.move(child_game.pit_index(move))
                node.children[move] = Node(game=child_game, prior=float(masked[move]))
            else:
                node.children[move].prior = float(masked[move])

        node.expanded = True
        return masked, value

    def apply_tactical_root_bias(self, game: KalahGame, priors: np.ndarray) -> np.ndarray:
        legal_moves = game.possible_moves()
        biased = np.asarray(priors, dtype=np.float32).copy()
        if self.tactical_root_bias <= 0 or not legal_moves:
            return biased

        tactical_moves = [move for move in legal_moves if self._is_immediate_tactical_move(game, move)]
        if not tactical_moves:
            return biased

        for move in tactical_moves:
            biased[move] += self.tactical_root_bias

        total = float(np.sum(biased[legal_moves]))
        if total <= 0:
            return biased

        normalized = np.zeros_like(biased)
        normalized[legal_moves] = biased[legal_moves] / total
        return normalized.astype(np.float32)

    def _is_immediate_tactical_move(self, game: KalahGame, move: int) -> bool:
        return self._is_immediate_extra_turn(game, move) or self._is_immediate_capture(game, move)

    def _is_immediate_extra_turn(self, game: KalahGame, move: int) -> bool:
        simulated = game.clone()
        player = simulated.current_player
        if not simulated.move(simulated.pit_index(move)):
            return False
        return simulated.current_player == player

    def _is_immediate_capture(self, game: KalahGame, move: int) -> bool:
        absolute_index = game.pit_index(move)
        seeds = game.pits[absolute_index]
        store_before = game.captured_seeds[game.current_player]
        simulated = game.clone()
        if not simulated.move(simulated.pit_index(move)):
            return False

        store_gain = simulated.captured_seeds[game.current_player] - store_before
        return store_gain > self._own_store_passes(move, seeds)

    def _own_store_passes(self, move: int, seeds: int) -> int:
        distance_to_store = PITS_PER_PLAYER - move
        if seeds < distance_to_store:
            return 0
        return 1 + ((seeds - distance_to_store) // ((PITS_PER_PLAYER * 2) + 1))


def terminal_value(game: KalahGame) -> float | None:
    if not game.over():
        return None
    if game.winner is None:
        return 0.0
    return 1.0 if game.winner == game.current_player else -1.0


def policy_from_visits(visits: np.ndarray, legal_moves: list[int], temperature: float) -> list[float]:
    policy = np.zeros(PITS_PER_PLAYER, dtype=np.float32)
    if not legal_moves:
        return policy.tolist()

    legal_visits = np.array([visits[move] for move in legal_moves], dtype=np.float64)
    if temperature <= 0.05:
        best_move = legal_moves[int(np.argmax(legal_visits))]
        policy[best_move] = 1.0
        return policy.tolist()

    scaled = np.power(legal_visits + 1e-8, 1.0 / temperature)
    scaled_sum = np.sum(scaled)
    if scaled_sum <= 0:
        scaled = np.ones_like(scaled)
        scaled_sum = np.sum(scaled)
    scaled /= scaled_sum
    for idx, move in enumerate(legal_moves):
        policy[move] = float(scaled[idx])
    return policy.tolist()


def sample_move(policy: list[float], legal_moves: list[int], rng: random.Random) -> int:
    threshold = rng.random()
    running = 0.0
    for move in legal_moves:
        running += float(policy[move])
        if threshold <= running:
            return move
    return legal_moves[-1]


def outcome_for_player(winner: int | None, player: int) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner == player else -1.0


def search_seed_for_classic_mcts(base_seed: int, game_index: int, ply: int) -> int:
    """Deterministic per-position seed for classic MCTS in self-play."""
    return (base_seed * 1_000_003) + (game_index * 10_007) + ply


def visits_from_classic_mcts_root(root: "Any") -> list[float]:
    """Convert classic MCTS root children visit counts to a visits array indexed by action (0-5)."""
    result = [0.0] * PITS_PER_PLAYER
    for action, child in root.children.items():
        if 0 <= action < PITS_PER_PLAYER:
            result[action] = float(child.visits)
    return result


def value_from_classic_mcts_root(root: "Any") -> float:
    """Win rate from root, converted to [-1, 1] from the current player's perspective."""
    if root.visits == 0:
        return 0.0
    return 2.0 * (root.wins / float(root.visits)) - 1.0


def teacher_targets_for_state(
    *,
    state: dict,
    ply: int,
    opening_cache,
    player_mode: str,
    search_profile: dict,
    teacher_runner,
    policy_target_mode: str,
) -> tuple[list[float], list[float], float, str, dict, str, str, dict | None]:
    if opening_cache is not None:
        cached = opening_cache.lookup(state, ply=ply)
        if cached is not None:
            cached_policy = list(cached["policy"])
            cached_policy_target = build_policy_target_from_distribution(cached_policy, mode=policy_target_mode)
            cached_search_profile = dict(getattr(opening_cache, "search_profile", {}) or {})
            cached_search_profile_hash = str(
                cached.get("provenance", {}).get("search_profile_hash")
                or cached_search_profile.get("hash")
                or ""
            )
            return (
                cached_policy,
                cached_policy_target,
                float(cached["value"]),
                "opening_cache",
                cached_search_profile,
                cached_search_profile_hash,
                str(policy_target_mode),
                None,
            )

    gameplay_policy, policy_target, value, teacher_root_summary = teacher_runner()
    return (
        list(gameplay_policy),
        list(policy_target),
        float(value),
        str(player_mode),
        dict(search_profile),
        str(search_profile["hash"]),
        str(policy_target_mode),
        teacher_root_summary,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None, help="Optional model checkpoint .npz")
    parser.add_argument("--opponent-pool-config", default=None, help="Optional JSON config for opponent checkpoint pool")
    parser.add_argument("--opening-cache", default=None, help="Optional JSON opening-cache artifact path")
    parser.add_argument("--input-encoding", choices=SUPPORTED_INPUT_ENCODINGS, default=DEFAULT_INPUT_ENCODING)
    parser.add_argument("--simulations", type=int, default=96)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--temperature-threshold", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-late", type=float, default=0.1)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    parser.add_argument("--schedule-progress-mode", choices=["none", "linear"], default="none")
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--total-iterations", type=int, default=1)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--evaluator-cache-size", type=non_negative_int, default=0)
    parser.add_argument("--seed-sweep", default=None, help="Comma-separated seeds to mix within one run")
    parser.add_argument("--tree-reuse-enabled", action="store_true")
    parser.add_argument("--policy-target-mode", choices=sorted(SUPPORTED_POLICY_TARGET_MODES), default=DEFAULT_POLICY_TARGET_MODE)
    parser.add_argument("--value-target-mode", choices=sorted(SUPPORTED_VALUE_TARGET_MODES), default=DEFAULT_VALUE_TARGET_MODE)
    parser.add_argument(
        "--player-mode",
        choices=sorted(SUPPORTED_PLAYER_MODES),
        default=DEFAULT_PLAYER_MODE,
        help="Search algorithm used for game generation: 'puct' (default) or 'classic_mcts'",
    )
    add_search_option_args(parser)
    return parser.parse_args()


def merge_worker_shards(results: list[dict], out_path: Path) -> int:
    rows_written = 0
    with out_path.open("w", encoding="utf-8") as out_handle:
        for result in results:
            rows_written += int(result["rows_written"])
            with Path(result["shard_path"]).open("r", encoding="utf-8") as shard_handle:
                for line in shard_handle:
                    out_handle.write(line)
    return rows_written


def schedule_exploration_params(
    *,
    mode: str,
    iteration: int,
    total_iterations: int,
    temperature: float,
    temperature_late: float,
    temperature_threshold: int,
    dirichlet_epsilon: float,
) -> dict[str, float | int]:
    if mode == "none":
        return {
            "temperature": float(temperature),
            "temperature_late": float(temperature_late),
            "temperature_threshold": int(temperature_threshold),
            "dirichlet_epsilon": float(dirichlet_epsilon),
        }

    total = max(1, int(total_iterations))
    current = min(max(1, int(iteration)), total)
    progress = (current - 1) / max(1, total - 1)

    decay = 1.0 - (0.5 * progress)
    threshold_decay = 1.0 - (0.5 * progress)
    epsilon_decay = 1.0 - (0.6 * progress)

    return {
        "temperature": max(0.1, float(temperature) * decay),
        "temperature_late": max(0.02, float(temperature_late) * decay),
        "temperature_threshold": max(2, int(round(float(temperature_threshold) * threshold_decay))),
        "dirichlet_epsilon": max(0.05, float(dirichlet_epsilon) * epsilon_decay),
    }


def partition_counts(total: int, workers: int) -> list[int]:
    workers = max(1, workers)
    base = total // workers
    remainder = total % workers
    return [base + (1 if i < remainder else 0) for i in range(workers)]


def parse_seed_pool(seed: int, seed_sweep: str | None) -> list[int]:
    if not seed_sweep:
        return [int(seed)]

    pool = [int(value.strip()) for value in seed_sweep.split(",") if value.strip()]
    return pool or [int(seed)]


def build_checkpoint_evaluator(
    checkpoint_path: Path,
    *,
    input_encoding: str,
    cache_size: int,
) -> CheckpointEvaluator:
    if cache_size > 0:
        return CheckpointEvaluator(checkpoint_path, input_encoding=input_encoding, cache_size=cache_size)
    return CheckpointEvaluator(checkpoint_path, input_encoding=input_encoding)


def run_self_play_worker(
    *,
    worker_id: int,
    start_index: int,
    games: int,
    seed: int,
    seed_pool: list[int],
    checkpoint: str | None,
    input_encoding: str,
    simulations: int,
    c_puct: float,
    temperature_threshold: int,
    temperature: float,
    temperature_late: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    max_moves: int,
    shard_path: str,
    evaluator_cache_size: int = 0,
    tree_reuse_enabled: bool = False,
    fpu_mode: str = DEFAULT_SEARCH_OPTIONS["fpu_mode"],
    reuse_subtree: bool = DEFAULT_SEARCH_OPTIONS["reuse_subtree"],
    normalize_values: bool = DEFAULT_SEARCH_OPTIONS["normalize_values"],
    root_policy_mode: str = DEFAULT_SEARCH_OPTIONS["root_policy_mode"],
    tactical_root_bias: float = DEFAULT_SEARCH_OPTIONS["tactical_root_bias"],
    policy_target_mode: str = DEFAULT_POLICY_TARGET_MODE,
    value_target_mode: str = DEFAULT_VALUE_TARGET_MODE,
    player_mode: str = DEFAULT_PLAYER_MODE,
    value_trust_schedule: dict | None = None,
    opponent_pool_config: str | None = None,
    opening_cache=None,
    opening_cache_path: str | None = None,
) -> dict:
    shard = Path(shard_path)
    policy_target_mode = normalize_policy_target_mode(policy_target_mode)
    value_target_mode = normalize_value_target_mode(value_target_mode)

    if player_mode not in SUPPORTED_PLAYER_MODES:
        raise ValueError(f"Unsupported player_mode {player_mode!r}. Must be one of {sorted(SUPPORTED_PLAYER_MODES)}")

    evaluator: Evaluator | None = None
    opponent_evaluator_cache: dict[str, Evaluator] = {}
    opponent_checkpoints: list[str] = []
    if player_mode == "puct":
        opponent_checkpoints = load_opponent_checkpoints(opponent_pool_config)
        if checkpoint:
            evaluator = build_checkpoint_evaluator(
                Path(checkpoint),
                input_encoding=input_encoding,
                cache_size=evaluator_cache_size,
            )
        else:
            evaluator = HeuristicEvaluator()

    effective_reuse_subtree = bool(reuse_subtree or tree_reuse_enabled)
    normalized_search_options = build_search_options(
        fpu_mode=fpu_mode,
        reuse_subtree=effective_reuse_subtree,
        normalize_values=normalize_values,
        root_policy_mode=root_policy_mode,
        tactical_root_bias=tactical_root_bias,
        value_trust_schedule=value_trust_schedule,
    )
    profile_extra_fields: dict[str, str] = {}
    if opponent_checkpoints:
        profile_extra_fields["opponent_pool_fingerprint"] = opponent_pool_fingerprint(opponent_checkpoints)

    search_profile = build_search_profile(
        kind="self_play",
        player_mode=player_mode,
        simulations=simulations,
        c_puct=c_puct,
        search_options=normalized_search_options,
        extra_fields=profile_extra_fields,
    )

    if opening_cache is None and opening_cache_path:
        opening_cache_payload = json.loads(Path(opening_cache_path).read_text(encoding="utf-8"))
        opening_cache = load_opening_cache(opening_cache_payload)

    rows_written = 0
    with shard.open("w", encoding="utf-8") as handle:
        for local_index in range(games):
            global_index = start_index + local_index
            game_seed = seed_pool[global_index % len(seed_pool)]
            rng = random.Random((game_seed * 1_000_003) + global_index + (worker_id * 9_973))
            game = KalahGame.from_state(
                {
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                }
            )
            positions: list[tuple[list[float], list[float], int, float, int, str, dict, str, str, dict | None]] = []
            reusable_root: Node | None = None
            opponent_checkpoint_for_game: str | None = None
            if opponent_checkpoints:
                opponent_checkpoint_for_game = sample_opponent_checkpoint(
                    opponent_checkpoints,
                    base_seed=seed,
                    game_index=global_index,
                    worker_id=worker_id,
                )

            for ply in range(max_moves):
                if game.over():
                    break

                legal_moves = game.possible_moves()
                if not legal_moves:
                    break

                state = game.to_state()
                puct_root: Node | None = None
                temp = temperature if ply < temperature_threshold else temperature_late

                def run_teacher() -> tuple[list[float], list[float], float, dict | None]:
                    nonlocal puct_root
                    if player_mode == "classic_mcts":
                        mcts_seed = search_seed_for_classic_mcts(seed, global_index, ply)
                        classic_kwargs = {}
                        if "value_trust_schedule" in normalized_search_options:
                            classic_kwargs["value_trust_schedule"] = normalized_search_options["value_trust_schedule"]
                        mcts = ClassicMCTS(game.clone(), simulations=simulations, seed=mcts_seed, **classic_kwargs)
                        mcts_root = mcts.search_root()
                        mcts_summary = None
                        if "value_trust_schedule" in normalized_search_options:
                            mcts_summary = mcts.root_summary()
                        visits = np.array(visits_from_classic_mcts_root(mcts_root), dtype=np.float32)
                        gameplay_policy = policy_from_visits(visits, legal_moves=legal_moves, temperature=temp)
                        return (
                            gameplay_policy,
                            build_policy_target(
                                visits,
                                legal_moves=legal_moves,
                                temperature=temp,
                                mode=policy_target_mode,
                            ),
                            value_from_classic_mcts_root(mcts_root),
                            mcts_summary,
                        )

                    selected_evaluator = evaluator
                    if game.current_player == 1 and opponent_checkpoint_for_game is not None:
                        selected_evaluator = opponent_evaluator_cache.get(opponent_checkpoint_for_game)
                        if selected_evaluator is None:
                            selected_evaluator = build_checkpoint_evaluator(
                                Path(opponent_checkpoint_for_game),
                                input_encoding=input_encoding,
                                cache_size=evaluator_cache_size,
                            )
                            opponent_evaluator_cache[opponent_checkpoint_for_game] = selected_evaluator

                    puct_kwargs = {}
                    if "value_trust_schedule" in normalized_search_options:
                        puct_kwargs["value_trust_schedule"] = normalized_search_options["value_trust_schedule"]
                    search = PUCT(
                        evaluator=selected_evaluator,
                        simulations=simulations,
                        c_puct=c_puct,
                        rng=rng,
                        root=reusable_root,
                        fpu_mode=str(normalized_search_options["fpu_mode"]),
                        reuse_subtree=bool(normalized_search_options["reuse_subtree"]),
                        normalize_values=bool(normalized_search_options["normalize_values"]),
                        root_policy_mode=str(normalized_search_options["root_policy_mode"]),
                        tactical_root_bias=float(normalized_search_options["tactical_root_bias"]),
                        **puct_kwargs,
                    )
                    visits, puct_root = search.run(
                        game,
                        dirichlet_alpha=dirichlet_alpha if ply < temperature_threshold else None,
                        dirichlet_epsilon=dirichlet_epsilon,
                    )
                    gameplay_policy = policy_from_visits(visits, legal_moves=legal_moves, temperature=temp)
                    root_summary = None
                    if "value_trust_schedule" in normalized_search_options and hasattr(search, "root_summary"):
                        root_summary = search.root_summary()
                    return (
                        gameplay_policy,
                        build_policy_target(
                            visits,
                            legal_moves=legal_moves,
                            temperature=temp,
                            mode=policy_target_mode,
                        ),
                        puct_root.q_value,
                        root_summary,
                    )

                (
                    gameplay_policy,
                    policy_target,
                    search_value,
                    teacher_source,
                    teacher_search_profile,
                    teacher_search_profile_hash,
                    policy_target_actual_mode,
                    teacher_root_summary,
                ) = teacher_targets_for_state(
                    state=state,
                    ply=ply,
                    opening_cache=opening_cache,
                    player_mode=player_mode,
                    search_profile=search_profile,
                    teacher_runner=run_teacher,
                    policy_target_mode=policy_target_mode,
                )
                positions.append(
                    (
                        encode_state(state, input_encoding=input_encoding),
                        policy_target,
                        game.current_player,
                        search_value,
                        ply,
                        teacher_source,
                        teacher_search_profile,
                        teacher_search_profile_hash,
                        policy_target_actual_mode,
                        teacher_root_summary,
                    )
                )

                move = sample_move(gameplay_policy, legal_moves=legal_moves, rng=rng)
                if player_mode == "classic_mcts" or teacher_source == "opening_cache":
                    reusable_root = None
                else:
                    reusable_root = puct_root.child_for_action(move) if effective_reuse_subtree else None
                absolute_move = game.pit_index(move)
                if not game.move(absolute_move):
                    break

            winner = game.winner
            for (
                state,
                policy,
                player,
                search_value,
                move_index,
                teacher_source,
                teacher_search_profile,
                teacher_search_profile_hash,
                policy_target_actual_mode,
                teacher_root_summary,
            ) in positions:
                row = {
                    "state": state,
                    "policy": policy,
                    "value": derive_self_play_value_target(
                        outcome_value=outcome_for_player(winner, player),
                        search_value=search_value,
                        move_index=move_index,
                        mode=value_target_mode,
                    ),
                    "player": int(player),
                    "move_index": int(move_index),
                    "winner": winner,
                    "teacher_source": teacher_source,
                    "policy_target_mode": policy_target_mode,
                    "policy_target_actual_mode": policy_target_actual_mode,
                    "value_target_mode": value_target_mode,
                    "search_profile": search_profile,
                    "search_profile_hash": search_profile["hash"],
                    "teacher_search_profile": teacher_search_profile,
                    "teacher_search_profile_hash": teacher_search_profile_hash,
                }
                if teacher_root_summary is not None:
                    row["teacher_root_summary"] = teacher_root_summary
                handle.write(json.dumps(row) + "\n")
                rows_written += 1

    return {
        "worker_id": worker_id,
        "rows_written": rows_written,
        "shard_path": str(shard),
        **cache_metrics_for([evaluator, *opponent_evaluator_cache.values()]),
    }


def main() -> None:
    args = parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workers = max(1, args.workers)
    seed_pool = parse_seed_pool(args.seed, args.seed_sweep)
    search_options = search_options_from_args(args)
    exploration = schedule_exploration_params(
        mode=args.schedule_progress_mode,
        iteration=args.iteration,
        total_iterations=args.total_iterations,
        temperature=args.temperature,
        temperature_late=args.temperature_late,
        temperature_threshold=args.temperature_threshold,
        dirichlet_epsilon=args.dirichlet_epsilon,
    )

    game_counts = partition_counts(args.games, workers)
    starts: list[int] = []
    cursor = 0
    for count in game_counts:
        starts.append(cursor)
        cursor += count
    with tempfile.TemporaryDirectory(prefix="azlite-self-play-shards-", dir=out_path.parent) as shard_dir:
        shard_paths = [str(Path(shard_dir) / f"worker_{worker_id}.jsonl") for worker_id in range(workers)]

        futures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for worker_id, (start_index, game_count) in enumerate(zip(starts, game_counts, strict=True)):
                if game_count <= 0:
                    Path(shard_paths[worker_id]).write_text("", encoding="utf-8")
                    continue

                worker_kwargs = {
                    "worker_id": worker_id,
                    "start_index": start_index,
                    "games": game_count,
                    "seed": args.seed,
                    "seed_pool": seed_pool,
                    "checkpoint": args.checkpoint,
                    "opponent_pool_config": args.opponent_pool_config,
                    "input_encoding": args.input_encoding,
                    "simulations": args.simulations,
                    "c_puct": args.c_puct,
                    "temperature_threshold": int(exploration["temperature_threshold"]),
                    "temperature": float(exploration["temperature"]),
                    "temperature_late": float(exploration["temperature_late"]),
                    "dirichlet_alpha": args.dirichlet_alpha,
                    "dirichlet_epsilon": float(exploration["dirichlet_epsilon"]),
                    "max_moves": args.max_moves,
                    "shard_path": shard_paths[worker_id],
                    "evaluator_cache_size": args.evaluator_cache_size,
                    "tree_reuse_enabled": args.tree_reuse_enabled,
                    "fpu_mode": str(search_options["fpu_mode"]),
                    "reuse_subtree": bool(search_options["reuse_subtree"]),
                    "normalize_values": bool(search_options["normalize_values"]),
                    "root_policy_mode": str(search_options["root_policy_mode"]),
                    "tactical_root_bias": float(search_options["tactical_root_bias"]),
                    "policy_target_mode": args.policy_target_mode,
                    "value_target_mode": args.value_target_mode,
                    "player_mode": args.player_mode,
                    "opening_cache_path": args.opening_cache,
                }
                if "value_trust_schedule" in search_options:
                    worker_kwargs["value_trust_schedule"] = search_options["value_trust_schedule"]
                futures.append(pool.submit(run_self_play_worker, **worker_kwargs))

            results = [future.result() for future in futures]
            results.sort(key=lambda item: item["worker_id"])

        rows_written = merge_worker_shards(results, out_path)

    print(f"wrote {rows_written} rows to {out_path}")
    cache_metrics = cache_metrics_for([])
    cache_metrics["cache_enabled"] = any(bool(result.get("cache_enabled", False)) for result in results)
    cache_metrics["cache_hits"] = sum(int(result.get("cache_hits", 0)) for result in results)
    cache_metrics["cache_misses"] = sum(int(result.get("cache_misses", 0)) for result in results)
    print(
        format_metrics_line(
            prefix="cache_metrics",
            extra_fields={"cache_enabled": cache_metrics.get("cache_enabled", False)},
            cache_hits=int(cache_metrics["cache_hits"]),
            cache_misses=int(cache_metrics["cache_misses"]),
        )
    )


if __name__ == "__main__":
    main()
