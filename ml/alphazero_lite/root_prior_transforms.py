from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np

from ml.alphazero_lite.capture_002_003_rule_collision_diagnostic import (
    simulate_move_rule_features,
)


PITS_PER_PLAYER = 6
ALL_TRANSFORM_NAMES = (
    "original_prior",
    "uniform_legal_prior",
    "extra_turn_damp_050",
    "extra_turn_damp_025",
    "no_extra_turn_capture_boost_2x",
    "hybrid_damp050_captureboost2x",
    "prior_temperature_2x",
)
FOLLOWUP_TRANSFORM_NAMES = (
    "seed4_extra_turn_damp_050_when_two_captures_noncapture5",
    "seed4_extra_turn_damp_025_when_two_captures_noncapture5",
    "seed4_extra_turn_damp_010_when_two_captures_noncapture5",
    "seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5",
)
ARENA_TRANSFORM_NAMES = (
    "seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5",
)
SUPPORTED_TRANSFORM_NAMES = ALL_TRANSFORM_NAMES + FOLLOWUP_TRANSFORM_NAMES

MoveFeatureAnnotations = dict[int, dict[str, int | bool]]


def normalize_legal_prior(
    prior: list[float] | np.ndarray, legal_moves: list[int]
) -> np.ndarray:
    normalized = np.zeros(PITS_PER_PLAYER, dtype=np.float32)
    if not legal_moves:
        return normalized

    values = np.asarray(prior, dtype=np.float32)
    normalized[legal_moves] = values[legal_moves]
    total = float(np.sum(normalized[legal_moves]))
    if total <= 0.0:
        normalized[legal_moves] = 1.0 / len(legal_moves)
        return normalized
    normalized[legal_moves] /= total
    return normalized


def move_feature_annotations_for(
    *, state: dict, legal_moves: list[int]
) -> MoveFeatureAnnotations:
    current_player = int(state.get("current_player", 0))
    side_to_move_pits = (
        list(state.get("player_pits") or [])
        if current_player == 0
        else list(state.get("opponent_pits") or [])
    )
    annotations: MoveFeatureAnnotations = {}
    for move in legal_moves:
        rule_features = simulate_move_rule_features(state=state, move=int(move))
        post_move_state = rule_features.get("post_move_state") or {}
        annotations[int(move)] = {
            "gives_extra_turn": bool(rule_features.get("extra_turn_available", False)),
            "produces_capture": bool(rule_features.get("capture_legal", False)),
            "seed_count": int(side_to_move_pits[int(move)]),
            "pit_index": int(move),
            "resulting_side_to_move": int(
                post_move_state.get("current_player", current_player)
            ),
        }
    return annotations


def _mass_shift(before: np.ndarray, after: np.ndarray, legal_moves: list[int]) -> float:
    if not legal_moves:
        return 0.0
    return round(
        0.5 * float(np.sum(np.abs(after[legal_moves] - before[legal_moves]))), 4
    )


def _telemetry(
    *,
    transform_name: str,
    legal_moves: list[int],
    before: np.ndarray,
    after: np.ndarray,
    move_feature_annotations: MoveFeatureAnnotations,
    scale_factors: dict[int, float] | None = None,
) -> dict:
    return {
        "transform_name": transform_name,
        "activated": any(
            not np.isclose(float(before[move]), float(after[move])) for move in legal_moves
        ),
        "legal_move_count": len(legal_moves),
        "context_counts": _context_counts(move_feature_annotations),
        "mass_shift": _mass_shift(before, after, legal_moves),
        "per_move": {
            str(move): {
                "before": round(float(before[move]), 4),
                "after": round(float(after[move]), 4),
                "delta": round(float(after[move] - before[move]), 4),
                "scale_factor": None
                if scale_factors is None
                else round(float(scale_factors.get(int(move), 1.0)), 4),
                "features": dict(move_feature_annotations.get(int(move), {})),
            }
            for move in legal_moves
        },
    }


def summarize_root_prior_telemetry(entries: list[dict] | None) -> dict:
    telemetry_entries = [entry for entry in (entries or []) if isinstance(entry, dict)]
    root_states_evaluated = len(telemetry_entries)
    activated_entries = [entry for entry in telemetry_entries if bool(entry.get("activated"))]
    activation_count = len(activated_entries)
    activation_rate = (
        None
        if root_states_evaluated <= 0
        else round(activation_count / root_states_evaluated, 4)
    )
    average_mass_shift = (
        None
        if activation_count <= 0
        else round(
            sum(float(entry.get("mass_shift", 0.0)) for entry in activated_entries)
            / activation_count,
            4,
        )
    )
    context_totals = {
        "seed4_extra_turn_count": 0,
        "no_extra_turn_capture_count": 0,
        "no_extra_turn_noncapture_seed5_count": 0,
        "legal_move_count": 0,
    }
    changed_pattern_counts: Counter[str] = Counter()
    for entry in activated_entries:
        context = entry.get("context_counts") if isinstance(entry.get("context_counts"), dict) else {}
        context_totals["seed4_extra_turn_count"] += int(
            context.get("seed4_extra_turn_count", 0)
        )
        context_totals["no_extra_turn_capture_count"] += int(
            context.get("no_extra_turn_capture_count", 0)
        )
        context_totals["no_extra_turn_noncapture_seed5_count"] += int(
            context.get("no_extra_turn_noncapture_seed5_count", 0)
        )
        context_totals["legal_move_count"] += int(entry.get("legal_move_count", 0))

        per_move = entry.get("per_move") if isinstance(entry.get("per_move"), dict) else {}
        for move_payload in per_move.values():
            if not isinstance(move_payload, dict):
                continue
            delta = float(move_payload.get("delta", 0.0))
            if np.isclose(delta, 0.0):
                continue
            features = move_payload.get("features") if isinstance(move_payload.get("features"), dict) else {}
            pattern = (
                f"extra_turn={int(bool(features.get('gives_extra_turn', False)))}|"
                f"capture={int(bool(features.get('produces_capture', False)))}|"
                f"seed_count={int(features.get('seed_count', 0))}|"
                f"delta_sign={'up' if delta > 0.0 else 'down'}"
            )
            changed_pattern_counts[pattern] += 1

    top_changed_move_feature_patterns = [
        {"pattern": pattern, "count": count}
        for pattern, count in changed_pattern_counts.most_common(5)
    ]
    return {
        "root_states_evaluated": root_states_evaluated,
        "activation_count": activation_count,
        "activation_rate": activation_rate,
        "average_mass_shift": average_mass_shift,
        "legal_move_context_counts": context_totals,
        "top_changed_move_feature_patterns": top_changed_move_feature_patterns,
    }


def merge_root_prior_telemetry_summaries(summaries: list[dict] | None) -> dict:
    normalized = [summary for summary in (summaries or []) if isinstance(summary, dict)]
    root_states_evaluated = sum(
        int(summary.get("root_states_evaluated", 0)) for summary in normalized
    )
    activation_count = sum(int(summary.get("activation_count", 0)) for summary in normalized)
    activation_rate = (
        None
        if root_states_evaluated <= 0
        else round(activation_count / root_states_evaluated, 4)
    )
    weighted_mass_shift = sum(
        float(summary.get("average_mass_shift", 0.0))
        * int(summary.get("activation_count", 0))
        for summary in normalized
        if summary.get("average_mass_shift") is not None
    )
    average_mass_shift = (
        None if activation_count <= 0 else round(weighted_mass_shift / activation_count, 4)
    )
    context_totals = {
        "seed4_extra_turn_count": 0,
        "no_extra_turn_capture_count": 0,
        "no_extra_turn_noncapture_seed5_count": 0,
        "legal_move_count": 0,
    }
    changed_pattern_counts: Counter[str] = Counter()
    for summary in normalized:
        context = summary.get("legal_move_context_counts")
        if isinstance(context, dict):
            for key in context_totals:
                context_totals[key] += int(context.get(key, 0))
        top_patterns = summary.get("top_changed_move_feature_patterns")
        if isinstance(top_patterns, list):
            for row in top_patterns:
                if not isinstance(row, dict):
                    continue
                pattern = row.get("pattern")
                if not isinstance(pattern, str) or not pattern:
                    continue
                changed_pattern_counts[pattern] += int(row.get("count", 0))

    top_changed_move_feature_patterns = [
        {"pattern": pattern, "count": count}
        for pattern, count in changed_pattern_counts.most_common(5)
    ]
    return {
        "root_states_evaluated": root_states_evaluated,
        "activation_count": activation_count,
        "activation_rate": activation_rate,
        "average_mass_shift": average_mass_shift,
        "legal_move_context_counts": context_totals,
        "top_changed_move_feature_patterns": top_changed_move_feature_patterns,
    }


def _scaled_transform(
    *,
    transform_name: str,
    legal_moves: list[int],
    original_prior: np.ndarray,
    move_feature_annotations: MoveFeatureAnnotations,
    scale_for_move,
) -> tuple[np.ndarray, dict]:
    adjusted = np.asarray(original_prior, dtype=np.float32).copy()
    scale_factors = {
        int(move): float(scale_for_move(int(move))) for move in legal_moves
    }
    for move in legal_moves:
        adjusted[move] *= scale_factors[int(move)]
    transformed = normalize_legal_prior(adjusted, legal_moves)
    return transformed, _telemetry(
        transform_name=transform_name,
        legal_moves=legal_moves,
        before=original_prior,
        after=transformed,
        move_feature_annotations=move_feature_annotations,
        scale_factors=scale_factors,
    )


def _context_counts(move_feature_annotations: MoveFeatureAnnotations) -> dict[str, int]:
    features = list(move_feature_annotations.values())
    return {
        "extra_turn_count": sum(1 for item in features if item["gives_extra_turn"]),
        "seed4_extra_turn_count": sum(
            1
            for item in features
            if item["gives_extra_turn"] and int(item["seed_count"]) == 4
        ),
        "no_extra_turn_capture_count": sum(
            1
            for item in features
            if item["produces_capture"] and not item["gives_extra_turn"]
        ),
        "no_extra_turn_noncapture_seed5_count": sum(
            1
            for item in features
            if not item["gives_extra_turn"]
            and not item["produces_capture"]
            and int(item["seed_count"]) == 5
        ),
    }


def _state_condition_two_captures_noncapture5(
    *, legal_moves: list[int], move_feature_annotations: MoveFeatureAnnotations
) -> bool:
    counts = _context_counts(move_feature_annotations)
    return (
        counts["seed4_extra_turn_count"] >= 1
        and counts["no_extra_turn_capture_count"] >= 2
        and counts["no_extra_turn_noncapture_seed5_count"] >= 1
        and len(legal_moves) >= 4
    )


def _temperature_transform(
    *,
    transform_name: str,
    legal_moves: list[int],
    original_prior: np.ndarray,
    move_feature_annotations: MoveFeatureAnnotations,
    temperature: float,
) -> tuple[np.ndarray, dict]:
    adjusted = np.zeros_like(original_prior, dtype=np.float32)
    if not legal_moves:
        return adjusted, _telemetry(
            transform_name=transform_name,
            legal_moves=legal_moves,
            before=original_prior,
            after=adjusted,
            move_feature_annotations=move_feature_annotations,
        )
    legal_prior = np.asarray(original_prior[legal_moves], dtype=np.float32)
    adjusted[legal_moves] = np.power(legal_prior, 1.0 / float(temperature))
    transformed = normalize_legal_prior(adjusted, legal_moves)
    return transformed, _telemetry(
        transform_name=transform_name,
        legal_moves=legal_moves,
        before=original_prior,
        after=transformed,
        move_feature_annotations=move_feature_annotations,
    )


def _conditional_seed4_extra_turn_damp(
    *,
    transform_name: str,
    legal_moves: list[int],
    original_prior: np.ndarray,
    move_feature_annotations: MoveFeatureAnnotations,
    scale: float,
    require_exactly_four_legal: bool,
) -> tuple[np.ndarray, dict]:
    state_condition = _state_condition_two_captures_noncapture5(
        legal_moves=legal_moves,
        move_feature_annotations=move_feature_annotations,
    ) and (not require_exactly_four_legal or len(legal_moves) == 4)
    return _scaled_transform(
        transform_name=transform_name,
        legal_moves=legal_moves,
        original_prior=original_prior,
        move_feature_annotations=move_feature_annotations,
        scale_for_move=lambda move: (
            scale
            if state_condition
            and move_feature_annotations[int(move)]["gives_extra_turn"]
            and int(move_feature_annotations[int(move)]["seed_count"]) == 4
            else 1.0
        ),
    )


def apply_root_prior_transform(
    *,
    state: dict,
    legal_moves: list[int],
    original_root_prior: list[float] | np.ndarray,
    move_feature_annotations: MoveFeatureAnnotations,
    transform_name: str,
) -> tuple[np.ndarray, dict]:
    del state
    if transform_name not in SUPPORTED_TRANSFORM_NAMES:
        raise ValueError(f"unsupported root prior transform: {transform_name}")

    original = normalize_legal_prior(original_root_prior, legal_moves)
    if transform_name == "original_prior":
        return original, _telemetry(
            transform_name=transform_name,
            legal_moves=legal_moves,
            before=original,
            after=original,
            move_feature_annotations=move_feature_annotations,
            scale_factors={int(move): 1.0 for move in legal_moves},
        )

    if transform_name == "uniform_legal_prior":
        transformed = np.zeros_like(original, dtype=np.float32)
        if legal_moves:
            transformed[legal_moves] = 1.0 / len(legal_moves)
        return transformed, _telemetry(
            transform_name=transform_name,
            legal_moves=legal_moves,
            before=original,
            after=transformed,
            move_feature_annotations=move_feature_annotations,
        )

    if transform_name == "extra_turn_damp_050":
        return _scaled_transform(
            transform_name=transform_name,
            legal_moves=legal_moves,
            original_prior=original,
            move_feature_annotations=move_feature_annotations,
            scale_for_move=lambda move: (
                0.50 if move_feature_annotations[int(move)]["gives_extra_turn"] else 1.0
            ),
        )

    if transform_name == "extra_turn_damp_025":
        return _scaled_transform(
            transform_name=transform_name,
            legal_moves=legal_moves,
            original_prior=original,
            move_feature_annotations=move_feature_annotations,
            scale_for_move=lambda move: (
                0.25 if move_feature_annotations[int(move)]["gives_extra_turn"] else 1.0
            ),
        )

    if transform_name == "no_extra_turn_capture_boost_2x":
        return _scaled_transform(
            transform_name=transform_name,
            legal_moves=legal_moves,
            original_prior=original,
            move_feature_annotations=move_feature_annotations,
            scale_for_move=lambda move: (
                2.0
                if move_feature_annotations[int(move)]["produces_capture"]
                and not move_feature_annotations[int(move)]["gives_extra_turn"]
                else 1.0
            ),
        )

    if transform_name == "hybrid_damp050_captureboost2x":
        return _scaled_transform(
            transform_name=transform_name,
            legal_moves=legal_moves,
            original_prior=original,
            move_feature_annotations=move_feature_annotations,
            scale_for_move=lambda move: (
                (
                    0.50
                    if move_feature_annotations[int(move)]["gives_extra_turn"]
                    else 1.0
                )
                * (
                    2.0
                    if move_feature_annotations[int(move)]["produces_capture"]
                    and not move_feature_annotations[int(move)]["gives_extra_turn"]
                    else 1.0
                )
            ),
        )

    if transform_name == "prior_temperature_2x":
        return _temperature_transform(
            transform_name=transform_name,
            legal_moves=legal_moves,
            original_prior=original,
            move_feature_annotations=move_feature_annotations,
            temperature=2.0,
        )

    if transform_name == "seed4_extra_turn_damp_050_when_two_captures_noncapture5":
        return _conditional_seed4_extra_turn_damp(
            transform_name=transform_name,
            legal_moves=legal_moves,
            original_prior=original,
            move_feature_annotations=move_feature_annotations,
            scale=0.50,
            require_exactly_four_legal=False,
        )

    if transform_name == "seed4_extra_turn_damp_025_when_two_captures_noncapture5":
        return _conditional_seed4_extra_turn_damp(
            transform_name=transform_name,
            legal_moves=legal_moves,
            original_prior=original,
            move_feature_annotations=move_feature_annotations,
            scale=0.25,
            require_exactly_four_legal=False,
        )

    if transform_name == "seed4_extra_turn_damp_010_when_two_captures_noncapture5":
        return _conditional_seed4_extra_turn_damp(
            transform_name=transform_name,
            legal_moves=legal_moves,
            original_prior=original,
            move_feature_annotations=move_feature_annotations,
            scale=0.10,
            require_exactly_four_legal=False,
        )

    if (
        transform_name
        == "seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5"
    ):
        return _conditional_seed4_extra_turn_damp(
            transform_name=transform_name,
            legal_moves=legal_moves,
            original_prior=original,
            move_feature_annotations=move_feature_annotations,
            scale=0.10,
            require_exactly_four_legal=True,
        )

    raise AssertionError(f"unhandled root prior transform: {transform_name}")


def build_root_prior_override(
    transform_name: str,
) -> Callable[..., np.ndarray] | None:
    if transform_name == "original_prior":
        return None
    if transform_name not in SUPPORTED_TRANSFORM_NAMES:
        raise ValueError(f"unsupported root prior transform: {transform_name}")

    def override(*, game, legal_moves: list[int], priors: np.ndarray) -> np.ndarray:
        state = game.to_state()
        annotations = move_feature_annotations_for(state=state, legal_moves=legal_moves)
        transformed, telemetry_payload = apply_root_prior_transform(
            state=state,
            legal_moves=legal_moves,
            original_root_prior=np.asarray(priors, dtype=np.float32),
            move_feature_annotations=annotations,
            transform_name=transform_name,
        )
        override.last_telemetry = telemetry_payload
        return transformed

    override.last_telemetry = None
    return override
