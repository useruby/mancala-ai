from __future__ import annotations

import json
import math
from hashlib import sha256
from typing import Any

PHASE_KEYS = ("opening", "midgame", "late")
VALUE_TRANSFORM_SCHEMA_VERSION = "v1"
DIAGNOSTIC_PHASE_MODES = frozenset(
    {"identity", "zero", "negate", "sign_only", "scale_clamp"}
)


def identity_transform_config(*, name: str = "identity_ref") -> dict[str, Any]:
    return {
        "version": VALUE_TRANSFORM_SCHEMA_VERSION,
        "name": str(name),
        "kind": "identity",
        "phase_params": {},
    }


def clamp_unit_value(value: float) -> float:
    return max(-0.999999, min(0.999999, float(value)))


def affine_tanh_transform(value: float, *, a: float, b: float) -> float:
    clamped = clamp_unit_value(value)
    return float(math.tanh((float(a) * math.atanh(clamped)) + float(b)))


def _normalize_phase_name(phase_name: str) -> str:
    normalized = str(phase_name)
    if normalized not in PHASE_KEYS:
        raise ValueError(f"unsupported value-transform phase: {phase_name}")
    return normalized


def _normalize_affine_phase_params(params: dict[str, Any]) -> dict[str, float]:
    a = float(params.get("a", 1.0))
    b = float(params.get("b", 0.0))
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("affine_tanh value-transform parameters must be finite")
    return {"a": a, "b": b}


def _normalize_isotonic_phase_params(params: dict[str, Any]) -> dict[str, list[float]]:
    raw_x = params.get("x")
    raw_y = params.get("y")
    if not isinstance(raw_x, list) or not isinstance(raw_y, list):
        raise ValueError("phase_isotonic parameters must include x/y lists")
    if len(raw_x) != len(raw_y) or not raw_x:
        raise ValueError("phase_isotonic x/y lists must be non-empty and aligned")
    x_values = [float(value) for value in raw_x]
    y_values = [float(value) for value in raw_y]
    previous_x = -float("inf")
    previous_y = -float("inf")
    for x_value, y_value in zip(x_values, y_values, strict=True):
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            raise ValueError("phase_isotonic values must be finite")
        if x_value < previous_x:
            raise ValueError("phase_isotonic x values must be sorted")
        if y_value < previous_y:
            raise ValueError("phase_isotonic y values must be monotonic")
        previous_x = x_value
        previous_y = y_value
    return {"x": x_values, "y": y_values}


def _normalize_diagnostic_phase_params(params: dict[str, Any]) -> dict[str, Any]:
    mode = str(params.get("mode") or "identity").strip()
    if mode not in DIAGNOSTIC_PHASE_MODES:
        raise ValueError(f"unsupported diagnostic value-transform mode: {mode}")
    normalized: dict[str, Any] = {"mode": mode}
    if mode == "scale_clamp":
        scale = float(params.get("scale", 1.0))
        if not math.isfinite(scale):
            raise ValueError("diagnostic scale_clamp scale must be finite")
        normalized["scale"] = scale
    return normalized


def normalize_value_transform_config(
    value_transform: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if value_transform is None:
        return None
    if not isinstance(value_transform, dict):
        raise ValueError("value_transform must be a JSON object")
    kind = str(value_transform.get("kind", "")).strip()
    if kind == "identity":
        return identity_transform_config(
            name=str(value_transform.get("name") or "identity_ref")
        )
    if kind not in {"affine_tanh", "phase_isotonic", "diagnostic_phase_transform"}:
        raise ValueError(f"unsupported value_transform kind: {kind}")
    raw_phase_params = value_transform.get("phase_params")
    if not isinstance(raw_phase_params, dict):
        raise ValueError("value_transform.phase_params must be an object")
    phase_params: dict[str, Any] = {}
    for phase_name in PHASE_KEYS:
        raw_params = raw_phase_params.get(phase_name)
        if raw_params is None:
            continue
        if not isinstance(raw_params, dict):
            raise ValueError(
                f"value_transform.phase_params.{phase_name} must be an object"
            )
        phase_params[_normalize_phase_name(phase_name)] = (
            _normalize_affine_phase_params(raw_params)
            if kind == "affine_tanh"
            else (
                _normalize_isotonic_phase_params(raw_params)
                if kind == "phase_isotonic"
                else _normalize_diagnostic_phase_params(raw_params)
            )
        )
    return {
        "version": VALUE_TRANSFORM_SCHEMA_VERSION,
        "name": str(value_transform.get("name") or kind),
        "kind": kind,
        "phase_params": phase_params,
    }


def parse_value_transform_json(text: str | None) -> dict[str, Any] | None:
    if text is None:
        return None
    payload = str(text).strip()
    if not payload:
        return None
    return normalize_value_transform_config(json.loads(payload))


def effective_value_transform_config(
    value_transform: dict[str, Any] | None, *, identity_name: str = "identity_ref"
) -> dict[str, Any]:
    normalized = normalize_value_transform_config(value_transform)
    if normalized is None:
        return identity_transform_config(name=identity_name)
    return normalized


def value_transform_hash(
    value_transform: dict[str, Any] | None, *, identity_name: str = "identity_ref"
) -> str:
    normalized = effective_value_transform_config(
        value_transform, identity_name=identity_name
    )
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def phase_bucket_from_seed_count(seeds_remaining: int) -> str:
    if int(seeds_remaining) <= 12:
        return "late"
    if int(seeds_remaining) <= 24:
        return "midgame"
    return "opening"


def phase_bucket_for_game(game) -> str:
    return phase_bucket_from_seed_count(sum(int(seed) for seed in game.pits))


def _interpolate_isotonic(
    value: float, *, x_values: list[float], y_values: list[float]
) -> float:
    if len(x_values) == 1:
        return float(y_values[0])
    clamped = float(value)
    if clamped <= x_values[0]:
        return float(y_values[0])
    if clamped >= x_values[-1]:
        return float(y_values[-1])
    for index in range(1, len(x_values)):
        right_x = x_values[index]
        if clamped > right_x:
            continue
        left_x = x_values[index - 1]
        left_y = y_values[index - 1]
        right_y = y_values[index]
        if right_x <= left_x:
            return float(right_y)
        ratio = (clamped - left_x) / (right_x - left_x)
        return float(left_y + (ratio * (right_y - left_y)))
    return float(y_values[-1])


def _apply_diagnostic_phase_transform(value: float, *, params: dict[str, Any]) -> float:
    mode = str(params["mode"])
    if mode == "identity":
        return float(value)
    if mode == "zero":
        return 0.0
    if mode == "negate":
        return float(-value)
    if mode == "sign_only":
        if value > 0:
            return 1.0
        if value < 0:
            return -1.0
        return 0.0
    scale = float(params.get("scale", 1.0))
    return clamp_unit_value(float(value) * scale)


def apply_value_transform(
    value: float,
    *,
    phase_bucket: str,
    value_transform: dict[str, Any] | None,
) -> float:
    normalized = normalize_value_transform_config(value_transform)
    if normalized is None or normalized["kind"] == "identity":
        return float(value)
    phase_name = _normalize_phase_name(phase_bucket)
    phase_params = normalized["phase_params"].get(phase_name)
    if phase_params is None:
        return float(value)
    if normalized["kind"] == "affine_tanh":
        return affine_tanh_transform(
            value,
            a=float(phase_params["a"]),
            b=float(phase_params["b"]),
        )
    if normalized["kind"] == "diagnostic_phase_transform":
        return _apply_diagnostic_phase_transform(float(value), params=phase_params)
    return _interpolate_isotonic(
        clamp_unit_value(value),
        x_values=[float(item) for item in phase_params["x"]],
        y_values=[float(item) for item in phase_params["y"]],
    )


def value_transform_summary_for_phase(
    value_transform: dict[str, Any] | None,
    *,
    phase_bucket: str | None,
    identity_name: str = "identity_ref",
) -> dict[str, Any]:
    normalized = effective_value_transform_config(
        value_transform, identity_name=identity_name
    )
    phase_params = None
    if phase_bucket in PHASE_KEYS:
        phase_params = normalized["phase_params"].get(str(phase_bucket))
    return {
        "name": normalized["name"],
        "kind": normalized["kind"],
        "hash": value_transform_hash(normalized, identity_name=identity_name),
        "phase_bucket": phase_bucket,
        "parameters": phase_params,
        "config": normalized,
    }
