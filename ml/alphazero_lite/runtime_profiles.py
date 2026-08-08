"""Normalized, budget-aware runtime search profiles."""

from __future__ import annotations

import json
import math
from typing import Any

from ml.alphazero_lite.cpuct_schedule import (
    budget_pair_label,
    parse_budget_pair_label,
    resolve_budget_cpuct,
)
from ml.alphazero_lite.evaluation_seed_contract import stable_hash


def _normalize_overrides(
    values: dict[str, float] | None, *, field: str
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in (values or {}).items():
        try:
            challenger, current = parse_budget_pair_label(str(key))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {field} budget key: {key!r}") from exc
        if challenger <= 0 or current <= 0:
            raise ValueError(f"invalid {field} budget key: {key!r}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"invalid {field} for {key}: {value!r}")
        normalized[budget_pair_label(challenger, current)] = number
    return {key: normalized[key] for key in sorted(normalized)}


def runtime_profile_definition(
    *,
    name: str,
    default_tactical_root_bias: float,
    tactical_root_bias_overrides: dict[str, float] | None,
    default_c_puct: float,
    c_puct_overrides: dict[str, float] | None,
    search_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a canonical profile definition and content hash."""
    tactical = float(default_tactical_root_bias)
    cpuct = float(default_c_puct)
    if not math.isfinite(tactical) or not math.isfinite(cpuct) or cpuct <= 0.0:
        raise ValueError("runtime-profile defaults must be finite and c_puct positive")
    definition = {
        "name": str(name),
        "default_tactical_root_bias": tactical,
        "tactical_root_bias_overrides": _normalize_overrides(
            tactical_root_bias_overrides, field="tactical_root_bias"
        ),
        "default_c_puct": cpuct,
        "c_puct_overrides": _normalize_overrides(c_puct_overrides, field="c_puct"),
        "search_options": search_options
        or {
            "root_policy_mode": "deterministic",
            "normalize_values": False,
            "value_transform": None,
            "root_prior_transform": None,
        },
    }
    return {**definition, "hash": stable_hash(definition)}


def parse_runtime_profile_json(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("runtime profile must be a JSON object")
    required = {
        "name",
        "default_tactical_root_bias",
        "tactical_root_bias_overrides",
        "default_c_puct",
        "c_puct_overrides",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"runtime profile missing fields: {sorted(missing)}")
    return runtime_profile_definition(
        name=str(payload["name"]),
        default_tactical_root_bias=float(payload["default_tactical_root_bias"]),
        tactical_root_bias_overrides=payload["tactical_root_bias_overrides"],
        default_c_puct=float(payload["default_c_puct"]),
        c_puct_overrides=payload["c_puct_overrides"],
        search_options=payload.get("search_options"),
    )


def resolve_runtime_profile(profile: dict[str, Any], budget: str) -> dict[str, Any]:
    challenger, current = parse_budget_pair_label(budget)
    label = budget_pair_label(challenger, current)
    tactical = profile["tactical_root_bias_overrides"].get(
        label, profile["default_tactical_root_bias"]
    )
    return {
        "tactical_root_bias": float(tactical),
        "c_puct": resolve_budget_cpuct(
            schedule=profile["c_puct_overrides"],
            challenger_simulations=challenger,
            current_simulations=current,
            default_c_puct=float(profile["default_c_puct"]),
        ),
        "search_options": dict(profile["search_options"]),
    }
