from __future__ import annotations

import json
import math
from typing import Any


def budget_pair_label(challenger_simulations: int, current_simulations: int) -> str:
    return f"{int(challenger_simulations)}:{int(current_simulations)}"


def parse_budget_pair_label(label: str) -> tuple[int, int]:
    left, right = str(label).split(":", 1)
    return int(left), int(right)


def parse_cpuct_schedule_json(text: str | None) -> dict[str, float]:
    if text is None:
        return {}
    payload = str(text).strip()
    if not payload:
        return {}
    loaded = json.loads(payload)
    if not isinstance(loaded, dict):
        raise ValueError("c_puct schedule must be a JSON object")
    schedule: dict[str, float] = {}
    for key, value in loaded.items():
        challenger_simulations, current_simulations = parse_budget_pair_label(str(key))
        cpuct = float(value)
        if not math.isfinite(cpuct) or cpuct <= 0.0:
            raise ValueError(f"invalid c_puct for {key}: {value}")
        schedule[budget_pair_label(challenger_simulations, current_simulations)] = cpuct
    return schedule


def normalize_cpuct_schedule(schedule: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for label in sorted(schedule):
        challenger_simulations, current_simulations = parse_budget_pair_label(label)
        normalized[budget_pair_label(challenger_simulations, current_simulations)] = (
            float(schedule[label])
        )
    return normalized


def resolve_budget_cpuct(
    *,
    schedule: dict[str, float] | None,
    challenger_simulations: int,
    current_simulations: int,
    default_c_puct: float,
) -> float:
    if not schedule:
        return float(default_c_puct)
    return float(
        schedule.get(
            budget_pair_label(challenger_simulations, current_simulations),
            default_c_puct,
        )
    )


def schedule_definition(
    *, default_c_puct: float, schedule: dict[str, float] | None
) -> dict[str, Any]:
    return {
        "default_c_puct": float(default_c_puct),
        "overrides": normalize_cpuct_schedule(schedule or {}),
    }
