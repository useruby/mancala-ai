"""Exact-root training-target construction and frozen-replay relabeling."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

POLICY_SIZE = 6
SELECTED_ONE_HOT = "selected_one_hot"
OPTIMAL_SET_UNIFORM = "optimal_set_uniform"
SUPPORTED_EXACT_ROOT_POLICY_TARGET_MODES = frozenset(
    {SELECTED_ONE_HOT, OPTIMAL_SET_UNIFORM}
)
EXACT_ROOT_ONE_HOT_POLICY_TARGET_MODE = "exact_root_one_hot"
EXACT_ROOT_OPTIMAL_SET_UNIFORM_POLICY_TARGET_MODE = "exact_root_optimal_set_uniform"


def normalize_exact_root_policy_target_mode(mode: str) -> str:
    normalized = str(mode)
    if normalized not in SUPPORTED_EXACT_ROOT_POLICY_TARGET_MODES:
        raise ValueError(f"unsupported exact_root_policy_target_mode: {mode}")
    return normalized


def exact_root_policy_target(
    *, selected_action: int, optimal_actions: list[int], mode: str
) -> list[float]:
    mode = normalize_exact_root_policy_target_mode(mode)
    optimal = sorted({int(action) for action in optimal_actions})
    if not optimal or any(action < 0 or action >= POLICY_SIZE for action in optimal):
        raise ValueError("exact optimal actions must be non-empty policy moves")
    if mode == SELECTED_ONE_HOT:
        if selected_action not in optimal:
            raise ValueError("exact selected action must be optimal")
        target = [0.0] * POLICY_SIZE
        target[int(selected_action)] = 1.0
        return target
    mass = 1.0 / len(optimal)
    return [mass if action in optimal else 0.0 for action in range(POLICY_SIZE)]


def exact_root_actual_policy_target_mode(mode: str) -> str:
    mode = normalize_exact_root_policy_target_mode(mode)
    if mode == SELECTED_ONE_HOT:
        return EXACT_ROOT_ONE_HOT_POLICY_TARGET_MODE
    return EXACT_ROOT_OPTIMAL_SET_UNIFORM_POLICY_TARGET_MODE


def validate_exact_root_metadata(row: dict[str, Any]) -> tuple[int, list[int]]:
    selected = row.get("exact_selected_action")
    optimal = row.get("exact_optimal_actions")
    margins = row.get("exact_action_margins")
    if (
        not isinstance(selected, int)
        or not isinstance(optimal, list)
        or not isinstance(margins, dict)
    ):
        raise ValueError("exact-root row is missing exact target metadata")
    optimal = sorted({int(action) for action in optimal})
    if selected not in optimal:
        raise ValueError("exact selected action is not optimal")
    normalized_margins = {
        int(action): float(value) for action, value in margins.items()
    }
    if not set(optimal).issubset(normalized_margins):
        raise ValueError("exact action margins must cover exact-optimal moves")
    try:
        optimal_margins = [normalized_margins[action] for action in optimal]
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("exact action margins are invalid") from error
    if not all(math.isfinite(value) for value in optimal_margins):
        raise ValueError("exact optimal action margins must be finite")
    if any(value != max(optimal_margins) for value in optimal_margins):
        raise ValueError("exact optimal actions do not have equal margins")
    if any(value > optimal_margins[0] for value in normalized_margins.values()):
        raise ValueError("exact optimal actions are not maximum-margin actions")
    return selected, optimal


def relabel_exact_root_policy_row(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return a copy with only tied exact-root supervision made uniform."""
    if row.get("teacher_source") != "exact_root_tablebase":
        return copy.deepcopy(row), False
    selected, optimal = validate_exact_root_metadata(row)
    if len(optimal) == 1:
        return copy.deepcopy(row), False
    transformed = copy.deepcopy(row)
    policy = exact_root_policy_target(
        selected_action=selected, optimal_actions=optimal, mode=OPTIMAL_SET_UNIFORM
    )
    transformed["policy"] = policy
    if "stored_policy_target" in transformed:
        transformed["stored_policy_target"] = policy
    transformed["policy_target_actual_mode"] = (
        EXACT_ROOT_OPTIMAL_SET_UNIFORM_POLICY_TARGET_MODE
    )
    return transformed, True


def relabel_exact_root_policy_jsonl(source: Path, output: Path) -> dict[str, Any]:
    """Write a deterministic immutable uniform-optimal-set replay artifact."""
    counters: Counter[str] = Counter()
    with (
        source.open("r", encoding="utf-8") as source_handle,
        output.open("w", encoding="utf-8") as output_handle,
    ):
        for line_number, line in enumerate(source_handle, start=1):
            row = json.loads(line)
            transformed, changed = relabel_exact_root_policy_row(row)
            counters["source_rows"] += 1
            is_exact = row.get("teacher_source") == "exact_root_tablebase"
            if is_exact:
                _selected, optimal = validate_exact_root_metadata(row)
                counters["exact_root_rows"] += 1
                if len(optimal) == 1:
                    counters["unique_optimum_rows"] += 1
                else:
                    counters["tied_optimum_rows"] += 1
            if changed:
                counters["policy_rows_changed"] += 1
            if not is_exact and transformed != row:
                raise ValueError(f"non-exact row changed at line {line_number}")
            output_handle.write(json.dumps(transformed, separators=(",", ":")) + "\n")
            counters["output_rows"] += 1
    if counters["policy_rows_changed"] != counters["tied_optimum_rows"]:
        raise ValueError("exact_optimal_set_relabel_scope_violation")
    return {
        **dict(counters),
        "source_sha256": sha256_file(source),
        "transformed_sha256": sha256_file(output),
        "state_mismatches": 0,
        "value_mismatches": 0,
        "winner_mismatches": 0,
        "ordering_mismatches": 0,
        "non_exact_policy_mismatches": 0,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
