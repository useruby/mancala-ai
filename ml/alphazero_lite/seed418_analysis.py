"""Pure metric and exact-action analysis for seed418 evidence."""

from __future__ import annotations

import math
import statistics
from typing import Any


def choose_exact_action(
    margins: dict[int, int], priors: list[float], legal: list[int]
) -> int:
    if set(margins) != set(legal):
        raise ValueError("exact_action_coverage_mismatch")
    if not legal:
        raise ValueError("terminal_state_has_no_legal_actions")
    best = max(margins.values())
    return max(
        (move for move in legal if margins[move] == best),
        key=lambda move: (priors[move], -move),
    )


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 128 or any(
        row.get("coverage_complete") is not True for row in rows
    ):
        raise ValueError("incomplete_exact_coverage")
    regrets: list[int] = []
    optimal: list[bool] = []
    inferior: list[bool] = []
    by_stones: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        margins = {int(key): int(value) for key, value in row["action_margins"].items()}
        legal = [int(move) for move in row["legal_moves"]]
        if set(margins) != set(legal):
            raise ValueError("exact_action_coverage_mismatch")
        chosen = int(row["search_selected_move"])
        best = max(margins.values())
        regret = best - margins[chosen]
        best_wdl = 1 if best > 0 else 0 if best == 0 else -1
        chosen_wdl = 1 if margins[chosen] > 0 else 0 if margins[chosen] == 0 else -1
        is_inferior = chosen_wdl < best_wdl
        regrets.append(regret)
        optimal.append(regret == 0)
        inferior.append(is_inferior)
        by_stones.setdefault(str(row["active_pit_stones"]), []).append(
            {"regret": regret, "optimal": regret == 0, "wdl_inferior": is_inferior}
        )
    slices = {
        stone: {
            "n": len(values),
            "mean_final_margin_regret": statistics.fmean(
                item["regret"] for item in values
            ),
            "exact_optimal_action_rate": statistics.fmean(
                item["optimal"] for item in values
            ),
            "wdl_inferior_choice_rate": statistics.fmean(
                item["wdl_inferior"] for item in values
            ),
        }
        for stone, values in sorted(by_stones.items(), key=lambda pair: int(pair[0]))
    }
    latencies = [float(row["native_decision_latency_ms"]) for row in rows]
    mean_regret = statistics.fmean(regrets)
    inferior_rate = statistics.fmean(inferior)
    mean_latency = statistics.fmean(latencies)
    p95 = sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1]
    advance = (
        inferior_rate >= 0.05
        and mean_regret >= 1
        and mean_latency <= 200
        and p95 <= 350
    )
    return {
        "schema": "seed418-analysis-v1",
        "n": len(rows),
        "complete_exact_coverage": True,
        "mean_final_margin_regret": mean_regret,
        "exact_optimal_action_rate": statistics.fmean(optimal),
        "wdl_inferior_choice_rate": inferior_rate,
        "native_decision_latency_ms": {"mean": mean_latency, "p95_nearest_rank": p95},
        "by_active_pit_stones": slices,
        "decision": "advance_to_separately_registered_runtime_experiment"
        if advance
        else "stop_seed418_branch",
        "interpretation": "retrospective feasibility diagnostic; not strength, promotion, or runtime-change evidence",
    }
