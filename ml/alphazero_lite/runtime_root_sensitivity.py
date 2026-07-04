from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.cpuct_schedule import resolve_budget_cpuct, schedule_definition
from ml.alphazero_lite.run_post_schedule_value_trust_calibration import (
    budget_pair_label,
    load_suite_entries,
    parse_budget_pairs,
)
from ml.alphazero_lite.run_promoted_current_opening_puct_disagreement import (
    sha256_file,
)
from ml.alphazero_lite.self_play import build_eval_search_options


def _diagnostic_root_value(summary: dict[str, Any]) -> float:
    search_root_value = summary.get("search_root_value")
    if isinstance(search_root_value, (int, float)):
        return float(search_root_value)
    return float(summary.get("value") or 0.0)


def runtime_sensitivity_diagnostic_for_opening_suite(
    *,
    current_path: Path,
    suite_path: Path,
    lane_specs: list[dict[str, Any]],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    seed: int,
    workdir: Path,
    budget_labels: list[str],
) -> dict[str, Any]:
    out_path = workdir / "runtime_sensitivity.json"
    cache_key = {
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "lanes": {
            str(lane_spec["name"]): lane_spec.get("value_transform")
            for lane_spec in lane_specs
        },
        "default_c_puct": float(default_c_puct),
        "cpuct_schedule": schedule_definition(
            default_c_puct=float(default_c_puct), schedule=cpuct_schedule
        ),
        "seed": int(seed),
        "budget_labels": list(budget_labels),
    }
    if out_path.is_file():
        cached = json.loads(out_path.read_text(encoding="utf-8"))
        if cached.get("cache_key") == cache_key:
            return cached

    suite_entries = load_suite_entries(suite_path)
    evaluator = ArtifactEvaluator(current_path)
    budgets = parse_budget_pairs(",".join(budget_labels))
    default_results: dict[str, list[dict[str, Any]]] = {}
    identity_lane = next(
        lane_spec
        for lane_spec in lane_specs
        if str(lane_spec["name"]) == "identity_ref"
    )
    for challenger_sims, current_sims in budgets:
        budget_label = budget_pair_label(challenger_sims, current_sims)
        effective_c_puct = resolve_budget_cpuct(
            schedule=cpuct_schedule,
            challenger_simulations=challenger_sims,
            current_simulations=current_sims,
            default_c_puct=default_c_puct,
        )
        default_rows: list[dict[str, Any]] = []
        for index, entry in enumerate(suite_entries):
            summary = evaluate_artifact_position(
                evaluator=evaluator,
                state=entry["state"],
                simulations=challenger_sims,
                seed=int(seed) + index,
                c_puct=effective_c_puct,
                search_options=build_eval_search_options(
                    root_policy_mode="deterministic",
                    tactical_root_bias=0.0,
                    value_transform=identity_lane.get("value_transform"),
                ),
            )
            default_rows.append(summary)
        default_results[budget_label] = default_rows

    diagnostic = {"cache_key": cache_key, "suite": str(suite_path), "budgets": {}}
    for challenger_sims, current_sims in budgets:
        budget_label = budget_pair_label(challenger_sims, current_sims)
        effective_c_puct = resolve_budget_cpuct(
            schedule=cpuct_schedule,
            challenger_simulations=challenger_sims,
            current_simulations=current_sims,
            default_c_puct=default_c_puct,
        )
        baseline_rows = default_results[budget_label]
        budget_rows: dict[str, dict[str, float | int]] = {}
        for lane_spec in lane_specs:
            lane_name = str(lane_spec["name"])
            if lane_name == "identity_ref":
                budget_rows[lane_name] = {
                    "move_change_count": 0,
                    "move_change_rate": 0.0,
                    "mean_abs_value_delta": 0.0,
                    "max_abs_value_delta": 0.0,
                }
                continue
            move_change_count = 0
            value_deltas: list[float] = []
            for index, entry in enumerate(suite_entries):
                summary = evaluate_artifact_position(
                    evaluator=evaluator,
                    state=entry["state"],
                    simulations=challenger_sims,
                    seed=int(seed) + index,
                    c_puct=effective_c_puct,
                    search_options=build_eval_search_options(
                        root_policy_mode="deterministic",
                        tactical_root_bias=0.0,
                        value_transform=lane_spec.get("value_transform"),
                    ),
                )
                if summary.get("selected_move") != baseline_rows[index].get(
                    "selected_move"
                ):
                    move_change_count += 1
                value_deltas.append(
                    abs(
                        _diagnostic_root_value(summary)
                        - _diagnostic_root_value(baseline_rows[index])
                    )
                )
            budget_rows[lane_name] = {
                "move_change_count": move_change_count,
                "move_change_rate": float(
                    move_change_count / max(len(suite_entries), 1)
                ),
                "mean_abs_value_delta": float(statistics.fmean(value_deltas))
                if value_deltas
                else 0.0,
                "max_abs_value_delta": float(max(value_deltas))
                if value_deltas
                else 0.0,
            }
        diagnostic["budgets"][budget_label] = budget_rows
    out_path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    return diagnostic


def root_sensitivity_prefilter(
    *,
    runtime_sensitivity: dict[str, Any],
    lane_names: list[str],
    default_name: str,
    min_move_change_rate: float,
    min_mean_abs_value_delta: float,
) -> dict[str, Any]:
    budget_map = runtime_sensitivity.get("budgets", {})
    retained = [default_name]
    filtered_out: list[str] = []
    decisions: dict[str, dict[str, Any]] = {
        default_name: {
            "retain": True,
            "reason": "reference_lane",
            "max_move_change_rate": None,
            "max_mean_abs_value_delta": None,
        }
    }
    for lane_name in lane_names:
        if lane_name == default_name:
            continue
        lane_rows = [
            lanes[lane_name]
            for lanes in budget_map.values()
            if isinstance(lanes, dict) and lane_name in lanes
        ]
        max_move_change_rate = max(
            (float(row.get("move_change_rate") or 0.0) for row in lane_rows),
            default=0.0,
        )
        max_mean_abs_value_delta = max(
            (float(row.get("mean_abs_value_delta") or 0.0) for row in lane_rows),
            default=0.0,
        )
        retain = max_move_change_rate >= float(
            min_move_change_rate
        ) or max_mean_abs_value_delta >= float(min_mean_abs_value_delta)
        decisions[lane_name] = {
            "retain": bool(retain),
            "reason": (
                "root_sensitivity_threshold_met"
                if retain
                else "below_root_sensitivity_threshold"
            ),
            "max_move_change_rate": float(max_move_change_rate),
            "max_mean_abs_value_delta": float(max_mean_abs_value_delta),
        }
        if retain:
            retained.append(lane_name)
        else:
            filtered_out.append(lane_name)
    return {
        "retain": retained,
        "filtered_out": filtered_out,
        "decisions": decisions,
        "thresholds": {
            "min_move_change_rate": float(min_move_change_rate),
            "min_mean_abs_value_delta": float(min_mean_abs_value_delta),
        },
    }
