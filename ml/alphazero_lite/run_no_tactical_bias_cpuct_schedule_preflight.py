#!/usr/bin/env python3
"""Budget-conditioned no-tactical-bias c_puct schedule preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    normalize_cpuct_schedule,
    resolve_budget_cpuct,
    schedule_definition,
)
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    bootstrap_ci,
)
from ml.alphazero_lite.run_opening_suite_seat_benchmark import (  # noqa: E402
    compute_per_opening_metrics,
    compute_seat_metrics,
    load_suite,
    parse_game_jsonl,
)
from ml.alphazero_lite.run_pr123_weighted_candidate_preflight import (  # noqa: E402
    fmt,
    markdown_table,
)
from ml.alphazero_lite.run_promoted_current_opening_puct_disagreement import (  # noqa: E402
    build_input_summary,
    require_existing_file,
    sha256_file,
    verify_expected_hash,
    write_json,
)


SUMMARY_SCHEMA = "azlite_no_tactical_bias_cpuct_schedule_preflight_v1"
SUMMARY_FILENAME = "summary_metrics.json"
REPORT_PATH = (
    REPO_ROOT
    / "docs/alphazero-lite-no-tactical-bias-cpuct-schedule-preflight-results.md"
)
DEFAULT_WORKDIR = "/tmp/azlite_no_tactical_bias_cpuct_schedule_preflight"
DEFAULT_SCHEDULES = "default_1_25,global_1_00,768eq_1_00,equal_budget_1_00,low_budget_1_00,768eq_0_90,768eq_1_10"
DEFAULT_BUDGET_AXES = (
    "384:256",
    "768:256",
    "768:768",
    "1200:1200",
    "1200:256",
    "256:768",
)
ROBUSTNESS_AXES = ("384:256", "768:768", "1200:1200", "1200:256")
DEFAULT_GATE_BUDGET_PAIRS = "384:256,1200:1200,1200:256,256:768"
DEFAULT_TACTICAL_ROOT_BIAS = 0.0
DEFAULT_GAMES_PER_OPENING = 2
DEFAULT_TIMEOUT = 14400
DEFAULT_GATE_GAMES = 120


def _python() -> str:
    candidate = REPO_ROOT / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


@dataclass(frozen=True)
class LaneConfig:
    name: str
    description: str
    default_c_puct: float
    schedule_overrides: dict[str, float]
    tactical_root_bias: float = DEFAULT_TACTICAL_ROOT_BIAS
    is_schedule_lane: bool = False

    def schedule_definition(self) -> dict[str, Any]:
        return schedule_definition(
            default_c_puct=self.default_c_puct,
            schedule=self.schedule_overrides,
        )

    def schedule_json(self) -> str:
        return json.dumps(self.schedule_definition()["overrides"], sort_keys=True)

    def effective_cpuct_for_axis(self, axis: str) -> float:
        challenger_simulations, current_simulations = parse_budget_axis(axis)
        return resolve_budget_cpuct(
            schedule=self.schedule_overrides,
            challenger_simulations=challenger_simulations,
            current_simulations=current_simulations,
            default_c_puct=self.default_c_puct,
        )


def parse_budget_axis(axis: str) -> tuple[int, int]:
    left, right = axis.split(":", 1)
    return int(left), int(right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--tactical-root-bias", type=float, default=0.0)
    parser.add_argument("--schedules", default=DEFAULT_SCHEDULES)
    parser.add_argument(
        "--games-per-opening", type=int, default=DEFAULT_GAMES_PER_OPENING
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--gate-games", type=int, default=DEFAULT_GATE_GAMES)
    return parser.parse_args()


def parse_csv_paths(text: str) -> list[Path]:
    values = [Path(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one path")
    return values


def parse_csv_strings(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one value")
    return values


def build_schedule_lanes(
    *, names: list[str], tactical_root_bias: float
) -> list[LaneConfig]:
    lane_specs: dict[str, tuple[str, float, dict[str, float], bool]] = {
        "default_1_25": (
            "default_cpuct_1_25_ref",
            1.25,
            {},
            False,
        ),
        "global_1_00": (
            "global_cpuct_1_00_ref",
            1.00,
            {},
            False,
        ),
        "768eq_1_00": (
            "schedule_768eq_cpuct_1_00",
            1.25,
            {"768:768": 1.00},
            True,
        ),
        "equal_budget_1_00": (
            "schedule_equal_budget_cpuct_1_00",
            1.25,
            {"768:768": 1.00, "1200:1200": 1.00},
            True,
        ),
        "low_budget_1_00": (
            "schedule_low_budget_cpuct_1_00",
            1.25,
            {"384:256": 1.00, "768:768": 1.00},
            True,
        ),
        "768eq_0_90": (
            "schedule_768eq_cpuct_0_90",
            1.25,
            {"768:768": 0.90},
            True,
        ),
        "768eq_1_10": (
            "schedule_768eq_cpuct_1_10",
            1.25,
            {"768:768": 1.10},
            True,
        ),
    }
    lanes: list[LaneConfig] = []
    for name in names:
        if name not in lane_specs:
            raise ValueError(f"unknown schedule lane: {name}")
        lane_name, default_c_puct, overrides, is_schedule_lane = lane_specs[name]
        normalized_overrides = normalize_cpuct_schedule(overrides)
        description = json.dumps(
            schedule_definition(
                default_c_puct=default_c_puct,
                schedule=normalized_overrides,
            ),
            sort_keys=True,
        )
        lanes.append(
            LaneConfig(
                name=lane_name,
                description=description,
                default_c_puct=default_c_puct,
                schedule_overrides=normalized_overrides,
                tactical_root_bias=float(tactical_root_bias),
                is_schedule_lane=is_schedule_lane,
            )
        )
    return lanes


def write_suite_prefixes(path: Path, suite_entries: list[dict[str, Any]]) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in suite_entries:
            handle.write(json.dumps({"prefix_moves": entry["prefix_moves"]}) + "\n")


def run_arena_for_budget(
    *,
    lane: LaneConfig,
    suite_name: str,
    suite_path: Path,
    suite_entries: list[dict[str, Any]],
    current_path: Path,
    axis: str,
    workdir: Path,
    games_per_opening: int,
    workers: int,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    challenger_simulations, current_simulations = parse_budget_axis(axis)
    effective_c_puct = lane.effective_cpuct_for_axis(axis)
    lane_dir = workdir / suite_name / lane.name / axis.replace(":", "_")
    lane_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = lane_dir / "metrics.json"
    arena_json = lane_dir / "arena.json"
    games_jsonl = lane_dir / "games.jsonl"
    suite_jsonl = lane_dir / "opening_suite.jsonl"
    write_suite_prefixes(suite_jsonl, suite_entries)
    total_games = len(suite_entries) * max(1, int(games_per_opening))
    cache_context = {
        "lane": lane.name,
        "suite_path": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "suite_size": len(suite_entries),
        "current_path": str(current_path),
        "current_sha256": sha256_file(current_path / "weights.json"),
        "challenger_simulations": challenger_simulations,
        "current_simulations": current_simulations,
        "effective_c_puct": effective_c_puct,
        "games_per_opening": games_per_opening,
        "seed": seed,
        "workers": workers,
        "schedule": lane.schedule_definition(),
        "tactical_root_bias": lane.tactical_root_bias,
    }
    shared_cache_context = {
        "suite_path": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "suite_size": len(suite_entries),
        "current_path": str(current_path),
        "current_sha256": sha256_file(current_path / "weights.json"),
        "challenger_simulations": challenger_simulations,
        "current_simulations": current_simulations,
        "effective_c_puct": effective_c_puct,
        "games_per_opening": games_per_opening,
        "seed": seed,
        "workers": workers,
        "tactical_root_bias": lane.tactical_root_bias,
    }
    shared_cache_key = hashlib.sha256(
        json.dumps(shared_cache_context, sort_keys=True).encode("utf-8")
    ).hexdigest()
    shared_metrics_path = (
        workdir
        / "_shared"
        / suite_name
        / axis.replace(":", "_")
        / shared_cache_key
        / "metrics.json"
    )
    if metrics_path.is_file() and arena_json.is_file() and games_jsonl.is_file():
        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        if cached.get("cache_context") == cache_context:
            return cached
    if shared_metrics_path.is_file():
        cached = json.loads(shared_metrics_path.read_text(encoding="utf-8"))
        if cached.get("cache_context") == shared_cache_context:
            lane_cached = dict(cached)
            lane_cached["cache_context"] = cache_context
            write_json(metrics_path, lane_cached)
            return lane_cached

    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/arena.py"),
        "--challenger",
        str(current_path),
        "--current",
        str(current_path),
        "--challenger-simulations",
        str(challenger_simulations),
        "--current-simulations",
        str(current_simulations),
        "--games",
        str(total_games),
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--min-score",
        "0.0",
        "--out",
        str(arena_json),
        "--game-jsonl",
        str(games_jsonl),
        "--games-per-opening",
        str(games_per_opening),
        "--opening-prefixes-jsonl",
        str(suite_jsonl),
        "--root-policy-mode",
        "deterministic",
        "--c-puct",
        str(effective_c_puct),
        "--tactical-root-bias",
        str(lane.tactical_root_bias),
    ]
    print(
        f"[arena] {suite_name} {lane.name} {axis} c_puct={effective_c_puct:.2f}",
        flush=True,
    )
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"arena failed for {suite_name} {lane.name} {axis}: {result.stderr[-2000:]}"
        )
    arena_report = json.loads(arena_json.read_text(encoding="utf-8"))
    entries = parse_game_jsonl(str(games_jsonl))
    seat_metrics = compute_seat_metrics(entries)
    per_opening_metrics = compute_per_opening_metrics(entries)
    notes = (
        arena_report.get("notes", {})
        if isinstance(arena_report.get("notes"), dict)
        else {}
    )
    budget_summary = (
        arena_report.get("budget_summary", {})
        if isinstance(arena_report.get("budget_summary"), dict)
        else {}
    )
    metrics = {
        "cache_context": shared_cache_context,
        "suite_name": suite_name,
        "suite_path": str(suite_path),
        "lane": lane.name,
        "budget_pair": axis,
        "challenger_simulations": challenger_simulations,
        "current_simulations": current_simulations,
        "effective_c_puct": effective_c_puct,
        "tactical_root_bias": lane.tactical_root_bias,
        "games": total_games,
        "ds": seat_metrics["ds"],
        "p0_score": seat_metrics["p0_score"],
        "p1_score": seat_metrics["p1_score"],
        "duplicate_trajectory_count": seat_metrics["duplicate_trajectory_count"],
        "duplicate_trajectory_rate": seat_metrics["duplicate_trajectory_rate"],
        "per_opening_metrics": per_opening_metrics,
        "search_profile": notes.get("search_profile"),
        "search_profile_hash": notes.get("search_profile_hash"),
        "move_time_mean_ms": notes.get("move_time_mean_ms"),
        "move_time_p95_ms": notes.get("move_time_p95_ms"),
        "mean_final_simulations": budget_summary.get("mean_final_simulations"),
        "budget_summary": budget_summary,
    }
    shared_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(shared_metrics_path, metrics)
    metrics["cache_context"] = cache_context
    write_json(metrics_path, metrics)
    return metrics


def evaluate_suite(
    *,
    suite_name: str,
    suite_path: Path,
    current_path: Path,
    lanes: list[LaneConfig],
    workdir: Path,
    games_per_opening: int,
    workers: int,
    seed: int,
    timeout: int,
) -> dict[str, dict[str, Any]]:
    suite_entries = load_suite(str(suite_path))
    results: dict[str, dict[str, Any]] = {}
    for lane in lanes:
        lane_results: dict[str, Any] = {}
        for axis in DEFAULT_BUDGET_AXES:
            lane_results[axis] = run_arena_for_budget(
                lane=lane,
                suite_name=suite_name,
                suite_path=suite_path,
                suite_entries=suite_entries,
                current_path=current_path,
                axis=axis,
                workdir=workdir,
                games_per_opening=games_per_opening,
                workers=workers,
                seed=seed,
                timeout=timeout,
            )
        results[lane.name] = lane_results
    return results


def delta_vs_default(
    suite_results: dict[str, dict[str, Any]], lane: LaneConfig, default_lane: LaneConfig
) -> dict[str, float]:
    lane_results = suite_results[lane.name]
    default_results = suite_results[default_lane.name]
    return {
        axis: float(lane_results[axis]["ds"]) - float(default_results[axis]["ds"])
        for axis in DEFAULT_BUDGET_AXES
    }


def aggregate_stage_summary(
    *,
    suite_results: dict[str, dict[str, Any]],
    lanes: list[LaneConfig],
    default_lane: LaneConfig,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for lane in lanes:
        row: dict[str, Any] = {
            "schedule": lane.schedule_definition(),
            "description": lane.description,
        }
        deltas = delta_vs_default(suite_results, lane, default_lane)
        for axis in DEFAULT_BUDGET_AXES:
            axis_metrics = suite_results[lane.name][axis]
            row[axis] = {
                "ds": float(axis_metrics["ds"]),
                "delta_vs_default": float(deltas[axis]),
                "p0_score": float(axis_metrics["p0_score"]),
                "p1_score": float(axis_metrics["p1_score"]),
                "duplicate_trajectory_count": int(
                    axis_metrics["duplicate_trajectory_count"]
                ),
                "duplicate_trajectory_rate": float(
                    axis_metrics["duplicate_trajectory_rate"]
                ),
                "effective_c_puct": float(axis_metrics["effective_c_puct"]),
                "tactical_root_bias": float(axis_metrics["tactical_root_bias"]),
                "search_profile": axis_metrics.get("search_profile"),
                "search_profile_hash": axis_metrics.get("search_profile_hash"),
                "challenger_simulations": int(axis_metrics["challenger_simulations"]),
                "current_simulations": int(axis_metrics["current_simulations"]),
                "move_time_mean_ms": axis_metrics.get("move_time_mean_ms"),
                "move_time_p95_ms": axis_metrics.get("move_time_p95_ms"),
                "average_simulations_per_game": axis_metrics.get(
                    "mean_final_simulations"
                ),
            }
        summary[lane.name] = row
    return summary


def aggregate_heldout_summary(
    *,
    heldout_results: dict[str, dict[str, dict[str, Any]]],
    lanes: list[LaneConfig],
    default_lane: LaneConfig,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for lane in lanes:
        row: dict[str, Any] = {
            "schedule": lane.schedule_definition(),
            "suite_rows": {},
            "per_budget_runtime": {},
        }
        all_move_means: list[float] = []
        all_move_p95s: list[float] = []
        all_average_simulations: list[float] = []
        for axis in DEFAULT_BUDGET_AXES:
            ds_values: list[float] = []
            deltas: list[float] = []
            p0_scores: list[float] = []
            p1_scores: list[float] = []
            duplicate_counts: list[float] = []
            move_means: list[float] = []
            move_p95s: list[float] = []
            average_simulations: list[float] = []
            worst_suite_name: str | None = None
            worst_suite_ds: float | None = None
            for suite_name, suite_result in heldout_results.items():
                lane_metrics = suite_result[lane.name][axis]
                default_metrics = suite_result[default_lane.name][axis]
                ds = float(lane_metrics["ds"])
                delta = ds - float(default_metrics["ds"])
                row["suite_rows"].setdefault(suite_name, {})[axis] = {
                    "ds": ds,
                    "delta_vs_default": delta,
                    "effective_c_puct": float(lane_metrics["effective_c_puct"]),
                    "search_profile_hash": lane_metrics.get("search_profile_hash"),
                }
                ds_values.append(ds)
                deltas.append(delta)
                p0_scores.append(float(lane_metrics["p0_score"]))
                p1_scores.append(float(lane_metrics["p1_score"]))
                duplicate_counts.append(
                    float(lane_metrics["duplicate_trajectory_count"])
                )
                if lane_metrics.get("move_time_mean_ms") is not None:
                    move_means.append(float(lane_metrics["move_time_mean_ms"]))
                    all_move_means.append(float(lane_metrics["move_time_mean_ms"]))
                if lane_metrics.get("move_time_p95_ms") is not None:
                    move_p95s.append(float(lane_metrics["move_time_p95_ms"]))
                    all_move_p95s.append(float(lane_metrics["move_time_p95_ms"]))
                if lane_metrics.get("mean_final_simulations") is not None:
                    average_simulations.append(
                        float(lane_metrics["mean_final_simulations"])
                    )
                    all_average_simulations.append(
                        float(lane_metrics["mean_final_simulations"])
                    )
                if worst_suite_ds is None or ds < worst_suite_ds:
                    worst_suite_ds = ds
                    worst_suite_name = suite_name
            row[axis] = {
                "mean_ds": statistics.fmean(ds_values) if ds_values else None,
                "mean_delta_vs_default": statistics.fmean(deltas) if deltas else None,
                "worst_suite_ds": worst_suite_ds,
                "worst_suite_name": worst_suite_name,
                "mean_p0_score": statistics.fmean(p0_scores) if p0_scores else None,
                "mean_p1_score": statistics.fmean(p1_scores) if p1_scores else None,
                "mean_duplicate_trajectory_count": statistics.fmean(duplicate_counts)
                if duplicate_counts
                else None,
                "effective_c_puct": lane.effective_cpuct_for_axis(axis),
            }
            row["per_budget_runtime"][axis] = {
                "mean_move_latency_ms": statistics.fmean(move_means)
                if move_means
                else None,
                "p95_move_latency_ms": statistics.fmean(move_p95s)
                if move_p95s
                else None,
                "average_simulations_per_game": statistics.fmean(average_simulations)
                if average_simulations
                else None,
            }
        row["runtime_cost"] = {
            "mean_move_latency_ms": statistics.fmean(all_move_means)
            if all_move_means
            else None,
            "p95_move_latency_ms": statistics.fmean(all_move_p95s)
            if all_move_p95s
            else None,
            "average_simulations_per_game": statistics.fmean(all_average_simulations)
            if all_average_simulations
            else None,
        }
        summary[lane.name] = row
    default_latency = summary[default_lane.name]["runtime_cost"].get(
        "mean_move_latency_ms"
    )
    for lane in lanes:
        lane_latency = summary[lane.name]["runtime_cost"].get("mean_move_latency_ms")
        if (
            default_latency is not None
            and lane_latency is not None
            and float(default_latency) > 0.0
        ):
            summary[lane.name]["runtime_cost"]["relative_slowdown_vs_default"] = (
                float(lane_latency) / float(default_latency)
            ) - 1.0
        else:
            summary[lane.name]["runtime_cost"]["relative_slowdown_vs_default"] = None
    return summary


def pooled_per_opening_differences(
    *,
    heldout_results: dict[str, dict[str, dict[str, Any]]],
    lane: LaneConfig,
    default_lane: LaneConfig,
    axis: str,
) -> list[float]:
    pooled: list[float] = []
    for suite_result in heldout_results.values():
        lane_metrics = suite_result[lane.name][axis].get("per_opening_metrics", [])
        default_metrics = suite_result[default_lane.name][axis].get(
            "per_opening_metrics", []
        )
        lane_by_opening = {row["opening_prefix"]: row for row in lane_metrics}
        default_by_opening = {row["opening_prefix"]: row for row in default_metrics}
        for opening_prefix in sorted(set(lane_by_opening) & set(default_by_opening)):
            pooled.append(
                float(lane_by_opening[opening_prefix]["ds"])
                - float(default_by_opening[opening_prefix]["ds"])
            )
    return pooled


def compute_bootstrap_rows(
    *,
    heldout_results: dict[str, dict[str, dict[str, Any]]],
    lanes: list[LaneConfig],
    default_lane: LaneConfig,
    seed: int,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for lane in lanes:
        rows[lane.name] = {}
        for axis in ROBUSTNESS_AXES:
            diffs = [0.0]
            if lane.name != default_lane.name:
                diffs = pooled_per_opening_differences(
                    heldout_results=heldout_results,
                    lane=lane,
                    default_lane=default_lane,
                    axis=axis,
                )
            rows[lane.name][axis] = bootstrap_ci(diffs, seed=seed)
    return rows


def should_run_gate(lane_name: str, heldout_summary: dict[str, Any]) -> bool:
    row = heldout_summary[lane_name]
    return (
        float(row["768:768"]["mean_delta_vs_default"] or 0.0) >= 0.10
        and float(row["384:256"]["mean_delta_vs_default"] or 0.0) >= -0.03
        and float(row["1200:1200"]["mean_delta_vs_default"] or 0.0) >= -0.03
        and float(row["1200:256"]["mean_delta_vs_default"] or 0.0) >= -0.03
    )


def run_gate_for_lane(
    *,
    lane: LaneConfig,
    current_path: Path,
    out_path: Path,
    seed: int,
    workers: int,
    games: int,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _python(),
        str(REPO_ROOT / "script/ai/seat_aware_promotion_gate"),
        "--candidate-path",
        str(current_path),
        "--current-path",
        str(current_path),
        "--out",
        str(out_path),
        "--games",
        str(games),
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--budget-pairs",
        DEFAULT_GATE_BUDGET_PAIRS,
        "--c-puct",
        str(lane.default_c_puct),
        "--c-puct-schedule-json",
        lane.schedule_json(),
        "--root-policy-mode",
        "deterministic",
        "--root-temperature",
        "0.0",
        "--tactical-root-bias",
        str(lane.tactical_root_bias),
    ]
    print(f"[gate] {lane.name}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=DEFAULT_TIMEOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gate failed for {lane.name}: {result.stderr[-2000:]}")
    return json.loads(out_path.read_text(encoding="utf-8"))


def lane_assessment(
    *,
    lane: LaneConfig,
    heldout_summary: dict[str, Any],
    bootstrap_rows: dict[str, dict[str, Any]],
    gate_result: dict[str, Any] | None,
) -> dict[str, Any]:
    row = heldout_summary[lane.name]
    delta_384 = float(row["384:256"]["mean_delta_vs_default"] or 0.0)
    delta_768 = float(row["768:768"]["mean_delta_vs_default"] or 0.0)
    delta_1200 = float(row["1200:1200"]["mean_delta_vs_default"] or 0.0)
    delta_1200_256 = float(row["1200:256"]["mean_delta_vs_default"] or 0.0)
    ci_768 = bootstrap_rows[lane.name]["768:768"]
    slowdown = row["runtime_cost"].get("relative_slowdown_vs_default")
    gate_classification = gate_result.get("classification") if gate_result else None
    candidate = (
        lane.is_schedule_lane
        and delta_768 >= 0.10
        and float(ci_768["lower"]) > 0.03
        and delta_384 >= -0.03
        and delta_1200 >= -0.03
        and delta_1200_256 >= -0.03
        and slowdown is not None
        and float(slowdown) < 0.10
        and gate_classification == "high_search_breakthrough"
    )
    tradeoff = (
        lane.is_schedule_lane
        and delta_768 >= 0.10
        and (delta_384 < -0.03 or delta_1200 < -0.03 or delta_1200_256 < -0.03)
    )
    return {
        "mean_delta_384_256": delta_384,
        "mean_delta_768_768": delta_768,
        "mean_delta_1200_1200": delta_1200,
        "mean_delta_1200_256": delta_1200_256,
        "ci_768_768": ci_768,
        "runtime_slowdown": slowdown,
        "gate_classification": gate_classification,
        "preserves_high_search_breakthrough": gate_classification
        == "high_search_breakthrough",
        "classification": (
            "cpuct_schedule_candidate"
            if candidate
            else "equal_budget_cpuct_tradeoff"
            if tradeoff
            else "cpuct_schedule_no_better_than_default"
        ),
    }


def classify_run(
    lanes: list[LaneConfig], lane_assessments: dict[str, dict[str, Any]]
) -> str:
    schedule_lanes = [lane for lane in lanes if lane.is_schedule_lane]
    if any(
        lane_assessments[lane.name]["classification"] == "cpuct_schedule_candidate"
        for lane in schedule_lanes
    ):
        return "cpuct_schedule_candidate"
    if any(
        lane_assessments[lane.name]["classification"] == "equal_budget_cpuct_tradeoff"
        for lane in schedule_lanes
    ):
        return "equal_budget_cpuct_tradeoff"
    return "cpuct_schedule_no_better_than_default"


def stage_table_rows(
    stage_summary: dict[str, Any], lane_names: list[str]
) -> list[list[Any]]:
    rows = []
    for lane_name in lane_names:
        row = stage_summary[lane_name]
        rows.append(
            [
                lane_name,
                fmt(row["384:256"]["ds"]),
                fmt(row["768:256"]["ds"]),
                fmt(row["768:768"]["ds"]),
                fmt(row["1200:1200"]["ds"]),
                fmt(row["1200:256"]["ds"]),
                fmt(row["256:768"]["ds"]),
            ]
        )
    return rows


def search_profile_rows(
    stage_summary: dict[str, Any], lane_names: list[str]
) -> list[list[Any]]:
    rows = []
    for lane_name in lane_names:
        for axis in DEFAULT_BUDGET_AXES:
            row = stage_summary[lane_name][axis]
            rows.append(
                [
                    lane_name,
                    axis,
                    fmt(row["effective_c_puct"], digits=2),
                    fmt(row["tactical_root_bias"], digits=2),
                    row.get("search_profile_hash") or "n/a",
                    row["challenger_simulations"],
                    row["current_simulations"],
                ]
            )
    return rows


def heldout_mean_rows(
    heldout_summary: dict[str, Any], lane_names: list[str]
) -> list[list[Any]]:
    rows = []
    for lane_name in lane_names:
        row = heldout_summary[lane_name]
        rows.append(
            [
                lane_name,
                fmt(row["384:256"]["mean_ds"]),
                fmt(row["768:256"]["mean_ds"]),
                fmt(row["768:768"]["mean_ds"]),
                fmt(row["1200:1200"]["mean_ds"]),
                fmt(row["1200:256"]["mean_ds"]),
                fmt(row["256:768"]["mean_ds"]),
            ]
        )
    return rows


def heldout_worst_rows(
    heldout_summary: dict[str, Any], lane_names: list[str]
) -> list[list[Any]]:
    rows = []
    for lane_name in lane_names:
        row = heldout_summary[lane_name]
        rows.append(
            [
                lane_name,
                fmt(row["384:256"]["worst_suite_ds"]),
                fmt(row["768:256"]["worst_suite_ds"]),
                fmt(row["768:768"]["worst_suite_ds"]),
                fmt(row["1200:1200"]["worst_suite_ds"]),
                fmt(row["1200:256"]["worst_suite_ds"]),
                fmt(row["256:768"]["worst_suite_ds"]),
            ]
        )
    return rows


def bootstrap_table_rows(
    bootstrap_rows: dict[str, dict[str, Any]], lane_names: list[str]
) -> list[list[Any]]:
    rows = []
    for lane_name in lane_names:
        for axis in ROBUSTNESS_AXES:
            ci = bootstrap_rows[lane_name][axis]
            rows.append(
                [
                    f"{lane_name}_minus_default_{axis.replace(':', '_')}",
                    fmt(ci["mean"]),
                    fmt(ci["lower"]),
                    fmt(ci["upper"]),
                ]
            )
    return rows


def p0_p1_rows(
    heldout_summary: dict[str, Any], lane_names: list[str]
) -> list[list[Any]]:
    rows = []
    for lane_name in lane_names:
        row = heldout_summary[lane_name]["384:256"]
        p0 = float(row["mean_p0_score"] or 0.0)
        p1 = float(row["mean_p1_score"] or 0.0)
        rows.append(
            [
                lane_name,
                fmt(p0),
                fmt(p1),
                fmt(p1 - p0),
                fmt(row["mean_duplicate_trajectory_count"]),
            ]
        )
    return rows


def duplicate_rows(
    heldout_summary: dict[str, Any], lane_names: list[str]
) -> list[list[Any]]:
    rows = []
    for lane_name in lane_names:
        row = heldout_summary[lane_name]
        rows.append(
            [
                lane_name,
                fmt(row["384:256"]["mean_duplicate_trajectory_count"]),
                fmt(row["768:256"]["mean_duplicate_trajectory_count"]),
                fmt(row["768:768"]["mean_duplicate_trajectory_count"]),
                fmt(row["1200:1200"]["mean_duplicate_trajectory_count"]),
                fmt(row["1200:256"]["mean_duplicate_trajectory_count"]),
                fmt(row["256:768"]["mean_duplicate_trajectory_count"]),
            ]
        )
    return rows


def runtime_rows(
    heldout_summary: dict[str, Any], lane_names: list[str]
) -> list[list[Any]]:
    rows = []
    for lane_name in lane_names:
        runtime = heldout_summary[lane_name]["runtime_cost"]
        rows.append(
            [
                lane_name,
                fmt(runtime["mean_move_latency_ms"], digits=2),
                fmt(runtime["p95_move_latency_ms"], digits=2),
                fmt(runtime["average_simulations_per_game"], digits=2),
                fmt(runtime.get("relative_slowdown_vs_default"), digits=3),
            ]
        )
    return rows


def lane_definition_rows(
    lanes: list[LaneConfig] | list[dict[str, Any]],
) -> list[list[Any]]:
    rows = []
    for lane in lanes:
        if isinstance(lane, dict):
            lane_name = str(lane["name"])
            default_c_puct = float(lane["default_c_puct"])
            overrides = dict(lane["schedule_overrides"])
        else:
            lane_name = lane.name
            default_c_puct = lane.default_c_puct
            overrides = lane.schedule_definition()["overrides"]
        rows.append(
            [
                lane_name,
                fmt(default_c_puct, digits=2),
                json.dumps(overrides, sort_keys=True),
            ]
        )
    return rows


def gate_rows(gate_results: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for row in gate_results:
        rows.append([row["lane"], row["classification"], row["budget_pairs"]])
    return rows


def assessment_rows(
    lane_assessments: dict[str, dict[str, Any]], lane_names: list[str]
) -> list[list[Any]]:
    rows = []
    for lane_name in lane_names:
        assessment = lane_assessments[lane_name]
        rows.append(
            [
                lane_name,
                assessment["classification"],
                fmt(assessment["mean_delta_768_768"]),
                fmt(assessment["ci_768_768"]["lower"]),
                fmt(assessment["mean_delta_384_256"]),
                fmt(assessment["mean_delta_1200_1200"]),
                fmt(assessment["mean_delta_1200_256"]),
                fmt(assessment["runtime_slowdown"], digits=3),
                assessment["gate_classification"] or "not_run",
            ]
        )
    return rows


def render_report(summary: dict[str, Any]) -> str:
    inputs = summary["inputs"]
    lane_names = summary["evaluation"]["lane_names"]
    return "\n".join(
        [
            "# AlphaZero-Lite No Tactical Bias c_puct Schedule Preflight Results",
            "",
            f"**Date**: {summary['date']}",
            "",
            f"**Classification**: `{summary['classification']}`",
            "",
            "## Inputs",
            "",
            f"- Current artifact: `{inputs['current']['path']}`",
            f"- Current artifact weights SHA256: `{inputs['current']['actual_sha256']}`",
            f"- Expected SHA256: `{inputs['current']['expected_sha256']}`",
            f"- Medium suite: `{inputs['medium_suite']['path']}`",
            f"- Fixed large suite: `{inputs['fixed_large_suite']['path']}`",
            "- Held-out suites:",
            *[f"- `{row['path']}`" for row in inputs["heldout_suites"]],
            f"- Tactical root bias: `{fmt(inputs['tactical_root_bias'], digits=2)}`",
            f"- Games per opening: `{inputs['games_per_opening']}`",
            f"- Workers: `{inputs['workers']}`",
            f"- Seed: `{inputs['seed']}`",
            "",
            "## Lane Schedule Definitions",
            "",
            markdown_table(
                ["Lane", "Default c_puct", "Per-budget overrides"],
                lane_definition_rows(summary["lane_objects"]),
            ),
            "",
            "## Effective Search Profiles",
            "",
            markdown_table(
                [
                    "Lane",
                    "Budget",
                    "effective c_puct",
                    "tactical_root_bias",
                    "search_profile_hash",
                    "challenger sims",
                    "current sims",
                ],
                search_profile_rows(
                    summary["evaluation"]["medium_summary"], lane_names
                ),
            ),
            "",
            "## Medium DS Table",
            "",
            markdown_table(
                [
                    "Lane",
                    "384:256",
                    "768:256",
                    "768:768",
                    "1200:1200",
                    "1200:256",
                    "256:768",
                ],
                stage_table_rows(summary["evaluation"]["medium_summary"], lane_names),
            ),
            "",
            "## Fixed Large DS Table",
            "",
            markdown_table(
                [
                    "Lane",
                    "384:256",
                    "768:256",
                    "768:768",
                    "1200:1200",
                    "1200:256",
                    "256:768",
                ],
                stage_table_rows(
                    summary["evaluation"]["fixed_large_summary"], lane_names
                ),
            ),
            "",
            "## Held-Out Mean DS Table",
            "",
            markdown_table(
                [
                    "Lane",
                    "384:256",
                    "768:256",
                    "768:768",
                    "1200:1200",
                    "1200:256",
                    "256:768",
                ],
                heldout_mean_rows(summary["evaluation"]["heldout_summary"], lane_names),
            ),
            "",
            "## Held-Out Worst-Suite DS Table",
            "",
            markdown_table(
                [
                    "Lane",
                    "384:256",
                    "768:256",
                    "768:768",
                    "1200:1200",
                    "1200:256",
                    "256:768",
                ],
                heldout_worst_rows(
                    summary["evaluation"]["heldout_summary"], lane_names
                ),
            ),
            "",
            "## Bootstrap 95% CI",
            "",
            markdown_table(
                ["Comparison", "Mean", "Lower 95%", "Upper 95%"],
                bootstrap_table_rows(
                    summary["evaluation"]["bootstrap_cis"], lane_names
                ),
            ),
            "",
            "## P0/P1 Split For 384:256",
            "",
            markdown_table(
                ["Lane", "Mean P0", "Mean P1", "Gap", "Mean duplicates"],
                p0_p1_rows(summary["evaluation"]["heldout_summary"], lane_names),
            ),
            "",
            "## Duplicate Trajectory Counts",
            "",
            markdown_table(
                [
                    "Lane",
                    "384:256",
                    "768:256",
                    "768:768",
                    "1200:1200",
                    "1200:256",
                    "256:768",
                ],
                duplicate_rows(summary["evaluation"]["heldout_summary"], lane_names),
            ),
            "",
            "## Runtime Cost Versus Default",
            "",
            markdown_table(
                [
                    "Lane",
                    "Mean move latency",
                    "P95 move latency",
                    "Avg sims/game",
                    "Relative slowdown",
                ],
                runtime_rows(summary["evaluation"]["heldout_summary"], lane_names),
            ),
            "",
            "## Gate Classification",
            "",
            markdown_table(
                ["Lane", "Classification", "Gate budgets"],
                gate_rows(summary["gate_results"]),
            ),
            "",
            "## Lane Assessments",
            "",
            markdown_table(
                [
                    "Lane",
                    "Classification",
                    "Delta 768:768",
                    "CI lower 768:768",
                    "Delta 384:256",
                    "Delta 1200:1200",
                    "Delta 1200:256",
                    "Slowdown",
                    "Gate",
                ],
                assessment_rows(summary["lane_assessments"], lane_names),
            ),
            "",
            "## Decision Summary",
            "",
            f"- Gate lanes run: `{','.join(row['lane'] for row in summary['gate_results'])}`",
            f"- Final classification: `{summary['classification']}`",
        ]
    )


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    current_path = Path(args.current)
    current_weights = current_path / "weights.json"
    require_existing_file(current_weights, "current weights")
    medium_suite = Path(args.medium_suite)
    fixed_large_suite = Path(args.fixed_large_suite)
    heldout_suites = parse_csv_paths(args.heldout_suites)
    for suite_path in [medium_suite, fixed_large_suite, *heldout_suites]:
        require_existing_file(suite_path, f"suite {suite_path.name}")

    current_summary = verify_expected_hash(
        current_weights,
        args.expected_current_weights_sha256,
        "current artifact weights",
    )
    lane_aliases = parse_csv_strings(args.schedules)
    lanes = build_schedule_lanes(
        names=lane_aliases,
        tactical_root_bias=float(args.tactical_root_bias),
    )
    lane_map = {lane.name: lane for lane in lanes}
    default_lane = lane_map["default_cpuct_1_25_ref"]

    medium_results = evaluate_suite(
        suite_name="medium",
        suite_path=medium_suite,
        current_path=current_path,
        lanes=lanes,
        workdir=workdir,
        games_per_opening=args.games_per_opening,
        workers=args.workers,
        seed=args.seed,
        timeout=args.timeout,
    )
    fixed_large_results = evaluate_suite(
        suite_name="fixed_large",
        suite_path=fixed_large_suite,
        current_path=current_path,
        lanes=lanes,
        workdir=workdir,
        games_per_opening=args.games_per_opening,
        workers=args.workers,
        seed=args.seed,
        timeout=args.timeout,
    )
    heldout_results: dict[str, dict[str, dict[str, Any]]] = {}
    for suite_path in heldout_suites:
        heldout_results[suite_path.stem] = evaluate_suite(
            suite_name=suite_path.stem,
            suite_path=suite_path,
            current_path=current_path,
            lanes=lanes,
            workdir=workdir,
            games_per_opening=args.games_per_opening,
            workers=args.workers,
            seed=args.seed,
            timeout=args.timeout,
        )

    medium_summary = aggregate_stage_summary(
        suite_results=medium_results,
        lanes=lanes,
        default_lane=default_lane,
    )
    fixed_large_summary = aggregate_stage_summary(
        suite_results=fixed_large_results,
        lanes=lanes,
        default_lane=default_lane,
    )
    heldout_summary = aggregate_heldout_summary(
        heldout_results=heldout_results,
        lanes=lanes,
        default_lane=default_lane,
    )
    bootstrap_rows = compute_bootstrap_rows(
        heldout_results=heldout_results,
        lanes=lanes,
        default_lane=default_lane,
        seed=args.seed,
    )

    gate_lane_names = [default_lane.name]
    for lane in lanes:
        if lane.is_schedule_lane and should_run_gate(lane.name, heldout_summary):
            gate_lane_names.append(lane.name)
    gate_lane_names = list(dict.fromkeys(gate_lane_names))
    gate_results: list[dict[str, Any]] = []
    gate_report_by_lane: dict[str, dict[str, Any]] = {}
    for lane_name in gate_lane_names:
        lane = lane_map[lane_name]
        gate_report = run_gate_for_lane(
            lane=lane,
            current_path=current_path,
            out_path=workdir / "gate" / lane.name / "report.json",
            seed=args.seed,
            workers=args.workers,
            games=args.gate_games,
        )
        gate_report_by_lane[lane.name] = gate_report
        gate_results.append(
            {
                "lane": lane.name,
                "classification": gate_report.get("classification"),
                "budget_pairs": DEFAULT_GATE_BUDGET_PAIRS,
                "budget_results": gate_report.get("budget_results", {}),
            }
        )

    lane_assessments = {
        lane.name: lane_assessment(
            lane=lane,
            heldout_summary=heldout_summary,
            bootstrap_rows=bootstrap_rows,
            gate_result=gate_report_by_lane.get(lane.name),
        )
        for lane in lanes
    }
    classification = classify_run(lanes, lane_assessments)

    summary = {
        "schema": SUMMARY_SCHEMA,
        "date": str(date.today()),
        "classification": classification,
        "inputs": {
            "current": current_summary,
            "medium_suite": build_input_summary(medium_suite),
            "fixed_large_suite": build_input_summary(fixed_large_suite),
            "heldout_suites": [build_input_summary(path) for path in heldout_suites],
            "tactical_root_bias": float(args.tactical_root_bias),
            "games_per_opening": int(args.games_per_opening),
            "workers": int(args.workers),
            "seed": int(args.seed),
            "schedule_aliases": lane_aliases,
        },
        "lane_objects": [
            {
                "name": lane.name,
                "default_c_puct": lane.default_c_puct,
                "schedule_overrides": lane.schedule_overrides,
                "tactical_root_bias": lane.tactical_root_bias,
                "is_schedule_lane": lane.is_schedule_lane,
            }
            for lane in lanes
        ],
        "evaluation": {
            "lane_names": [lane.name for lane in lanes],
            "medium_summary": medium_summary,
            "fixed_large_summary": fixed_large_summary,
            "heldout_summary": heldout_summary,
            "bootstrap_cis": bootstrap_rows,
        },
        "gate_results": gate_results,
        "lane_assessments": lane_assessments,
    }

    summary_path = workdir / SUMMARY_FILENAME
    write_json(summary_path, summary)
    report_text = render_report(summary)
    REPORT_PATH.write_text(report_text + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
