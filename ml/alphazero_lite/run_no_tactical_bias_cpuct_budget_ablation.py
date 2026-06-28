#!/usr/bin/env python3
"""No-tactical-bias deterministic PUCT c_puct and budget ablation."""

from __future__ import annotations

import argparse
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

from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    bootstrap_ci,
    gate_budget_results,
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


SUMMARY_SCHEMA = "azlite_no_tactical_bias_cpuct_budget_ablation_v1"
SUMMARY_FILENAME = "summary_metrics.json"
REPORT_PATH = (
    REPO_ROOT / "docs/alphazero-lite-no-tactical-bias-cpuct-budget-ablation-results.md"
)
DEFAULT_C_PUCT_VALUES = "0.75,1.0,1.25,1.5,1.75,2.0"
DEFAULT_BUDGET_LANES = "384:384,768:768,384:512,512:384,768:1024,1024:768"
DEFAULT_GATE_BUDGET_PAIRS = "384:256,1200:1200,1200:256,256:768"
DEFAULT_BOOTSTRAP_BUDGETS = ("384:256", "768:768", "1200:1200", "1200:256")
DEFAULT_TACTICAL_ROOT_BIAS = 0.0
DEFAULT_C_PUCT = 1.25
CORE_AXIS_ORDER = ("384:256", "768:768", "1200:1200", "1200:256")


def _python() -> str:
    candidate = REPO_ROOT / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


@dataclass(frozen=True)
class LaneConfig:
    name: str
    description: str
    c_puct: float
    tactical_root_bias: float = DEFAULT_TACTICAL_ROOT_BIAS
    axis_budget_overrides: dict[str, tuple[int, int]] | None = None
    source_kind: str = "baseline"
    source_detail: str | None = None

    def search_profile(self) -> dict[str, Any]:
        return {
            "c_puct": float(self.c_puct),
            "tactical_root_bias": float(self.tactical_root_bias),
            "root_policy_mode": "deterministic",
            "search_mode": "full",
            "root_prior_transform": None,
            "value_trust_schedule": None,
        }

    def actual_budget_for_axis(self, axis: str) -> tuple[int, int]:
        if self.axis_budget_overrides and axis in self.axis_budget_overrides:
            return self.axis_budget_overrides[axis]
        left, right = axis.split(":", 1)
        return int(left), int(right)

    def actual_budget_label_for_axis(self, axis: str) -> str:
        return budget_pair_label(*self.actual_budget_for_axis(axis))

    def requested_budget_pairs(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for axis in CORE_AXIS_ORDER:
            pair = self.actual_budget_for_axis(axis)
            if pair not in pairs:
                pairs.append(pair)
        return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="/tmp/azlite_no_tactical_bias_cpuct_budget_ablation"
    )
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--cpuct-values", default=DEFAULT_C_PUCT_VALUES)
    parser.add_argument("--budget-lanes", default=DEFAULT_BUDGET_LANES)
    parser.add_argument("--tactical-root-bias", type=float, default=0.0)
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--gate-games", type=int, default=120)
    return parser.parse_args()


def parse_csv_paths(text: str) -> list[Path]:
    values = [Path(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one path")
    return values


def parse_csv_floats(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one float")
    return values


def parse_budget_pairs(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(":", 1)
        pairs.append((int(left), int(right)))
    if not pairs:
        raise ValueError("expected at least one budget pair")
    return pairs


def budget_pair_label(challenger: int, current: int) -> str:
    return f"{challenger}:{current}"


def build_cpuct_lanes(
    cpuct_values: list[float], tactical_root_bias: float
) -> list[LaneConfig]:
    name_map = {
        0.75: "no_tactical_cpuct_0_75",
        1.0: "no_tactical_cpuct_1_00",
        1.25: "no_tactical_cpuct_1_25_ref",
        1.5: "no_tactical_cpuct_1_50",
        1.75: "no_tactical_cpuct_1_75",
        2.0: "no_tactical_cpuct_2_00",
    }
    lanes: list[LaneConfig] = []
    for value in cpuct_values:
        lane_name = name_map.get(round(value, 2), f"no_tactical_cpuct_{value:.2f}")
        lanes.append(
            LaneConfig(
                name=lane_name,
                description=f"No tactical bias with c_puct={value:.2f}.",
                c_puct=float(value),
                tactical_root_bias=float(tactical_root_bias),
                source_kind="cpuct",
                source_detail=f"c_puct={value:.2f}",
            )
        )
    return lanes


def build_budget_lanes(
    budget_pairs: list[tuple[int, int]],
    *,
    base_c_puct: float,
    tactical_root_bias: float,
) -> list[LaneConfig]:
    lane_specs = {
        (384, 384): (
            "no_tactical_equalized_384_384",
            "Equalize the 384-axis budget while keeping other robustness axes at baseline.",
            {"384:256": (384, 384)},
        ),
        (768, 768): (
            "no_tactical_equalized_768_768",
            "Explicit equalized 768-axis budget lane.",
            {"768:768": (768, 768)},
        ),
        (384, 512): (
            "no_tactical_high_current_384_512",
            "Increase current simulations on the 384-axis only.",
            {"384:256": (384, 512)},
        ),
        (512, 384): (
            "no_tactical_high_challenger_512_384",
            "Increase challenger simulations on the 384-axis only.",
            {"384:256": (512, 384)},
        ),
        (768, 1024): (
            "no_tactical_high_current_768_1024",
            "Increase current simulations on the 768-axis only.",
            {"768:768": (768, 1024)},
        ),
        (1024, 768): (
            "no_tactical_high_challenger_1024_768",
            "Increase challenger simulations on the 768-axis only.",
            {"768:768": (1024, 768)},
        ),
    }
    lanes: list[LaneConfig] = []
    for pair in budget_pairs:
        spec = lane_specs.get(pair)
        if spec is None:
            continue
        name, description, overrides = spec
        lanes.append(
            LaneConfig(
                name=name,
                description=description,
                c_puct=float(base_c_puct),
                tactical_root_bias=float(tactical_root_bias),
                axis_budget_overrides=overrides,
                source_kind="budget",
                source_detail=budget_pair_label(*pair),
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
    pair: tuple[int, int],
    workdir: Path,
    games_per_opening: int,
    workers: int,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    challenger_sims, current_sims = pair
    label = budget_pair_label(challenger_sims, current_sims)
    lane_dir = workdir / suite_name / lane.name / label.replace(":", "_")
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
        "challenger_simulations": challenger_sims,
        "current_simulations": current_sims,
        "games_per_opening": games_per_opening,
        "seed": seed,
        "workers": workers,
        "search_profile": lane.search_profile(),
    }
    if metrics_path.is_file() and arena_json.is_file() and games_jsonl.is_file():
        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        if cached.get("cache_context") == cache_context:
            return cached

    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/arena.py"),
        "--challenger",
        str(current_path),
        "--current",
        str(current_path),
        "--challenger-simulations",
        str(challenger_sims),
        "--current-simulations",
        str(current_sims),
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
        str(lane.c_puct),
        "--tactical-root-bias",
        str(lane.tactical_root_bias),
    ]
    print(
        f"[arena] {suite_name} {lane.name} {label} c_puct={lane.c_puct:.2f}",
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
            f"arena failed for {suite_name} {lane.name} {label}: {result.stderr[-2000:]}"
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
        "cache_context": cache_context,
        "suite_name": suite_name,
        "suite_path": str(suite_path),
        "lane": lane.name,
        "budget_pair": label,
        "challenger_simulations": challenger_sims,
        "current_simulations": current_sims,
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
        budget_results: dict[str, Any] = {}
        for pair in lane.requested_budget_pairs():
            label = budget_pair_label(*pair)
            budget_results[label] = run_arena_for_budget(
                lane=lane,
                suite_name=suite_name,
                suite_path=suite_path,
                suite_entries=suite_entries,
                current_path=current_path,
                pair=pair,
                workdir=workdir,
                games_per_opening=games_per_opening,
                workers=workers,
                seed=seed,
                timeout=timeout,
            )
        results[lane.name] = budget_results
    return results


def lane_metrics_by_axis(
    suite_results: dict[str, dict[str, Any]], lane: LaneConfig
) -> dict[str, dict[str, Any]]:
    lane_results = suite_results[lane.name]
    axis_rows: dict[str, dict[str, Any]] = {}
    for axis in CORE_AXIS_ORDER:
        axis_rows[axis] = lane_results[lane.actual_budget_label_for_axis(axis)]
    return axis_rows


def delta_vs_baseline(
    suite_results: dict[str, dict[str, Any]],
    lane: LaneConfig,
    baseline_lane: LaneConfig,
) -> dict[str, float]:
    lane_axes = lane_metrics_by_axis(suite_results, lane)
    baseline_axes = lane_metrics_by_axis(suite_results, baseline_lane)
    return {
        axis: float(lane_axes[axis]["ds"]) - float(baseline_axes[axis]["ds"])
        for axis in CORE_AXIS_ORDER
    }


def balanced_score_from_deltas(deltas: dict[str, float]) -> float:
    return (
        float(deltas.get("384:256", 0.0))
        + 0.5 * float(deltas.get("768:768", 0.0))
        + 0.25 * float(deltas.get("1200:1200", 0.0))
        + 0.25 * float(deltas.get("1200:256", 0.0))
    )


def deltas_are_material(deltas: dict[str, float], *, epsilon: float = 1e-9) -> bool:
    return any(abs(float(value)) > epsilon for value in deltas.values())


def medium_improvement_criteria(deltas: dict[str, float]) -> bool:
    return (
        deltas.get("384:256", 0.0) >= 0.05 and deltas.get("768:768", 0.0) >= -0.08
    ) or (deltas.get("768:768", 0.0) >= 0.05 and deltas.get("384:256", 0.0) >= -0.08)


def robust_primary_candidate(deltas: dict[str, float]) -> bool:
    return (
        deltas.get("384:256", 0.0) >= 0.05
        and deltas.get("768:768", 0.0) >= -0.08
        and deltas.get("1200:1200", 0.0) >= -0.03
        and deltas.get("1200:256", 0.0) >= -0.03
    )


def robust_equal_candidate(deltas: dict[str, float]) -> bool:
    return (
        deltas.get("768:768", 0.0) >= 0.08
        and deltas.get("384:256", 0.0) >= -0.05
        and deltas.get("1200:1200", 0.0) >= -0.03
        and deltas.get("1200:256", 0.0) >= -0.03
    )


def select_budget_eval_cpuct(
    cpuct_lanes: list[LaneConfig],
    medium_results: dict[str, dict[str, Any]],
    baseline_lane: LaneConfig,
) -> float:
    ranked: list[tuple[float, LaneConfig, dict[str, float]]] = []
    for lane in cpuct_lanes:
        if lane.name == baseline_lane.name:
            continue
        deltas = delta_vs_baseline(medium_results, lane, baseline_lane)
        ranked.append((balanced_score_from_deltas(deltas), lane, deltas))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return baseline_lane.c_puct
    best_score, best_lane, best_deltas = ranked[0]
    if best_score >= 0.05 and (
        robust_primary_candidate(best_deltas) or robust_equal_candidate(best_deltas)
    ):
        return best_lane.c_puct
    return baseline_lane.c_puct


def build_combined_lanes(
    *,
    cpuct_lanes: list[LaneConfig],
    budget_lanes: list[LaneConfig],
    medium_results: dict[str, dict[str, Any]],
    baseline_lane: LaneConfig,
) -> list[LaneConfig]:
    cpuct_ranked: list[tuple[float, LaneConfig]] = []
    for lane in cpuct_lanes:
        if lane.name == baseline_lane.name:
            continue
        score = balanced_score_from_deltas(
            delta_vs_baseline(medium_results, lane, baseline_lane)
        )
        cpuct_ranked.append((score, lane))
    cpuct_ranked.sort(key=lambda item: item[0], reverse=True)

    budget_ranked: list[tuple[float, LaneConfig, dict[str, float]]] = []
    for lane in budget_lanes:
        deltas = delta_vs_baseline(medium_results, lane, baseline_lane)
        budget_ranked.append((balanced_score_from_deltas(deltas), lane, deltas))
    budget_ranked.sort(key=lambda item: item[0], reverse=True)
    if not cpuct_ranked or not budget_ranked:
        return []

    best_budget_score, best_budget_lane, best_budget_deltas = budget_ranked[0]
    if best_budget_score <= 0.0 or not medium_improvement_criteria(best_budget_deltas):
        return []

    lanes: list[LaneConfig] = []
    for index, (cpuct_score, cpuct_lane) in enumerate(cpuct_ranked[:2], start=1):
        cpuct_deltas = delta_vs_baseline(medium_results, cpuct_lane, baseline_lane)
        if cpuct_score <= 0.0 or not medium_improvement_criteria(cpuct_deltas):
            continue
        lanes.append(
            LaneConfig(
                name=(
                    f"combined_{index}_{cpuct_lane.name}_plus_{best_budget_lane.name}"
                ),
                description=(
                    f"Combined medium-promising c_puct lane {cpuct_lane.name} with "
                    f"budget lane {best_budget_lane.name}."
                ),
                c_puct=float(cpuct_lane.c_puct),
                tactical_root_bias=float(cpuct_lane.tactical_root_bias),
                axis_budget_overrides=best_budget_lane.axis_budget_overrides,
                source_kind="combined",
                source_detail=(
                    f"cpuct={cpuct_lane.name};budget={best_budget_lane.name}"
                ),
            )
        )
    return lanes


def select_fixed_large_lanes(
    *,
    lanes: list[LaneConfig],
    medium_results: dict[str, dict[str, Any]],
    baseline_lane: LaneConfig,
) -> list[str]:
    scored: list[tuple[float, str]] = []
    mandatory: set[str] = set()
    for lane in lanes:
        if lane.name == baseline_lane.name:
            continue
        deltas = delta_vs_baseline(medium_results, lane, baseline_lane)
        if not deltas_are_material(deltas):
            continue
        if medium_improvement_criteria(deltas):
            mandatory.add(lane.name)
        scored.append((balanced_score_from_deltas(deltas), lane.name))
    scored.sort(reverse=True)
    selected_non_baseline: list[str] = []
    for _score, name in scored:
        if name in mandatory and name not in selected_non_baseline:
            selected_non_baseline.append(name)
    for _score, name in scored:
        if name not in selected_non_baseline:
            selected_non_baseline.append(name)
        if len(selected_non_baseline) >= 6:
            break
    return [baseline_lane.name, *selected_non_baseline[:6]]


def select_heldout_lanes(
    *,
    fixed_large_results: dict[str, dict[str, Any]],
    fixed_large_lanes: list[LaneConfig],
    baseline_lane: LaneConfig,
) -> list[str]:
    lane_map = {lane.name: lane for lane in fixed_large_lanes}
    scored: list[tuple[float, str]] = []
    selected = [baseline_lane.name]
    for lane in fixed_large_lanes:
        if lane.name == baseline_lane.name:
            continue
        deltas = delta_vs_baseline(fixed_large_results, lane, baseline_lane)
        if not deltas_are_material(deltas):
            continue
        scored.append((balanced_score_from_deltas(deltas), lane.name))
    scored.sort(reverse=True)
    for _score, name in scored[:3]:
        if name not in selected:
            selected.append(name)
    for lane_name in [
        lane.name for lane in fixed_large_lanes if lane.name != baseline_lane.name
    ]:
        deltas = delta_vs_baseline(
            fixed_large_results, lane_map[lane_name], baseline_lane
        )
        if not deltas_are_material(deltas):
            continue
        if (
            deltas.get("384:256", 0.0) >= 0.08 and deltas.get("768:768", 0.0) >= -0.08
        ) or (
            deltas.get("768:768", 0.0) >= 0.08 and deltas.get("384:256", 0.0) >= -0.05
        ):
            if lane_name not in selected:
                selected.append(lane_name)
    return selected


def aggregate_stage_summary(
    *,
    suite_results: dict[str, dict[str, Any]],
    lanes: list[LaneConfig],
    baseline_lane: LaneConfig,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for lane in lanes:
        axis_rows = lane_metrics_by_axis(suite_results, lane)
        deltas = delta_vs_baseline(suite_results, lane, baseline_lane)
        row: dict[str, Any] = {
            "balanced_score": balanced_score_from_deltas(deltas),
        }
        for axis, axis_metrics in axis_rows.items():
            row[axis] = {
                "ds": float(axis_metrics["ds"]),
                "delta_vs_baseline": float(deltas[axis]),
                "p0_score": float(axis_metrics["p0_score"]),
                "p1_score": float(axis_metrics["p1_score"]),
                "duplicate_trajectory_count": int(
                    axis_metrics["duplicate_trajectory_count"]
                ),
                "duplicate_trajectory_rate": float(
                    axis_metrics["duplicate_trajectory_rate"]
                ),
                "mean_move_latency_ms": axis_metrics.get("move_time_mean_ms"),
                "p95_move_latency_ms": axis_metrics.get("move_time_p95_ms"),
                "average_simulations_per_game": axis_metrics.get(
                    "mean_final_simulations"
                ),
                "actual_budget_pair": lane.actual_budget_label_for_axis(axis),
                "search_profile": axis_metrics.get("search_profile"),
                "search_profile_hash": axis_metrics.get("search_profile_hash"),
            }
        summary[lane.name] = row
    return summary


def aggregate_heldout_summary(
    *,
    heldout_results: dict[str, dict[str, dict[str, Any]]],
    lanes: list[LaneConfig],
    baseline_lane: LaneConfig,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for lane in lanes:
        lane_row: dict[str, Any] = {"suite_rows": {}, "runtime_cost": {}}
        for axis in CORE_AXIS_ORDER:
            ds_values: list[float] = []
            deltas: list[float] = []
            p0_scores: list[float] = []
            p1_scores: list[float] = []
            duplicates: list[float] = []
            move_means: list[float] = []
            move_p95s: list[float] = []
            sims: list[float] = []
            worst_suite_name: str | None = None
            worst_suite_ds: float | None = None
            actual_budget = lane.actual_budget_label_for_axis(axis)
            for suite_name, suite_row in heldout_results.items():
                lane_metrics = lane_metrics_by_axis(suite_row, lane)[axis]
                baseline_metrics = lane_metrics_by_axis(suite_row, baseline_lane)[axis]
                ds = float(lane_metrics["ds"])
                delta = ds - float(baseline_metrics["ds"])
                lane_row["suite_rows"].setdefault(suite_name, {})[axis] = {
                    "ds": ds,
                    "delta_vs_baseline": delta,
                    "actual_budget_pair": actual_budget,
                }
                ds_values.append(ds)
                deltas.append(delta)
                p0_scores.append(float(lane_metrics["p0_score"]))
                p1_scores.append(float(lane_metrics["p1_score"]))
                duplicates.append(float(lane_metrics["duplicate_trajectory_count"]))
                if lane_metrics.get("move_time_mean_ms") is not None:
                    move_means.append(float(lane_metrics["move_time_mean_ms"]))
                if lane_metrics.get("move_time_p95_ms") is not None:
                    move_p95s.append(float(lane_metrics["move_time_p95_ms"]))
                if lane_metrics.get("mean_final_simulations") is not None:
                    sims.append(float(lane_metrics["mean_final_simulations"]))
                if worst_suite_ds is None or ds < worst_suite_ds:
                    worst_suite_ds = ds
                    worst_suite_name = suite_name
            lane_row[axis] = {
                "mean_ds": statistics.fmean(ds_values) if ds_values else None,
                "mean_delta_vs_baseline": statistics.fmean(deltas) if deltas else None,
                "worst_suite_ds": worst_suite_ds,
                "worst_suite_name": worst_suite_name,
                "mean_p0_score": statistics.fmean(p0_scores) if p0_scores else None,
                "mean_p1_score": statistics.fmean(p1_scores) if p1_scores else None,
                "mean_duplicate_trajectory_count": statistics.fmean(duplicates)
                if duplicates
                else None,
                "actual_budget_pair": actual_budget,
            }
            lane_row["runtime_cost"][axis] = {
                "mean_move_latency_ms": statistics.fmean(move_means)
                if move_means
                else None,
                "p95_move_latency_ms": statistics.fmean(move_p95s)
                if move_p95s
                else None,
                "average_simulations_per_game": statistics.fmean(sims)
                if sims
                else None,
            }
        summary[lane.name] = lane_row
    baseline_runtime = summary[baseline_lane.name]["runtime_cost"]["384:256"]
    baseline_latency = baseline_runtime.get("mean_move_latency_ms")
    for lane in lanes:
        runtime = summary[lane.name]["runtime_cost"]["384:256"]
        lane_latency = runtime.get("mean_move_latency_ms")
        if (
            baseline_latency is not None
            and lane_latency is not None
            and float(baseline_latency) > 0.0
        ):
            runtime["relative_slowdown_vs_baseline"] = (
                float(lane_latency) / float(baseline_latency)
            ) - 1.0
        else:
            runtime["relative_slowdown_vs_baseline"] = None
    return summary


def pooled_per_opening_differences(
    *,
    heldout_results: dict[str, dict[str, dict[str, Any]]],
    lane: LaneConfig,
    baseline_lane: LaneConfig,
    axis: str,
) -> list[float]:
    pooled: list[float] = []
    for suite_row in heldout_results.values():
        lane_metrics = lane_metrics_by_axis(suite_row, lane)[axis].get(
            "per_opening_metrics", []
        )
        baseline_metrics = lane_metrics_by_axis(suite_row, baseline_lane)[axis].get(
            "per_opening_metrics", []
        )
        lane_by_opening = {row["opening_prefix"]: row for row in lane_metrics}
        baseline_by_opening = {row["opening_prefix"]: row for row in baseline_metrics}
        common = sorted(set(lane_by_opening) & set(baseline_by_opening))
        for opening_prefix in common:
            pooled.append(
                float(lane_by_opening[opening_prefix]["ds"])
                - float(baseline_by_opening[opening_prefix]["ds"])
            )
    return pooled


def compute_bootstrap_rows(
    *,
    heldout_results: dict[str, dict[str, dict[str, Any]]],
    lanes: list[LaneConfig],
    baseline_lane: LaneConfig,
    seed: int,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for lane in lanes:
        lane_rows: dict[str, Any] = {}
        for axis in DEFAULT_BOOTSTRAP_BUDGETS:
            diffs = [0.0]
            if lane.name != baseline_lane.name:
                diffs = pooled_per_opening_differences(
                    heldout_results=heldout_results,
                    lane=lane,
                    baseline_lane=baseline_lane,
                    axis=axis,
                )
            lane_rows[axis] = bootstrap_ci(diffs, seed=seed)
        rows[lane.name] = lane_rows
    return rows


def gate_budget_pairs_for_lane(lane: LaneConfig) -> str:
    default_pairs = [
        item.strip() for item in DEFAULT_GATE_BUDGET_PAIRS.split(",") if item.strip()
    ]
    low_budget = lane.actual_budget_label_for_axis("384:256")
    return ",".join([low_budget, *default_pairs[1:]])


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
        gate_budget_pairs_for_lane(lane),
        "--c-puct",
        str(lane.c_puct),
        "--root-policy-mode",
        "deterministic",
        "--root-temperature",
        "0.0",
        "--tactical-root-bias",
        str(lane.tactical_root_bias),
    ]
    print(f"[gate] {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=14400,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gate failed for {lane.name}: {result.stderr[-2000:]}")
    return json.loads(out_path.read_text(encoding="utf-8"))


def lane_decision(
    *,
    lane: LaneConfig,
    heldout_summary: dict[str, Any],
    bootstrap_rows: dict[str, dict[str, Any]],
    gate_result: dict[str, Any] | None,
    baseline_lane: LaneConfig,
) -> dict[str, Any]:
    budgets = heldout_summary[lane.name]
    delta_384 = float(budgets["384:256"]["mean_delta_vs_baseline"] or 0.0)
    delta_768 = float(budgets["768:768"]["mean_delta_vs_baseline"] or 0.0)
    delta_1200 = float(budgets["1200:1200"]["mean_delta_vs_baseline"] or 0.0)
    delta_1200_256 = float(budgets["1200:256"]["mean_delta_vs_baseline"] or 0.0)
    slowdown = heldout_summary[lane.name]["runtime_cost"]["384:256"].get(
        "relative_slowdown_vs_baseline"
    )
    gate_classification = gate_result.get("classification") if gate_result else None
    return {
        "improves_384": delta_384 >= 0.05,
        "ci_384_lower_gt_001": float(bootstrap_rows[lane.name]["384:256"]["lower"])
        > 0.01,
        "robust_768": delta_768 >= -0.08,
        "robust_1200_1200": delta_1200 >= -0.03,
        "robust_1200_256": delta_1200_256 >= -0.03,
        "slowdown_lt_25pct": slowdown is not None and float(slowdown) < 0.25,
        "gate_classification": gate_classification,
        "preserves_high_search_breakthrough": gate_classification
        == "high_search_breakthrough",
        "is_baseline": lane.name == baseline_lane.name,
    }


def classify_run(
    *,
    heldout_lanes: list[LaneConfig],
    heldout_summary: dict[str, Any],
    lane_decisions: dict[str, dict[str, Any]],
    baseline_lane: LaneConfig,
) -> str:
    upgrade_candidates: list[str] = []
    expensive_candidates: list[str] = []
    cpuct_material = False
    budget_material = False
    for lane in heldout_lanes:
        if lane.name == baseline_lane.name:
            continue
        decisions = lane_decisions[lane.name]
        strength_ok = (
            decisions["improves_384"]
            and decisions["ci_384_lower_gt_001"]
            and decisions["robust_768"]
            and decisions["robust_1200_1200"]
            and decisions["robust_1200_256"]
            and decisions["preserves_high_search_breakthrough"]
        )
        if strength_ok:
            if decisions["slowdown_lt_25pct"]:
                upgrade_candidates.append(lane.name)
            else:
                expensive_candidates.append(lane.name)
        delta_384 = float(
            heldout_summary[lane.name]["384:256"]["mean_delta_vs_baseline"] or 0.0
        )
        delta_768 = float(
            heldout_summary[lane.name]["768:768"]["mean_delta_vs_baseline"] or 0.0
        )
        if lane.source_kind in {"cpuct", "combined"} and (
            abs(delta_384) >= 0.05 or abs(delta_768) >= 0.08
        ):
            cpuct_material = True
        if lane.source_kind in {"budget", "combined"} and (
            abs(delta_384) >= 0.05 or abs(delta_768) >= 0.08
        ):
            budget_material = True
    if upgrade_candidates:
        return "runtime_profile_upgrade_candidate"
    if expensive_candidates:
        return "expensive_runtime_profile_candidate"
    if cpuct_material:
        return "cpuct_sensitive_runtime"
    if budget_material:
        return "budget_allocation_sensitive_runtime"
    return "no_tactical_default_still_best"


def lane_to_summary_row(lane: LaneConfig) -> dict[str, Any]:
    return {
        "name": lane.name,
        "description": lane.description,
        "source_kind": lane.source_kind,
        "source_detail": lane.source_detail,
        "c_puct": float(lane.c_puct),
        "tactical_root_bias": float(lane.tactical_root_bias),
        "search_profile": lane.search_profile(),
        "axis_budget_pairs": {
            axis: lane.actual_budget_label_for_axis(axis) for axis in CORE_AXIS_ORDER
        },
    }


def build_stage_table_rows(
    stage_summary: dict[str, Any], lane_names: list[str]
) -> list[list[Any]]:
    rows = []
    for lane_name in lane_names:
        row = stage_summary[lane_name]
        rows.append(
            [
                lane_name,
                fmt(row["384:256"]["ds"]),
                fmt(row["768:768"]["ds"]),
                fmt(row["1200:1200"]["ds"]),
                fmt(row["1200:256"]["ds"]),
                fmt(row["balanced_score"]),
            ]
        )
    return rows


def build_budget_map_rows(lanes: list[LaneConfig]) -> list[list[Any]]:
    rows = []
    for lane in lanes:
        rows.append(
            [
                lane.name,
                fmt(lane.c_puct, digits=2),
                fmt(lane.tactical_root_bias, digits=2),
                lane.actual_budget_label_for_axis("384:256"),
                lane.actual_budget_label_for_axis("768:768"),
                lane.actual_budget_label_for_axis("1200:1200"),
                lane.actual_budget_label_for_axis("1200:256"),
            ]
        )
    return rows


def build_search_profile_hash_rows(
    stage_summary: dict[str, Any], lane_names: list[str]
) -> list[list[Any]]:
    rows = []
    for lane_name in lane_names:
        for axis in CORE_AXIS_ORDER:
            row = stage_summary[lane_name][axis]
            rows.append(
                [
                    lane_name,
                    axis,
                    row["actual_budget_pair"],
                    row.get("search_profile_hash") or "n/a",
                    json.dumps(row.get("search_profile") or {}, sort_keys=True),
                ]
            )
    return rows


def build_heldout_rows(
    heldout_summary: dict[str, Any], lane_names: list[str]
) -> list[list[Any]]:
    rows = []
    for lane_name in lane_names:
        row = heldout_summary[lane_name]
        rows.append(
            [
                lane_name,
                fmt(row["384:256"]["mean_ds"]),
                fmt(row["384:256"]["worst_suite_ds"]),
                fmt(row["768:768"]["mean_ds"]),
                fmt(row["768:768"]["worst_suite_ds"]),
                fmt(row["1200:1200"]["mean_ds"]),
                fmt(row["1200:256"]["mean_ds"]),
            ]
        )
    return rows


def build_bootstrap_rows(
    bootstrap_rows_by_lane: dict[str, dict[str, Any]], lane_names: list[str]
) -> list[list[Any]]:
    rows = []
    for lane_name in lane_names:
        for axis in DEFAULT_BOOTSTRAP_BUDGETS:
            ci = bootstrap_rows_by_lane[lane_name][axis]
            rows.append(
                [
                    f"{lane_name}_minus_baseline_{axis.replace(':', '_')}",
                    fmt(ci["mean"]),
                    fmt(ci["lower"]),
                    fmt(ci["upper"]),
                ]
            )
    return rows


def build_p0_rows(
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


def build_runtime_rows(
    heldout_summary: dict[str, Any], lane_names: list[str]
) -> list[list[Any]]:
    rows = []
    for lane_name in lane_names:
        runtime = heldout_summary[lane_name]["runtime_cost"]["384:256"]
        rows.append(
            [
                lane_name,
                fmt(runtime["mean_move_latency_ms"], digits=2),
                fmt(runtime["p95_move_latency_ms"], digits=2),
                fmt(runtime["average_simulations_per_game"], digits=2),
                fmt(runtime.get("relative_slowdown_vs_baseline"), digits=3),
            ]
        )
    return rows


def render_report(summary: dict[str, Any]) -> str:
    inputs = summary["inputs"]
    heldout_lane_names = summary["evaluation"]["heldout_lane_names"]
    gate_lines = [
        f"- {row['lane']}: `{row['classification']}` using gate budgets `{row['budget_pairs']}`"
        for row in summary["gate_results"]
    ]
    return "\n".join(
        [
            "# AlphaZero-Lite No Tactical Bias c_puct Budget Ablation Results",
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
            f"- Requested c_puct values: `{','.join(inputs['requested_cpuct_values'])}`",
            f"- Requested budget lanes: `{','.join(inputs['requested_budget_lanes'])}`",
            f"- Tactical root bias: `{fmt(inputs['tactical_root_bias'], digits=2)}`",
            f"- Games per opening: `{inputs['games_per_opening']}`",
            f"- Workers: `{inputs['workers']}`",
            f"- Seed: `{inputs['seed']}`",
            "",
            "## Lane Budget Map",
            "",
            markdown_table(
                [
                    "Lane",
                    "c_puct",
                    "tactical_root_bias",
                    "384-axis",
                    "768-axis",
                    "1200:1200",
                    "1200:256",
                ],
                build_budget_map_rows(
                    [
                        LaneConfig(**lane) if False else lane
                        for lane in summary["_lane_objects"]
                    ]
                ),
            ),
            "",
            "## Effective Search Profiles",
            "",
            markdown_table(
                [
                    "Lane",
                    "Axis",
                    "Actual budget",
                    "search_profile_hash",
                    "search_profile",
                ],
                build_search_profile_hash_rows(
                    summary["evaluation"]["medium_summary"],
                    summary["evaluation"]["medium_lane_names"],
                ),
            ),
            "",
            "## Medium DS Table",
            "",
            markdown_table(
                ["Lane", "384-axis", "768-axis", "1200:1200", "1200:256", "Balanced"],
                build_stage_table_rows(
                    summary["evaluation"]["medium_summary"],
                    summary["evaluation"]["medium_lane_names"],
                ),
            ),
            "",
            "## Fixed Large DS Table",
            "",
            markdown_table(
                ["Lane", "384-axis", "768-axis", "1200:1200", "1200:256", "Balanced"],
                build_stage_table_rows(
                    summary["evaluation"]["fixed_large_summary"],
                    summary["evaluation"]["fixed_large_lane_names"],
                ),
            ),
            "",
            "## Held-Out Mean And Worst-Suite DS Table",
            "",
            markdown_table(
                [
                    "Lane",
                    "Mean 384-axis",
                    "Worst 384-axis",
                    "Mean 768-axis",
                    "Worst 768-axis",
                    "Mean 1200:1200",
                    "Mean 1200:256",
                ],
                build_heldout_rows(
                    summary["evaluation"]["heldout_summary"], heldout_lane_names
                ),
            ),
            "",
            "## Bootstrap CI",
            "",
            markdown_table(
                ["Comparison", "Mean", "Lower 95%", "Upper 95%"],
                build_bootstrap_rows(
                    summary["evaluation"]["bootstrap_cis"], heldout_lane_names
                ),
            ),
            "",
            "## P0/P1 Split For 384-axis",
            "",
            markdown_table(
                ["Lane", "Mean P0", "Mean P1", "Gap", "Mean duplicates"],
                build_p0_rows(
                    summary["evaluation"]["heldout_summary"], heldout_lane_names
                ),
            ),
            "",
            "## Runtime Cost",
            "",
            markdown_table(
                [
                    "Lane",
                    "Mean move latency",
                    "P95 move latency",
                    "Avg sims/game",
                    "Slowdown vs baseline",
                ],
                build_runtime_rows(
                    summary["evaluation"]["heldout_summary"], heldout_lane_names
                ),
            ),
            "",
            "## Gate Classification",
            "",
            *(gate_lines or ["- baseline-only gate run was not required"]),
            "",
            "## Decision Summary",
            "",
            f"- Budget-lane c_puct used after medium c_puct pass: `{fmt(summary['selection']['budget_lane_cpuct'], digits=2)}`",
            f"- Fixed-large carry lanes: `{','.join(summary['evaluation']['fixed_large_lane_names'])}`",
            f"- Held-out carry lanes: `{','.join(summary['evaluation']['heldout_lane_names'])}`",
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
    cpuct_values = parse_csv_floats(args.cpuct_values)
    requested_budget_lanes = parse_budget_pairs(args.budget_lanes)
    cpuct_lanes = build_cpuct_lanes(cpuct_values, args.tactical_root_bias)
    lane_map: dict[str, LaneConfig] = {lane.name: lane for lane in cpuct_lanes}
    baseline_lane = lane_map["no_tactical_cpuct_1_25_ref"]

    medium_cpuct_results = evaluate_suite(
        suite_name="medium",
        suite_path=medium_suite,
        current_path=current_path,
        lanes=cpuct_lanes,
        workdir=workdir,
        games_per_opening=args.games_per_opening,
        workers=args.workers,
        seed=args.seed,
        timeout=args.timeout,
    )
    budget_lane_cpuct = select_budget_eval_cpuct(
        cpuct_lanes,
        medium_cpuct_results,
        baseline_lane,
    )
    budget_lanes = build_budget_lanes(
        requested_budget_lanes,
        base_c_puct=budget_lane_cpuct,
        tactical_root_bias=args.tactical_root_bias,
    )
    for lane in budget_lanes:
        lane_map[lane.name] = lane
    medium_budget_results = evaluate_suite(
        suite_name="medium",
        suite_path=medium_suite,
        current_path=current_path,
        lanes=budget_lanes,
        workdir=workdir,
        games_per_opening=args.games_per_opening,
        workers=args.workers,
        seed=args.seed,
        timeout=args.timeout,
    )
    medium_results = {**medium_cpuct_results, **medium_budget_results}

    combined_lanes = build_combined_lanes(
        cpuct_lanes=cpuct_lanes,
        budget_lanes=budget_lanes,
        medium_results=medium_results,
        baseline_lane=baseline_lane,
    )
    for lane in combined_lanes:
        lane_map[lane.name] = lane
    if combined_lanes:
        medium_combined_results = evaluate_suite(
            suite_name="medium",
            suite_path=medium_suite,
            current_path=current_path,
            lanes=combined_lanes,
            workdir=workdir,
            games_per_opening=args.games_per_opening,
            workers=args.workers,
            seed=args.seed,
            timeout=args.timeout,
        )
        medium_results.update(medium_combined_results)

    all_medium_lanes = [*cpuct_lanes, *budget_lanes, *combined_lanes]
    fixed_large_lane_names = select_fixed_large_lanes(
        lanes=all_medium_lanes,
        medium_results=medium_results,
        baseline_lane=baseline_lane,
    )
    fixed_large_lanes = [lane_map[name] for name in fixed_large_lane_names]
    fixed_large_results = evaluate_suite(
        suite_name="fixed_large",
        suite_path=fixed_large_suite,
        current_path=current_path,
        lanes=fixed_large_lanes,
        workdir=workdir,
        games_per_opening=args.games_per_opening,
        workers=args.workers,
        seed=args.seed,
        timeout=args.timeout,
    )
    heldout_lane_names = select_heldout_lanes(
        fixed_large_results=fixed_large_results,
        fixed_large_lanes=fixed_large_lanes,
        baseline_lane=baseline_lane,
    )
    heldout_lanes = [lane_map[name] for name in heldout_lane_names]
    heldout_results: dict[str, dict[str, dict[str, Any]]] = {}
    for suite_path in heldout_suites:
        heldout_results[suite_path.stem] = evaluate_suite(
            suite_name=suite_path.stem,
            suite_path=suite_path,
            current_path=current_path,
            lanes=heldout_lanes,
            workdir=workdir,
            games_per_opening=args.games_per_opening,
            workers=args.workers,
            seed=args.seed,
            timeout=args.timeout,
        )

    medium_summary = aggregate_stage_summary(
        suite_results=medium_results,
        lanes=all_medium_lanes,
        baseline_lane=baseline_lane,
    )
    fixed_large_summary = aggregate_stage_summary(
        suite_results=fixed_large_results,
        lanes=fixed_large_lanes,
        baseline_lane=baseline_lane,
    )
    heldout_summary = aggregate_heldout_summary(
        heldout_results=heldout_results,
        lanes=heldout_lanes,
        baseline_lane=baseline_lane,
    )
    bootstrap_rows = compute_bootstrap_rows(
        heldout_results=heldout_results,
        lanes=heldout_lanes,
        baseline_lane=baseline_lane,
        seed=args.seed,
    )

    gate_targets = [baseline_lane.name]
    for lane in heldout_lanes:
        if lane.name == baseline_lane.name:
            continue
        deltas = {
            axis: float(
                heldout_summary[lane.name][axis]["mean_delta_vs_baseline"] or 0.0
            )
            for axis in CORE_AXIS_ORDER
        }
        if robust_primary_candidate(deltas):
            gate_targets.append(lane.name)
    gate_results: list[dict[str, Any]] = []
    gate_reports: dict[str, dict[str, Any]] = {}
    for lane_name in gate_targets:
        lane = lane_map[lane_name]
        gate_path = workdir / "gate" / lane_name / "gate_report.json"
        report = run_gate_for_lane(
            lane=lane,
            current_path=current_path,
            out_path=gate_path,
            seed=args.seed,
            workers=args.workers,
            games=args.gate_games,
        )
        gate_reports[lane_name] = report
        gate_results.append(
            {
                "lane": lane_name,
                "classification": report.get("classification", "unknown"),
                "budget_pairs": gate_budget_pairs_for_lane(lane),
                "budget_results": gate_budget_results(report),
            }
        )

    lane_decisions = {
        lane.name: lane_decision(
            lane=lane,
            heldout_summary=heldout_summary,
            bootstrap_rows=bootstrap_rows,
            gate_result=gate_reports.get(lane.name),
            baseline_lane=baseline_lane,
        )
        for lane in heldout_lanes
    }
    classification = classify_run(
        heldout_lanes=heldout_lanes,
        heldout_summary=heldout_summary,
        lane_decisions=lane_decisions,
        baseline_lane=baseline_lane,
    )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "date": str(date.today()),
        "classification": classification,
        "inputs": {
            "current": current_summary,
            "medium_suite": build_input_summary(medium_suite),
            "fixed_large_suite": build_input_summary(fixed_large_suite),
            "heldout_suites": [build_input_summary(path) for path in heldout_suites],
            "requested_cpuct_values": [f"{value:g}" for value in cpuct_values],
            "requested_budget_lanes": [
                budget_pair_label(*pair) for pair in requested_budget_lanes
            ],
            "tactical_root_bias": float(args.tactical_root_bias),
            "games_per_opening": int(args.games_per_opening),
            "workers": int(args.workers),
            "seed": int(args.seed),
        },
        "selection": {
            "budget_lane_cpuct": float(budget_lane_cpuct),
        },
        "lanes": [
            lane_to_summary_row(lane)
            for lane in [
                baseline_lane,
                *[lane for lane in all_medium_lanes if lane.name != baseline_lane.name],
            ]
        ],
        "evaluation": {
            "medium_results": medium_results,
            "fixed_large_results": fixed_large_results,
            "heldout_results": heldout_results,
            "medium_summary": medium_summary,
            "fixed_large_summary": fixed_large_summary,
            "heldout_summary": heldout_summary,
            "medium_lane_names": [lane.name for lane in all_medium_lanes],
            "fixed_large_lane_names": fixed_large_lane_names,
            "heldout_lane_names": heldout_lane_names,
            "bootstrap_cis": bootstrap_rows,
            "lane_decisions": lane_decisions,
        },
        "gate_results": gate_results,
    }
    lane_objects_for_report = [
        baseline_lane,
        *[lane for lane in all_medium_lanes if lane.name != baseline_lane.name],
    ]
    summary["_lane_objects"] = lane_objects_for_report
    summary_path = workdir / SUMMARY_FILENAME
    write_json(
        summary_path,
        {key: value for key, value in summary.items() if key != "_lane_objects"},
    )
    REPORT_PATH.write_text(render_report(summary) + "\n", encoding="utf-8")
    print(f"classification={classification}")
    print(f"summary={summary_path}")
    print(f"report={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
