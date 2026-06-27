#!/usr/bin/env python3
"""Balanced-current deterministic search-profile calibration ablation."""

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


SUMMARY_SCHEMA = "azlite_balanced_current_search_calibration_v1"
SUMMARY_FILENAME = "summary_metrics.json"
REPORT_PATH = (
    REPO_ROOT / "docs/alphazero-lite-balanced-current-search-calibration-results.md"
)
DEFAULT_BUDGET_PAIRS = "384:256,768:256,768:768,1200:1200,1200:256,256:768"
DEFAULT_GATE_BUDGET_PAIRS = "384:256,1200:1200,1200:256,256:768"
DEFAULT_BOOTSTRAP_BUDGETS = ("384:256", "768:768", "1200:1200", "1200:256")
DEFAULT_ROOT_TRANSFORM = (
    "seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5"
)
DEFAULT_C_PUCT = 1.25
DEFAULT_TACTICAL_ROOT_BIAS = 0.1


def _python() -> str:
    candidate = REPO_ROOT / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


@dataclass(frozen=True)
class LaneConfig:
    name: str
    description: str
    supported: bool
    unsupported_reason: str | None
    c_puct: float = DEFAULT_C_PUCT
    tactical_root_bias: float = DEFAULT_TACTICAL_ROOT_BIAS
    value_trust_schedule: dict[str, bool | float] | None = None
    challenger_root_prior_transform: str | None = None
    current_root_prior_transform: str | None = None

    def search_profile(self) -> dict[str, Any]:
        profile: dict[str, Any] = {
            "c_puct": float(self.c_puct),
            "tactical_root_bias": float(self.tactical_root_bias),
            "root_policy_mode": "deterministic",
            "search_mode": "full",
        }
        if self.value_trust_schedule is not None:
            profile["value_trust_schedule"] = dict(self.value_trust_schedule)
        if self.challenger_root_prior_transform is not None:
            profile["challenger_root_prior_transform"] = str(
                self.challenger_root_prior_transform
            )
        if self.current_root_prior_transform is not None:
            profile["current_root_prior_transform"] = str(
                self.current_root_prior_transform
            )
        return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="/tmp/azlite_balanced_current_search_calibration"
    )
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument(
        "--search-modes", default="full,policy_only,value_only,classic_only"
    )
    parser.add_argument("--cpuct-values", default="0.75,1.0,1.25,1.5,2.0")
    parser.add_argument(
        "--root-prior-transforms",
        default=f"none,{DEFAULT_ROOT_TRANSFORM}",
    )
    parser.add_argument("--budget-pairs", default=DEFAULT_BUDGET_PAIRS)
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--gate-games", type=int, default=60)
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


def inspect_supported_controls() -> dict[str, Any]:
    return {
        "search_modes": {
            "requested": ["classic_only", "policy_only", "value_only", "full"],
            "supported_in_active_arena_cli": ["full"],
            "unsupported": ["classic_only", "policy_only", "value_only"],
            "notes": (
                "arena.evaluate_artifact_position has an ablation_mode helper, but the "
                "active arena CLI and game loop do not expose or pass ablation_mode, so "
                "opening-suite evaluation cannot run search-mode lanes beyond full."
            ),
        },
        "c_puct": {
            "supported": True,
            "scope": "global_only",
            "notes": "arena CLI exposes a single --c-puct applied to both challenger and current.",
        },
        "root_prior_transform": {
            "supported": True,
            "scope": "challenger_only_current_only_or_both",
            "notes": (
                "arena CLI exposes --root-prior-transform, --challenger-root-prior-transform, "
                "and --current-root-prior-transform. In the active path, the generic transform "
                "falls back only to challenger unless current-specific is also provided."
            ),
        },
        "value_trust": {
            "supported": True,
            "scope": "global_only",
            "active_non_stub_path": True,
            "notes": (
                "arena CLI passes value-trust schedule into live PUCT search and emits "
                "value_trust_summary telemetry when configured. Multipliers must be > 0."
            ),
        },
        "tactical_root_bias": {
            "supported": True,
            "scope": "global_only",
            "notes": "arena CLI exposes a single --tactical-root-bias applied to both sides.",
        },
        "search_profile_hash": {
            "arena_report_support": True,
            "benchmark_metrics_support": False,
            "notes": (
                "arena report notes include search_profile and search_profile_hash, but "
                "run_opening_suite_seat_benchmark metrics.json does not preserve them."
            ),
        },
        "gate_custom_search_profile": {
            "supported": True,
            "notes": (
                "script/ai/seat_aware_promotion_gate can pass deterministic search controls through to arena."
            ),
        },
    }


def lane_catalog(
    *,
    requested_search_modes: set[str],
    requested_cpuct_values: set[float],
    requested_root_transforms: set[str],
    controls: dict[str, Any],
) -> list[LaneConfig]:
    supports_only_full = set(controls["search_modes"]["supported_in_active_arena_cli"])
    lanes = [
        LaneConfig(
            name="default_full_ref",
            description="Current default deterministic PUCT settings.",
            supported=True,
            unsupported_reason=None,
        )
    ]
    for mode_name in (
        "policy_only_default",
        "value_only_default",
        "classic_only_default",
    ):
        requested = mode_name.replace("_default", "")
        lanes.append(
            LaneConfig(
                name=mode_name,
                description=f"Requested {requested} search lane.",
                supported=requested in requested_search_modes
                and requested in supports_only_full,
                unsupported_reason=None
                if requested in requested_search_modes
                and requested in supports_only_full
                else "active arena CLI does not expose ablation search modes beyond full",
            )
        )
    lanes.extend(
        [
            LaneConfig(
                name="full_default_no_tactical_bias",
                description="Full search with tactical_root_bias set to 0.0.",
                supported=True,
                unsupported_reason=None,
                tactical_root_bias=0.0,
            ),
            LaneConfig(
                name="full_value_trust_half",
                description="Full search with value-trust multiplier 0.5 in all phases.",
                supported=True,
                unsupported_reason=None,
                value_trust_schedule={
                    "enabled": True,
                    "opening": 0.5,
                    "midgame": 0.5,
                    "late": 0.5,
                },
            ),
            LaneConfig(
                name="full_value_trust_zero",
                description="Requested full search with value disabled or neutralized.",
                supported=False,
                unsupported_reason=(
                    "active arena CLI does not expose value disable / neutral value mode, and "
                    "value-trust multipliers must be finite numbers > 0"
                ),
            ),
        ]
    )
    for value, name in (
        (0.75, "full_cpuct_0_75"),
        (1.0, "full_cpuct_1_00"),
        (1.5, "full_cpuct_1_50"),
        (2.0, "full_cpuct_2_00"),
    ):
        lanes.append(
            LaneConfig(
                name=name,
                description=f"Full search with global c_puct={value:.2f}.",
                supported=value in requested_cpuct_values,
                unsupported_reason=None
                if value in requested_cpuct_values
                else "not requested by --cpuct-values",
                c_puct=value,
            )
        )
    transform_requested = DEFAULT_ROOT_TRANSFORM in requested_root_transforms
    lanes.append(
        LaneConfig(
            name="full_seed4_extra_turn_damp",
            description=(
                "Full search with the arena-supported seed4 extra-turn damp transform applied to both sides."
            ),
            supported=transform_requested,
            unsupported_reason=None
            if transform_requested
            else "not requested by --root-prior-transforms",
            challenger_root_prior_transform=DEFAULT_ROOT_TRANSFORM,
            current_root_prior_transform=DEFAULT_ROOT_TRANSFORM,
        )
    )
    return lanes


def lane_by_name(lanes: list[LaneConfig]) -> dict[str, LaneConfig]:
    return {lane.name: lane for lane in lanes}


def write_suite_prefixes(path: Path, suite_entries: list[dict[str, Any]]) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in suite_entries:
            handle.write(json.dumps({"prefix_moves": entry["prefix_moves"]}) + "\n")


def run_arena_for_lane(
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
    if lane.value_trust_schedule is not None:
        cmd.append("--value-trust-enabled")
        cmd.extend(
            [
                "--value-trust-opening",
                str(lane.value_trust_schedule["opening"]),
                "--value-trust-midgame",
                str(lane.value_trust_schedule["midgame"]),
                "--value-trust-late",
                str(lane.value_trust_schedule["late"]),
            ]
        )
    if lane.challenger_root_prior_transform is not None:
        cmd.extend(
            [
                "--challenger-root-prior-transform",
                str(lane.challenger_root_prior_transform),
            ]
        )
    if lane.current_root_prior_transform is not None:
        cmd.extend(
            [
                "--current-root-prior-transform",
                str(lane.current_root_prior_transform),
            ]
        )
    print(
        f"[arena] {suite_name} {lane.name} {label} sims={challenger_sims}:{current_sims}",
        flush=True,
    )
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
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
        "root_prior_transform": notes.get("root_prior_transform"),
        "challenger_root_prior_transform": notes.get("challenger_root_prior_transform"),
        "current_root_prior_transform": notes.get("current_root_prior_transform"),
        "root_prior_telemetry": arena_report.get("root_prior_telemetry"),
        "value_trust_summary": arena_report.get("value_trust_summary"),
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
    budget_pairs: list[tuple[int, int]],
    workdir: Path,
    games_per_opening: int,
    workers: int,
    seed: int,
    timeout: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    suite_entries = load_suite(str(suite_path))
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for lane in lanes:
        if not lane.supported:
            continue
        results[lane.name] = {}
        for pair in budget_pairs:
            row = run_arena_for_lane(
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
            results[lane.name][budget_pair_label(*pair)] = row
    return results


def medium_deltas(
    medium_results: dict[str, dict[str, dict[str, Any]]], lane_name: str
) -> dict[str, float]:
    default_results = medium_results["default_full_ref"]
    lane_results = medium_results[lane_name]
    deltas: dict[str, float] = {}
    for budget_pair, row in lane_results.items():
        deltas[budget_pair] = float(row["ds"]) - float(
            default_results[budget_pair]["ds"]
        )
    return deltas


def balanced_score_from_deltas(deltas: dict[str, float]) -> float:
    return (
        float(deltas.get("384:256", 0.0))
        + 0.5 * float(deltas.get("768:768", 0.0))
        + 0.25 * float(deltas.get("1200:1200", 0.0))
    )


def select_fixed_large_lanes(
    lanes: list[LaneConfig],
    medium_results: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    supported_names = [lane.name for lane in lanes if lane.supported]
    candidate_names = [name for name in supported_names if name != "default_full_ref"]
    scored: list[tuple[float, str]] = []
    mandatory: set[str] = set()
    for name in candidate_names:
        deltas = medium_deltas(medium_results, name)
        if deltas.get("384:256", 0.0) >= 0.05 and deltas.get("768:768", 0.0) >= -0.08:
            mandatory.add(name)
        if deltas.get("768:768", 0.0) >= 0.05 and deltas.get("384:256", 0.0) >= -0.08:
            mandatory.add(name)
        scored.append((balanced_score_from_deltas(deltas), name))
    scored.sort(reverse=True)
    selected_non_ref: list[str] = []
    for _score, name in scored:
        if name in mandatory and name not in selected_non_ref:
            selected_non_ref.append(name)
    for _score, name in scored:
        if name not in selected_non_ref:
            selected_non_ref.append(name)
        if len(selected_non_ref) >= 6:
            break
    selected_non_ref = selected_non_ref[:6]
    return ["default_full_ref", *selected_non_ref]


def aggregate_suite_budget_rows(
    *,
    suite_results: dict[str, dict[str, dict[str, Any]]],
    lane_names: list[str],
    reference_name: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    budget_pairs = next(iter(suite_results.values())).keys() if suite_results else []
    for lane_name in lane_names:
        lane_budgets = suite_results[lane_name]
        row: dict[str, Any] = {}
        for budget_pair in budget_pairs:
            ds = float(lane_budgets[budget_pair]["ds"])
            ref_ds = float(suite_results[reference_name][budget_pair]["ds"])
            row[budget_pair] = {
                "ds": ds,
                "delta_vs_default": ds - ref_ds,
                "p0_score": float(lane_budgets[budget_pair]["p0_score"]),
                "p1_score": float(lane_budgets[budget_pair]["p1_score"]),
                "duplicate_trajectory_count": int(
                    lane_budgets[budget_pair]["duplicate_trajectory_count"]
                ),
                "move_time_mean_ms": lane_budgets[budget_pair].get("move_time_mean_ms"),
                "move_time_p95_ms": lane_budgets[budget_pair].get("move_time_p95_ms"),
                "mean_final_simulations": lane_budgets[budget_pair].get(
                    "mean_final_simulations"
                ),
            }
        summary[lane_name] = row
    return summary


def aggregate_heldout_summary(
    *,
    heldout_results: dict[str, dict[str, dict[str, dict[str, Any]]]],
    lane_names: list[str],
    reference_name: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    budget_pairs = (
        next(iter(next(iter(heldout_results.values())).values())).keys()
        if heldout_results
        else []
    )
    for lane_name in lane_names:
        lane_row: dict[str, Any] = {
            "suite_rows": {},
            "runtime_cost": {},
        }
        for budget_pair in budget_pairs:
            suite_ds: list[float] = []
            suite_deltas: list[float] = []
            p0_scores: list[float] = []
            p1_scores: list[float] = []
            duplicates: list[float] = []
            move_means: list[float] = []
            move_p95s: list[float] = []
            sims: list[float] = []
            worst_suite_name: str | None = None
            worst_suite_ds: float | None = None
            for suite_name, suite_row in heldout_results.items():
                candidate_budget = suite_row[lane_name][budget_pair]
                ref_budget = suite_row[reference_name][budget_pair]
                ds = float(candidate_budget["ds"])
                delta = ds - float(ref_budget["ds"])
                lane_row["suite_rows"].setdefault(suite_name, {})[budget_pair] = {
                    "ds": ds,
                    "delta_vs_default": delta,
                }
                suite_ds.append(ds)
                suite_deltas.append(delta)
                p0_scores.append(float(candidate_budget["p0_score"]))
                p1_scores.append(float(candidate_budget["p1_score"]))
                duplicates.append(float(candidate_budget["duplicate_trajectory_count"]))
                if candidate_budget.get("move_time_mean_ms") is not None:
                    move_means.append(float(candidate_budget["move_time_mean_ms"]))
                if candidate_budget.get("move_time_p95_ms") is not None:
                    move_p95s.append(float(candidate_budget["move_time_p95_ms"]))
                if candidate_budget.get("mean_final_simulations") is not None:
                    sims.append(float(candidate_budget["mean_final_simulations"]))
                if worst_suite_ds is None or ds < worst_suite_ds:
                    worst_suite_ds = ds
                    worst_suite_name = suite_name
            lane_row[budget_pair] = {
                "mean_ds": statistics.fmean(suite_ds) if suite_ds else None,
                "mean_delta_vs_default": statistics.fmean(suite_deltas)
                if suite_deltas
                else None,
                "worst_suite_ds": worst_suite_ds,
                "worst_suite_name": worst_suite_name,
                "mean_p0_score": statistics.fmean(p0_scores) if p0_scores else None,
                "mean_p1_score": statistics.fmean(p1_scores) if p1_scores else None,
                "mean_duplicate_trajectory_count": statistics.fmean(duplicates)
                if duplicates
                else None,
            }
            lane_row["runtime_cost"][budget_pair] = {
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
        summary[lane_name] = lane_row
    return summary


def pooled_per_opening_differences(
    *,
    suite_results: dict[str, dict[str, dict[str, dict[str, Any]]]],
    lane_name: str,
    reference_name: str,
    budget_pair: str,
) -> list[float]:
    pooled: list[float] = []
    for suite_row in suite_results.values():
        lane_metrics = suite_row[lane_name][budget_pair].get("per_opening_metrics", [])
        ref_metrics = suite_row[reference_name][budget_pair].get(
            "per_opening_metrics", []
        )
        lane_by_opening = {row["opening_prefix"]: row for row in lane_metrics}
        ref_by_opening = {row["opening_prefix"]: row for row in ref_metrics}
        common = sorted(set(lane_by_opening) & set(ref_by_opening))
        for opening_prefix in common:
            pooled.append(
                float(lane_by_opening[opening_prefix]["ds"])
                - float(ref_by_opening[opening_prefix]["ds"])
            )
    return pooled


def compute_bootstrap_rows(
    *,
    suite_results: dict[str, dict[str, dict[str, dict[str, Any]]]],
    lane_names: list[str],
    reference_name: str,
    seed: int,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for lane_name in lane_names:
        lane_rows: dict[str, Any] = {}
        for budget_pair in DEFAULT_BOOTSTRAP_BUDGETS:
            if lane_name == reference_name:
                diffs = [0.0]
            else:
                diffs = pooled_per_opening_differences(
                    suite_results=suite_results,
                    lane_name=lane_name,
                    reference_name=reference_name,
                    budget_pair=budget_pair,
                )
            lane_rows[budget_pair] = bootstrap_ci(diffs, seed=seed)
        rows[lane_name] = lane_rows
    return rows


def run_gate_for_lane(
    *,
    lane: LaneConfig,
    current_path: Path,
    out_path: Path,
    seed: int,
    workers: int,
    games: int,
    budget_pairs: str,
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
        budget_pairs,
        "--c-puct",
        str(lane.c_puct),
        "--root-policy-mode",
        "deterministic",
        "--root-temperature",
        "0.0",
        "--tactical-root-bias",
        str(lane.tactical_root_bias),
    ]
    if lane.value_trust_schedule is not None:
        cmd.append("--value-trust-enabled")
        cmd.extend(
            [
                "--value-trust-opening",
                str(lane.value_trust_schedule["opening"]),
                "--value-trust-midgame",
                str(lane.value_trust_schedule["midgame"]),
                "--value-trust-late",
                str(lane.value_trust_schedule["late"]),
            ]
        )
    if lane.challenger_root_prior_transform is not None:
        cmd.extend(
            [
                "--challenger-root-prior-transform",
                str(lane.challenger_root_prior_transform),
            ]
        )
    if lane.current_root_prior_transform is not None:
        cmd.extend(
            [
                "--current-root-prior-transform",
                str(lane.current_root_prior_transform),
            ]
        )
    print(f"[gate] {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=14400,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gate failed: {result.stderr[-2000:]}")
    return json.loads(out_path.read_text(encoding="utf-8"))


def select_heldout_lanes(
    *,
    fixed_large_results: dict[str, dict[str, dict[str, Any]]],
    lane_names: list[str],
) -> list[str]:
    scored: list[tuple[float, str]] = []
    beat_default_384: set[str] = set()
    default_budgets = fixed_large_results["default_full_ref"]
    for lane_name in lane_names:
        if lane_name == "default_full_ref":
            continue
        deltas = {
            budget_pair: float(row["ds"]) - float(default_budgets[budget_pair]["ds"])
            for budget_pair, row in fixed_large_results[lane_name].items()
        }
        scored.append((balanced_score_from_deltas(deltas), lane_name))
        if deltas.get("384:256", 0.0) >= 0.10:
            beat_default_384.add(lane_name)
    scored.sort(reverse=True)
    selected = ["default_full_ref"]
    for _score, lane_name in scored[:3]:
        if lane_name not in selected:
            selected.append(lane_name)
    for lane_name in sorted(beat_default_384):
        if lane_name not in selected:
            selected.append(lane_name)
    return selected


def lane_decision(
    *,
    lane_name: str,
    heldout_summary: dict[str, Any],
    bootstrap_rows: dict[str, dict[str, Any]],
    gate_result: dict[str, Any] | None,
    gate_supported: bool,
) -> dict[str, Any]:
    budgets = heldout_summary[lane_name]
    delta_384 = float(budgets["384:256"]["mean_delta_vs_default"] or 0.0)
    delta_768 = float(budgets["768:768"]["mean_delta_vs_default"] or 0.0)
    delta_1200 = float(budgets["1200:1200"]["mean_delta_vs_default"] or 0.0)
    delta_1200_256 = float(budgets["1200:256"]["mean_delta_vs_default"] or 0.0)
    ci_384 = bootstrap_rows[lane_name]["384:256"]
    gate_classification = gate_result.get("classification") if gate_result else None
    gate_ok = gate_supported and gate_classification == "high_search_breakthrough"
    search_profile_improvement = (
        delta_384 >= 0.05
        and float(ci_384["lower"]) > 0.01
        and delta_768 >= -0.08
        and delta_1200 >= -0.03
        and delta_1200_256 >= -0.03
        and gate_ok
    )
    equal_budget_search_fix = (
        delta_768 >= 0.08
        and delta_384 >= -0.05
        and delta_1200 >= -0.03
        and delta_1200_256 >= -0.03
    )
    return {
        "search_profile_improvement": search_profile_improvement,
        "equal_budget_search_fix": equal_budget_search_fix,
        "gate_supported": gate_supported,
        "gate_classification": gate_classification,
    }


def classify_run(
    *,
    evaluated_lane_names: list[str],
    heldout_summary: dict[str, Any],
    lane_decisions: dict[str, dict[str, Any]],
) -> str:
    candidates = [name for name in evaluated_lane_names if name != "default_full_ref"]
    gated_winners = [
        name
        for name in candidates
        if lane_decisions[name]["gate_supported"]
        and lane_decisions[name]["gate_classification"] == "high_search_breakthrough"
        and (
            lane_decisions[name]["search_profile_improvement"]
            or lane_decisions[name]["equal_budget_search_fix"]
        )
    ]
    if gated_winners:
        winner = gated_winners[0]
        runtime = heldout_summary[winner]["runtime_cost"]
        default_runtime = heldout_summary["default_full_ref"]["runtime_cost"]
        runtime_not_materially_worse = True
        for budget_pair in ("384:256", "768:768", "1200:1200", "1200:256"):
            cand_mean = runtime[budget_pair].get("mean_move_latency_ms")
            ref_mean = default_runtime[budget_pair].get("mean_move_latency_ms")
            if cand_mean is None or ref_mean is None or ref_mean <= 0:
                continue
            if float(cand_mean) > (float(ref_mean) * 1.2):
                runtime_not_materially_worse = False
                break
        if runtime_not_materially_worse:
            return "runtime_config_candidate"
        return "search_profile_improvement"
    ungated_or_nonpreserving = [
        name
        for name in candidates
        if lane_decisions[name]["search_profile_improvement"]
        or lane_decisions[name]["equal_budget_search_fix"]
    ]
    if ungated_or_nonpreserving:
        return "search_profile_improvement"
    low_cpuct = [
        name for name in candidates if name in {"full_cpuct_0_75", "full_cpuct_1_00"}
    ]
    high_cpuct = [
        name for name in candidates if name in {"full_cpuct_1_50", "full_cpuct_2_00"}
    ]
    if low_cpuct and high_cpuct:
        low_good = any(
            float(heldout_summary[name]["384:256"]["mean_delta_vs_default"] or 0.0)
            >= 0.05
            or float(heldout_summary[name]["768:768"]["mean_delta_vs_default"] or 0.0)
            >= 0.08
            for name in low_cpuct
        )
        high_bad = any(
            float(heldout_summary[name]["384:256"]["mean_delta_vs_default"] or 0.0)
            < 0.0
            and float(heldout_summary[name]["768:768"]["mean_delta_vs_default"] or 0.0)
            <= 0.0
            for name in high_cpuct
        )
        if low_good and high_bad:
            return "policy_prior_overtrusted"
    if "full_value_trust_half" in candidates:
        vt_delta_384 = float(
            heldout_summary["full_value_trust_half"]["384:256"]["mean_delta_vs_default"]
            or 0.0
        )
        vt_delta_768 = float(
            heldout_summary["full_value_trust_half"]["768:768"]["mean_delta_vs_default"]
            or 0.0
        )
        if vt_delta_384 >= 0.05 or vt_delta_768 >= 0.08:
            return "value_head_overtrusted"
    return "search_knobs_not_enough"


def build_supported_controls_rows(controls: dict[str, Any]) -> list[list[Any]]:
    return [
        [
            "search modes",
            ", ".join(controls["search_modes"]["supported_in_active_arena_cli"]),
            ", ".join(controls["search_modes"]["unsupported"]),
            controls["search_modes"]["notes"],
        ],
        ["c_puct", "yes", controls["c_puct"]["scope"], controls["c_puct"]["notes"]],
        [
            "root-prior transform",
            "yes",
            controls["root_prior_transform"]["scope"],
            controls["root_prior_transform"]["notes"],
        ],
        [
            "value trust",
            "yes",
            controls["value_trust"]["scope"],
            controls["value_trust"]["notes"],
        ],
        [
            "tactical_root_bias",
            "yes",
            controls["tactical_root_bias"]["scope"],
            controls["tactical_root_bias"]["notes"],
        ],
        [
            "search_profile_hash",
            "arena yes / benchmark no",
            "reporting",
            controls["search_profile_hash"]["notes"],
        ],
        [
            "custom gate search profile",
            "yes",
            "supported",
            controls["gate_custom_search_profile"]["notes"],
        ],
    ]


def build_lane_rows(lanes: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for lane in lanes:
        rows.append(
            [
                lane["name"],
                str(lane["supported"]),
                json.dumps(lane["search_profile"], sort_keys=True),
                lane.get("unsupported_reason") or "",
            ]
        )
    return rows


def build_fixed_large_rows(
    fixed_large_summary: dict[str, Any], lane_names: list[str]
) -> list[list[Any]]:
    rows = []
    for lane_name in lane_names:
        row = fixed_large_summary[lane_name]
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


def build_search_profile_hash_rows(
    stage_results: dict[str, dict[str, dict[str, Any]]], lane_names: list[str]
) -> list[list[Any]]:
    rows = []
    for lane_name in lane_names:
        for budget_pair, row in stage_results[lane_name].items():
            rows.append(
                [lane_name, budget_pair, row.get("search_profile_hash") or "n/a"]
            )
    return rows


def build_value_trust_rows(
    stage_results: dict[str, dict[str, dict[str, Any]]], lane_names: list[str]
) -> list[list[Any]]:
    rows = []
    for lane_name in lane_names:
        for budget_pair, row in stage_results[lane_name].items():
            summary = row.get("value_trust_summary")
            if isinstance(summary, dict):
                rows.append(
                    [lane_name, budget_pair, json.dumps(summary, sort_keys=True)]
                )
                break
    return rows


def build_root_transform_rows(
    stage_results: dict[str, dict[str, dict[str, Any]]], lane_names: list[str]
) -> list[list[Any]]:
    rows = []
    for lane_name in lane_names:
        for budget_pair, row in stage_results[lane_name].items():
            telemetry = row.get("root_prior_telemetry")
            challenger_transform = row.get("challenger_root_prior_transform")
            current_transform = row.get("current_root_prior_transform")
            if challenger_transform or current_transform:
                rows.append(
                    [
                        lane_name,
                        budget_pair,
                        challenger_transform or "n/a",
                        current_transform or "n/a",
                        json.dumps(telemetry or {}, sort_keys=True),
                    ]
                )
                break
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
                fmt(row["1200:1200"]["mean_ds"]),
            ]
        )
    return rows


def build_bootstrap_rows(
    bootstrap_rows: dict[str, dict[str, Any]], lane_names: list[str]
) -> list[list[Any]]:
    rows = []
    for lane_name in lane_names:
        for budget_pair in DEFAULT_BOOTSTRAP_BUDGETS:
            ci = bootstrap_rows[lane_name][budget_pair]
            rows.append(
                [
                    f"{lane_name}_minus_default_{budget_pair.replace(':', '_')}",
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
        rows.append(
            [
                lane_name,
                fmt(row["mean_p0_score"]),
                fmt(row["mean_p1_score"]),
                fmt(
                    float(row["mean_p1_score"] or 0.0)
                    - float(row["mean_p0_score"] or 0.0)
                ),
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
            ]
        )
    return rows


def render_report(summary: dict[str, Any]) -> str:
    inputs = summary["inputs"]
    lane_names = summary["evaluation"]["heldout_lane_names"]
    supported_lane_names = [
        lane["name"] for lane in summary["lanes"] if bool(lane["supported"])
    ]
    value_trust_rows = build_value_trust_rows(
        summary["evaluation"]["medium_results"], supported_lane_names
    )
    root_transform_rows = build_root_transform_rows(
        summary["evaluation"]["medium_results"], supported_lane_names
    )
    gate_lines = [
        f"- {row['lane']}: `{row['classification']}`"
        + (
            " (custom search-profile gate unsupported)"
            if row.get("unsupported_custom_profile_gate")
            else ""
        )
        for row in summary["gate_results"]
    ]
    heldout_suite_lines = [f"- `{row['path']}`" for row in inputs["heldout_suites"]]
    return "\n".join(
        [
            "# AlphaZero-Lite Balanced Current Search Calibration Results",
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
            *heldout_suite_lines,
            f"- Budget pairs: `{','.join(inputs['budget_pairs'])}`",
            f"- Games per opening: `{inputs['games_per_opening']}`",
            f"- Workers: `{inputs['workers']}`",
            f"- Seed: `{inputs['seed']}`",
            "",
            "## Supported Controls",
            "",
            markdown_table(
                ["Control", "Supported", "Scope", "Notes"],
                build_supported_controls_rows(summary["supported_controls"]),
            ),
            "",
            "## Lane Search Profiles",
            "",
            markdown_table(
                ["Lane", "Supported", "Search profile", "Unsupported reason"],
                build_lane_rows(summary["lanes"]),
            ),
            "",
            "## Search Profile Hashes",
            "",
            "Medium-suite arena hashes by lane and budget pair.",
            "",
            markdown_table(
                ["Lane", "Budget", "search_profile_hash"],
                build_search_profile_hash_rows(
                    summary["evaluation"]["medium_results"], supported_lane_names
                ),
            ),
            "",
            "## Value-Trust Telemetry",
            "",
            markdown_table(
                ["Lane", "Budget", "Telemetry"],
                value_trust_rows or [["none", "n/a", "n/a"]],
            ),
            "",
            "## Root-Prior Transform Telemetry",
            "",
            markdown_table(
                [
                    "Lane",
                    "Budget",
                    "Challenger transform",
                    "Current transform",
                    "Telemetry",
                ],
                root_transform_rows or [["none", "n/a", "n/a", "n/a", "n/a"]],
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
                build_fixed_large_rows(
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
                    "Mean 384:256",
                    "Worst 384:256",
                    "Mean 768:768",
                    "Mean 1200:1200",
                ],
                build_heldout_rows(
                    summary["evaluation"]["heldout_summary"], lane_names
                ),
            ),
            "",
            "## Bootstrap CI",
            "",
            markdown_table(
                ["Comparison", "Mean", "Lower 95%", "Upper 95%"],
                build_bootstrap_rows(
                    summary["evaluation"]["bootstrap_cis"], lane_names
                ),
            ),
            "",
            "## P0/P1 Split For 384:256",
            "",
            markdown_table(
                ["Lane", "Mean P0", "Mean P1", "Gap", "Mean duplicates"],
                build_p0_rows(summary["evaluation"]["heldout_summary"], lane_names),
            ),
            "",
            "## Runtime Cost",
            "",
            "Held-out 384:256 aggregate runtime cost.",
            "",
            markdown_table(
                ["Lane", "Mean move latency", "P95 move latency", "Avg sims/game"],
                build_runtime_rows(
                    summary["evaluation"]["heldout_summary"], lane_names
                ),
            ),
            "",
            "## Gate Classification",
            "",
            *gate_lines,
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
    controls = inspect_supported_controls()
    lanes = lane_catalog(
        requested_search_modes=set(
            item.strip() for item in args.search_modes.split(",") if item.strip()
        ),
        requested_cpuct_values=set(parse_csv_floats(args.cpuct_values)),
        requested_root_transforms=set(
            item.strip()
            for item in args.root_prior_transforms.split(",")
            if item.strip()
        ),
        controls=controls,
    )
    supported_lanes = [lane for lane in lanes if lane.supported]
    budget_pairs = parse_budget_pairs(args.budget_pairs)

    medium_results = evaluate_suite(
        suite_name="medium",
        suite_path=medium_suite,
        current_path=current_path,
        lanes=supported_lanes,
        budget_pairs=budget_pairs,
        workdir=workdir,
        games_per_opening=args.games_per_opening,
        workers=args.workers,
        seed=args.seed,
        timeout=args.timeout,
    )
    fixed_large_lane_names = select_fixed_large_lanes(lanes, medium_results)
    fixed_large_lanes = [
        lane for lane in supported_lanes if lane.name in fixed_large_lane_names
    ]
    fixed_large_results = evaluate_suite(
        suite_name="fixed_large",
        suite_path=fixed_large_suite,
        current_path=current_path,
        lanes=fixed_large_lanes,
        budget_pairs=budget_pairs,
        workdir=workdir,
        games_per_opening=args.games_per_opening,
        workers=args.workers,
        seed=args.seed,
        timeout=args.timeout,
    )
    heldout_lane_names = select_heldout_lanes(
        fixed_large_results=fixed_large_results,
        lane_names=fixed_large_lane_names,
    )
    heldout_lanes = [
        lane for lane in supported_lanes if lane.name in heldout_lane_names
    ]
    heldout_results: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for suite_path in heldout_suites:
        heldout_results[suite_path.stem] = evaluate_suite(
            suite_name=suite_path.stem,
            suite_path=suite_path,
            current_path=current_path,
            lanes=heldout_lanes,
            budget_pairs=budget_pairs,
            workdir=workdir,
            games_per_opening=args.games_per_opening,
            workers=args.workers,
            seed=args.seed,
            timeout=args.timeout,
        )

    fixed_large_summary = aggregate_suite_budget_rows(
        suite_results=fixed_large_results,
        lane_names=fixed_large_lane_names,
        reference_name="default_full_ref",
    )
    heldout_summary = aggregate_heldout_summary(
        heldout_results=heldout_results,
        lane_names=heldout_lane_names,
        reference_name="default_full_ref",
    )
    bootstrap_rows = compute_bootstrap_rows(
        suite_results=heldout_results,
        lane_names=heldout_lane_names,
        reference_name="default_full_ref",
        seed=args.seed,
    )

    lane_map = lane_by_name(lanes)
    gate_targets = ["default_full_ref"]
    for lane_name in heldout_lane_names:
        if lane_name == "default_full_ref":
            continue
        budgets = heldout_summary[lane_name]
        if (
            float(budgets["384:256"]["mean_delta_vs_default"] or 0.0) >= 0.05
            and float(budgets["768:768"]["mean_delta_vs_default"] or 0.0) >= -0.08
            and float(budgets["1200:1200"]["mean_delta_vs_default"] or 0.0) >= -0.03
            and float(budgets["1200:256"]["mean_delta_vs_default"] or 0.0) >= -0.03
        ):
            gate_targets.append(lane_name)
    gate_results: list[dict[str, Any]] = []
    gate_reports: dict[str, dict[str, Any]] = {}
    for lane_name in gate_targets:
        gate_path = workdir / "gate" / lane_name / "gate_report.json"
        report = run_gate_for_lane(
            lane=lane_map[lane_name],
            current_path=current_path,
            out_path=gate_path,
            seed=args.seed,
            workers=args.workers,
            games=args.gate_games,
            budget_pairs=DEFAULT_GATE_BUDGET_PAIRS,
        )
        gate_reports[lane_name] = report
        gate_results.append(
            {
                "lane": lane_name,
                "classification": report.get("classification", "unknown"),
                "budget_results": gate_budget_results(report),
                "unsupported_custom_profile_gate": False,
            }
        )

    lane_decisions = {}
    for lane_name in heldout_lane_names:
        gate_report = gate_reports.get(lane_name)
        lane_decisions[lane_name] = lane_decision(
            lane_name=lane_name,
            heldout_summary=heldout_summary,
            bootstrap_rows=bootstrap_rows,
            gate_result=gate_report,
            gate_supported=gate_report is not None,
        )

    classification = classify_run(
        evaluated_lane_names=heldout_lane_names,
        heldout_summary=heldout_summary,
        lane_decisions=lane_decisions,
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
            "budget_pairs": [budget_pair_label(*pair) for pair in budget_pairs],
            "games_per_opening": int(args.games_per_opening),
            "workers": int(args.workers),
            "seed": int(args.seed),
        },
        "supported_controls": controls,
        "lanes": [
            {
                "name": lane.name,
                "description": lane.description,
                "supported": lane.supported,
                "unsupported_reason": lane.unsupported_reason,
                "search_profile": lane.search_profile(),
            }
            for lane in lanes
        ],
        "evaluation": {
            "medium_results": medium_results,
            "fixed_large_results": fixed_large_results,
            "heldout_results": heldout_results,
            "fixed_large_lane_names": fixed_large_lane_names,
            "heldout_lane_names": heldout_lane_names,
            "fixed_large_summary": fixed_large_summary,
            "heldout_summary": heldout_summary,
            "bootstrap_cis": bootstrap_rows,
            "lane_decisions": lane_decisions,
        },
        "gate_results": gate_results,
    }
    summary_path = workdir / SUMMARY_FILENAME
    write_json(summary_path, summary)
    REPORT_PATH.write_text(render_report(summary) + "\n", encoding="utf-8")
    print(f"classification={classification}")
    print(f"summary={summary_path}")
    print(f"report={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
