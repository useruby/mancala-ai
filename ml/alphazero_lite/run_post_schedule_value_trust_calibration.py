#!/usr/bin/env python3
"""Post-schedule value-trust calibration and ablation experiment."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position  # noqa: E402
from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    parse_cpuct_schedule_json,
    resolve_budget_cpuct,
    schedule_definition,
)
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_ci,
    gate_budget_results,
    pooled_per_opening_differences,
)
from ml.alphazero_lite.run_opening_suite_seat_benchmark import (  # noqa: E402
    compute_by_ply_metrics,
    compute_per_opening_metrics,
    compute_seat_metrics,
    parse_game_jsonl,
)
from ml.alphazero_lite.run_post_schedule_pairwise_puct_update import (  # noqa: E402
    fmt,
)
from ml.alphazero_lite.run_promoted_current_opening_puct_disagreement import (  # noqa: E402
    artifact_forward_details,
    build_input_summary,
    canonical_state_hash,
    raw_margin,
    read_jsonl,
    require_existing_file,
    sha256_file,
    top_policy_move,
    verify_expected_hash,
    write_json,
    write_jsonl,
)
from ml.alphazero_lite.self_play import (  # noqa: E402
    build_eval_search_options,
    build_search_profile,
)

SUMMARY_SCHEMA = "azlite_post_schedule_value_trust_calibration_v1"
SUMMARY_FILENAME = "summary_metrics.json"
VALUE_AUDIT_FILENAME = "value_calibration_audit.json"
VALUE_STATES_FILENAME = "value_calibration_states.jsonl"
REPORT_PATH = (
    REPO_ROOT / "docs/alphazero-lite-post-schedule-value-trust-calibration-results.md"
)
DEFAULT_BUDGET_PAIRS = "384:256,768:256,768:768,1200:1200,1200:256,256:768"
AUDIT_BUDGETS = ("384:256", "768:768", "1200:1200", "1200:256")
BOOTSTRAP_BUDGETS = ("384:256", "768:768", "1200:1200", "1200:256")
VALUE_BUCKETS = (
    (-1.0, -0.75),
    (-0.75, -0.5),
    (-0.5, -0.25),
    (-0.25, 0.0),
    (0.0, 0.25),
    (0.25, 0.5),
    (0.5, 0.75),
    (0.75, 1.000001),
)


@dataclass(frozen=True)
class LaneConfig:
    name: str
    description: str
    value_trust_schedule: dict[str, bool | float] | None = None

    @property
    def supported(self) -> bool:
        return True

    def search_profile(self) -> dict[str, Any]:
        profile: dict[str, Any] = {
            "c_puct": 1.25,
            "c_puct_schedule": {"768:768": 0.9},
            "root_policy_mode": "deterministic",
            "search_mode": "full",
            "root_prior_transform": None,
            "tactical_root_bias": 0.0,
        }
        if self.value_trust_schedule is not None:
            profile["value_trust_schedule"] = dict(self.value_trust_schedule)
        return profile


def _python() -> str:
    candidate = REPO_ROOT / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    table = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        table.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--default-c-puct", type=float, default=1.25)
    parser.add_argument("--cpuct-schedule", required=True)
    parser.add_argument("--tactical-root-bias", type=float, default=0.0)
    parser.add_argument(
        "--value-trust-lanes",
        default=(
            "all0.75,all0.50,all0.25,opening0.50,opening0.75,mid0.50,"
            "late0.50,all1.25,opening1.25"
        ),
    )
    parser.add_argument("--budget-pairs", default=DEFAULT_BUDGET_PAIRS)
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--gate-games", type=int, default=60)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--audit-target-states", type=int, default=4096)
    parser.add_argument("--audit-min-states", type=int, default=2000)
    return parser.parse_args()


def parse_budget_pairs(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in text.split(","):
        token = item.strip()
        if not token:
            continue
        left, right = token.split(":", 1)
        pairs.append((int(left), int(right)))
    if not pairs:
        raise ValueError("expected at least one budget pair")
    return pairs


def budget_pair_label(challenger: int, current: int) -> str:
    return f"{challenger}:{current}"


def lane_catalog(requested: str) -> list[LaneConfig]:
    supported = {
        "all0.75": LaneConfig(
            name="value_trust_all_0_75",
            description="Uniform value trust 0.75 in opening, midgame, and late.",
            value_trust_schedule={
                "enabled": True,
                "opening": 0.75,
                "midgame": 0.75,
                "late": 0.75,
            },
        ),
        "all0.50": LaneConfig(
            name="value_trust_all_0_50",
            description="Uniform value trust 0.50 in opening, midgame, and late.",
            value_trust_schedule={
                "enabled": True,
                "opening": 0.50,
                "midgame": 0.50,
                "late": 0.50,
            },
        ),
        "all0.25": LaneConfig(
            name="value_trust_all_0_25",
            description="Uniform value trust 0.25 in opening, midgame, and late.",
            value_trust_schedule={
                "enabled": True,
                "opening": 0.25,
                "midgame": 0.25,
                "late": 0.25,
            },
        ),
        "opening0.50": LaneConfig(
            name="value_trust_opening_0_50_mid_1_00_late_1_00",
            description="Opening-only trust reduction to 0.50.",
            value_trust_schedule={
                "enabled": True,
                "opening": 0.50,
                "midgame": 1.00,
                "late": 1.00,
            },
        ),
        "opening0.75": LaneConfig(
            name="value_trust_opening_0_75_mid_1_00_late_1_00",
            description="Opening-only trust reduction to 0.75.",
            value_trust_schedule={
                "enabled": True,
                "opening": 0.75,
                "midgame": 1.00,
                "late": 1.00,
            },
        ),
        "mid0.50": LaneConfig(
            name="value_trust_opening_1_00_mid_0_50_late_1_00",
            description="Midgame-only trust reduction to 0.50.",
            value_trust_schedule={
                "enabled": True,
                "opening": 1.00,
                "midgame": 0.50,
                "late": 1.00,
            },
        ),
        "late0.50": LaneConfig(
            name="value_trust_opening_1_00_mid_1_00_late_0_50",
            description="Late-only trust reduction to 0.50.",
            value_trust_schedule={
                "enabled": True,
                "opening": 1.00,
                "midgame": 1.00,
                "late": 0.50,
            },
        ),
        "all1.25": LaneConfig(
            name="value_trust_all_1_25",
            description="Uniform value trust 1.25 in opening, midgame, and late.",
            value_trust_schedule={
                "enabled": True,
                "opening": 1.25,
                "midgame": 1.25,
                "late": 1.25,
            },
        ),
        "opening1.25": LaneConfig(
            name="value_trust_opening_1_25_mid_1_00_late_1_00",
            description="Opening-only trust increase to 1.25.",
            value_trust_schedule={
                "enabled": True,
                "opening": 1.25,
                "midgame": 1.00,
                "late": 1.00,
            },
        ),
    }
    lanes = [
        LaneConfig(
            name="default_value_trust_1_00",
            description="Promoted runtime default with no configured value-trust override.",
            value_trust_schedule=None,
        )
    ]
    for token in [item.strip() for item in requested.split(",") if item.strip()]:
        if token not in supported:
            raise ValueError(f"unsupported lane token: {token}")
        lanes.append(supported[token])
    return lanes


def phase_for_entry(entry: dict[str, Any]) -> str:
    ply = int(entry.get("ply", len(entry.get("prefix_moves", []))))
    state = entry["state"]
    seeds_remaining = sum(int(v) for v in state["player_pits"] + state["opponent_pits"])
    if ply <= 8:
        return "opening"
    if seeds_remaining <= 16:
        return "late"
    return "mid"


def seat_context_for_entry(entry: dict[str, Any]) -> str:
    return f"P{int(entry['state'].get('current_player', entry.get('side_to_move', 0)))}"


def load_suite_entries(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in read_jsonl(path):
        row = dict(entry)
        row["suite_name"] = path.stem
        row["phase"] = phase_for_entry(row)
        row["seat_context"] = seat_context_for_entry(row)
        row["state_hash"] = canonical_state_hash(row["state"])
        row["prefix_key"] = ",".join(
            str(int(move)) for move in row.get("prefix_moves", [])
        )
        rows.append(row)
    return rows


def ensure_suite_prefixes(path: Path, suite_entries: list[dict[str, Any]]) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in suite_entries:
            handle.write(json.dumps({"prefix_moves": entry["prefix_moves"]}) + "\n")


def value_trust_args(schedule: dict[str, bool | float] | None) -> list[str]:
    if schedule is None:
        return []
    return [
        "--value-trust-enabled",
        "--value-trust-opening",
        str(float(schedule["opening"])),
        "--value-trust-midgame",
        str(float(schedule["midgame"])),
        "--value-trust-late",
        str(float(schedule["late"])),
    ]


def run_opening_suite_lane(
    *,
    lane: LaneConfig,
    suite_name: str,
    suite_path: Path,
    suite_entries: list[dict[str, Any]],
    current_path: Path,
    pair: tuple[int, int],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    games_per_opening: int,
    workers: int,
    seed: int,
    timeout: int,
    workdir: Path,
) -> dict[str, Any]:
    challenger_sims, current_sims = pair
    label = budget_pair_label(challenger_sims, current_sims)
    pair_dir = workdir / suite_name / lane.name / label.replace(":", "_")
    pair_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = pair_dir / "metrics.json"
    effective_c_puct = resolve_budget_cpuct(
        schedule=cpuct_schedule,
        challenger_simulations=challenger_sims,
        current_simulations=current_sims,
        default_c_puct=default_c_puct,
    )
    cache_context = {
        "lane": lane.name,
        "suite_path": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "suite_size": len(suite_entries),
        "current_path": str(current_path),
        "current_sha256": sha256_file(current_path / "weights.json"),
        "challenger_simulations": challenger_sims,
        "current_simulations": current_sims,
        "effective_c_puct": effective_c_puct,
        "c_puct_schedule": schedule_definition(
            default_c_puct=default_c_puct,
            schedule=cpuct_schedule,
        )["overrides"],
        "games_per_opening": games_per_opening,
        "workers": workers,
        "seed": seed,
        "root_policy_mode": "deterministic",
        "tactical_root_bias": tactical_root_bias,
        "value_trust_schedule": lane.value_trust_schedule,
    }
    if metrics_path.is_file():
        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        if cached.get("cache_context") == cache_context:
            return cached

    total_games = len(suite_entries) * max(1, int(games_per_opening))
    all_game_entries: list[dict[str, Any]] = []
    seat_reports: list[dict[str, Any]] = []
    for seat in (0, 1):
        seat_dir = pair_dir / f"starts_{seat}"
        seat_dir.mkdir(parents=True, exist_ok=True)
        arena_json = seat_dir / "arena.json"
        games_jsonl = seat_dir / "games.jsonl"
        suite_jsonl = seat_dir / "opening_suite.jsonl"
        seat_context = {**cache_context, "challenger_starts": seat}
        meta_path = seat_dir / "metadata.json"
        ensure_suite_prefixes(suite_jsonl, suite_entries)
        if meta_path.is_file() and arena_json.is_file() and games_jsonl.is_file():
            cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if cached_meta.get("cache_context") == seat_context:
                game_entries = parse_game_jsonl(str(games_jsonl))
                if game_entries:
                    all_game_entries.extend(game_entries)
                    seat_reports.append(
                        json.loads(arena_json.read_text(encoding="utf-8"))
                    )
                    continue

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
            "--challenger-starts",
            str(seat),
            "--games-per-opening",
            str(games_per_opening),
            "--opening-prefixes-jsonl",
            str(suite_jsonl),
            "--root-policy-mode",
            "deterministic",
            "--c-puct",
            str(effective_c_puct),
            "--tactical-root-bias",
            str(tactical_root_bias),
            *value_trust_args(lane.value_trust_schedule),
        ]
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
                f"arena failed for {lane.name} {suite_name} {label}: {result.stderr[-2000:]}"
            )
        report = json.loads(arena_json.read_text(encoding="utf-8"))
        game_entries = parse_game_jsonl(str(games_jsonl))
        all_game_entries.extend(game_entries)
        seat_reports.append(report)
        meta_path.write_text(
            json.dumps(
                {
                    "cache_context": seat_context,
                    "games": len(game_entries),
                    "arena_score": report.get("score"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    metrics = compute_seat_metrics(all_game_entries)
    per_opening_metrics = compute_per_opening_metrics(all_game_entries)
    by_ply_metrics = compute_by_ply_metrics(all_game_entries)
    report_notes = seat_reports[0].get("notes", {}) if seat_reports else {}
    mean_move_values = [
        float(report.get("notes", {}).get("move_time_mean_ms"))
        for report in seat_reports
        if report.get("notes", {}).get("move_time_mean_ms") is not None
    ]
    p95_move_values = [
        float(report.get("notes", {}).get("move_time_p95_ms"))
        for report in seat_reports
        if report.get("notes", {}).get("move_time_p95_ms") is not None
    ]
    value_trust_summary = next(
        (
            report.get("value_trust_summary")
            for report in seat_reports
            if isinstance(report.get("value_trust_summary"), dict)
        ),
        None,
    )
    payload = {
        **metrics,
        "cache_context": cache_context,
        "suite_name": suite_name,
        "budget_pair": label,
        "challenger_simulations": challenger_sims,
        "current_simulations": current_sims,
        "effective_c_puct": effective_c_puct,
        "tactical_root_bias": tactical_root_bias,
        "per_opening_metrics": per_opening_metrics,
        "by_ply_metrics": {str(key): value for key, value in by_ply_metrics.items()},
        "search_profile": report_notes.get("search_profile"),
        "search_profile_hash": report_notes.get("search_profile_hash"),
        "value_trust_summary": value_trust_summary,
        "move_time_mean_ms": statistics.fmean(mean_move_values)
        if mean_move_values
        else None,
        "move_time_p95_ms": statistics.fmean(p95_move_values)
        if p95_move_values
        else None,
        "total_games": len(all_game_entries),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_suite_benchmark(
    *,
    lanes: list[LaneConfig],
    suite_path: Path,
    current_path: Path,
    budget_pairs: list[tuple[int, int]],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    games_per_opening: int,
    workers: int,
    seed: int,
    timeout: int,
    workdir: Path,
) -> dict[str, Any]:
    suite_entries = load_suite_entries(suite_path)
    report: dict[str, Any] = {
        "suite_path": str(suite_path),
        "suite_name": suite_path.stem,
        "suite_sha256": sha256_file(suite_path),
        "suite_size": len(suite_entries),
        "lanes": {},
    }
    for lane in lanes:
        lane_result: dict[str, Any] = {"candidate": lane.name, "budget_results": {}}
        for pair in budget_pairs:
            budget_result = run_opening_suite_lane(
                lane=lane,
                suite_name=suite_path.stem,
                suite_path=suite_path,
                suite_entries=suite_entries,
                current_path=current_path,
                pair=pair,
                default_c_puct=default_c_puct,
                cpuct_schedule=cpuct_schedule,
                tactical_root_bias=tactical_root_bias,
                games_per_opening=games_per_opening,
                workers=workers,
                seed=seed,
                timeout=timeout,
                workdir=workdir,
            )
            lane_result["budget_results"][budget_result["budget_pair"]] = budget_result
        report["lanes"][lane.name] = lane_result
    return report


def inspect_supported_controls() -> dict[str, Any]:
    return {
        "value_trust": {
            "supported": True,
            "scope": "global_opening_midgame_late_schedule",
            "notes": (
                "The active arena and gate paths support global value-trust schedules with "
                "opening/midgame/late multipliers > 0 and emit value_trust_summary telemetry."
            ),
        },
        "value_disable_or_zero": {
            "supported": False,
            "notes": (
                "The active path does not expose a disabled-value runtime lane through the opening-suite "
                "benchmark, and value-trust multipliers must stay finite and > 0."
            ),
        },
        "search_modes": {
            "supported": ["full"],
            "unsupported": ["policy_only", "value_only", "classic_only"],
            "notes": "The active opening-suite benchmark path does not expose search-mode ablations beyond full.",
        },
    }


def sample_audit_rows(
    *,
    rows: list[dict[str, Any]],
    target_total_states: int,
    min_total_states: int,
    budget_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    unique_target = max(min_total_states // budget_count, 1)
    unique_target = max(
        unique_target, min(target_total_states // budget_count, len(rows))
    )
    unique_target = min(unique_target, len(rows))
    by_hash: dict[str, dict[str, Any]] = {}
    for row in rows:
        state_hash = str(row["state_hash"])
        if state_hash not in by_hash:
            by_hash[state_hash] = row
    unique_rows = list(by_hash.values())
    rng = random.Random(seed)
    rng.shuffle(unique_rows)
    bins: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in unique_rows:
        bins[(str(row["phase"]), str(row["seat_context"]))].append(row)
    prefix_cap = max(
        2,
        math.ceil(
            unique_target / max(1, len({row["prefix_key"] for row in unique_rows})) * 8
        ),
    )
    selected: list[dict[str, Any]] = []
    per_prefix = Counter()
    bin_keys = sorted(bins)
    while len(selected) < unique_target:
        progressed = False
        for key in bin_keys:
            candidates = bins[key]
            while candidates:
                row = candidates.pop()
                prefix_key = str(row.get("prefix_key", ""))
                if per_prefix[prefix_key] >= prefix_cap:
                    continue
                selected.append(row)
                per_prefix[prefix_key] += 1
                progressed = True
                break
            if len(selected) >= unique_target:
                break
        if not progressed:
            break
    return selected


def sample_audit_seed_rows(
    *,
    rows: list[dict[str, Any]],
    target_unique_states: int,
    seed: int,
) -> list[dict[str, Any]]:
    root_target = max(64, math.ceil(target_unique_states / 4))
    root_target = min(root_target, len(rows))
    rng = random.Random(seed)
    shuffled_rows = list(rows)
    rng.shuffle(shuffled_rows)
    bins: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in shuffled_rows:
        bins[(str(row["suite_name"]), str(row["seat_context"]))].append(row)
    prefix_cap = max(
        2, math.ceil(root_target / max(1, len({row["prefix_key"] for row in rows})) * 8)
    )
    selected: list[dict[str, Any]] = []
    per_prefix = Counter()
    bin_keys = sorted(bins)
    while len(selected) < root_target:
        progressed = False
        for key in bin_keys:
            candidates = bins[key]
            while candidates:
                row = candidates.pop()
                prefix_key = str(row.get("prefix_key", ""))
                if per_prefix[prefix_key] >= prefix_cap:
                    continue
                selected.append(row)
                per_prefix[prefix_key] += 1
                progressed = True
                break
            if len(selected) >= root_target:
                break
        if not progressed:
            break
    return selected


def sign_value(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def phase_for_state(*, total_ply: int, state: dict[str, Any]) -> str:
    seeds_remaining = sum(
        int(value) for value in state["player_pits"] + state["opponent_pits"]
    )
    if total_ply <= 8:
        return "opening"
    if seeds_remaining <= 16:
        return "late"
    return "mid"


def spearman_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None

    def average_ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
        ranks = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position + 1
            while end < len(order) and values[order[end]] == values[order[position]]:
                end += 1
            average_rank = (position + 1 + end) / 2.0
            for slot in range(position, end):
                ranks[order[slot]] = average_rank
            position = end
        return ranks

    x_ranks = np.asarray(average_ranks(xs), dtype=np.float64)
    y_ranks = np.asarray(average_ranks(ys), dtype=np.float64)
    x_std = float(x_ranks.std())
    y_std = float(y_ranks.std())
    if x_std <= 0.0 or y_std <= 0.0:
        return None
    return float(np.corrcoef(x_ranks, y_ranks)[0, 1])


def bucket_label(low: float, high: float) -> str:
    right = "]" if high > 1.0 else ")"
    upper = 1 if high > 1.0 else high
    return f"[{low:g},{upper:g}{right}"


def summarize_value_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {
                "count": 0,
                "mse": None,
                "mae": None,
                "sign_accuracy": None,
                "spearman": None,
                "overconfidence_rate": None,
            }
        raw_values = [float(row["raw_value"]) for row in rows]
        outcomes = [float(row["continuation_outcome"]) for row in rows]
        mse = statistics.fmean((rv - ov) ** 2 for rv, ov in zip(raw_values, outcomes))
        mae = statistics.fmean(abs(rv - ov) for rv, ov in zip(raw_values, outcomes))
        sign_accuracy = statistics.fmean(
            1.0 if sign_value(rv) == sign_value(ov) else 0.0
            for rv, ov in zip(raw_values, outcomes)
        )
        overconfidence = [
            row
            for row in rows
            if abs(float(row["raw_value"])) >= 0.75
            and sign_value(float(row["raw_value"]))
            != sign_value(float(row["continuation_outcome"]))
        ]
        return {
            "count": len(rows),
            "mse": float(mse),
            "mae": float(mae),
            "sign_accuracy": float(sign_accuracy),
            "spearman": spearman_correlation(raw_values, outcomes),
            "overconfidence_rate": float(len(overconfidence) / len(rows)),
        }

    by_bucket: dict[str, dict[str, Any]] = {}
    for low, high in VALUE_BUCKETS:
        label = bucket_label(low, high)
        bucket_rows = [
            row
            for row in records
            if float(row["raw_value"]) >= low and float(row["raw_value"]) < high
        ]
        by_bucket[label] = summarize_subset(bucket_rows)
    by_phase = {
        phase: summarize_subset([row for row in records if row["phase"] == phase])
        for phase in ("opening", "mid", "late")
    }
    budget_labels = sorted({str(row["budget_pair"]) for row in records})
    by_budget = {
        label: summarize_subset([row for row in records if row["budget_pair"] == label])
        for label in budget_labels
    }
    seat_labels = sorted({str(row["seat_context"]) for row in records})
    by_seat = {
        label: summarize_subset(
            [row for row in records if row["seat_context"] == label]
        )
        for label in seat_labels
    }
    overconfident_wrong = [
        row
        for row in records
        if abs(float(row["raw_value"])) >= 0.75
        and sign_value(float(row["raw_value"]))
        != sign_value(float(row["continuation_outcome"]))
    ]
    overconfident_wrong.sort(
        key=lambda row: (
            -abs(float(row["raw_value"])),
            -abs(float(row["final_margin_from_side_to_move"])),
            str(row["state_hash"]),
        )
    )
    return {
        "overall": summarize_subset(records),
        "by_value_bucket": by_bucket,
        "by_phase": by_phase,
        "by_budget_pair": by_budget,
        "by_seat_context": by_seat,
        "top_overconfident_wrong_states": overconfident_wrong[:50],
    }


def play_deterministic_continuation(
    *,
    evaluator: ArtifactEvaluator,
    state: dict[str, Any],
    challenger_sims: int,
    current_sims: int,
    effective_c_puct: float,
    tactical_root_bias: float,
    seed: int,
) -> dict[str, Any]:
    game = KalahGame.from_state(state)
    side_to_move = int(game.current_player)
    trajectory: list[int] = []
    while not game.over():
        legal_moves = game.possible_moves()
        if not legal_moves:
            break
        simulations = challenger_sims if game.current_player == 0 else current_sims
        summary = evaluate_artifact_position(
            evaluator=evaluator,
            state=game.to_state(),
            simulations=simulations,
            seed=seed + len(trajectory),
            c_puct=effective_c_puct,
            search_options=build_eval_search_options(
                root_policy_mode="deterministic",
                tactical_root_bias=tactical_root_bias,
            ),
        )
        selected_move = summary.get("selected_move")
        if selected_move is None:
            break
        absolute_move = game.pit_index(int(selected_move))
        if not game.move(absolute_move):
            break
        trajectory.append(int(selected_move))
    final_margin = int(game.captured_seeds[0] - game.captured_seeds[1])
    perspective_margin = final_margin if side_to_move == 0 else -final_margin
    return {
        "continuation_outcome": float(sign_value(perspective_margin)),
        "final_margin_from_side_to_move": int(perspective_margin),
        "final_stores": {
            "player0": int(game.captured_seeds[0]),
            "player1": int(game.captured_seeds[1]),
        },
        "trajectory": trajectory,
    }


def trace_deterministic_audit_states(
    *,
    evaluator: ArtifactEvaluator,
    seed_rows: list[dict[str, Any]],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
) -> list[dict[str, Any]]:
    trace_budget_pair = "384:256"
    challenger_sims, current_sims = [
        int(part) for part in trace_budget_pair.split(":", 1)
    ]
    effective_c_puct = resolve_budget_cpuct(
        schedule=cpuct_schedule,
        challenger_simulations=challenger_sims,
        current_simulations=current_sims,
        default_c_puct=default_c_puct,
    )
    traced_rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for row_index, seed_row in enumerate(seed_rows):
        game = KalahGame.from_state(seed_row["state"])
        total_ply = int(seed_row.get("ply", len(seed_row.get("prefix_moves", []))))
        local_ply = 0
        while True:
            state = game.to_state()
            state_hash = canonical_state_hash(state)
            if state_hash not in seen_hashes:
                traced_rows.append(
                    {
                        "state": state,
                        "state_hash": state_hash,
                        "suite_name": str(seed_row["suite_name"]),
                        "prefix_moves": list(seed_row.get("prefix_moves", [])),
                        "prefix_key": str(seed_row.get("prefix_key", "")),
                        "phase": phase_for_state(total_ply=total_ply, state=state),
                        "seat_context": f"P{int(state.get('current_player', 0))}",
                        "ply": total_ply,
                    }
                )
                seen_hashes.add(state_hash)
            if game.over():
                break
            legal_moves = game.possible_moves()
            if not legal_moves:
                break
            simulations = challenger_sims if game.current_player == 0 else current_sims
            summary = evaluate_artifact_position(
                evaluator=evaluator,
                state=state,
                simulations=simulations,
                seed=seed + (row_index * 97) + local_ply,
                c_puct=effective_c_puct,
                search_options=build_eval_search_options(
                    root_policy_mode="deterministic",
                    tactical_root_bias=tactical_root_bias,
                ),
            )
            selected_move = summary.get("selected_move")
            if selected_move is None:
                break
            absolute_move = game.pit_index(int(selected_move))
            if not game.move(absolute_move):
                break
            total_ply += 1
            local_ply += 1
    return traced_rows


def run_value_calibration_audit(
    *,
    current_path: Path,
    fixed_large_suite: Path,
    heldout_suites: list[Path],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    workdir: Path,
    seed: int,
    target_states: int,
    min_states: int,
) -> dict[str, Any]:
    evaluator = ArtifactEvaluator(current_path)
    rows = load_suite_entries(fixed_large_suite)
    for suite_path in heldout_suites:
        rows.extend(load_suite_entries(suite_path))
    target_unique_states = max(min_states // len(AUDIT_BUDGETS), 1)
    target_unique_states = max(
        target_unique_states, target_states // len(AUDIT_BUDGETS)
    )
    seed_rows = sample_audit_seed_rows(
        rows=rows,
        target_unique_states=target_unique_states,
        seed=seed,
    )
    traced_rows = trace_deterministic_audit_states(
        evaluator=evaluator,
        seed_rows=seed_rows,
        default_c_puct=default_c_puct,
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=tactical_root_bias,
        seed=seed,
    )
    sampled_rows = sample_audit_rows(
        rows=traced_rows,
        target_total_states=target_states,
        min_total_states=min_states,
        budget_count=len(AUDIT_BUDGETS),
        seed=seed + 1,
    )
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(sampled_rows):
        for budget_index, budget_pair in enumerate(AUDIT_BUDGETS):
            challenger_sims, current_sims = [
                int(part) for part in budget_pair.split(":", 1)
            ]
            effective_c_puct = resolve_budget_cpuct(
                schedule=cpuct_schedule,
                challenger_simulations=challenger_sims,
                current_simulations=current_sims,
                default_c_puct=default_c_puct,
            )
            search_options = build_eval_search_options(
                root_policy_mode="deterministic",
                tactical_root_bias=tactical_root_bias,
            )
            profile = build_search_profile(
                kind="arena_eval",
                player_mode="puct",
                simulations=max(challenger_sims, current_sims),
                c_puct=effective_c_puct,
                search_options=search_options,
                extra_fields={
                    "challenger_simulations": challenger_sims,
                    "current_simulations": current_sims,
                },
            )
            game = KalahGame.from_state(row["state"])
            legal_moves = [int(move) for move in game.possible_moves()]
            _logits, raw_policy, raw_value = artifact_forward_details(evaluator, game)
            raw_top_move = top_policy_move(raw_policy, legal_moves)
            root_simulations = (
                challenger_sims if game.current_player == 0 else current_sims
            )
            root_summary = evaluate_artifact_position(
                evaluator=evaluator,
                state=row["state"],
                simulations=root_simulations,
                seed=seed + (row_index * 31) + budget_index,
                c_puct=effective_c_puct,
                search_options=search_options,
            )
            continuation = play_deterministic_continuation(
                evaluator=evaluator,
                state=row["state"],
                challenger_sims=challenger_sims,
                current_sims=current_sims,
                effective_c_puct=effective_c_puct,
                tactical_root_bias=tactical_root_bias,
                seed=seed + (row_index * 101) + budget_index,
            )
            record = {
                "state_hash": str(row["state_hash"]),
                "suite": str(row["suite_name"]),
                "budget_pair": budget_pair,
                "phase": str(row["phase"]),
                "seat_context": str(row["seat_context"]),
                "side_to_move": int(row["state"].get("current_player", 0)),
                "raw_value": float(raw_value),
                "raw_policy_top_move": raw_top_move,
                "raw_policy_top_margin": float(raw_margin(raw_policy, legal_moves)),
                "puct_root_value": (
                    root_summary.get("selection_breakdown", {}).get("parent_q_value")
                    if isinstance(root_summary.get("selection_breakdown"), dict)
                    else None
                ),
                "puct_root_visits": int(
                    sum(float(value) for value in root_summary.get("visits", []))
                ),
                "continuation_outcome": continuation["continuation_outcome"],
                "final_margin_from_side_to_move": continuation[
                    "final_margin_from_side_to_move"
                ],
                "final_stores": continuation["final_stores"],
                "legal_moves": legal_moves,
                "search_profile_hash": str(profile["hash"]),
                "search_profile": profile,
                "prefix_moves": list(row.get("prefix_moves", [])),
            }
            records.append(record)
    write_jsonl(workdir / VALUE_STATES_FILENAME, records)
    audit = {
        "schema": "azlite_post_schedule_value_calibration_audit_v1",
        "state_count": len(records),
        "unique_state_count": len({str(row["state_hash"]) for row in records}),
        "seed_state_count": len(seed_rows),
        "trace_pool_state_count": len(traced_rows),
        "budget_pairs": list(AUDIT_BUDGETS),
        "summary": summarize_value_records(records),
    }
    write_json(workdir / VALUE_AUDIT_FILENAME, audit)
    return audit


def suite_delta(
    report: dict[str, Any], lane_name: str, default_name: str, budget_pair: str
) -> float:
    lane_ds = float(report["lanes"][lane_name]["budget_results"][budget_pair]["ds"])
    default_ds = float(
        report["lanes"][default_name]["budget_results"][budget_pair]["ds"]
    )
    return lane_ds - default_ds


def score_lane(report: dict[str, Any], lane_name: str, default_name: str) -> float:
    return (
        suite_delta(report, lane_name, default_name, "384:256")
        + (0.5 * suite_delta(report, lane_name, default_name, "768:768"))
        + (0.25 * suite_delta(report, lane_name, default_name, "1200:1200"))
        + (0.25 * suite_delta(report, lane_name, default_name, "1200:256"))
    )


def carry_lanes_from_medium(report: dict[str, Any], default_name: str) -> list[str]:
    candidates: set[str] = {default_name}
    for lane_name in report["lanes"]:
        if lane_name == default_name:
            continue
        delta_384 = suite_delta(report, lane_name, default_name, "384:256")
        delta_768 = suite_delta(report, lane_name, default_name, "768:768")
        delta_1200 = suite_delta(report, lane_name, default_name, "1200:1200")
        delta_1200_256 = suite_delta(report, lane_name, default_name, "1200:256")
        if delta_384 >= 0.05 and delta_768 >= -0.05:
            candidates.add(lane_name)
        if (delta_1200 >= 0.05 or delta_1200_256 >= 0.05) and delta_384 >= -0.05:
            candidates.add(lane_name)
    ranked = sorted(
        (name for name in candidates if name != default_name),
        key=lambda name: (score_lane(report, name, default_name), name),
        reverse=True,
    )
    return [default_name, *ranked[:6]]


def heldout_lane_summary(
    *,
    heldout_reports: dict[str, dict[str, Any]],
    lane_name: str,
    default_name: str,
    budget_pairs: list[str],
) -> dict[str, Any]:
    def aggregate_for(candidate_name: str) -> dict[str, Any]:
        aggregate: dict[str, Any] = {}
        for budget_pair in budget_pairs:
            rows: list[dict[str, Any]] = []
            for suite_name, report in heldout_reports.items():
                budget_result = report["lanes"][candidate_name]["budget_results"].get(
                    budget_pair
                )
                if budget_result is None:
                    continue
                rows.append(
                    {
                        "suite": suite_name,
                        "ds": budget_result.get("ds"),
                        "p0_score": budget_result.get("p0_score"),
                        "p1_score": budget_result.get("p1_score"),
                        "duplicate_trajectory_count": budget_result.get(
                            "duplicate_trajectory_count"
                        ),
                    }
                )
            ds_values = [float(row["ds"]) for row in rows if row["ds"] is not None]
            aggregate[budget_pair] = {
                "available": bool(rows),
                "rows": rows,
                "mean_ds": statistics.fmean(ds_values) if ds_values else None,
                "worst_suite_ds": min(ds_values) if ds_values else None,
                "best_suite_ds": max(ds_values) if ds_values else None,
            }
        return aggregate

    lane_summary = aggregate_for(lane_name)
    default_summary = aggregate_for(default_name)
    lane_summary["deltas_vs_default"] = {
        budget_pair: (
            None
            if lane_summary[budget_pair]["mean_ds"] is None
            or default_summary[budget_pair]["mean_ds"] is None
            else float(lane_summary[budget_pair]["mean_ds"])
            - float(default_summary[budget_pair]["mean_ds"])
        )
        for budget_pair in budget_pairs
    }
    return lane_summary


def carry_lanes_from_fixed_large(
    *,
    fixed_large_report: dict[str, Any],
    heldout_reports: dict[str, dict[str, Any]] | None,
    default_name: str,
) -> list[str]:
    ranked = sorted(
        (name for name in fixed_large_report["lanes"] if name != default_name),
        key=lambda name: (score_lane(fixed_large_report, name, default_name), name),
        reverse=True,
    )
    selected: set[str] = {default_name, *ranked[:3]}
    for lane_name in fixed_large_report["lanes"]:
        if lane_name == default_name:
            continue
        delta_384 = suite_delta(fixed_large_report, lane_name, default_name, "384:256")
        delta_1200 = suite_delta(
            fixed_large_report, lane_name, default_name, "1200:1200"
        )
        delta_768 = suite_delta(fixed_large_report, lane_name, default_name, "768:768")
        delta_1200_256 = suite_delta(
            fixed_large_report, lane_name, default_name, "1200:256"
        )
        robustness_regression = delta_768 < -0.05 or delta_1200_256 < -0.03
        if delta_384 >= 0.05 and not robustness_regression:
            selected.add(lane_name)
        if delta_1200 >= 0.08 and delta_384 >= -0.03:
            selected.add(lane_name)
    return [default_name, *sorted(name for name in selected if name != default_name)]


def effective_profile_hashes(
    report: dict[str, Any],
) -> dict[str, dict[str, str | None]]:
    output: dict[str, dict[str, str | None]] = {}
    for lane_name, lane_report in report["lanes"].items():
        output[lane_name] = {
            budget_pair: budget.get("search_profile_hash")
            for budget_pair, budget in lane_report["budget_results"].items()
        }
    return output


def telemetry_summary(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        lane_name: {
            budget_pair: budget.get("value_trust_summary")
            for budget_pair, budget in lane_report["budget_results"].items()
            if budget.get("value_trust_summary") is not None
        }
        for lane_name, lane_report in report["lanes"].items()
    }


def runtime_cost_summary(
    report: dict[str, Any], default_name: str
) -> dict[str, dict[str, Any]]:
    default_budget = report["lanes"][default_name]["budget_results"]["384:256"]
    default_mean = float(default_budget.get("move_time_mean_ms") or 0.0)
    summary: dict[str, dict[str, Any]] = {}
    for lane_name, lane_report in report["lanes"].items():
        budget = lane_report["budget_results"]["384:256"]
        mean_latency = float(budget.get("move_time_mean_ms") or 0.0)
        summary[lane_name] = {
            "mean_move_latency_ms": budget.get("move_time_mean_ms"),
            "p95_move_latency_ms": budget.get("move_time_p95_ms"),
            "relative_slowdown": None
            if default_mean <= 0.0
            else float((mean_latency / default_mean) - 1.0),
        }
    return summary


def build_bootstrap_rows(
    *,
    heldout_reports: dict[str, dict[str, Any]],
    lane_names: list[str],
    default_name: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    suite_rows = {
        suite_name: {
            "candidates": {
                lane_name: {"budget_results": lane_report["budget_results"]}
                for lane_name, lane_report in report["lanes"].items()
            }
        }
        for suite_name, report in heldout_reports.items()
    }
    result: dict[str, dict[str, Any]] = {}
    for lane_name in lane_names:
        if lane_name == default_name:
            continue
        lane_entry: dict[str, Any] = {}
        for budget_pair in BOOTSTRAP_BUDGETS:
            diffs = pooled_per_opening_differences(
                suite_rows=suite_rows,
                candidate_a=lane_name,
                candidate_b=default_name,
                budget_pair=budget_pair,
                metric_key="ds",
            )
            lane_entry[budget_pair] = bootstrap_ci(
                diffs,
                seed=seed + abs(hash((lane_name, budget_pair))) % 100000,
                samples=bootstrap_samples,
            )
        result[lane_name] = lane_entry
    return result


def aggregate_p0_p1(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        lane_name: {
            "p0_score": lane_report["budget_results"]["384:256"].get("p0_score"),
            "p1_score": lane_report["budget_results"]["384:256"].get("p1_score"),
            "gap": float(lane_report["budget_results"]["384:256"].get("ds") or 0.0),
        }
        for lane_name, lane_report in report["lanes"].items()
    }


def aggregate_duplicates(report: dict[str, Any]) -> dict[str, float | None]:
    return {
        lane_name: lane_report["budget_results"]["384:256"].get(
            "duplicate_trajectory_count"
        )
        for lane_name, lane_report in report["lanes"].items()
    }


def run_gate(
    *,
    lane: LaneConfig,
    current_path: Path,
    out_path: Path,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
    workers: int,
    games: int,
) -> dict[str, Any]:
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
        DEFAULT_BUDGET_PAIRS,
        "--c-puct",
        str(default_c_puct),
        "--c-puct-schedule-json",
        json.dumps(cpuct_schedule, sort_keys=True),
        "--root-policy-mode",
        "deterministic",
        "--tactical-root-bias",
        str(tactical_root_bias),
        *value_trust_args(lane.value_trust_schedule),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=7200,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gate failed for {lane.name}: {result.stderr[-2000:]}")
    return json.loads(out_path.read_text(encoding="utf-8"))


def classify_experiment(
    *,
    audit: dict[str, Any],
    heldout_means: dict[str, Any],
    bootstrap: dict[str, dict[str, Any]],
    gate_results: dict[str, dict[str, Any]],
    default_name: str,
) -> dict[str, Any]:
    overconf_opening = (
        float(audit["summary"]["by_phase"]["opening"].get("overconfidence_rate") or 0.0)
        > 0.05
    )
    winners: list[str] = []
    candidates: list[str] = []
    phase_specific_help = False
    for lane_name, summary in heldout_means.items():
        if lane_name == default_name:
            continue
        delta_384 = float(summary["deltas_vs_default"].get("384:256") or 0.0)
        delta_768 = float(summary["deltas_vs_default"].get("768:768") or 0.0)
        delta_1200 = float(summary["deltas_vs_default"].get("1200:1200") or 0.0)
        delta_1200_256 = float(summary["deltas_vs_default"].get("1200:256") or 0.0)
        ci_384 = bootstrap.get(lane_name, {}).get("384:256", {})
        ci_1200 = bootstrap.get(lane_name, {}).get("1200:1200", {})
        improved = delta_384 >= 0.05 or delta_1200 >= 0.08
        if improved:
            winners.append(lane_name)
        if "opening" in lane_name or "mid" in lane_name or "late" in lane_name:
            if improved:
                phase_specific_help = True
        gate_ok = True
        if lane_name in gate_results:
            gate_ok = (
                gate_results[lane_name].get("classification")
                == "high_search_breakthrough"
            )
        if (
            (
                (delta_384 >= 0.05 and float(ci_384.get("lower") or 0.0) > 0.01)
                or (delta_1200 >= 0.08 and float(ci_1200.get("lower") or 0.0) > 0.01)
            )
            and delta_768 >= -0.05
            and delta_1200_256 >= -0.03
            and gate_ok
        ):
            candidates.append(lane_name)
    if winners and overconf_opening:
        classification = "value_head_overtrusted"
    elif phase_specific_help and overconf_opening:
        classification = "phase_specific_value_miscalibration"
    elif candidates:
        classification = "value_trust_runtime_candidate"
    elif winners:
        classification = "value_trust_not_enough"
    else:
        classification = "value_head_useful_default"
    return {
        "classification": classification,
        "runtime_candidates": candidates,
        "improving_lanes": winners,
    }


def build_summary(
    *,
    args: argparse.Namespace,
    workdir: Path,
    current_path: Path,
    current_hash_info: dict[str, Any],
    controls: dict[str, Any],
    lanes: list[LaneConfig],
    medium_report: dict[str, Any],
    fixed_large_report: dict[str, Any],
    heldout_reports: dict[str, dict[str, Any]],
    audit: dict[str, Any],
    bootstrap: dict[str, dict[str, Any]],
    gate_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    default_name = lanes[0].name
    heldout_means = {
        lane.name: heldout_lane_summary(
            heldout_reports=heldout_reports,
            lane_name=lane.name,
            default_name=default_name,
            budget_pairs=list(
                fixed_large_report["lanes"][default_name]["budget_results"].keys()
            ),
        )
        for lane in lanes
        if lane.name in next(iter(heldout_reports.values()))["lanes"]
    }
    classification = classify_experiment(
        audit=audit,
        heldout_means=heldout_means,
        bootstrap=bootstrap,
        gate_results=gate_results,
        default_name=default_name,
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "date": str(date.today()),
        "classification": classification["classification"],
        "decision_details": classification,
        "inputs": {
            "current_artifact": build_input_summary(current_path / "weights.json"),
            "current_hash_verification": current_hash_info,
            "medium_suite": build_input_summary(Path(args.medium_suite)),
            "fixed_large_suite": build_input_summary(Path(args.fixed_large_suite)),
            "heldout_suites": [
                build_input_summary(path)
                for path in parse_csv_paths(args.heldout_suites)
            ],
        },
        "promoted_runtime_profile": {
            "artifact": str(current_path),
            "default_c_puct": float(args.default_c_puct),
            "c_puct_schedule": schedule_definition(
                default_c_puct=float(args.default_c_puct),
                schedule=parse_cpuct_schedule_json(args.cpuct_schedule),
            ),
            "tactical_root_bias": float(args.tactical_root_bias),
            "root_policy_mode": "deterministic",
            "root_prior_transform": None,
            "search_mode": "full",
        },
        "supported_controls": controls,
        "lane_definitions": {
            lane.name: {
                "description": lane.description,
                "supported": lane.supported,
                "search_profile": lane.search_profile(),
            }
            for lane in lanes
        },
        "value_calibration_audit": audit,
        "effective_search_profile_hashes": {
            "medium": effective_profile_hashes(medium_report),
            "fixed_large": effective_profile_hashes(fixed_large_report),
        },
        "value_trust_telemetry": {
            "medium": telemetry_summary(medium_report),
            "fixed_large": telemetry_summary(fixed_large_report),
        },
        "medium": medium_report,
        "fixed_large": fixed_large_report,
        "heldout": heldout_reports,
        "heldout_mean_summary": heldout_means,
        "bootstrap": bootstrap,
        "p0_p1_split_384_256": aggregate_p0_p1(fixed_large_report),
        "duplicate_trajectory_count": aggregate_duplicates(fixed_large_report),
        "runtime_cost": runtime_cost_summary(fixed_large_report, default_name),
        "gate_results": gate_results,
        "workdir": str(workdir),
    }
    return summary


def parse_csv_paths(text: str) -> list[Path]:
    return [Path(item.strip()) for item in text.split(",") if item.strip()]


def write_report(summary: dict[str, Any], lanes: list[LaneConfig]) -> None:
    default_name = lanes[0].name
    audit = summary["value_calibration_audit"]
    medium = summary["medium"]
    fixed_large = summary["fixed_large"]
    heldout_means = summary["heldout_mean_summary"]
    bootstrap = summary["bootstrap"]
    runtime_cost = summary["runtime_cost"]
    gate_results = summary["gate_results"]
    calibration_rows = [
        [
            "overall",
            audit["summary"]["overall"]["count"],
            fmt(audit["summary"]["overall"]["mse"]),
            fmt(audit["summary"]["overall"]["mae"]),
            fmt(audit["summary"]["overall"]["sign_accuracy"]),
            fmt(audit["summary"]["overall"]["spearman"]),
            fmt(audit["summary"]["overall"]["overconfidence_rate"]),
        ]
    ]
    for phase in ("opening", "mid", "late"):
        row = audit["summary"]["by_phase"][phase]
        calibration_rows.append(
            [
                phase,
                row["count"],
                fmt(row["mse"]),
                fmt(row["mae"]),
                fmt(row["sign_accuracy"]),
                fmt(row["spearman"]),
                fmt(row["overconfidence_rate"]),
            ]
        )
    overconf_rows = []
    for row in audit["summary"]["top_overconfident_wrong_states"][:10]:
        overconf_rows.append(
            [
                row["state_hash"][:12],
                row["suite"],
                row["budget_pair"],
                row["phase"],
                row["seat_context"],
                fmt(row["raw_value"]),
                fmt(row["continuation_outcome"]),
                row["final_margin_from_side_to_move"],
            ]
        )
    lane_rows = [
        [
            lane.name,
            lane.description,
            json.dumps(lane.value_trust_schedule, sort_keys=True),
        ]
        for lane in lanes
    ]
    profile_rows = []
    for lane_name, budget_map in summary["effective_search_profile_hashes"][
        "fixed_large"
    ].items():
        for budget_pair, profile_hash in budget_map.items():
            profile_rows.append([lane_name, budget_pair, profile_hash])
    telemetry_rows = []
    for lane_name, budget_map in summary["value_trust_telemetry"][
        "fixed_large"
    ].items():
        for budget_pair, telemetry in budget_map.items():
            telemetry_rows.append(
                [lane_name, budget_pair, json.dumps(telemetry, sort_keys=True)]
            )

    def budget_table(report: dict[str, Any]) -> str:
        rows = []
        for lane in lanes:
            if lane.name not in report["lanes"]:
                continue
            budgets = report["lanes"][lane.name]["budget_results"]
            rows.append(
                [
                    lane.name,
                    fmt(budgets["384:256"]["ds"]),
                    fmt(budgets["768:256"]["ds"]),
                    fmt(budgets["768:768"]["ds"]),
                    fmt(budgets["1200:1200"]["ds"]),
                    fmt(budgets["1200:256"]["ds"]),
                    fmt(budgets["256:768"]["ds"]),
                ]
            )
        return markdown_table(
            [
                "Lane",
                "384:256",
                "768:256",
                "768:768",
                "1200:1200",
                "1200:256",
                "256:768",
            ],
            rows,
        )

    heldout_rows = []
    for lane_name, row in heldout_means.items():
        heldout_rows.append(
            [
                lane_name,
                fmt(row["384:256"]["mean_ds"]),
                fmt(row["deltas_vs_default"]["384:256"]),
                fmt(row["384:256"]["worst_suite_ds"]),
                fmt(row["768:768"]["mean_ds"]),
                fmt(row["1200:1200"]["mean_ds"]),
            ]
        )
    bootstrap_rows = []
    for lane_name, budget_map in bootstrap.items():
        for budget_pair in BOOTSTRAP_BUDGETS:
            ci = budget_map[budget_pair]
            bootstrap_rows.append(
                [
                    f"{lane_name} minus {default_name} @ {budget_pair}",
                    fmt(ci["mean"]),
                    fmt(ci["lower"]),
                    fmt(ci["upper"]),
                ]
            )
    p0_rows = []
    for lane_name, row in summary["p0_p1_split_384_256"].items():
        p0_rows.append(
            [lane_name, fmt(row["p0_score"]), fmt(row["p1_score"]), fmt(row["gap"])]
        )
    dup_rows = []
    for lane_name, value in summary["duplicate_trajectory_count"].items():
        dup_rows.append([lane_name, fmt(value)])
    runtime_rows = []
    for lane_name, row in runtime_cost.items():
        runtime_rows.append(
            [
                lane_name,
                fmt(row["mean_move_latency_ms"]),
                fmt(row["p95_move_latency_ms"]),
                fmt(row["relative_slowdown"], digits=3),
            ]
        )
    gate_rows = []
    for lane_name, report in gate_results.items():
        gate_rows.append([lane_name, report.get("classification")])
    report_text = "\n".join(
        [
            "# AlphaZero-Lite Post-Schedule Value-Trust Calibration Results",
            "",
            f"**Date**: {summary['date']}",
            "",
            f"**Classification**: `{summary['classification']}`",
            "",
            "## Artifact Hash",
            "",
            f"- Current artifact: `{summary['inputs']['current_artifact']['path']}`",
            f"- Current weights SHA256: `{summary['inputs']['current_artifact']['sha256']}`",
            f"- Expected SHA256: `{summary['inputs']['current_hash_verification']['expected_sha256']}`",
            "",
            "## Promoted Search Schedule Confirmation",
            "",
            f"- Runtime profile: `{json.dumps(summary['promoted_runtime_profile'], sort_keys=True)}`",
            "",
            "## Supported And Unsupported Value Controls",
            "",
            f"- Controls: `{json.dumps(summary['supported_controls'], sort_keys=True)}`",
            "",
            "## Value Calibration Audit Table",
            "",
            markdown_table(
                [
                    "Slice",
                    "Count",
                    "MSE",
                    "MAE",
                    "Sign acc",
                    "Spearman",
                    "Overconf rate",
                ],
                calibration_rows,
            ),
            "",
            "## Overconfident-Wrong-State Table",
            "",
            markdown_table(
                [
                    "State",
                    "Suite",
                    "Budget",
                    "Phase",
                    "Seat",
                    "Raw value",
                    "Outcome",
                    "Final margin",
                ],
                overconf_rows,
            ),
            "",
            "## Value-Trust Lane Definitions",
            "",
            markdown_table(["Lane", "Description", "Schedule"], lane_rows),
            "",
            "## Effective Search Profile Hashes",
            "",
            markdown_table(["Lane", "Budget", "Hash"], profile_rows),
            "",
            "## Value-Trust Telemetry",
            "",
            markdown_table(["Lane", "Budget", "Telemetry"], telemetry_rows),
            "",
            "## Medium DS Table",
            "",
            budget_table(medium),
            "",
            "## Fixed Large DS Table",
            "",
            budget_table(fixed_large),
            "",
            "## Held-Out Mean/Worst-Suite DS Table",
            "",
            markdown_table(
                [
                    "Lane",
                    "Held-out mean 384:256",
                    "Delta vs default",
                    "Worst-suite 384:256",
                    "Held-out mean 768:768",
                    "Held-out mean 1200:1200",
                ],
                heldout_rows,
            ),
            "",
            "## Bootstrap CI",
            "",
            markdown_table(
                ["Comparison", "Mean", "Lower 95%", "Upper 95%"], bootstrap_rows
            ),
            "",
            "## P0/P1 Split For 384:256",
            "",
            markdown_table(["Lane", "Mean P0", "Mean P1", "Gap"], p0_rows),
            "",
            "## Duplicate Trajectory Count",
            "",
            markdown_table(["Lane", "Mean duplicates"], dup_rows),
            "",
            "## Runtime Cost",
            "",
            markdown_table(
                ["Lane", "Mean move latency", "P95 move latency", "Relative slowdown"],
                runtime_rows,
            ),
            "",
            "## Gate Classification",
            "",
            markdown_table(["Lane", "Classification"], gate_rows),
            "",
        ]
    )
    REPORT_PATH.write_text(report_text + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    current_path = Path(args.current)
    require_existing_file(current_path / "weights.json", "current weights")
    require_existing_file(Path(args.medium_suite), "medium suite")
    require_existing_file(Path(args.fixed_large_suite), "fixed large suite")
    for path in parse_csv_paths(args.heldout_suites):
        require_existing_file(path, f"heldout suite {path.name}")

    current_hash_info = verify_expected_hash(
        current_path / "weights.json",
        args.expected_current_weights_sha256,
        "current artifact",
    )
    cpuct_schedule = parse_cpuct_schedule_json(args.cpuct_schedule)
    controls = inspect_supported_controls()
    lanes = lane_catalog(args.value_trust_lanes)
    budget_pairs = parse_budget_pairs(args.budget_pairs)

    audit = run_value_calibration_audit(
        current_path=current_path,
        fixed_large_suite=Path(args.fixed_large_suite),
        heldout_suites=parse_csv_paths(args.heldout_suites),
        default_c_puct=float(args.default_c_puct),
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=float(args.tactical_root_bias),
        workdir=workdir,
        seed=int(args.seed),
        target_states=int(args.audit_target_states),
        min_states=int(args.audit_min_states),
    )

    medium_report = run_suite_benchmark(
        lanes=lanes,
        suite_path=Path(args.medium_suite),
        current_path=current_path,
        budget_pairs=budget_pairs,
        default_c_puct=float(args.default_c_puct),
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=float(args.tactical_root_bias),
        games_per_opening=int(args.games_per_opening),
        workers=int(args.workers),
        seed=int(args.seed),
        timeout=int(args.timeout),
        workdir=workdir / "medium_benchmark",
    )
    medium_carry = carry_lanes_from_medium(medium_report, lanes[0].name)
    fixed_large_lanes = [lane for lane in lanes if lane.name in set(medium_carry)]
    fixed_large_report = run_suite_benchmark(
        lanes=fixed_large_lanes,
        suite_path=Path(args.fixed_large_suite),
        current_path=current_path,
        budget_pairs=budget_pairs,
        default_c_puct=float(args.default_c_puct),
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=float(args.tactical_root_bias),
        games_per_opening=int(args.games_per_opening),
        workers=int(args.workers),
        seed=int(args.seed),
        timeout=int(args.timeout),
        workdir=workdir / "fixed_large_benchmark",
    )
    heldout_carry = carry_lanes_from_fixed_large(
        fixed_large_report=fixed_large_report,
        heldout_reports=None,
        default_name=lanes[0].name,
    )
    heldout_lanes = [lane for lane in lanes if lane.name in set(heldout_carry)]
    heldout_reports: dict[str, dict[str, Any]] = {}
    for suite_path in parse_csv_paths(args.heldout_suites):
        heldout_reports[suite_path.stem] = run_suite_benchmark(
            lanes=heldout_lanes,
            suite_path=suite_path,
            current_path=current_path,
            budget_pairs=budget_pairs,
            default_c_puct=float(args.default_c_puct),
            cpuct_schedule=cpuct_schedule,
            tactical_root_bias=float(args.tactical_root_bias),
            games_per_opening=int(args.games_per_opening),
            workers=int(args.workers),
            seed=int(args.seed),
            timeout=int(args.timeout),
            workdir=workdir / "heldout_benchmark",
        )
    bootstrap = build_bootstrap_rows(
        heldout_reports=heldout_reports,
        lane_names=[lane.name for lane in heldout_lanes],
        default_name=lanes[0].name,
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    heldout_means = {
        lane.name: heldout_lane_summary(
            heldout_reports=heldout_reports,
            lane_name=lane.name,
            default_name=lanes[0].name,
            budget_pairs=[budget_pair_label(*pair) for pair in budget_pairs],
        )
        for lane in heldout_lanes
    }
    gate_candidates = {lanes[0].name}
    for lane in heldout_lanes:
        if lane.name == lanes[0].name:
            continue
        row = heldout_means[lane.name]
        delta_384 = float(row["deltas_vs_default"].get("384:256") or 0.0)
        delta_1200 = float(row["deltas_vs_default"].get("1200:1200") or 0.0)
        delta_768 = float(row["deltas_vs_default"].get("768:768") or 0.0)
        delta_1200_256 = float(row["deltas_vs_default"].get("1200:256") or 0.0)
        if (
            (delta_384 >= 0.05 or delta_1200 >= 0.08)
            and delta_768 >= -0.05
            and delta_1200_256 >= -0.03
        ):
            gate_candidates.add(lane.name)
    gate_results: dict[str, dict[str, Any]] = {}
    lane_map = {lane.name: lane for lane in lanes}
    for lane_name in sorted(gate_candidates):
        gate_report = run_gate(
            lane=lane_map[lane_name],
            current_path=current_path,
            out_path=workdir / f"gate_{lane_name}.json",
            default_c_puct=float(args.default_c_puct),
            cpuct_schedule=cpuct_schedule,
            tactical_root_bias=float(args.tactical_root_bias),
            seed=int(args.seed),
            workers=int(args.workers),
            games=int(args.gate_games),
        )
        gate_results[lane_name] = {
            "classification": gate_report.get("classification"),
            "budget_results": gate_budget_results(gate_report),
        }

    summary = build_summary(
        args=args,
        workdir=workdir,
        current_path=current_path,
        current_hash_info=current_hash_info,
        controls=controls,
        lanes=lanes,
        medium_report=medium_report,
        fixed_large_report=fixed_large_report,
        heldout_reports=heldout_reports,
        audit=audit,
        bootstrap=bootstrap,
        gate_results=gate_results,
    )
    write_json(workdir / SUMMARY_FILENAME, summary)
    write_report(summary, lanes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
