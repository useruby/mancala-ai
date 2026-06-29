#!/usr/bin/env python3
"""Guarded promotion of the default 768:768 c_puct=0.90 runtime schedule."""

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

from ml.alphazero_lite import arena  # noqa: E402
from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    DEFAULT_RUNTIME_C_PUCT,
    default_runtime_schedule_definition,
    parse_cpuct_schedule_json,
    resolve_budget_cpuct,
    schedule_definition,
)
from ml.alphazero_lite.run_opening_suite_seat_benchmark import (  # noqa: E402
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
from ml.alphazero_lite.seat_aware_arena import BUDGET_PAIR_LABELS  # noqa: E402


SUMMARY_SCHEMA = "azlite_cpuct_768eq_090_schedule_promotion_v1"
SUMMARY_FILENAME = "summary_metrics.json"
REPORT_PATH = (
    REPO_ROOT / "docs/alphazero-lite-cpuct-768eq-090-schedule-promotion-results.md"
)
DEFAULT_WORKDIR = "/tmp/azlite_cpuct_768eq_090_schedule_promotion"
DEFAULT_BUDGET_PAIRS = "384:256,768:256,768:768,1200:1200,1200:256,256:768"
DEFAULT_GATE_BUDGET_PAIRS = "384:256,1200:1200,1200:256,256:768"
DEFAULT_SEARCH_MODE = "full"
DEFAULT_ROOT_POLICY_MODE = "deterministic"
DEFAULT_TACTICAL_ROOT_BIAS = 0.0
DEFAULT_GAMES_PER_OPENING = 2
DEFAULT_TIMEOUT = 14400
DEFAULT_GATE_GAMES = 120
ACCEPTED_GATE_CLASSIFICATIONS = {"high_search_breakthrough"}
BENCHMARK_BUDGET_PAIR_LABELS = {
    (384, 256): "standard",
    (768, 256): "challenger_768_vs_256",
    (768, 768): "equal_768",
    (1200, 1200): "equal_high",
    (1200, 256): "1200_vs_256",
    (256, 768): "current_high_asymmetry",
}
FILES_CHANGED = [
    "ml/alphazero_lite/cpuct_schedule.py",
    "ml/alphazero_lite/run_opening_suite_seat_benchmark.py",
    "script/ai/seat_aware_promotion_gate",
    "ml/alphazero_lite/test_run_opening_suite_seat_benchmark.py",
    "ml/alphazero_lite/run_cpuct_768eq_090_schedule_promotion.py",
    "docs/alphazero-lite-cpuct-768eq-090-schedule-promotion-results.md",
]


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    omit_schedule_flag: bool
    schedule_overrides: dict[str, float]

    def schedule_manifest(self) -> dict[str, Any]:
        return schedule_definition(
            default_c_puct=DEFAULT_RUNTIME_C_PUCT,
            schedule=self.schedule_overrides,
        )

    def schedule_json(self) -> str:
        return json.dumps(self.schedule_manifest()["overrides"], sort_keys=True)

    def effective_cpuct(self, axis: str) -> float:
        challenger_simulations, current_simulations = parse_budget_pair(axis)
        return resolve_budget_cpuct(
            schedule=self.schedule_overrides,
            challenger_simulations=challenger_simulations,
            current_simulations=current_simulations,
            default_c_puct=DEFAULT_RUNTIME_C_PUCT,
        )


def _python() -> str:
    candidate = REPO_ROOT / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--budget-pairs", default=DEFAULT_BUDGET_PAIRS)
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


def parse_budget_pair(axis: str) -> tuple[int, int]:
    left, right = axis.split(":", 1)
    return int(left), int(right)


def budget_pair_label(challenger: int, current: int) -> str:
    return f"{challenger}:{current}"


def gate_budget_key(pair: tuple[int, int]) -> str:
    return BUDGET_PAIR_LABELS.get(pair, f"{pair[0]}_vs_{pair[1]}")


def run_command(cmd: list[str], *, timeout: int) -> None:
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        raise RuntimeError(
            stderr or stdout or f"command failed: {' '.join(str(part) for part in cmd)}"
        )


def variants(promoted_schedule: dict[str, float]) -> list[Variant]:
    return [
        Variant(
            name="old_no_schedule_default_ref",
            description="Explicit historical no-schedule runtime default.",
            omit_schedule_flag=False,
            schedule_overrides={},
        ),
        Variant(
            name="promoted_schedule_default",
            description="No explicit schedule flag; relies on the checked-in default.",
            omit_schedule_flag=True,
            schedule_overrides=promoted_schedule,
        ),
        Variant(
            name="explicit_768eq_090_schedule",
            description="Explicit promoted schedule control.",
            omit_schedule_flag=False,
            schedule_overrides=promoted_schedule,
        ),
    ]


def current_runtime_profile() -> dict[str, Any]:
    search_options = arena.build_eval_search_options(
        root_policy_mode=DEFAULT_ROOT_POLICY_MODE,
        tactical_root_bias=DEFAULT_TACTICAL_ROOT_BIAS,
    )
    search_profile = arena.build_search_profile(
        kind="arena_eval",
        player_mode="puct",
        simulations=384,
        c_puct=DEFAULT_RUNTIME_C_PUCT,
        search_options=search_options,
        extra_fields={
            "challenger_simulations": 384,
            "current_simulations": 256,
        },
    )
    return {
        "applicable": False,
        "reason": "direct single-budget runtime path has no challenger:current schedule input",
        "search_profile": search_profile,
    }


def run_runtime_loader_probe(current_path: Path, *, seed: int) -> dict[str, Any]:
    evaluator = arena.ArtifactEvaluator(current_path)
    return arena.evaluate_artifact_position(
        artifact_path=current_path,
        evaluator=evaluator,
        state={
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        },
        simulations=0,
        seed=seed,
        c_puct=DEFAULT_RUNTIME_C_PUCT,
        search_options=arena.build_eval_search_options(
            root_policy_mode=DEFAULT_ROOT_POLICY_MODE,
            tactical_root_bias=DEFAULT_TACTICAL_ROOT_BIAS,
        ),
    )


def tiny_suite_from(source_suite: Path, *, workdir: Path) -> Path:
    entries = load_suite(str(source_suite))[:1]
    tiny_path = workdir / "diagnostics" / "tiny_suite.jsonl"
    tiny_path.parent.mkdir(parents=True, exist_ok=True)
    with tiny_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    return tiny_path


def run_benchmark_report(
    *,
    variant: Variant,
    suite_path: Path,
    current_path: Path,
    workdir: Path,
    budget_pairs: list[tuple[int, int]],
    games_per_opening: int,
    workers: int,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    budget_pairs_csv = ",".join(budget_pair_label(*pair) for pair in budget_pairs)
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/run_opening_suite_seat_benchmark.py"),
        "--workdir",
        str(workdir),
        "--suite",
        str(suite_path),
        "--current",
        str(current_path),
        "--candidates",
        str(current_path),
        "--budget-pairs",
        budget_pairs_csv,
        "--games-per-opening",
        str(games_per_opening),
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--timeout",
        str(timeout),
        "--root-policy-mode",
        DEFAULT_ROOT_POLICY_MODE,
        "--tactical-root-bias",
        str(DEFAULT_TACTICAL_ROOT_BIAS),
        "--c-puct",
        str(DEFAULT_RUNTIME_C_PUCT),
    ]
    if not variant.omit_schedule_flag:
        cmd.extend(["--c-puct-schedule-json", variant.schedule_json()])
    run_command(cmd, timeout=timeout)
    report_path = workdir / "temperature_benchmark_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report["temperature_reports"][0]["seed_reports"][0]["candidate_reports"][0]


def run_gate_report(
    *,
    variant: Variant,
    current_path: Path,
    workdir: Path,
    budget_pairs: list[tuple[int, int]],
    games: int,
    workers: int,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    out_path = workdir / "gate_report.json"
    budget_pairs_csv = ",".join(budget_pair_label(*pair) for pair in budget_pairs)
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
        budget_pairs_csv,
        "--workdir",
        str(workdir),
        "--root-policy-mode",
        DEFAULT_ROOT_POLICY_MODE,
        "--tactical-root-bias",
        str(DEFAULT_TACTICAL_ROOT_BIAS),
        "--c-puct",
        str(DEFAULT_RUNTIME_C_PUCT),
    ]
    if not variant.omit_schedule_flag:
        cmd.extend(["--c-puct-schedule-json", variant.schedule_json()])
    run_command(cmd, timeout=timeout)
    return json.loads(out_path.read_text(encoding="utf-8"))


def benchmark_diagnostics_by_budget(
    candidate_report: dict[str, Any], budget_pairs: list[tuple[int, int]]
) -> dict[str, dict[str, Any]]:
    budget_results = candidate_report.get("budget_results", {})
    diagnostics: dict[str, dict[str, Any]] = {}
    for pair in budget_pairs:
        axis = budget_pair_label(*pair)
        budget_key = BENCHMARK_BUDGET_PAIR_LABELS.get(pair, f"{pair[0]}_vs_{pair[1]}")
        budget_result = budget_results.get(budget_key, {})
        diagnostics[axis] = {
            "effective_c_puct": budget_result.get("effective_c_puct"),
            "tactical_root_bias": budget_result.get("tactical_root_bias"),
            "search_profile": budget_result.get("search_profile"),
            "search_profile_hash": budget_result.get("search_profile_hash"),
        }
    return diagnostics


def normalized_benchmark_results(
    candidate_report: dict[str, Any], budget_pairs: list[tuple[int, int]]
) -> dict[str, dict[str, Any]]:
    budget_results = candidate_report.get("budget_results", {})
    normalized: dict[str, dict[str, Any]] = {}
    for pair in budget_pairs:
        axis = budget_pair_label(*pair)
        budget_key = BENCHMARK_BUDGET_PAIR_LABELS.get(pair, f"{pair[0]}_vs_{pair[1]}")
        normalized[axis] = budget_results.get(budget_key, {})
    return normalized


def gate_diagnostics_by_budget(
    *, current_path: Path, workdir: Path, budget_pairs: list[tuple[int, int]]
) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    for pair in budget_pairs:
        axis = budget_pair_label(*pair)
        budget_key = gate_budget_key(pair)
        arena_path = workdir / current_path.name / budget_key / "alternating_arena.json"
        arena_report = json.loads(arena_path.read_text(encoding="utf-8"))
        notes = (
            arena_report.get("notes", {})
            if isinstance(arena_report.get("notes"), dict)
            else {}
        )
        search_profile = notes.get("search_profile")
        diagnostics[axis] = {
            "effective_c_puct": search_profile.get("c_puct")
            if isinstance(search_profile, dict)
            else None,
            "tactical_root_bias": search_profile.get("search_options", {}).get(
                "tactical_root_bias"
            )
            if isinstance(search_profile, dict)
            else None,
            "search_profile": search_profile,
            "search_profile_hash": notes.get("search_profile_hash"),
        }
    return diagnostics


def gate_standard_profile(current_path: Path, workdir: Path) -> dict[str, Any] | None:
    arena_path = workdir / current_path.name / "standard" / "alternating_arena.json"
    arena_report = json.loads(arena_path.read_text(encoding="utf-8"))
    notes = (
        arena_report.get("notes", {})
        if isinstance(arena_report.get("notes"), dict)
        else {}
    )
    search_profile = notes.get("search_profile")
    return search_profile if isinstance(search_profile, dict) else None


def write_suite_prefixes(path: Path, suite_entries: list[dict[str, Any]]) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in suite_entries:
            handle.write(json.dumps({"prefix_moves": entry["prefix_moves"]}) + "\n")


def run_arena_for_variant(
    *,
    variant: Variant,
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
    axis = budget_pair_label(challenger_sims, current_sims)
    effective_c_puct = variant.effective_cpuct(axis)
    lane_dir = workdir / suite_name / variant.name / axis.replace(":", "_")
    lane_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = lane_dir / "metrics.json"
    arena_json = lane_dir / "arena.json"
    games_jsonl = lane_dir / "games.jsonl"
    suite_jsonl = lane_dir / "opening_suite.jsonl"
    write_suite_prefixes(suite_jsonl, suite_entries)
    total_games = len(suite_entries) * max(1, int(games_per_opening))
    cache_context = {
        "variant": variant.name,
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
        "effective_c_puct": effective_c_puct,
        "tactical_root_bias": DEFAULT_TACTICAL_ROOT_BIAS,
        "schedule": variant.schedule_manifest(),
    }
    shared_cache_context = {
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
        "effective_c_puct": effective_c_puct,
        "tactical_root_bias": DEFAULT_TACTICAL_ROOT_BIAS,
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
        DEFAULT_ROOT_POLICY_MODE,
        "--tactical-root-bias",
        str(DEFAULT_TACTICAL_ROOT_BIAS),
        "--c-puct",
        str(effective_c_puct),
    ]
    run_command(cmd, timeout=timeout)
    arena_report = json.loads(arena_json.read_text(encoding="utf-8"))
    game_entries = parse_game_jsonl(str(games_jsonl))
    seat_metrics = compute_seat_metrics(game_entries)
    notes = (
        arena_report.get("notes", {})
        if isinstance(arena_report.get("notes"), dict)
        else {}
    )
    metrics = {
        "cache_context": cache_context,
        "suite_name": suite_name,
        "variant": variant.name,
        "budget_pair": axis,
        "p0_score": seat_metrics["p0_score"],
        "p1_score": seat_metrics["p1_score"],
        "ds": seat_metrics["ds"],
        "total_games": seat_metrics["total_games"],
        "duplicate_trajectory_rate": seat_metrics["duplicate_trajectory_rate"],
        "duplicate_trajectory_count": seat_metrics["duplicate_trajectory_count"],
        "move_time_mean_ms": notes.get("move_time_mean_ms"),
        "move_time_p95_ms": notes.get("move_time_p95_ms"),
        "search_profile": notes.get("search_profile"),
        "search_profile_hash": notes.get("search_profile_hash"),
        "arena_score": arena_report.get("score"),
        "effective_c_puct": effective_c_puct,
        "tactical_root_bias": DEFAULT_TACTICAL_ROOT_BIAS,
    }
    shared_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    shared_metrics = dict(metrics)
    shared_metrics["cache_context"] = shared_cache_context
    write_json(shared_metrics_path, shared_metrics)
    write_json(metrics_path, metrics)
    return metrics


def evaluate_suite(
    *,
    suite_name: str,
    suite_path: Path,
    current_path: Path,
    variant_list: list[Variant],
    budget_pairs: list[tuple[int, int]],
    workdir: Path,
    games_per_opening: int,
    workers: int,
    seed: int,
    timeout: int,
) -> dict[str, dict[str, Any]]:
    suite_entries = load_suite(str(suite_path))
    results: dict[str, dict[str, Any]] = {}
    for variant in variant_list:
        results[variant.name] = {}
        for pair in budget_pairs:
            axis = budget_pair_label(*pair)
            results[variant.name][axis] = run_arena_for_variant(
                variant=variant,
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
    return results


def suite_table_rows(
    suite_results: dict[str, dict[str, Any]],
    variant_list: list[Variant],
    budget_labels: list[str],
) -> list[list[Any]]:
    rows = []
    for variant in variant_list:
        rows.append(
            [variant.name]
            + [fmt(suite_results[variant.name][label]["ds"]) for label in budget_labels]
        )
    return rows


def heldout_all_suite_rows(
    heldout_results: dict[str, dict[str, dict[str, Any]]],
    variant_list: list[Variant],
    budget_labels: list[str],
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for suite_name, results in heldout_results.items():
        for variant in variant_list:
            rows.append(
                [suite_name, variant.name]
                + [fmt(results[variant.name][label]["ds"]) for label in budget_labels]
            )
    return rows


def mean_budget_summary(
    suite_results_by_name: dict[str, dict[str, dict[str, Any]]],
    variant_list: list[Variant],
    budget_labels: list[str],
) -> dict[str, dict[str, dict[str, float | None]]]:
    summary: dict[str, dict[str, dict[str, float | None]]] = {}
    for variant in variant_list:
        summary[variant.name] = {}
        for label in budget_labels:
            ds_values = [
                float(results[variant.name][label]["ds"])
                for results in suite_results_by_name.values()
            ]
            move_means = [
                float(results[variant.name][label]["move_time_mean_ms"])
                for results in suite_results_by_name.values()
                if results[variant.name][label]["move_time_mean_ms"] is not None
            ]
            move_p95 = [
                float(results[variant.name][label]["move_time_p95_ms"])
                for results in suite_results_by_name.values()
                if results[variant.name][label]["move_time_p95_ms"] is not None
            ]
            summary[variant.name][label] = {
                "mean_ds": statistics.fmean(ds_values) if ds_values else None,
                "mean_move_time_ms": statistics.fmean(move_means)
                if move_means
                else None,
                "mean_move_time_p95_ms": statistics.fmean(move_p95)
                if move_p95
                else None,
            }
    return summary


def compare_variants(
    suite_results_by_name: dict[str, dict[str, dict[str, Any]]],
    *,
    left_name: str,
    right_name: str,
    budget_labels: list[str],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for suite_name, results in suite_results_by_name.items():
        for label in budget_labels:
            left = results[left_name][label]
            right = results[right_name][label]
            if left.get("search_profile_hash") != right.get(
                "search_profile_hash"
            ) or any(
                left.get(field) != right.get(field)
                for field in (
                    "ds",
                    "p0_score",
                    "p1_score",
                    "arena_score",
                    "duplicate_trajectory_count",
                )
            ):
                mismatches.append(
                    {
                        "suite": suite_name,
                        "budget_pair": label,
                        left_name: left,
                        right_name: right,
                    }
                )
    return {"matches": not mismatches, "mismatches": mismatches}


def compare_non_768_budget_stability(
    suite_results_by_name: dict[str, dict[str, dict[str, Any]]],
    *,
    baseline_name: str,
    candidate_name: str,
    budget_labels: list[str],
) -> dict[str, Any]:
    stable_labels = [label for label in budget_labels if label != "768:768"]
    return compare_variants(
        suite_results_by_name,
        left_name=baseline_name,
        right_name=candidate_name,
        budget_labels=stable_labels,
    )


def improvement_checks(
    *,
    fixed_large_results: dict[str, dict[str, Any]],
    heldout_mean_summary: dict[str, dict[str, dict[str, float | None]]],
) -> dict[str, Any]:
    main_labels = ["384:256", "768:768", "1200:1200", "1200:256"]
    fixed_large: dict[str, dict[str, Any]] = {}
    heldout_means: dict[str, dict[str, Any]] = {}
    fixed_large_ok = True
    heldout_ok = True
    for label in main_labels:
        old_fixed = float(
            fixed_large_results["old_no_schedule_default_ref"][label]["ds"]
        )
        new_fixed = float(fixed_large_results["promoted_schedule_default"][label]["ds"])
        fixed_pass = (
            new_fixed > old_fixed if label == "768:768" else new_fixed == old_fixed
        )
        fixed_large[label] = {
            "old_no_schedule_default_ref": old_fixed,
            "promoted_schedule_default": new_fixed,
            "passed": fixed_pass,
        }
        fixed_large_ok = fixed_large_ok and fixed_pass

        old_mean = heldout_mean_summary["old_no_schedule_default_ref"][label]["mean_ds"]
        new_mean = heldout_mean_summary["promoted_schedule_default"][label]["mean_ds"]
        if old_mean is None or new_mean is None:
            heldout_pass = False
        else:
            heldout_pass = (
                new_mean > old_mean if label == "768:768" else new_mean == old_mean
            )
        heldout_means[label] = {
            "old_no_schedule_default_ref": old_mean,
            "promoted_schedule_default": new_mean,
            "passed": heldout_pass,
        }
        heldout_ok = heldout_ok and heldout_pass
    return {
        "main_budget_pairs": main_labels,
        "fixed_large": fixed_large,
        "heldout_means": heldout_means,
        "passed": fixed_large_ok and heldout_ok,
    }


def default_schedule_active(
    diagnostics: dict[str, dict[str, Any]],
    expected_schedule: dict[str, float],
    budget_labels: list[str],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for path_name, per_budget in diagnostics.items():
        for label in budget_labels:
            entry = per_budget.get(label, {})
            search_profile = entry.get("search_profile")
            expected_cpuct = resolve_budget_cpuct(
                schedule=expected_schedule,
                challenger_simulations=parse_budget_pair(label)[0],
                current_simulations=parse_budget_pair(label)[1],
                default_c_puct=DEFAULT_RUNTIME_C_PUCT,
            )
            actual_cpuct = None
            actual_bias = None
            if isinstance(search_profile, dict):
                actual_cpuct = search_profile.get("c_puct")
                actual_bias = search_profile.get("search_options", {}).get(
                    "tactical_root_bias"
                )
            if (
                actual_cpuct != expected_cpuct
                or actual_bias != DEFAULT_TACTICAL_ROOT_BIAS
            ):
                issues.append(
                    {
                        "path": path_name,
                        "budget_pair": label,
                        "expected_c_puct": expected_cpuct,
                        "actual_c_puct": actual_cpuct,
                        "actual_tactical_root_bias": actual_bias,
                    }
                )
    return {"passed": not issues, "issues": issues}


def runtime_cost_summary(
    heldout_mean_summary: dict[str, dict[str, dict[str, float | None]]],
    budget_labels: list[str],
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    baseline = []
    for label in budget_labels:
        value = heldout_mean_summary["old_no_schedule_default_ref"][label][
            "mean_move_time_ms"
        ]
        if value is not None:
            baseline.append(float(value))
    baseline_mean = statistics.fmean(baseline) if baseline else None
    for variant_name in (
        "old_no_schedule_default_ref",
        "promoted_schedule_default",
        "explicit_768eq_090_schedule",
    ):
        means = []
        p95s = []
        for label in budget_labels:
            mean_value = heldout_mean_summary[variant_name][label]["mean_move_time_ms"]
            p95_value = heldout_mean_summary[variant_name][label][
                "mean_move_time_p95_ms"
            ]
            if mean_value is not None:
                means.append(float(mean_value))
            if p95_value is not None:
                p95s.append(float(p95_value))
        mean_latency = statistics.fmean(means) if means else None
        mean_p95 = statistics.fmean(p95s) if p95s else None
        slowdown = None
        if baseline_mean not in (None, 0.0) and mean_latency is not None:
            slowdown = (mean_latency / baseline_mean) - 1.0
        rows.append(
            [
                variant_name,
                fmt(mean_latency, digits=2),
                fmt(mean_p95, digits=2),
                fmt(slowdown, digits=3),
            ]
        )
    return rows


def classify_summary(summary: dict[str, Any]) -> str:
    checks = summary["checks"]
    if (
        not checks["artifact_integrity"]["passed"]
        or not checks["runtime_loader"]["passed"]
        or not checks["metadata_parse"]["passed"]
    ):
        return "promotion_blocked_runtime_failure"
    if not checks["default_profile"]["default_active"]:
        return "promotion_blocked_default_not_active"
    if (
        not checks["evaluation_match"]["passed"]
        or not checks["non_768_budget_stability"]["passed"]
        or not checks["improvement_direction"]["passed"]
    ):
        return "promotion_blocked_eval_mismatch"
    if not checks["gate"]["passed"]:
        return "promotion_blocked_gate_regression"
    return "promoted_cpuct_768eq_090_schedule_default"


def render_effective_profile_diagnostics(
    path_name: str, diagnostics: dict[str, dict[str, Any]], budget_labels: list[str]
) -> str:
    rows = []
    for label in budget_labels:
        entry = diagnostics.get(label, {})
        rows.append(
            [
                path_name,
                label,
                fmt(entry.get("effective_c_puct"), digits=2),
                fmt(entry.get("search_profile", {}).get("c_puct"), digits=2)
                if isinstance(entry.get("search_profile"), dict)
                else "N/A",
                fmt(entry.get("tactical_root_bias"), digits=2),
                entry.get("search_profile_hash"),
            ]
        )
    return markdown_table(
        [
            "Path",
            "Budget",
            "effective c_puct",
            "profile c_puct",
            "tactical_root_bias",
            "search_profile_hash",
        ],
        rows,
    )


def render_report(summary: dict[str, Any]) -> str:
    inputs = summary["inputs"]
    fixed_large_rows = suite_table_rows(
        summary["fixed_large_results"],
        variants(summary["new_default_search_profile"]["c_puct_schedule"]),
        inputs["budget_pairs"],
    )
    heldout_rows = heldout_all_suite_rows(
        summary["heldout_results"],
        variants(summary["new_default_search_profile"]["c_puct_schedule"]),
        inputs["budget_pairs"],
    )
    return "\n".join(
        [
            "# AlphaZero-Lite c_puct 768:768 0.90 Schedule Promotion Results",
            "",
            f"**Date**: {summary['date']}",
            "",
            f"**Classification**: `{summary['classification']}`",
            "",
            "## Inputs",
            "",
            f"- Current artifact: `{inputs['current']['path']}`",
            f"- Unchanged artifact weights SHA256: `{inputs['current']['actual_sha256']}`",
            f"- Expected weights SHA256: `{inputs['current']['expected_sha256']}`",
            f"- Large suite: `{inputs['large_suite']['path']}`",
            "- Held-out suites:",
            *[f"- `{row['path']}`" for row in inputs["heldout_suites"]],
            f"- Budget pairs: `{','.join(inputs['budget_pairs'])}`",
            f"- Games per opening: `{inputs['games_per_opening']}`",
            f"- Workers: `{inputs['workers']}`",
            f"- Seed: `{inputs['seed']}`",
            "",
            "## Runtime Profiles",
            "",
            f"- Old runtime profile: `{json.dumps(summary['old_default_search_profile'], sort_keys=True)}`",
            f"- New runtime profile: `{json.dumps(summary['new_default_search_profile'], sort_keys=True)}`",
            "",
            "## Files Changed",
            "",
            *[f"- `{path}`" for path in summary["files_changed"]],
            "",
            "## Effective-Profile Diagnostics",
            "",
            render_effective_profile_diagnostics(
                "opening_suite_benchmark_path",
                summary["diagnostics"]["opening_suite_benchmark_path"],
                inputs["budget_pairs"],
            ),
            "",
            render_effective_profile_diagnostics(
                "seat_aware_gate_path",
                summary["diagnostics"]["seat_aware_gate_path"],
                inputs["budget_pairs"],
            ),
            "",
            f"- Runtime/app path applicability: `{summary['diagnostics']['runtime_app_path']['applicable']}`",
            f"- Runtime/app path note: `{summary['diagnostics']['runtime_app_path']['reason']}`",
            "",
            "## Fixed Large Before/After",
            "",
            markdown_table(["Variant", *inputs["budget_pairs"]], fixed_large_rows),
            "",
            "## Held-Out All-Suite Table",
            "",
            markdown_table(["Suite", "Variant", *inputs["budget_pairs"]], heldout_rows),
            "",
            "## Gate Result",
            "",
            f"- Default gate classification: `{summary['gate']['report'].get('classification')}`",
            f"- Default gate schedule manifest: `{json.dumps(summary['gate']['report'].get('c_puct_schedule'), sort_keys=True)}`",
            f"- Standard budget effective profile: `{json.dumps(summary['gate'].get('standard_arena_profile'), sort_keys=True)}`",
            "",
            "## Runtime Cost Comparison",
            "",
            markdown_table(
                [
                    "Variant",
                    "Mean move latency ms",
                    "Mean p95 latency ms",
                    "Relative slowdown",
                ],
                runtime_cost_summary(
                    summary["heldout_mean_summary"], inputs["budget_pairs"]
                ),
            ),
            "",
            "## Decision Summary",
            "",
            f"- Model weights unchanged: `{summary['checks']['artifact_integrity']['passed']}`",
            f"- Default benchmark/gate paths activate the promoted schedule: `{summary['checks']['default_profile']['default_active']}`",
            f"- Old no-schedule global 1.25 remains reproducible: `{summary['checks']['non_768_budget_stability']['passed']}`",
            f"- Promoted default equals explicit 768:768=0.90 schedule: `{summary['checks']['evaluation_match']['passed']}`",
            f"- 768:768 improvement direction reproduced: `{summary['checks']['improvement_direction']['passed']}`",
            f"- Gate preserved: `{summary['checks']['gate']['passed']}`",
            f"- Final classification: `{summary['classification']}`",
        ]
    )


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    current_path = Path(args.current)
    current_weights = current_path / "weights.json"
    current_metadata = current_path / "metadata.json"
    require_existing_file(current_weights, "current weights")
    require_existing_file(current_metadata, "current metadata")
    large_suite = Path(args.large_suite)
    heldout_suites = parse_csv_paths(args.heldout_suites)
    require_existing_file(large_suite, "large suite")
    for suite_path in heldout_suites:
        require_existing_file(suite_path, f"heldout suite {suite_path.name}")

    promoted_schedule = parse_cpuct_schedule_json(args.schedule)
    if promoted_schedule != default_runtime_schedule_definition()["overrides"]:
        raise ValueError(
            "--schedule must match the checked-in promoted default schedule exactly"
        )
    variant_list = variants(promoted_schedule)
    budget_pairs = parse_budget_pairs(args.budget_pairs)
    budget_labels = [budget_pair_label(*pair) for pair in budget_pairs]
    diagnostic_gate_budget_pairs = budget_pairs

    current_summary = verify_expected_hash(
        current_weights,
        args.expected_current_weights_sha256,
        "current artifact weights",
    )
    metadata_payload = json.loads(current_metadata.read_text(encoding="utf-8"))
    runtime_probe = run_runtime_loader_probe(current_path, seed=args.seed)

    tiny_suite = tiny_suite_from(large_suite, workdir=workdir)
    benchmark_default_diag = run_benchmark_report(
        variant=variant_list[1],
        suite_path=tiny_suite,
        current_path=current_path,
        workdir=workdir / "diagnostics" / "benchmark_default",
        budget_pairs=budget_pairs,
        games_per_opening=2,
        workers=1,
        seed=args.seed,
        timeout=1200,
    )
    benchmark_old_diag = run_benchmark_report(
        variant=variant_list[0],
        suite_path=tiny_suite,
        current_path=current_path,
        workdir=workdir / "diagnostics" / "benchmark_old_no_schedule",
        budget_pairs=budget_pairs,
        games_per_opening=2,
        workers=1,
        seed=args.seed,
        timeout=1200,
    )
    run_gate_report(
        variant=variant_list[1],
        current_path=current_path,
        workdir=workdir / "diagnostics" / "gate_default",
        budget_pairs=diagnostic_gate_budget_pairs,
        games=4,
        workers=1,
        seed=args.seed,
        timeout=1200,
    )
    run_gate_report(
        variant=variant_list[0],
        current_path=current_path,
        workdir=workdir / "diagnostics" / "gate_old_no_schedule",
        budget_pairs=diagnostic_gate_budget_pairs,
        games=4,
        workers=1,
        seed=args.seed,
        timeout=1200,
    )
    diagnostics = {
        "runtime_app_path": current_runtime_profile(),
        "opening_suite_benchmark_path": benchmark_diagnostics_by_budget(
            benchmark_default_diag, budget_pairs
        ),
        "opening_suite_benchmark_old_no_schedule": benchmark_diagnostics_by_budget(
            benchmark_old_diag, budget_pairs
        ),
        "seat_aware_gate_path": gate_diagnostics_by_budget(
            current_path=current_path,
            workdir=workdir / "diagnostics" / "gate_default",
            budget_pairs=diagnostic_gate_budget_pairs,
        ),
        "seat_aware_gate_old_no_schedule": gate_diagnostics_by_budget(
            current_path=current_path,
            workdir=workdir / "diagnostics" / "gate_old_no_schedule",
            budget_pairs=diagnostic_gate_budget_pairs,
        ),
    }

    fixed_large_results = evaluate_suite(
        suite_name="large_suite",
        suite_path=large_suite,
        current_path=current_path,
        variant_list=variant_list,
        budget_pairs=budget_pairs,
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
            variant_list=variant_list,
            budget_pairs=budget_pairs,
            workdir=workdir,
            games_per_opening=args.games_per_opening,
            workers=args.workers,
            seed=args.seed,
            timeout=args.timeout,
        )
    heldout_mean_summary = mean_budget_summary(
        heldout_results, variant_list, budget_labels
    )
    suite_results_all = {"large_suite": fixed_large_results, **heldout_results}
    evaluation_match = compare_variants(
        suite_results_all,
        left_name="promoted_schedule_default",
        right_name="explicit_768eq_090_schedule",
        budget_labels=budget_labels,
    )
    non_768_budget_stability = compare_non_768_budget_stability(
        suite_results_all,
        baseline_name="old_no_schedule_default_ref",
        candidate_name="promoted_schedule_default",
        budget_labels=budget_labels,
    )
    improvement_direction = improvement_checks(
        fixed_large_results=fixed_large_results,
        heldout_mean_summary=heldout_mean_summary,
    )
    default_profile = default_schedule_active(
        {
            "opening_suite_benchmark_path": diagnostics["opening_suite_benchmark_path"],
            "seat_aware_gate_path": diagnostics["seat_aware_gate_path"],
        },
        promoted_schedule,
        budget_labels,
    )

    gate = run_gate_report(
        variant=variant_list[1],
        current_path=current_path,
        workdir=workdir / "gate",
        budget_pairs=parse_budget_pairs(DEFAULT_GATE_BUDGET_PAIRS),
        games=args.gate_games,
        workers=args.workers,
        seed=args.seed,
        timeout=args.timeout,
    )
    gate_manifest = gate.get("c_puct_schedule")
    gate_classification = gate.get("classification")
    gate_passed = (
        gate_classification in ACCEPTED_GATE_CLASSIFICATIONS
        and gate_manifest == default_runtime_schedule_definition()
    )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "date": str(date.today()),
        "inputs": {
            "current": current_summary,
            "large_suite": build_input_summary(large_suite),
            "heldout_suites": [build_input_summary(path) for path in heldout_suites],
            "budget_pairs": budget_labels,
            "games_per_opening": int(args.games_per_opening),
            "workers": int(args.workers),
            "seed": int(args.seed),
        },
        "files_changed": FILES_CHANGED,
        "old_default_search_profile": {
            "search_mode": DEFAULT_SEARCH_MODE,
            "root_policy_mode": DEFAULT_ROOT_POLICY_MODE,
            "c_puct": float(DEFAULT_RUNTIME_C_PUCT),
            "c_puct_schedule": {},
            "tactical_root_bias": float(DEFAULT_TACTICAL_ROOT_BIAS),
            "root_prior_transform": None,
        },
        "new_default_search_profile": {
            "search_mode": DEFAULT_SEARCH_MODE,
            "root_policy_mode": DEFAULT_ROOT_POLICY_MODE,
            "c_puct": float(DEFAULT_RUNTIME_C_PUCT),
            "c_puct_schedule": promoted_schedule,
            "tactical_root_bias": float(DEFAULT_TACTICAL_ROOT_BIAS),
            "root_prior_transform": None,
        },
        "runtime_probe": {
            "selected_move": runtime_probe.get("selected_move"),
            "search_options": runtime_probe.get("search_options"),
        },
        "diagnostics": diagnostics,
        "fixed_large_results": fixed_large_results,
        "heldout_results": heldout_results,
        "heldout_mean_summary": heldout_mean_summary,
        "gate": {
            "report": gate,
            "standard_arena_profile": gate_standard_profile(
                current_path, workdir / "gate"
            ),
        },
        "checks": {
            "artifact_integrity": {
                "passed": current_summary["actual_sha256"]
                == args.expected_current_weights_sha256,
            },
            "runtime_loader": {
                "passed": runtime_probe.get("selected_move") is not None,
            },
            "metadata_parse": {
                "passed": isinstance(metadata_payload, dict)
                and metadata_payload.get("schema_version") is not None,
            },
            "default_profile": {
                "default_active": default_profile["passed"],
                "details": default_profile,
            },
            "evaluation_match": {
                "passed": evaluation_match["matches"],
                "details": evaluation_match,
            },
            "non_768_budget_stability": {
                "passed": non_768_budget_stability["matches"],
                "details": non_768_budget_stability,
            },
            "improvement_direction": improvement_direction,
            "gate": {
                "passed": gate_passed,
                "classification": gate_classification,
                "manifest": gate_manifest,
            },
        },
    }
    summary["classification"] = classify_summary(summary)

    summary_path = workdir / SUMMARY_FILENAME
    write_json(summary_path, summary)
    REPORT_PATH.write_text(render_report(summary) + "\n", encoding="utf-8")
    print(f"classification={summary['classification']}")
    print(f"summary={summary_path}")
    print(f"report={REPORT_PATH}")
    return (
        0
        if summary["classification"] == "promoted_cpuct_768eq_090_schedule_default"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
