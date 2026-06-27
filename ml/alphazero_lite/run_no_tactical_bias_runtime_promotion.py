#!/usr/bin/env python3
"""Guarded promotion of the no-tactical-bias runtime search profile."""

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

from ml.alphazero_lite import arena  # noqa: E402
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
from ml.alphazero_lite.self_play import DEFAULT_EVAL_SEARCH_OPTIONS  # noqa: E402


SUMMARY_SCHEMA = "azlite_no_tactical_bias_runtime_promotion_v1"
REPORT_PATH = (
    REPO_ROOT / "docs/alphazero-lite-no-tactical-bias-runtime-promotion-results.md"
)
SUMMARY_FILENAME = "summary_metrics.json"
DEFAULT_BUDGET_PAIRS = "384:256,768:256,768:768,1200:1200,1200:256,256:768"
DEFAULT_GATE_BUDGET_PAIRS = "384:256,1200:1200,1200:256,256:768"
DEFAULT_SEARCH_MODE = "full"
DEFAULT_ROOT_POLICY_MODE = "deterministic"
DEFAULT_C_PUCT = 1.25
OLD_DEFAULT_TACTICAL_ROOT_BIAS = 0.1
PROMOTED_TACTICAL_ROOT_BIAS = 0.0
ACCEPTED_GATE_CLASSIFICATIONS = {
    "high_search_breakthrough",
    "standard_budget_breakthrough",
}
FILES_CHANGED = [
    "ml/alphazero_lite/self_play.py",
    "ml/alphazero_lite/arena.py",
    "ml/alphazero_lite/benchmark.py",
    "ml/alphazero_lite/test_benchmark.py",
    "ml/alphazero_lite/run_opening_suite_seat_benchmark.py",
    "script/ai/seat_aware_promotion_gate",
    "ml/alphazero_lite/run_no_tactical_bias_runtime_promotion.py",
    "docs/alphazero-lite-no-tactical-bias-runtime-promotion-results.md",
]


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    cli_tactical_root_bias: float | None

    @property
    def effective_tactical_root_bias(self) -> float:
        if self.cli_tactical_root_bias is None:
            return float(DEFAULT_EVAL_SEARCH_OPTIONS["tactical_root_bias"])
        return float(self.cli_tactical_root_bias)

    def search_profile(self) -> dict[str, Any]:
        return {
            "search_mode": DEFAULT_SEARCH_MODE,
            "root_policy_mode": DEFAULT_ROOT_POLICY_MODE,
            "c_puct": float(DEFAULT_C_PUCT),
            "tactical_root_bias": self.effective_tactical_root_bias,
            "root_prior_transform": None,
            "value_trust_schedule": None,
        }


def _python() -> str:
    candidate = REPO_ROOT / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="/tmp/azlite_no_tactical_bias_runtime_promotion"
    )
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--budget-pairs", default=DEFAULT_BUDGET_PAIRS)
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


def variants() -> list[Variant]:
    return [
        Variant(
            name="old_default_ref",
            description="Explicit historical baseline with tactical_root_bias=0.1.",
            cli_tactical_root_bias=OLD_DEFAULT_TACTICAL_ROOT_BIAS,
        ),
        Variant(
            name="promoted_default",
            description="No tactical-root-bias flag; relies on the checked-in default.",
            cli_tactical_root_bias=None,
        ),
        Variant(
            name="explicit_no_tactical_bias",
            description="Explicit tactical_root_bias=0.0 control.",
            cli_tactical_root_bias=PROMOTED_TACTICAL_ROOT_BIAS,
        ),
    ]


def write_suite_prefixes(path: Path, suite_entries: list[dict[str, Any]]) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in suite_entries:
            handle.write(json.dumps({"prefix_moves": entry["prefix_moves"]}) + "\n")


def run_command(
    cmd: list[str], *, timeout: int, env: dict[str, str] | None = None
) -> None:
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        raise RuntimeError(
            stderr or stdout or f"command failed: {' '.join(str(part) for part in cmd)}"
        )


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
    label = budget_pair_label(challenger_sims, current_sims)
    lane_dir = workdir / suite_name / variant.name / label.replace(":", "_")
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
        "c_puct": float(DEFAULT_C_PUCT),
        "root_policy_mode": DEFAULT_ROOT_POLICY_MODE,
        "explicit_tactical_root_bias": variant.cli_tactical_root_bias,
        "effective_tactical_root_bias": variant.effective_tactical_root_bias,
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
        DEFAULT_ROOT_POLICY_MODE,
        "--c-puct",
        str(DEFAULT_C_PUCT),
    ]
    if variant.cli_tactical_root_bias is not None:
        cmd.extend(["--tactical-root-bias", str(variant.cli_tactical_root_bias)])
    run_command(cmd, timeout=timeout)

    arena_report = json.loads(arena_json.read_text(encoding="utf-8"))
    game_entries = parse_game_jsonl(str(games_jsonl))
    seat_metrics = compute_seat_metrics(game_entries)
    notes = (
        arena_report.get("notes", {})
        if isinstance(arena_report.get("notes"), dict)
        else {}
    )
    search_profile = (
        notes.get("search_profile")
        if isinstance(notes.get("search_profile"), dict)
        else {}
    )
    metrics = {
        "cache_context": cache_context,
        "suite_name": suite_name,
        "variant": variant.name,
        "budget_pair": label,
        "p0_score": seat_metrics["p0_score"],
        "p1_score": seat_metrics["p1_score"],
        "ds": seat_metrics["ds"],
        "total_games": seat_metrics["total_games"],
        "duplicate_trajectory_rate": seat_metrics["duplicate_trajectory_rate"],
        "duplicate_trajectory_count": seat_metrics["duplicate_trajectory_count"],
        "move_time_mean_ms": notes.get("move_time_mean_ms"),
        "move_time_p95_ms": notes.get("move_time_p95_ms"),
        "search_profile": search_profile,
        "search_profile_hash": notes.get("search_profile_hash"),
        "arena_score": arena_report.get("score"),
    }
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
            label = budget_pair_label(*pair)
            results[variant.name][label] = run_arena_for_variant(
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


def compare_promoted_and_explicit(
    suite_results_by_name: dict[str, dict[str, dict[str, Any]]],
    budget_labels: list[str],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for suite_name, results in suite_results_by_name.items():
        for label in budget_labels:
            promoted = results["promoted_default"][label]
            explicit = results["explicit_no_tactical_bias"][label]
            if promoted["search_profile_hash"] != explicit[
                "search_profile_hash"
            ] or any(
                promoted[field] != explicit[field]
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
                        "promoted_default": promoted,
                        "explicit_no_tactical_bias": explicit,
                    }
                )
    return {
        "matches": not mismatches,
        "mismatches": mismatches,
    }


def improvement_checks(
    *,
    fixed_large_results: dict[str, dict[str, Any]],
    heldout_mean_summary: dict[str, dict[str, dict[str, float | None]]],
) -> dict[str, Any]:
    main_labels = ["384:256", "768:768", "1200:1200", "1200:256"]
    fixed_large = {}
    heldout_means = {}
    fixed_large_ok = True
    heldout_ok = True
    for label in main_labels:
        old_fixed = float(fixed_large_results["old_default_ref"][label]["ds"])
        new_fixed = float(fixed_large_results["promoted_default"][label]["ds"])
        fixed_pass = (
            new_fixed >= old_fixed if label == "768:768" else new_fixed > old_fixed
        )
        fixed_large[label] = {
            "old_default_ref": old_fixed,
            "promoted_default": new_fixed,
            "passed": fixed_pass,
        }
        fixed_large_ok = fixed_large_ok and fixed_pass

        old_mean = heldout_mean_summary["old_default_ref"][label]["mean_ds"]
        new_mean = heldout_mean_summary["promoted_default"][label]["mean_ds"]
        if old_mean is None or new_mean is None:
            heldout_pass = False
        else:
            heldout_pass = (
                new_mean >= old_mean if label == "768:768" else new_mean > old_mean
            )
        heldout_means[label] = {
            "old_default_ref": old_mean,
            "promoted_default": new_mean,
            "passed": heldout_pass,
        }
        heldout_ok = heldout_ok and heldout_pass
    return {
        "main_budget_pairs": main_labels,
        "fixed_large": fixed_large,
        "heldout_means": heldout_means,
        "passed": fixed_large_ok and heldout_ok,
    }


def current_runtime_profile() -> dict[str, Any]:
    search_options = arena.build_eval_search_options(
        root_policy_mode=DEFAULT_ROOT_POLICY_MODE
    )
    search_profile = arena.build_search_profile(
        kind="arena_eval",
        player_mode="puct",
        simulations=384,
        c_puct=DEFAULT_C_PUCT,
        search_options=search_options,
        extra_fields={
            "challenger_simulations": 384,
            "current_simulations": 256,
        },
    )
    return {
        "search_options": search_options,
        "search_profile": search_profile,
    }


def run_arena_profile_diagnostic(
    *, current_path: Path, workdir: Path, tactical_root_bias: float | None
) -> dict[str, Any]:
    out_path = workdir / "arena_report.json"
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/arena.py"),
        "--challenger",
        str(current_path),
        "--current",
        str(current_path),
        "--games",
        "2",
        "--challenger-simulations",
        "384",
        "--current-simulations",
        "256",
        "--seed",
        "42",
        "--workers",
        "1",
        "--min-score",
        "0.0",
        "--out",
        str(out_path),
        "--root-policy-mode",
        DEFAULT_ROOT_POLICY_MODE,
        "--c-puct",
        str(DEFAULT_C_PUCT),
    ]
    if tactical_root_bias is not None:
        cmd.extend(["--tactical-root-bias", str(tactical_root_bias)])
    run_command(cmd, timeout=1200)
    report = json.loads(out_path.read_text(encoding="utf-8"))
    notes = report.get("notes", {}) if isinstance(report.get("notes"), dict) else {}
    return {
        "explicit_tactical_root_bias": tactical_root_bias,
        "search_profile": notes.get("search_profile"),
        "search_profile_hash": notes.get("search_profile_hash"),
    }


def run_opening_benchmark_diagnostic(
    *,
    current_path: Path,
    suite_path: Path,
    workdir: Path,
    tactical_root_bias: float | None,
) -> dict[str, Any]:
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
        "384:256",
        "--games-per-opening",
        "2",
        "--seed",
        "42",
        "--workers",
        "1",
        "--timeout",
        "1200",
    ]
    if tactical_root_bias is not None:
        cmd.extend(["--tactical-root-bias", str(tactical_root_bias)])
    run_command(cmd, timeout=1200)
    candidate_dir = workdir / "temp_0_0" / "seed_42" / current_path.name / "standard"
    arena_path = candidate_dir / "starts_0" / "arena.json"
    report = json.loads(arena_path.read_text(encoding="utf-8"))
    notes = report.get("notes", {}) if isinstance(report.get("notes"), dict) else {}
    return {
        "explicit_tactical_root_bias": tactical_root_bias,
        "search_profile": notes.get("search_profile"),
        "search_profile_hash": notes.get("search_profile_hash"),
        "arena_path": str(arena_path),
    }


def run_gate_diagnostic(
    *,
    current_path: Path,
    workdir: Path,
    tactical_root_bias: float | None,
) -> dict[str, Any]:
    out_path = workdir / "gate_report.json"
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
        "4",
        "--seed",
        "42",
        "--workers",
        "1",
        "--budget-pairs",
        "384:256",
        "--workdir",
        str(workdir),
    ]
    if tactical_root_bias is not None:
        cmd.extend(["--tactical-root-bias", str(tactical_root_bias)])
    run_command(cmd, timeout=1200)
    arena_path = workdir / current_path.name / "standard" / "alternating_arena.json"
    report = json.loads(arena_path.read_text(encoding="utf-8"))
    notes = report.get("notes", {}) if isinstance(report.get("notes"), dict) else {}
    return {
        "explicit_tactical_root_bias": tactical_root_bias,
        "search_profile": notes.get("search_profile"),
        "search_profile_hash": notes.get("search_profile_hash"),
        "arena_path": str(arena_path),
    }


def run_full_gate(
    *, current_path: Path, workdir: Path, workers: int, seed: int, games: int
) -> dict[str, Any]:
    out_path = workdir / "gate_report.json"
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
        "--workdir",
        str(workdir),
    ]
    run_command(cmd, timeout=14400)
    report = json.loads(out_path.read_text(encoding="utf-8"))
    arena_path = workdir / current_path.name / "standard" / "alternating_arena.json"
    arena_report = json.loads(arena_path.read_text(encoding="utf-8"))
    notes = (
        arena_report.get("notes", {})
        if isinstance(arena_report.get("notes"), dict)
        else {}
    )
    return {
        "report": report,
        "standard_arena_profile": notes.get("search_profile"),
        "standard_arena_profile_hash": notes.get("search_profile_hash"),
    }


def render_profile_diagnostics(diagnostics: dict[str, Any]) -> str:
    rows = []
    for path_name in (
        "default_runtime_path",
        "opening_suite_benchmark_path",
        "seat_aware_gate_path",
    ):
        default_entry = diagnostics[path_name]["default"]
        explicit_entry = diagnostics[path_name]["explicit_old_default_ref"]
        rows.append(
            [
                path_name,
                json.dumps(default_entry["search_profile"], sort_keys=True),
                json.dumps(explicit_entry["search_profile"], sort_keys=True),
            ]
        )
    return markdown_table(
        ["Path", "Default effective profile", "Explicit tactical_root_bias=0.1"],
        rows,
    )


def runtime_cost_rows(
    heldout_means: dict[str, dict[str, dict[str, float | None]]],
) -> list[list[Any]]:
    rows = []
    for name in ("old_default_ref", "promoted_default", "explicit_no_tactical_bias"):
        runtime = heldout_means[name]["384:256"]
        rows.append(
            [
                name,
                fmt(runtime["mean_move_time_ms"], digits=2),
                fmt(runtime["mean_move_time_p95_ms"], digits=2),
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
        or not checks["improvement_direction"]["passed"]
    ):
        return "promotion_blocked_eval_mismatch"
    if not checks["gate"]["passed"]:
        return "promotion_blocked_gate_regression"
    return "promoted_no_tactical_bias_runtime_default"


def render_report(summary: dict[str, Any]) -> str:
    inputs = summary["inputs"]
    fixed_large_rows = suite_table_rows(
        summary["fixed_large_results"],
        variants(),
        inputs["budget_pairs"],
    )
    heldout_rows = []
    for suite_name in summary["heldout_results"]:
        row = [suite_name]
        for variant_name in (
            "old_default_ref",
            "promoted_default",
            "explicit_no_tactical_bias",
        ):
            row.append(
                fmt(
                    summary["heldout_results"][suite_name][variant_name]["384:256"][
                        "ds"
                    ]
                )
            )
            row.append(
                fmt(
                    summary["heldout_results"][suite_name][variant_name]["768:768"][
                        "ds"
                    ]
                )
            )
            row.append(
                fmt(
                    summary["heldout_results"][suite_name][variant_name]["1200:1200"][
                        "ds"
                    ]
                )
            )
            row.append(
                fmt(
                    summary["heldout_results"][suite_name][variant_name]["1200:256"][
                        "ds"
                    ]
                )
            )
        heldout_rows.append(row)
    return "\n".join(
        [
            "# AlphaZero-Lite No Tactical Bias Runtime Promotion Results",
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
            "## Search Profiles",
            "",
            f"- Old default search profile: `{json.dumps(summary['old_default_search_profile'], sort_keys=True)}`",
            f"- New default search profile: `{json.dumps(summary['new_default_search_profile'], sort_keys=True)}`",
            "",
            "## Files Changed",
            "",
            *[f"- `{path}`" for path in summary["files_changed"]],
            "",
            "## Effective-Profile Diagnostics",
            "",
            render_profile_diagnostics(summary["diagnostics"]),
            "",
            "## Fixed Large Before/After",
            "",
            markdown_table(
                ["Variant", *inputs["budget_pairs"]],
                fixed_large_rows,
            ),
            "",
            "## Held-Out Spot Check",
            "",
            markdown_table(
                [
                    "Suite",
                    "old 384:256",
                    "old 768:768",
                    "old 1200:1200",
                    "old 1200:256",
                    "promoted 384:256",
                    "promoted 768:768",
                    "promoted 1200:1200",
                    "promoted 1200:256",
                    "explicit 384:256",
                    "explicit 768:768",
                    "explicit 1200:1200",
                    "explicit 1200:256",
                ],
                heldout_rows,
            ),
            "",
            "## Gate Result",
            "",
            f"- Default gate classification: `{summary['gate']['report'].get('classification')}`",
            f"- Standard budget effective profile: `{json.dumps(summary['gate']['standard_arena_profile'], sort_keys=True)}`",
            "",
            "## Runtime Cost Comparison",
            "",
            "Held-out mean move latency at `384:256`.",
            "",
            markdown_table(
                ["Variant", "Mean move latency ms", "Mean p95 move latency ms"],
                runtime_cost_rows(summary["heldout_mean_summary"]),
            ),
            "",
            "## Decision Summary",
            "",
            f"- Default active in runtime/benchmark/gate paths: `{summary['checks']['default_profile']['default_active']}`",
            f"- Explicit `tactical_root_bias=0.1` still reproducible: `{summary['checks']['default_profile']['explicit_old_default_supported']}`",
            f"- Promoted default equals explicit no-tactical-bias: `{summary['checks']['evaluation_match']['passed']}`",
            f"- Improvement direction reproduced: `{summary['checks']['improvement_direction']['passed']}`",
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

    current_summary = verify_expected_hash(
        current_weights,
        args.expected_current_weights_sha256,
        "current artifact weights",
    )
    metadata_payload = json.loads(current_metadata.read_text(encoding="utf-8"))

    evaluator = arena.ArtifactEvaluator(current_path)
    probe = arena.evaluate_artifact_position(
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
        seed=args.seed,
        c_puct=DEFAULT_C_PUCT,
        search_options=arena.build_eval_search_options(
            root_policy_mode=DEFAULT_ROOT_POLICY_MODE
        ),
    )

    variant_list = variants()
    budget_pairs = parse_budget_pairs(args.budget_pairs)
    budget_labels = [budget_pair_label(*pair) for pair in budget_pairs]

    diagnostic_suite_entries = load_suite(str(large_suite))[:1]
    diagnostic_suite_path = workdir / "diagnostics" / "tiny_suite.jsonl"
    diagnostic_suite_path.parent.mkdir(parents=True, exist_ok=True)
    with diagnostic_suite_path.open("w", encoding="utf-8") as handle:
        for entry in diagnostic_suite_entries:
            handle.write(json.dumps(entry) + "\n")

    diagnostics = {
        "default_runtime_path": {
            "default": current_runtime_profile(),
            "explicit_old_default_ref": run_arena_profile_diagnostic(
                current_path=current_path,
                workdir=workdir / "diagnostics" / "arena_explicit_old_default_ref",
                tactical_root_bias=OLD_DEFAULT_TACTICAL_ROOT_BIAS,
            ),
        },
        "opening_suite_benchmark_path": {
            "default": run_opening_benchmark_diagnostic(
                current_path=current_path,
                suite_path=diagnostic_suite_path,
                workdir=workdir / "diagnostics" / "benchmark_default",
                tactical_root_bias=None,
            ),
            "explicit_old_default_ref": run_opening_benchmark_diagnostic(
                current_path=current_path,
                suite_path=diagnostic_suite_path,
                workdir=workdir / "diagnostics" / "benchmark_explicit_old_default_ref",
                tactical_root_bias=OLD_DEFAULT_TACTICAL_ROOT_BIAS,
            ),
        },
        "seat_aware_gate_path": {
            "default": run_gate_diagnostic(
                current_path=current_path,
                workdir=workdir / "diagnostics" / "gate_default",
                tactical_root_bias=None,
            ),
            "explicit_old_default_ref": run_gate_diagnostic(
                current_path=current_path,
                workdir=workdir / "diagnostics" / "gate_explicit_old_default_ref",
                tactical_root_bias=OLD_DEFAULT_TACTICAL_ROOT_BIAS,
            ),
        },
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
    evaluation_match = compare_promoted_and_explicit(
        heldout_results | {"large_suite": fixed_large_results}, budget_labels
    )
    improvement_direction = improvement_checks(
        fixed_large_results=fixed_large_results,
        heldout_mean_summary=heldout_mean_summary,
    )
    gate = run_full_gate(
        current_path=current_path,
        workdir=workdir / "gate",
        workers=args.workers,
        seed=args.seed,
        games=args.gate_games,
    )

    default_profiles = [
        diagnostics["default_runtime_path"]["default"]["search_profile"],
        diagnostics["opening_suite_benchmark_path"]["default"]["search_profile"],
        diagnostics["seat_aware_gate_path"]["default"]["search_profile"],
    ]
    explicit_old_profiles = [
        diagnostics["default_runtime_path"]["explicit_old_default_ref"][
            "search_profile"
        ],
        diagnostics["opening_suite_benchmark_path"]["explicit_old_default_ref"][
            "search_profile"
        ],
        diagnostics["seat_aware_gate_path"]["explicit_old_default_ref"][
            "search_profile"
        ],
    ]
    default_active = all(
        isinstance(profile, dict)
        and profile.get("search_options", {}).get("tactical_root_bias")
        == PROMOTED_TACTICAL_ROOT_BIAS
        for profile in default_profiles
    )
    explicit_old_supported = all(
        isinstance(profile, dict)
        and profile.get("search_options", {}).get("tactical_root_bias")
        == OLD_DEFAULT_TACTICAL_ROOT_BIAS
        for profile in explicit_old_profiles
    )
    gate_classification = gate["report"].get("classification")
    gate_profile = gate.get("standard_arena_profile")
    gate_passed = (
        gate_classification in ACCEPTED_GATE_CLASSIFICATIONS
        and isinstance(gate_profile, dict)
        and gate_profile.get("search_options", {}).get("tactical_root_bias")
        == PROMOTED_TACTICAL_ROOT_BIAS
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
            "c_puct": float(DEFAULT_C_PUCT),
            "tactical_root_bias": float(OLD_DEFAULT_TACTICAL_ROOT_BIAS),
            "root_prior_transform": None,
            "value_trust_schedule": None,
        },
        "new_default_search_profile": {
            "search_mode": DEFAULT_SEARCH_MODE,
            "root_policy_mode": DEFAULT_ROOT_POLICY_MODE,
            "c_puct": float(DEFAULT_C_PUCT),
            "tactical_root_bias": float(PROMOTED_TACTICAL_ROOT_BIAS),
            "root_prior_transform": None,
            "value_trust_schedule": None,
        },
        "metadata_schema_version": metadata_payload.get("schema_version"),
        "runtime_probe": {
            "selected_move": probe.get("selected_move"),
            "search_options": probe.get("search_options"),
        },
        "diagnostics": diagnostics,
        "fixed_large_results": fixed_large_results,
        "heldout_results": heldout_results,
        "heldout_mean_summary": heldout_mean_summary,
        "gate": gate,
        "checks": {
            "artifact_integrity": {
                "passed": current_summary["actual_sha256"]
                == args.expected_current_weights_sha256,
            },
            "runtime_loader": {
                "passed": probe.get("selected_move") is not None,
            },
            "metadata_parse": {
                "passed": isinstance(metadata_payload, dict)
                and metadata_payload.get("schema_version") is not None,
            },
            "default_profile": {
                "default_active": default_active,
                "explicit_old_default_supported": explicit_old_supported,
            },
            "evaluation_match": {
                "passed": evaluation_match["matches"],
                "details": evaluation_match,
            },
            "improvement_direction": improvement_direction,
            "gate": {
                "passed": gate_passed,
                "classification": gate_classification,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
