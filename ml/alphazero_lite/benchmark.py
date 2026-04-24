#!/usr/bin/env python3
"""Benchmark contract runner skeleton for AlphaZero-lite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.report_validation import ArenaReportValidationError, validate_arena_report
from ml.alphazero_lite.self_play import add_search_option_args, build_eval_search_options, search_options_from_args


EXPECTED_MCTS_SCHEMA = "azlite_vs_mcts_v1"


def opening_cache_summary_fields(summary: dict | None) -> dict | None:
    if not isinstance(summary, dict):
        return None

    return {
        "runtime_hit_rate": summary.get("runtime_hit_rate"),
        "training_hit_rate": summary.get("training_hit_rate"),
        "opening_bucket_quality_delta": summary.get("opening_bucket_quality_delta"),
        "latency_delta_ms": summary.get("latency_delta_ms"),
    }


def mcts_integer_field(report: dict, key: str, report_name: str) -> int:
    if key not in report:
        raise SystemExit(f"{report_name} missing required field: {key}")

    value = report[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"{report_name} field {key} must be an integer")

    return value


def validate_mcts_report(report: dict, report_name: str) -> tuple[int, int, int, int]:
    schema = report.get("schema")
    if schema is not None and schema != EXPECTED_MCTS_SCHEMA:
        raise SystemExit(f"{report_name} schema must be {EXPECTED_MCTS_SCHEMA} when present")

    games = mcts_integer_field(report, "games", report_name)
    az_wins = mcts_integer_field(report, "az_wins", report_name)
    mcts_wins = mcts_integer_field(report, "mcts_wins", report_name)
    draws = mcts_integer_field(report, "draws", report_name)

    if games <= 0:
        raise SystemExit(f"{report_name} games must be > 0")
    if az_wins < 0 or mcts_wins < 0 or draws < 0:
        raise SystemExit(f"{report_name} az_wins/mcts_wins/draws must be non-negative")
    if az_wins + mcts_wins + draws != games:
        raise SystemExit(f"{report_name} az_wins + mcts_wins + draws must equal games")

    return games, az_wins, mcts_wins, draws


def validated_numeric_score(report: dict, report_name: str) -> float:
    if "score" not in report:
        raise SystemExit(f"dynamic budget comparison requires explicit score fields; missing {report_name} score")
    value = report.get("score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"dynamic budget comparison requires explicit score fields; {report_name} score must be numeric")
    score = float(value)
    games, az_wins, _mcts_wins, draws = validate_mcts_report(report, report_name)
    derived_score = round((az_wins + (0.5 * draws)) / games, 4)
    if score != derived_score:
        raise SystemExit(
            f"dynamic budget comparison requires explicit score fields derived from raw counts; {report_name} score does not match games/az_wins/draws"
        )
    return score


def validate_matching_fixed_arm_configuration(candidate_report: dict, baseline_report: dict) -> None:
    candidate_profile = candidate_report.get("search_profile")
    baseline_profile = baseline_report.get("search_profile")
    if not isinstance(candidate_profile, dict) or not isinstance(baseline_profile, dict):
        raise SystemExit("dynamic budget comparison requires producer provenance metadata in both reports")
    if candidate_profile.get("kind") != baseline_profile.get("kind"):
        raise SystemExit("dynamic budget comparison requires baseline report from the matching fixed arm configuration")
    invariant_fields = (
        "player_mode",
        "classic_mcts_simulations",
        "az_base_simulations",
        "mcts_simulations",
        "exact_solve_enabled",
        "exact_solve_stone_threshold",
    )
    for field in invariant_fields:
        candidate_value = candidate_profile.get(field)
        baseline_value = baseline_profile.get(field)
        if candidate_value != baseline_value:
            raise SystemExit("dynamic budget comparison requires baseline report from the matching fixed arm configuration")

    candidate_config = candidate_report.get("classic_mcts_dynamic_budget_config")
    baseline_config = baseline_report.get("classic_mcts_dynamic_budget_config")
    if not isinstance(candidate_config, dict) or bool(candidate_config.get("enabled")) is not True:
        raise SystemExit("dynamic budget comparison requires candidate dynamic budget config metadata")
    if not isinstance(baseline_config, dict) or bool(baseline_config.get("enabled")) is not False:
        raise SystemExit("dynamic budget comparison requires baseline report from the matching fixed arm configuration")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sanity", "promotion"], required=True)
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument("--challenger-path", default=None)
    parser.add_argument("--current-path", default="storage/ai/alphazero_lite/current")
    parser.add_argument("--arena-report", default=None)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--min-confidence-lower-bound", type=float, default=None)
    parser.add_argument("--mcts-report", default=None)
    parser.add_argument("--current-baseline-mcts-report", default=None)
    parser.add_argument("--min-mcts-score", type=float, default=0.45)
    parser.add_argument("--dry-run", action="store_true")
    add_search_option_args(parser)
    parser.set_defaults(root_policy_mode="deterministic", tactical_root_bias=0.1)
    return parser.parse_args(argv)


def dynamic_budget_comparison(mcts_report: dict, baseline_report: dict | None, *, strict: bool = True) -> dict | None:
    if baseline_report is None:
        return None

    candidate_mode = mcts_report.get("comparison_mode")
    if candidate_mode is None:
        if strict:
            raise SystemExit("dynamic budget comparison requires explicit fixed-vs-dynamic ClassicMCTS candidate comparison_mode")
        return None
    if candidate_mode != "classic_dynamic_vs_fixed":
        if strict:
            raise SystemExit("dynamic budget comparison requires explicit fixed-vs-dynamic ClassicMCTS candidate comparison_mode")
        return None

    candidate_budget = mcts_report.get("budget_summary", {})
    baseline_budget = baseline_report.get("budget_summary", {})
    candidate_source = candidate_budget.get("source")
    baseline_source = baseline_budget.get("source")
    candidate_classic_mode = mcts_report.get("classic_mcts_mode")
    baseline_classic_mode = baseline_report.get("classic_mcts_mode")
    if candidate_classic_mode != "dynamic" or baseline_classic_mode != "fixed":
        raise SystemExit("dynamic budget comparison requires dynamic candidate and fixed baseline ClassicMCTS reports")
    if candidate_source != "classic_mcts_dynamic_runtime" or baseline_source != "classic_mcts_fixed_runtime":
        raise SystemExit("dynamic budget comparison requires fixed-vs-dynamic ClassicMCTS comparison metric sources")
    validate_matching_fixed_arm_configuration(mcts_report, baseline_report)

    embedded_candidate = mcts_report.get("dynamic_budget_comparison")
    if not isinstance(embedded_candidate, dict):
        raise SystemExit("dynamic budget comparison requires embedded fixed-vs-dynamic ClassicMCTS candidate comparison data")

    candidate_mean_final_simulations = candidate_budget.get("mean_final_simulations")
    candidate_mean_root_latency_ms = candidate_budget.get("mean_root_latency_ms")
    baseline_mean_final_simulations = baseline_budget.get("mean_final_simulations")
    baseline_mean_root_latency_ms = baseline_budget.get("mean_root_latency_ms")
    candidate_seat_bias_neutralized = embedded_candidate.get("seat_bias_neutralized")
    if not isinstance(candidate_seat_bias_neutralized, bool):
        raise SystemExit("dynamic budget comparison requires explicit seat-bias allocation metadata")
    if candidate_mean_root_latency_ms is None or baseline_mean_root_latency_ms is None:
        raise SystemExit("dynamic budget comparison requires candidate and baseline latency metrics")
    runtime_target_ms = baseline_mean_root_latency_ms
    runtime_target_matched = candidate_mean_root_latency_ms >= runtime_target_ms

    candidate_score = validated_numeric_score(mcts_report, "candidate")
    baseline_score = validated_numeric_score(baseline_report, "baseline")

    result = {
        "comparison_mode": "classic_dynamic_vs_fixed",
        "runtime_target_ms": runtime_target_ms,
        "runtime_target_matched": bool(runtime_target_matched),
        "seat_bias_neutralized": candidate_seat_bias_neutralized,
        "dynamic_mean_final_simulations": None
        if candidate_mean_final_simulations is None
        else float(candidate_mean_final_simulations),
        "dynamic_mean_root_latency_ms": None
        if candidate_mean_root_latency_ms is None
        else float(candidate_mean_root_latency_ms),
        "fixed_mean_final_simulations": None
        if baseline_mean_final_simulations is None
        else float(baseline_mean_final_simulations),
        "fixed_mean_root_latency_ms": None
        if baseline_mean_root_latency_ms is None
        else float(baseline_mean_root_latency_ms),
        "dynamic_score": candidate_score,
        "fixed_score": baseline_score,
    }
    if embedded_candidate != result:
        raise SystemExit("dynamic budget comparison requires truthful fixed-vs-dynamic ClassicMCTS comparison data")
    return result


def checks_for_mode(mode: str) -> list[dict[str, str]]:
    if mode == "sanity":
        return [
            {"id": "az_identity", "description": "AlphaZero identity check"},
            {"id": "mcts_identity", "description": "MCTS1200 identity check"},
            {"id": "mcts_monotonic", "description": "MCTS1800 vs MCTS1200"},
            {"id": "runtime_parity", "description": "Chooser vs preloaded parity"},
        ]

    return [
        {"id": "promotion_arena", "description": "Candidate versus current arena gate"},
    ]


def promotion_checks(args: argparse.Namespace) -> list[dict]:
    if args.dry_run:
        return checks_for_mode("promotion")

    if not args.arena_report:
        raise SystemExit("--arena-report is required for promotion mode when not using --dry-run")
    if not args.mcts_report:
        raise SystemExit("--mcts-report is required for promotion mode when not using --dry-run")

    arena_path = Path(args.arena_report)
    if not arena_path.exists():
        raise SystemExit(f"arena report not found: {arena_path}")

    arena = json.loads(arena_path.read_text(encoding="utf-8"))
    try:
        arena_result = validate_arena_report(
            report=arena,
            min_score=float(args.min_score),
            min_confidence_lower_bound=args.min_confidence_lower_bound,
        )
    except ArenaReportValidationError as error:
        raise SystemExit(str(error)) from error

    games_played = int(arena_result["games_played"])
    wins = int(arena_result["wins"])
    losses = int(arena_result["losses"])
    draws = int(arena_result["draws"])
    score = float(arena_result["score"])
    declared = bool(arena.get("promotion_decision", {}).get("passed", False))
    check_passed = bool(arena_result["passed"])

    mcts_path = Path(args.mcts_report)
    if not mcts_path.exists():
        raise SystemExit(f"mcts report not found: {mcts_path}")

    mcts = json.loads(mcts_path.read_text(encoding="utf-8"))
    mcts_games, mcts_wins, mcts_losses, mcts_draws = validate_mcts_report(mcts, "mcts report")

    mcts_score = (mcts_wins + (0.5 * mcts_draws)) / mcts_games
    mcts_min_score = float(args.min_mcts_score)
    mcts_passed = mcts_score >= mcts_min_score
    mcts_check: dict[str, int | float | bool | str] = {
        "id": "mcts1200_gate",
        "description": "Candidate versus MCTS1200 minimum strength",
        "passed": bool(mcts_passed),
        "score": round(mcts_score, 4),
        "games_played": mcts_games,
        "wins": mcts_wins,
        "losses": mcts_losses,
        "draws": mcts_draws,
        "min_score": mcts_min_score,
    }

    if args.current_baseline_mcts_report:
        baseline_mcts_path = Path(args.current_baseline_mcts_report)
        if not baseline_mcts_path.exists():
            raise SystemExit(f"current baseline mcts report not found: {baseline_mcts_path}")

        baseline_mcts = json.loads(baseline_mcts_path.read_text(encoding="utf-8"))
        baseline_games, baseline_wins, _baseline_losses, baseline_draws = validate_mcts_report(
            baseline_mcts,
            "current baseline mcts report",
        )

        baseline_score = (baseline_wins + (0.5 * baseline_draws)) / baseline_games
        mcts_check.update(
            {
                "description": "Candidate versus current baseline MCTS1200 strength",
                "passed": bool(mcts_score >= baseline_score),
                "baseline_score": round(baseline_score, 4),
                "comparison": "current_baseline",
            }
        )
        del mcts_check["min_score"]

    arena_check: dict[str, int | float | bool | str] = {
        "id": "promotion_arena",
        "description": "Candidate versus current arena gate",
        "passed": check_passed,
        "score": round(score, 4),
        "games_played": games_played,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "declared_passed": declared,
        "min_score": float(args.min_score),
    }
    if args.min_confidence_lower_bound is not None:
        arena_check.update(
            {
                "confidence_lower_bound": round(float(arena_result["confidence_lower_bound"]), 4),
                "confidence_passed": bool(arena_result["confidence_passed"]),
                "min_confidence_lower_bound": float(args.min_confidence_lower_bound),
            }
        )

    return [arena_check, mcts_check]


def build_report(args: argparse.Namespace) -> dict:
    checks = checks_for_mode(args.mode)
    dynamic_budget = None
    dynamic_budget_metric_source = None
    classic_mcts_dynamic_budget_config = None
    opening_cache_summary = None
    if args.mode == "promotion":
        checks = promotion_checks(args)
        if not args.dry_run and args.arena_report:
            arena = json.loads(Path(args.arena_report).read_text(encoding="utf-8"))
            opening_cache_summary = opening_cache_summary_fields(arena.get("opening_cache_summary"))
        if not args.dry_run and args.mcts_report:
            mcts = json.loads(Path(args.mcts_report).read_text(encoding="utf-8"))
            dynamic_budget_metric_source = {
                "candidate": mcts.get("budget_summary", {}).get("source"),
            }
            classic_mcts_dynamic_budget_config = {
                "candidate": mcts.get("classic_mcts_dynamic_budget_config"),
            }
            if args.current_baseline_mcts_report:
                baseline_mcts = json.loads(Path(args.current_baseline_mcts_report).read_text(encoding="utf-8"))
                dynamic_budget_metric_source["baseline"] = baseline_mcts.get("budget_summary", {}).get("source")
                classic_mcts_dynamic_budget_config["baseline"] = baseline_mcts.get("classic_mcts_dynamic_budget_config")
                dynamic_budget = dynamic_budget_comparison(mcts, baseline_mcts, strict=True)

    return {
        "schema": "azlite_benchmark_v1",
        "mode": args.mode,
        "games": args.games,
        "seed": args.seed,
        "challenger_path": args.challenger_path,
        "current_path": args.current_path,
        "arena_report": args.arena_report,
        "mcts_report": args.mcts_report,
        "current_baseline_mcts_report": args.current_baseline_mcts_report,
        "min_score": float(args.min_score),
        "min_confidence_lower_bound": args.min_confidence_lower_bound,
        "min_mcts_score": float(args.min_mcts_score),
        "dry_run": bool(args.dry_run),
        "search_options": build_eval_search_options(**search_options_from_args(args)),
        "dynamic_budget_metric_source": dynamic_budget_metric_source,
        "classic_mcts_dynamic_budget_config": classic_mcts_dynamic_budget_config,
        "dynamic_budget_comparison": dynamic_budget,
        "opening_cache_summary": opening_cache_summary,
        "checks": checks,
    }


def build_report_from_inputs(*, arena_report: dict, mcts_report: dict, opening_cache_summary: dict | None = None) -> dict:
    validate_arena_report(report=arena_report, min_score=0.0, min_confidence_lower_bound=None)
    validate_mcts_report(mcts_report, "mcts report")
    resolved_opening_cache_summary = opening_cache_summary
    if resolved_opening_cache_summary is None:
        resolved_opening_cache_summary = arena_report.get("opening_cache_summary")

    return {
        "schema": "azlite_benchmark_v1",
        "mode": "promotion",
        "arena_report": arena_report,
        "mcts_report": mcts_report,
        "opening_cache_summary": opening_cache_summary_fields(resolved_opening_cache_summary),
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = build_report(args)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"wrote benchmark report to {out_path}")


if __name__ == "__main__":
    main()
