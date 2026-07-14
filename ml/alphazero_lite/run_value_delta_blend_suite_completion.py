#!/usr/bin/env python3
"""Complete staged, inference-only opening-suite evaluation for PR #158 blends."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    parse_cpuct_schedule_json,
    resolve_budget_cpuct,
    schedule_definition,
)
from ml.alphazero_lite.run_opening_suite_seat_benchmark import (  # noqa: E402
    compute_per_opening_metrics,
    compute_seat_metrics,
    parse_game_jsonl,
    run_arena,
)

BUDGETS = ("384:256", "768:256", "768:768", "1200:1200", "1200:256", "256:768")
EQUAL_BUDGETS = frozenset({"768:768", "1200:1200"})
REPORT_PATH = (
    REPO_ROOT / "docs/alphazero-lite-value-delta-blend-suite-completion-results.md"
)
SUMMARY_PATH = (
    REPO_ROOT
    / "docs/data/alphazero-lite-value-delta-blend-suite-completion-summary.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8192), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def lane_schedule(name: str) -> dict[str, float]:
    if name in {"alpha000", "blend_alpha_000"}:
        return {budget: 0.0 for budget in BUDGETS}
    if name in {"alpha010", "blend_alpha_010"}:
        return {budget: 0.1 for budget in BUDGETS}
    if name in {"alpha025", "blend_alpha_025", "global025"}:
        return {budget: 0.25 for budget in BUDGETS}
    if name in {"alpha100", "blend_alpha_100"}:
        return {budget: 1.0 for budget in BUDGETS}
    if name == "asym050_equal010":
        return {budget: 0.1 if budget in EQUAL_BUDGETS else 0.5 for budget in BUDGETS}
    raise ValueError(f"unknown lane: {name}")


def canonical_lane_name(name: str) -> str:
    return {
        "alpha000": "blend_alpha_000",
        "alpha010": "blend_alpha_010",
        "alpha025": "blend_alpha_025",
        "alpha100": "blend_alpha_100",
        "global025": "blend_alpha_025",
    }.get(name, name)


def deduplicate_lanes(names: list[str]) -> tuple[list[str], dict[str, str]]:
    """Retain one lane per effective budget schedule, including global025."""
    retained: list[str] = []
    aliases: dict[str, str] = {}
    schedules: dict[tuple[float, ...], str] = {}
    for requested in names:
        name = canonical_lane_name(requested)
        key = tuple(lane_schedule(name)[budget] for budget in BUDGETS)
        if key in schedules:
            aliases[requested] = schedules[key]
        else:
            schedules[key] = name
            retained.append(name)
            if requested == "global025":
                aliases[requested] = name
    return retained, aliases


def probe_passing_lanes(preflight: dict[str, Any]) -> list[str]:
    return [
        name
        for name, lane in preflight.get("probes", {}).items()
        if lane.get("role") == "candidate_lane"
        and lane.get("candidate_probe_pass")
        and lane.get("carry_to_medium")
    ]


def stage_complete(stage: dict[str, Any]) -> bool:
    return bool(
        isinstance(stage, dict)
        and stage.get("status") == "completed"
        and any(
            isinstance(item, dict)
            and item.get("role") == "candidate_lane"
            and item.get("completed")
            for item in stage.get("lanes", {}).values()
        )
    )


def medium_pass(metrics: dict[str, Any]) -> bool:
    values = metrics["by_budget"]
    return (
        values["384:256"]["candidate_minus_current_ds"] >= 0.03
        and values["768:768"]["candidate_minus_current_ds"] >= -0.03
        and values["1200:1200"]["candidate_minus_current_ds"] >= -0.03
        and values["1200:256"]["candidate_minus_current_ds"] >= -0.03
    )


def fixed_large_pass(metrics: dict[str, Any]) -> bool:
    values = metrics["by_budget"]
    return (
        values["384:256"]["candidate_minus_current_ds"] >= 0.05
        and values["768:768"]["candidate_minus_current_ds"] >= -0.05
        and values["1200:1200"]["candidate_minus_current_ds"] >= -0.03
        and values["1200:256"]["candidate_minus_current_ds"] >= -0.03
    )


def heldout_pass(metrics: dict[str, Any]) -> bool:
    values = metrics["by_budget"]
    return (
        values["384:256"]["candidate_minus_current_ds"] >= 0.05
        and values["384:256"]["bootstrap_ci95"]["lower"] > 0.01
        and values["768:768"]["candidate_minus_current_ds"] >= -0.05
        and values["1200:1200"]["candidate_minus_current_ds"] >= -0.03
        and values["1200:256"]["candidate_minus_current_ds"] >= -0.03
    )


def classify(summary: dict[str, Any]) -> str:
    candidates = summary["probe_passing_candidates"]
    medium = summary["stages"]["medium"]
    if candidates and not stage_complete(medium):
        return "value_delta_blend_suite_incomplete"
    lanes = medium.get("lanes", {})
    intermediates = [x for x in lanes.values() if x.get("role") == "candidate_lane"]
    if intermediates and all(
        item["by_budget"]["384:256"]["candidate_minus_current_ds"] < 0.03
        and item["by_budget"]["768:768"]["candidate_minus_current_ds"] >= -0.05
        and item["by_budget"]["1200:1200"]["candidate_minus_current_ds"] >= -0.03
        and item["by_budget"]["1200:256"]["candidate_minus_current_ds"] >= -0.03
        for item in intermediates
    ):
        return "blended_value_safe_but_strength_neutral"
    if any(
        item["by_budget"]["384:256"]["candidate_minus_current_ds"] > 0
        and any(
            item["by_budget"][budget]["candidate_minus_current_ds"] < limit
            for budget, limit in (
                ("768:768", -0.05),
                ("1200:1200", -0.03),
                ("1200:256", -0.03),
            )
        )
        for item in intermediates
    ):
        return "blended_value_tradeoff"
    fixed = summary["stages"]["fixed_large"]
    heldout = summary["stages"]["heldout"]
    if stage_complete(heldout):
        for item in heldout["lanes"].values():
            if item.get("role") == "candidate_lane" and heldout_pass(item):
                if summary["gate"].get("passed"):
                    return "blended_value_runtime_candidate"
                return "value_update_magnitude_too_large_but_blend_promising"
    if stage_complete(fixed) and any(
        item.get("role") == "candidate_lane" and fixed_large_pass(item)
        for item in fixed["lanes"].values()
    ):
        return "value_update_magnitude_too_large_but_blend_promising"
    return "outcome_value_update_not_strengthening"


def _write_openings(path: Path, suite: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps({"prefix_moves": row["prefix_moves"]}) + "\n" for row in suite
        ),
        encoding="utf-8",
    )


def evaluate_lane(
    *,
    lane: str,
    role: str,
    suite_path: Path,
    workdir: Path,
    current: Path,
    candidate: Path,
    schedule: dict[str, float],
    default_cpuct: float,
    workers: int,
    seed: int,
    games_per_opening: int,
) -> dict[str, Any]:
    suite = load_jsonl(suite_path)
    alpha = lane_schedule(lane)
    by_budget: dict[str, Any] = {}
    for budget in BUDGETS:
        challenger_sims, current_sims = (int(x) for x in budget.split(":"))
        target = workdir / lane / budget.replace(":", "_")
        metrics_path = target / "metrics.json"
        if metrics_path.is_file():
            by_budget[budget] = json.loads(metrics_path.read_text(encoding="utf-8"))
            continue
        target.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, Any]] = []
        reports: list[dict[str, Any]] = []
        cpuct = resolve_budget_cpuct(
            schedule=schedule,
            challenger_simulations=challenger_sims,
            current_simulations=current_sims,
            default_c_puct=default_cpuct,
        )
        for seat in (0, 1):
            seat_dir = target / f"starts_{seat}"
            seat_dir.mkdir(exist_ok=True)
            openings = seat_dir / "opening_suite.jsonl"
            _write_openings(openings, suite)
            report = run_arena(
                challenger=str(candidate),
                current=str(current),
                challenger_sims=challenger_sims,
                current_sims=current_sims,
                games=len(suite) * games_per_opening,
                seed=seed,
                workers=workers,
                out_json=str(seat_dir / "arena.json"),
                out_jsonl=str(seat_dir / "games.jsonl"),
                opening_prefixes_jsonl=str(openings),
                challenger_starts=seat,
                games_per_opening=games_per_opening,
                root_policy_mode="deterministic",
                root_temperature=0.0,
                normalize_values=False,
                c_puct=cpuct,
                tactical_root_bias=0.0,
                challenger_blend_current=True,
                challenger_value_alpha=alpha[budget],
            )
            reports.append(report)
            entries.extend(parse_game_jsonl(str(seat_dir / "games.jsonl")))
        metrics = compute_seat_metrics(entries)
        notes = reports[0].get("notes", {}) if reports else {}
        result = {
            **metrics,
            "effective_alpha": alpha[budget],
            "effective_c_puct": cpuct,
            "move_time_mean_ms": statistics.fmean(
                float(r["notes"]["move_time_mean_ms"])
                for r in reports
                if r.get("notes", {}).get("move_time_mean_ms") is not None
            )
            if reports
            else None,
            "move_time_p95_ms": statistics.fmean(
                float(r["notes"]["move_time_p95_ms"])
                for r in reports
                if r.get("notes", {}).get("move_time_p95_ms") is not None
            )
            if reports
            else None,
            "search_profile_hash": notes.get("search_profile_hash"),
            "per_opening_artifact": str(target / "per_opening_metrics.json"),
        }
        write_json(
            target / "per_opening_metrics.json",
            {"metrics": compute_per_opening_metrics(entries)},
        )
        write_json(metrics_path, result)
        by_budget[budget] = result
    return {
        "role": role,
        "completed": True,
        "effective_alpha_by_budget": alpha,
        "per_opening_artifact": str(workdir / lane),
        "by_budget": by_budget,
    }


def add_deltas(lanes: dict[str, Any], reference: str = "blend_alpha_000") -> None:
    baseline = lanes[reference]["by_budget"]
    for result in lanes.values():
        for budget, metrics in result["by_budget"].items():
            current_ds = baseline[budget]["ds"]
            metrics["raw_candidate_ds"] = metrics["ds"]
            metrics["raw_current_ds"] = current_ds
            metrics["candidate_minus_current_ds"] = metrics["ds"] - current_ds
            metrics["current_minus_candidate_ds"] = current_ds - metrics["ds"]


def evaluate_stage(**kwargs: Any) -> dict[str, Any]:
    lanes = {
        lane: evaluate_lane(lane=lane, role=role, **kwargs)
        for lane, role in kwargs.pop("lane_roles").items()
    }
    add_deltas(lanes)
    return {"status": "completed", "lanes": lanes}


def aggregate_heldout(stage_results: list[dict[str, Any]], lane: str) -> dict[str, Any]:
    aggregate = {"role": "candidate_lane", "completed": True, "by_budget": {}}
    for budget in BUDGETS:
        values = [stage["lanes"][lane]["by_budget"][budget] for stage in stage_results]
        deltas = [item["candidate_minus_current_ds"] for item in values]
        rng = np.random.default_rng(42)
        samples = [
            float(np.mean(rng.choice(deltas, size=len(deltas), replace=True)))
            for _ in range(10000)
        ]
        aggregate["by_budget"][budget] = {
            "mean_candidate_ds": statistics.fmean(
                item["raw_candidate_ds"] for item in values
            ),
            "mean_current_ds": statistics.fmean(
                item["raw_current_ds"] for item in values
            ),
            "worst_suite_candidate_ds": min(
                item["raw_candidate_ds"] for item in values
            ),
            "candidate_minus_current_ds": statistics.fmean(deltas),
            "current_minus_candidate_ds": -statistics.fmean(deltas),
            "bootstrap_ci95": {
                "lower": float(np.percentile(samples, 2.5)),
                "upper": float(np.percentile(samples, 97.5)),
            },
            "p0_score": statistics.fmean(item["p0_score"] for item in values),
            "p1_score": statistics.fmean(item["p1_score"] for item in values),
            "duplicate_trajectory_count": sum(
                item["duplicate_trajectory_count"] for item in values
            ),
        }
    aggregate["per_opening_artifacts"] = [
        stage["lanes"][lane]["per_opening_artifact"] for stage in stage_results
    ]
    return aggregate


def run_gate(
    *,
    workdir: Path,
    current: Path,
    candidate: Path,
    lane: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    gate_path = workdir / "deterministic_gate.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "script/ai/seat_aware_promotion_gate"),
            "--candidate-path",
            str(candidate),
            "--current-path",
            str(current),
            "--out",
            str(gate_path),
            "--workdir",
            str(workdir / "deterministic_gate"),
            "--seed",
            str(args.seed),
            "--workers",
            str(args.workers),
            "--c-puct",
            str(args.default_c_puct),
            "--c-puct-schedule-json",
            args.cpuct_schedule,
            "--root-policy-mode",
            "deterministic",
            "--budget-pairs",
            ",".join(BUDGETS),
            "--tactical-root-bias",
            "0.0",
            "--challenger-blend-current",
            "--challenger-value-alpha-by-budget",
            json.dumps(lane_schedule(lane), sort_keys=True),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return {"status": "failed", "passed": False, "error": result.stderr.strip()}
    report = json.loads(gate_path.read_text(encoding="utf-8"))
    return {
        "status": "completed",
        "passed": report.get("classification") == "promote",
        "artifact": str(gate_path),
        "report": report,
    }


def report(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-Lite Value-Delta Blend Suite Completion",
        "",
        f"- classification: `{summary['classification']}`",
        "",
    ]
    for stage_name, stage in summary["stages"].items():
        lines.extend(
            [
                f"## {stage_name.replace('_', ' ').title()}",
                "",
                f"- status: `{stage['status']}`",
            ]
        )
        for lane, result in stage.get("lanes", {}).items():
            lines.extend(
                [
                    "",
                    f"### {lane}",
                    "",
                    "| budget | raw DS | candidate-current DS | P0 | P1 | unique / duplicate trajectories | latency mean/p95 ms |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for budget, item in result["by_budget"].items():
                lines.append(
                    f"| {budget} | {item.get('raw_candidate_ds', item.get('mean_candidate_ds', 0)):+.4f} | {item['candidate_minus_current_ds']:+.4f} | {item.get('p0_score', 0):.4f} | {item.get('p1_score', 0):.4f} | {item.get('unique_trajectories', '-')}/{item.get('duplicate_trajectory_count', 0)} | {item.get('move_time_mean_ms', 0)}/{item.get('move_time_p95_ms', 0)} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--expected-candidate-weights-sha256", required=True)
    parser.add_argument("--pr158-summary", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--lanes", required=True)
    parser.add_argument("--default-c-puct", type=float, default=1.25)
    parser.add_argument("--cpuct-schedule", default='{"768:768": 0.90}')
    parser.add_argument("--tactical-root-bias", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.tactical_root_bias != 0.0:
        raise ValueError("suite completion requires tactical_root_bias=0")
    current, candidate, workdir = (
        Path(args.current),
        Path(args.candidate),
        Path(args.workdir),
    )
    if sha256(current / "weights.json") != args.expected_current_weights_sha256:
        raise RuntimeError("current weights hash mismatch")
    if sha256(candidate / "weights.json") != args.expected_candidate_weights_sha256:
        raise RuntimeError("candidate weights hash mismatch")
    preflight_path = Path(args.pr158_summary)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    requested = [x.strip() for x in args.lanes.split(",") if x.strip()]
    lanes, aliases = deduplicate_lanes(requested)
    if "global025" in preflight.get("probes", {}):
        aliases["global025"] = "blend_alpha_025"
    if "blend_alpha_000" not in lanes:
        lanes.insert(0, "blend_alpha_000")
    if "blend_alpha_100" not in lanes:
        lanes.append("blend_alpha_100")
    schedule = parse_cpuct_schedule_json(args.cpuct_schedule)
    common = {
        "current": current,
        "candidate": candidate,
        "schedule": schedule,
        "default_cpuct": args.default_c_puct,
        "workers": args.workers,
        "seed": args.seed,
        "games_per_opening": 2,
    }
    roles = {lane: "candidate_lane" for lane in lanes}
    roles["blend_alpha_000"] = "required_reference"
    roles["blend_alpha_100"] = "diagnostic_reference"
    medium = evaluate_stage(
        suite_path=Path(args.medium_suite),
        workdir=workdir / "medium",
        lane_roles=roles,
        **common,
    )
    intermediate = [
        name
        for name, result in medium["lanes"].items()
        if result["role"] == "candidate_lane" and medium_pass(result)
    ][:2]
    fixed_roles = {
        "blend_alpha_000": "required_reference",
        "blend_alpha_100": "diagnostic_reference",
        **{name: "candidate_lane" for name in intermediate},
    }
    fixed = (
        evaluate_stage(
            suite_path=Path(args.fixed_large_suite),
            workdir=workdir / "fixed_large",
            lane_roles=fixed_roles,
            **common,
        )
        if intermediate
        else {
            "status": "stopped",
            "stop_reason": "no_intermediate_medium_pass",
            "lanes": {},
        }
    )
    fixed_winners = [
        name
        for name in intermediate
        if fixed.get("lanes", {}).get(name) and fixed_large_pass(fixed["lanes"][name])
    ]
    heldout: dict[str, Any] = {"status": "not_reached", "lanes": {}}
    if fixed_winners:
        winner = max(
            fixed_winners,
            key=lambda name: fixed["lanes"][name]["by_budget"]["384:256"][
                "candidate_minus_current_ds"
            ],
        )
        per_suite = []
        for index, suite in enumerate(x for x in args.heldout_suites.split(",") if x):
            per_suite.append(
                evaluate_stage(
                    suite_path=Path(suite),
                    workdir=workdir / "heldout" / f"suite_{index}",
                    lane_roles={
                        "blend_alpha_000": "required_reference",
                        winner: "candidate_lane",
                    },
                    **common,
                )
            )
        heldout = {
            "status": "completed",
            "selected_lane": winner,
            "suite_results": per_suite,
            "lanes": {winner: aggregate_heldout(per_suite, winner)},
        }
    elif intermediate:
        heldout = {
            "status": "stopped",
            "stop_reason": "no_fixed_large_intermediate_pass",
            "lanes": {},
        }
    gate = {"status": "not_run", "passed": False}
    if heldout.get("status") == "completed":
        heldout_lane = heldout["selected_lane"]
        if heldout_pass(heldout["lanes"][heldout_lane]):
            gate = run_gate(
                workdir=workdir,
                current=current,
                candidate=candidate,
                lane=heldout_lane,
                args=args,
            )
    summary = {
        "schema": "azlite_value_delta_blend_suite_completion_v1",
        "pr158_input_hash": sha256(preflight_path),
        "current_artifact_hash": sha256(current / "weights.json"),
        "candidate_artifact_hash": sha256(candidate / "weights.json"),
        "evaluated_lane_schedules": {name: lane_schedule(name) for name in lanes},
        "deduplicated_lanes": aliases,
        "probe_passing_candidates": probe_passing_lanes(preflight),
        "runtime_profile": {
            "default_c_puct": args.default_c_puct,
            "c_puct_schedule": schedule_definition(
                default_c_puct=args.default_c_puct, schedule=schedule
            ),
            "tactical_root_bias": 0.0,
            "normalize_values": False,
            "root_policy_mode": "deterministic",
        },
        "stages": {"medium": medium, "fixed_large": fixed, "heldout": heldout},
        "gate": gate,
    }
    summary["classification"] = classify(summary)
    write_json(workdir / "summary_metrics.json", summary)
    write_json(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(report(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
