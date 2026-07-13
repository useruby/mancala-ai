#!/usr/bin/env python3
# ruff: noqa: E402
"""Evaluate best value-head-only probe candidate under baseline vs normalize-values search."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (
    benchmark_budget_results,
)  # noqa: E402
from ml.alphazero_lite.run_current_init_policy_only_distillation_preflight import (
    matching_candidate_report,
)  # noqa: E402

SUMMARY_DEFAULT = "/tmp/azlite_terminal_outcome_selfplay_iteration/summary_metrics.json"
REPORT_DEFAULT = (
    REPO_ROOT
    / "docs/alphazero-lite-value-head-best-probe-normalize-values-suite-eval-results.md"
)
DEFAULT_BUDGET_PAIRS = "384:256,768:768,1200:1200,1200:256,256:768"


def _python() -> str:
    candidate = REPO_ROOT / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        rendered.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(rendered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=SUMMARY_DEFAULT)
    parser.add_argument(
        "--workdir", default="/tmp/azlite_value_head_best_probe_normalize_eval"
    )
    parser.add_argument("--fixed-large-suite", default=None)
    parser.add_argument("--heldout-suites", default=None)
    parser.add_argument("--budget-pairs", default=DEFAULT_BUDGET_PAIRS)
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--out-md", default=str(REPORT_DEFAULT))
    return parser.parse_args()


def run_benchmark(
    *,
    workdir: Path,
    suite_path: Path,
    current_artifact: Path,
    candidate_artifact: Path,
    budget_pairs: str,
    games_per_opening: int,
    workers: int,
    seed: int,
    timeout: int,
    normalize_values: bool,
) -> dict[str, Any]:
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/run_opening_suite_seat_benchmark.py"),
        "--workdir",
        str(workdir),
        "--suite",
        str(suite_path),
        "--current",
        str(current_artifact),
        "--candidates",
        str(candidate_artifact),
        "--budget-pairs",
        budget_pairs,
        "--games-per-opening",
        str(games_per_opening),
        "--seed",
        str(seed),
        "--root-policy-mode",
        "deterministic",
        "--workers",
        str(workers),
        "--timeout",
        str(timeout),
    ]
    if normalize_values:
        cmd.append("--normalize-values")
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout + 300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"benchmark failed: {result.stderr[-2000:]}")
    return read_json(workdir / "temperature_benchmark_report.json")


def suite_budget_map(
    *,
    report: dict[str, Any],
    suite_path: Path,
    current_artifact: Path,
    candidate_name: str,
    candidate_artifact: Path,
) -> dict[str, dict[str, Any]]:
    candidate_report = matching_candidate_report(
        report=report,
        candidate_name=candidate_name,
        artifact_path=candidate_artifact,
        current_artifact=current_artifact,
    )
    if candidate_report is None:
        raise RuntimeError(
            f"missing candidate report for {candidate_name} in {suite_path.name}"
        )
    return benchmark_budget_results(candidate_report)


def build_report(summary: dict[str, Any]) -> str:
    comparison_rows = []
    for suite_name, suite_summary in sorted(summary["suites"].items()):
        for budget, budget_summary in sorted(suite_summary["budgets"].items()):
            comparison_rows.append(
                [
                    suite_name,
                    budget,
                    fmt(float(budget_summary["baseline_candidate_ds"])),
                    fmt(float(budget_summary["normalize_candidate_ds"])),
                    fmt(float(budget_summary["normalize_minus_baseline"])),
                ]
            )
    lines = [
        "# AlphaZero-Lite Value-Head Best-Probe Normalize-Values Suite Eval",
        "",
        f"- source summary: `{summary['source_summary']}`",
        f"- candidate: `{summary['candidate_name']}`",
        f"- candidate artifact: `{summary['candidate_artifact']}`",
        f"- budget pairs: `{summary['budget_pairs']}`",
        "",
        "## Budget Comparison",
        "",
        markdown_table(
            ["Suite", "Budget", "Baseline DS", "Normalize DS", "Normalize-Baseline"],
            comparison_rows,
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary)
    smoke_summary = read_json(summary_path)
    current_artifact = Path(
        str((smoke_summary.get("candidate_rows") or {})["current_ref"]["artifact_dir"])
    )
    candidate_row = (smoke_summary.get("candidate_rows") or {}).get(
        "value_head_only_best_probe"
    )
    if not isinstance(candidate_row, dict):
        raise RuntimeError("summary missing value_head_only_best_probe")
    candidate_artifact = Path(str(candidate_row["artifact_dir"]))
    candidate_name = str(candidate_row["name"])
    fixed_large_suite = Path(
        args.fixed_large_suite
        or smoke_summary.get("inputs", {}).get("fixed_large_suite", {}).get("path")
    )
    heldout_suite_paths = []
    if args.heldout_suites:
        heldout_suite_paths = [
            Path(part.strip())
            for part in args.heldout_suites.split(",")
            if part.strip()
        ]
    else:
        heldout_suite_paths = [
            Path(payload["path"])
            for _name, payload in sorted(
                (smoke_summary.get("inputs", {}).get("heldout_suites") or {}).items()
            )
        ]

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    suite_paths = [
        ("fixed_large", fixed_large_suite),
        *[(path.stem, path) for path in heldout_suite_paths],
    ]
    suites_summary: dict[str, Any] = {}
    for suite_name, suite_path in suite_paths:
        baseline_dir = workdir / suite_name / "baseline"
        normalize_dir = workdir / suite_name / "normalize_values"
        baseline_start = time.time()
        baseline_report = run_benchmark(
            workdir=baseline_dir,
            suite_path=suite_path,
            current_artifact=current_artifact,
            candidate_artifact=candidate_artifact,
            budget_pairs=args.budget_pairs,
            games_per_opening=args.games_per_opening,
            workers=args.workers,
            seed=args.seed,
            timeout=args.timeout,
            normalize_values=False,
        )
        baseline_elapsed = time.time() - baseline_start
        normalize_start = time.time()
        normalize_report = run_benchmark(
            workdir=normalize_dir,
            suite_path=suite_path,
            current_artifact=current_artifact,
            candidate_artifact=candidate_artifact,
            budget_pairs=args.budget_pairs,
            games_per_opening=args.games_per_opening,
            workers=args.workers,
            seed=args.seed,
            timeout=args.timeout,
            normalize_values=True,
        )
        normalize_elapsed = time.time() - normalize_start
        baseline_budgets = suite_budget_map(
            report=baseline_report,
            suite_path=suite_path,
            current_artifact=current_artifact,
            candidate_name=candidate_name,
            candidate_artifact=candidate_artifact,
        )
        normalize_budgets = suite_budget_map(
            report=normalize_report,
            suite_path=suite_path,
            current_artifact=current_artifact,
            candidate_name=candidate_name,
            candidate_artifact=candidate_artifact,
        )
        budgets_summary: dict[str, Any] = {}
        for budget in sorted(set(baseline_budgets) & set(normalize_budgets)):
            baseline_ds = float(baseline_budgets[budget]["ds"])
            normalize_ds = float(normalize_budgets[budget]["ds"])
            budgets_summary[budget] = {
                "baseline_candidate_ds": baseline_ds,
                "normalize_candidate_ds": normalize_ds,
                "normalize_minus_baseline": normalize_ds - baseline_ds,
            }
        suites_summary[suite_name] = {
            "suite_path": str(suite_path),
            "baseline_elapsed_s": baseline_elapsed,
            "normalize_elapsed_s": normalize_elapsed,
            "budgets": budgets_summary,
        }

    output = {
        "source_summary": str(summary_path.parent),
        "candidate_name": candidate_name,
        "candidate_artifact": str(candidate_artifact),
        "budget_pairs": args.budget_pairs,
        "suites": suites_summary,
    }
    write_json(workdir / "normalize_values_suite_eval_summary.json", output)
    Path(args.out_md).write_text(build_report(output), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
