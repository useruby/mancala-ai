#!/usr/bin/env python3
"""Preflight for balanced-current runtime PUCT override evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))


SUMMARY_SCHEMA = "azlite_balanced_current_runtime_puct_override_v1"
SUMMARY_FILENAME = "summary_metrics.json"
REPORT_PATH = (
    REPO_ROOT / "docs/alphazero-lite-balanced-current-runtime-puct-override-results.md"
)
DEFAULT_BUDGET_PAIRS = "384:256,768:256,768:768,1200:1200,1200:256,256:768"


@dataclass(frozen=True)
class SemanticsEvidence:
    benchmark_file: str
    arena_file: str
    benchmark_invocation: str
    arena_behavior: str
    classification: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="/tmp/azlite_balanced_current_runtime_puct_override"
    )
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--override-budgets", default="384,768,1200")
    parser.add_argument("--visit-share-thresholds", default="0.55,0.70")
    parser.add_argument("--budget-pairs", default=DEFAULT_BUDGET_PAIRS)
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing required file: {path}")
    return path


def parse_csv_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def parse_csv_floats(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one float value")
    return values


def parse_csv_paths(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one path")
    return values


def inspect_benchmark_semantics() -> SemanticsEvidence:
    return SemanticsEvidence(
        benchmark_file="ml/alphazero_lite/run_opening_suite_seat_benchmark.py",
        arena_file="ml/alphazero_lite/arena.py",
        benchmark_invocation=(
            "benchmark invokes arena with --root-policy-mode deterministic and "
            "challenger/current simulation budgets"
        ),
        arena_behavior=(
            "arena move selection always constructs PUCT, runs search, and chooses "
            "the move via select_root_move(root, legal_moves) when legal moves exist"
        ),
        classification="search_already_in_eval",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_summary(args: argparse.Namespace, artifact_sha: str) -> dict[str, Any]:
    evidence = inspect_benchmark_semantics()
    return {
        "schema": SUMMARY_SCHEMA,
        "date": str(date.today()),
        "classification": evidence.classification,
        "artifact": {
            "current_path": str(Path(args.current)),
            "weights_sha256": artifact_sha,
            "expected_weights_sha256": args.expected_current_weights_sha256,
        },
        "inputs": {
            "medium_suite": args.medium_suite,
            "fixed_large_suite": args.fixed_large_suite,
            "heldout_suites": parse_csv_paths(args.heldout_suites),
            "override_budgets": parse_csv_ints(args.override_budgets),
            "visit_share_thresholds": parse_csv_floats(args.visit_share_thresholds),
            "budget_pairs": [
                item.strip() for item in args.budget_pairs.split(",") if item.strip()
            ],
            "games_per_opening": int(args.games_per_opening),
            "workers": int(args.workers),
            "seed": int(args.seed),
        },
        "benchmark_semantics_check": {
            "result": evidence.classification,
            "benchmark_file": evidence.benchmark_file,
            "arena_file": evidence.arena_file,
            "benchmark_invocation": evidence.benchmark_invocation,
            "arena_behavior": evidence.arena_behavior,
            "equivalence_reason": (
                "The requested runtime override substitutes a PUCT move for a raw "
                "policy move at decision time, but deterministic opening-suite eval "
                "already makes decisions from runtime PUCT search rather than raw "
                "policy top-1."
            ),
        },
        "override_controller": {
            "implemented": False,
            "reason": evidence.classification,
        },
        "override_audit": {
            "ran": False,
            "reason": evidence.classification,
        },
        "opening_suite_evaluation": {
            "ran": False,
            "reason": evidence.classification,
        },
        "gate": {
            "ran": False,
            "classification": "not_run",
            "reason": evidence.classification,
        },
        "runtime_cost_estimate": {
            "available": False,
            "reason": evidence.classification,
        },
    }


def render_report(summary: dict[str, Any]) -> str:
    semantics = summary["benchmark_semantics_check"]
    artifact = summary["artifact"]
    inputs = summary["inputs"]
    heldout_rows = "\n".join(
        f"- `{path}`" for path in summary["inputs"]["heldout_suites"]
    )
    return "\n".join(
        [
            "# AlphaZero-Lite Balanced Current Runtime PUCT Override Results",
            "",
            f"**Date**: {summary['date']}",
            "",
            f"**Classification**: `{summary['classification']}`",
            "",
            "## Artifact Hash",
            "",
            f"- Current artifact: `{artifact['current_path']}`",
            f"- Current artifact weights SHA256: `{artifact['weights_sha256']}`",
            f"- Expected SHA256: `{artifact['expected_weights_sha256']}`",
            "",
            "## Benchmark Semantics Check",
            "",
            f"- Result: `{semantics['result']}`",
            f"- Benchmark file: `{semantics['benchmark_file']}`",
            f"- Arena file: `{semantics['arena_file']}`",
            f"- Benchmark invocation: {semantics['benchmark_invocation']}",
            f"- Arena behavior: {semantics['arena_behavior']}",
            f"- Equivalence reason: {semantics['equivalence_reason']}",
            "",
            "Deterministic opening-suite evaluation already uses runtime PUCT root search for move choice. The proposed override would therefore be behaviorally identical to existing evaluation, so the experiment stops in Phase A by design.",
            "",
            "## Override Controller Description",
            "",
            "- Not implemented beyond semantics preflight.",
            "- Reason: deterministic eval already selects moves from runtime PUCT, so a raw-policy-to-PUCT decision override does not create a distinct controller lane.",
            "",
            "## Inputs",
            "",
            f"- Medium suite: `{inputs['medium_suite']}`",
            f"- Fixed large suite: `{inputs['fixed_large_suite']}`",
            "- Held-out suites:",
            heldout_rows,
            f"- Override budgets: `{','.join(str(v) for v in inputs['override_budgets'])}`",
            f"- Visit-share thresholds: `{','.join(f'{v:.2f}' for v in inputs['visit_share_thresholds'])}`",
            f"- Budget pairs: `{','.join(inputs['budget_pairs'])}`",
            f"- Games per opening: `{inputs['games_per_opening']}`",
            f"- Workers: `{inputs['workers']}`",
            f"- Seed: `{inputs['seed']}`",
            "",
            "## Override Audit Table",
            "",
            "Not run because the semantics check classified the proposal as `search_already_in_eval`.",
            "",
            "## Fixed Large DS Table",
            "",
            "Not run because the semantics check classified the proposal as `search_already_in_eval`.",
            "",
            "## Held-Out Mean And Worst-Suite DS Table",
            "",
            "Not run because the semantics check classified the proposal as `search_already_in_eval`.",
            "",
            "## Bootstrap CI",
            "",
            "Not run because the semantics check classified the proposal as `search_already_in_eval`.",
            "",
            "## P0/P1 Split For 384:256",
            "",
            "Not run because the semantics check classified the proposal as `search_already_in_eval`.",
            "",
            "## Duplicate Trajectory Counts",
            "",
            "Not run because the semantics check classified the proposal as `search_already_in_eval`.",
            "",
            "## Gate Classification",
            "",
            "- `not_run` because the semantics check classified the proposal as `search_already_in_eval`.",
            "",
            "## Runtime Cost Estimate",
            "",
            "Not estimated because no distinct override controller was run. Existing deterministic benchmark cost already includes runtime PUCT search per move.",
        ]
    )


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    current_weights = require_file(Path(args.current) / "weights.json")
    artifact_sha = sha256_file(current_weights)
    if artifact_sha != args.expected_current_weights_sha256:
        raise ValueError(
            "current artifact weights SHA256 mismatch: "
            f"expected {args.expected_current_weights_sha256}, got {artifact_sha}"
        )

    summary = build_summary(args, artifact_sha)
    summary_path = workdir / SUMMARY_FILENAME
    write_json(summary_path, summary)
    REPORT_PATH.write_text(render_report(summary) + "\n", encoding="utf-8")

    print(f"classification={summary['classification']}")
    print(f"summary={summary_path}")
    print(f"report={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
