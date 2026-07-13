#!/usr/bin/env python3
"""Focused 768:768 search-variant diagnostic on value-driven changed rows."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position  # noqa: E402
from ml.alphazero_lite.self_play import build_eval_search_options  # noqa: E402

SUMMARY_DEFAULT = "/tmp/azlite_terminal_outcome_selfplay_iteration/summary_metrics.json"
REPORT_DEFAULT = (
    REPO_ROOT / "docs/alphazero-lite-768eq-search-variant-diagnostic-results.md"
)
OUTPUT_JSON_NAME = "search_variant_768eq_diagnostic.json"


@dataclass(frozen=True)
class Variant:
    name: str
    normalize_values: bool = False
    value_trust_schedule: dict[str, bool | float] | None = None


VARIANTS = [
    Variant(name="default"),
    Variant(name="normalize_values", normalize_values=True),
    Variant(
        name="value_trust_half",
        value_trust_schedule={
            "enabled": True,
            "opening": 0.5,
            "midgame": 0.5,
            "late": 0.5,
        },
    ),
    Variant(
        name="normalize_values_plus_value_trust_half",
        normalize_values=True,
        value_trust_schedule={
            "enabled": True,
            "opening": 0.5,
            "midgame": 0.5,
            "late": 0.5,
        },
    ),
]


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
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=str(REPORT_DEFAULT))
    parser.add_argument("--max-cases", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate_variant(
    *,
    evaluator: ArtifactEvaluator,
    raw_state: dict[str, Any],
    simulations: int,
    cpuct: float,
    seed: int,
    variant: Variant,
) -> dict[str, Any]:
    return evaluate_artifact_position(
        evaluator=evaluator,
        state=raw_state,
        simulations=simulations,
        seed=seed,
        c_puct=cpuct,
        search_options=build_eval_search_options(
            root_policy_mode="deterministic",
            tactical_root_bias=0.0,
            normalize_values=variant.normalize_values,
            value_trust_schedule=variant.value_trust_schedule,
        ),
        ablation_mode="full",
    )


def build_report(diagnostic: dict[str, Any]) -> str:
    aggregate_rows = []
    for variant_name, aggregate in diagnostic["variants"].items():
        aggregate_rows.append(
            [
                variant_name,
                fmt(float(aggregate["candidate_changed_rate_vs_current"])),
                fmt(float(aggregate["candidate_teacher_rate"])),
                fmt(float(aggregate["current_teacher_rate"])),
                fmt(float(aggregate["candidate_minus_current_teacher_rate"])),
                fmt(float(aggregate["mean_abs_root_value_delta"])),
                fmt(float(aggregate["mean_abs_selected_score_delta"])),
            ]
        )
    case_rows = []
    for row in diagnostic["sample_cases"][:12]:
        case_rows.append(
            [
                row["game_index"],
                row["ply"],
                row["phase"],
                row["teacher_move"],
                row["baseline_current_move"],
                row["baseline_candidate_move"],
                row["best_variant"],
                fmt(row["best_variant_teacher_delta"]),
            ]
        )
    lines = [
        "# AlphaZero-Lite 768:768 Search Variant Diagnostic",
        "",
        f"- source summary: `{diagnostic['source_summary']}`",
        f"- candidate artifact: `{diagnostic['candidate_artifact']}`",
        f"- cases analyzed: `{diagnostic['rows']}`",
        "",
        "## Variant Aggregate",
        "",
        markdown_table(
            [
                "Variant",
                "Cand changed rate",
                "Cand teacher rate",
                "Cur teacher rate",
                "Cand-cur teacher",
                "Mean |dV|",
                "Mean |dScore|",
            ],
            aggregate_rows,
        ),
        "",
        "## Sample Cases",
        "",
        markdown_table(
            [
                "Game",
                "Ply",
                "Phase",
                "Teacher",
                "Base cur",
                "Base cand",
                "Best variant",
                "Teacher delta",
            ],
            case_rows or [["n/a", "", "", "", "", "", "", ""]],
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary)
    summary = read_json(summary_path)
    drift = summary.get("value_search_drift_diagnostic") or {}
    changed_cases = list((drift.get("top_changed_cases") or [])[: args.max_cases])
    if not changed_cases:
        raise RuntimeError("summary has no changed drift cases")
    candidate_row = (summary.get("candidate_rows") or {}).get(
        "value_head_only_best_probe"
    )
    if not isinstance(candidate_row, dict):
        raise RuntimeError("summary missing value_head_only_best_probe")
    current_artifact = Path(
        str((summary.get("candidate_rows") or {})["current_ref"]["artifact_dir"])
    )
    candidate_artifact = Path(str(candidate_row["artifact_dir"]))
    current_evaluator = ArtifactEvaluator(current_artifact)
    candidate_evaluator = ArtifactEvaluator(candidate_artifact)
    manifest = (summary.get("runtime_profile_confirmation") or {}).get(
        "schedule_manifest", {}
    )
    cpuct = float(
        (manifest.get("overrides") or {}).get(
            "768:768", manifest.get("default_c_puct", 1.25)
        )
    )

    variant_cases: dict[str, list[dict[str, Any]]] = {
        variant.name: [] for variant in VARIANTS
    }
    sample_cases: list[dict[str, Any]] = []
    for index, case in enumerate(changed_cases):
        raw_state = case.get("raw_state")
        if not isinstance(raw_state, dict):
            raise RuntimeError("changed diagnostic case is missing raw_state")
        teacher_move = int(case.get("teacher_selected_move", -1))
        baseline_current_move = int(case.get("current_selected_move", -1))
        baseline_candidate_move = int(case.get("candidate_selected_move", -1))
        per_variant: dict[str, dict[str, Any]] = {}
        for variant in VARIANTS:
            current_result = evaluate_variant(
                evaluator=current_evaluator,
                raw_state=raw_state,
                simulations=768,
                cpuct=cpuct,
                seed=args.seed + index,
                variant=variant,
            )
            candidate_result = evaluate_variant(
                evaluator=candidate_evaluator,
                raw_state=raw_state,
                simulations=768,
                cpuct=cpuct,
                seed=args.seed + index,
                variant=variant,
            )
            current_move = int(current_result["selected_move"])
            candidate_move = int(candidate_result["selected_move"])
            current_root_value = float(
                current_result.get(
                    "search_root_value", current_result.get("value", 0.0)
                )
            )
            candidate_root_value = float(
                candidate_result.get(
                    "search_root_value", candidate_result.get("value", 0.0)
                )
            )
            per_variant[variant.name] = {
                "current_move": current_move,
                "candidate_move": candidate_move,
                "current_teacher": 1.0 if current_move == teacher_move else 0.0,
                "candidate_teacher": 1.0 if candidate_move == teacher_move else 0.0,
                "changed": 1.0 if current_move != candidate_move else 0.0,
                "abs_root_value_delta": abs(candidate_root_value - current_root_value),
                "abs_selected_score_delta": abs(
                    candidate_root_value - current_root_value
                ),
            }
            variant_cases[variant.name].append(per_variant[variant.name])
        best_variant_name = max(
            per_variant,
            key=lambda name: (
                per_variant[name]["candidate_teacher"]
                - per_variant[name]["current_teacher"],
                -per_variant[name]["changed"],
                -per_variant[name]["abs_root_value_delta"],
            ),
        )
        sample_cases.append(
            {
                "game_index": int(case.get("game_index", -1)),
                "ply": int(case.get("ply", 0)),
                "phase": str(case.get("phase", "")),
                "teacher_move": teacher_move,
                "baseline_current_move": baseline_current_move,
                "baseline_candidate_move": baseline_candidate_move,
                "best_variant": best_variant_name,
                "best_variant_teacher_delta": per_variant[best_variant_name][
                    "candidate_teacher"
                ]
                - per_variant[best_variant_name]["current_teacher"],
            }
        )

    variants_summary: dict[str, dict[str, Any]] = {}
    for variant_name, rows in variant_cases.items():
        variants_summary[variant_name] = {
            "candidate_changed_rate_vs_current": statistics.fmean(
                row["changed"] for row in rows
            )
            if rows
            else 0.0,
            "candidate_teacher_rate": statistics.fmean(
                row["candidate_teacher"] for row in rows
            )
            if rows
            else 0.0,
            "current_teacher_rate": statistics.fmean(
                row["current_teacher"] for row in rows
            )
            if rows
            else 0.0,
            "candidate_minus_current_teacher_rate": (
                statistics.fmean(row["candidate_teacher"] for row in rows)
                - statistics.fmean(row["current_teacher"] for row in rows)
            )
            if rows
            else 0.0,
            "mean_abs_root_value_delta": statistics.fmean(
                row["abs_root_value_delta"] for row in rows
            )
            if rows
            else 0.0,
            "mean_abs_selected_score_delta": statistics.fmean(
                row["abs_selected_score_delta"] for row in rows
            )
            if rows
            else 0.0,
        }

    diagnostic = {
        "source_summary": str(summary_path.parent),
        "candidate_artifact": str(candidate_artifact),
        "rows": len(changed_cases),
        "variants": variants_summary,
        "sample_cases": sample_cases,
    }
    out_json = (
        Path(args.out_json) if args.out_json else summary_path.parent / OUTPUT_JSON_NAME
    )
    out_md = Path(args.out_md)
    write_json(out_json, diagnostic)
    out_md.write_text(build_report(diagnostic), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
