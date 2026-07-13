#!/usr/bin/env python3
"""Focused 768:768 value-search integration diagnostic.

Reads the terminal-outcome self-play smoke summary, re-evaluates the changed
768:768 states for current and the selected best value-only candidate, and
summarizes how value/Q reordering changes root move selection.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position  # noqa: E402
from ml.alphazero_lite.cpuct_schedule import parse_cpuct_schedule_json  # noqa: E402
from ml.alphazero_lite.self_play import build_eval_search_options  # noqa: E402

SUMMARY_DEFAULT = "/tmp/azlite_terminal_outcome_selfplay_iteration/summary_metrics.json"
REPORT_DEFAULT = (
    REPO_ROOT
    / "docs/alphazero-lite-768eq-value-search-integration-diagnostic-results.md"
)
OUTPUT_JSON_NAME = "value_search_integration_768eq_diagnostic.json"


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


def selection_map(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    breakdown = result.get("selection_breakdown") or {}
    moves = breakdown.get("moves") or []
    return {
        int(entry["move"]): entry
        for entry in moves
        if isinstance(entry, dict) and entry.get("move") is not None
    }


def row_digest(result: dict[str, Any], move: int | None) -> dict[str, Any] | None:
    if move is None:
        return None
    entry = selection_map(result).get(int(move))
    if not isinstance(entry, dict):
        return None
    return {
        "move": int(move),
        "selection_score": float(entry.get("selection_score", 0.0)),
        "q_value": float(entry.get("q_value", 0.0)),
        "prior": float(entry.get("prior", 0.0)),
        "u_value": float(entry.get("u_value", 0.0)),
        "visits": int(entry.get("visits", 0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=SUMMARY_DEFAULT)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=str(REPORT_DEFAULT))
    parser.add_argument("--max-cases", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_report(summary: dict[str, Any], diagnostic: dict[str, Any]) -> str:
    aggregate = diagnostic["aggregate"]
    case_rows = []
    for case in diagnostic["cases"][:10]:
        current_selected = case["current_selected"]
        candidate_selected = case["candidate_selected"]
        case_rows.append(
            [
                case["game_index"],
                case["ply"],
                case["phase"],
                case["seat_context"],
                current_selected["move"],
                candidate_selected["move"],
                fmt(case["root_value_delta"]),
                fmt(candidate_selected["q_value"] - current_selected["q_value"]),
                fmt(
                    candidate_selected["selection_score"]
                    - current_selected["selection_score"]
                ),
                fmt(candidate_selected["prior"] - current_selected["prior"]),
            ]
        )
    lines = [
        "# AlphaZero-Lite 768:768 Value-Search Integration Diagnostic",
        "",
        f"- source summary: `{summary.get('workdir')}`",
        f"- candidate: `{diagnostic['candidate_name']}`",
        f"- current artifact: `{diagnostic['current_artifact']}`",
        f"- candidate artifact: `{diagnostic['candidate_artifact']}`",
        "",
        "## Aggregate",
        "",
        f"- changed rows analyzed: `{aggregate['rows']}`",
        f"- mean root value delta: `{fmt(aggregate['mean_root_value_delta'])}`",
        f"- mean |root value delta|: `{fmt(aggregate['mean_abs_root_value_delta'])}`",
        f"- mean selected-q delta: `{fmt(aggregate['mean_selected_q_delta'])}`",
        f"- mean selected-score delta: `{fmt(aggregate['mean_selected_score_delta'])}`",
        f"- mean selected-prior delta: `{fmt(aggregate['mean_selected_prior_delta'])}`",
        f"- candidate-selected move was teacher move: `{fmt(aggregate['candidate_selected_teacher_rate'])}`",
        f"- current-selected move was teacher move: `{fmt(aggregate['current_selected_teacher_rate'])}`",
        f"- phase counts: `{json.dumps(aggregate['phase_counts'], sort_keys=True)}`",
        f"- current->candidate move flips: `{json.dumps(aggregate['move_flip_counts'], sort_keys=True)}`",
        "",
        "## Changed Cases",
        "",
        markdown_table(
            [
                "Game",
                "Ply",
                "Phase",
                "Seat",
                "Cur move",
                "Cand move",
                "Delta V",
                "Delta Q(sel)",
                "Delta Score(sel)",
                "Delta Prior(sel)",
            ],
            case_rows or [["n/a", "", "", "", "", "", "", "", "", ""]],
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary)
    summary = read_json(summary_path)
    drift = summary.get("value_search_drift_diagnostic") or {}
    candidate_name = str(drift.get("candidate") or "")
    if not candidate_name:
        raise RuntimeError(
            "summary does not include value_search_drift_diagnostic candidate"
        )
    candidate_row = (summary.get("candidate_rows") or {}).get(candidate_name)
    if not isinstance(candidate_row, dict):
        raise RuntimeError(f"missing candidate row for {candidate_name}")
    candidate_artifact = Path(str(candidate_row["artifact_dir"]))
    current_artifact = Path(
        str((summary.get("candidate_rows") or {})["current_ref"]["artifact_dir"])
    )
    cases = list((drift.get("top_changed_cases") or [])[: args.max_cases])
    if not cases:
        raise RuntimeError("no changed cases available in diagnostic")

    cpuct_schedule = parse_cpuct_schedule_json(
        json.dumps(
            (summary.get("runtime_profile_confirmation") or {})
            .get("schedule_manifest", {})
            .get("overrides", {})
        )
    )
    default_c_puct = float(
        (summary.get("runtime_profile_confirmation") or {})
        .get("schedule_manifest", {})
        .get("default_c_puct", 1.25)
    )
    search_options = build_eval_search_options(
        root_policy_mode="deterministic", tactical_root_bias=0.0
    )
    current_evaluator = ArtifactEvaluator(current_artifact)
    candidate_evaluator = ArtifactEvaluator(candidate_artifact)

    analyzed_cases: list[dict[str, Any]] = []
    root_value_deltas: list[float] = []
    selected_q_deltas: list[float] = []
    selected_score_deltas: list[float] = []
    selected_prior_deltas: list[float] = []
    phase_counts: Counter[str] = Counter()
    move_flip_counts: Counter[str] = Counter()
    candidate_selected_teacher_hits = 0
    current_selected_teacher_hits = 0

    for index, case in enumerate(cases):
        raw_state = case.get("raw_state")
        if not isinstance(raw_state, dict):
            raise RuntimeError(
                "diagnostic case is missing raw_state; rerun smoke diagnostic"
            )
        current_result = evaluate_artifact_position(
            evaluator=current_evaluator,
            state=raw_state,
            simulations=768,
            seed=args.seed + index,
            c_puct=float(cpuct_schedule.get("768:768", default_c_puct)),
            search_options=search_options,
            ablation_mode="full",
        )
        candidate_result = evaluate_artifact_position(
            evaluator=candidate_evaluator,
            state=raw_state,
            simulations=768,
            seed=args.seed + index,
            c_puct=float(cpuct_schedule.get("768:768", default_c_puct)),
            search_options=search_options,
            ablation_mode="full",
        )
        current_selected_move = int(current_result["selected_move"])
        candidate_selected_move = int(candidate_result["selected_move"])
        current_selected = row_digest(current_result, current_selected_move) or {
            "move": current_selected_move,
            "q_value": 0.0,
            "selection_score": 0.0,
            "prior": 0.0,
        }
        candidate_selected = row_digest(candidate_result, candidate_selected_move) or {
            "move": candidate_selected_move,
            "q_value": 0.0,
            "selection_score": 0.0,
            "prior": 0.0,
        }
        current_root_value = float(
            current_result.get("search_root_value", current_result.get("value", 0.0))
        )
        candidate_root_value = float(
            candidate_result.get(
                "search_root_value", candidate_result.get("value", 0.0)
            )
        )
        root_value_delta = candidate_root_value - current_root_value
        root_value_deltas.append(root_value_delta)
        selected_q_deltas.append(
            float(candidate_selected.get("q_value", 0.0))
            - float(current_selected.get("q_value", 0.0))
        )
        selected_score_deltas.append(
            float(candidate_selected.get("selection_score", 0.0))
            - float(current_selected.get("selection_score", 0.0))
        )
        selected_prior_deltas.append(
            float(candidate_selected.get("prior", 0.0))
            - float(current_selected.get("prior", 0.0))
        )
        teacher_move = int(case.get("teacher_selected_move", -1))
        candidate_selected_teacher_hits += (
            1 if candidate_selected_move == teacher_move else 0
        )
        current_selected_teacher_hits += (
            1 if current_selected_move == teacher_move else 0
        )
        phase_counts[str(case.get("phase"))] += 1
        move_flip_counts[f"{current_selected_move}->{candidate_selected_move}"] += 1
        analyzed_cases.append(
            {
                "game_index": int(case.get("game_index", -1)),
                "ply": int(case.get("ply", 0)),
                "phase": str(case.get("phase")),
                "seat_context": str(case.get("seat_context")),
                "teacher_selected_move": teacher_move,
                "current_root_value": current_root_value,
                "candidate_root_value": candidate_root_value,
                "root_value_delta": root_value_delta,
                "current_selected": current_selected,
                "candidate_selected": candidate_selected,
            }
        )

    diagnostic = {
        "candidate_name": candidate_name,
        "current_artifact": str(current_artifact),
        "candidate_artifact": str(candidate_artifact),
        "aggregate": {
            "rows": len(analyzed_cases),
            "mean_root_value_delta": statistics.fmean(root_value_deltas)
            if root_value_deltas
            else 0.0,
            "mean_abs_root_value_delta": statistics.fmean(
                abs(value) for value in root_value_deltas
            )
            if root_value_deltas
            else 0.0,
            "mean_selected_q_delta": statistics.fmean(selected_q_deltas)
            if selected_q_deltas
            else 0.0,
            "mean_selected_score_delta": statistics.fmean(selected_score_deltas)
            if selected_score_deltas
            else 0.0,
            "mean_selected_prior_delta": statistics.fmean(selected_prior_deltas)
            if selected_prior_deltas
            else 0.0,
            "candidate_selected_teacher_rate": candidate_selected_teacher_hits
            / max(len(analyzed_cases), 1),
            "current_selected_teacher_rate": current_selected_teacher_hits
            / max(len(analyzed_cases), 1),
            "phase_counts": dict(sorted(phase_counts.items())),
            "move_flip_counts": dict(move_flip_counts.most_common(10)),
        },
        "cases": analyzed_cases,
    }

    out_json = (
        Path(args.out_json) if args.out_json else summary_path.parent / OUTPUT_JSON_NAME
    )
    out_md = Path(args.out_md)
    write_json(out_json, diagnostic)
    out_md.write_text(build_report(summary, diagnostic), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
