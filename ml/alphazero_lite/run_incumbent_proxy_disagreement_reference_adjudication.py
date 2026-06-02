#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.forensic_suite import canonical_state_key, load_suite
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import build_eval_search_options


DEFAULT_AUDIT_SUMMARY_PATH = Path(
    "/tmp/azlite_incumbent_proxy_value_backup_audit/"
    "incumbent_proxy_value_backup_audit_summary.json"
)
DEFAULT_REFERENCE_PATH = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
DEFAULT_CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
DEFAULT_OUTPUT_DIR = Path("/tmp/azlite_incumbent_proxy_reference_adjudication")
DEFAULT_SUMMARY_OUT = (
    DEFAULT_OUTPUT_DIR / "incumbent_proxy_reference_adjudication_summary.json"
)
DEFAULT_PATCH_OUT = (
    DEFAULT_OUTPUT_DIR / "incumbent_proxy_reference_adjudication_patch.json"
)
DEFAULT_REPORT_OUT = Path(
    "docs/alphazero-lite-incumbent-proxy-disagreement-reference-adjudication-results.md"
)
SEARCH_OPTIONS = build_eval_search_options(
    fpu_mode="parent_q",
    reuse_subtree=True,
    normalize_values=True,
    root_policy_mode="deterministic",
    tactical_root_bias=0.1,
)
CLASSIC_BUDGETS = (5000, 10000)
OPTIONAL_CLASSIC_BUDGET = 20000
SEEDS = (11, 23, 37, 42, 101)
MAX_PROJECTED_OPTIONAL_SECONDS = 900.0
SCHEMA = "azlite_incumbent_proxy_disagreement_reference_adjudication_v1"
FAMILY = "incumbent_proxy_disagreement"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-summary-path", type=Path, default=DEFAULT_AUDIT_SUMMARY_PATH
    )
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--suite-path", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument(
        "--current-artifact", type=Path, default=DEFAULT_CURRENT_ARTIFACT
    )
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--patch-out", type=Path, default=DEFAULT_PATCH_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument(
        "--max-projected-optional-seconds",
        type=float,
        default=MAX_PROJECTED_OPTIONAL_SECONDS,
    )
    parser.add_argument("--skip-optional-budget", action="store_true")
    return parser.parse_args(argv)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def round_float(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def format_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return str(bool(value)).lower()


def q_from_win_rate(win_rate: float) -> float:
    return (2.0 * float(win_rate)) - 1.0


def majority_summary(moves: list[int | None]) -> dict[str, Any]:
    observed = [int(move) for move in moves if move is not None]
    counts = Counter(observed)
    if not observed:
        return {
            "observed_moves": [],
            "majority_move": None,
            "majority_fraction": None,
            "stable": False,
        }
    majority_move, majority_count = max(
        counts.items(), key=lambda item: (int(item[1]), -int(item[0]))
    )
    return {
        "observed_moves": sorted(counts),
        "majority_move": int(majority_move),
        "majority_fraction": round_float(majority_count / len(observed)),
        "stable": len(counts) == 1,
    }


def top_q_move(summary: dict[str, Any]) -> int | None:
    best_move = None
    best_q = -float("inf")
    for child in summary.get("child_stats") or []:
        move = int(child["move"])
        q_value = q_from_win_rate(float(child.get("win_rate", 0.0)))
        if q_value > best_q or (
            q_value == best_q and (best_move is None or move < best_move)
        ):
            best_q = q_value
            best_move = move
    return best_move


def classic_run_for_state(
    *, state: dict[str, Any], budget: int, seed: int
) -> dict[str, Any]:
    started = time.perf_counter()
    summary = ClassicMCTS(
        KalahGame.from_state(state), simulations=int(budget), seed=int(seed)
    ).root_summary()
    duration_ms = (time.perf_counter() - started) * 1000.0
    q_by_move: dict[int, float] = {}
    visits_by_move: dict[int, int] = {}
    for child in summary.get("child_stats") or []:
        move = int(child["move"])
        visits_by_move[move] = int(child.get("visits", 0))
        q_by_move[move] = q_from_win_rate(float(child.get("win_rate", 0.0)))
    return {
        "selected_move": summary.get("selected_move"),
        "top_q_move": top_q_move(summary),
        "q_by_move": q_by_move,
        "visits_by_move": visits_by_move,
        "duration_ms": duration_ms,
    }


def puct_baseline(
    *, evaluator: ArtifactEvaluator, artifact_path: Path, state: dict[str, Any]
) -> dict[str, Any]:
    result = evaluate_artifact_position(
        artifact_path=artifact_path,
        evaluator=evaluator,
        state=state,
        simulations=1200,
        seed=17,
        c_puct=1.25,
        search_options=dict(SEARCH_OPTIONS),
        ablation_mode="full",
    )
    return {
        "selected_move": result.get("selected_move"),
        "root_value": round_float(result.get("value")),
    }


def estimate_optional_budget(
    *,
    suspicious_rows: list[dict[str, Any]],
    sample_state: dict[str, Any],
    max_seconds: float,
) -> tuple[bool, str | None]:
    started = time.perf_counter()
    ClassicMCTS(
        KalahGame.from_state(sample_state),
        simulations=CLASSIC_BUDGETS[0],
        seed=int(SEEDS[0]),
    ).root_summary()
    elapsed = max(time.perf_counter() - started, 1e-6)
    projected = (
        elapsed
        * (OPTIONAL_CLASSIC_BUDGET / CLASSIC_BUDGETS[0])
        * len(suspicious_rows)
        * len(SEEDS)
    )
    if projected > max_seconds:
        return (
            False,
            f"skipped {OPTIONAL_CLASSIC_BUDGET} budget: projected ~{projected:.1f}s across {len(suspicious_rows)} rows and {len(SEEDS)} seeds",
        )
    return True, None


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(row) + " |")
    return output


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-lite Incumbent Proxy Disagreement Reference Adjudication Results",
        "",
        "## Context",
        "",
        f"- Focused suspicious rows came from `{summary['inputs']['audit_summary_path']}`.",
        f"- Corrected references stayed read-only at `{summary['inputs']['reference_path']}`.",
        "- No training, arena, promotion, or fixture mutation was performed.",
        "",
        "## Focus rows",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "row_id",
                "corrected_reference_move",
                "current_selected_1200",
                "classic_majority_move",
                "majority_fraction",
                "stable",
                "adjudication",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["corrected_reference_move"]),
                    str(row["current_selected_1200"]),
                    str(
                        row["highest_budget_majority_move"]
                        if row["highest_budget_majority_move"] is not None
                        else "-"
                    ),
                    format_float(row["highest_budget_majority_fraction"]),
                    format_bool(row["highest_budget_stable"]),
                    row["adjudication"],
                    row["notes"],
                ]
                for row in summary["focus_row_table"]
            ],
        )
    )
    lines.extend(["", "## Deep Classic MCTS rows", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "budget",
                "seed",
                "selected_move",
                "top_q_move",
                "corrected_reference_q",
                "selected_q",
                "selected_minus_reference_q_margin",
                "corrected_reference_visits",
                "selected_visits",
                "selected_is_reference",
            ],
            [
                [
                    row["row_id"],
                    str(row["budget"]),
                    str(row["seed"]),
                    str(
                        row["selected_move"]
                        if row["selected_move"] is not None
                        else "-"
                    ),
                    str(row["top_q_move"] if row["top_q_move"] is not None else "-"),
                    format_float(row["corrected_reference_q"]),
                    format_float(row["selected_q"]),
                    format_float(row["selected_minus_reference_q_margin"]),
                    str(row["corrected_reference_visits"]),
                    str(row["selected_visits"]),
                    format_bool(row["selected_is_reference"]),
                ]
                for row in summary["classic_rows"]
            ],
        )
    )
    if summary.get("optional_budget_skip_reason"):
        lines.extend(["", f"- {summary['optional_budget_skip_reason']}"])
    lines.extend(["", "## Suggested non-mutating patch", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "old_reference_move",
                "suggested_reference_move",
                "status",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["old_reference_move"]),
                    str(
                        row["suggested_reference_move"]
                        if row["suggested_reference_move"] is not None
                        else "-"
                    ),
                    row["status"],
                    row["notes"],
                ]
                for row in summary["patch_rows"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Exactly one recommended next action",
            "",
            f"Recommendation: **{summary['recommended_next_action']}**",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit_summary = load_json(args.audit_summary_path)
    suite = load_suite(args.suite_path)
    suite_by_id = {position.id: position for position in suite}
    reference_payload = load_json(args.reference_path)
    reference_by_id = {
        str(row["id"]): row
        for row in list(reference_payload.get("rows") or [])
        if isinstance(row, dict) and row.get("id") is not None
    }

    suspicious_target_ids = [
        str(row["row_id"])
        for row in list(audit_summary.get("classification_table") or [])
        if isinstance(row, dict)
        and str(row.get("role")) == "target_candidate"
        and str(row.get("row_classification")) == "corrected_reference_suspicious"
    ]
    if not suspicious_target_ids:
        raise SystemExit("no suspicious target rows found in audit summary")

    validation_table = {
        str(row["row_id"]): row
        for row in list(audit_summary.get("row_validation_table") or [])
        if isinstance(row, dict)
    }
    evaluator = ArtifactEvaluator(args.current_artifact)

    suspicious_rows: list[dict[str, Any]] = []
    for row_id in suspicious_target_ids:
        suite_row = suite_by_id[row_id]
        reference_row = reference_by_id[row_id]
        suspicious_rows.append(
            {
                "row_id": row_id,
                "state": dict(suite_row.state),
                "legal_moves": list(suite_row.legal_moves),
                "corrected_reference_move": int(reference_row["reference_move"]),
                "canonical_state": canonical_state_key(suite_row.state),
                "current_selected_1200": int(
                    validation_table[row_id]["current_selected_1200"]
                ),
                "current_puct": puct_baseline(
                    evaluator=evaluator,
                    artifact_path=args.current_artifact,
                    state=dict(suite_row.state),
                ),
            }
        )

    classic_budgets: list[int] = list(CLASSIC_BUDGETS)
    optional_budget_skip_reason = None
    if not args.skip_optional_budget:
        should_run_optional, optional_budget_skip_reason = estimate_optional_budget(
            suspicious_rows=suspicious_rows,
            sample_state=dict(suspicious_rows[0]["state"]),
            max_seconds=float(args.max_projected_optional_seconds),
        )
        if should_run_optional:
            classic_budgets.append(OPTIONAL_CLASSIC_BUDGET)

    classic_rows: list[dict[str, Any]] = []
    focus_row_table: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    adjudication_counts: Counter[str] = Counter()

    for row in suspicious_rows:
        highest_budget_rows: list[dict[str, Any]] = []
        for budget in classic_budgets:
            selected_moves_for_budget: list[int | None] = []
            for seed in SEEDS:
                result = classic_run_for_state(
                    state=dict(row["state"]), budget=int(budget), seed=int(seed)
                )
                selected_move = (
                    None
                    if result["selected_move"] is None
                    else int(result["selected_move"])
                )
                selected_moves_for_budget.append(selected_move)
                corrected_reference_q = result["q_by_move"].get(
                    int(row["corrected_reference_move"])
                )
                selected_q = (
                    result["q_by_move"].get(int(selected_move))
                    if selected_move is not None
                    else None
                )
                classic_row = {
                    "row_id": row["row_id"],
                    "budget": int(budget),
                    "seed": int(seed),
                    "selected_move": selected_move,
                    "top_q_move": result["top_q_move"],
                    "corrected_reference_q": round_float(corrected_reference_q),
                    "selected_q": round_float(selected_q),
                    "selected_minus_reference_q_margin": round_float(
                        float(selected_q) - float(corrected_reference_q)
                    )
                    if corrected_reference_q is not None and selected_q is not None
                    else None,
                    "corrected_reference_visits": int(
                        result["visits_by_move"].get(
                            int(row["corrected_reference_move"]), 0
                        )
                    ),
                    "selected_visits": int(
                        result["visits_by_move"].get(int(selected_move), 0)
                    )
                    if selected_move is not None
                    else 0,
                    "selected_is_reference": selected_move
                    == int(row["corrected_reference_move"]),
                    "duration_ms": round_float(result["duration_ms"]),
                }
                classic_rows.append(classic_row)
                if budget == classic_budgets[-1]:
                    highest_budget_rows.append(classic_row)

            majority = majority_summary(selected_moves_for_budget)
            row.setdefault("budget_majorities", []).append(
                {"budget": int(budget), **majority}
            )

        highest_budget_majority = row["budget_majorities"][-1]
        majority_move = highest_budget_majority["majority_move"]
        majority_fraction = highest_budget_majority["majority_fraction"]
        stable = bool(highest_budget_majority["stable"])
        corrected_reference_move = int(row["corrected_reference_move"])
        if (
            majority_move == corrected_reference_move
            and (majority_fraction or 0.0) >= 0.8
        ):
            adjudication = "reference_upheld"
            suggested_move = corrected_reference_move
            notes = "deep classic root search keeps the corrected reference"
        elif (
            majority_move is not None
            and majority_move != corrected_reference_move
            and (majority_fraction or 0.0) >= 0.8
        ):
            adjudication = "reference_overturned"
            suggested_move = int(majority_move)
            notes = "deep classic root search consistently prefers a different move"
        else:
            adjudication = "reference_uncertain"
            suggested_move = None
            notes = "deeper root search remains mixed across seeds"
        adjudication_counts[adjudication] += 1
        focus_row_table.append(
            {
                "row_id": row["row_id"],
                "corrected_reference_move": corrected_reference_move,
                "current_selected_1200": row["current_selected_1200"],
                "highest_budget_majority_move": majority_move,
                "highest_budget_majority_fraction": majority_fraction,
                "highest_budget_stable": stable,
                "adjudication": adjudication,
                "notes": notes,
            }
        )
        patch_rows.append(
            {
                "row_id": row["row_id"],
                "old_reference_move": corrected_reference_move,
                "suggested_reference_move": suggested_move,
                "status": adjudication,
                "notes": notes,
            }
        )

    if adjudication_counts["reference_uncertain"] > 0:
        recommended_next_action = "run an even deeper or exact adjudication pass only on the remaining uncertain incumbent_proxy_disagreement rows before any training."
    elif adjudication_counts["reference_overturned"] > 0:
        recommended_next_action = "prepare a non-mutating candidate reference patch for the overturned incumbent_proxy_disagreement rows, then review it before any training."
    elif adjudication_counts["reference_upheld"] == len(focus_row_table):
        recommended_next_action = "treat the corrected references as upheld and resume mechanism-specific value/PUCT debugging on the remaining incumbent_proxy_disagreement failures."
    else:
        recommended_next_action = "review the focused adjudication outputs before taking the next training-related branch."

    patch_payload = {
        "schema": "azlite_incumbent_proxy_disagreement_reference_patch_proposal_v1",
        "family": FAMILY,
        "rows": patch_rows,
    }
    summary = {
        "schema": SCHEMA,
        "inputs": {
            "audit_summary_path": str(args.audit_summary_path),
            "reference_path": str(args.reference_path),
            "suite_path": str(args.suite_path),
            "current_artifact": str(args.current_artifact),
        },
        "guardrails": {
            "ran_training": False,
            "ran_arena": False,
            "promoted_model": False,
            "mutated_corrected_references": False,
        },
        "suspicious_target_row_ids": suspicious_target_ids,
        "classic_budgets": classic_budgets,
        "seeds": list(SEEDS),
        "optional_budget_skip_reason": optional_budget_skip_reason,
        "focus_row_table": focus_row_table,
        "classic_rows": classic_rows,
        "patch_rows": patch_rows,
        "adjudication_counts": dict(adjudication_counts),
        "recommended_next_action": recommended_next_action,
    }
    write_json(args.summary_out, summary)
    write_json(args.patch_out, patch_payload)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(build_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(args.summary_out),
                "patch_path": str(args.patch_out),
                "report_path": str(args.report_out),
                "adjudication_counts": dict(adjudication_counts),
                "recommended_next_action": recommended_next_action,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
