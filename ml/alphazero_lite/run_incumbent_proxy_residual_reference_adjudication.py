#!/usr/bin/env python3

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.forensic_suite import canonical_state_key, load_suite
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_incumbent_proxy_disagreement_value_backup_audit import (
    child_state_from_move,
    load_json,
    round_float,
    state_to_root_perspective_value,
    write_json,
)
from ml.alphazero_lite.self_play import build_eval_search_options


DEFAULT_REFERENCE_PATH = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
DEFAULT_CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
DEFAULT_REBASELINE_SUMMARY_PATH = Path(
    "/tmp/azlite_incumbent_proxy_post_adjudication_rebaseline/"
    "incumbent_proxy_post_adjudication_rebaseline_summary.json"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/azlite_incumbent_proxy_residual_reference_adjudication")
DEFAULT_SUMMARY_OUT = (
    DEFAULT_OUTPUT_DIR / "residual_reference_adjudication_summary.json"
)
DEFAULT_PATCH_OUT = (
    DEFAULT_OUTPUT_DIR / "incumbent_proxy_residual_reference_review_patch_v1.json"
)
DEFAULT_REPORT_OUT = Path(
    "docs/alphazero-lite-incumbent-proxy-disagreement-residual-reference-adjudication-results.md"
)
SCHEMA = "azlite_incumbent_proxy_residual_reference_adjudication_v1"
FAMILY = "incumbent_proxy_disagreement"
SUSPICIOUS_ROW_IDS = (
    "incumbent_proxy_disagreement-007",
    "incumbent_proxy_disagreement-009",
    "incumbent_proxy_disagreement-010",
    "incumbent_proxy_disagreement-018",
    "incumbent_proxy_disagreement-021",
    "incumbent_proxy_disagreement-023",
    "incumbent_proxy_disagreement-024",
    "incumbent_proxy_disagreement-032",
    "incumbent_proxy_disagreement-033",
)
ROOT_BUDGETS = (1200, 2400, 5000, 10000)
OPTIONAL_ROOT_BUDGET = 30000
ROOT_SEEDS = (11, 23, 37, 42, 101, 202, 303)
CHILD_BUDGETS = (1200, 2400, 5000)
CHILD_SEEDS = (11, 23, 37, 42, 101)
PUCT_BUDGETS = (384, 1200, 2400, 5000)
C_PUCT = 1.25
MAX_PROJECTED_30000_SECONDS = 1800.0
SEARCH_OPTIONS = build_eval_search_options(
    fpu_mode="parent_q",
    reuse_subtree=True,
    normalize_values=True,
    root_policy_mode="deterministic",
    tactical_root_bias=0.1,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--suite-path", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument(
        "--current-artifact", type=Path, default=DEFAULT_CURRENT_ARTIFACT
    )
    parser.add_argument(
        "--rebaseline-summary-path",
        type=Path,
        default=DEFAULT_REBASELINE_SUMMARY_PATH,
    )
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--patch-out", type=Path, default=DEFAULT_PATCH_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument(
        "--max-projected-30000-seconds",
        type=float,
        default=MAX_PROJECTED_30000_SECONDS,
    )
    parser.add_argument("--skip-optional-30000", action="store_true")
    return parser.parse_args(argv)


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def format_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return str(bool(value)).lower()


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(row) + " |")
    return output


def canonical_reference_rows(
    reference_payload: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = list(reference_payload.get("rows") or [])
    by_id: dict[str, dict[str, Any]] = {}
    by_canonical: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("id") is None:
            continue
        row_id = str(row["id"])
        canonical = str(row.get("canonical_state") or canonical_state_key(row["state"]))
        by_id[row_id] = row
        by_canonical[canonical] = row
    return by_id, by_canonical


def legal_move_set(state: dict[str, Any]) -> list[int]:
    return KalahGame.from_state(state).possible_moves()


def selection_map(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    moves = (result.get("selection_breakdown") or {}).get("moves") or []
    return {
        int(entry["move"]): entry
        for entry in moves
        if isinstance(entry, dict) and entry.get("move") is not None
    }


def visit_share(visits: list[float], move: int) -> float | None:
    total = sum(float(value) for value in visits)
    if total <= 0 or move >= len(visits):
        return None
    return round_float(float(visits[move]) / total)


def q_by_move_from_puct(result: dict[str, Any]) -> dict[int, float]:
    return {
        int(child["move"]): round_float(float(child.get("q_value", 0.0))) or 0.0
        for child in list(result.get("child_stats") or [])
        if child.get("move") is not None
    }


def prior_by_move_from_puct(
    result: dict[str, Any], legal_moves: list[int]
) -> dict[int, float]:
    policy = list(result.get("policy") or [])
    return {
        int(move): round_float(float(policy[move])) or 0.0
        for move in legal_moves
        if move < len(policy)
    }


def top_margin_from_q_map(q_by_move: dict[int, float]) -> float | None:
    if len(q_by_move) < 2:
        return None
    ordered = sorted(
        q_by_move.items(), key=lambda item: (-float(item[1]), int(item[0]))
    )
    return round_float(float(ordered[0][1]) - float(ordered[1][1]))


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return round_float(mean), round_float(std)


def counter_to_text(counter: Counter[int]) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{move}:{count}" for move, count in sorted(counter.items()))


def majority_from_counter(
    counter: Counter[int], total: int
) -> tuple[int | None, float | None]:
    if not counter or total <= 0:
        return None, None
    move, count = max(counter.items(), key=lambda item: (int(item[1]), -int(item[0])))
    return int(move), round_float(count / total)


def classic_root_run(
    state: dict[str, Any], *, budget: int, seed: int
) -> dict[str, Any]:
    started = time.perf_counter()
    summary = ClassicMCTS(
        KalahGame.from_state(state), simulations=int(budget), seed=int(seed)
    ).root_summary()
    duration_ms = (time.perf_counter() - started) * 1000.0
    q_by_move: dict[int, float] = {}
    visits_by_move: dict[int, int] = {}
    for child in list(summary.get("child_stats") or []):
        move = int(child["move"])
        visits_by_move[move] = int(child.get("visits", 0))
        q_by_move[move] = (
            round_float((2.0 * float(child.get("win_rate", 0.0))) - 1.0) or 0.0
        )
    return {
        "selected_move": None
        if summary.get("selected_move") is None
        else int(summary["selected_move"]),
        "visits_by_move": visits_by_move,
        "q_by_move": q_by_move,
        "top1_margin": top_margin_from_q_map(q_by_move),
        "duration_ms": round_float(duration_ms),
    }


def classic_child_run(
    child_state: dict[str, Any], *, budget: int, seed: int, root_player: int
) -> dict[str, Any]:
    summary = ClassicMCTS(
        KalahGame.from_state(child_state), simulations=int(budget), seed=int(seed)
    ).root_summary()
    selected_move = summary.get("selected_move")
    raw_value = 0.0
    for child in list(summary.get("child_stats") or []):
        if selected_move is not None and int(child["move"]) == int(selected_move):
            raw_value = (2.0 * float(child.get("win_rate", 0.5))) - 1.0
            break
    root_value = state_to_root_perspective_value(
        raw_value=raw_value, state=child_state, root_player=root_player
    )
    return {
        "selected_move": None if selected_move is None else int(selected_move),
        "raw_value": round_float(raw_value),
        "root_value": round_float(root_value),
    }


def deterministic_puct_run(
    evaluator: ArtifactEvaluator, state: dict[str, Any], *, budget: int
) -> dict[str, Any]:
    result = evaluate_artifact_position(
        artifact_path=None,
        evaluator=evaluator,
        state=state,
        simulations=int(budget),
        seed=17,
        c_puct=C_PUCT,
        search_options=dict(SEARCH_OPTIONS),
        ablation_mode="full",
    )
    legal_moves = [int(move) for move in result.get("legal_moves") or []]
    visits = [float(value) for value in list(result.get("visits") or [])]
    selected_move = (
        None if result.get("selected_move") is None else int(result["selected_move"])
    )
    return {
        "selected_move": selected_move,
        "visit_share_by_move": {
            int(move): visit_share(visits, int(move)) for move in legal_moves
        },
        "q_by_move": q_by_move_from_puct(result),
        "prior_by_move": prior_by_move_from_puct(result, legal_moves),
        "root_value": round_float(float(result.get("value", 0.0))),
        "legal_moves": legal_moves,
        "selection_breakdown": result.get("selection_breakdown") or {},
    }


def estimate_optional_budget(
    state: dict[str, Any], *, row_count: int, max_seconds: float
) -> tuple[bool, str | None]:
    started = time.perf_counter()
    ClassicMCTS(
        KalahGame.from_state(state), simulations=5000, seed=int(ROOT_SEEDS[0])
    ).root_summary()
    elapsed = max(time.perf_counter() - started, 1e-6)
    projected = elapsed * (OPTIONAL_ROOT_BUDGET / 5000.0) * row_count * len(ROOT_SEEDS)
    if projected > max_seconds:
        return (
            False,
            (
                f"skipped {OPTIONAL_ROOT_BUDGET} budget: projected ~{projected:.1f}s "
                f"across {row_count} rows and {len(ROOT_SEEDS)} seeds"
            ),
        )
    return True, None


def tablebase_preferred_move(
    tablebase: EndgameTablebase, state: dict[str, Any], *, root_player: int
) -> tuple[float | None, int | None]:
    game = KalahGame.from_state(state)
    value = tablebase.lookup_cached(game, root_player)
    if value is None:
        value = tablebase.lookup(game, root_player)
    if value is None:
        return None, None
    legal_moves = game.possible_moves()
    if not legal_moves:
        return round_float((2.0 * float(value)) - 1.0), None
    best_move = None
    best_value = None
    for move in legal_moves:
        child_state = child_state_from_move(state, move)
        child_game = KalahGame.from_state(child_state)
        child_value = tablebase.lookup_cached(child_game, root_player)
        if child_value is None:
            child_value = tablebase.lookup(child_game, root_player)
        if child_value is None:
            return round_float((2.0 * float(value)) - 1.0), None
        candidate = float(child_value)
        if (
            best_value is None
            or candidate > best_value
            or (candidate == best_value and (best_move is None or move < best_move))
        ):
            best_value = candidate
            best_move = int(move)
    return round_float((2.0 * float(value)) - 1.0), best_move


def validation_rows(
    *,
    suite_by_id: dict[str, Any],
    reference_by_id: dict[str, dict[str, Any]],
    evaluator: ArtifactEvaluator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    for row_id in SUSPICIOUS_ROW_IDS:
        suite_row = suite_by_id.get(row_id)
        reference_row = reference_by_id.get(row_id)
        if suite_row is None or reference_row is None:
            output_rows.append(
                {
                    "row_id": row_id,
                    "active_reference_move": None,
                    "legal": None,
                    "reference_unstable": None,
                    "canonical_state_match": False,
                    "current_selected_384": None,
                    "current_selected_1200": None,
                    "status": "reference_integrity_error",
                    "notes": "missing from suite or active references",
                }
            )
            continue
        suite_state = dict(suite_row.state)
        suite_canonical = canonical_state_key(suite_state)
        reference_canonical = str(
            reference_row.get("canonical_state")
            or canonical_state_key(reference_row["state"])
        )
        canonical_match = suite_canonical == reference_canonical
        reference_move = reference_row.get("reference_move")
        legal_moves = list(suite_row.legal_moves)
        legal = reference_move is not None and int(reference_move) in legal_moves
        probe_384 = deterministic_puct_run(evaluator, suite_state, budget=384)
        probe_1200 = deterministic_puct_run(evaluator, suite_state, budget=1200)
        status = "ok"
        notes: list[str] = []
        if not canonical_match:
            status = "reference_integrity_error"
            notes.append("canonical state hash mismatch")
        if not legal:
            status = "reference_integrity_error"
            notes.append("active reference move is illegal")
        if bool(reference_row.get("reference_unstable", False)):
            notes.append("active reference already marked unstable")
        output_rows.append(
            {
                "row_id": row_id,
                "active_reference_move": None
                if reference_move is None
                else int(reference_move),
                "legal": None if reference_move is None else bool(legal),
                "reference_unstable": bool(
                    reference_row.get("reference_unstable", False)
                ),
                "canonical_state_match": bool(canonical_match),
                "current_selected_384": probe_384["selected_move"],
                "current_selected_1200": probe_1200["selected_move"],
                "status": status,
                "notes": "; ".join(notes)
                if notes
                else "validated against active references",
                "observed_reference_moves": list(
                    reference_row.get("observed_reference_moves") or []
                ),
                "canonical_state_hash": suite_canonical,
                "legal_moves": legal_moves,
            }
        )
        if status == "ok" and reference_move is not None:
            valid_rows.append(
                {
                    "row_id": row_id,
                    "suite_state": suite_state,
                    "canonical_state_hash": suite_canonical,
                    "active_reference_move": int(reference_move),
                    "observed_reference_moves": list(
                        reference_row.get("observed_reference_moves") or []
                    ),
                    "reference_unstable": bool(
                        reference_row.get("reference_unstable", False)
                    ),
                    "legal_moves": legal_moves,
                    "current_selected_384": probe_384["selected_move"],
                    "current_selected_1200": probe_1200["selected_move"],
                }
            )
    return valid_rows, output_rows


def classify_row(row: dict[str, Any]) -> dict[str, Any]:
    reference_move = int(row["active_reference_move"])
    highest_root = row["root_budget_summaries"][str(row["highest_root_budget"])]
    highest_child = row["child_budget_summaries"][str(row["highest_child_budget"])]
    highest_puct = row["puct_budget_summaries"][str(row["highest_puct_budget"])]
    majority_move = highest_root.get("majority_move")
    majority_fraction = float(highest_root.get("majority_fraction") or 0.0)
    reference_fraction = float(highest_root.get("reference_selected_fraction") or 0.0)
    majority_child = (
        highest_child.get(str(majority_move), {}) if majority_move is not None else {}
    )
    puct_selected_move = highest_puct.get("puct_selected_move")
    majority_delta_vs_reference = majority_child.get("delta_vs_reference")
    reference_beats_majority = False
    if majority_move == reference_move:
        reference_beats_majority = True
    elif majority_delta_vs_reference is not None:
        reference_beats_majority = float(majority_delta_vs_reference) < -0.01
    root_puct_agrees_reference = puct_selected_move == reference_move
    root_classic_supports_reference = (
        majority_move == reference_move and majority_fraction >= 0.71
    )
    root_classic_supports_flip = (
        majority_move != reference_move and majority_fraction >= 0.71
    )
    root_unstable = majority_fraction < 0.71
    decisions_seen = {
        budget_summary["majority_move"]
        for budget_summary in row["root_budget_summaries"].values()
        if budget_summary.get("majority_move") is not None
    }
    high_budget_decisions_seen = {
        budget_summary["majority_move"]
        for budget_key, budget_summary in row["root_budget_summaries"].items()
        if int(budget_key) >= 5000 and budget_summary.get("majority_move") is not None
    }
    if len(decisions_seen) >= 3 and majority_fraction < 0.86:
        root_unstable = True
    if len(high_budget_decisions_seen) >= 2:
        root_unstable = True
    if (
        root_classic_supports_flip
        and reference_beats_majority is False
        and not root_unstable
    ):
        decision = "reference_should_flip"
        proposed_reference_move = majority_move
        proposed_unstable = False
        recommended_use = "review patch before any training"
        reason = "high-budget ClassicMCTS consistently prefers another move"
    elif (
        root_classic_supports_reference
        and reference_beats_majority
        and root_puct_agrees_reference
    ):
        decision = "active_reference_confirmed"
        proposed_reference_move = None
        proposed_unstable = False
        recommended_use = "keep as active reference; exclude from training unless it remains a stable mechanism row"
        reason = (
            "classic root, child-afterstate, and PUCT all support the active reference"
        )
    elif (
        root_classic_supports_reference
        and reference_beats_majority
        and not root_puct_agrees_reference
    ):
        decision = "puct_teacher_divergence"
        proposed_reference_move = None
        proposed_unstable = False
        recommended_use = "diagnostic-only until teacher/reference policy is chosen"
        reason = "ClassicMCTS supports the active reference while deterministic PUCT keeps selecting away"
    elif root_unstable:
        decision = "reference_unstable"
        proposed_reference_move = None
        proposed_unstable = True
        recommended_use = "exclude from hard gates and training targets"
        reason = "seeds or budgets do not converge on one stable move"
    else:
        decision = "still_inconclusive"
        proposed_reference_move = None
        proposed_unstable = False
        recommended_use = "keep excluded pending more evidence"
        reason = "evidence stays mixed across root, child, and PUCT adjudication"
    return {
        "adjudicated_decision": decision,
        "proposed_reference_move": proposed_reference_move,
        "proposed_unstable": proposed_unstable,
        "recommended_use": recommended_use,
        "evidence_summary": (
            f"highest_budget_majority={majority_move} fraction={format_float(majority_fraction)}; "
            f"reference_fraction={format_float(reference_fraction)}; "
            f"highest_puct_selected={puct_selected_move}; "
            f"child_reference_beats_majority={str(reference_beats_majority).lower()}; "
            f"root_puct_agrees_reference={str(bool(root_puct_agrees_reference)).lower()}"
        ),
        "notes": reason,
    }


def projected_buckets(
    rebaseline_summary: dict[str, Any], row_results: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_table = {
        str(entry["mechanism"]): entry
        for entry in list(rebaseline_summary.get("mechanism_summary_table") or [])
        if isinstance(entry, dict) and entry.get("mechanism") is not None
    }
    buckets = {
        "confirmed_value_head_miscalibration_candidates": list(
            summary_table.get("stable_value_head_miscalibration", {}).get("rows") or []
        ),
        "confirmed_root_selection_pressure_candidates": list(
            summary_table.get("stable_root_selection_pressure", {}).get("rows") or []
        ),
        "confirmed_puct_child_mismatch_candidates": list(
            summary_table.get("stable_puct_child_mismatch", {}).get("rows") or []
        ),
        "reference_flip_candidates": [],
        "unstable_or_excluded": [],
        "stable_controls": list(
            summary_table.get("controls_stable", {}).get("rows") or []
        ),
    }
    for row in row_results:
        row_id = row["row_id"]
        decision = row["adjudicated_decision"]
        reference_move = int(row["active_reference_move"])
        highest_puct_selected = row["puct_budget_summaries"][
            str(row["highest_puct_budget"])
        ]
        if decision == "reference_should_flip":
            buckets["reference_flip_candidates"].append(row_id)
        elif decision in {"reference_unstable", "still_inconclusive"}:
            buckets["unstable_or_excluded"].append(row_id)
        elif decision in {"active_reference_confirmed", "puct_teacher_divergence"}:
            if highest_puct_selected.get("puct_selected_move") == reference_move:
                buckets["confirmed_value_head_miscalibration_candidates"].append(row_id)
            else:
                buckets["confirmed_root_selection_pressure_candidates"].append(row_id)
        else:
            buckets["unstable_or_excluded"].append(row_id)
    projected_rows: list[dict[str, Any]] = []
    for bucket_name, rows in buckets.items():
        unique_rows = sorted(set(rows))
        train_eligible = (
            len(unique_rows)
            if bucket_name == "confirmed_value_head_miscalibration_candidates"
            else 0
        )
        risks = {
            "confirmed_value_head_miscalibration_candidates": "teacher-confirmed labels can still overfit if the set stays small",
            "confirmed_root_selection_pressure_candidates": "root-prior pressure may need search-stack fixes before training",
            "confirmed_puct_child_mismatch_candidates": "child search disagreement may indicate backup/search defects rather than label quality",
            "reference_flip_candidates": "fixture changes need explicit review before they are safe to use",
            "unstable_or_excluded": "unstable labels would contaminate any hard target set",
            "stable_controls": "controls must stay unchanged during any follow-up",
        }[bucket_name]
        next_action = {
            "confirmed_value_head_miscalibration_candidates": "usable for small train-only value calibration if references are otherwise clean",
            "confirmed_root_selection_pressure_candidates": "diagnose root selection pressure separately from training",
            "confirmed_puct_child_mismatch_candidates": "diagnose child search mismatch before treating as training targets",
            "reference_flip_candidates": "review and apply only via a separate explicit fixture patch",
            "unstable_or_excluded": "exclude from hard pass/fail gates and training targets",
            "stable_controls": "preserve unchanged as regression checks",
        }[bucket_name]
        projected_rows.append(
            {
                "bucket": bucket_name,
                "row_count": len(unique_rows),
                "rows": unique_rows,
                "train_target_eligible_count": train_eligible,
                "risks": risks,
                "next_action": next_action,
            }
        )
    usable_value_bucket = next(
        (
            entry
            for entry in projected_rows
            if entry["bucket"] == "confirmed_value_head_miscalibration_candidates"
        ),
        None,
    )
    usable_buckets = [
        entry
        for entry in projected_rows
        if int(entry["train_target_eligible_count"]) > 0
    ]
    largest_usable_bucket = (
        max(
            usable_buckets,
            key=lambda entry: (
                int(entry["train_target_eligible_count"]),
                -len(entry["bucket"]),
            ),
        )
        if usable_buckets
        else {"bucket": None, "train_target_eligible_count": 0}
    )
    flip_count = len(buckets["reference_flip_candidates"])
    unstable_count = len(buckets["unstable_or_excluded"])
    divergence_count = sum(
        1
        for row in row_results
        if row["adjudicated_decision"] == "puct_teacher_divergence"
    )
    confirmed_or_flip_count = sum(
        1
        for row in row_results
        if row["adjudicated_decision"]
        in {"active_reference_confirmed", "reference_should_flip"}
    )
    if (
        confirmed_or_flip_count >= 7
        and usable_value_bucket is not None
        and largest_usable_bucket["bucket"]
        == "confirmed_value_head_miscalibration_candidates"
    ):
        decision = "references_clean_enough_for_value_calibration"
        recommendation = (
            "build a small train-only value-calibration artifact from confirmed stable "
            "value-head rows, with stable controls and no arena until local value metrics improve"
        )
    elif flip_count >= 3:
        decision = "reference_patch_needed"
        recommendation = "review and apply the proposed patch artifact, then rerun post-adjudication rebaseline"
    elif unstable_count >= 4:
        decision = "reference_suite_too_noisy_for_this_family"
        recommendation = "exclude unstable rows and either target the smaller stable bucket or select the next non-opening family"
    elif divergence_count >= 4:
        decision = "teacher_family_divergence"
        recommendation = "decide which teacher should define references before training"
    else:
        decision = "still_reference_uncertain"
        recommendation = "do not train on incumbent_proxy_disagreement; rerun non-opening family mining and pick the next family"
    meta = {
        "projected_family_decision": decision,
        "recommended_next_action": recommendation,
        "value_training_eligible_after_cleanup": int(
            usable_value_bucket["train_target_eligible_count"]
            if usable_value_bucket is not None
            else 0
        ),
        "stable_value_head_largest_usable_bucket": bool(
            largest_usable_bucket["bucket"]
            == "confirmed_value_head_miscalibration_candidates"
        ),
    }
    return projected_rows, meta


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-lite Incumbent Proxy Disagreement Residual Reference Adjudication Results",
        "",
        "## 1. Context",
        "",
        "- No training was run.",
        "- No arena was run.",
        "- No model was promoted.",
        "- No replay artifacts were created.",
        f"- Active references stayed read-only at `{summary['inputs']['reference_path']}`.",
        f"- Current artifact: `{summary['inputs']['current_artifact']}`.",
        "",
        "## 2. Why PR #44 blocked training",
        "",
        "- PR #44 fixed row `021`, reran the post-adjudication rebaseline, and showed that `residual_reference_suspicious` still had 9 rows.",
        "- That suspicious-reference bucket remained larger than the usable `stable_value_head_miscalibration` bucket, so training would risk baking incorrect labels into any value-calibration pass.",
        "- This run adjudicates only those residual suspicious rows and keeps the active fixture unchanged.",
        "",
        "## 3. Suspicious row validation",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "row_id",
                "active_reference_move",
                "legal",
                "reference_unstable",
                "canonical_state_match",
                "current_selected_384",
                "current_selected_1200",
                "status",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(
                        row["active_reference_move"]
                        if row["active_reference_move"] is not None
                        else "-"
                    ),
                    format_bool(row["legal"]),
                    format_bool(row["reference_unstable"]),
                    format_bool(row["canonical_state_match"]),
                    str(
                        row["current_selected_384"]
                        if row["current_selected_384"] is not None
                        else "-"
                    ),
                    str(
                        row["current_selected_1200"]
                        if row["current_selected_1200"] is not None
                        else "-"
                    ),
                    row["status"],
                    row["notes"],
                ]
                for row in summary["validation_table"]
            ],
        )
    )
    lines.extend(["", "## 4. Root ClassicMCTS adjudication", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "budget",
                "seeds",
                "active_reference_move",
                "observed_top_moves",
                "majority_move",
                "majority_fraction",
                "reference_selected_fraction",
                "top1_margin_mean",
                "decision",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["budget"]),
                    str(row["seeds"]),
                    str(row["active_reference_move"]),
                    row["observed_top_moves"],
                    str(
                        row["majority_move"]
                        if row["majority_move"] is not None
                        else "-"
                    ),
                    format_float(row["majority_fraction"]),
                    format_float(row["reference_selected_fraction"]),
                    format_float(row["top1_margin_mean"]),
                    row["decision"],
                    row["notes"],
                ]
                for row in summary["root_adjudication_table"]
            ],
        )
    )
    if summary.get("optional_30000_skip_reason"):
        lines.extend(["", f"- {summary['optional_30000_skip_reason']}"])
    lines.extend(["", "## 5. Child-afterstate adjudication", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "child_from_move",
                "budget",
                "child_value_root_mean",
                "child_value_root_std",
                "child_selected_moves",
                "root_perspective_value_delta_vs_reference",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["child_from_move"]),
                    str(row["budget"]),
                    format_float(row["child_value_root_mean"]),
                    format_float(row["child_value_root_std"]),
                    row["child_selected_moves"],
                    format_float(row["root_perspective_value_delta_vs_reference"]),
                    row["notes"],
                ]
                for row in summary["child_adjudication_table"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "- Perspective conversion used the repo's existing convention: `+1` when `child.current_player == root.current_player`, otherwise sign-flipped with `-1` to convert child values back to the root player's perspective.",
            "",
            "## 6. Tablebase availability",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "row_id",
                "state_label",
                "remaining_seed_count",
                "tablebase_available",
                "tablebase_value_root",
                "tablebase_preferred_move",
                "agrees_with_classic_majority",
                "agrees_with_active_reference",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["state_label"],
                    str(row["remaining_seed_count"]),
                    format_bool(row["tablebase_available"]),
                    format_float(row["tablebase_value_root"]),
                    str(
                        row["tablebase_preferred_move"]
                        if row["tablebase_preferred_move"] is not None
                        else "-"
                    ),
                    format_bool(row["agrees_with_classic_majority"]),
                    format_bool(row["agrees_with_active_reference"]),
                    row["notes"],
                ]
                for row in summary["tablebase_table"]
            ],
        )
    )
    lines.extend(["", "## 7. PUCT/artifact teacher comparison", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "budget",
                "active_reference_move",
                "puct_selected_move",
                "puct_reference_visit_share",
                "puct_selected_visit_share",
                "puct_agrees_with_classic_majority",
                "puct_agrees_with_active_reference",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["budget"]),
                    str(row["active_reference_move"]),
                    str(
                        row["puct_selected_move"]
                        if row["puct_selected_move"] is not None
                        else "-"
                    ),
                    format_float(row["puct_reference_visit_share"]),
                    format_float(row["puct_selected_visit_share"]),
                    format_bool(row["puct_agrees_with_classic_majority"]),
                    format_bool(row["puct_agrees_with_active_reference"]),
                    row["notes"],
                ]
                for row in summary["puct_comparison_table"]
            ],
        )
    )
    lines.extend(["", "## 8. Row decisions", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "active_reference_move",
                "adjudicated_decision",
                "proposed_reference_move",
                "proposed_unstable",
                "evidence_summary",
                "recommended_use",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["active_reference_move"]),
                    row["adjudicated_decision"],
                    str(
                        row["proposed_reference_move"]
                        if row["proposed_reference_move"] is not None
                        else "-"
                    ),
                    format_bool(row["proposed_unstable"]),
                    row["evidence_summary"],
                    row["recommended_use"],
                    row["notes"],
                ]
                for row in summary["row_decision_table"]
            ],
        )
    )
    lines.extend(["", "## 9. Proposed non-mutating patch artifact", ""])
    if summary.get("patch_artifact"):
        lines.append(f"- Proposed review artifact: `{summary['inputs']['patch_out']}`.")
        lines.append("- `do_not_auto_apply` is set for every proposed row.")
        lines.append("- The active fixture was not edited by this run.")
    else:
        lines.append(
            "- No patch artifact was written because no rows were classified as `reference_should_flip` or `reference_unstable`."
        )
    lines.extend(["", "## 10. Projected clean mechanism buckets", ""])
    lines.extend(
        markdown_table(
            [
                "bucket",
                "row_count",
                "rows",
                "train_target_eligible_count",
                "risks",
                "next_action",
            ],
            [
                [
                    row["bucket"],
                    str(row["row_count"]),
                    ", ".join(row["rows"]) if row["rows"] else "-",
                    str(row["train_target_eligible_count"]),
                    row["risks"],
                    row["next_action"],
                ]
                for row in summary["projected_bucket_table"]
            ],
        )
    )
    lines.extend(
        [
            "",
            f"- Projected family decision: `{summary['projected_family_decision']}`.",
            f"- Value-calibration rows still eligible after excluding or patching suspicious rows: `{summary['value_training_eligible_after_cleanup']}`.",
            f"- Stable value-head remains the largest usable mechanism bucket: `{str(bool(summary['stable_value_head_largest_usable_bucket'])).lower()}`.",
            "",
            "## 11. Exactly one recommended next action",
            "",
            f"Recommendation: **{summary['recommended_next_action']}**",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    suite = load_suite(args.suite_path)
    suite_by_id = {position.id: position for position in suite}
    reference_payload = load_json(args.reference_path)
    reference_by_id, _reference_by_canonical = canonical_reference_rows(
        reference_payload
    )
    rebaseline_summary = load_json(args.rebaseline_summary_path)
    evaluator = ArtifactEvaluator(args.current_artifact)
    valid_rows, validation_table = validation_rows(
        suite_by_id=suite_by_id, reference_by_id=reference_by_id, evaluator=evaluator
    )

    optional_30000_skip_reason = None
    root_budgets: list[int] = list(ROOT_BUDGETS)
    if valid_rows and not args.skip_optional_30000:
        can_run_30000, skip_reason = estimate_optional_budget(
            valid_rows[0]["suite_state"],
            row_count=len(valid_rows),
            max_seconds=float(args.max_projected_30000_seconds),
        )
        if can_run_30000:
            root_budgets.append(OPTIONAL_ROOT_BUDGET)
        else:
            optional_30000_skip_reason = skip_reason
    elif args.skip_optional_30000:
        optional_30000_skip_reason = "skipped 30000 budget by explicit flag"

    root_seed_rows: list[dict[str, Any]] = []
    root_adjudication_table: list[dict[str, Any]] = []
    child_seed_rows: list[dict[str, Any]] = []
    child_adjudication_table: list[dict[str, Any]] = []
    puct_comparison_table: list[dict[str, Any]] = []
    tablebase_table: list[dict[str, Any]] = []
    row_results: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    tablebase = EndgameTablebase()

    for row in valid_rows:
        row_id = row["row_id"]
        state = row["suite_state"]
        reference_move = int(row["active_reference_move"])
        root_player = int(state["current_player"])

        root_budget_summaries: dict[str, dict[str, Any]] = {}
        for budget in root_budgets:
            selected_counter: Counter[int] = Counter()
            top1_margins: list[float] = []
            reference_selected_count = 0
            for seed in ROOT_SEEDS:
                run = classic_root_run(state, budget=int(budget), seed=int(seed))
                selected_move = run["selected_move"]
                if selected_move is not None:
                    selected_counter[int(selected_move)] += 1
                if selected_move == reference_move:
                    reference_selected_count += 1
                if run["top1_margin"] is not None:
                    top1_margins.append(float(run["top1_margin"]))
                root_seed_rows.append(
                    {
                        "row_id": row_id,
                        "budget": int(budget),
                        "seed": int(seed),
                        "selected_move": selected_move,
                        "active_reference_move": reference_move,
                        "visits_by_move": run["visits_by_move"],
                        "q_by_move": run["q_by_move"],
                        "top1_margin": run["top1_margin"],
                        "active_reference_value": run["q_by_move"].get(reference_move),
                        "selected_value": None
                        if selected_move is None
                        else run["q_by_move"].get(int(selected_move)),
                        "selected_minus_reference_value": None
                        if selected_move is None
                        else round_float(
                            float(run["q_by_move"].get(int(selected_move), 0.0))
                            - float(run["q_by_move"].get(reference_move, 0.0))
                        ),
                        "selected_is_reference": selected_move == reference_move,
                        "duration_ms": run["duration_ms"],
                    }
                )
            majority_move, majority_fraction = majority_from_counter(
                selected_counter, len(ROOT_SEEDS)
            )
            top1_margin_mean, _top1_margin_std = mean_std(top1_margins)
            if (
                majority_move == reference_move
                and float(majority_fraction or 0.0) >= 0.71
            ):
                decision = "supports_active_reference"
                notes = "highest visit majority stays on the active reference"
            elif (
                majority_move is not None
                and majority_move != reference_move
                and float(majority_fraction or 0.0) >= 0.71
            ):
                decision = "supports_flip_candidate"
                notes = "high-budget majority prefers a different move"
            else:
                decision = "unstable_or_mixed"
                notes = "seed majorities remain mixed at this budget"
            budget_summary = {
                "row_id": row_id,
                "budget": int(budget),
                "seeds": len(ROOT_SEEDS),
                "active_reference_move": reference_move,
                "observed_top_moves": dict(selected_counter),
                "majority_move": majority_move,
                "majority_fraction": majority_fraction,
                "reference_selected_fraction": round_float(
                    reference_selected_count / len(ROOT_SEEDS)
                ),
                "top1_margin_mean": top1_margin_mean,
                "decision": decision,
                "notes": notes,
            }
            root_budget_summaries[str(budget)] = budget_summary
            root_adjudication_table.append(
                {
                    **budget_summary,
                    "observed_top_moves": counter_to_text(selected_counter),
                }
            )

        highest_root_budget = max(root_budgets)
        highest_root_summary = root_budget_summaries[str(highest_root_budget)]
        majority_move = highest_root_summary["majority_move"]
        current_puct_move = int(row["current_selected_1200"])
        candidate_moves = [reference_move]
        for move in (majority_move, current_puct_move):
            if move is not None and move not in candidate_moves:
                candidate_moves.append(int(move))

        child_states = {
            int(move): child_state_from_move(state, int(move))
            for move in candidate_moves
        }
        child_budget_summaries: dict[str, dict[str, dict[str, Any]]] = {}
        for budget in CHILD_BUDGETS:
            budget_child_summary: dict[str, dict[str, Any]] = {}
            for move, child_state in child_states.items():
                root_values: list[float] = []
                raw_values: list[float] = []
                selected_counter: Counter[int] = Counter()
                for seed in CHILD_SEEDS:
                    child_run = classic_child_run(
                        child_state,
                        budget=int(budget),
                        seed=int(seed),
                        root_player=root_player,
                    )
                    if child_run["root_value"] is not None:
                        root_values.append(float(child_run["root_value"]))
                    if child_run["raw_value"] is not None:
                        raw_values.append(float(child_run["raw_value"]))
                    if child_run["selected_move"] is not None:
                        selected_counter[int(child_run["selected_move"])] += 1
                    child_seed_rows.append(
                        {
                            "row_id": row_id,
                            "child_from_move": int(move),
                            "budget": int(budget),
                            "seed": int(seed),
                            "child_raw_value": child_run["raw_value"],
                            "child_value_root": child_run["root_value"],
                            "child_selected_move": child_run["selected_move"],
                        }
                    )
                root_mean, root_std = mean_std(root_values)
                raw_mean, _raw_std = mean_std(raw_values)
                budget_child_summary[str(move)] = {
                    "child_from_move": int(move),
                    "child_raw_value_mean": raw_mean,
                    "child_value_root_mean": root_mean,
                    "child_value_root_std": root_std,
                    "child_selected_moves_majority": majority_from_counter(
                        selected_counter, len(CHILD_SEEDS)
                    )[0],
                    "child_selected_moves": dict(selected_counter),
                    "delta_vs_reference": None,
                }
            reference_mean = budget_child_summary[str(reference_move)][
                "child_value_root_mean"
            ]
            for move in candidate_moves:
                child_mean = budget_child_summary[str(move)]["child_value_root_mean"]
                if reference_mean is None or child_mean is None:
                    delta = None
                else:
                    delta = round_float(float(child_mean) - float(reference_mean))
                budget_child_summary[str(move)]["delta_vs_reference"] = delta
                child_adjudication_table.append(
                    {
                        "row_id": row_id,
                        "child_from_move": int(move),
                        "budget": int(budget),
                        "child_value_root_mean": budget_child_summary[str(move)][
                            "child_value_root_mean"
                        ],
                        "child_value_root_std": budget_child_summary[str(move)][
                            "child_value_root_std"
                        ],
                        "child_selected_moves": counter_to_text(
                            Counter(
                                {
                                    int(candidate_move): int(candidate_count)
                                    for candidate_move, candidate_count in budget_child_summary[
                                        str(move)
                                    ]["child_selected_moves"].items()
                                }
                            )
                        ),
                        "root_perspective_value_delta_vs_reference": delta,
                        "notes": (
                            "reference child baseline"
                            if move == reference_move
                            else "negative means worse than active reference"
                        ),
                    }
                )
            child_budget_summaries[str(budget)] = budget_child_summary

        highest_child_budget = max(CHILD_BUDGETS)
        puct_budget_summaries: dict[str, dict[str, Any]] = {}
        for budget in PUCT_BUDGETS:
            puct_run = deterministic_puct_run(evaluator, state, budget=int(budget))
            puct_selected_move = puct_run["selected_move"]
            puct_budget_summaries[str(budget)] = {
                "row_id": row_id,
                "budget": int(budget),
                "active_reference_move": reference_move,
                "puct_selected_move": puct_selected_move,
                "puct_reference_visit_share": puct_run["visit_share_by_move"].get(
                    reference_move
                ),
                "puct_selected_visit_share": None
                if puct_selected_move is None
                else puct_run["visit_share_by_move"].get(puct_selected_move),
                "puct_agrees_with_classic_majority": puct_selected_move
                == majority_move,
                "puct_agrees_with_active_reference": puct_selected_move
                == reference_move,
                "visit_share_by_move": puct_run["visit_share_by_move"],
                "q_by_move": puct_run["q_by_move"],
                "prior_by_move": puct_run["prior_by_move"],
                "notes": "deterministic artifact PUCT with PR #44 audit settings",
            }
            puct_comparison_table.append(puct_budget_summaries[str(budget)])

        row_result = {
            "row_id": row_id,
            "canonical_state_hash": row["canonical_state_hash"],
            "active_reference_move": reference_move,
            "observed_reference_moves": row["observed_reference_moves"],
            "reference_unstable": row["reference_unstable"],
            "current_selected_384": row["current_selected_384"],
            "current_selected_1200": row["current_selected_1200"],
            "highest_root_budget": highest_root_budget,
            "highest_child_budget": highest_child_budget,
            "highest_puct_budget": max(PUCT_BUDGETS),
            "root_budget_summaries": root_budget_summaries,
            "child_budget_summaries": child_budget_summaries,
            "puct_budget_summaries": puct_budget_summaries,
        }
        row_result.update(classify_row(row_result))
        row_results.append(row_result)

        for label, move, tb_state in [("root", None, state)] + [
            (f"child_from_{move}", move, child_state)
            for move, child_state in child_states.items()
        ]:
            remaining_seed_count = int(sum(KalahGame.from_state(tb_state).pits))
            tablebase_value_root, tablebase_move = tablebase_preferred_move(
                tablebase, tb_state, root_player=root_player
            )
            tablebase_available = tablebase_value_root is not None
            tablebase_table.append(
                {
                    "row_id": row_id,
                    "state_label": label,
                    "remaining_seed_count": remaining_seed_count,
                    "tablebase_available": tablebase_available,
                    "tablebase_value_root": tablebase_value_root,
                    "tablebase_preferred_move": tablebase_move,
                    "agrees_with_classic_majority": None
                    if tablebase_move is None
                    else tablebase_move == majority_move,
                    "agrees_with_active_reference": None
                    if tablebase_move is None
                    else tablebase_move == reference_move,
                    "notes": (
                        "not solvable under the repo threshold"
                        if not tablebase_available
                        else "exact tablebase value available under <=16 remaining seeds"
                    ),
                }
            )

        if row_result["adjudicated_decision"] in {
            "reference_should_flip",
            "reference_unstable",
        }:
            patch_rows.append(
                {
                    "row_id": row_id,
                    "canonical_state_hash": row["canonical_state_hash"],
                    "current_active_reference_move": reference_move,
                    "proposed_reference_move": row_result["proposed_reference_move"],
                    "proposed_reference_unstable": bool(
                        row_result["proposed_unstable"]
                    ),
                    "observed_reference_moves": row["observed_reference_moves"],
                    "budgets_seeds_used": {
                        "root_budgets": root_budgets,
                        "root_seeds": list(ROOT_SEEDS),
                        "child_budgets": list(CHILD_BUDGETS),
                        "child_seeds": list(CHILD_SEEDS),
                        "puct_budgets": list(PUCT_BUDGETS),
                    },
                    "evidence_summary": row_result["evidence_summary"],
                    "reason": row_result["notes"],
                    "do_not_auto_apply": True,
                }
            )

    row_results.sort(key=lambda row: row["row_id"])
    projected_bucket_table, projected_meta = projected_buckets(
        rebaseline_summary, row_results
    )
    row_decision_table = [
        {
            "row_id": row["row_id"],
            "active_reference_move": row["active_reference_move"],
            "adjudicated_decision": row["adjudicated_decision"],
            "proposed_reference_move": row["proposed_reference_move"],
            "proposed_unstable": row["proposed_unstable"],
            "evidence_summary": row["evidence_summary"],
            "recommended_use": row["recommended_use"],
            "notes": row["notes"],
        }
        for row in row_results
    ]
    summary = {
        "schema": SCHEMA,
        "family": FAMILY,
        "inputs": {
            "reference_path": str(args.reference_path),
            "suite_path": str(args.suite_path),
            "current_artifact": str(args.current_artifact),
            "rebaseline_summary_path": str(args.rebaseline_summary_path),
            "patch_out": str(args.patch_out),
        },
        "validation_table": validation_table,
        "root_adjudication_table": root_adjudication_table,
        "root_seed_rows": root_seed_rows,
        "child_adjudication_table": child_adjudication_table,
        "child_seed_rows": child_seed_rows,
        "tablebase_table": tablebase_table,
        "puct_comparison_table": puct_comparison_table,
        "row_decision_table": row_decision_table,
        "projected_bucket_table": projected_bucket_table,
        "patch_artifact": bool(patch_rows),
        "patch_rows": patch_rows,
        "optional_30000_skip_reason": optional_30000_skip_reason,
        **projected_meta,
    }

    write_json(args.summary_out, summary)
    if patch_rows:
        write_json(
            args.patch_out,
            {
                "schema": SCHEMA,
                "family": FAMILY,
                "reference_path": str(args.reference_path),
                "suite_path": str(args.suite_path),
                "rows": patch_rows,
            },
        )
    report = build_report(summary)
    args.report_out.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
