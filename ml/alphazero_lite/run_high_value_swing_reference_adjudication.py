#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
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
    format_bool,
    format_float,
    load_jsonl,
    load_reference_maps,
    python_bin,
    repo_root,
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
DEFAULT_SELECTED_ROWS_PATH = Path(
    "/tmp/azlite_corrected_non_opening_failure_mining_v2/"
    "selected_non_opening_family_rows_v2.jsonl"
)
DEFAULT_SUMMARY_OUT = Path(
    "/tmp/azlite_high_value_swing_reference_adjudication/"
    "high_value_swing_reference_adjudication_summary.json"
)
DEFAULT_PATCH_OUT = Path(
    "/tmp/azlite_high_value_swing_reference_adjudication/"
    "high_value_swing_reference_review_patch_v1.json"
)
DEFAULT_REPORT_OUT = Path(
    "docs/alphazero-lite-high-value-swing-reference-adjudication-results.md"
)
FAMILY = "high_value_swing"
SCHEMA = "azlite_high_value_swing_reference_adjudication_v1"
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
ADJUDICATED_ROW_IDS = (
    "high_value_swing-007",
    "high_value_swing-025",
    "high_value_swing-023",
    "high_value_swing-001",
    "high_value_swing-021",
    "high_value_swing-013",
    "high_value_swing-008",
    "high_value_swing-003",
    "high_value_swing-016",
    "high_value_swing-020",
    "high_value_swing-009",
    "high_value_swing-015",
    "high_value_swing-024",
    "high_value_swing-018",
)
HOLDOUT_ROW_IDS = (
    "high_value_swing-011",
    "high_value_swing-022",
)
PRESERVATION_CONTROL_ROW_IDS = (
    "high_value_swing-010",
    "high_value_swing-026",
    "high_value_swing-027",
    "high_value_swing-017",
)
ALL_REPORTED_ROW_IDS = (
    *ADJUDICATED_ROW_IDS,
    *HOLDOUT_ROW_IDS,
    *PRESERVATION_CONTROL_ROW_IDS,
)
PR52_CLASSIFICATION = {
    "high_value_swing-007": "corrected_reference_suspicious",
    "high_value_swing-025": "corrected_reference_suspicious",
    "high_value_swing-023": "corrected_reference_suspicious",
    "high_value_swing-001": "corrected_reference_suspicious",
    "high_value_swing-021": "corrected_reference_suspicious",
    "high_value_swing-013": "corrected_reference_suspicious",
    "high_value_swing-008": "corrected_reference_suspicious",
    "high_value_swing-003": "corrected_reference_suspicious",
    "high_value_swing-016": "corrected_reference_suspicious",
    "high_value_swing-020": "corrected_reference_suspicious",
    "high_value_swing-009": "corrected_reference_suspicious",
    "high_value_swing-015": "corrected_reference_suspicious",
    "high_value_swing-024": "puct_child_search_mismatch",
    "high_value_swing-018": "inconclusive",
    "high_value_swing-011": "holdout_context",
    "high_value_swing-022": "holdout_context",
    "high_value_swing-010": "preservation_control",
    "high_value_swing-026": "preservation_control",
    "high_value_swing-027": "preservation_control",
    "high_value_swing-017": "preservation_control",
}
FALLBACK_ROLE_BY_ROW = {
    **{row_id: "target_candidate" for row_id in ADJUDICATED_ROW_IDS},
    **{row_id: "holdout_candidate" for row_id in HOLDOUT_ROW_IDS},
    **{row_id: "preservation_control" for row_id in PRESERVATION_CONTROL_ROW_IDS},
}
PR52_CONTEXT = {
    "family_classification": "reference_family_uncertain",
    "value_head_miscalibration_count": 0,
    "puct_child_search_mismatch_count": 1,
    "root_selection_pressure_count": 0,
    "corrected_reference_suspicious_count": 12,
    "inconclusive_count": 1,
    "blocked_next_action": "adjudicate high_value_swing references before training.",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--suite-path", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument(
        "--current-artifact", type=Path, default=DEFAULT_CURRENT_ARTIFACT
    )
    parser.add_argument(
        "--selected-rows-path", type=Path, default=DEFAULT_SELECTED_ROWS_PATH
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


def rerun_family_mining_if_needed(root: Path, args: argparse.Namespace) -> None:
    if args.selected_rows_path.exists():
        return
    command = [
        python_bin(root),
        "-m",
        "ml.alphazero_lite.run_corrected_non_opening_failure_family_mining_v2",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            "missing selected rows file and rerun failed with exit code "
            f"{completed.returncode}"
        )
    if not args.selected_rows_path.exists():
        raise SystemExit(
            f"selected rows file still missing after rerun: {args.selected_rows_path}"
        )


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(row) + " |")
    return output


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


def terminal_state_raw_value(state: dict[str, Any]) -> float:
    game = KalahGame.from_state(state)
    settled_scores = game.captured_seeds.copy()
    for player in (0, 1):
        start = player * 6
        settled_scores[player] += sum(game.pits[start : start + 6])
    current_player = int(game.current_player)
    opponent = 1 - current_player
    if settled_scores[current_player] > settled_scores[opponent]:
        return 1.0
    if settled_scores[current_player] < settled_scores[opponent]:
        return -1.0
    return 0.0


def load_selected_row_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    return {
        str(row["row_id"]): row
        for row in rows
        if isinstance(row, dict) and row.get("row_id") is not None
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
    q_by_move = {
        int(child["move"]): round_float(float(child.get("q_value", 0.0))) or 0.0
        for child in list(result.get("child_stats") or [])
        if child.get("move") is not None
    }
    prior_by_move = {
        int(move): round_float(float(result.get("policy", [])[move])) or 0.0
        for move in legal_moves
        if move < len(result.get("policy", []))
    }
    visit_share_by_move = {
        int(move): round_float(float(visits[move]) / float(sum(visits)))
        if sum(visits) > 0 and move < len(visits)
        else None
        for move in legal_moves
    }
    return {
        "selected_move": selected_move,
        "visit_share_by_move": visit_share_by_move,
        "q_by_move": q_by_move,
        "prior_by_move": prior_by_move,
        "root_value": round_float(float(result.get("value", 0.0))),
        "legal_moves": legal_moves,
    }


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
    game = KalahGame.from_state(child_state)
    summary = ClassicMCTS(game, simulations=int(budget), seed=int(seed)).root_summary()
    selected_move = summary.get("selected_move")
    raw_value = 0.0
    for child in list(summary.get("child_stats") or []):
        if selected_move is not None and int(child["move"]) == int(selected_move):
            raw_value = (2.0 * float(child.get("win_rate", 0.5))) - 1.0
            break
    if selected_move is None and game.over():
        raw_value = terminal_state_raw_value(child_state)
    root_value = state_to_root_perspective_value(
        raw_value=raw_value, state=child_state, root_player=root_player
    )
    return {
        "selected_move": None if selected_move is None else int(selected_move),
        "raw_value": round_float(raw_value),
        "root_value": round_float(root_value),
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
        child_game = KalahGame.from_state(child_state_from_move(state, move))
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


def representative_optional_budget_state(
    valid_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not valid_rows:
        return None
    chosen_row = max(
        valid_rows,
        key=lambda row: (
            len(list(row.get("legal_moves") or [])),
            sum(KalahGame.from_state(row["suite_state"]).pits),
            str(row["row_id"]),
        ),
    )
    return dict(chosen_row["suite_state"])


def high_budget_keys(summary_map: dict[str, Any], *, minimum: int) -> list[int]:
    keys = sorted(int(key) for key in summary_map)
    filtered = [key for key in keys if key >= minimum]
    return filtered if filtered else keys[-2:]


def consistent_majority(
    summary_map: dict[str, Any], *, minimum: int
) -> tuple[bool, int | None]:
    keys = high_budget_keys(summary_map, minimum=minimum)
    majority_moves = [summary_map[str(key)].get("majority_move") for key in keys]
    fractions = [
        float(summary_map[str(key)].get("majority_fraction") or 0.0) for key in keys
    ]
    if not majority_moves or any(move is None for move in majority_moves):
        return False, None
    if len(set(int(move) for move in majority_moves)) != 1:
        return False, None
    if any(fraction < 0.71 for fraction in fractions):
        return False, int(majority_moves[0])
    return True, int(majority_moves[0])


def child_support_label(
    child_comparison_rows: list[dict[str, Any]],
    *,
    comparison_move: int | None,
) -> str:
    if comparison_move is None:
        return "not_needed"
    relevant = [
        row
        for row in child_comparison_rows
        if row.get("comparison_move") == int(comparison_move)
        and int(row["budget"]) >= 2400
    ]
    if not relevant:
        return "mixed"
    deltas = [
        float(row.get("active_minus_comparison_value") or 0.0) for row in relevant
    ]
    stable_flags = [bool(row.get("stable_support")) for row in relevant]
    if all(delta > 0.01 for delta in deltas) and all(stable_flags):
        return "supports_active_reference"
    if all(delta < -0.01 for delta in deltas) and all(stable_flags):
        return "supports_comparison_move"
    if any(abs(delta) <= 0.01 for delta in deltas):
        return "mixed"
    return "mixed"


def classify_row_result(row: dict[str, Any]) -> dict[str, Any]:
    reference_move = int(row["active_reference_move"])
    root_consistent, root_consensus_move = consistent_majority(
        row["root_budget_summaries"], minimum=5000
    )
    puct_consistent, puct_consensus_move = consistent_majority(
        row["puct_budget_summaries"], minimum=1200
    )
    comparison_move = row.get("classic_majority_move")
    child_support = child_support_label(
        row.get("child_comparison_rows", []), comparison_move=comparison_move
    )
    tablebase_root_move = row.get("tablebase_root_preferred_move")
    tablebase_supports_reference = (
        tablebase_root_move is None or int(tablebase_root_move) == reference_move
    )
    tablebase_supports_consensus = (
        tablebase_root_move is None
        or root_consensus_move is None
        or int(tablebase_root_move) == int(root_consensus_move)
    )
    if (
        root_consistent
        and root_consensus_move is not None
        and root_consensus_move != reference_move
    ):
        if child_support == "supports_comparison_move" and tablebase_supports_consensus:
            decision = "reference_should_flip"
            proposed_reference_move = int(root_consensus_move)
            proposed_unstable = False
            recommended_use = "requires reviewed reference patch before training use"
            notes = "high-budget ClassicMCTS root and child-afterstate evidence consistently prefer another move"
        elif child_support == "mixed":
            decision = "still_inconclusive"
            proposed_reference_move = None
            proposed_unstable = False
            recommended_use = "exclude pending more evidence"
            notes = "root search prefers another move, but child-afterstate evidence stays mixed"
        else:
            decision = "still_inconclusive"
            proposed_reference_move = None
            proposed_unstable = False
            recommended_use = "exclude pending more evidence"
            notes = (
                "root search and child-afterstate do not align cleanly enough to flip"
            )
    elif root_consistent and root_consensus_move == reference_move:
        if (
            child_support in {"supports_active_reference", "not_needed"}
            and tablebase_supports_reference
        ):
            if (
                puct_consistent
                and puct_consensus_move is not None
                and puct_consensus_move != reference_move
            ):
                decision = "puct_teacher_divergence"
                proposed_reference_move = None
                proposed_unstable = False
                recommended_use = "diagnostic-only until teacher policy is chosen"
                notes = "ClassicMCTS supports the active reference while deterministic PUCT keeps selecting away"
            else:
                decision = "active_reference_confirmed"
                proposed_reference_move = None
                proposed_unstable = False
                recommended_use = (
                    "eligible under active references if the family remains usable"
                )
                notes = "root ClassicMCTS, child-afterstate adjudication, and available tablebase evidence support the active reference"
        elif child_support == "mixed":
            decision = "still_inconclusive"
            proposed_reference_move = None
            proposed_unstable = False
            recommended_use = "exclude pending more evidence"
            notes = "root search supports the active reference, but child-afterstate evidence stays mixed"
        else:
            decision = "still_inconclusive"
            proposed_reference_move = None
            proposed_unstable = False
            recommended_use = "exclude pending more evidence"
            notes = "active reference is not strongly enough supported after child-afterstate review"
    else:
        decision = "reference_unstable"
        proposed_reference_move = None
        proposed_unstable = True
        recommended_use = "exclude from hard gates and training targets"
        notes = "high-budget seeds or budgets do not converge on one stable move"
    highest_root = row["root_budget_summaries"][str(row["highest_root_budget"])]
    highest_puct = row["puct_budget_summaries"][str(row["highest_puct_budget"])]
    return {
        "adjudicated_decision": decision,
        "proposed_reference_move": proposed_reference_move,
        "proposed_unstable": proposed_unstable,
        "recommended_use": recommended_use,
        "evidence_summary": (
            f"highest_root_majority={highest_root.get('majority_move')} fraction="
            f"{format_float(highest_root.get('majority_fraction'))}; "
            f"highest_puct_selected={highest_puct.get('puct_selected_move')}; "
            f"child_support={child_support}; "
            f"tablebase_root_move={tablebase_root_move if tablebase_root_move is not None else '-'}"
        ),
        "notes": notes,
    }


def projected_buckets(
    *,
    row_decision_table: list[dict[str, Any]],
    validation_table: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    confirmed_targets = sorted(
        row["row_id"]
        for row in row_decision_table
        if row["adjudicated_decision"] == "active_reference_confirmed"
    )
    flip_candidates = sorted(
        row["row_id"]
        for row in row_decision_table
        if row["adjudicated_decision"] == "reference_should_flip"
    )
    unstable_or_excluded = sorted(
        row["row_id"]
        for row in row_decision_table
        if row["adjudicated_decision"] in {"reference_unstable", "still_inconclusive"}
    )
    divergence_rows = sorted(
        row["row_id"]
        for row in row_decision_table
        if row["adjudicated_decision"] == "puct_teacher_divergence"
    )
    confirmed_controls = sorted(
        row["row_id"]
        for row in validation_table
        if row["row_id"] in PRESERVATION_CONTROL_ROW_IDS
        and row["status"] == "ok"
        and row["current_selected_1200"] == row["active_reference_move"]
        and row["current_selected_2400"] == row["active_reference_move"]
    )
    holdout_rows = sorted(
        row["row_id"]
        for row in validation_table
        if row["row_id"] in HOLDOUT_ROW_IDS and row["status"] == "ok"
    )
    rows = [
        {
            "bucket": "confirmed_active_reference_targets",
            "row_count": len(confirmed_targets),
            "rows": confirmed_targets,
            "train_target_eligible_count": len(confirmed_targets),
            "risks": "still limited to rows with confirmed active references only",
            "next_action": "usable only if the family still clears the final branch decision",
        },
        {
            "bucket": "reference_flip_candidates",
            "row_count": len(flip_candidates),
            "rows": flip_candidates,
            "train_target_eligible_count": 0,
            "risks": "fixture changes need explicit review before they are safe to use",
            "next_action": "review the proposed patch artifact before any training",
        },
        {
            "bucket": "unstable_or_excluded",
            "row_count": len(unstable_or_excluded),
            "rows": unstable_or_excluded,
            "train_target_eligible_count": 0,
            "risks": "unstable or mixed labels would contaminate any hard target set",
            "next_action": "exclude from hard gates and training targets",
        },
        {
            "bucket": "puct_teacher_divergence_rows",
            "row_count": len(divergence_rows),
            "rows": divergence_rows,
            "train_target_eligible_count": 0,
            "risks": "teacher-policy ambiguity remains unresolved for these rows",
            "next_action": "decide teacher policy before any training use",
        },
        {
            "bucket": "confirmed_preservation_controls",
            "row_count": len(confirmed_controls),
            "rows": confirmed_controls,
            "train_target_eligible_count": 0,
            "risks": "controls must stay unchanged during any follow-up",
            "next_action": "preserve unchanged as regression checks",
        },
        {
            "bucket": "holdout_context_rows",
            "row_count": len(holdout_rows),
            "rows": holdout_rows,
            "train_target_eligible_count": 0,
            "risks": "holdouts should remain context-only during follow-up",
            "next_action": "report only; do not train on them in this branch",
        },
    ]
    flip_count = len(flip_candidates)
    confirmed_count = len(confirmed_targets)
    unstable_count = len(unstable_or_excluded)
    divergence_count = len(divergence_rows)
    adjudicated_count = len(row_decision_table)
    if flip_count >= 4:
        decision = "reference_patch_needed"
        recommendation = "review and apply the proposed patch artifact, then rerun high_value_swing value/backup audit."
    elif (
        confirmed_count >= max(8, (adjudicated_count * 3 + 4) // 5)
        and confirmed_count >= 6
    ):
        decision = "references_clean_enough_for_value_calibration"
        recommendation = "build a small train-only high_value_swing value-calibration artifact with preservation controls; no arena until local value metrics improve."
    elif unstable_count >= 4:
        decision = "reference_suite_too_noisy_for_high_value_swing"
        recommendation = "exclude unstable rows and either target the smaller stable bucket or select the next non-opening family."
    elif divergence_count >= 4:
        decision = "teacher_family_divergence"
        recommendation = "decide teacher policy for high_value_swing before training, as we did for incumbent_proxy_disagreement."
    else:
        decision = "still_reference_uncertain"
        recommendation = "do not train on high_value_swing; rerun non-opening family mining and pick the next family."
    return (
        rows,
        {
            "projected_family_decision": decision,
            "recommended_next_action": recommendation,
            "target_candidate_rows_eligible_under_active_references": len(
                confirmed_targets
            ),
            "target_candidate_rows_requiring_reference_patch": len(flip_candidates),
            "target_candidate_rows_excluded": len(unstable_or_excluded)
            + len(divergence_rows),
            "high_value_swing_good_training_target_after_adjudication": decision
            == "references_clean_enough_for_value_calibration",
        },
    )


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-lite High Value Swing Reference Adjudication Results",
        "",
        "## 1. Context",
        "",
        "- No training was run.",
        "- No arena was run.",
        "- No model was promoted.",
        "- No replay/value-calibration artifacts were created.",
        "- Active corrected references were not mutated.",
        f"- Active corrected references: `{summary['inputs']['reference_path']}`.",
        f"- Current artifact: `{summary['inputs']['current_artifact']}`.",
        f"- Selected family rows path: `{summary['inputs']['selected_rows_path']}`.",
        "",
        "## 2. Why PR #52 blocked value training",
        "",
        "- PR #52 did not confirm a value-head family gap for `high_value_swing`.",
        f"- Family-level classification: `{summary['pr52_context']['family_classification']}`.",
        f"- Mechanism counts: `{json.dumps(summary['pr52_context'], sort_keys=True)}`.",
        f"- Exact PR #52 recommendation: **{summary['pr52_context']['blocked_next_action']}**",
        "",
        "## 3. Row validation",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "row_id",
                "pr52_classification",
                "active_reference_move",
                "legal",
                "reference_unstable",
                "canonical_state_match",
                "current_selected_384",
                "current_selected_1200",
                "current_selected_2400",
                "status",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["pr52_classification"],
                    str(row["active_reference_move"])
                    if row["active_reference_move"] is not None
                    else "-",
                    format_bool(row["legal"]),
                    format_bool(row["reference_unstable"]),
                    format_bool(row["canonical_state_match"]),
                    str(row["current_selected_384"])
                    if row["current_selected_384"] is not None
                    else "-",
                    str(row["current_selected_1200"])
                    if row["current_selected_1200"] is not None
                    else "-",
                    str(row["current_selected_2400"])
                    if row["current_selected_2400"] is not None
                    else "-",
                    row["status"],
                    row["notes"],
                ]
                for row in summary["validation_table"]
            ],
        )
    )
    lines.extend(
        [
            "",
            f"- Perspective conversion: `{summary['perspective_conversion']['conversion_rule']}`.",
            f"- Repo convention: `{summary['perspective_conversion']['implementation_rule']}`.",
            "",
            "## 4. Root ClassicMCTS adjudication",
            "",
        ]
    )
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
                    str(row["majority_move"])
                    if row["majority_move"] is not None
                    else "-",
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
                "active_reference_move",
                "comparison_move",
                "budget",
                "active_reference_child_value_root_mean",
                "comparison_child_value_root_mean",
                "active_minus_comparison_value",
                "child_selected_moves",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["active_reference_move"]),
                    str(row["comparison_move"]),
                    str(row["budget"]),
                    format_float(row["active_reference_child_value_root_mean"]),
                    format_float(row["comparison_child_value_root_mean"]),
                    format_float(row["active_minus_comparison_value"]),
                    row["child_selected_moves"],
                    row["notes"],
                ]
                for row in summary["child_adjudication_table"]
            ],
        )
    )
    lines.extend(["", "## 6. Tablebase availability", ""])
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
                    str(row["tablebase_preferred_move"])
                    if row["tablebase_preferred_move"] is not None
                    else "-",
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
                    str(row["puct_selected_move"])
                    if row["puct_selected_move"] is not None
                    else "-",
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
                    str(row["proposed_reference_move"])
                    if row["proposed_reference_move"] is not None
                    else "-",
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
        lines.append("- The active fixture remained unchanged.")
    else:
        lines.append(
            "- No patch artifact was written because no rows were classified as `reference_should_flip` or `reference_unstable`."
        )
    lines.extend(["", "## 10. Projected clean targetability", ""])
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
            f"- Target-candidate rows still eligible under active references: `{summary['target_candidate_rows_eligible_under_active_references']}`.",
            f"- Target-candidate rows that would require reference patching: `{summary['target_candidate_rows_requiring_reference_patch']}`.",
            f"- Target-candidate rows that should stay excluded from training: `{summary['target_candidate_rows_excluded']}`.",
            f"- `high_value_swing` remains a good training target after adjudication: `{str(bool(summary['high_value_swing_good_training_target_after_adjudication'])).lower()}`.",
            f"- Projected family decision: `{summary['projected_family_decision']}`.",
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
    root = repo_root()
    rerun_family_mining_if_needed(root, args)

    suite = load_suite(args.suite_path)
    suite_by_id = {position.id: position for position in suite}
    reference_by_id, reference_by_canonical = load_reference_maps(args.reference_path)
    selected_row_map = load_selected_row_map(args.selected_rows_path)
    evaluator = ArtifactEvaluator(args.current_artifact)

    validation_table: list[dict[str, Any]] = []
    valid_adjudicated_rows: list[dict[str, Any]] = []
    for row_id in ALL_REPORTED_ROW_IDS:
        suite_row = suite_by_id.get(row_id)
        selected_row = selected_row_map.get(row_id, {})
        reference_row = None
        canonical_hash = None
        if suite_row is not None:
            canonical_hash = canonical_state_key(dict(suite_row.state))
            reference_row = reference_by_id.get(row_id) or reference_by_canonical.get(
                canonical_hash
            )
        active_reference_move = (
            None if reference_row is None else reference_row.get("reference_move")
        )
        legal_moves = [] if suite_row is None else list(suite_row.legal_moves)
        legal = (
            None
            if active_reference_move is None
            else int(active_reference_move) in legal_moves
        )
        reference_unstable = (
            None
            if reference_row is None
            else bool(reference_row.get("reference_unstable", False))
        )
        reference_canonical = None
        canonical_state_match = False
        if reference_row is not None and suite_row is not None:
            reference_canonical = str(
                reference_row.get("canonical_state")
                or canonical_state_key(reference_row["state"])
            )
            canonical_state_match = reference_canonical == canonical_hash
        current_selected_384 = None
        current_selected_1200 = None
        current_selected_2400 = None
        notes: list[str] = []
        status = "ok"
        if suite_row is None or reference_row is None:
            status = "reference_integrity_error"
            notes.append("missing from suite or active reference fixture")
        else:
            puct_384 = deterministic_puct_run(
                evaluator, dict(suite_row.state), budget=384
            )
            puct_1200 = deterministic_puct_run(
                evaluator, dict(suite_row.state), budget=1200
            )
            puct_2400 = deterministic_puct_run(
                evaluator, dict(suite_row.state), budget=2400
            )
            current_selected_384 = puct_384["selected_move"]
            current_selected_1200 = puct_1200["selected_move"]
            current_selected_2400 = puct_2400["selected_move"]
            if not canonical_state_match:
                status = "reference_integrity_error"
                notes.append("canonical state hash mismatch")
            if not legal:
                status = "reference_integrity_error"
                notes.append("active reference move is illegal")
            if reference_unstable:
                notes.append("active reference already marked unstable")
        role = str(selected_row.get("recommended_role") or FALLBACK_ROLE_BY_ROW[row_id])
        if not notes:
            if row_id in HOLDOUT_ROW_IDS:
                notes.append("reported as holdout context only")
            elif row_id in PRESERVATION_CONTROL_ROW_IDS:
                notes.append("reported as preservation control only")
            else:
                notes.append("validated for adjudication")
        validation_row = {
            "row_id": row_id,
            "role": role,
            "pr52_classification": PR52_CLASSIFICATION[row_id],
            "active_reference_move": None
            if active_reference_move is None
            else int(active_reference_move),
            "legal": None if legal is None else bool(legal),
            "reference_unstable": reference_unstable,
            "canonical_state_match": bool(canonical_state_match),
            "current_selected_384": current_selected_384,
            "current_selected_1200": current_selected_1200,
            "current_selected_2400": current_selected_2400,
            "status": status,
            "notes": "; ".join(notes),
            "observed_reference_moves": []
            if reference_row is None
            else list(reference_row.get("observed_reference_moves") or []),
            "canonical_state_hash": canonical_hash,
            "legal_moves": legal_moves,
        }
        validation_table.append(validation_row)
        if (
            row_id in ADJUDICATED_ROW_IDS
            and status == "ok"
            and suite_row is not None
            and reference_row is not None
            and active_reference_move is not None
        ):
            valid_adjudicated_rows.append(
                {
                    "row_id": row_id,
                    "role": role,
                    "suite_state": dict(suite_row.state),
                    "legal_moves": legal_moves,
                    "canonical_state_hash": canonical_hash,
                    "active_reference_move": int(active_reference_move),
                    "reference_unstable": bool(reference_unstable),
                    "observed_reference_moves": list(
                        reference_row.get("observed_reference_moves") or []
                    ),
                    "pr52_classification": PR52_CLASSIFICATION[row_id],
                    "current_selected_384": current_selected_384,
                    "current_selected_1200": current_selected_1200,
                    "current_selected_2400": current_selected_2400,
                }
            )

    optional_30000_skip_reason = None
    root_budgets: list[int] = list(ROOT_BUDGETS)
    if valid_adjudicated_rows and not args.skip_optional_30000:
        optional_budget_state = representative_optional_budget_state(
            valid_adjudicated_rows
        )
        can_run_30000, skip_reason = estimate_optional_budget(
            optional_budget_state
            if optional_budget_state is not None
            else valid_adjudicated_rows[0]["suite_state"],
            row_count=len(valid_adjudicated_rows),
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
    tablebase_table: list[dict[str, Any]] = []
    puct_comparison_table: list[dict[str, Any]] = []
    row_results: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    tablebase = EndgameTablebase()

    for row in valid_adjudicated_rows:
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
                        "active_reference_visits": int(
                            run["visits_by_move"].get(reference_move, 0)
                        ),
                        "active_reference_value": run["q_by_move"].get(reference_move),
                        "selected_minus_reference_value": None
                        if selected_move is None
                        else round_float(
                            float(run["q_by_move"].get(int(selected_move), 0.0))
                            - float(run["q_by_move"].get(reference_move, 0.0))
                        ),
                        "selected_is_reference": selected_move == reference_move,
                        "top1_margin": run["top1_margin"],
                        "duration_ms": run["duration_ms"],
                    }
                )
            majority_move, majority_fraction = majority_from_counter(
                selected_counter, len(ROOT_SEEDS)
            )
            top1_margin_mean, _top1_margin_std = mean_std(top1_margins)
            decision_note = ""
            if (
                majority_move == reference_move
                and float(majority_fraction or 0.0) >= 0.71
            ):
                decision = "supports_active_reference"
                decision_note = "high-budget majority stays on the active reference"
            elif majority_move is not None and float(majority_fraction or 0.0) >= 0.71:
                decision = "supports_flip_candidate"
                decision_note = "high-budget majority prefers another move"
            else:
                decision = "unstable_or_mixed"
                decision_note = "seed majorities remain mixed at this budget"
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
                "notes": decision_note,
            }
            root_budget_summaries[str(budget)] = budget_summary
            root_adjudication_table.append(
                {
                    **budget_summary,
                    "observed_top_moves": counter_to_text(selected_counter),
                }
            )

        highest_root_budget = max(root_budgets)
        classic_majority_move = root_budget_summaries[str(highest_root_budget)][
            "majority_move"
        ]
        current_puct_comparison_move = row["current_selected_2400"]
        candidate_moves = [reference_move]
        for move in (classic_majority_move, current_puct_comparison_move):
            if move is not None and int(move) not in candidate_moves:
                candidate_moves.append(int(move))

        child_states = {
            move: child_state_from_move(state, int(move)) for move in candidate_moves
        }
        child_budget_summaries: dict[str, dict[str, Any]] = {}
        child_comparison_rows: list[dict[str, Any]] = []
        highest_child_majorities: dict[int, int | None] = {}
        for budget in CHILD_BUDGETS:
            per_move_summary: dict[str, Any] = {}
            for move, child_state in child_states.items():
                raw_values: list[float] = []
                root_values: list[float] = []
                selected_counter: Counter[int] = Counter()
                for seed in CHILD_SEEDS:
                    child_run = classic_child_run(
                        child_state,
                        budget=int(budget),
                        seed=int(seed),
                        root_player=root_player,
                    )
                    if child_run["raw_value"] is not None:
                        raw_values.append(float(child_run["raw_value"]))
                    if child_run["root_value"] is not None:
                        root_values.append(float(child_run["root_value"]))
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
                child_majority_move, child_majority_fraction = majority_from_counter(
                    selected_counter, len(CHILD_SEEDS)
                )
                per_move_summary[str(move)] = {
                    "child_from_move": int(move),
                    "child_raw_value_mean": raw_mean,
                    "child_value_root_mean": root_mean,
                    "child_value_root_std": root_std,
                    "child_selected_moves": dict(selected_counter),
                    "child_majority_move": child_majority_move,
                    "child_majority_fraction": child_majority_fraction,
                }
                if int(budget) == max(CHILD_BUDGETS):
                    highest_child_majorities[int(move)] = child_majority_move
            child_budget_summaries[str(budget)] = per_move_summary
            reference_summary = per_move_summary[str(reference_move)]
            for comparison_move in candidate_moves:
                if comparison_move == reference_move:
                    continue
                comparison_summary = per_move_summary[str(comparison_move)]
                reference_mean = reference_summary["child_value_root_mean"]
                comparison_mean = comparison_summary["child_value_root_mean"]
                active_minus_comparison = None
                if reference_mean is not None and comparison_mean is not None:
                    active_minus_comparison = round_float(
                        float(reference_mean) - float(comparison_mean)
                    )
                stable_support = (
                    active_minus_comparison is not None
                    and abs(float(active_minus_comparison)) > 0.01
                    and float(reference_summary["child_value_root_std"] or 0.0) <= 0.35
                    and float(comparison_summary["child_value_root_std"] or 0.0) <= 0.35
                )
                label = "classic_majority"
                if (
                    comparison_move == current_puct_comparison_move
                    and comparison_move != classic_majority_move
                ):
                    label = "current_puct_2400"
                child_row = {
                    "row_id": row_id,
                    "active_reference_move": reference_move,
                    "comparison_move": int(comparison_move),
                    "comparison_label": label,
                    "budget": int(budget),
                    "active_reference_child_value_root_mean": reference_mean,
                    "comparison_child_value_root_mean": comparison_mean,
                    "active_minus_comparison_value": active_minus_comparison,
                    "child_selected_moves": (
                        f"ref[{counter_to_text(Counter({int(k): int(v) for k, v in reference_summary['child_selected_moves'].items()}))}] "
                        f"cmp[{counter_to_text(Counter({int(k): int(v) for k, v in comparison_summary['child_selected_moves'].items()}))}]"
                    ),
                    "stable_support": stable_support,
                    "notes": (
                        f"comparison={label}; same player to move after child => +1, otherwise sign -1"
                    ),
                }
                child_comparison_rows.append(child_row)
                child_adjudication_table.append(child_row)

        puct_budget_summaries: dict[str, dict[str, Any]] = {}
        for budget in PUCT_BUDGETS:
            puct_run = deterministic_puct_run(evaluator, state, budget=int(budget))
            puct_selected_move = puct_run["selected_move"]
            puct_budget_summaries[str(budget)] = {
                "row_id": row_id,
                "budget": int(budget),
                "active_reference_move": reference_move,
                "puct_selected_move": puct_selected_move,
                "majority_move": puct_selected_move,
                "majority_fraction": 1.0 if puct_selected_move is not None else None,
                "puct_reference_visit_share": puct_run["visit_share_by_move"].get(
                    reference_move
                ),
                "puct_selected_visit_share": None
                if puct_selected_move is None
                else puct_run["visit_share_by_move"].get(int(puct_selected_move)),
                "puct_agrees_with_classic_majority": puct_selected_move
                == classic_majority_move,
                "puct_agrees_with_active_reference": puct_selected_move
                == reference_move,
                "visit_share_by_move": puct_run["visit_share_by_move"],
                "q_by_move": puct_run["q_by_move"],
                "prior_by_move": puct_run["prior_by_move"],
                "notes": "deterministic artifact PUCT with PR #52 settings",
            }
            puct_comparison_table.append(puct_budget_summaries[str(budget)])

        tablebase_root_value, tablebase_root_move = tablebase_preferred_move(
            tablebase, state, root_player=root_player
        )
        tablebase_table.append(
            {
                "row_id": row_id,
                "state_label": "root",
                "remaining_seed_count": int(sum(KalahGame.from_state(state).pits)),
                "tablebase_available": tablebase_root_value is not None,
                "tablebase_value_root": tablebase_root_value,
                "tablebase_preferred_move": tablebase_root_move,
                "agrees_with_classic_majority": None
                if tablebase_root_move is None or classic_majority_move is None
                else int(tablebase_root_move) == int(classic_majority_move),
                "agrees_with_active_reference": None
                if tablebase_root_move is None
                else int(tablebase_root_move) == reference_move,
                "notes": (
                    "exact tablebase value available under <=16 remaining seeds"
                    if tablebase_root_value is not None
                    else "not solvable under the repo threshold"
                ),
            }
        )
        for move, child_state in child_states.items():
            child_tb_value, child_tb_move = tablebase_preferred_move(
                tablebase, child_state, root_player=root_player
            )
            tablebase_table.append(
                {
                    "row_id": row_id,
                    "state_label": f"child_after_move_{move}",
                    "remaining_seed_count": int(
                        sum(KalahGame.from_state(child_state).pits)
                    ),
                    "tablebase_available": child_tb_value is not None,
                    "tablebase_value_root": child_tb_value,
                    "tablebase_preferred_move": child_tb_move,
                    "agrees_with_classic_majority": None
                    if child_tb_move is None
                    else child_tb_move == highest_child_majorities.get(int(move)),
                    "agrees_with_active_reference": None,
                    "notes": (
                        "exact tablebase value available under <=16 remaining seeds"
                        if child_tb_value is not None
                        else "not solvable under the repo threshold"
                    ),
                }
            )

        row_result = {
            "row_id": row_id,
            "canonical_state_hash": row["canonical_state_hash"],
            "pr52_classification": row["pr52_classification"],
            "active_reference_move": reference_move,
            "observed_reference_moves": row["observed_reference_moves"],
            "reference_unstable": row["reference_unstable"],
            "current_selected_384": row["current_selected_384"],
            "current_selected_1200": row["current_selected_1200"],
            "current_selected_2400": row["current_selected_2400"],
            "classic_majority_move": classic_majority_move,
            "highest_root_budget": highest_root_budget,
            "highest_child_budget": max(CHILD_BUDGETS),
            "highest_puct_budget": max(PUCT_BUDGETS),
            "root_budget_summaries": root_budget_summaries,
            "child_budget_summaries": child_budget_summaries,
            "child_comparison_rows": child_comparison_rows,
            "puct_budget_summaries": puct_budget_summaries,
            "tablebase_root_preferred_move": tablebase_root_move,
        }
        row_result.update(classify_row_result(row_result))
        row_results.append(row_result)

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
    projected_bucket_table, projected_meta = projected_buckets(
        row_decision_table=row_decision_table,
        validation_table=validation_table,
    )

    summary = {
        "schema": SCHEMA,
        "family": FAMILY,
        "inputs": {
            "reference_path": str(args.reference_path),
            "suite_path": str(args.suite_path),
            "current_artifact": str(args.current_artifact),
            "selected_rows_path": str(args.selected_rows_path),
            "patch_out": str(args.patch_out),
        },
        "guardrails": {
            "ran_training": False,
            "ran_arena": False,
            "promoted_model": False,
            "created_replay_artifacts": False,
            "mutated_active_references": False,
        },
        "pr52_context": PR52_CONTEXT,
        "perspective_conversion": {
            "conversion_rule": "+1 when child current_player == root current_player, else -1",
            "implementation_rule": "follow state_to_root_perspective_value: child values stay positive for the same player-to-move and are sign-flipped when the turn hands off",
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
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(build_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(args.summary_out),
                "patch_path": str(args.patch_out) if patch_rows else None,
                "report_path": str(args.report_out),
                "projected_family_decision": summary["projected_family_decision"],
                "recommended_next_action": summary["recommended_next_action"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
