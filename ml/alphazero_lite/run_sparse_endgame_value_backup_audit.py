#!/usr/bin/env python3
"""
Child-afterstate value/backup audit for sparse_endgame.

Run child-afterstate value/backup audit for the sparse_endgame corrected
non-opening failure family selected by PR #63.

This audit does not train, does not run arena, does not promote, does not
create replay artifacts, and does not mutate corrected reference fixtures.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.forensic_suite import canonical_state_key, load_suite
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_incumbent_proxy_disagreement_value_backup_audit import (
    AuditRow,
    CHILD_PUCT_BUDGETS,
    CHILD_TEACHER_BASE_BUDGETS,
    CHILD_TEACHER_SEEDS,
    C_PUCT,
    OPTIONAL_CHILD_TEACHER_BUDGET,
    ROOT_BUDGETS,
    child_neural_audit,
    child_puct_audit,
    child_state_from_move,
    counterfactual_rows,
    estimate_teacher_5000_budget,
    format_bool,
    format_float,
    load_json,
    load_jsonl,
    load_reference_maps,
    mining_inventory_map,
    python_bin,
    representative_metric_map,
    repo_root,
    root_baseline_for_row,
    round_float,
    teacher_child_audit,
    write_json,
)

DEFAULT_REFERENCE_PATH = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
DEFAULT_CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
DEFAULT_SELECTED_ROWS_PATH = Path(
    "/tmp/azlite_corrected_non_opening_failure_mining_v6/"
    "selected_non_opening_family_rows_v6.jsonl"
)
DEFAULT_MINING_SUMMARY_PATH = Path(
    "/tmp/azlite_corrected_non_opening_failure_mining_v6/"
    "non_opening_failure_family_summary_v6.json"
)
DEFAULT_SUMMARY_OUT = Path(
    "/tmp/azlite_sparse_endgame_value_backup_audit/"
    "sparse_endgame_value_backup_audit_summary.json"
)
DEFAULT_REPORT_OUT = Path(
    "docs/alphazero-lite-sparse-endgame-value-backup-audit-results.md"
)
EPS = 1e-9
FAMILY = "sparse_endgame"
SCHEMA = "azlite_sparse_endgame_value_backup_audit_v1"
OPTIONAL_ROOT_BUDGET = 2400
MAX_PROJECTED_ROOT_2400_SECONDS = 600.0

# Representative target rows from PR #63
REQUIRED_TARGET_ROW_IDS = (
    "sparse_endgame-003",
    "sparse_endgame-007",
    "sparse_endgame-009",
    "sparse_endgame-024",
    "sparse_endgame-026",
    "sparse_endgame-025",
    "sparse_endgame-023",
    "sparse_endgame-021",
    "sparse_endgame-013",
    "sparse_endgame-017",
)

# Controls from PR #63
REQUIRED_CONTROL_ROW_IDS = (
    "sparse_endgame-002",
    "sparse_endgame-019",
    "sparse_endgame-020",
    "sparse_endgame-011",
)


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
    parser.add_argument(
        "--mining-summary-path", type=Path, default=DEFAULT_MINING_SUMMARY_PATH
    )
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cpuct", type=float, default=C_PUCT)
    parser.add_argument(
        "--reference-patch",
        type=Path,
        default=None,
        help="Path to a tablebase reference patch artifact to overlay in-memory only",
    )
    return parser.parse_args(argv)


def rerun_family_mining_if_needed(root: Path, args: argparse.Namespace) -> None:
    if args.selected_rows_path.exists():
        return
    command = [
        python_bin(root),
        "-m",
        "ml.alphazero_lite.run_corrected_non_opening_failure_family_mining_v6",
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


def policy_rank(root_policy: dict[str, float], move: int) -> int | None:
    ranked = sorted(
        ((int(candidate), float(prob)) for candidate, prob in root_policy.items()),
        key=lambda item: (-item[1], item[0]),
    )
    for index, (candidate, _prob) in enumerate(ranked, start=1):
        if candidate == int(move):
            return index
    return None


def baseline_notes(row: AuditRow, baseline: dict[str, Any]) -> str:
    notes = [row.role]
    if baseline["selected_move"] == row.corrected_reference_move:
        notes.append("selected reference")
    else:
        notes.append("selected away from reference")
    return ", ".join(notes)


def estimate_root_2400_budget(
    *,
    evaluator: ArtifactEvaluator,
    target_rows: list[AuditRow],
    seed: int,
    cpuct: float,
) -> tuple[bool, str]:
    if not target_rows:
        return False, "skipped 2400 root budget: no target rows"
    started = time.perf_counter()
    root_baseline_for_row(
        evaluator=evaluator,
        row=target_rows[0],
        budget=ROOT_BUDGETS[-1],
        seed=int(seed),
        cpuct=float(cpuct),
    )
    elapsed = max(time.perf_counter() - started, 1e-6)
    projected = elapsed * (OPTIONAL_ROOT_BUDGET / ROOT_BUDGETS[-1]) * len(target_rows)
    if projected > MAX_PROJECTED_ROOT_2400_SECONDS:
        return (
            False,
            "skipped 2400 root budget: projected "
            f"~{projected:.1f}s across {len(target_rows)} target rows",
        )
    return (
        True,
        "ran 2400 root budget: projected "
        f"~{projected:.1f}s across {len(target_rows)} target rows",
    )


def tablebase_preferred_move(
    tablebase: EndgameTablebase, state: dict[str, Any], *, root_player: int
) -> tuple[float | None, int | None]:
    """Return (root-perspective value, preferred move) for a state, or (None, None).

    Values are returned in the [-1.0, 1.0] root-perspective convention:
    +1.0 forced win, 0.0 forced draw, -1.0 forced loss.
    """
    game = KalahGame.from_state(state)
    win_rate = tablebase.lookup_cached(game, root_player)
    if win_rate is None:
        win_rate = tablebase.lookup(game, root_player)
    if win_rate is None:
        return None, None
    root_value = (2.0 * float(win_rate)) - 1.0
    legal_moves = game.possible_moves()
    if not legal_moves:
        return round_float(root_value), None
    best_move: int | None = None
    best_value: float | None = None
    for move in legal_moves:
        child_game = KalahGame.from_state(child_state_from_move(state, move))
        child_win_rate = tablebase.lookup_cached(child_game, root_player)
        if child_win_rate is None:
            child_win_rate = tablebase.lookup(child_game, root_player)
        if child_win_rate is None:
            return round_float(root_value), None
        candidate = float(child_win_rate)
        if (
            best_value is None
            or candidate > best_value
            or (candidate == best_value and (best_move is None or move < best_move))
        ):
            best_value = candidate
            best_move = int(move)
    return round_float(root_value), best_move


def tablebase_legal_move_values(
    tablebase: EndgameTablebase, state: dict[str, Any], *, root_player: int
) -> dict[str, Any]:
    game = KalahGame.from_state(state)
    win_rate = tablebase.lookup_cached(game, root_player)
    if win_rate is None:
        win_rate = tablebase.lookup(game, root_player)
    if win_rate is None:
        return {
            "root_value": None,
            "legal_moves": [],
            "child_value_by_move": {},
            "best_value": None,
            "optimal_moves": [],
            "preferred_move": None,
            "preferred_move_is_tiebreak": False,
            "is_forcing": False,
            "tablebase_available": False,
        }
    root_value = (2.0 * float(win_rate)) - 1.0
    legal_moves = game.possible_moves()
    child_value_by_move: dict[int, float] = {}
    best_value = -float("inf")
    for move in legal_moves:
        child_game = KalahGame.from_state(child_state_from_move(state, move))
        child_wr = tablebase.lookup_cached(child_game, root_player)
        if child_wr is None:
            child_wr = tablebase.lookup(child_game, root_player)
        if child_wr is None:
            child_value_by_move[move] = 0.0
        else:
            cv = (2.0 * float(child_wr)) - 1.0
            child_value_by_move[move] = round_float(cv)
            if cv > best_value:
                best_value = cv
    if best_value == -float("inf"):
        best_value = root_value
    optimal_moves = sorted(
        m for m, v in child_value_by_move.items() if abs(v - best_value) < EPS
    )
    preferred_move = optimal_moves[0] if optimal_moves else None
    return {
        "root_value": round_float(root_value),
        "legal_moves": legal_moves,
        "child_value_by_move": child_value_by_move,
        "best_value": round_float(best_value),
        "optimal_moves": [int(m) for m in optimal_moves],
        "preferred_move": int(preferred_move) if preferred_move is not None else None,
        "preferred_move_is_tiebreak": len(optimal_moves) > 1,
        "is_forcing": abs(root_value) >= 1.0 - EPS,
        "tablebase_available": True,
    }


def tablebase_state_summary(
    tablebase: EndgameTablebase, state: dict[str, Any], *, root_player: int
) -> dict[str, Any]:
    remaining = int(sum(KalahGame.from_state(state).pits))
    over_threshold = remaining > EndgameTablebase.MAX_SOLVED_SEEDS
    value, preferred = tablebase_preferred_move(
        tablebase, state, root_player=root_player
    )
    if over_threshold and value is None:
        notes = (
            f"not solvable under the {EndgameTablebase.MAX_SOLVED_SEEDS}-seed threshold"
        )
    elif value is None:
        notes = "tablebase lookup unavailable"
    else:
        notes = (
            f"exact tablebase value available under {EndgameTablebase.MAX_SOLVED_SEEDS} "
            "remaining seeds"
        )
    return {
        "remaining_seed_count": int(remaining),
        "tablebase_available": value is not None,
        "tablebase_value_root": value,
        "tablebase_preferred_move": preferred,
        "notes": notes,
    }


def load_reference_patch_overlay(patch_path: Path | None) -> dict[str, int]:
    """Load a reference patch artifact and return {row_id: proposed_reference_move}."""
    if patch_path is None or not patch_path.exists():
        return {}
    payload = load_json(patch_path)
    entries = payload.get("entries") or []
    overlay: dict[str, int] = {}
    for entry in entries:
        row_id = str(entry.get("row_id", ""))
        proposed = entry.get("proposed_reference_move")
        if row_id and proposed is not None:
            overlay[row_id] = int(proposed)
    if overlay:
        print(
            f"Loaded {len(overlay)} reference patch overrides from {patch_path}",
            file=sys.stderr,
        )
    return overlay


def validate_and_enrich_rows(
    args: argparse.Namespace,
) -> tuple[list[AuditRow], list[dict[str, Any]], dict[str, Any]]:
    suite = load_suite(args.suite_path)
    suite_by_id = {position.id: position for position in suite}
    reference_by_id, reference_by_canonical = load_reference_maps(args.reference_path)
    inventory_by_id = mining_inventory_map(args.mining_summary_path)
    representative_by_id = representative_metric_map(args.mining_summary_path)
    patch_overlay = load_reference_patch_overlay(args.reference_patch)
    selected_rows = load_jsonl(args.selected_rows_path)
    roles_seen: Counter[str] = Counter()
    audit_rows: list[AuditRow] = []
    validation_rows: list[dict[str, Any]] = []
    required_target_set = set(REQUIRED_TARGET_ROW_IDS)
    required_control_set = set(REQUIRED_CONTROL_ROW_IDS)

    for row in selected_rows:
        row_id = str(row["row_id"])
        role = str(row["recommended_role"])
        roles_seen[role] += 1
        if str(row.get("family")) != FAMILY:
            raise ValueError(f"{row_id} family mismatch: expected {FAMILY}")
        suite_row = suite_by_id.get(row_id)
        if suite_row is None:
            raise ValueError(f"{row_id} missing from forensic suite")
        legal_moves = list(suite_row.legal_moves)
        corrected_reference_move = int(row["corrected_reference_move"])
        if corrected_reference_move not in legal_moves:
            raise ValueError(
                f"{row_id} corrected reference move {corrected_reference_move} is illegal"
            )
        canonical = canonical_state_key(suite_row.state)
        if canonical != str(row["canonical_state_hash"]):
            raise ValueError(
                f"{row_id} canonical_state_hash does not match suite canonical state"
            )
        reference_row = reference_by_id.get(row_id) or reference_by_canonical.get(
            canonical
        )
        if reference_row is None:
            raise ValueError(f"{row_id} missing from corrected reference data")
        reference_unstable = bool(reference_row.get("reference_unstable", False))
        if reference_unstable:
            raise ValueError(f"{row_id} is marked reference_unstable")
        # Apply reference patch overlay in-memory only (no fixture mutation)
        orig_reference_move = corrected_reference_move
        if row_id in patch_overlay:
            corrected_reference_move = int(patch_overlay[row_id])
            if corrected_reference_move not in legal_moves:
                raise ValueError(
                    f"{row_id} patch overlay move {corrected_reference_move} is illegal"
                )
            print(
                f"  [patch-overlay] {row_id}: reference move {orig_reference_move} -> "
                f"{corrected_reference_move}",
                file=sys.stderr,
            )
        inventory_row = inventory_by_id.get(row_id, {})
        failure_status = str(inventory_row.get("failure_status") or "")
        if failure_status == "reference_integrity_error":
            raise ValueError(f"{row_id} is marked reference_integrity_error")
        mining_metrics = representative_by_id.get(row_id, {})
        audit_rows.append(
            AuditRow(
                row_id=row_id,
                role=role,
                severity=str(row.get("severity") or "none"),
                failure_mode=str(row.get("failure_mode") or "unknown"),
                corrected_reference_move=corrected_reference_move,
                current_selected_move=int(row["current_selected_move"]),
                suite_state=dict(suite_row.state),
                legal_moves=legal_moves,
                canonical_state_hash=canonical,
                reference_teacher_value=round_float(reference_row.get("teacher_value")),
                reference_unstable=False,
                reference_integrity_error=False,
                inventory_failure_status=failure_status or None,
                mining_metrics=dict(mining_metrics),
            )
        )
        remaining_seed_count = int(sum(KalahGame.from_state(suite_row.state).pits))
        notes = ["validated"]
        if row_id in required_target_set:
            notes.append("representative target row present")
        if row_id in required_control_set:
            notes.append("representative control row present")
        validation_rows.append(
            {
                "row_id": row_id,
                "role": role,
                "corrected_reference_move": corrected_reference_move,
                "legal": True,
                "reference_unstable": False,
                "remaining_seed_count": remaining_seed_count,
                "status": "ok",
                "notes": ", ".join(notes),
            }
        )

    for expected_role in (
        "target_candidate",
        "preservation_control",
        "holdout_candidate",
    ):
        if roles_seen[expected_role] == 0:
            raise ValueError(
                f"selected rows are missing required role bucket {expected_role}"
            )

    validation_context = {
        "required_target_rows_present": [
            row_id
            for row_id in REQUIRED_TARGET_ROW_IDS
            if row_id in {row.row_id for row in audit_rows}
        ],
        "required_control_rows_present": [
            row_id
            for row_id in REQUIRED_CONTROL_ROW_IDS
            if row_id in {row.row_id for row in audit_rows}
        ],
        "context_rows_present": [
            row_id
            for row_id in (
                "sparse_endgame-021",
                "sparse_endgame-013",
                "sparse_endgame-017",
            )
            if row_id in {row.row_id for row in audit_rows}
        ],
        "selected_row_count": len(audit_rows),
    }
    return audit_rows, validation_rows, validation_context


def build_root_baseline_record(
    row: AuditRow, baseline: dict[str, Any]
) -> dict[str, Any]:
    remaining_seed_count = int(sum(KalahGame.from_state(row.suite_state).pits))
    return {
        **baseline,
        "selected_is_reference": bool(baseline["pass"]),
        "reference_policy_probability": baseline["root_policy"].get(
            str(row.corrected_reference_move)
        ),
        "selected_policy_probability": None
        if baseline["selected_move"] is None
        else baseline["root_policy"].get(str(int(baseline["selected_move"]))),
        "reference_policy_rank": policy_rank(
            baseline["root_policy"], row.corrected_reference_move
        ),
        "selected_policy_rank": None
        if baseline["selected_move"] is None
        else policy_rank(baseline["root_policy"], int(baseline["selected_move"])),
        "remaining_seed_count": remaining_seed_count,
        "notes": baseline_notes(row, baseline),
    }


def compare_move_consequences(
    *, row: AuditRow, baseline_1200: dict[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    selected_move = baseline_1200["selected_move"]
    baseline_legal_moves = [
        int(move) for move in baseline_1200.get("legal_moves") or []
    ]
    indexed = {
        int(entry["move"]): entry
        for entry in baseline_1200.get("move_consequences") or []
    }
    rows: list[dict[str, Any]] = []
    reference_entry = indexed.get(int(row.corrected_reference_move), {})
    selected_entry = (
        indexed.get(int(selected_move), {}) if selected_move is not None else {}
    )
    notes: list[str] = []
    if selected_move is not None and selected_move != row.corrected_reference_move:
        if bool(selected_entry.get("gives_extra_turn")) and not bool(
            reference_entry.get("gives_extra_turn")
        ):
            notes.append("selected move gains extra turn")
        if int(selected_entry.get("capture_count", 0)) > int(
            reference_entry.get("capture_count", 0)
        ):
            notes.append("selected move captures more immediately")
        if int(selected_entry.get("immediate_store_delta", 0)) > int(
            reference_entry.get("immediate_store_delta", 0)
        ):
            notes.append("selected move has larger immediate store gain")
        if int(selected_entry.get("remaining_seed_count", 0)) < int(
            reference_entry.get("remaining_seed_count", 0)
        ):
            notes.append("selected move leaves fewer seeds in pits")
        if int(selected_entry.get("side_to_move_after", -1)) != int(
            row.suite_state["current_player"]
        ) and int(reference_entry.get("side_to_move_after", -1)) == int(
            row.suite_state["current_player"]
        ):
            notes.append("selected move hands turn off while reference keeps move")
        if int(selected_entry.get("side_to_move_after", -1)) == int(
            row.suite_state["current_player"]
        ) and int(reference_entry.get("side_to_move_after", -1)) != int(
            row.suite_state["current_player"]
        ):
            notes.append("selected move keeps the move")
        if bool(selected_entry.get("game_over_after_move")) and not bool(
            reference_entry.get("game_over_after_move")
        ):
            notes.append("selected move ends the game immediately")
    for move in baseline_legal_moves:
        entry = indexed.get(int(move))
        if entry is None:
            raise ValueError(
                f"{row.row_id} missing move consequence for legal move {move}"
            )
        rows.append(
            {
                "row_id": row.row_id,
                "move": int(move),
                "is_corrected_reference": int(move)
                == int(row.corrected_reference_move),
                "is_selected": int(move) == int(selected_move)
                if selected_move is not None
                else False,
                "gives_extra_turn": bool(entry["gives_extra_turn"]),
                "produces_capture": bool(entry["produces_capture"]),
                "capture_count": int(entry["capture_count"]),
                "immediate_store_delta": int(entry["immediate_store_delta"]),
                "side_to_move_after": int(entry["side_to_move_after"]),
                "game_over_after_move": bool(entry["game_over_after_move"]),
                "remaining_seed_count": int(entry["remaining_seed_count"]),
                "store_delta": int(entry["immediate_store_delta"]),
                "pit_index": int(move),
                "move_index": int(move),
                "notes": "; ".join(notes)
                if selected_move is not None
                and int(move) == int(selected_move)
                and notes
                else "",
            }
        )
    return rows, "; ".join(notes) if notes else "no obvious immediate heuristic edge"


def tablebase_audit_for_row(
    *,
    tablebase: EndgameTablebase,
    row: AuditRow,
    child_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    root_player = int(row.suite_state["current_player"])
    root_summary = tablebase_state_summary(
        tablebase, row.suite_state, root_player=root_player
    )
    child_ref_summary = tablebase_state_summary(
        tablebase, child_summary["child_ref_state"], root_player=root_player
    )
    child_selected_summary = tablebase_state_summary(
        tablebase, child_summary["child_selected_state"], root_player=root_player
    )

    root_tb_info = tablebase_legal_move_values(
        tablebase, row.suite_state, root_player=root_player
    )

    active_ref = int(row.corrected_reference_move)
    active_is_optimal = active_ref in root_tb_info["optimal_moves"]
    value_gap = None
    if root_tb_info["best_value"] is not None:
        child_val = root_tb_info["child_value_by_move"].get(active_ref)
        if child_val is not None:
            value_gap = round_float(root_tb_info["best_value"] - child_val)

    if not root_tb_info["tablebase_available"]:
        corrected_classification = "tablebase_unavailable"
    elif not active_is_optimal and value_gap is not None and value_gap > EPS:
        corrected_classification = "tablebase_real_conflict"
    elif active_is_optimal and int(root_tb_info["preferred_move"]) != active_ref:
        corrected_classification = "tablebase_tie_not_conflict"
    elif active_is_optimal:
        corrected_classification = "tablebase_confirmed"
    else:
        corrected_classification = "tablebase_ambiguous_or_error"

    return {
        "row_id": row.row_id,
        "root_player": int(root_player),
        "root": root_summary,
        "child_ref": child_ref_summary,
        "child_selected": child_selected_summary,
        "root_tablebase_agrees_with_active_reference": (
            root_summary["tablebase_preferred_move"] is not None
            and int(root_summary["tablebase_preferred_move"])
            == int(row.corrected_reference_move)
        ),
        "root_tablebase_disagrees_with_active_reference": (
            root_summary["tablebase_preferred_move"] is not None
            and int(root_summary["tablebase_preferred_move"])
            != int(row.corrected_reference_move)
        ),
        "root_legal_move_values": root_tb_info,
        "active_reference_is_optimal": bool(active_is_optimal),
        "value_gap": value_gap,
        "corrected_tablebase_classification": corrected_classification,
    }


def child_summary_payload(neural_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "child_ref_state": dict(neural_summary["child_ref_state"]),
        "child_selected_state": dict(neural_summary["child_selected_state"]),
        "child_ref_root_value": round_float(
            float(neural_summary.get("child_ref_root", 0.0))
        ),
        "child_selected_root_value": round_float(
            float(neural_summary.get("child_selected_root", 0.0))
        ),
        "neural_prefers_reference": bool(neural_summary.get("agrees", False)),
    }


def classify_row(
    *,
    row: AuditRow,
    neural_summary: dict[str, Any] | None,
    teacher_budget_summary: dict[int, dict[str, Any]] | None,
    child_puct_budget_summary: dict[int, dict[str, Any]] | None,
    root_baseline_1200: dict[str, Any],
    counterfactual: list[dict[str, Any]],
    tablebase_row: dict[str, Any] | None,
) -> dict[str, Any]:
    if row.role == "preservation_control":
        baseline_pass = bool(root_baseline_1200["pass"])
        return {
            "row_id": row.row_id,
            "role": row.role,
            "row_classification": "inconclusive",
            "supporting_evidence": (
                "preservation control retained to guard against regression"
            ),
            "recommended_use": "preserve_control",
            "notes": "baseline pass" if baseline_pass else "unexpected control failure",
        }
    if row.role == "holdout_candidate":
        return {
            "row_id": row.row_id,
            "role": row.role,
            "row_classification": "inconclusive",
            "supporting_evidence": "holdout row reserved for out-of-sample follow-up rather than mechanism counting",
            "recommended_use": "holdout",
            "notes": "root still fails"
            if not root_baseline_1200["pass"]
            else "baseline pass",
        }
    assert neural_summary is not None
    assert teacher_budget_summary is not None
    assert child_puct_budget_summary is not None
    deepest_teacher_budget = max(teacher_budget_summary)
    deepest_puct_budget = max(child_puct_budget_summary)
    teacher_prefers_reference = bool(
        teacher_budget_summary[deepest_teacher_budget][
            "teacher_prefers_corrected_reference"
        ]
    )
    teacher_stable = bool(teacher_budget_summary[deepest_teacher_budget]["stable"])
    child_puct_prefers_reference = bool(
        child_puct_budget_summary[deepest_puct_budget][
            "puct_prefers_corrected_reference"
        ]
    )
    neural_prefers_reference = bool(neural_summary["agrees"])
    root_prefers_reference = bool(root_baseline_1200["pass"])
    same_player_ref = int(neural_summary["child_ref_state"]["current_player"]) == int(
        row.suite_state["current_player"]
    )
    same_player_selected = int(
        neural_summary["child_selected_state"]["current_player"]
    ) == int(row.suite_state["current_player"])
    teacher_flip = any(
        entry["intervention"] == "teacher_child_value_override" and entry["flipped"]
        for entry in counterfactual
    )
    prior_flip = any(
        entry["intervention"] == "equalize_root_priors" and entry["flipped"]
        for entry in counterfactual
    )
    swap_flip = any(
        entry["intervention"] == "neural_child_value_swap" and entry["flipped"]
        for entry in counterfactual
    )
    corrected_tb_classification = (
        str(tablebase_row.get("corrected_tablebase_classification", ""))
        if tablebase_row
        else ""
    )
    if not teacher_stable:
        classification = "inconclusive"
        evidence = "ClassicMCTS child teacher is unstable across seeds"
    elif corrected_tb_classification == "tablebase_real_conflict":
        classification = "tablebase_reference_conflict"
        evidence = (
            "tablebase is available at the root and the active corrected reference "
            "is NOT among the optimal tablebase moves (value gap "
            + str(tablebase_row.get("value_gap", "?"))
            + ")"
        )
    elif corrected_tb_classification == "tablebase_tie_not_conflict":
        classification = "tablebase_tie_not_conflict"
        evidence = (
            "tablebase preferred move differs from active reference only by "
            "tie-breaking; active reference IS optimal in the tablebase solution"
        )
    elif not teacher_prefers_reference:
        classification = "corrected_reference_suspicious"
        evidence = (
            "ClassicMCTS child teacher does not support the corrected reference child"
        )
    elif (
        same_player_ref != same_player_selected
        and swap_flip
        and not neural_prefers_reference
    ):
        classification = "backup_perspective_suspect"
        evidence = (
            "sign-sensitive child comparison flips under value swap and perspective "
            "differs across children"
        )
    elif teacher_prefers_reference and not neural_prefers_reference:
        classification = "value_head_miscalibration"
        evidence = (
            "ClassicMCTS child teacher prefers corrected reference child but raw "
            "neural child values prefer selected child"
        )
    elif (
        teacher_prefers_reference
        and neural_prefers_reference
        and not child_puct_prefers_reference
    ):
        classification = "puct_child_search_value_mismatch"
        evidence = (
            "ClassicMCTS child teacher and neural child values prefer corrected "
            "reference child but child PUCT does not"
        )
    elif (
        teacher_prefers_reference
        and neural_prefers_reference
        and child_puct_prefers_reference
        and not root_prefers_reference
    ):
        classification = "root_selection_pressure"
        evidence = (
            "neural, ClassicMCTS child teacher, and child PUCT support corrected "
            "reference child but root PUCT still selects another move"
        )
    elif teacher_flip and not root_prefers_reference:
        classification = "puct_child_search_value_mismatch"
        evidence = "teacher child value override flips the root decision"
    elif prior_flip and not root_prefers_reference:
        classification = "root_selection_pressure"
        evidence = "equalizing root priors flips the root decision"
    elif (
        same_player_ref != same_player_selected
        and not neural_prefers_reference
        and child_puct_prefers_reference
    ):
        classification = "backup_perspective_suspect"
        evidence = (
            "disagreement is concentrated in a perspective-changing child comparison"
        )
    else:
        classification = "inconclusive"
        evidence = (
            "evidence remains mixed after neural, ClassicMCTS, child PUCT, tablebase, "
            "and counterfactual diagnostics"
        )
    return {
        "row_id": row.row_id,
        "role": row.role,
        "row_classification": classification,
        "supporting_evidence": evidence,
        "recommended_use": "target_candidate",
        "notes": "root still fails" if not root_prefers_reference else "baseline pass",
    }


def classify_family(
    row_classifications: list[dict[str, Any]],
) -> tuple[str, str, dict[str, int]]:
    target_rows = [
        row for row in row_classifications if row["role"] == "target_candidate"
    ]
    counts = Counter(str(row["row_classification"]) for row in target_rows)
    real_conflicts = counts.get("tablebase_reference_conflict", 0)
    tie_not_conflicts = counts.get("tablebase_tie_not_conflict", 0)
    combined_tablebase_issue = real_conflicts + tie_not_conflicts

    tot_non_tie = sum(
        counts.get(k, 0)
        for k in (
            "value_head_miscalibration",
            "puct_child_search_value_mismatch",
            "root_selection_pressure",
            "corrected_reference_suspicious",
            "backup_perspective_suspect",
        )
    )

    if real_conflicts >= max(2, max(0, len(target_rows)) // 3):
        return (
            "tablebase_reference_patch_needed",
            "produce a non-mutating tablebase-backed reference patch artifact for sparse_endgame and rerun the audit before training.",
            dict(counts),
        )
    if counts.get("value_head_miscalibration", 0) > max(
        counts.get("puct_child_search_value_mismatch", 0),
        counts.get("root_selection_pressure", 0),
        real_conflicts,
        counts.get("corrected_reference_suspicious", 0),
        counts.get("backup_perspective_suspect", 0),
        0,
    ):
        return (
            "value_head_family_gap",
            "build a small train-only child-afterstate value-calibration artifact for sparse_endgame, with controls and no arena until local value metrics improve.",
            dict(counts),
        )
    if counts.get("puct_child_search_value_mismatch", 0) > max(
        counts.get("value_head_miscalibration", 0),
        counts.get("root_selection_pressure", 0),
        real_conflicts,
        counts.get("corrected_reference_suspicious", 0),
        counts.get("backup_perspective_suspect", 0),
        0,
    ):
        return (
            "puct_child_search_family_gap",
            "audit child PUCT expansion/backup/value normalization before training.",
            dict(counts),
        )
    if counts.get("root_selection_pressure", 0) > max(
        counts.get("value_head_miscalibration", 0),
        counts.get("puct_child_search_value_mismatch", 0),
        real_conflicts,
        counts.get("corrected_reference_suspicious", 0),
        counts.get("backup_perspective_suspect", 0),
        0,
    ):
        return (
            "root_selection_family_gap",
            "run cpuct/root-prior calibration diagnostics for sparse_endgame.",
            dict(counts),
        )
    if counts.get("corrected_reference_suspicious", 0) > max(
        counts.get("value_head_miscalibration", 0),
        counts.get("puct_child_search_value_mismatch", 0),
        counts.get("root_selection_pressure", 0),
        real_conflicts,
        counts.get("backup_perspective_suspect", 0),
        0,
    ):
        return (
            "reference_family_uncertain",
            "adjudicate sparse_endgame references before training.",
            dict(counts),
        )
    if combined_tablebase_issue > 0 and tot_non_tie == 0:
        return (
            "tablebase_tie_logic_fixed_sparse_endgame_reauditable",
            "tablebase tie-break artifacts have been reclassified; rerun sparse_endgame value/backup audit cleanly under the fixed logic.",
            dict(counts),
        )
    return (
        "mixed_family_gap",
        "split sparse_endgame into mechanism-specific buckets and choose the largest stable bucket.",
        dict(counts),
    )


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
        "# AlphaZero-lite Sparse Endgame Value/Backup Audit Results",
        "",
        "## 1. Context",
        "",
        "- This audit evaluates child-afterstate values and backup behavior for the `sparse_endgame` corrected non-opening failure family.",
        "- No training was run.",
        "- No arena was run.",
        "- No model was promoted.",
        "- No replay artifacts were created.",
        "- Corrected references were not mutated.",
        f"- Corrected references: `{summary['inputs']['reference_path']}`.",
        f"- Current artifact: `{summary['inputs']['current_artifact']}`.",
        f"- Selected family rows: `{summary['inputs']['selected_rows_path']}`.",
        "",
        "## 2. Why sparse_endgame was selected",
        "",
        "- PR #63 selected `sparse_endgame` as the next corrected non-opening failure family.",
        f"- Family stats: `{json.dumps(summary['family_context'], sort_keys=True)}`.",
        "",
        "## 3. Row validation",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "row_id",
                "role",
                "corrected_reference_move",
                "legal",
                "reference_unstable",
                "remaining_seed_count",
                "status",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["role"],
                    str(row["corrected_reference_move"]),
                    format_bool(row["legal"]),
                    format_bool(row["reference_unstable"]),
                    str(row["remaining_seed_count"]),
                    row["status"],
                    row["notes"],
                ]
                for row in summary["row_validation_table"]
            ],
        )
    )
    lines.extend(
        [
            "",
            f"- Perspective conversion: `{summary['perspective_and_backup_check']['conversion_rule']}`.",
            f"- Implementation rule: `{summary['perspective_and_backup_check']['implementation_rule']}`.",
            "",
            "## 4. Root PUCT baseline",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "row_id",
                "role",
                "budget",
                "corrected_reference_move",
                "selected_move",
                "selected_is_reference",
                "reference_visit_share",
                "selected_visit_share",
                "reference_q",
                "selected_q",
                "selected_minus_reference_q_margin",
                "reference_policy_probability",
                "selected_policy_probability",
                "remaining_seed_count",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["role"],
                    str(row["budget"]),
                    str(row["corrected_reference_move"]),
                    str(
                        row["selected_move"]
                        if row["selected_move"] is not None
                        else "-"
                    ),
                    format_bool(row["selected_is_reference"]),
                    format_float(row["reference_visit_share"]),
                    format_float(row["selected_visit_share"]),
                    format_float(row["reference_q"]),
                    format_float(row["selected_q"]),
                    format_float(row["selected_minus_reference_q_margin"]),
                    format_float(row["reference_policy_probability"]),
                    format_float(row["selected_policy_probability"]),
                    str(row["remaining_seed_count"]),
                    row["notes"],
                ]
                for row in summary["root_baseline_table"]
            ],
        )
    )
    lines.extend(["", f"- {summary['root_2400_note']}"])
    lines.extend(["", "## 5. Move consequence and endgame audit", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "move",
                "is_corrected_reference",
                "is_selected",
                "gives_extra_turn",
                "produces_capture",
                "capture_count",
                "immediate_store_delta",
                "side_to_move_after",
                "game_over_after_move",
                "remaining_seed_count_after",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["move"]),
                    format_bool(row["is_corrected_reference"]),
                    format_bool(row["is_selected"]),
                    format_bool(row["gives_extra_turn"]),
                    format_bool(row["produces_capture"]),
                    str(row["capture_count"]),
                    str(row["immediate_store_delta"]),
                    str(row["side_to_move_after"]),
                    format_bool(row["game_over_after_move"]),
                    str(row["remaining_seed_count"]),
                    row["notes"],
                ]
                for row in summary["move_consequence_table"]
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
                "tablebase_preferred_move",
                "tablebase_value_root",
                "agrees_with_active_reference",
                "agrees_with_puct_selected",
                "agrees_with_classic_teacher",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["state_label"],
                    str(row["remaining_seed_count"]),
                    format_bool(row["tablebase_available"]),
                    str(
                        row["tablebase_preferred_move"]
                        if row["tablebase_preferred_move"] is not None
                        else "-"
                    ),
                    format_float(row["tablebase_value_root"]),
                    (
                        format_bool(row["agrees_with_active_reference"])
                        if row["agrees_with_active_reference"] is not None
                        else "-"
                    ),
                    (
                        format_bool(row["agrees_with_puct_selected"])
                        if row["agrees_with_puct_selected"] is not None
                        else "-"
                    ),
                    (
                        format_bool(row["agrees_with_classic_teacher"])
                        if row["agrees_with_classic_teacher"] is not None
                        else "-"
                    ),
                    row["notes"],
                ]
                for row in summary["tablebase_table"]
            ],
        )
    )
    lines.extend(["", "## 7. Neural child-afterstate value audit", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "corrected_reference_move",
                "selected_move",
                "child",
                "raw_value",
                "root_perspective_value",
                "tablebase_value_root_if_available",
                "child_ref_minus_child_selected",
                "neural_prefers_reference",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["corrected_reference_move"]),
                    str(row["selected_move"]),
                    row["child"],
                    format_float(row["raw_value"]),
                    format_float(row["root_perspective_value"]),
                    format_float(row.get("tablebase_value_root_if_available")),
                    format_float(row["child_ref_minus_child_selected"]),
                    format_bool(row["neural_prefers_reference"]),
                    row["notes"],
                ]
                for row in summary["child_neural_table"]
            ],
        )
    )
    lines.extend(["", "## 8. ClassicMCTS child-afterstate audit", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "budget",
                "seeds",
                "child_ref_value_root_mean",
                "child_selected_value_root_mean",
                "child_ref_minus_child_selected",
                "teacher_prefers_reference",
                "stable",
                "tablebase_agrees_if_available",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["budget"]),
                    str(row["seeds"]),
                    format_float(row["child_ref_value_root_mean"]),
                    format_float(row["child_selected_value_root_mean"]),
                    format_float(row["child_ref_minus_child_selected"]),
                    format_bool(row["teacher_prefers_reference"]),
                    format_bool(row["stable"]),
                    (
                        format_bool(row["tablebase_agrees_if_available"])
                        if row["tablebase_agrees_if_available"] is not None
                        else "-"
                    ),
                    row["notes"],
                ]
                for row in summary["child_classic_table"]
            ],
        )
    )
    if summary.get("teacher_5000_skip_reason"):
        lines.extend(["", f"- {summary['teacher_5000_skip_reason']}"])
    lines.extend(["", "## 9. PUCT child-afterstate audit", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "budget",
                "child_ref_value_root",
                "child_selected_value_root",
                "child_ref_minus_child_selected",
                "puct_prefers_reference",
                "tablebase_agreement_if_available",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["budget"]),
                    format_float(row["child_ref_value_root"]),
                    format_float(row["child_selected_value_root"]),
                    format_float(row["child_ref_minus_child_selected"]),
                    format_bool(row["puct_prefers_reference"]),
                    (
                        format_bool(row["tablebase_agreement_if_available"])
                        if row["tablebase_agreement_if_available"] is not None
                        else "-"
                    ),
                    row["notes"],
                ]
                for row in summary["child_puct_table"]
            ],
        )
    )
    lines.extend(["", "## 10. Root counterfactual diagnostics", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "intervention",
                "budget",
                "selected_move",
                "selected_is_reference",
                "reference_visit_share",
                "selected_visit_share",
                "selected_minus_reference_q_margin",
                "flipped",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["intervention"],
                    str(row["budget"]),
                    str(
                        row["selected_move"]
                        if row["selected_move"] is not None
                        else "-"
                    ),
                    format_bool(row["selected_is_corrected_reference"]),
                    format_float(row["reference_visit_share"]),
                    format_float(row["selected_visit_share"]),
                    format_float(row["selected_minus_reference_q_margin"]),
                    format_bool(row["flipped"]),
                    row["notes"],
                ]
                for row in summary["counterfactual_table"]
            ],
        )
    )
    lines.extend(["", "## 11. Row classifications", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "role",
                "row_classification",
                "supporting_evidence",
                "recommended_use",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["role"],
                    row["row_classification"],
                    row["supporting_evidence"],
                    row["recommended_use"],
                    row["notes"],
                ]
                for row in summary["classification_table"]
            ],
        )
    )
    lines.extend(["", "## 12. Family-level interpretation", ""])
    lines.extend(
        markdown_table(
            [
                "family_classification",
                "value_head_miscalibration_count",
                "puct_child_search_mismatch_count",
                "root_selection_pressure_count",
                "tablebase_reference_conflict_count",
                "tablebase_tie_not_conflict_count",
                "corrected_reference_suspicious_count",
                "backup_perspective_suspect_count",
                "inconclusive_count",
                "next_action",
            ],
            [
                [
                    summary["family_decision_table"]["family_classification"],
                    str(
                        summary["family_decision_table"][
                            "value_head_miscalibration_count"
                        ]
                    ),
                    str(
                        summary["family_decision_table"][
                            "puct_child_search_mismatch_count"
                        ]
                    ),
                    str(
                        summary["family_decision_table"][
                            "root_selection_pressure_count"
                        ]
                    ),
                    str(
                        summary["family_decision_table"][
                            "tablebase_reference_conflict_count"
                        ]
                    ),
                    str(
                        summary["family_decision_table"][
                            "tablebase_tie_not_conflict_count"
                        ]
                    ),
                    str(
                        summary["family_decision_table"][
                            "corrected_reference_suspicious_count"
                        ]
                    ),
                    str(
                        summary["family_decision_table"][
                            "backup_perspective_suspect_count"
                        ]
                    ),
                    str(summary["family_decision_table"]["inconclusive_count"]),
                    summary["family_decision_table"]["next_action"],
                ]
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 13. Exactly one recommended next action",
            "",
            f"Recommendation: **{summary['family_decision_table']['next_action']}**",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    rerun_family_mining_if_needed(root, args)
    audit_rows, validation_rows, validation_context = validate_and_enrich_rows(args)
    evaluator = ArtifactEvaluator(args.current_artifact)
    mining_summary = (
        load_json(args.mining_summary_path) if args.mining_summary_path.exists() else {}
    )
    family_context = next(
        (
            row
            for row in list(mining_summary.get("family_rankings") or [])
            if isinstance(row, dict) and str(row.get("family")) == FAMILY
        ),
        {},
    )

    target_rows = [row for row in audit_rows if row.role == "target_candidate"]
    holdout_rows = [row for row in audit_rows if row.role == "holdout_candidate"]
    control_rows = [row for row in audit_rows if row.role == "preservation_control"]

    run_root_2400, root_2400_note = estimate_root_2400_budget(
        evaluator=evaluator,
        target_rows=target_rows,
        seed=int(args.seed),
        cpuct=float(args.cpuct),
    )
    root_budgets = list(ROOT_BUDGETS)
    if run_root_2400:
        root_budgets.append(OPTIONAL_ROOT_BUDGET)

    root_baselines: dict[tuple[str, int], dict[str, Any]] = {}
    for row in audit_rows:
        for budget in root_budgets:
            root_baselines[(row.row_id, int(budget))] = build_root_baseline_record(
                row,
                root_baseline_for_row(
                    evaluator=evaluator,
                    row=row,
                    budget=int(budget),
                    seed=int(args.seed),
                    cpuct=float(args.cpuct),
                ),
            )

    move_consequence_table: list[dict[str, Any]] = []
    row_move_comparison_notes: dict[str, str] = {}
    for row in audit_rows:
        consequence_rows, note = compare_move_consequences(
            row=row,
            baseline_1200=root_baselines[(row.row_id, 1200)],
        )
        move_consequence_table.extend(consequence_rows)
        row_move_comparison_notes[row.row_id] = note

    teacher_budgets: list[int] = list(CHILD_TEACHER_BASE_BUDGETS)
    teacher_5000_skip_reason = None
    if target_rows:
        can_run_5000, teacher_5000_skip_reason = estimate_teacher_5000_budget(
            target_rows,
            child_state_from_move(
                target_rows[0].suite_state, target_rows[0].corrected_reference_move
            ),
            CHILD_TEACHER_SEEDS,
        )
        if can_run_5000:
            teacher_budgets.append(OPTIONAL_CHILD_TEACHER_BUDGET)

    child_neural_rows: list[dict[str, Any]] = []
    child_classic_rows: list[dict[str, Any]] = []
    child_classic_seed_details: list[dict[str, Any]] = []
    child_puct_rows: list[dict[str, Any]] = []
    counterfactual_table: list[dict[str, Any]] = []
    tablebase_rows: list[dict[str, Any]] = []
    classification_table: list[dict[str, Any]] = []
    neural_summary_by_row: dict[str, dict[str, Any]] = {}
    teacher_summary_by_row: dict[str, dict[int, dict[str, Any]]] = {}
    child_puct_summary_by_row: dict[str, dict[int, dict[str, Any]]] = {}
    tablebase_by_row: dict[str, dict[str, Any]] = {}

    tablebase = EndgameTablebase()

    for row in audit_rows:
        baseline_1200 = root_baselines[(row.row_id, 1200)]
        # augment validation_rows in main output with root-perspective p0 remaining seeds
        validation_lookup = {entry["row_id"]: entry for entry in validation_rows}
        if row.row_id in validation_lookup:
            validation_lookup[row.row_id]["remaining_seed_count"] = int(
                sum(KalahGame.from_state(row.suite_state).pits)
            )

        for root_entry in move_consequence_table:
            if root_entry["row_id"] == row.row_id:
                root_entry["remaining_seed_count_after"] = int(
                    root_entry["remaining_seed_count"]
                )

    for row in target_rows:
        baseline_1200 = root_baselines[(row.row_id, 1200)]
        neural_rows, neural_summary = child_neural_audit(
            evaluator=evaluator,
            row=row,
            baseline_1200=baseline_1200,
        )
        neural_summary_by_row[row.row_id] = neural_summary
        tablebase_row = tablebase_audit_for_row(
            tablebase=tablebase,
            row=row,
            child_summary=neural_summary,
        )
        tablebase_by_row[row.row_id] = tablebase_row
        child_tb_ref = tablebase_row["child_ref"]["tablebase_value_root"]
        child_tb_selected = tablebase_row["child_selected"]["tablebase_value_root"]
        for neural_row in neural_rows:
            child_name = neural_row["child"]
            tb_value = (
                child_tb_ref
                if child_name == "corrected_reference_child"
                else child_tb_selected
            )
            child_neural_rows.append(
                {
                    "row_id": neural_row["row_id"],
                    "corrected_reference_move": neural_row["corrected_reference_move"],
                    "selected_move": neural_row["selected_move"],
                    "child": child_name,
                    "raw_value": neural_row["raw_value"],
                    "root_perspective_value": neural_row["root_perspective_value"],
                    "child_ref_minus_child_selected": neural_row[
                        "child_ref_minus_child_selected"
                    ],
                    "neural_prefers_reference": neural_row[
                        "agrees_with_corrected_reference"
                    ],
                    "child_legal_moves": neural_row["child_legal_moves"],
                    "child_policy_top_move": neural_row["child_policy_top_move"],
                    "tablebase_value_root_if_available": tb_value,
                    "notes": neural_row["notes"],
                }
            )
        teacher_rows, teacher_budget_summary = teacher_child_audit(
            row=row,
            child_summary=neural_summary,
            budgets=tuple(teacher_budgets),
            seeds=CHILD_TEACHER_SEEDS,
        )
        child_classic_seed_details.extend(teacher_rows)
        teacher_summary_by_row[row.row_id] = teacher_budget_summary
        child_ref_tb_move = tablebase_row["child_ref"]["tablebase_preferred_move"]
        child_ref_tb_value = tablebase_row["child_ref"]["tablebase_value_root"]
        child_selected_tb_value = tablebase_row["child_selected"][
            "tablebase_value_root"
        ]
        for budget in teacher_budgets:
            aggregate = teacher_budget_summary[int(budget)]
            tablebase_agreement = None
            if child_ref_tb_value is not None and child_selected_tb_value is not None:
                tablebase_agreement = (
                    (child_ref_tb_value - child_selected_tb_value)
                    > 0.0
                    == bool(aggregate["teacher_prefers_corrected_reference"])
                )
            child_classic_rows.append(
                {
                    "row_id": row.row_id,
                    "budget": int(budget),
                    "seeds": list(CHILD_TEACHER_SEEDS),
                    "child_ref_value_root_mean": aggregate["mean_child_ref_value_root"],
                    "child_selected_value_root_mean": aggregate[
                        "mean_child_selected_value_root"
                    ],
                    "child_ref_minus_child_selected": aggregate["mean_diff"],
                    "teacher_prefers_reference": aggregate[
                        "teacher_prefers_corrected_reference"
                    ],
                    "stable": aggregate["stable"],
                    "tablebase_agrees_if_available": tablebase_agreement,
                    "notes": "ClassicMCTS child-afterstate teacher aggregate",
                }
            )
        puct_rows, puct_budget_summary = child_puct_audit(
            evaluator=evaluator,
            row=row,
            child_summary=neural_summary,
            budgets=CHILD_PUCT_BUDGETS,
            seed=int(args.seed),
            cpuct=float(args.cpuct),
        )
        child_puct_summary_by_row[row.row_id] = puct_budget_summary
        deepest_teacher_budget = max(teacher_budget_summary)
        for puct_row in puct_rows:
            teacher_pref = bool(
                teacher_budget_summary[deepest_teacher_budget][
                    "teacher_prefers_corrected_reference"
                ]
            )
            neural_pref = bool(neural_summary["agrees"])
            puct_pref = bool(puct_row["puct_prefers_corrected_reference"])
            tablebase_agreement = None
            if child_ref_tb_value is not None and child_selected_tb_value is not None:
                tablebase_agreement = (
                    (child_ref_tb_value - child_selected_tb_value)
                    > 0.0
                    == bool(puct_pref)
                )
            notes = []
            notes.append(
                "agrees with teacher"
                if puct_pref == teacher_pref
                else "disagrees with teacher"
            )
            notes.append(
                "agrees with neural"
                if puct_pref == neural_pref
                else "disagrees with neural"
            )
            child_puct_rows.append(
                {
                    **puct_row,
                    "puct_prefers_reference": puct_pref,
                    "agrees_with_teacher": puct_pref == teacher_pref,
                    "agrees_with_neural": puct_pref == neural_pref,
                    "tablebase_agreement_if_available": tablebase_agreement,
                    "notes": "; ".join(notes),
                }
            )
        counterfactual_table.extend(
            counterfactual_rows(
                evaluator=evaluator,
                row=row,
                child_summary=neural_summary,
                teacher_budget_summary=teacher_budget_summary,
                neural_summary=neural_summary,
                seed=int(args.seed),
                cpuct=float(args.cpuct),
            )
        )

    for row in target_rows:
        baseline_1200 = root_baselines[(row.row_id, 1200)]
        tablebase_row = tablebase_by_row[row.row_id]
        child_ref_state = tablebase_row["child_ref"]
        child_selected_state = tablebase_row["child_selected"]
        child_ref_tb_move = child_ref_state["tablebase_preferred_move"]
        child_ref_tb_value = child_ref_state["tablebase_value_root"]
        child_selected_tb_value = child_selected_state["tablebase_value_root"]
        for state_label, state_summary in (
            ("root", tablebase_row["root"]),
            (
                "child_after_corrected_reference",
                child_ref_state,
            ),
            (
                "child_after_current_selected",
                child_selected_state,
            ),
        ):
            active_reference_move = (
                row.corrected_reference_move if state_label == "root" else None
            )
            puct_selected_move = None
            if state_label == "root":
                puct_selected_move = baseline_1200["selected_move"]
            teacher_preferred_move = None
            if state_label == "child_after_corrected_reference":
                teacher_preferred_move = child_ref_tb_move
            tablebase_rows.append(
                {
                    "row_id": row.row_id,
                    "state_label": state_label,
                    "remaining_seed_count": int(state_summary["remaining_seed_count"]),
                    "tablebase_available": bool(state_summary["tablebase_available"]),
                    "tablebase_preferred_move": state_summary[
                        "tablebase_preferred_move"
                    ],
                    "tablebase_value_root": state_summary["tablebase_value_root"],
                    "agrees_with_active_reference": (
                        bool(
                            active_reference_move is not None
                            and state_summary["tablebase_preferred_move"] is not None
                            and int(state_summary["tablebase_preferred_move"])
                            == int(active_reference_move)
                        )
                        if state_label == "root"
                        else None
                    ),
                    "agrees_with_puct_selected": (
                        bool(
                            puct_selected_move is not None
                            and state_summary["tablebase_preferred_move"] is not None
                            and int(state_summary["tablebase_preferred_move"])
                            == int(puct_selected_move)
                        )
                        if state_label == "root"
                        else None
                    ),
                    "agrees_with_classic_teacher": (
                        bool(
                            teacher_preferred_move is not None
                            and state_summary["tablebase_preferred_move"] is not None
                            and int(state_summary["tablebase_preferred_move"])
                            == int(teacher_preferred_move)
                        )
                        if state_label == "child_after_corrected_reference"
                        else None
                    ),
                    "notes": state_summary["notes"],
                }
            )

    for row in audit_rows:
        baseline_1200 = root_baselines[(row.row_id, 1200)]
        classification_table.append(
            classify_row(
                row=row,
                neural_summary=neural_summary_by_row.get(row.row_id),
                teacher_budget_summary=teacher_summary_by_row.get(row.row_id),
                child_puct_budget_summary=child_puct_summary_by_row.get(row.row_id),
                root_baseline_1200=baseline_1200,
                counterfactual=[
                    entry
                    for entry in counterfactual_table
                    if entry["row_id"] == row.row_id
                ],
                tablebase_row=tablebase_by_row.get(row.row_id),
            )
        )

    family_label, family_action, family_counts = classify_family(classification_table)
    root_baseline_table = [
        root_baselines[key]
        for key in sorted(root_baselines, key=lambda item: (item[0], item[1]))
    ]
    family_decision_table = {
        "family_classification": family_label,
        "value_head_miscalibration_count": int(
            family_counts.get("value_head_miscalibration", 0)
        ),
        "puct_child_search_mismatch_count": int(
            family_counts.get("puct_child_search_value_mismatch", 0)
        ),
        "root_selection_pressure_count": int(
            family_counts.get("root_selection_pressure", 0)
        ),
        "tablebase_reference_conflict_count": int(
            family_counts.get("tablebase_reference_conflict", 0)
        ),
        "tablebase_tie_not_conflict_count": int(
            family_counts.get("tablebase_tie_not_conflict", 0)
        ),
        "corrected_reference_suspicious_count": int(
            family_counts.get("corrected_reference_suspicious", 0)
        ),
        "backup_perspective_suspect_count": int(
            family_counts.get("backup_perspective_suspect", 0)
        ),
        "inconclusive_count": int(family_counts.get("inconclusive", 0)),
        "next_action": family_action,
    }
    summary = {
        "schema": SCHEMA,
        "selected_family": FAMILY,
        "inputs": {
            "reference_path": str(args.reference_path),
            "suite_path": str(args.suite_path),
            "current_artifact": str(args.current_artifact),
            "selected_rows_path": str(args.selected_rows_path),
            "mining_summary_path": str(args.mining_summary_path),
        },
        "guardrails": {
            "ran_training": False,
            "ran_arena": False,
            "promoted_model": False,
            "created_replay_artifacts": False,
            "mutated_corrected_references": False,
        },
        "family_context": family_context,
        "row_selection": {
            "target_row_ids": [row.row_id for row in target_rows],
            "holdout_row_ids": [row.row_id for row in holdout_rows],
            "control_row_ids": [row.row_id for row in control_rows],
            **validation_context,
        },
        "perspective_and_backup_check": {
            "conversion_rule": "+1 when child current_player == root current_player, else -1",
            "implementation_rule": "PUCT _search negates the returned child value exactly when child.game.current_player != parent.game.current_player",
            "notes": "extra turns keep sign; turn handoff flips sign back to root-player perspective",
        },
        "root_2400_note": root_2400_note,
        "teacher_5000_skip_reason": teacher_5000_skip_reason,
        "row_validation_table": validation_rows,
        "root_baseline_table": root_baseline_table,
        "move_consequence_table": move_consequence_table,
        "row_move_comparison_notes": row_move_comparison_notes,
        "tablebase_table": tablebase_rows,
        "child_neural_table": child_neural_rows,
        "child_classic_table": child_classic_rows,
        "child_classic_seed_details": child_classic_seed_details,
        "child_puct_table": child_puct_rows,
        "counterfactual_table": counterfactual_table,
        "classification_table": classification_table,
        "family_decision_table": family_decision_table,
    }
    write_json(args.summary_out, summary)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(build_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(args.summary_out),
                "report_path": str(args.report_out),
                "family_classification": family_label,
                "recommended_next_action": family_action,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
