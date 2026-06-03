#!/usr/bin/env python3
"""
Tablebase preferred-move convention audit for sparse_endgame.

Documents tablebase API semantics, enumerates legal-move values, and
determines whether apparent tablebase conflicts are genuine or tie-break
artifacts in forced positions.

Does not train, run arena, promote, create replay artifacts, or mutate
active reference fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.forensic_suite import canonical_state_key, load_suite
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_incumbent_proxy_disagreement_value_backup_audit import (
    child_state_from_move,
    load_jsonl,
    load_reference_maps,
    round_float,
    write_json,
)

EPS = 1e-9
FAMILY = "sparse_endgame"
REFERENCE_PATH = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
SELECTED_ROWS_PATH = Path(
    "/tmp/azlite_corrected_non_opening_failure_mining_v6/"
    "selected_non_opening_family_rows_v6.jsonl"
)
OUTPUT_DIR = Path("/tmp/azlite_sparse_endgame_tablebase_convention_audit")

TARGET_ROW_IDS = (
    "sparse_endgame-003",
    "sparse_endgame-007",
    "sparse_endgame-009",
    "sparse_endgame-024",
    "sparse_endgame-025",
    "sparse_endgame-026",
)
CONTROL_ROW_IDS = (
    "sparse_endgame-002",
    "sparse_endgame-011",
    "sparse_endgame-019",
    "sparse_endgame-020",
    "sparse_endgame-023",
)
ALL_ROW_IDS = TARGET_ROW_IDS + CONTROL_ROW_IDS


# ---------------------------------------------------------------------------
# Diagnostic: legal-move value enumeration
# ---------------------------------------------------------------------------


def tablebase_legal_move_values(
    tablebase: EndgameTablebase,
    state: dict[str, Any],
    *,
    root_player: int,
) -> dict[str, Any]:
    """Enumerate legal-move values from root-player perspective.

    Returns a dict with:
        root_value: float in [-1.0, 1.0] or None if unsolvable
        legal_moves: list[int]
        child_value_by_move: dict[int, float] (root-player perspective)
        best_value: float (max child value)
        optimal_moves: list[int] (all moves within EPS of best_value)
        preferred_move: int (first optimal move by move-index tie-break)
        preferred_move_is_tiebreak: bool
        is_forcing: bool (abs(root_value) >= 1.0 - EPS)
    """
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
    preferred_move_is_tiebreak = len(optimal_moves) > 1
    is_forcing = abs(root_value) >= 1.0 - EPS

    return {
        "root_value": round_float(root_value),
        "legal_moves": legal_moves,
        "child_value_by_move": {
            int(k): float(v) for k, v in child_value_by_move.items()
        },
        "best_value": round_float(best_value),
        "optimal_moves": [int(m) for m in optimal_moves],
        "preferred_move": int(preferred_move) if preferred_move is not None else None,
        "preferred_move_is_tiebreak": bool(preferred_move_is_tiebreak),
        "is_forcing": bool(is_forcing),
        "tablebase_available": True,
    }


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def classify_tablebase_row(
    tb_info: dict[str, Any],
    active_reference_move: int,
    puct_selected_move: int | None,
    role: str,
) -> dict[str, Any]:
    """Classify a row using corrected tablebase conflict logic.

    Returns dict with:
        corrected_tablebase_classification: str
        active_reference_is_optimal: bool
        puct_selected_is_optimal: bool | None
        value_gap: float | None
    """
    if not tb_info["tablebase_available"]:
        return {
            "corrected_tablebase_classification": "tablebase_unavailable",
            "active_reference_is_optimal": None,
            "puct_selected_is_optimal": None,
            "value_gap": None,
        }

    child_values = tb_info["child_value_by_move"]
    optimal_moves = tb_info["optimal_moves"]
    best_value = tb_info["best_value"]
    is_forcing = tb_info["is_forcing"]
    preferred_move = tb_info["preferred_move"]
    preferred_is_tiebreak = tb_info["preferred_move_is_tiebreak"]

    active_ref_value = child_values.get(active_reference_move)
    active_is_optimal = active_reference_move in optimal_moves

    puct_is_optimal = None
    if puct_selected_move is not None:
        puct_is_optimal = puct_selected_move in optimal_moves

    value_gap = None
    if active_ref_value is not None and best_value is not None:
        value_gap = round_float(best_value - active_ref_value)

    if not active_is_optimal and value_gap is not None and value_gap > EPS:
        classification = "tablebase_real_conflict"
    elif active_is_optimal and preferred_move != active_reference_move:
        classification = "tablebase_tie_not_conflict"
    elif active_is_optimal and preferred_move == active_reference_move:
        if role == "target_candidate":
            classification = "tablebase_confirmed"
        else:
            classification = "tablebase_confirmed"
    elif not active_is_optimal and (value_gap is None or value_gap <= EPS):
        classification = "tablebase_ambiguous_or_error"
    else:
        classification = "tablebase_ambiguous_or_error"

    return {
        "corrected_tablebase_classification": classification,
        "active_reference_is_optimal": bool(active_is_optimal),
        "puct_selected_is_optimal": puct_is_optimal,
        "value_gap": value_gap,
        "is_forcing": is_forcing,
        "preferred_move_is_tiebreak": preferred_is_tiebreak,
    }


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def format_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return str(bool(value)).lower()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suite = load_suite(SUITE_PATH)
    suite_by_id = {position.id: position for position in suite}
    reference_by_id, reference_by_canonical = load_reference_maps(REFERENCE_PATH)
    selected_rows = load_jsonl(SELECTED_ROWS_PATH)
    selected_by_id = {str(row["row_id"]): row for row in selected_rows}

    # Build row info
    rows: dict[str, dict[str, Any]] = {}
    for row_id in ALL_ROW_IDS:
        suite_row = suite_by_id.get(row_id)
        if suite_row is None:
            print(f"ERROR: {row_id} missing from forensic suite", file=sys.stderr)
            return 1
        selected = selected_by_id.get(row_id)
        if selected is None:
            print(f"ERROR: {row_id} missing from selected rows", file=sys.stderr)
            return 1
        reference_row = reference_by_id.get(row_id) or reference_by_canonical.get(
            canonical_state_key(suite_row.state)
        )
        if reference_row is None:
            print(f"ERROR: {row_id} missing from reference data", file=sys.stderr)
            return 1
        canonical = canonical_state_key(suite_row.state)
        if canonical != str(selected["canonical_state_hash"]):
            print(
                f"ERROR: {row_id} canonical state hash mismatch",
                file=sys.stderr,
            )
            return 1
        corrected_reference_move = int(selected["corrected_reference_move"])
        legal_moves = list(suite_row.legal_moves)
        if corrected_reference_move not in legal_moves:
            print(
                f"ERROR: {row_id} reference move {corrected_reference_move} illegal",
                file=sys.stderr,
            )
            return 1
        rows[row_id] = {
            "id": row_id,
            "state": dict(suite_row.state),
            "legal_moves": legal_moves,
            "corrected_reference_move": corrected_reference_move,
            "current_selected_move": int(selected["current_selected_move"]),
            "canonical_state_hash": canonical,
            "reference_row": reference_row,
            "role": str(selected.get("recommended_role", "unknown")),
        }

    # Audit each row
    tablebase = EndgameTablebase()
    row_results: list[dict[str, Any]] = []
    for row_id in ALL_ROW_IDS:
        info = rows[row_id]
        state = info["state"]
        root_player = int(state["current_player"])
        active_ref = info["corrected_reference_move"]
        puct_selected = info["current_selected_move"]

        tb_info = tablebase_legal_move_values(tablebase, state, root_player=root_player)
        classification = classify_tablebase_row(
            tb_info, active_ref, puct_selected, info["role"]
        )

        # Previous (uncorrected) decision for comparison
        old_preferred_move = tb_info["preferred_move"]
        old_disagrees = (
            old_preferred_move is not None and old_preferred_move != active_ref
        )
        old_decision = (
            "tablebase_conflict"
            if old_disagrees
            else "tablebase_confirmed"
            if old_preferred_move is not None
            else "tablebase_unavailable"
        )

        row_results.append(
            {
                "row_id": row_id,
                "role": info["role"],
                "legal_moves": info["legal_moves"],
                "root_value": tb_info["root_value"],
                "active_reference_move": active_ref,
                "puct_selected_move": puct_selected,
                "preferred_move": tb_info["preferred_move"],
                "optimal_moves": tb_info["optimal_moves"],
                "child_value_by_move": tb_info["child_value_by_move"],
                "best_value": tb_info["best_value"],
                "active_reference_is_optimal": classification[
                    "active_reference_is_optimal"
                ],
                "puct_selected_is_optimal": classification["puct_selected_is_optimal"],
                "preferred_move_is_tiebreak": tb_info["preferred_move_is_tiebreak"],
                "is_forcing": tb_info["is_forcing"],
                "value_gap": classification["value_gap"],
                "old_uncorrected_decision": old_decision,
                "corrected_tablebase_classification": classification[
                    "corrected_tablebase_classification"
                ],
            }
        )

    # Build summary
    api_semantics = {
        "value_convention": (
            "current-player perspective minimax; returned as win-rate in [0.0, 1.0] "
            "from perspective_player argument; converted to [-1.0, 1.0] root-perspective "
            "via root_value = 2*win_rate - 1"
        ),
        "preferred_move_convention": (
            "first legal move (by move index) among optimal child values; "
            "optimal = child with highest win-rate from root-player perspective"
        ),
        "move_selection": (
            "evaluates child win-rate for each legal move from root-player perspective; "
            "picks max win-rate; tie-breaks by lower move index"
        ),
        "can_expose": {
            "value_per_legal_move": True,
            "all_optimal_moves": True,
            "value_gap_best_vs_active": True,
            "tie_equivalence_set": True,
        },
        "implication": (
            "In forcing positions (|root_value| = 1.0), all legal moves have identical "
            "child win-rates from root perspective. The 'preferred move' is a tie-breaker "
            "(first legal move by index), not a genuine policy distinction. "
            "Conflicts in forcing positions should be classified as tablebase_tie_not_conflict "
            "when the active reference is among the optimal (tied) moves."
        ),
    }

    summary = {
        "schema": "azlite_sparse_endgame_tablebase_convention_audit_v1",
        "family": FAMILY,
        "description": (
            "Tablebase preferred-move convention audit for sparse_endgame. "
            "Documents tablebase API semantics and reclassifies apparent "
            "tablebase conflicts using corrected tie-break-aware logic."
        ),
        "guardrails": {
            "mutated_active_fixture": False,
            "ran_training": False,
            "ran_arena": False,
            "promoted_model": False,
            "created_replay_artifacts": False,
        },
        "api_semantics": api_semantics,
        "eps": EPS,
        "row_results": row_results,
    }
    summary_path = OUTPUT_DIR / "tablebase_convention_audit_summary.json"
    write_json(summary_path, summary)
    print(f"Wrote: {summary_path}")

    # Print summary table
    print(f"\n{'=' * 120}")
    print("TABLEBASE CONVENTION AUDIT — SPARSE_ENDGAME")
    print(f"{'=' * 120}")
    print(
        f"{'row_id':30s} {'role':20s} {'root_val':10s} {'ref':4s} "
        f"{'pref':5s} {'optimals':24s} {'ref_opt':7s} {'tiebreak':9s} "
        f"{'forcing':8s} {'old_dec':20s} {'corr_dec':24s}"
    )
    print(f"{'-' * 120}")
    for r in row_results:
        print(
            f"{r['row_id']:30s} {r['role']:20s} "
            f"{format_float(r['root_value']):10s} "
            f"{str(r['active_reference_move']):4s} "
            f"{str(r['preferred_move'] if r['preferred_move'] is not None else '-'):5s} "
            f"{str(r['optimal_moves']):24s} "
            f"{format_bool(r['active_reference_is_optimal']):7s} "
            f"{format_bool(r['preferred_move_is_tiebreak']):9s} "
            f"{format_bool(r['is_forcing']):8s} "
            f"{r['old_uncorrected_decision']:20s} "
            f"{r['corrected_tablebase_classification']:24s}"
        )
    print(f"{'=' * 120}\n")

    # Print child value details
    print(f"{'CHILD VALUE DETAILS':^80}")
    print(f"{'=' * 80}")
    for r in row_results:
        print(
            f"\n{r['row_id']} (ref={r['active_reference_move']}, "
            f"root_val={format_float(r['root_value'])})"
        )
        for move, val in sorted(r["child_value_by_move"].items()):
            marker = ""
            if move == r["active_reference_move"]:
                marker += " [REF]"
            if move == r["preferred_move"]:
                marker += " [PREF]"
            if r["puct_selected_move"] is not None and move == r["puct_selected_move"]:
                marker += " [PUCT]"
            if r["optimal_moves"] and move in r["optimal_moves"]:
                marker += " [OPT]"
            print(f"  move {move}: {format_float(val)}{marker}")
        print(
            f"  best_value={format_float(r['best_value'])}, "
            f"optimal_moves={r['optimal_moves']}, "
            f"value_gap={format_float(r['value_gap'])}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
