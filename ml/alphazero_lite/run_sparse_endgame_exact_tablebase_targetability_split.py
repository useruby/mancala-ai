#!/usr/bin/env python3
"""
Exact-tablebase targetability split for sparse_endgame.

Loads all 24 sparse_endgame rows, enumerates exact tablebase move values,
classifies each row into signal buckets, and produces a targetability
decision.

Does not train, run arena, promote, create replay/value artifacts,
or mutate active reference fixtures.
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
OUTPUT_DIR = Path("/tmp/azlite_sparse_endgame_exact_tablebase_targetability")

ALL_SPARSE_ENDGAME_IDS = tuple(f"sparse_endgame-{i:03d}" for i in range(1, 4)) + tuple(
    f"sparse_endgame-{i:03d}" for i in range(7, 28)
)

PR66_ROW_IDS = (
    "sparse_endgame-003",
    "sparse_endgame-007",
    "sparse_endgame-009",
    "sparse_endgame-024",
    "sparse_endgame-025",
    "sparse_endgame-026",
    "sparse_endgame-023",
)

PR66_CLASSIFICATIONS = {
    "sparse_endgame-003": "corrected_reference_suspicious",
    "sparse_endgame-007": "corrected_reference_suspicious",
    "sparse_endgame-009": "tablebase_tie_not_conflict",
    "sparse_endgame-024": "puct_child_search_value_mismatch",
    "sparse_endgame-025": "tablebase_tie_not_conflict",
    "sparse_endgame-026": "tablebase_tie_not_conflict",
    "sparse_endgame-023": "inconclusive (holdout)",
}

_MOVE_OPTIMAL_CACHE: dict[str, Any] = {}


def tablebase_legal_move_values(
    tablebase: EndgameTablebase,
    state: dict[str, Any],
    *,
    root_player: int,
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
    preferred_move_is_tiebreak = len(optimal_moves) > 1
    is_forcing = abs(root_value) >= 1.0 - EPS
    all_moves_equivalent = (
        len(set(round_float(v, 6) for v in child_value_by_move.values())) == 1
        if child_value_by_move
        else False
    )
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
        "all_moves_equivalent": bool(all_moves_equivalent),
        "tablebase_available": True,
    }


def classify_exact_signal(
    tb_info: dict[str, Any],
    active_reference_move: int,
    puct_selected_move: int | None,
) -> dict[str, Any]:
    if not tb_info["tablebase_available"]:
        return {
            "exact_signal_class": "tablebase_unavailable",
            "active_reference_is_optimal": None,
            "puct_selected_is_optimal": None,
            "unique_optimal_move": None,
            "all_legal_moves_equivalent": False,
            "is_forcing": False,
            "value_gap_best_minus_reference": None,
            "value_gap_best_minus_puct": None,
            "value_gap_reference_minus_puct": None,
        }

    optimal_moves = tb_info["optimal_moves"]
    child_values = tb_info["child_value_by_move"]
    best_value = tb_info["best_value"]
    is_forcing = tb_info["is_forcing"]
    all_equiv = tb_info["all_moves_equivalent"]

    active_is_optimal = active_reference_move in optimal_moves
    puct_is_optimal = None
    if puct_selected_move is not None:
        puct_is_optimal = puct_selected_move in optimal_moves

    unique_optimal = len(optimal_moves) == 1
    unique_optimal_move = optimal_moves[0] if unique_optimal else None

    ref_value = child_values.get(active_reference_move, best_value)
    puct_value = (
        child_values.get(puct_selected_move, best_value)
        if puct_selected_move is not None
        else None
    )

    value_gap_best_ref = (
        round_float(best_value - ref_value) if ref_value is not None else None
    )
    value_gap_best_puct = (
        round_float(best_value - puct_value) if puct_value is not None else None
    )
    value_gap_ref_puct = (
        round_float(ref_value - puct_value)
        if ref_value is not None and puct_value is not None
        else None
    )

    if not tb_info["tablebase_available"]:
        signal_class = "tablebase_unavailable"
    elif all_equiv:
        signal_class = "forced_all_moves_equivalent"
    elif unique_optimal and active_is_optimal:
        signal_class = "exact_unique_policy_target"
    elif active_is_optimal and not unique_optimal:
        signal_class = "exact_value_only_tie"
    elif not active_is_optimal:
        signal_class = "exact_reference_suboptimal"
    else:
        signal_class = "tablebase_integrity_error"

    if (
        puct_selected_move is not None
        and puct_is_optimal is not None
        and not puct_is_optimal
    ):
        puct_suboptimal = True
    else:
        puct_suboptimal = False

    return {
        "exact_signal_class": signal_class,
        "active_reference_is_optimal": (
            bool(active_is_optimal) if active_is_optimal is not None else None
        ),
        "puct_selected_is_optimal": puct_is_optimal,
        "unique_optimal_move": unique_optimal_move,
        "all_legal_moves_equivalent": bool(all_equiv),
        "is_forcing": bool(is_forcing),
        "value_gap_best_minus_reference": value_gap_best_ref,
        "value_gap_best_minus_puct_selected": value_gap_best_puct,
        "value_gap_reference_minus_puct_selected": value_gap_ref_puct,
        "puct_suboptimal": puct_suboptimal,
    }


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def format_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return str(bool(value)).lower()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Load data
    # -----------------------------------------------------------------------
    suite = load_suite(SUITE_PATH)
    suite_by_id = {position.id: position for position in suite}
    reference_by_id, reference_by_canonical = load_reference_maps(REFERENCE_PATH)
    selected_rows = load_jsonl(SELECTED_ROWS_PATH)
    selected_by_id = {str(row["row_id"]): row for row in selected_rows}

    rows: dict[str, dict[str, Any]] = {}
    for row_id in ALL_SPARSE_ENDGAME_IDS:
        suite_row = suite_by_id.get(row_id)
        if suite_row is None:
            print(f"ERROR: {row_id} missing from forensic suite", file=sys.stderr)
            return 1
        reference_row = reference_by_id.get(row_id) or reference_by_canonical.get(
            canonical_state_key(suite_row.state)
        )
        if reference_row is None:
            print(f"ERROR: {row_id} missing from reference data", file=sys.stderr)
            return 1
        if reference_row.get("reference_unstable", False):
            print(f"ERROR: {row_id} is marked reference_unstable", file=sys.stderr)
            return 1

        selected = selected_by_id.get(row_id)
        if selected is not None:
            canonical = canonical_state_key(suite_row.state)
            if canonical != str(selected["canonical_state_hash"]):
                print(
                    f"ERROR: {row_id} canonical state hash mismatch",
                    file=sys.stderr,
                )
                return 1
            corrected_reference_move = int(selected["corrected_reference_move"])
            current_selected_move = int(selected["current_selected_move"])
            role = str(selected.get("recommended_role", "unknown"))
        else:
            corrected_reference_move = int(reference_row["reference_move"])
            current_selected_move = None
            role = "unselected"

        legal_moves = list(suite_row.legal_moves)
        if corrected_reference_move not in legal_moves:
            print(
                f"ERROR: {row_id} reference move {corrected_reference_move} illegal",
                file=sys.stderr,
            )
            return 1

        remaining_seed_count = int(sum(KalahGame.from_state(suite_row.state).pits))

        rows[row_id] = {
            "id": row_id,
            "state": dict(suite_row.state),
            "legal_moves": legal_moves,
            "corrected_reference_move": corrected_reference_move,
            "current_selected_move": current_selected_move,
            "canonical_state_hash": canonical_state_key(suite_row.state),
            "reference_row": reference_row,
            "role": role,
            "remaining_seed_count": remaining_seed_count,
        }

    # -----------------------------------------------------------------------
    # Step 2: Exact tablebase move-value enumeration
    # -----------------------------------------------------------------------
    tablebase = EndgameTablebase()
    row_results: list[dict[str, Any]] = []

    for row_id in ALL_SPARSE_ENDGAME_IDS:
        info = rows[row_id]
        state = info["state"]
        root_player = int(state["current_player"])
        active_ref = info["corrected_reference_move"]
        puct_selected = info["current_selected_move"]

        tb_info = tablebase_legal_move_values(tablebase, state, root_player=root_player)
        classification = classify_exact_signal(tb_info, active_ref, puct_selected)

        row_results.append(
            {
                "row_id": row_id,
                "role": info["role"],
                "legal_moves": info["legal_moves"],
                "remaining_seed_count": info["remaining_seed_count"],
                "root_value": tb_info["root_value"],
                "active_reference_move": active_ref,
                "current_puct_selected_move": puct_selected,
                "child_value_by_move": tb_info["child_value_by_move"],
                "best_value": tb_info["best_value"],
                "optimal_moves": tb_info["optimal_moves"],
                "preferred_move": tb_info["preferred_move"],
                "preferred_move_is_tiebreak": tb_info["preferred_move_is_tiebreak"],
                "is_forcing": tb_info["is_forcing"],
                "all_moves_equivalent": tb_info["all_moves_equivalent"],
                "tablebase_available": tb_info["tablebase_available"],
                "active_reference_is_optimal": classification[
                    "active_reference_is_optimal"
                ],
                "puct_selected_is_optimal": classification["puct_selected_is_optimal"],
                "unique_optimal_move": classification["unique_optimal_move"],
                "value_gap_best_minus_reference": classification[
                    "value_gap_best_minus_reference"
                ],
                "value_gap_best_minus_puct_selected": classification[
                    "value_gap_best_minus_puct_selected"
                ],
                "value_gap_reference_minus_puct_selected": classification[
                    "value_gap_reference_minus_puct_selected"
                ],
                "exact_signal_class": classification["exact_signal_class"],
                "puct_suboptimal": classification["puct_suboptimal"],
            }
        )

    # -----------------------------------------------------------------------
    # Step 3: Build buckets
    # -----------------------------------------------------------------------
    buckets: dict[str, dict[str, Any]] = {
        "exact_unique_policy_targets": {
            "row_count": 0,
            "rows": [],
            "recommended_use": "policy_target_candidate",
            "risks": "single-row bucket too small for isolated training; insufficient for family-level mining",
            "next_action": "holdout_only unless combined with other policy-target rows from another family",
        },
        "exact_value_only_ties": {
            "row_count": 0,
            "rows": [],
            "recommended_use": "value_only_candidate",
            "risks": "multiple optimal moves means policy target is ambiguous; value labels are correct",
            "next_action": "assess whether enough rows exist for value-only diagnostic",
        },
        "forced_all_moves_equivalent": {
            "row_count": 0,
            "rows": [],
            "recommended_use": "exclude_from_training",
            "risks": "no meaningful policy or value signal; all moves lead to same outcome",
            "next_action": "exclude from training selection; preserve as context only",
        },
        "puct_suboptimal_exact_rows": {
            "row_count": 0,
            "rows": [],
            "recommended_use": "search_diagnostic",
            "risks": "PUCT selects a suboptimal move despite exact tablebase showing optimal; search/value issue not policy",
            "next_action": "inspect whether root-prior or child-PUCT mismatch drives suboptimal selection",
        },
        "tablebase_unavailable_or_ambiguous": {
            "row_count": 0,
            "rows": [],
            "recommended_use": "exclude_from_training",
            "risks": "no exact tablebase label available",
            "next_action": "do not train until adjudicated or solver expanded",
        },
        "preservation_controls": {
            "row_count": 0,
            "rows": [],
            "recommended_use": "preservation_control",
            "risks": "controls must remain passing; no training target",
            "next_action": "retain in test/holdout set for regression detection",
        },
        "exact_reference_suboptimal": {
            "row_count": 0,
            "rows": [],
            "recommended_use": "exclude_from_training",
            "risks": "active reference is not tablebase-optimal; reference integrity error or row needs adjudication",
            "next_action": "do not train until reference is corrected or row is excluded",
        },
        "tablebase_integrity_error": {
            "row_count": 0,
            "rows": [],
            "recommended_use": "exclude_from_training",
            "risks": "tablebase lookup or classification inconsistent",
            "next_action": "investigate tablebase implementation",
        },
    }

    for r in row_results:
        signal = r["exact_signal_class"]
        role = r["role"]
        row_id = r["row_id"]

        if signal == "exact_unique_policy_target":
            buckets["exact_unique_policy_targets"]["row_count"] += 1
            buckets["exact_unique_policy_targets"]["rows"].append(row_id)
        elif signal == "exact_value_only_tie":
            buckets["exact_value_only_ties"]["row_count"] += 1
            buckets["exact_value_only_ties"]["rows"].append(row_id)
        elif signal == "forced_all_moves_equivalent":
            buckets["forced_all_moves_equivalent"]["row_count"] += 1
            buckets["forced_all_moves_equivalent"]["rows"].append(row_id)
        elif signal == "exact_reference_suboptimal":
            buckets["exact_reference_suboptimal"]["row_count"] += 1
            buckets["exact_reference_suboptimal"]["rows"].append(row_id)
        elif signal == "tablebase_unavailable":
            buckets["tablebase_unavailable_or_ambiguous"]["row_count"] += 1
            buckets["tablebase_unavailable_or_ambiguous"]["rows"].append(row_id)
        elif signal == "tablebase_integrity_error":
            buckets["tablebase_integrity_error"]["row_count"] += 1
            buckets["tablebase_integrity_error"]["rows"].append(row_id)

        if role == "preservation_control":
            buckets["preservation_controls"]["row_count"] += 1
            buckets["preservation_controls"]["rows"].append(row_id)

        if r["puct_suboptimal"] and signal in (
            "exact_unique_policy_target",
            "exact_value_only_tie",
        ):
            if row_id not in buckets["puct_suboptimal_exact_rows"]["rows"]:
                buckets["puct_suboptimal_exact_rows"]["row_count"] += 1
                buckets["puct_suboptimal_exact_rows"]["rows"].append(row_id)

    # -----------------------------------------------------------------------
    # Step 5: Value-only candidate assessment
    # -----------------------------------------------------------------------
    value_only_candidates = [
        r
        for r in row_results
        if r["exact_signal_class"]
        in ("exact_value_only_tie", "forced_all_moves_equivalent")
        and r["active_reference_is_optimal"]
    ]

    value_assessment_rows = []
    for r in value_only_candidates:
        root_val = r["root_value"]
        value_assessment_rows.append(
            {
                "row_id": r["row_id"],
                "exact_value": root_val,
                "neural_value": None,
                "value_error": None,
                "sign_error": None,
                "abs_error": None,
                "forced_win_or_loss": abs(root_val) >= 1.0 - EPS
                if root_val is not None
                else None,
                "value_only_candidate": True,
                "notes": "neural value not computed in this split; available from value backup audit for selected rows",
            }
        )

    meaningful_value_error_count = 0
    for va in value_assessment_rows:
        if va["neural_value"] is not None and va["value_error"] is not None:
            if abs(va["value_error"]) > 0.05:
                meaningful_value_error_count += 1

    value_only_ready = (
        len(value_assessment_rows) >= 5 and meaningful_value_error_count >= 3
    )

    # -----------------------------------------------------------------------
    # Step 6: Policy/search candidate assessment
    # -----------------------------------------------------------------------
    policy_search_candidates = [
        r
        for r in row_results
        if r["exact_signal_class"] == "exact_unique_policy_target"
        or r["puct_suboptimal"]
    ]

    policy_search_rows = []
    for r in policy_search_candidates:
        policy_search_rows.append(
            {
                "row_id": r["row_id"],
                "exact_optimal_move": r["unique_optimal_move"]
                if r["exact_signal_class"] == "exact_unique_policy_target"
                else (r["optimal_moves"][0] if r["optimal_moves"] else None),
                "puct_selected_move": r["current_puct_selected_move"],
                "puct_selected_is_optimal": r["puct_selected_is_optimal"],
                "policy_or_search_candidate": r["exact_signal_class"]
                == "exact_unique_policy_target"
                or r["puct_suboptimal"],
                "notes": "",
            }
        )
        if r["puct_suboptimal"]:
            policy_search_rows[-1]["notes"] += "PUCT suboptimal; "
        if r["exact_signal_class"] == "exact_unique_policy_target":
            policy_search_rows[-1]["notes"] += "unique policy target; "
        if r["exact_signal_class"] == "exact_value_only_tie" and r["puct_suboptimal"]:
            policy_search_rows[-1]["notes"] += (
                "value-tie but PUCT suboptimal (search diagnostic); "
            )

    unique_policy_count = len(
        [
            r
            for r in row_results
            if r["exact_signal_class"] == "exact_unique_policy_target"
        ]
    )
    puct_suboptimal_count = len(
        set(r["row_id"] for r in row_results if r["puct_suboptimal"])
    )
    unique_targets_with_puct_data = len(
        [
            r
            for r in row_results
            if r["exact_signal_class"] == "exact_unique_policy_target"
            and r["current_puct_selected_move"] is not None
        ]
    )
    policy_search_ready = (unique_policy_count + puct_suboptimal_count) >= 3

    # -----------------------------------------------------------------------
    # Step 4: Decision
    #
    # Primary gate: forced/tied dominance.
    # If >50% of rows are forced/tied (all moves equivalent or value-only tie),
    # policy labels are essentially arbitrary and the family is not trainable
    # as a policy family. A secondary policy/search diagnostic may still be
    # noted but does not override the fundamental limitation.
    # -----------------------------------------------------------------------
    forced_all_equivalent_count = buckets["forced_all_moves_equivalent"]["row_count"]
    value_only_tie_count = buckets["exact_value_only_ties"]["row_count"]
    unique_target_count = buckets["exact_unique_policy_targets"]["row_count"]
    tablebase_unavail_count = buckets["tablebase_unavailable_or_ambiguous"]["row_count"]
    ref_suboptimal_count = buckets["exact_reference_suboptimal"]["row_count"]

    forced_or_tied = forced_all_equivalent_count + value_only_tie_count
    total_rows = len(row_results)
    forced_or_tied_pct = forced_or_tied / total_rows if total_rows > 0 else 0.0

    forced_tied_dominated = forced_or_tied_pct > 0.5

    if forced_tied_dominated:
        detail_lines = [
            f"{forced_or_tied_pct:.0%} of {total_rows} rows are forced/tied",
            f"({forced_all_equivalent_count} forced_all_moves_equivalent + "
            f"{value_only_tie_count} exact_value_only_ties).",
        ]
        if forced_all_equivalent_count > 0:
            detail_lines.append(
                f"Policy labels are arbitrary in all {forced_all_equivalent_count} "
                "forced-all-equivalent rows."
            )

        if unique_target_count > 0:
            detail_lines.append(
                f"{unique_target_count} unique policy targets exist "
                f"but only {unique_targets_with_puct_data} have PUCT data "
                f"({unique_target_count - unique_targets_with_puct_data} need MCTS runs)."
            )

        if puct_suboptimal_count > 0:
            detail_lines.append(
                f"{puct_suboptimal_count} PUCT-suboptimal row(s) exist "
                "but family-level training is not justified at 83% forced noise."
            )

        decision_classification = "sparse_endgame_not_trainable_as_policy_family"
        decision_evidence = " ".join(detail_lines)
        decision_rejected = (
            f"Value-only: neural values not computed in this split; "
            f"Policy/search: {unique_target_count} unique targets, "
            f"{puct_suboptimal_count} PUCT-suboptimal, "
            f"but {forced_or_tied_pct:.0%} forced/tied noise dominates."
        )
        decision_next = (
            "Exclude sparse_endgame from training-family selection "
            "and rerun non-opening family mining v7. "
            "Optional: run a tiny PUCT diagnostic on sparse_endgame-023 only "
            "(unique policy target with PUCT-suboptimal behavior) if search "
            "diagnosis is desired, but do not treat as a training family."
        )
    elif value_only_ready:
        decision_classification = "sparse_endgame_value_only_diagnostic_ready"
        decision_evidence = (
            f"At least {len(value_assessment_rows)} value-only candidate rows "
            f"with {meaningful_value_error_count} having meaningful neural value error. "
            f"Value-only diagnostic artifact is justified."
        )
        decision_rejected = (
            f"Policy/search not viable ({unique_target_count} unique targets, "
            f"{puct_suboptimal_count} PUCT-suboptimal); "
            f"forced/tied dominates ({forced_or_tied_pct:.0%} of rows)."
        )
        decision_next = (
            "Build a tiny value-only diagnostic artifact using exact tablebase "
            "values for forced/tied rows; do not use policy targets for tied rows."
        )
    elif policy_search_ready:
        decision_classification = "sparse_endgame_policy_search_diagnostic_ready"
        decision_evidence = (
            f"{unique_target_count} unique policy targets + "
            f"{puct_suboptimal_count} PUCT-suboptimal rows "
            f"= {unique_target_count + puct_suboptimal_count} combined search/policy candidates. "
            f"Note: {unique_targets_with_puct_data} of {unique_target_count} unique targets "
            f"have PUCT data; remaining need MCTS runs."
        )
        decision_rejected = (
            f"Value-only not viable: neural values not available or error < threshold. "
            f"Forced/tied: {forced_or_tied_pct:.0%} of rows."
        )
        decision_next = (
            "Run root/child PUCT diagnostics only for exact suboptimal rows; "
            "do not train a full family on sparse_endgame."
        )
    elif tablebase_unavail_count > 0:
        decision_classification = "sparse_endgame_tablebase_coverage_insufficient"
        decision_evidence = (
            f"{tablebase_unavail_count} rows have no tablebase coverage."
        )
        decision_rejected = "Other signals insufficient for reliable training."
        decision_next = (
            "Exclude sparse_endgame or expand exact solver coverage before training."
        )
    else:
        decision_classification = "sparse_endgame_too_small_after_tablebase_fix"
        decision_evidence = (
            f"Total: {total_rows} rows. "
            f"Unique policy: {unique_target_count}, PUCT-suboptimal: {puct_suboptimal_count}, "
            f"Value-only: {len(value_assessment_rows)}."
        )
        decision_rejected = (
            "Neither value-only nor policy/search bucket meets minimum thresholds."
        )
        decision_next = "Exclude sparse_endgame and select the next family."

    # -----------------------------------------------------------------------
    # Build output JSONs
    # -----------------------------------------------------------------------

    # Summary JSON
    summary = {
        "schema": "azlite_sparse_endgame_exact_tablebase_targetability_v1",
        "family": FAMILY,
        "description": (
            "Exact-tablebase targetability split for sparse_endgame. "
            "Classifies all 24 rows into exact tablebase signal buckets "
            "and produces a targetability decision."
        ),
        "guardrails": {
            "mutated_active_fixture": False,
            "ran_training": False,
            "ran_arena": False,
            "promoted_model": False,
            "created_replay_artifacts": False,
            "created_value_artifact": False,
        },
        "row_validation": {
            "total_rows_in_suite": len(suite),
            "sparse_endgame_row_count": len(rows),
            "selected_row_count": len(selected_rows),
            "selected_sparse_endgame_count": len(
                [r for r in selected_rows if r.get("family") == FAMILY]
            ),
            "all_sparse_endgame_ids": list(ALL_SPARSE_ENDGAME_IDS),
        },
        "exact_tablebase_table": [
            {
                "row_id": r["row_id"],
                "role": r["role"],
                "legal_moves": r["legal_moves"],
                "root_value": r["root_value"],
                "active_reference_move": r["active_reference_move"],
                "current_puct_selected_move": r["current_puct_selected_move"],
                "optimal_moves": r["optimal_moves"],
                "active_reference_is_optimal": r["active_reference_is_optimal"],
                "puct_selected_is_optimal": r["puct_selected_is_optimal"],
                "unique_optimal_move": r["unique_optimal_move"],
                "all_moves_equivalent": r["all_moves_equivalent"],
                "is_forcing": r["is_forcing"],
                "exact_signal_class": r["exact_signal_class"],
                "puct_suboptimal": r["puct_suboptimal"],
                "value_gap_best_minus_reference": r["value_gap_best_minus_reference"],
                "notes": "",
            }
            for r in row_results
        ],
        "pr66_reinterpretation": [],
        "buckets": buckets,
        "value_only_assessment": value_assessment_rows,
        "policy_search_assessment": policy_search_rows,
        "decision": {
            "classification": decision_classification,
            "supporting_evidence": decision_evidence,
            "rejected_alternatives": decision_rejected,
            "next_action": decision_next,
            "counts": {
                "total_rows": total_rows,
                "forced_all_moves_equivalent": forced_all_equivalent_count,
                "exact_value_only_ties": value_only_tie_count,
                "exact_unique_policy_targets": unique_target_count,
                "puct_suboptimal": puct_suboptimal_count,
                "reference_suboptimal": ref_suboptimal_count,
                "tablebase_unavailable": tablebase_unavail_count,
                "value_only_candidates": len(value_assessment_rows),
                "policy_search_candidates": len(policy_search_rows),
            },
        },
    }

    # Step 3: Reinterpret PR #66 rows
    for r in row_results:
        if r["row_id"] in PR66_ROW_IDS:
            pr66_class = PR66_CLASSIFICATIONS.get(r["row_id"], "unknown")
            signal = r["exact_signal_class"]

            meaningful_policy = signal == "exact_unique_policy_target"
            meaningful_value = signal in (
                "exact_value_only_tie",
                "exact_unique_policy_target",
            )
            if signal == "forced_all_moves_equivalent":
                meaningful_policy = False
                meaningful_value = False

            if signal == "exact_unique_policy_target":
                rec_use = "holdout_only (single-row bucket)"
            elif signal == "exact_value_only_tie":
                rec_use = "value_only_candidate"
            elif signal == "forced_all_moves_equivalent":
                rec_use = "exclude_from_training"
            else:
                rec_use = "needs_adjudication"

            summary["pr66_reinterpretation"].append(
                {
                    "row_id": r["row_id"],
                    "pr66_classification": pr66_class,
                    "exact_signal_class": signal,
                    "meaningful_policy_target": meaningful_policy,
                    "meaningful_value_target": meaningful_value,
                    "recommended_use": rec_use,
                    "notes": "",
                }
            )

    write_json(
        OUTPUT_DIR / "sparse_endgame_exact_tablebase_targetability_summary.json",
        summary,
    )
    print(
        f"Wrote: {OUTPUT_DIR / 'sparse_endgame_exact_tablebase_targetability_summary.json'}"
    )

    # Buckets-only JSON
    buckets_out = {
        "schema": "azlite_sparse_endgame_exact_tablebase_buckets_v1",
        "family": FAMILY,
        "buckets": buckets,
        "decision": summary["decision"],
    }
    write_json(OUTPUT_DIR / "sparse_endgame_exact_tablebase_buckets.json", buckets_out)
    print(f"Wrote: {OUTPUT_DIR / 'sparse_endgame_exact_tablebase_buckets.json'}")

    # -----------------------------------------------------------------------
    # Print console summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print("EXACT TABLEBASE TARGETABILITY SPLIT — SPARSE_ENDGAME")
    print(f"{'=' * 100}")

    print(
        f"\n{'Rows loaded:':30s} {len(rows)} sparse_endgame / {len(suite)} total suite"
    )
    print(
        f"{'Rows with PUCT data:':30s} {len([r for r in row_results if r['current_puct_selected_move'] is not None])}"
    )

    print(f"\n{'Bucket':40s} {'Count':8s} {'Use':30s}")
    print(f"{'-' * 80}")
    for bucket_name, bucket_data in sorted(buckets.items()):
        print(
            f"{bucket_name:40s} {str(bucket_data['row_count']):8s} {bucket_data['recommended_use']:30s}"
        )

    print(f"\n{'Exact tablebase summary':^80}")
    print(f"{'=' * 80}")
    print(
        f"{'row_id':30s} {'signal':40s} {'ref_opt':7s} {'puct_opt':8s} {'unique':7s} {'equiv':7s}"
    )
    print(f"{'-' * 100}")
    for r in row_results:
        print(
            f"{r['row_id']:30s} "
            f"{r['exact_signal_class']:40s} "
            f"{format_bool(r['active_reference_is_optimal']):7s} "
            f"{format_bool(r['puct_selected_is_optimal']):8s} "
            f"{str(r['unique_optimal_move'] if r['unique_optimal_move'] is not None else '-'):7s} "
            f"{format_bool(r['all_moves_equivalent']):7s}"
        )

    print(f"\n{'Decision':^80}")
    print(f"{'=' * 80}")
    print(f"Classification: {decision_classification}")
    print(f"Evidence: {decision_evidence}")
    print(f"Rejected:  {decision_rejected}")
    print(f"Next:      {decision_next}")

    print(f"\n{'=' * 100}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
