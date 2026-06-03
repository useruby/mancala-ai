#!/usr/bin/env python3
"""
Build a non-mutating tablebase-backed reference patch artifact for sparse_endgame.

Steps:
1. Validate sparse_endgame rows
2. Tablebase root adjudication
3. Tablebase child adjudication
4. ClassicMCTS cross-check
5. Build non-mutating patch artifact

Does not train, run arena, promote, create replay artifacts, or mutate
active reference fixtures.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
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

FAMILY = "sparse_endgame"
REFERENCE_PATH = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
SELECTED_ROWS_PATH = Path(
    "/tmp/azlite_corrected_non_opening_failure_mining_v6/"
    "selected_non_opening_family_rows_v6.jsonl"
)
PATCH_DIR = Path("/tmp/azlite_sparse_endgame_tablebase_patch")
PATCH_ARTIFACT_PATH = PATCH_DIR / "sparse_endgame_tablebase_reference_patch_v1.json"
PATCH_SUMMARY_PATH = PATCH_DIR / "sparse_endgame_tablebase_reference_patch_summary.json"

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

CLASSIC_BUDGETS = (1200, 2400, 5000, 10000)
CLASSIC_SEEDS = (11, 23, 37, 42, 101, 202, 303)


def tablebase_preferred_move_and_value(
    tablebase: EndgameTablebase, state: dict[str, Any], *, root_player: int
) -> tuple[float | None, int | None]:
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
    best_win_rate: float | None = None
    for move in legal_moves:
        child_game = KalahGame.from_state(child_state_from_move(state, move))
        child_win_rate = tablebase.lookup_cached(child_game, root_player)
        if child_win_rate is None:
            child_win_rate = tablebase.lookup(child_game, root_player)
        if child_win_rate is None:
            return round_float(root_value), None
        if best_win_rate is None or float(child_win_rate) > best_win_rate:
            best_win_rate = float(child_win_rate)
            best_move = int(move)
    return round_float(root_value), best_move


def tablebase_child_value(
    tablebase: EndgameTablebase, state: dict[str, Any], *, root_player: int
) -> float | None:
    game = KalahGame.from_state(state)
    win_rate = tablebase.lookup_cached(game, root_player)
    if win_rate is None:
        win_rate = tablebase.lookup(game, root_player)
    if win_rate is None:
        return None
    return round_float((2.0 * float(win_rate)) - 1.0)


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def format_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return str(bool(value)).lower()


def main() -> int:
    PATCH_DIR.mkdir(parents=True, exist_ok=True)

    suite = load_suite(SUITE_PATH)
    suite_by_id = {position.id: position for position in suite}
    reference_by_id, reference_by_canonical = load_reference_maps(REFERENCE_PATH)
    selected_rows = load_jsonl(SELECTED_ROWS_PATH)
    selected_by_id = {str(row["row_id"]): row for row in selected_rows}

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

    # Step 1: Validate rows
    tablebase = EndgameTablebase()
    validation_table: list[dict[str, Any]] = []
    for row_id in ALL_ROW_IDS:
        info = rows[row_id]
        state = info["state"]
        game = KalahGame.from_state(state)
        remaining_seed_count = int(sum(game.pits))
        root_player = int(state["current_player"])
        tb_value, tb_move = tablebase_preferred_move_and_value(
            tablebase, state, root_player=root_player
        )
        canonical = canonical_state_key(state)
        ref_canonical = str(
            info["reference_row"].get("canonical_state")
            or canonical_state_key(info["reference_row"]["state"])
        )
        canonical_match = canonical == ref_canonical
        validation_table.append(
            {
                "row_id": row_id,
                "active_reference_move": info["corrected_reference_move"],
                "legal": True,
                "canonical_state_match": canonical_match,
                "remaining_seed_count": remaining_seed_count,
                "root_tablebase_available": tb_value is not None,
                "status": "ok" if canonical_match else "canonical_mismatch",
                "notes": "validated" if canonical_match else "state hash mismatch",
            }
        )
        if not canonical_match:
            print(
                f"WARNING: {row_id} canonical state mismatch between suite and reference",
                file=sys.stderr,
            )

    # Step 2: Tablebase root adjudication
    root_adjudication_table: list[dict[str, Any]] = []
    for row_id in ALL_ROW_IDS:
        info = rows[row_id]
        state = info["state"]
        root_player = int(state["current_player"])
        tb_value, tb_move = tablebase_preferred_move_and_value(
            tablebase, state, root_player=root_player
        )
        active_ref = info["corrected_reference_move"]
        puct_selected = info["current_selected_move"]
        if tb_value is None:
            decision = "tablebase_unavailable"
            notes = "tablebase cannot solve this position"
        elif tb_move is None:
            decision = "tablebase_tie_or_ambiguous"
            notes = "tablebase has value but no preferred move (terminal or tied)"
        elif tb_move == active_ref:
            decision = "tablebase_confirmed"
            notes = f"tablebase prefers move {tb_move}, matches active reference"
        else:
            decision = "tablebase_conflict"
            notes = (
                f"tablebase prefers move {tb_move} (value {format_float(tb_value)}), "
                f"active reference is move {active_ref}"
            )
        root_adjudication_table.append(
            {
                "row_id": row_id,
                "active_reference_move": active_ref,
                "tablebase_preferred_move": tb_move,
                "tablebase_value_root": tb_value,
                "current_puct_selected_move": puct_selected,
                "tablebase_agrees_with_active_reference": (
                    tb_move is not None and tb_move == active_ref
                ),
                "tablebase_agrees_with_puct": (
                    tb_move is not None and tb_move == puct_selected
                ),
                "decision": decision,
                "notes": notes,
            }
        )

    # Step 3: Tablebase child adjudication
    child_adjudication_table: list[dict[str, Any]] = []
    for row_id in TARGET_ROW_IDS:
        info = rows[row_id]
        state = info["state"]
        root_player = int(state["current_player"])
        active_ref = info["corrected_reference_move"]
        puct_selected = info["current_selected_move"]
        _, tb_move = tablebase_preferred_move_and_value(
            tablebase, state, root_player=root_player
        )
        tb_move_int = tb_move if tb_move is not None else active_ref

        moves_to_evaluate = set([active_ref, puct_selected, tb_move_int])
        for move in sorted(moves_to_evaluate):
            child_state = child_state_from_move(state, move)
            child_value = tablebase_child_value(
                tablebase, child_state, root_player=root_player
            )
            if move == active_ref:
                child_ref_value = child_value
            elif move == tb_move_int:
                child_tb_value = child_value
            else:
                child_puct_value = child_value

        child_ref_value = tablebase_child_value(
            tablebase, child_state_from_move(state, active_ref), root_player=root_player
        )
        child_tb_value = tablebase_child_value(
            tablebase,
            child_state_from_move(state, tb_move_int),
            root_player=root_player,
        )
        child_puct_value = tablebase_child_value(
            tablebase,
            child_state_from_move(state, puct_selected),
            root_player=root_player,
        )

        active_minus_tb = (
            round_float(child_ref_value - child_tb_value)
            if child_ref_value is not None and child_tb_value is not None
            else None
        )
        active_minus_puct = (
            round_float(child_ref_value - child_puct_value)
            if child_ref_value is not None and child_puct_value is not None
            else None
        )

        for move, label in [
            (active_ref, "active_reference"),
            (tb_move_int, "tablebase_preferred"),
            (puct_selected, "puct_selected"),
        ]:
            child_state = child_state_from_move(state, move)
            child_val = tablebase_child_value(
                tablebase, child_state, root_player=root_player
            )
            child_adjudication_table.append(
                {
                    "row_id": row_id,
                    "child_from_move": move,
                    "child_label": label,
                    "child_value_root": child_val,
                    "active_minus_tablebase_preferred": (
                        active_minus_tb if label == "active_reference" else None
                    ),
                    "active_minus_puct_selected": (
                        active_minus_puct if label == "active_reference" else None
                    ),
                    "notes": f"tablebase evaluator for move {move}",
                }
            )

    # Step 4: ClassicMCTS cross-check
    classic_cross_check_table: list[dict[str, Any]] = []
    for row_id in TARGET_ROW_IDS:
        info = rows[row_id]
        state = info["state"]
        root_player = int(state["current_player"])
        _, tb_move = tablebase_preferred_move_and_value(
            tablebase, state, root_player=root_player
        )
        tb_pref = tb_move
        active_ref = info["corrected_reference_move"]
        for budget in CLASSIC_BUDGETS:
            seed_results: dict[int, dict[str, Any]] = {}
            for seed in CLASSIC_SEEDS:
                mcts = ClassicMCTS(
                    KalahGame.from_state(state),
                    simulations=budget,
                    seed=seed,
                )
                summary = mcts.root_summary()
                selected = summary.get("selected_move")
                child_stats_list = summary.get("child_stats") or []
                total_visits = sum(c.get("visits", 0) for c in child_stats_list)
                seed_results[seed] = {
                    "selected_move": selected,
                    "child_stats": child_stats_list,
                    "total_visits": total_visits,
                }
            move_counter: Counter[int] = Counter()
            for seed in CLASSIC_SEEDS:
                sel = seed_results[seed].get("selected_move")
                if sel is not None:
                    move_counter[int(sel)] += 1
            if move_counter:
                majority_move, majority_count = move_counter.most_common(1)[0]
                majority_fraction = round_float(majority_count / len(CLASSIC_SEEDS))
            else:
                majority_move = None
                majority_fraction = None
            top_two = move_counter.most_common(2)
            top1_top2_margin = None
            if len(top_two) >= 2:
                top1_top2_margin = top_two[0][1] - top_two[1][1]
            elif len(top_two) == 1:
                top1_top2_margin = top_two[0][1]

            agrees_with_tb = (
                tb_pref is not None
                and majority_move is not None
                and int(majority_move) == int(tb_pref)
            )
            agrees_with_ref = majority_move is not None and int(majority_move) == int(
                active_ref
            )

            classic_cross_check_table.append(
                {
                    "row_id": row_id,
                    "budget": budget,
                    "seeds": list(CLASSIC_SEEDS),
                    "tablebase_preferred_move": tb_pref,
                    "classic_majority_move": majority_move,
                    "classic_majority_fraction": majority_fraction,
                    "top1_top2_margin": top1_top2_margin,
                    "classic_agrees_with_tablebase": agrees_with_tb,
                    "classic_agrees_with_active_reference": agrees_with_ref,
                    "notes": (f"ClassicMCTS cross-check at budget {budget}"),
                }
            )

    # Step 5: Build patch artifact
    patch_entries: list[dict[str, Any]] = []
    for row_id in TARGET_ROW_IDS:
        info = rows[row_id]
        active_ref = info["corrected_reference_move"]
        state = info["state"]
        root_player = int(state["current_player"])
        tb_value, tb_move = tablebase_preferred_move_and_value(
            tablebase, state, root_player=root_player
        )

        # Determine proposed reference move
        proposed_move = active_ref
        proposed_unstable = False
        evidence: list[str] = []

        tblookup = next(
            (r for r in root_adjudication_table if r["row_id"] == row_id), {}
        )
        decision = tblookup.get("decision", "unknown")

        # Check if tablebase position is forcing (all legal moves are equivalent)
        is_forcing_position = tb_value is not None and abs(float(tb_value)) >= 1.0

        if decision == "tablebase_conflict":
            proposed_move = tb_move
            evidence.append(
                f"tablebase prefers move {tb_move} "
                f"(value {format_float(tb_value)}) over active reference {active_ref}"
            )
            if is_forcing_position:
                proposed_unstable = True
                evidence.append(
                    f"root value is {format_float(tb_value)} (forcing position): "
                    "all legal moves have equal win rate from root perspective; "
                    "tablebase preference is a tie-breaker, not a meaningful distinction"
                )
        elif decision == "tablebase_unavailable":
            proposed_unstable = True
            evidence.append("tablebase unavailable for root position")
        elif decision == "tablebase_tie_or_ambiguous":
            proposed_unstable = True
            evidence.append("tablebase value exists but no clear preferred move")

        # Add ClassicMCTS evidence
        classic_rows_for_id = [
            r for r in classic_cross_check_table if r["row_id"] == row_id
        ]
        for cr in classic_rows_for_id:
            if cr["classic_majority_move"] is not None:
                evidence.append(
                    f"ClassicMCTS budget={cr['budget']} majority={cr['classic_majority_move']} "
                    f"(fraction={format_float(cr['classic_majority_fraction'])}, "
                    f"agrees_with_tb={cr['classic_agrees_with_tablebase']})"
                )
                # If ClassicMCTS disagrees with tablebase at highest budget, add stability concern
                if (
                    cr["budget"] == max(CLASSIC_BUDGETS)
                    and not cr["classic_agrees_with_tablebase"]
                ):
                    proposed_unstable = True
                    evidence.append(
                        "ClassicMCTS at highest budget disagrees with tablebase preference"
                    )

        # Observed reference moves from fixture
        ref_row = info["reference_row"]
        observed_moves = list(ref_row.get("observed_reference_moves") or [active_ref])

        classic_majority_move = None
        classic_majority_fraction = None
        if classic_rows_for_id:
            highest_budget = max(cr["budget"] for cr in classic_rows_for_id)
            cr_high = next(
                (cr for cr in classic_rows_for_id if cr["budget"] == highest_budget),
                {},
            )
            classic_majority_move = cr_high.get("classic_majority_move")
            classic_majority_fraction = cr_high.get("classic_majority_fraction")

        entry = {
            "row_id": row_id,
            "canonical_state_hash": info["canonical_state_hash"],
            "current_active_reference_move": active_ref,
            "proposed_reference_move": proposed_move,
            "proposed_reference_unstable": proposed_unstable,
            "observed_reference_moves": observed_moves,
            "tablebase_available": tb_value is not None,
            "tablebase_value_root": tb_value,
            "tablebase_preferred_move": tb_move,
            "classic_majority_move": classic_majority_move,
            "classic_majority_fraction": classic_majority_fraction,
            "evidence_summary": "; ".join(evidence) if evidence else "stable reference",
            "do_not_auto_apply": True,
        }
        patch_entries.append(entry)

    patch_artifact = {
        "schema": "azlite_sparse_endgame_tablebase_reference_patch_v1",
        "family": FAMILY,
        "description": (
            "Non-mutating tablebase-backed reference patch artifact for sparse_endgame. "
            "All entries have do_not_auto_apply=true. Review before applying."
        ),
        "guardrails": {
            "mutated_active_fixture": False,
            "ran_training": False,
            "ran_arena": False,
            "promoted_model": False,
            "created_replay_artifacts": False,
        },
        "source_references": str(REFERENCE_PATH),
        "source_suite": str(SUITE_PATH),
        "entries": patch_entries,
    }
    write_json(PATCH_ARTIFACT_PATH, patch_artifact)
    print(f"Wrote patch artifact: {PATCH_ARTIFACT_PATH}")

    # Write summary
    summary = {
        "schema": "azlite_sparse_endgame_tablebase_reference_patch_summary_v1",
        "family": FAMILY,
        "patch_artifact": str(PATCH_ARTIFACT_PATH),
        "guardrails": {
            "mutated_active_fixture": False,
            "ran_training": False,
            "ran_arena": False,
            "promoted_model": False,
            "created_replay_artifacts": False,
        },
        "row_validation_table": validation_table,
        "root_adjudication_table": root_adjudication_table,
        "child_adjudication_table": child_adjudication_table,
        "classic_cross_check_table": classic_cross_check_table,
        "patch_entries": patch_entries,
    }
    write_json(PATCH_SUMMARY_PATH, summary)
    print(f"Wrote patch summary: {PATCH_SUMMARY_PATH}")

    # Print summary tables
    print("\n=== Row Validation ===")
    print(
        f"{'row_id':30s} {'ref_move':8s} {'legal':6s} {'canonical':10s} "
        f"{'seeds':6s} {'tb_avail':9s} {'status':10s}"
    )
    for v in validation_table:
        print(
            f"{v['row_id']:30s} {str(v['active_reference_move']):8s} "
            f"{format_bool(v['legal']):6s} {format_bool(v['canonical_state_match']):10s} "
            f"{str(v['remaining_seed_count']):6s} "
            f"{format_bool(v['root_tablebase_available']):9s} "
            f"{v['status']:10s}"
        )

    print("\n=== Root Adjudication ===")
    print(
        f"{'row_id':30s} {'ref':4s} {'tb_pref':8s} {'tb_val':8s} "
        f"{'puct':4s} {'tb_agree_ref':12s} {'tb_agree_puct':12s} "
        f"{'decision':28s}"
    )
    for r in root_adjudication_table:
        print(
            f"{r['row_id']:30s} {str(r['active_reference_move']):4s} "
            f"{str(r['tablebase_preferred_move'] if r['tablebase_preferred_move'] is not None else '-'):8s} "
            f"{format_float(r['tablebase_value_root']):8s} "
            f"{str(r['current_puct_selected_move']):4s} "
            f"{format_bool(r['tablebase_agrees_with_active_reference']):12s} "
            f"{format_bool(r['tablebase_agrees_with_puct']):12s} "
            f"{r['decision']:28s}"
        )

    print("\n=== Classic Cross-Check (Highest Budget) ===")
    high_budget_rows = {}
    for cr in classic_cross_check_table:
        key = cr["row_id"]
        if (
            key not in high_budget_rows
            or cr["budget"] > high_budget_rows[key]["budget"]
        ):
            high_budget_rows[key] = cr
    print(
        f"{'row_id':30s} {'budget':8s} {'tb_pref':8s} {'maj_move':8s} {'maj_frac':8s} {'agrees_tb':10s} {'agrees_ref':10s}"
    )
    for cr in sorted(high_budget_rows.values(), key=lambda x: x["row_id"]):
        print(
            f"{cr['row_id']:30s} {str(cr['budget']):8s} "
            f"{str(cr['tablebase_preferred_move'] if cr['tablebase_preferred_move'] is not None else '-'):8s} "
            f"{str(cr['classic_majority_move'] if cr['classic_majority_move'] is not None else '-'):8s} "
            f"{format_float(cr['classic_majority_fraction']):8s} "
            f"{format_bool(cr['classic_agrees_with_tablebase']):10s} "
            f"{format_bool(cr['classic_agrees_with_active_reference']):10s}"
        )

    print("\n=== Patch Entries (changes only) ===")
    for entry in patch_entries:
        changed = (
            entry["proposed_reference_move"] != entry["current_active_reference_move"]
        )
        status = "CHANGED" if changed else "STABLE"
        print(
            f"{entry['row_id']:30s} {status:8s} "
            f"ref {entry['current_active_reference_move']} -> "
            f"{entry['proposed_reference_move']} "
            f"(tb_pref={entry['tablebase_preferred_move']}, "
            f"unstable={entry['proposed_reference_unstable']})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
