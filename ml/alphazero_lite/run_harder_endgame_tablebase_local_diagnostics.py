#!/usr/bin/env python3
"""Tablebase-backed local search/value diagnostics for harder endgame tablebase rows.

Steps:
  1. Load and validate selected hard rows from PR #73.
  2. Exact tablebase re-enumeration.
  3. Detailed PUCT budget sweep 32/64/128/256/384/768/1200/2400/5000.
  4. Root policy-prior audit.
  5. Neural value rank audit.
  6. Child PUCT audit.
  7. Root counterfactual diagnostics.
  8. Row mechanism classification.
  9. Build refined clean split.
  10. Decide next branch.
  11. Write report.

Does not train, run arena, promote, create replay artifacts, or mutate
active reference fixtures.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import (
    Node,
    PUCT,
    build_eval_search_options,
    terminal_value,
)

EPS = 1e-9
FAMILY = "harder_fresh_endgame_tablebase"
C_PUCT = 1.25
SEARCH_OPTIONS = build_eval_search_options(
    fpu_mode="parent_q",
    reuse_subtree=True,
    normalize_values=True,
    root_policy_mode="deterministic",
    tactical_root_bias=0.1,
)
ROOT_BUDGETS = (32, 64, 128, 256, 384, 768, 1200, 2400, 5000)
CHILD_PUCT_BUDGETS = (384, 1200, 2400)
COUNTERFACTUAL_BUDGETS = (384, 1200)
SEED = 17

CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
REFERENCE_PATH = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
SELECTED_ROWS_PATH = Path(
    "/tmp/azlite_harder_fresh_endgame_tablebase_mining/"
    "selected_harder_endgame_tablebase_rows.jsonl"
)
OUTPUT_DIR = Path("/tmp/azlite_harder_endgame_tablebase_local_diagnostics")

EXHAUSTED_ROW_ID_PREFIXES: frozenset[str] = frozenset(
    {
        "incumbent_proxy_disagreement",
        "incumbent_proxy_residual",
        "high_value_swing",
        "high_imbalance",
        "capture_available",
        "starvation_pressure",
        "sparse_endgame",
        "early_extra_turn",
        "opening_plies_1_8",
        "opening_extra_turn",
        "opening_edge_move",
        "opening_missed_extra_turn",
    }
)
EXHAUSTED_BUCKETS: frozenset[str] = frozenset(
    {
        "opening_plies_1_8",
        "opening_extra_turn_overbias",
        "opening_edge_move_5_preference",
        "opening_missed_extra_turn_continuation",
        "incumbent_proxy_disagreement",
        "incumbent_proxy_residual",
        "high_value_swing",
        "high_imbalance",
        "capture_available",
        "starvation_pressure",
        "sparse_endgame",
        "early_extra_turn",
    }
)
CORRECTED_GUARD_ROW_IDS: frozenset[str] = frozenset(
    {
        "capture_available-002",
        "capture_available-003",
        "capture_available-006",
        "capture_available-007",
        "capture_available-008",
    }
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def format_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return str(bool(value)).lower()


def child_state_from_move(root_state: dict[str, Any], move: int) -> dict[str, Any]:
    game = KalahGame.from_state(root_state)
    succeeded = game.move(game.pit_index(int(move)))
    if not succeeded:
        raise ValueError(f"illegal move {move} for state")
    return game.to_state()


def state_to_root_perspective_value(
    *, raw_value: float, state: dict[str, Any], root_player: int
) -> float:
    return (
        float(raw_value)
        if int(state["current_player"]) == int(root_player)
        else -float(raw_value)
    )


def root_to_state_perspective_value(
    *, root_value: float, state: dict[str, Any], root_player: int
) -> float:
    return (
        float(root_value)
        if int(state["current_player"]) == int(root_player)
        else -float(root_value)
    )


def is_exhausted_row_id(row_id: str) -> bool:
    for prefix in EXHAUSTED_ROW_ID_PREFIXES:
        if row_id.startswith(prefix):
            return True
    if row_id in CORRECTED_GUARD_ROW_IDS:
        return True
    return False


def visit_share(visits: list[float], move: int) -> float | None:
    total = sum(float(v) for v in visits)
    if total <= 0 or move >= len(visits):
        return None
    return round_float(float(visits[move]) / float(total))


def selection_entry_map(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    selection_breakdown = result.get("selection_breakdown") or {}
    return {
        int(entry["move"]): entry
        for entry in list(selection_breakdown.get("moves") or [])
        if isinstance(entry, dict) and entry.get("move") is not None
    }


def total_seeds_remaining(state: dict[str, Any]) -> int:
    return sum(state.get("player_pits", [])) + sum(state.get("opponent_pits", []))


def load_reference_maps(
    reference_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    by_id: dict[str, dict[str, Any]] = {}
    by_canonical: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("id") is None:
            continue
        by_id[str(row["id"])] = row
        canonical = str(row.get("canonical_state") or canonical_state_key(row["state"]))
        by_canonical[canonical] = row
    return by_id, by_canonical


def load_suite(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("rows", raw)
    return {str(row["id"]): row for row in raw if "id" in row}


def suite_canonical_states(suite: dict[str, dict]) -> set[str]:
    result: set[str] = set()
    for row in suite.values():
        if "canonical_state" in row:
            result.add(str(row["canonical_state"]))
        elif "state" in row:
            result.add(canonical_state_key(row["state"]))
    return result


# ── Step 1: Load and validate selected rows ─────────────────────────────────


def compute_tablebase_child_values(
    state: dict[str, Any],
) -> dict[str, Any]:
    game = KalahGame.from_state(state)
    tb = EndgameTablebase()
    legal = game.possible_moves()
    offset = game.current_player * 6
    child_values: dict[int, float] = {}
    optimal: list[int] = []
    best_val = -float("inf")
    for move in legal:
        child_game = game.clone()
        child_game.move(offset + move)
        c_wr = tb.lookup(child_game, game.current_player)
        if c_wr is not None:
            cv = (2.0 * float(c_wr)) - 1.0
            child_values[move] = round_float(cv)
            if cv > best_val:
                best_val = cv
    if best_val > -float("inf"):
        optimal = sorted(m for m, v in child_values.items() if abs(v - best_val) < EPS)
    return {
        "legal_moves": legal,
        "child_values": child_values,
        "optimal_moves": optimal,
        "best_val": None if best_val == -float("inf") else round_float(best_val),
    }


def validate_selected_rows(
    selected_rows: list[dict[str, Any]],
    suite_rows: dict[str, dict],
    ref_by_id: dict[str, dict],
    suite_canon: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation_entries: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    tb = EndgameTablebase()

    for row in selected_rows:
        cid = str(row.get("candidate_id", ""))
        family = str(row.get("provisional_family", ""))
        c_hash = str(row.get("canonical_state_hash", ""))
        state = row.get("state")
        role = str(row.get("assigned_role", ""))
        do_not_train = bool(row.get("do_not_train_yet", False))
        failure_class = str(row.get("failure_class", ""))
        exact_signal_class = str(row.get("exact_signal_class", ""))

        notes: list[str] = []
        valid = True

        # family check
        if family != FAMILY:
            notes.append(f"family mismatch: expected {FAMILY}, got {family}")
            valid = False

        # do_not_train check
        if not do_not_train:
            notes.append("do_not_train_yet is false/absent")
            valid = False

        # state and legal moves
        legal_moves: list[int] = []
        if not state:
            notes.append("missing state")
            valid = False
        else:
            game = KalahGame.from_state(state)
            legal_moves = game.possible_moves()
            if not legal_moves:
                notes.append("no legal moves")
                valid = False
            else:
                row["_legal_moves"] = legal_moves

        # canonical_state_hash
        if not c_hash:
            notes.append("missing canonical_state_hash")
            valid = False
        elif c_hash in seen_hashes:
            notes.append(f"duplicate canonical hash: {c_hash}")
            valid = False
        seen_hashes.add(c_hash)

        # role validation
        valid_roles = {
            "target_candidate",
            "preservation_control",
            "holdout_candidate",
            "low_budget_diagnostic_only",
        }
        if role not in valid_roles:
            notes.append(f"invalid assigned_role: {role}")
            valid = False

        # exhausted family overlap
        if is_exhausted_row_id(cid):
            notes.append("exhausted fixture row id prefix")
            valid = False

        # teacher conflict
        if failure_class == "teacher_conflict":
            notes.append("teacher conflict detected")
            valid = False

        # all_moves_equivalent
        if exact_signal_class == "all_moves_equivalent":
            notes.append("all_moves_equivalent=true")
            valid = False

        # tablebase availability at root and unique optimal
        if state and legal_moves:
            game = KalahGame.from_state(state)
            tb_wr = tb.lookup(game, game.current_player)
            if tb_wr is None:
                notes.append("tablebase unavailable at root")
                valid = False
            else:
                offset_i = game.current_player * 6
                child_vals: dict[int, float] = {}
                best_val = -float("inf")
                for move in legal_moves:
                    child_game = game.clone()
                    child_game.move(offset_i + move)
                    c_wr = tb.lookup(child_game, game.current_player)
                    if c_wr is not None:
                        cv = (2.0 * float(c_wr)) - 1.0
                        child_vals[move] = round_float(cv)
                        if cv > best_val:
                            best_val = cv
                if best_val > -float("inf"):
                    optimal_moves = sorted(
                        m for m, v in child_vals.items() if abs(v - best_val) < EPS
                    )
                    if len(optimal_moves) != 1:
                        notes.append(
                            f"not unique optimal: {len(optimal_moves)} optimal moves"
                        )
                        valid = False
                    else:
                        row["_tb_child_values"] = child_vals
                        row["_tb_optimal_moves"] = optimal_moves
                        row["_tb_unique_optimal"] = optimal_moves[0]
                else:
                    notes.append("no child values from tablebase")
                    valid = False

        # Check exhausted overlap via suite
        if state and not is_exhausted_row_id(cid):
            c_hash_val = canonical_state_key(state)
            if c_hash_val in suite_canon:
                for suite_id, suite_row in suite_rows.items():
                    suite_cs = suite_row.get("canonical_state") or canonical_state_key(
                        suite_row.get("state", {})
                    )
                    if suite_cs == c_hash_val:
                        if is_exhausted_row_id(suite_id):
                            notes.append(f"overlaps exhausted suite row {suite_id}")
                            valid = False
                        bucket = suite_row.get("bucket", "")
                        if bucket in EXHAUSTED_BUCKETS:
                            notes.append(
                                f"overlaps exhausted bucket {bucket} via {suite_id}"
                            )
                            valid = False
                        break

        validation_entries.append(
            {
                "candidate_id": cid,
                "assigned_role_from_pr73": role,
                "legal_moves": len(legal_moves),
                "tablebase_available": True,
                "unique_optimal": True,
                "exhausted_overlap": not valid and any("exhausted" in n for n in notes),
                "teacher_conflict": any("teacher conflict" in n for n in notes),
                "valid": valid,
                "notes": "; ".join(notes) if notes else "ok",
            }
        )

        if valid:
            valid_rows.append(row)

    return valid_rows, validation_entries


# ── Step 2: Exact tablebase re-enumeration ─────────────────────────────────


def exact_tablebase_enumeration(
    valid_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tb = EndgameTablebase()
    results: list[dict[str, Any]] = []

    for row in valid_rows:
        cid = str(row.get("candidate_id", ""))
        state = row.get("state")
        game = KalahGame.from_state(state)
        root_player = game.current_player

        tb_wr = tb.lookup(game, root_player)
        if tb_wr is None:
            results.append(
                {
                    "candidate_id": cid,
                    "root_value": None,
                    "optimal_move": None,
                    "second_best_move": None,
                    "best_minus_second_best": None,
                    "forced_win_or_loss": None,
                    "exact_signal_class": "tablebase_error",
                    "notes": "tablebase unavailable at root",
                }
            )
            continue

        root_value = (2.0 * float(tb_wr)) - 1.0
        legal_moves = game.possible_moves()
        offset = game.current_player * 6
        child_values: dict[int, float] = {}

        for move in legal_moves:
            child_game = game.clone()
            child_game.move(offset + move)
            c_wr = tb.lookup(child_game, root_player)
            if c_wr is not None:
                cv = (2.0 * float(c_wr)) - 1.0
                child_values[move] = round_float(cv)
            else:
                child_values[move] = 0.0

        best_val = max(
            (v for v in child_values.values() if v is not None),
            default=-float("inf"),
        )
        optimal_moves = sorted(
            m
            for m, v in child_values.items()
            if v is not None and abs(v - best_val) < EPS
        )
        unique_optimal = len(optimal_moves) == 1
        optimal_move = optimal_moves[0] if unique_optimal else None

        sorted_vals = sorted(
            [v for v in child_values.values() if v is not None], reverse=True
        )
        second_best = sorted_vals[1] if len(sorted_vals) >= 2 else None
        best_minus_second = (
            round_float(best_val - second_best) if second_best is not None else None
        )

        forced_win_loss = abs(root_value) > 0.99 if root_value is not None else None

        if not unique_optimal:
            signal = "tablebase_not_unique"
        elif best_minus_second is not None and best_minus_second < EPS:
            signal = "exact_unique_tiny_margin"
        else:
            signal = "exact_unique_clear_margin"

        results.append(
            {
                "candidate_id": cid,
                "root_value": round_float(root_value),
                "optimal_move": optimal_move,
                "optimal_moves": optimal_moves,
                "second_best_move": second_best,
                "best_minus_second_best": best_minus_second,
                "forced_win_or_loss": forced_win_loss,
                "exact_signal_class": signal,
                "notes": "ok",
            }
        )

    return results


# ── Step 3: Detailed PUCT budget sweep ──────────────────────────────────────


def run_single_puct(
    evaluator: ArtifactEvaluator,
    state: dict[str, Any],
    budget: int,
    seed: int,
) -> dict[str, Any]:
    """Run a single deterministic PUCT evaluation at the given budget."""
    result = evaluate_artifact_position(
        artifact_path=None,
        evaluator=evaluator,
        state=state,
        simulations=int(budget),
        seed=int(seed),
        c_puct=C_PUCT,
        search_options=dict(SEARCH_OPTIONS),
        ablation_mode="full",
    )
    return result


def policy_rank_of_move(policy: list[float], move: int) -> int | None:
    if not policy or move is None:
        return None
    sorted_moves = sorted(
        range(len(policy)),
        key=lambda m: (float(policy[m]), -m),
        reverse=True,
    )
    for rank, m in enumerate(sorted_moves):
        if m == move:
            return rank
    return None


def run_puct_budget_sweep_row(
    evaluator: ArtifactEvaluator,
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run all budgets for a single row."""
    cid = str(row.get("candidate_id", ""))
    state = row["state"]
    optimal_move = row.get("_tb_unique_optimal")
    budgets = list(ROOT_BUDGETS)

    row_results: list[dict[str, Any]] = []
    first_budget_optimal = None

    for budget in budgets:
        result = run_single_puct(evaluator, state, budget, SEED)
        selected_move = (
            None
            if result.get("selected_move") is None
            else int(result["selected_move"])
        )
        selection_map = selection_entry_map(result)
        sel_entry = (
            selection_map.get(selected_move) if selected_move is not None else {}
        )
        opt_entry = selection_map.get(optimal_move) if optimal_move is not None else {}
        selected_is_optimal = (
            selected_move == optimal_move if optimal_move is not None else None
        )
        visits_list = [float(v) for v in result.get("visits", [])]
        policy_list = [float(p) for p in result.get("policy", [])]

        optimal_policy_rank = policy_rank_of_move(policy_list, optimal_move)
        selected_policy_rank = policy_rank_of_move(policy_list, selected_move)

        if first_budget_optimal is None:
            first_budget_optimal = selected_is_optimal

        row_results.append(
            {
                "candidate_id": cid,
                "budget": int(budget),
                "optimal_move": optimal_move,
                "selected_move": selected_move,
                "selected_is_optimal": selected_is_optimal,
                "optimal_visit_share": visit_share(visits_list, optimal_move)
                if optimal_move is not None
                else None,
                "selected_visit_share": visit_share(visits_list, selected_move)
                if selected_move is not None
                else None,
                "optimal_q": round_float(float(opt_entry.get("q_value", 0.0)))
                if opt_entry
                else None,
                "selected_q": round_float(float(sel_entry.get("q_value", 0.0)))
                if sel_entry
                else None,
                "selected_minus_optimal_q_margin": round_float(
                    float(sel_entry.get("q_value", 0.0))
                    - float(opt_entry.get("q_value", 0.0))
                )
                if sel_entry and opt_entry
                else None,
                "optimal_policy_probability": round_float(
                    float(opt_entry.get("prior", 0.0))
                )
                if opt_entry
                else None,
                "selected_policy_probability": round_float(
                    float(sel_entry.get("prior", 0.0))
                )
                if sel_entry
                else None,
                "optimal_policy_rank": optimal_policy_rank,
                "selected_policy_rank": selected_policy_rank,
                "first_budget_optimal_selected": first_budget_optimal,
                "notes": "deterministic PUCT baseline",
            }
        )

    return row_results


def run_puct_budget_sweep(
    evaluator: ArtifactEvaluator,
    valid_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run PUCT budget sweep for all valid rows."""
    all_results: list[dict[str, Any]] = []

    for row in valid_rows:
        row_results = run_puct_budget_sweep_row(evaluator, row)
        all_results.extend(row_results)

        # Determine failure classifications
        passes_384 = any(
            r["budget"] == 384 and r["selected_is_optimal"] for r in row_results
        )
        passes_1200 = any(
            r["budget"] == 1200 and r["selected_is_optimal"] for r in row_results
        )
        passes_32 = any(
            r["budget"] == 32 and r["selected_is_optimal"] for r in row_results
        )
        passes_64 = any(
            r["budget"] == 64 and r["selected_is_optimal"] for r in row_results
        )

        # Persistent failure: fails at 384 and 1200
        persistent_failure = not passes_384 and not passes_1200
        # Medium-budget failure: fails at 384 but passes at 1200
        medium_budget_failure = not passes_384 and passes_1200
        # Low-budget only: fails below 384 but passes at 384+
        low_budget_failure = (
            (not passes_32 or not passes_64) and passes_384 and passes_1200
        )

        row["_persistent_failure_384_1200"] = persistent_failure
        row["_medium_budget_failure"] = medium_budget_failure
        row["_low_budget_only_failure"] = low_budget_failure

        # Check budget resolution for targetability purposes
        for r in row_results:
            if r["selected_is_optimal"]:
                row["_first_pass_budget"] = r["budget"]
                break
        else:
            row["_first_pass_budget"] = None

        row["_puct_results"] = row_results

    return all_results


# ── Step 4: Root policy-prior audit ────────────────────────────────────────


def root_policy_prior_audit(
    valid_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare policy priors on optimal vs selected moves."""
    results: list[dict[str, Any]] = []

    for row in valid_rows:
        cid = str(row.get("candidate_id", ""))
        state = row["state"]
        optimal_move = row.get("_tb_unique_optimal")

        game = KalahGame.from_state(state)

        # Evaluate network for policy
        evaluator = ArtifactEvaluator(CURRENT_ARTIFACT)
        policy, _ = evaluator.evaluate(game)

        # Find selected move from 1200 budget if available
        puct_results = row.get("_puct_results", [])
        r1200 = next((r for r in puct_results if r["budget"] == 1200), None)
        selected_move = r1200["selected_move"] if r1200 else None

        if optimal_move is None:
            continue

        optimal_prior = (
            float(policy[optimal_move]) if optimal_move < len(policy) else 0.0
        )
        selected_prior = (
            float(policy[selected_move])
            if selected_move is not None and selected_move < len(policy)
            else 0.0
        )
        optimal_prior_rank = policy_rank_of_move(
            [float(p) for p in policy], optimal_move
        )

        prior_diff = round_float(optimal_prior - selected_prior)

        # Classify prior relationship
        if selected_move is not None and selected_move != optimal_move:
            if optimal_prior_rank is not None and optimal_prior_rank >= 2:
                classification = "policy_prior_underweights_optimal"
            elif optimal_prior >= 0.5:
                classification = "policy_prior_supports_optimal"
            else:
                classification = "policy_prior_neutral"
        else:
            classification = "policy_prior_supports_optimal"

        results.append(
            {
                "candidate_id": cid,
                "optimal_move": optimal_move,
                "selected_move": selected_move,
                "optimal_policy_probability": round_float(optimal_prior),
                "selected_policy_probability": round_float(selected_prior),
                "optimal_policy_rank": optimal_prior_rank,
                "optimal_prior_minus_selected_prior": prior_diff,
                "prior_classification": classification,
                "notes": "",
            }
        )

        row["_policy_prior_audit"] = results[-1]

    return results


# ── Step 5: Neural value rank audit ────────────────────────────────────────


def neural_value_rank_audit(
    evaluator: ArtifactEvaluator,
    valid_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Audit neural child value rankings vs exact tablebase."""
    tb = EndgameTablebase()
    results: list[dict[str, Any]] = []

    for row in valid_rows:
        cid = str(row.get("candidate_id", ""))
        state = row["state"]
        root_player = KalahGame.from_state(state).current_player
        optimal_move = row.get("_tb_unique_optimal")
        legal_moves = row.get(
            "_legal_moves", KalahGame.from_state(state).possible_moves()
        )

        if optimal_move is None:
            continue

        # Evaluate all legal children
        neural_child_values: dict[int, float] = {}
        exact_child_values: dict[int, float] = {}

        for move in legal_moves:
            child_state = child_state_from_move(state, move)
            _, raw_nv = evaluator.evaluate(KalahGame.from_state(child_state))
            nv = state_to_root_perspective_value(
                raw_value=float(raw_nv), state=child_state, root_player=root_player
            )
            neural_child_values[move] = round_float(nv)

            c_wr = tb.lookup(KalahGame.from_state(child_state), root_player)
            if c_wr is not None:
                exact_child_values[move] = round_float((2.0 * float(c_wr)) - 1.0)

        if not neural_child_values:
            continue

        # Neural best child
        neural_best_move = max(
            neural_child_values, key=lambda m: neural_child_values[m]
        )
        neural_best_is_exact_optimal = neural_best_move == optimal_move

        # Value rank error: neural ranks a non-optimal child above optimal
        value_rank_error = False
        if not neural_best_is_exact_optimal and optimal_move in neural_child_values:
            if (
                neural_child_values[neural_best_move]
                > neural_child_values[optimal_move]
            ):
                value_rank_error = True

        # Sign error on root
        tb_wr = tb.lookup(KalahGame.from_state(state), root_player)
        exact_root = (2.0 * float(tb_wr)) - 1.0 if tb_wr is not None else None
        _, raw_root_nv = evaluator.evaluate(KalahGame.from_state(state))
        neural_root = float(raw_root_nv)
        sign_error = False
        if exact_root is not None and abs(exact_root) > EPS:
            exact_sign = math.copysign(1.0, exact_root)
            neural_sign = math.copysign(1.0, neural_root)
            sign_error = exact_sign != neural_sign

        optimal_neural = neural_child_values.get(optimal_move)
        neural_best_exact = exact_child_values.get(neural_best_move)
        optimal_exact = exact_child_values.get(optimal_move)
        neural_best_val = neural_child_values.get(neural_best_move)

        # Value error class
        if value_rank_error:
            v_class = "neural_prefers_nonoptimal_child"
        elif sign_error:
            v_class = "neural_sign_error"
        elif optimal_neural is not None and optimal_exact is not None:
            # Neural underestimates optimal: optimal child neural < exact optimal
            if abs(optimal_neural - optimal_exact) > 0.1:
                v_class = "neural_underestimates_optimal_child"
            else:
                puct_results = row.get("_puct_results", [])
                fails_384 = not any(
                    r["budget"] == 384 and r["selected_is_optimal"]
                    for r in puct_results
                )
                if fails_384:
                    v_class = "neural_ok_search_fails"
                else:
                    v_class = "neural_ok_search_fails"
        else:
            v_class = "neural_ok_search_fails"

        exact_value_gap = row.get(
            "_tb_best_minus_second", row.get("tb_best_minus_second")
        )
        if isinstance(exact_value_gap, str):
            try:
                exact_value_gap = float(exact_value_gap)
            except (ValueError, TypeError):
                exact_value_gap = None

        neural_value_gap = None
        if optimal_neural is not None and neural_best_val is not None:
            neural_value_gap = round_float(neural_best_val - optimal_neural)

        results.append(
            {
                "candidate_id": cid,
                "exact_optimal_move": optimal_move,
                "neural_best_child_move": neural_best_move,
                "neural_best_is_exact_optimal": neural_best_is_exact_optimal,
                "exact_optimal_child_neural_value": optimal_neural,
                "neural_best_child_exact_value": neural_best_exact,
                "exact_optimal_child_exact_value": optimal_exact,
                "neural_best_child_neural_value": neural_best_val,
                "neural_value_rank_error": value_rank_error,
                "neural_value_sign_error": sign_error,
                "exact_value_gap": exact_value_gap,
                "neural_value_gap": neural_value_gap,
                "value_error_class": v_class,
                "notes": "",
            }
        )

        row["_neural_value_audit"] = results[-1]

    return results


# ── Step 6: Child PUCT audit ──────────────────────────────────────────────


def child_puct_audit(
    evaluator: ArtifactEvaluator,
    valid_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run PUCT from child states to check if child PUCT values explain root failure."""
    tb = EndgameTablebase()
    results: list[dict[str, Any]] = []

    for row in valid_rows:
        cid = str(row.get("candidate_id", ""))
        state = row["state"]
        root_player = KalahGame.from_state(state).current_player
        optimal_move = row.get("_tb_unique_optimal")

        # Only audit rows where root PUCT fails at 384 or 1200
        puct_results = row.get("_puct_results", [])
        fails_384 = not any(
            r["budget"] == 384 and r["selected_is_optimal"] for r in puct_results
        )
        fails_1200 = not any(
            r["budget"] == 1200 and r["selected_is_optimal"] for r in puct_results
        )
        if not fails_384 and not fails_1200:
            continue

        # Get the selected (non-optimal) move from 384 or 1200
        r384 = next((r for r in puct_results if r["budget"] == 384), None)
        selected_move = r384["selected_move"] if r384 else None
        if selected_move is None or selected_move == optimal_move:
            selected_move = None
            r1200 = next((r for r in puct_results if r["budget"] == 1200), None)
            if r1200:
                selected_move = (
                    r1200["selected_move"]
                    if r1200["selected_move"] != optimal_move
                    else None
                )

        child_branches: list[tuple[str, int | None]] = [
            ("optimal_child", optimal_move),
            ("selected_child", selected_move),
        ]

        for budget in CHILD_PUCT_BUDGETS:
            for child_label, child_move in child_branches:
                if child_move is None:
                    continue
                if (
                    child_label == "selected_child"
                    and optimal_move is not None
                    and child_move == optimal_move
                ):
                    continue

                child_state = child_state_from_move(state, child_move)
                child_game = KalahGame.from_state(child_state)

                child_result = evaluate_artifact_position(
                    artifact_path=None,
                    evaluator=evaluator,
                    state=child_state,
                    simulations=int(budget),
                    seed=SEED,
                    c_puct=C_PUCT,
                    search_options=dict(SEARCH_OPTIONS),
                    ablation_mode="full",
                )

                # Child PUCT value (parent_q from root perspective)
                child_raw = float(
                    (child_result.get("selection_breakdown") or {}).get(
                        "parent_q_value", 0.0
                    )
                )
                child_puct_value_root = state_to_root_perspective_value(
                    raw_value=child_raw,
                    state=child_state,
                    root_player=root_player,
                )

                # Child exact value
                c_wr = tb.lookup(child_game, root_player)
                child_exact_root = (
                    (2.0 * float(c_wr)) - 1.0 if c_wr is not None else None
                )
                child_puct_error = (
                    round_float(child_puct_value_root - child_exact_root)
                    if child_exact_root is not None
                    else None
                )

                # Does child PUCT select tablebase-optimal from child state?
                child_selected = child_result.get("selected_move")
                c_legal = KalahGame.from_state(child_state).possible_moves()
                if c_legal:
                    c_offset = child_game.current_player * 6
                    cv_map: dict[int, float] = {}
                    best_cv = -float("inf")
                    for m in c_legal:
                        cg = child_game.clone()
                        cg.move(c_offset + m)
                        cwr2 = tb.lookup(cg, root_player)
                        if cwr2 is not None:
                            cv2 = (2.0 * float(cwr2)) - 1.0
                            cv_map[m] = cv2
                            if cv2 > best_cv:
                                best_cv = cv2
                    child_optimal_from_state_moves = sorted(
                        m for m, v in cv_map.items() if abs(v - best_cv) < EPS
                    )
                    child_selected_is_tb_optimal = (
                        child_selected in child_optimal_from_state_moves
                        if child_selected is not None
                        else None
                    )
                else:
                    child_selected_is_tb_optimal = None

                # Classification
                if (
                    child_exact_root is not None
                    and child_puct_error is not None
                    and abs(child_puct_error) > 0.05
                ):
                    child_class = "child_puct_value_error"
                elif child_selected_is_tb_optimal is False:
                    child_class = "child_puct_ok_root_selection_fails"
                else:
                    child_class = "child_puct_inconclusive"

                results.append(
                    {
                        "candidate_id": cid,
                        "child_branch": child_label,
                        "budget": int(budget),
                        "child_exact_value_root": child_exact_root,
                        "child_puct_value_root": round_float(child_puct_value_root),
                        "child_puct_error": child_puct_error,
                        "child_selected_move": child_selected,
                        "child_puct_agrees_with_exact": child_selected_is_tb_optimal,
                        "classification": child_class,
                        "notes": "",
                    }
                )

    return results


# ── Diagnostic PUCT for counterfactuals ──────────────────────────────────────


class DiagnosticPUCT(PUCT):
    def __init__(
        self,
        *args,
        child_value_overrides: dict[int, float] | None = None,
        override_child_selection_score: Callable | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.child_value_overrides = dict(child_value_overrides or {})
        self.root_move_by_state: dict[str, int] = {}

    def register_root_children(self, root: Node) -> None:
        for move, child in root.children.items():
            self.root_move_by_state[self._state_key(child.game)] = int(move)

    def _state_key(self, game: KalahGame) -> str:
        return json.dumps(game.to_state(), sort_keys=True, separators=(",", ":"))

    def _search(self, node: Node) -> float:
        terminal = terminal_value(node.game)
        if terminal is not None:
            return terminal
        if not node.expanded:
            _, value = self._expand(
                node,
                apply_dirichlet=False,
                dirichlet_alpha=None,
                dirichlet_epsilon=0.0,
                is_root=False,
            )
            root_move = self.root_move_by_state.get(self._state_key(node.game))
            if root_move is not None and root_move in self.child_value_overrides:
                value = float(self.child_value_overrides[root_move])
            return value
        if not node.children:
            return 0.0
        child = self._select_child(node)
        value = self._search(child)
        if child.game.current_player != node.game.current_player:
            value = -value
        child.visit_count += 1
        child.value_sum += value
        return value


def uniform_legal_prior_override(
    *, game, legal_moves: list[int], priors: np.ndarray
) -> np.ndarray:
    del game, priors
    adjusted = np.zeros(6, dtype=np.float32)
    if legal_moves:
        adjusted[legal_moves] = 1.0 / len(legal_moves)
    return adjusted


def make_equalize_optimal_selected_priors_override(
    optimal_move: int, selected_move: int
) -> Callable[..., np.ndarray]:
    def override(*, game, legal_moves: list[int], priors: np.ndarray) -> np.ndarray:
        del game
        adjusted = np.asarray(priors, dtype=np.float32).copy()
        if optimal_move in legal_moves and selected_move in legal_moves:
            target = max(
                float(adjusted[optimal_move]),
                float(adjusted[selected_move]),
            )
            adjusted[optimal_move] = target
            adjusted[selected_move] = target
        normalized = np.zeros_like(adjusted)
        normalized[legal_moves] = adjusted[legal_moves]
        total = float(np.sum(normalized[legal_moves]))
        if total > 0.0:
            normalized[legal_moves] /= total
        else:
            normalized[legal_moves] = 1.0 / len(legal_moves)
        return normalized.astype(np.float32)

    return override


def make_policy_prior_boost_optimal_override(
    optimal_move: int, legal_moves: list[int]
) -> Callable[..., np.ndarray]:
    def override(*, game, legal_moves: list[int], priors: np.ndarray) -> np.ndarray:
        del game
        adjusted = np.asarray(priors, dtype=np.float32).copy()
        if optimal_move in legal_moves:
            # Small diagnostic boost
            adjusted[optimal_move] += 0.05
        normalized = np.zeros_like(adjusted)
        normalized[legal_moves] = adjusted[legal_moves]
        total = float(np.sum(normalized[legal_moves]))
        if total > 0.0:
            normalized[legal_moves] /= total
        else:
            normalized[legal_moves] = 1.0 / len(legal_moves)
        return normalized.astype(np.float32)

    return override


def run_counterfactual(
    *,
    evaluator: ArtifactEvaluator,
    state: dict[str, Any],
    optimal_move: int,
    selected_move: int,
    budget: int,
    seed: int,
    root_prior_override,
    child_value_overrides: dict[int, float] | None,
) -> dict[str, Any]:
    root_game = KalahGame.from_state(state)
    search = DiagnosticPUCT(
        evaluator=evaluator,
        simulations=int(budget),
        c_puct=C_PUCT,
        rng=random.Random(int(seed)),
        fpu_mode=str(SEARCH_OPTIONS["fpu_mode"]),
        reuse_subtree=bool(SEARCH_OPTIONS["reuse_subtree"]),
        normalize_values=bool(SEARCH_OPTIONS["normalize_values"]),
        root_policy_mode=str(SEARCH_OPTIONS["root_policy_mode"]),
        tactical_root_bias=float(SEARCH_OPTIONS["tactical_root_bias"]),
        ablation_mode="full",
        root_prior_override=root_prior_override,
        child_value_overrides=child_value_overrides,
    )
    root = search._root_for(root_game)
    search._expand(
        root,
        apply_dirichlet=False,
        dirichlet_alpha=None,
        dirichlet_epsilon=0.0,
        is_root=True,
    )
    search.register_root_children(root)
    for _ in range(int(budget)):
        val = search._search(root)
        root.visit_count += 1
        root.value_sum += val
    search._last_root = root
    summary = search.root_summary()
    selection_map = {
        int(entry["move"]): entry
        for entry in list((summary.get("selection_breakdown") or {}).get("moves") or [])
        if isinstance(entry, dict) and entry.get("move") is not None
    }
    sel = summary.get("selected_move")
    chosen_entry = selection_map.get(int(sel), {}) if sel is not None else {}
    opt_entry = selection_map.get(optimal_move, {})
    total_visits = (
        sum(float(e.get("visit_count", 0)) for e in selection_map.values()) or 1.0
    )
    return {
        "selected_move": int(sel) if sel is not None else None,
        "selected_is_optimal": sel == optimal_move if sel is not None else None,
        "optimal_visit_share": round_float(
            float(opt_entry.get("visit_count", 0)) / total_visits
        )
        if opt_entry
        else None,
        "selected_visit_share": round_float(
            float(chosen_entry.get("visit_count", 0)) / total_visits
        )
        if sel is not None and chosen_entry
        else None,
        "optimal_q": round_float(float(opt_entry.get("q_value", 0.0))),
        "selected_q": round_float(float(chosen_entry.get("q_value", 0.0)))
        if sel is not None
        else None,
        "selected_minus_optimal_q_margin": round_float(
            float(chosen_entry.get("q_value", 0.0))
            - float(opt_entry.get("q_value", 0.0))
        )
        if sel is not None and chosen_entry and opt_entry
        else None,
        "flipped_to_optimal": sel == optimal_move if sel is not None else None,
    }


# ── Step 7: Root counterfactual diagnostics ────────────────────────────────


def run_counterfactual_diagnostics(
    evaluator: ArtifactEvaluator,
    valid_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run diagnostic-only interventions on rows that fail at 384 or 1200."""
    tb = EndgameTablebase()
    results: list[dict[str, Any]] = []

    for row in valid_rows:
        cid = str(row.get("candidate_id", ""))
        state = row["state"]
        root_player = KalahGame.from_state(state).current_player
        optimal_move = row.get("_tb_unique_optimal")
        legal_moves = row.get(
            "_legal_moves", KalahGame.from_state(state).possible_moves()
        )

        puct_results = row.get("_puct_results", [])
        fails_384 = not any(
            r["budget"] == 384 and r["selected_is_optimal"] for r in puct_results
        )
        fails_1200 = not any(
            r["budget"] == 1200 and r["selected_is_optimal"] for r in puct_results
        )
        if not fails_384 and not fails_1200:
            continue

        # Get selected non-optimal move
        r384 = next((r for r in puct_results if r["budget"] == 384), None)
        selected_move = r384["selected_move"] if r384 else None

        if (
            optimal_move is None
            or selected_move is None
            or selected_move == optimal_move
        ):
            continue

        sel_child_state = child_state_from_move(state, selected_move)
        opt_child_state = child_state_from_move(state, optimal_move)

        sel_c_wr = tb.lookup(KalahGame.from_state(sel_child_state), root_player)
        opt_c_wr = tb.lookup(KalahGame.from_state(opt_child_state), root_player)
        sel_child_exact = (
            (2.0 * float(sel_c_wr)) - 1.0 if sel_c_wr is not None else None
        )
        opt_child_exact = (
            (2.0 * float(opt_c_wr)) - 1.0 if opt_c_wr is not None else None
        )

        interventions: dict[str, dict[str, Any]] = {
            "original": {
                "root_prior_override": None,
                "child_value_overrides": None,
                "notes": "no intervention",
            },
            "uniform_legal_prior": {
                "root_prior_override": uniform_legal_prior_override,
                "child_value_overrides": None,
                "notes": "uniform legal prior",
            },
            "equalize_optimal_selected_priors": {
                "root_prior_override": make_equalize_optimal_selected_priors_override(
                    optimal_move, selected_move
                ),
                "child_value_overrides": None,
                "notes": "equalize optimal/selected priors",
            },
            "policy_prior_boost_optimal": {
                "root_prior_override": make_policy_prior_boost_optimal_override(
                    optimal_move, legal_moves
                ),
                "child_value_overrides": None,
                "notes": "diagnostic small boost to optimal move prior",
            },
        }

        if sel_child_exact is not None and opt_child_exact is not None:
            interventions["tablebase_child_value_override"] = {
                "root_prior_override": None,
                "child_value_overrides": {
                    optimal_move: root_to_state_perspective_value(
                        root_value=float(opt_child_exact),
                        state=opt_child_state,
                        root_player=root_player,
                    ),
                    selected_move: root_to_state_perspective_value(
                        root_value=float(sel_child_exact),
                        state=sel_child_state,
                        root_player=root_player,
                    ),
                },
                "notes": "override child backup values with exact tablebase child values",
            }

        for inter_name, config in interventions.items():
            for budget in COUNTERFACTUAL_BUDGETS:
                cf_result = run_counterfactual(
                    evaluator=evaluator,
                    state=state,
                    optimal_move=optimal_move,
                    selected_move=selected_move,
                    budget=int(budget),
                    seed=SEED,
                    root_prior_override=config["root_prior_override"],
                    child_value_overrides=config["child_value_overrides"],
                )
                results.append(
                    {
                        "candidate_id": cid,
                        "intervention": inter_name,
                        "budget": int(budget),
                        "selected_move": cf_result["selected_move"],
                        "selected_is_optimal": cf_result["selected_is_optimal"],
                        "optimal_visit_share": cf_result["optimal_visit_share"],
                        "selected_visit_share": cf_result["selected_visit_share"],
                        "optimal_q": cf_result["optimal_q"],
                        "selected_q": cf_result["selected_q"],
                        "selected_minus_optimal_q_margin": cf_result[
                            "selected_minus_optimal_q_margin"
                        ],
                        "flipped_to_optimal": cf_result["flipped_to_optimal"],
                        "notes": config["notes"],
                    }
                )

    return results


# ── Step 8: Mechanism classification ───────────────────────────────────────


def classify_row_mechanism(
    row: dict[str, Any],
    value_audit: dict[str, Any] | None,
    policy_audit: dict[str, Any] | None,
    counterfactual_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify exactly one causal mechanism per target candidate."""
    cid = str(row.get("candidate_id", ""))
    puct_results = row.get("_puct_results", [])

    fails_384 = not any(
        r["budget"] == 384 and r["selected_is_optimal"] for r in puct_results
    )
    fails_1200 = not any(
        r["budget"] == 1200 and r["selected_is_optimal"] for r in puct_results
    )
    passes_384 = any(
        r["budget"] == 384 and r["selected_is_optimal"] for r in puct_results
    )
    passes_1200 = any(
        r["budget"] == 1200 and r["selected_is_optimal"] for r in puct_results
    )

    cf_for_row = [r for r in counterfactual_results if r["candidate_id"] == cid]

    value_rank_error = False
    if value_audit:
        value_rank_error = value_audit.get("neural_value_rank_error", False)

    prior_underweight = False
    if policy_audit:
        prior_underweight = (
            policy_audit.get("prior_classification")
            == "policy_prior_underweights_optimal"
        )

    # Check counterfactual results
    tb_override_flips = any(
        r["intervention"] == "tablebase_child_value_override"
        and r["flipped_to_optimal"]
        for r in cf_for_row
    )
    uniform_prior_flips = any(
        r["intervention"] == "uniform_legal_prior" and r["flipped_to_optimal"]
        for r in cf_for_row
    )
    equalize_prior_flips = any(
        r["intervention"] == "equalize_optimal_selected_priors"
        and r["flipped_to_optimal"]
        for r in cf_for_row
    )
    prior_boost_flips = any(
        r["intervention"] == "policy_prior_boost_optimal" and r["flipped_to_optimal"]
        for r in cf_for_row
    )

    # Determine mechanism
    if not fails_384 and not fails_1200:
        if value_rank_error:
            mechanism = "value_error_but_search_compensates"
            evidence = (
                f"neural value rank error ({value_audit.get('value_error_class', '')}) "
                f"but PUCT selects optimal at 384+"
            )
        else:
            mechanism = "inconclusive"
            evidence = "row does not fail at 384 or 1200"
    elif not fails_384 and fails_1200:
        # Fails only at 1200
        if tb_override_flips:
            mechanism = "value_rank_error_drives_failure"
            evidence = (
                "value/backup error: tablebase child value override flips selection"
            )
        elif uniform_prior_flips or prior_boost_flips:
            mechanism = "policy_prior_underweight_drives_failure"
            evidence = "prior intervention flips selection"
        else:
            mechanism = "root_selection_pressure"
            evidence = (
                "exact child values support optimal but PUCT still selects non-optimal"
            )
    elif fails_384 and passes_1200:
        # Medium budget failure
        if tb_override_flips:
            mechanism = "value_rank_error_drives_failure"
            evidence = "value/backup error: tablebase override flips selection at 384"
        elif prior_underweight and (uniform_prior_flips or prior_boost_flips):
            mechanism = "policy_prior_underweight_drives_failure"
            evidence = "optimal move has low prior, prior interventions flip selection"
        else:
            mechanism = "root_selection_pressure"
            evidence = (
                "exact child values support optimal but PUCT selects non-optimal at 384"
            )
    elif fails_384 and not passes_1200:
        # Persistent failure
        if tb_override_flips:
            mechanism = "value_rank_error_drives_failure"
            evidence = (
                "value/backup error: tablebase child value override flips selection"
            )
        elif prior_underweight and (uniform_prior_flips or prior_boost_flips):
            mechanism = "policy_prior_underweight_drives_failure"
            evidence = "optimal move has low prior, prior interventions flip selection"
        elif uniform_prior_flips or equalize_prior_flips:
            mechanism = "root_selection_pressure"
            evidence = (
                "prior/selection pressure intervenable but not prior-underweight alone"
            )
        else:
            mechanism = "child_puct_value_error"
            evidence = "no counterfactual flips selection -> child PUCT or backup issue"
    else:
        mechanism = "low_budget_search_noise"
        evidence = "fails only below 384"

    # Refine: low-budget only
    passes_32 = any(
        r["budget"] == 32 and r["selected_is_optimal"] for r in puct_results
    )
    passes_64 = any(
        r["budget"] == 64 and r["selected_is_optimal"] for r in puct_results
    )
    if (not passes_32 or not passes_64) and passes_384 and passes_1200:
        mechanism = "low_budget_search_noise"
        evidence = "fails only below 384, passes at 384+"

    return {
        "candidate_id": cid,
        "row_mechanism": mechanism,
        "supporting_evidence": evidence,
        "recommended_role": "",
        "notes": "",
    }


def classify_all_mechanisms(
    valid_rows: list[dict[str, Any]],
    value_audit_results: list[dict[str, Any]],
    policy_audit_results: list[dict[str, Any]],
    counterfactual_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    classifications: list[dict[str, Any]] = []

    for row in valid_rows:
        cid = str(row.get("candidate_id", ""))
        value_audit = next(
            (r for r in value_audit_results if r["candidate_id"] == cid), None
        )
        policy_audit = next(
            (r for r in policy_audit_results if r["candidate_id"] == cid), None
        )

        cls = classify_row_mechanism(
            row, value_audit, policy_audit, counterfactual_results
        )
        classifications.append(cls)
        row["_mechanism"] = cls

    return classifications


# ── Step 9: Build refined clean split ──────────────────────────────────────


def build_clean_split(
    valid_rows: list[dict[str, Any]],
    exact_results: list[dict[str, Any]],
    mechanism_classifications: list[dict[str, Any]],
    policy_audit_results: list[dict[str, Any]],
) -> dict[str, Any]:
    split_rows: list[dict[str, Any]] = []

    prod_count = 0
    value_only_count = 0
    search_diag_count = 0
    control_count = 0
    holdout_count = 0
    exclude_count = 0

    for row in valid_rows:
        cid = str(row.get("candidate_id", ""))
        role = str(row.get("assigned_role", ""))
        exact = next((r for r in exact_results if r["candidate_id"] == cid), {})
        signal = exact.get("exact_signal_class", "")
        mech = next(
            (m for m in mechanism_classifications if m["candidate_id"] == cid), {}
        )
        mechanism = mech.get("row_mechanism", "inconclusive")

        puct_results = row.get("_puct_results", [])
        passes_384 = any(
            r["budget"] == 384 and r["selected_is_optimal"] for r in puct_results
        )
        passes_1200 = any(
            r["budget"] == 1200 and r["selected_is_optimal"] for r in puct_results
        )

        value_audit = next(
            (
                r
                for r in (row.get("_neural_value_audit"),)
                if r and r.get("candidate_id") == cid
            ),
            None,
        )
        value_rank_error = (
            bool(value_audit.get("neural_value_rank_error", False))
            if value_audit
            else False
        )
        sign_error = (
            bool(value_audit.get("neural_value_sign_error", False))
            if value_audit
            else False
        )

        # Exclude cases
        exclude = False
        bucket = ""

        if signal in (
            "tablebase_error",
            "tablebase_not_unique",
            "exact_unique_tiny_margin",
        ):
            exclude = True
            bucket = "exclude"
            reason = f"exact signal: {signal}"
        elif role != "target_candidate":
            if role == "preservation_control":
                bucket = "preservation_control"
                control_count += 1
            elif role == "holdout_candidate":
                bucket = "holdout_candidate"
                holdout_count += 1
            else:
                bucket = "exclude"
                exclude_count += 1
                reason = f"non-target role: {role}"
        elif mechanism in (
            "value_rank_error_drives_failure",
            "policy_prior_underweight_drives_failure",
            "root_selection_pressure",
            "child_puct_value_error",
        ):
            bucket = "production_candidate_later"
            prod_count += 1
            reason = mechanism
        elif value_rank_error or sign_error:
            bucket = "value_only_candidate"
            value_only_count += 1
            reason = "value rank/sign error but PUCT passes"
        elif mechanism == "low_budget_search_noise":
            bucket = "search_diagnostic_only"
            search_diag_count += 1
            reason = "low-budget-only failure"
        elif mechanism == "value_error_but_search_compensates":
            bucket = "value_only_candidate"
            value_only_count += 1
            reason = "value error but search compensates"
        else:
            bucket = "search_diagnostic_only"
            search_diag_count += 1
            reason = "inconclusive mechanism"

        split_rows.append(
            {
                "candidate_id": cid,
                "bucket": bucket,
                "mechanism": mechanism,
                "reason": reason if not exclude else reason,
                "exact_signal_class": signal,
                "value_rank_error": value_rank_error,
                "sign_error": sign_error,
                "passes_384": passes_384,
                "passes_1200": passes_1200,
                "notes": "",
            }
        )

    return {
        "split_rows": split_rows,
        "production_candidate_later": prod_count,
        "value_only_candidate": value_only_count,
        "search_diagnostic_only": search_diag_count,
        "preservation_control": control_count,
        "holdout_candidate": holdout_count,
        "exclude": exclude_count,
    }


# ── Step 10: Decision logic ───────────────────────────────────────────────


def make_diagnostic_decision(
    clean_split: dict[str, Any],
    mechanism_classifications: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
) -> tuple[str, str, str]:
    prod = clean_split["production_candidate_later"]
    value_only = clean_split["value_only_candidate"]
    search_only = clean_split["search_diagnostic_only"]
    total_targets = len(
        [r for r in valid_rows if str(r.get("assigned_role", "")) == "target_candidate"]
    )

    # Count mechanism distribution among targets
    mech_counts: dict[str, int] = {}
    for m in mechanism_classifications:
        mech = m["row_mechanism"]
        mech_counts[mech] = mech_counts.get(mech, 0) + 1

    dominant_mech = max(mech_counts, key=mech_counts.get) if mech_counts else "none"

    # Rule A: >=5 production candidates with coherent mechanism
    coherent_threshold = max(3, prod // 2)
    if prod >= 5 and mech_counts.get(dominant_mech, 0) >= coherent_threshold:
        decision = "exact_tablebase_diagnostic_artifact_ready"
        next_action = (
            "build a tiny train-only exact-tablebase diagnostic artifact "
            "with controls; no arena until local exact metrics improve"
        )
        dominant = dominant_mech
    # Rule B: Most are value-only
    elif value_only >= total_targets * 0.6 and value_only >= 5:
        decision = "exact_tablebase_value_only_ready"
        next_action = (
            "build a tiny value-only diagnostic artifact; do not use policy targets"
        )
        dominant = "value_error"
    # Rule C: Root selection pressure or prior-driven
    elif (
        mech_counts.get("root_selection_pressure", 0) >= 3
        or mech_counts.get("policy_prior_underweight_drives_failure", 0) >= 3
    ):
        decision = "exact_tablebase_search_calibration_ready"
        next_action = "run cpuct/prior calibration diagnostics before artifact training"
        dominant = "search_selection"
    # Rule D: Mostly low-budget noise
    elif search_only >= total_targets * 0.6 and search_only >= 5:
        decision = "exact_tablebase_low_budget_noise"
        next_action = "do not train; keep as search-budget diagnostics"
        dominant = "low_budget_noise"
    # Rule E: Mixed mechanisms
    elif prod >= 3:
        decision = "exact_tablebase_mechanism_split_required"
        next_action = (
            "split into value-rank, prior, and root-selection buckets; "
            "target the largest stable bucket"
        )
        dominant = "mixed"
    # Rule F: Too few
    else:
        decision = "exact_tablebase_too_small_after_diagnostics"
        next_action = "mine more hard exact-tablebase rows"
        dominant = "too_few_targets"

    return decision, next_action, dominant


# ── Report generation ──────────────────────────────────────────────────────


def generate_report(
    summary: dict[str, Any],
    output_path: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Harder Endgame Tablebase Local Diagnostics — Results")
    lines.append("")
    lines.append("**Date:** 2026-06-04")
    lines.append("**Family:** `harder_fresh_endgame_tablebase`")
    lines.append(
        "**Script:** `ml/alphazero_lite/run_harder_endgame_tablebase_local_diagnostics.py`"
    )
    lines.append("")

    # Section 1: Context
    lines.append("## 1. Context")
    lines.append("")
    dec = summary.get("decision_data", {})
    lines.append(
        "- Run classification: {}".format(dec.get("classification", "unknown"))
    )
    lines.append("- Selected family: harder_fresh_endgame_tablebase (from PR #73)")
    lines.append("- Current artifact: storage/ai/alphazero_lite/current")
    lines.append("- Active references not mutated")
    rep = summary.get("report_counts", {})
    lines.append(f"- Selected rows loaded: {rep.get('total_selected', 0)}")
    lines.append(f"- Valid rows: {rep.get('valid', 0)}")
    lines.append(f"- Invalid rows: {rep.get('invalid', 0)}")
    lines.append("- No training was run.")
    lines.append("- No arena was run.")
    lines.append("- No model was promoted.")
    lines.append("- Active references were not mutated.")

    # Section 2: Why PR #73 found a promising exact-teacher family
    lines.append("")
    lines.append("## 2. Why PR #73 found a promising exact-teacher family")
    lines.append("")
    lines.append(
        "PR #73 mined harder endgame positions with four strategies: "
        "deeper self-play, low-budget PUCT failure prescreening, adversarial "
        "near-threshold sampling, and PUCT-vs-tablebase disagreement sampling. "
        "It found 32 target candidates with exact-tablebase-unique optimal moves, "
        "7 persistent failures at 384+1200, 5 medium-budget failures, and 32 value "
        "rank errors. All rows are metadata candidates only — no replay artifacts "
        "were created."
    )
    lines.append("")
    lines.append(
        "This diagnostics run determines the causal mechanism for each failure: "
        "neural value rank error, PUCT selection pressure, child PUCT error, "
        "policy prior underweighting, backup/perspective issue, or low-budget noise."
    )

    # Section 3: Row validation
    lines.append("")
    lines.append("## 3. Row validation")
    lines.append("")
    vt = summary.get("validation_table", [])
    if vt:
        lines.append(
            "| candidate_id | assigned_role_from_pr73 | legal_moves | tablebase_available | "
            "unique_optimal | exhausted_overlap | teacher_conflict | valid | notes |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in vt:
            lines.append(
                f"| {r.get('candidate_id', '')} | {r.get('assigned_role_from_pr73', '')} | "
                f"{r.get('legal_moves', '')} | {format_bool(r.get('tablebase_available'))} | "
                f"{format_bool(r.get('unique_optimal'))} | {format_bool(r.get('exhausted_overlap'))} | "
                f"{format_bool(r.get('teacher_conflict'))} | {format_bool(r.get('valid'))} | "
                f"{r.get('notes', '')} |"
            )

    # Section 4: Exact tablebase re-enumeration
    lines.append("")
    lines.append("## 4. Exact tablebase re-enumeration")
    lines.append("")
    et = summary.get("exact_tablebase_table", [])
    if et:
        lines.append(
            "| candidate_id | root_value | optimal_move | second_best_move | "
            "best_minus_second_best | forced_win_or_loss | exact_signal_class | notes |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in et:
            lines.append(
                f"| {r.get('candidate_id', '')} | {format_float(r.get('root_value'))} | "
                f"{r.get('optimal_move', '')} | {r.get('second_best_move', '')} | "
                f"{format_float(r.get('best_minus_second_best'))} | "
                f"{format_bool(r.get('forced_win_or_loss'))} | {r.get('exact_signal_class', '')} | "
                f"{r.get('notes', '')} |"
            )

    # Section 5: Detailed PUCT budget sweep
    lines.append("")
    lines.append("## 5. Detailed PUCT budget sweep")
    lines.append("")
    pb = summary.get("puct_budget_summary", {})
    lines.append(
        f"- Rows with persistent failures (384+1200): {pb.get('persistent_failures_384_1200', 0)}"
    )
    lines.append(
        f"- Rows with medium-budget failures (384 only): {pb.get('medium_budget_failures', 0)}"
    )
    lines.append(
        f"- Rows with low-budget failures (32/64/128/256 only): {pb.get('low_budget_only_failures', 0)}"
    )
    lines.append(f"- Clean controls (all budgets pass): {pb.get('clean_controls', 0)}")
    lines.append("")
    pb_tbl = summary.get("puct_budget_table", [])
    if pb_tbl:
        lines.append(
            "| candidate_id | budget | optimal_move | selected_move | selected_is_optimal | "
            "optimal_visit_share | selected_visit_share | "
            "selected_minus_optimal_q_margin | optimal_policy_rank | "
            "first_budget_optimal_selected | notes |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in pb_tbl[:80]:
            lines.append(
                f"| {r.get('candidate_id', '')} | {r.get('budget', '')} | "
                f"{r.get('optimal_move', '')} | {r.get('selected_move', '')} | "
                f"{format_bool(r.get('selected_is_optimal'))} | "
                f"{format_float(r.get('optimal_visit_share'))} | "
                f"{format_float(r.get('selected_visit_share'))} | "
                f"{format_float(r.get('selected_minus_optimal_q_margin'))} | "
                f"{r.get('optimal_policy_rank', '')} | "
                f"{format_bool(r.get('first_budget_optimal_selected'))} | "
                f"{r.get('notes', '')} |"
            )
        if len(pb_tbl) > 80:
            lines.append(
                f"| ... and {len(pb_tbl) - 80} more rows | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |"
            )

    # Section 6: Root policy-prior audit
    lines.append("")
    lines.append("## 6. Root policy-prior audit")
    lines.append("")
    pp = summary.get("policy_prior_table", [])
    if pp:
        lines.append(
            "| candidate_id | optimal_move | selected_move | optimal_policy_probability | "
            "selected_policy_probability | optimal_policy_rank | "
            "optimal_prior_minus_selected_prior | prior_classification | notes |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in pp:
            lines.append(
                f"| {r.get('candidate_id', '')} | {r.get('optimal_move', '')} | "
                f"{r.get('selected_move', '')} | "
                f"{format_float(r.get('optimal_policy_probability'))} | "
                f"{format_float(r.get('selected_policy_probability'))} | "
                f"{r.get('optimal_policy_rank', '')} | "
                f"{format_float(r.get('optimal_prior_minus_selected_prior'))} | "
                f"{r.get('prior_classification', '')} | {r.get('notes', '')} |"
            )

    # Section 7: Neural value rank audit
    lines.append("")
    lines.append("## 7. Neural value rank audit")
    lines.append("")
    nv = summary.get("neural_value_table", [])
    if nv:
        lines.append(
            "| candidate_id | exact_optimal_move | neural_best_child_move | "
            "neural_best_is_exact_optimal | exact_optimal_child_neural_value | "
            "neural_best_child_exact_value | neural_value_rank_error | "
            "neural_value_sign_error | value_error_class | notes |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in nv:
            lines.append(
                f"| {r.get('candidate_id', '')} | {r.get('exact_optimal_move', '')} | "
                f"{r.get('neural_best_child_move', '')} | "
                f"{format_bool(r.get('neural_best_is_exact_optimal'))} | "
                f"{format_float(r.get('exact_optimal_child_neural_value'))} | "
                f"{format_float(r.get('neural_best_child_exact_value'))} | "
                f"{format_bool(r.get('neural_value_rank_error'))} | "
                f"{format_bool(r.get('neural_value_sign_error'))} | "
                f"{r.get('value_error_class', '')} | {r.get('notes', '')} |"
            )

    # Section 8: Child PUCT audit
    lines.append("")
    lines.append("## 8. Child PUCT audit")
    lines.append("")
    cp = summary.get("child_puct_table", [])
    if cp:
        lines.append(
            "| candidate_id | child_branch | budget | child_exact_value_root | "
            "child_puct_value_root | child_puct_error | "
            "child_selected_is_tablebase_optimal | classification | notes |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in cp:
            lines.append(
                f"| {r.get('candidate_id', '')} | {r.get('child_branch', '')} | "
                f"{r.get('budget', '')} | {format_float(r.get('child_exact_value_root'))} | "
                f"{format_float(r.get('child_puct_value_root'))} | "
                f"{format_float(r.get('child_puct_error'))} | "
                f"{format_bool(r.get('child_puct_agrees_with_exact'))} | "
                f"{r.get('classification', '')} | {r.get('notes', '')} |"
            )

    # Section 9: Root counterfactual diagnostics
    lines.append("")
    lines.append("## 9. Root counterfactual diagnostics")
    lines.append("")
    cf = summary.get("counterfactual_table", [])
    if cf:
        lines.append(
            "| candidate_id | intervention | budget | selected_move | selected_is_optimal | "
            "optimal_visit_share | selected_minus_optimal_q_margin | "
            "flipped_to_optimal | notes |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in cf:
            lines.append(
                f"| {r.get('candidate_id', '')} | {r.get('intervention', '')} | "
                f"{r.get('budget', '')} | {r.get('selected_move', '')} | "
                f"{format_bool(r.get('selected_is_optimal'))} | "
                f"{format_float(r.get('optimal_visit_share'))} | "
                f"{format_float(r.get('selected_minus_optimal_q_margin'))} | "
                f"{format_bool(r.get('flipped_to_optimal'))} | "
                f"{r.get('notes', '')} |"
            )

    # Section 10: Row mechanism classifications
    lines.append("")
    lines.append("## 10. Row mechanism classifications")
    lines.append("")
    mc = summary.get("mechanism_table", [])
    if mc:
        lines.append(
            "| candidate_id | row_mechanism | supporting_evidence | recommended_role | notes |"
        )
        lines.append("|---|---|---|---|---|")
        for r in mc:
            lines.append(
                f"| {r.get('candidate_id', '')} | {r.get('row_mechanism', '')} | "
                f"{r.get('supporting_evidence', '')} | "
                f"{r.get('recommended_role', '')} | {r.get('notes', '')} |"
            )

    # Section 11: Refined clean split
    lines.append("")
    lines.append("## 11. Refined clean split")
    lines.append("")
    cs = summary.get("clean_split", {})
    buckets = cs.get("split_rows", [])
    bucket_counts: dict[str, int] = {}
    for b in buckets:
        k = b.get("bucket", "exclude")
        bucket_counts[k] = bucket_counts.get(k, 0) + 1
    rows_in_buckets: dict[str, list[str]] = {}
    for b in buckets:
        k = b.get("bucket", "exclude")
        if k not in rows_in_buckets:
            rows_in_buckets[k] = []
        rows_in_buckets[k].append(b.get("candidate_id", ""))

    if bucket_counts:
        lines.append(
            "| bucket | row_count | rows | recommended_use | risks | next_action |"
        )
        lines.append("|---|---|---|---|---|---|")
        bucket_use = {
            "production_candidate_later": (
                "train policy+value with exact targets",
                "may need larger dataset for generalization",
                "build tiny diagnostic artifact with controls",
            ),
            "value_only_candidate": (
                "value calibration only",
                "not useful for policy targets",
                "include in value-only diagnostic artifact",
            ),
            "search_diagnostic_only": (
                "search-budget diagnostics",
                "do not train yet",
                "keep as budget sweep test cases",
            ),
            "preservation_control": (
                "control group for testing",
                "distribution must match targets",
                "use as clean controls in artifact",
            ),
            "holdout_candidate": (
                "unseen holdout for evaluation",
                "do not use in training",
                "set aside for final validation",
            ),
            "exclude": (
                "removed from consideration",
                "tablebase or validation issue",
                "do not use",
            ),
        }
        for bucket_name in sorted(bucket_counts.keys()):
            use = bucket_use.get(bucket_name, ("", "", ""))
            lines.append(
                f"| {bucket_name} | {bucket_counts[bucket_name]} | "
                f"{', '.join(rows_in_buckets[bucket_name][:5])}{'...' if len(rows_in_buckets[bucket_name]) > 5 else ''} | "
                f"{use[0]} | {use[1]} | {use[2]} |"
            )

    # Section 12: Final decision
    lines.append("")
    lines.append("## 12. Final decision")
    lines.append("")
    dd = summary.get("decision_data", {})
    lines.append(
        "| classification | supporting_evidence | rejected_alternatives | next_action |"
    )
    lines.append("|---|---|---|---|")
    classification = dd.get("classification", "")
    next_action = dd.get("next_action", "")
    dominant = dd.get("dominant_mechanism", "")
    counts = dd.get("counts", {})
    evidence = (
        f"production_candidate_later={counts.get('production_candidate_later', 0)}, "
        f"value_only={counts.get('value_only_candidate', 0)}, "
        f"search_only={counts.get('search_diagnostic_only', 0)}, "
        f"persistent_failures={counts.get('persistent_failures_384_1200', 0)}, "
        f"value_rank_errors={counts.get('value_rank_errors', 0)}, "
        f"dominant_mechanism={dominant}"
    )
    lines.append(
        f"| {classification} | {evidence} | see mechanism distribution | {next_action} |"
    )
    lines.append("")

    # Section 13: Recommended next action
    lines.append("## 13. Exactly one recommended next action")
    lines.append("")
    lines.append(f"Recommendation: **{next_action}**")
    lines.append("")
    lines.append("### Acceptance criteria")
    lines.append("")
    lines.append("- No training was run.")
    lines.append("- No arena was run.")
    lines.append("- No model was promoted.")
    lines.append("- Active references were not mutated.")
    lines.append("- Exhausted families were excluded from selection.")
    lines.append("- Teacher-conflict filtering was used.")
    lines.append("- Exact tablebase labels were used only diagnostically.")
    lines.append("- Final report recommends exactly one next branch.")

    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load fixture references for dedup
    print("\nLoading fixture inventory...")
    suite_rows = load_suite(SUITE_PATH)
    ref_by_id, _ = load_reference_maps(REFERENCE_PATH)
    suite_canon = suite_canonical_states(suite_rows)

    # Load artifact evaluator
    print("Loading artifact evaluator...")
    evaluator = ArtifactEvaluator(CURRENT_ARTIFACT)

    # Load selected rows
    selected_rows = load_jsonl(SELECTED_ROWS_PATH)
    print(f"Loaded {len(selected_rows)} selected rows from {SELECTED_ROWS_PATH}")

    target_count = sum(
        1 for r in selected_rows if r.get("assigned_role") == "target_candidate"
    )
    print(f"  Target candidates: {target_count}")
    print("  Expected target candidates: 32")

    # ── Step 1: Validate ──────────────────────────────────────────────────
    print("\n=== Step 1: Validation ===")
    valid_rows, validation_entries = validate_selected_rows(
        selected_rows, suite_rows, ref_by_id, suite_canon
    )
    valid_count = len(valid_rows)
    invalid_count = len(selected_rows) - valid_count
    print(f"  Valid: {valid_count}, Invalid: {invalid_count}")

    # Log invalid rows
    for ve in validation_entries:
        if not ve["valid"]:
            print(f"  INVALID: {ve['candidate_id']}: {ve['notes']}")

    # Ensure _tb_unique_optimal is set for all valid rows
    for row in valid_rows:
        if "_tb_unique_optimal" not in row:
            tb_data = compute_tablebase_child_values(row["state"])
            optimal = tb_data["optimal_moves"]
            if len(optimal) == 1:
                row["_tb_unique_optimal"] = optimal[0]
                row["_tb_child_values"] = tb_data["child_values"]
                row["_tb_optimal_moves"] = optimal

    # ── Step 2: Exact tablebase re-enumeration ───────────────────────────
    print("\n=== Step 2: Exact Tablebase Enumeration ===")
    exact_results = exact_tablebase_enumeration(valid_rows)
    for r in exact_results:
        print(
            f"  {r['candidate_id']}: {r['exact_signal_class']} "
            f"optimal={r['optimal_move']} "
            f"best-second={format_float(r['best_minus_second_best'])}"
        )

    # ── Step 3: Detailed PUCT budget sweep ────────────────────────────────
    print("\n=== Step 3: Detailed PUCT Budget Sweep ===")
    puct_budget_results = run_puct_budget_sweep(evaluator, valid_rows)
    print(f"  Total PUCT evaluations: {len(puct_budget_results)}")

    persistent_fails = sum(
        1 for r in valid_rows if r.get("_persistent_failure_384_1200")
    )
    medium_fails = sum(1 for r in valid_rows if r.get("_medium_budget_failure"))
    low_budget_fails = sum(1 for r in valid_rows if r.get("_low_budget_only_failure"))
    clean_controls = (
        len(valid_rows) - persistent_fails - medium_fails - low_budget_fails
    )
    print(f"  Persistent failures (384+1200): {persistent_fails}")
    print(f"  Medium-budget failures (384 only): {medium_fails}")
    print(f"  Low-budget failures (32/64/128/256 only): {low_budget_fails}")
    print(f"  Clean controls (all pass): {clean_controls}")

    # Build PUCT budget table
    puct_budget_table: list[dict[str, Any]] = []
    for r in puct_budget_results:
        puct_budget_table.append(
            {
                "candidate_id": r.get("candidate_id"),
                "budget": r.get("budget"),
                "optimal_move": r.get("optimal_move"),
                "selected_move": r.get("selected_move"),
                "selected_is_optimal": r.get("selected_is_optimal"),
                "optimal_visit_share": r.get("optimal_visit_share"),
                "selected_visit_share": r.get("selected_visit_share"),
                "selected_minus_optimal_q_margin": r.get(
                    "selected_minus_optimal_q_margin"
                ),
                "optimal_policy_rank": r.get("optimal_policy_rank"),
                "first_budget_optimal_selected": r.get("first_budget_optimal_selected"),
                "notes": "deterministic PUCT baseline",
            }
        )

    # ── Step 4: Root policy-prior audit ─────────────────────────────────
    print("\n=== Step 4: Root Policy-Prior Audit ===")
    policy_prior_results = root_policy_prior_audit(valid_rows)
    underweight_count = sum(
        1
        for r in policy_prior_results
        if r.get("prior_classification") == "policy_prior_underweights_optimal"
    )
    print(
        f"  Prior underweights optimal: {underweight_count} / {len(policy_prior_results)}"
    )
    for r in policy_prior_results:
        print(
            f"  {r['candidate_id']}: {r['prior_classification']} "
            f"opt_prior={format_float(r['optimal_policy_probability'])} "
            f"opt_rank={r['optimal_policy_rank']}"
        )

    # ── Step 5: Neural value rank audit ──────────────────────────────────
    print("\n=== Step 5: Neural Value Rank Audit ===")
    value_audit_results = neural_value_rank_audit(evaluator, valid_rows)
    value_rank_errors = sum(
        1 for r in value_audit_results if r.get("neural_value_rank_error")
    )
    sign_errors = sum(
        1 for r in value_audit_results if r.get("neural_value_sign_error")
    )
    print(f"  Value rank errors: {value_rank_errors}")
    print(f"  Sign errors: {sign_errors}")
    for r in value_audit_results:
        print(
            f"  {r['candidate_id']}: {r['value_error_class']} "
            f"rank_err={r['neural_value_rank_error']} "
            f"sign_err={r['neural_value_sign_error']}"
        )

    # ── Step 6: Child PUCT audit ────────────────────────────────────────
    print("\n=== Step 6: Child PUCT Audit ===")
    child_puct_results = child_puct_audit(evaluator, valid_rows)
    for r in child_puct_results:
        print(
            f"  {r['candidate_id']} {r['child_branch']}@{r['budget']}: "
            f"{r['classification']} err={format_float(r['child_puct_error'])}"
        )

    # ── Step 7: Root counterfactual diagnostics ─────────────────────────
    print("\n=== Step 7: Root Counterfactual Diagnostics ===")
    counterfactual_results = run_counterfactual_diagnostics(evaluator, valid_rows)
    for r in counterfactual_results:
        flip_str = "FLIP" if r.get("flipped_to_optimal") else "no flip"
        print(
            f"  {r['candidate_id']} {r['intervention']}@{r['budget']}: "
            f"sel={r['selected_move']} optimal={r['selected_is_optimal']} "
            f"{flip_str}"
        )

    # ── Step 8: Mechanism classification ────────────────────────────────
    print("\n=== Step 8: Mechanism Classification ===")
    mechanism_classifications = classify_all_mechanisms(
        valid_rows,
        value_audit_results,
        policy_prior_results,
        counterfactual_results,
    )
    mech_counts: dict[str, int] = {}
    for m in mechanism_classifications:
        mech = m["row_mechanism"]
        mech_counts[mech] = mech_counts.get(mech, 0) + 1
    print(f"  Mechanism distribution: {mech_counts}")
    for m in mechanism_classifications:
        print(
            f"  {m['candidate_id']}: {m['row_mechanism']} — {m['supporting_evidence']}"
        )

    # ── Step 9: Build refined clean split ───────────────────────────────
    print("\n=== Step 9: Refined Clean Split ===")
    clean_split = build_clean_split(
        valid_rows,
        exact_results,
        mechanism_classifications,
        policy_prior_results,
    )
    print(
        f"  Production: {clean_split['production_candidate_later']}, "
        f"Value-only: {clean_split['value_only_candidate']}, "
        f"Search-diagnostic: {clean_split['search_diagnostic_only']}, "
        f"Control: {clean_split['preservation_control']}, "
        f"Holdout: {clean_split['holdout_candidate']}, "
        f"Exclude: {clean_split['exclude']}"
    )

    # ── Step 10: Decision ───────────────────────────────────────────────
    print("\n=== Step 10: Decision ===")
    decision, next_action, dominant_mechanism = make_diagnostic_decision(
        clean_split, mechanism_classifications, valid_rows
    )
    print(f"  Classification: {decision}")
    print(f"  Next action: {next_action}")
    print(f"  Dominant mechanism: {dominant_mechanism}")

    decision_data = {
        "classification": decision,
        "next_action": next_action,
        "dominant_mechanism": dominant_mechanism,
        "counts": {
            "production_candidate_later": clean_split["production_candidate_later"],
            "value_only_candidate": clean_split["value_only_candidate"],
            "search_diagnostic_only": clean_split["search_diagnostic_only"],
            "preservation_control": clean_split["preservation_control"],
            "holdout_candidate": clean_split["holdout_candidate"],
            "exclude": clean_split["exclude"],
            "persistent_failures_384_1200": persistent_fails,
            "medium_budget_failures": medium_fails,
            "low_budget_only_failures": low_budget_fails,
            "value_rank_errors": value_rank_errors,
            "value_sign_errors": sign_errors,
            "valid_rows": valid_count,
            "invalid_rows": invalid_count,
            "total_selected": len(selected_rows),
        },
    }

    # ── Write summary JSON ──────────────────────────────────────────────
    summary = {
        "schema": "azlite_harder_endgame_tablebase_local_diagnostics_v1",
        "family": FAMILY,
        "description": (
            "Tablebase-backed local search/value diagnostics for "
            "harder_fresh_endgame_tablebase. Determines causal mechanism "
            "for exact-tablebase hard endgame failures."
        ),
        "guardrails": {
            "mutated_active_fixture": False,
            "ran_training": False,
            "ran_arena": False,
            "promoted_model": False,
            "created_replay_artifacts": False,
        },
        "inputs": {
            "selected_rows_path": str(SELECTED_ROWS_PATH),
            "current_artifact": str(CURRENT_ARTIFACT),
        },
        "report_counts": {
            "total_selected": len(selected_rows),
            "valid": valid_count,
            "invalid": invalid_count,
        },
        "validation_table": validation_entries,
        "exact_tablebase_table": exact_results,
        "puct_budget_summary": {
            "total_rows": len(valid_rows),
            "persistent_failures_384_1200": persistent_fails,
            "medium_budget_failures": medium_fails,
            "low_budget_only_failures": low_budget_fails,
            "clean_controls": clean_controls,
        },
        "puct_budget_table": puct_budget_table,
        "policy_prior_table": policy_prior_results,
        "neural_value_table": value_audit_results,
        "child_puct_table": child_puct_results,
        "counterfactual_table": counterfactual_results,
        "mechanism_table": mechanism_classifications,
        "mechanism_counts": mech_counts,
        "clean_split": clean_split,
        "decision": decision,
        "decision_data": decision_data,
    }

    write_json(
        OUTPUT_DIR / "harder_endgame_tablebase_local_diagnostics_summary.json",
        summary,
    )
    print(
        f"\nWritten summary: {OUTPUT_DIR / 'harder_endgame_tablebase_local_diagnostics_summary.json'}"
    )

    # ── Write row diagnostics JSONL ─────────────────────────────────────
    row_diagnostics: list[dict[str, Any]] = []
    for row in valid_rows:
        cid = str(row.get("candidate_id", ""))
        exact = next((r for r in exact_results if r["candidate_id"] == cid), {})
        val_audit = next(
            (r for r in value_audit_results if r["candidate_id"] == cid), {}
        )
        pol_audit = next(
            (r for r in policy_prior_results if r["candidate_id"] == cid), {}
        )
        baselines = [r for r in puct_budget_results if r.get("candidate_id") == cid]
        child_audits = [r for r in child_puct_results if r["candidate_id"] == cid]
        cfs = [r for r in counterfactual_results if r["candidate_id"] == cid]
        mech = next(
            (m for m in mechanism_classifications if m["candidate_id"] == cid), {}
        )
        split_info = next(
            (s for s in clean_split["split_rows"] if s["candidate_id"] == cid), {}
        )
        valid_info = next(
            (v for v in validation_entries if v["candidate_id"] == cid), {}
        )

        row_diagnostics.append(
            {
                "candidate_id": cid,
                "provisional_family": FAMILY,
                "canonical_state_hash": str(row.get("canonical_state_hash", "")),
                "state": row.get("state"),
                "legal_moves": row.get("_legal_moves", []),
                "validation": valid_info,
                "exact_tablebase": exact,
                "puct_baseline": baselines,
                "policy_prior_audit": pol_audit,
                "neural_value_audit": val_audit,
                "child_puct_audit": child_audits,
                "counterfactual": cfs,
                "mechanism": mech,
                "assigned_role": split_info.get("bucket", "unassigned"),
                "pr73_role": str(row.get("assigned_role", "")),
            }
        )

    # Also include invalid rows
    for row, ve in zip(selected_rows, validation_entries):
        if not ve["valid"]:
            row_diagnostics.append(
                {
                    "candidate_id": str(row.get("candidate_id", "")),
                    "provisional_family": str(row.get("provisional_family", "")),
                    "canonical_state_hash": str(row.get("canonical_state_hash", "")),
                    "state": row.get("state"),
                    "legal_moves": KalahGame.from_state(row["state"]).possible_moves()
                    if row.get("state")
                    else [],
                    "validation": ve,
                    "exact_tablebase": {},
                    "puct_baseline": [],
                    "policy_prior_audit": {},
                    "neural_value_audit": {},
                    "child_puct_audit": [],
                    "counterfactual": [],
                    "mechanism": {},
                    "assigned_role": "excluded",
                }
            )

    write_jsonl(
        OUTPUT_DIR / "harder_endgame_tablebase_local_row_diagnostics.jsonl",
        row_diagnostics,
    )
    print(
        f"Written row diagnostics: {OUTPUT_DIR / 'harder_endgame_tablebase_local_row_diagnostics.jsonl'}"
    )

    # ── Write clean split JSON ──────────────────────────────────────────
    clean_split_json = {
        "schema": "azlite_harder_endgame_tablebase_clean_split_v1",
        "family": FAMILY,
        "description": "Refined clean split after local diagnostics.",
        "guardrails": {
            "mutated_active_fixture": False,
            "ran_training": False,
            "ran_arena": False,
            "promoted_model": False,
            "created_replay_artifacts": False,
        },
        "split_rows": clean_split["split_rows"],
        "counts": {
            "production_candidate_later": clean_split["production_candidate_later"],
            "value_only_candidate": clean_split["value_only_candidate"],
            "search_diagnostic_only": clean_split["search_diagnostic_only"],
            "preservation_control": clean_split["preservation_control"],
            "holdout_candidate": clean_split["holdout_candidate"],
            "exclude": clean_split["exclude"],
        },
        "mechanism_counts": mech_counts,
        "decision": {
            "classification": decision,
            "next_action": next_action,
            "dominant_mechanism": dominant_mechanism,
        },
    }
    write_json(
        OUTPUT_DIR / "harder_endgame_tablebase_clean_split.json",
        clean_split_json,
    )
    print(
        f"Written clean split: {OUTPUT_DIR / 'harder_endgame_tablebase_clean_split.json'}"
    )

    # ── Generate report markdown ────────────────────────────────────────
    report_path = Path(
        "docs/alphazero-lite-harder-endgame-tablebase-local-diagnostics-results.md"
    )
    generate_report(summary, report_path)
    print(f"\nReport written to {report_path}")

    # ── Console summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("HARDER ENDGAME TABLEBASE LOCAL DIAGNOSTICS")
    print(f"{'=' * 90}")
    print(f"\n{'Validation':30s} {valid_count} valid / {invalid_count} invalid")
    print(f"{'Persistent failures (384+1200)':30s} {persistent_fails}")
    print(f"{'Medium-budget failures (384 only)':30s} {medium_fails}")
    print(f"{'Low-budget only failures':30s} {low_budget_fails}")
    print(f"{'Clean controls':30s} {clean_controls}")
    print(f"\n{'Value rank errors':30s} {value_rank_errors}")
    print(f"{'Sign errors':30s} {sign_errors}")
    print(f"{'Prior underweights optimal':30s} {underweight_count}")
    print(f"\n{'Mechanisms':30s}")
    for mech, cnt in sorted(mech_counts.items()):
        print(f"  {mech:40s} {cnt}")
    print(f"\n{'Clean split':30s}")
    print(
        f"  {'Production candidates':30s} {clean_split['production_candidate_later']}"
    )
    print(f"  {'Value-only candidates':30s} {clean_split['value_only_candidate']}")
    print(f"  {'Search diagnostic':30s} {clean_split['search_diagnostic_only']}")
    print(f"  {'Preservation controls':30s} {clean_split['preservation_control']}")
    print(f"  {'Holdouts':30s} {clean_split['holdout_candidate']}")
    print(f"  {'Excluded':30s} {clean_split['exclude']}")
    print(f"\n{'Classification':30s} {decision}")
    print(f"{'Dominant mechanism':30s} {dominant_mechanism}")
    print(f"{'Next action':30s} {next_action}")
    print(f"\n{'Outputs':30s}")
    print(
        f"  Summary: {OUTPUT_DIR}/harder_endgame_tablebase_local_diagnostics_summary.json"
    )
    print(
        f"  Row diagnostics: {OUTPUT_DIR}/harder_endgame_tablebase_local_row_diagnostics.jsonl"
    )
    print(f"  Clean split: {OUTPUT_DIR}/harder_endgame_tablebase_clean_split.json")
    print(f"  Report: {report_path}")
    print(f"{'=' * 90}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
