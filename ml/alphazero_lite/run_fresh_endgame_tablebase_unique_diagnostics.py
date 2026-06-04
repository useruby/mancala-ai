#!/usr/bin/env python3
"""Tablebase-backed local search/value diagnostics for fresh_endgame_tablebase_unique.

Steps:
  1. Load and validate selected fresh rows.
  2. Exact tablebase enumeration for every valid row.
  3. Current PUCT baseline at 128/384/1200/2400/5000.
  4. Neural value audit.
  5. Child PUCT audit for failing rows.
  6. Root counterfactual diagnostics for failing rows.
  7. Build clean target/control/holdout split.
  8. Decide whether a later diagnostic artifact is warranted.
  9. Write report.

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
FAMILY = "fresh_endgame_tablebase_unique"
C_PUCT = 1.25
SEARCH_OPTIONS = build_eval_search_options(
    fpu_mode="parent_q",
    reuse_subtree=True,
    normalize_values=True,
    root_policy_mode="deterministic",
    tactical_root_bias=0.1,
)
ROOT_BUDGETS = (128, 384, 1200, 2400, 5000)
CHILD_PUCT_BUDGETS = (384, 1200)
COUNTERFACTUAL_BUDGETS = (384, 1200)
SEED = 17

CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
REFERENCE_PATH = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
SELECTED_ROWS_PATH = Path(
    "/tmp/azlite_fresh_hard_state_mining_teacher_filtered/"
    "selected_fresh_family_rows.jsonl"
)
OUTPUT_DIR = Path("/tmp/azlite_fresh_endgame_tablebase_unique_diagnostics")

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


def selected_move_from_policy(
    policy: list[float], legal_moves: list[int]
) -> int | None:
    if not legal_moves:
        return None
    return max(legal_moves, key=lambda move: (float(policy[move]), -int(move)))


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


# ── Step 1: Load and validate ──────────────────────────────────────────────


def validate_selected_rows(
    selected_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    validation: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for row in selected_rows:
        cid = str(row.get("candidate_id", ""))
        family = str(row.get("provisional_family", ""))
        c_hash = str(row.get("canonical_state_hash", ""))
        state = row.get("state")
        do_not_train = bool(row.get("do_not_train_yet", False))

        notes: list[str] = []
        valid = True

        if family != FAMILY:
            notes.append(f"family mismatch: expected {FAMILY}, got {family}")
            valid = False

        if not do_not_train:
            notes.append("do_not_train_yet is false/absent")
            valid = False

        if not state:
            notes.append("missing state")
            valid = False
        else:
            game = KalahGame.from_state(state)
            legal = game.possible_moves()
            if not legal:
                notes.append("no legal moves")
                valid = False
            else:
                row["_legal_moves"] = legal

        if c_hash in seen_hashes:
            notes.append(f"duplicate canonical hash: {c_hash}")
            valid = False
        seen_hashes.add(c_hash)

        if is_exhausted_row_id(cid):
            notes.append("exhausted fixture row id prefix")
            valid = False

        legal: list[int] = []
        if state:
            game = KalahGame.from_state(state)
            legal = game.possible_moves()
            if not legal:
                notes.append("no legal moves")
                valid = False
            else:
                row["_legal_moves"] = legal
            tb_local = EndgameTablebase()
            offset = game.current_player * 6
            child_values: dict[int, float] = {}
            optimal: list[int] = []
            best_val = -float("inf")
            for move in legal:
                child_game = game.clone()
                child_game.move(offset + move)
                c_wr = tb_local.lookup(child_game, game.current_player)
                if c_wr is not None:
                    cv = (2.0 * float(c_wr)) - 1.0
                    child_values[move] = round_float(cv)
                    if cv > best_val:
                        best_val = cv
            if best_val > -float("inf"):
                optimal = sorted(
                    m for m, v in child_values.items() if abs(v - best_val) < EPS
                )
            row["_tb_child_values"] = child_values
            row["_tb_optimal_moves"] = optimal

        validation.append(
            {
                "candidate_id": cid,
                "valid": valid,
                "notes": "; ".join(notes) if notes else "ok",
            }
        )

    return validation


# ── Step 2: Exact tablebase enumeration ──────────────────────────────────────


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
                    "tablebase_optimal_move": None,
                    "second_best_value": None,
                    "best_minus_second_best": None,
                    "all_moves_equivalent": False,
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
                child_values[move] = None

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
        best_minus_second = None
        if len(sorted_vals) >= 2:
            best_minus_second = round_float(sorted_vals[0] - sorted_vals[1])

        all_equiv = (
            len(set(round_float(v, 6) for v in child_values.values() if v is not None))
            == 1
            if child_values
            else False
        )

        if not unique_optimal:
            signal = "tablebase_not_unique"
        elif all_equiv:
            signal = "exact_unique_but_forced_all_bad"
        elif best_minus_second is not None and best_minus_second < EPS:
            signal = "exact_unique_tiny_margin"
        else:
            signal = "exact_unique_clear_margin"

        notes_parts = []
        if signal == "exact_unique_tiny_margin":
            notes_parts.append(f"tiny margin = {best_minus_second}")
        if signal == "exact_unique_but_forced_all_bad":
            notes_parts.append("all moves equivalent despite unique optimal")
        if signal == "tablebase_not_unique":
            notes_parts.append(f"optimal moves: {optimal_moves}")

        results.append(
            {
                "candidate_id": cid,
                "root_value": round_float(root_value),
                "tablebase_optimal_move": optimal_move,
                "optimal_moves": optimal_moves,
                "second_best_value": round_float(
                    sorted_vals[1] if len(sorted_vals) >= 2 else None
                ),
                "best_minus_second_best": best_minus_second,
                "all_moves_equivalent": all_equiv,
                "exact_signal_class": signal,
                "notes": "; ".join(notes_parts) if notes_parts else "ok",
            }
        )

    return results


# ── Step 3: Current PUCT baseline ────────────────────────────────────────────


def root_puct_run(
    evaluator: ArtifactEvaluator,
    row: dict[str, Any],
    budget: int,
    seed: int,
) -> dict[str, Any]:
    result = evaluate_artifact_position(
        artifact_path=None,
        evaluator=evaluator,
        state=row["state"],
        simulations=int(budget),
        seed=int(seed),
        c_puct=C_PUCT,
        search_options=dict(SEARCH_OPTIONS),
        ablation_mode="full",
    )
    selected_move = (
        None if result.get("selected_move") is None else int(result["selected_move"])
    )
    selection_map = selection_entry_map(result)
    optimal_move = (
        row.get("_tb_optimal_moves", [None])[0]
        if row.get("_tb_optimal_moves")
        else None
    )
    opt_entry = selection_map.get(optimal_move) if optimal_move is not None else {}
    sel_entry = selection_map.get(selected_move) if selected_move is not None else {}
    selected_is_optimal = (
        selected_move == optimal_move if optimal_move is not None else None
    )
    visits_list = [float(v) for v in result.get("visits", [])]

    return {
        "row_id": str(row["candidate_id"]),
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
            float(sel_entry.get("q_value", 0.0)) - float(opt_entry.get("q_value", 0.0))
        )
        if sel_entry and opt_entry
        else None,
        "optimal_policy_probability": round_float(float(opt_entry.get("prior", 0.0)))
        if opt_entry
        else None,
        "selected_policy_probability": round_float(float(sel_entry.get("prior", 0.0)))
        if sel_entry
        else None,
        "optimal_policy_rank": None,
        "pass_status": "pass" if selected_is_optimal else "fail",
        "notes": "deterministic PUCT baseline",
    }


def run_baseline(
    evaluator: ArtifactEvaluator,
    valid_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    all_results: list[dict[str, Any]] = []
    for row in valid_rows:
        for budget in ROOT_BUDGETS:
            r = root_puct_run(evaluator, row, budget, SEED)
            all_results.append(r)

    for row in valid_rows:
        cid = str(row["candidate_id"])
        budget_results = [r for r in all_results if r["row_id"] == cid]
        passes_128 = any(
            r["pass_status"] == "pass" and r["budget"] == 128 for r in budget_results
        )
        passes_384 = any(
            r["pass_status"] == "pass" and r["budget"] == 384 for r in budget_results
        )
        passes_1200 = any(
            r["pass_status"] == "pass" and r["budget"] == 1200 for r in budget_results
        )
        passes_2400 = any(
            r["pass_status"] == "pass" and r["budget"] == 2400 for r in budget_results
        )
        passes_5000 = any(
            r["pass_status"] == "pass" and r["budget"] == 5000 for r in budget_results
        )

        if passes_2400 or passes_5000:
            row["_root_puct_classification"] = "passes_at_high_budget"
        elif passes_1200:
            row["_root_puct_classification"] = "passes_at_1200"
        elif passes_384:
            row["_root_puct_classification"] = "passes_at_384"
        elif passes_128:
            row["_root_puct_classification"] = "low_budget_only_failure"
        else:
            row["_root_puct_classification"] = "persistent_failure"

        row["_puct_pass_128"] = passes_128
        row["_puct_pass_384"] = passes_384
        row["_puct_pass_1200"] = passes_1200
        row["_puct_pass_2400"] = passes_2400
        row["_puct_pass_5000"] = passes_5000

    return all_results


# ── Step 4: Neural value audit ──────────────────────────────────────────────


def neural_value_audit(
    evaluator: ArtifactEvaluator,
    valid_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tb = EndgameTablebase()
    results: list[dict[str, Any]] = []

    for row in valid_rows:
        cid = str(row["candidate_id"])
        state = row["state"]
        root_player = KalahGame.from_state(state).current_player

        tb_wr = tb.lookup(KalahGame.from_state(state), root_player)
        exact_root_value = (2.0 * float(tb_wr)) - 1.0 if tb_wr is not None else None

        game = KalahGame.from_state(state)
        policy, raw_value = evaluator.evaluate(game)
        neural_root_value = float(raw_value)
        root_value_error = (
            round_float(neural_root_value - exact_root_value)
            if exact_root_value is not None
            else None
        )

        optimal_move = (
            row.get("_tb_optimal_moves", [None])[0]
            if row.get("_tb_optimal_moves")
            else None
        )
        puct_results_for_row = [
            r for r in row.get("_baseline_results", []) if r["row_id"] == cid
        ]
        budget_1200 = next(
            (r for r in puct_results_for_row if r["budget"] == 1200),
            puct_results_for_row[-1] if puct_results_for_row else {},
        )
        selected_move = budget_1200.get("selected_move")

        child_optimal_exact = None
        child_selected_exact = None
        child_optimal_neural = None
        child_selected_neural = None

        if optimal_move is not None:
            child_state_opt = child_state_from_move(state, optimal_move)
            c_wr = tb.lookup(KalahGame.from_state(child_state_opt), root_player)
            child_optimal_exact = (
                round_float((2.0 * float(c_wr)) - 1.0) if c_wr is not None else None
            )
            _, cnv = evaluator.evaluate(KalahGame.from_state(child_state_opt))
            child_optimal_neural = round_float(
                state_to_root_perspective_value(
                    raw_value=float(cnv),
                    state=child_state_opt,
                    root_player=root_player,
                )
            )

        if (
            selected_move is not None
            and optimal_move is not None
            and selected_move != optimal_move
        ):
            child_state_sel = child_state_from_move(state, selected_move)
            c_wr = tb.lookup(KalahGame.from_state(child_state_sel), root_player)
            child_selected_exact = (
                round_float((2.0 * float(c_wr)) - 1.0) if c_wr is not None else None
            )
            _, cnv = evaluator.evaluate(KalahGame.from_state(child_state_sel))
            child_selected_neural = round_float(
                state_to_root_perspective_value(
                    raw_value=float(cnv),
                    state=child_state_sel,
                    root_player=root_player,
                )
            )
        elif (
            selected_move is not None
            and optimal_move is not None
            and selected_move == optimal_move
        ):
            child_selected_exact = child_optimal_exact
            child_selected_neural = child_optimal_neural

        neural_child_optimal_minus_selected = None
        if child_optimal_neural is not None and child_selected_neural is not None:
            neural_child_optimal_minus_selected = round_float(
                child_optimal_neural - child_selected_neural
            )

        exact_child_optimal_minus_selected = None
        if child_optimal_exact is not None and child_selected_exact is not None:
            exact_child_optimal_minus_selected = round_float(
                child_optimal_exact - child_selected_exact
            )

        value_rank_error = False
        sign_error = False
        if (
            child_optimal_neural is not None
            and child_selected_neural is not None
            and child_optimal_exact is not None
            and child_selected_exact is not None
        ):
            exact_rank_correct = child_optimal_exact >= child_selected_exact
            neural_rank_correct = child_optimal_neural >= child_selected_neural
            value_rank_error = exact_rank_correct and not neural_rank_correct

            if exact_root_value is not None:
                exact_sign = math.copysign(1.0, exact_root_value)
                neural_sign = math.copysign(1.0, neural_root_value)
                sign_error = exact_sign != neural_sign

        if value_rank_error and sign_error:
            classification = "value_head_prefers_wrong_child"
        elif (
            child_optimal_neural is not None
            and child_selected_neural is not None
            and child_optimal_exact is not None
            and child_selected_exact is not None
            and neural_child_optimal_minus_selected is not None
            and exact_child_optimal_minus_selected is not None
            and neural_child_optimal_minus_selected
            < exact_child_optimal_minus_selected * 0.5
        ):
            classification = "value_head_underestimates_optimal_child"
        elif value_rank_error and not sign_error:
            classification = "value_head_prefers_wrong_child"
        elif (
            child_optimal_neural is not None
            and child_selected_neural is not None
            and child_optimal_exact is not None
            and child_selected_exact is not None
            and abs(neural_child_optimal_minus_selected or 0) < 0.02
            and abs(exact_child_optimal_minus_selected or 0) > 0.05
        ):
            classification = "value_head_underestimates_optimal_child"
        elif root_value_error is not None and abs(root_value_error) < 0.05:
            classification = "value_error_small"
        elif row.get("_root_puct_classification") == "persistent_failure":
            classification = "value_head_ok_search_fails"
        else:
            classification = "value_error_small"

        results.append(
            {
                "candidate_id": cid,
                "exact_root_value": round_float(exact_root_value),
                "neural_root_value": round_float(neural_root_value),
                "root_value_error": root_value_error,
                "optimal_child_exact_value": child_optimal_exact,
                "selected_child_exact_value": child_selected_exact,
                "optimal_child_neural_value": child_optimal_neural,
                "selected_child_neural_value": child_selected_neural,
                "neural_child_optimal_minus_selected": neural_child_optimal_minus_selected,
                "exact_child_optimal_minus_selected": exact_child_optimal_minus_selected,
                "value_rank_error": value_rank_error,
                "sign_error": sign_error,
                "classification": classification,
                "notes": "",
            }
        )

    return results


# ── Step 5: Child PUCT audit ────────────────────────────────────────────────


def child_puct_audit(
    evaluator: ArtifactEvaluator,
    valid_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tb = EndgameTablebase()
    results: list[dict[str, Any]] = []

    for row in valid_rows:
        cid = str(row["candidate_id"])
        classification = row.get("_root_puct_classification", "")
        if classification == "passes_at_high_budget":
            continue

        state = row["state"]
        root_player = KalahGame.from_state(state).current_player
        optimal_move = (
            row.get("_tb_optimal_moves", [None])[0]
            if row.get("_tb_optimal_moves")
            else None
        )
        baseline_1200 = next(
            (
                r
                for r in row.get("_baseline_results", [])
                if r["row_id"] == cid and r["budget"] == 1200
            ),
            None,
        )
        selected_move = baseline_1200["selected_move"] if baseline_1200 else None

        for budget in CHILD_PUCT_BUDGETS:
            for child_label, child_move in [
                ("optimal_child", optimal_move),
                ("selected_child", selected_move),
            ]:
                if child_move is None:
                    continue
                if child_label == "selected_child" and (
                    optimal_move is not None and child_move == optimal_move
                ):
                    continue

                child_s = child_state_from_move(state, child_move)

                child_result = evaluate_artifact_position(
                    artifact_path=None,
                    evaluator=evaluator,
                    state=child_s,
                    simulations=int(budget),
                    seed=SEED,
                    c_puct=C_PUCT,
                    search_options=dict(SEARCH_OPTIONS),
                    ablation_mode="full",
                )
                child_raw = float(
                    (child_result.get("selection_breakdown") or {}).get(
                        "parent_q_value", 0.0
                    )
                )
                child_puct_value = state_to_root_perspective_value(
                    raw_value=child_raw,
                    state=child_s,
                    root_player=root_player,
                )

                child_game = KalahGame.from_state(child_s)
                c_wr = tb.lookup(child_game, root_player)
                child_exact = (2.0 * float(c_wr)) - 1.0 if c_wr is not None else None
                child_puct_error = (
                    round_float(child_puct_value - child_exact)
                    if child_exact is not None
                    else None
                )

                child_selected = child_result.get("selected_move")
                child_legal = [int(m) for m in child_result.get("legal_moves", [])]
                if child_exact is not None:
                    c_offset = child_game.current_player * 6
                    cv_map: dict[int, float] = {}
                    best_cv = -float("inf")
                    for m in child_legal:
                        cg = child_game.clone()
                        cg.move(c_offset + m)
                        cwr = tb.lookup(cg, root_player)
                        if cwr is not None:
                            cv = (2.0 * float(cwr)) - 1.0
                            cv_map[m] = cv
                            if cv > best_cv:
                                best_cv = cv
                    child_optimal_moves = sorted(
                        m for m, v in cv_map.items() if abs(v - best_cv) < EPS
                    )
                    child_puct_agrees = (
                        child_selected in child_optimal_moves
                        if child_selected is not None
                        else None
                    )
                else:
                    child_puct_agrees = None

                if (
                    child_exact is not None
                    and child_puct_error is not None
                    and abs(child_puct_error) > 0.05
                ):
                    child_classification = "child_puct_value_error"
                elif child_puct_agrees is False:
                    child_classification = "child_puct_ok_root_selection_fails"
                elif child_exact is None:
                    child_classification = "child_tablebase_unavailable"
                else:
                    child_classification = "inconclusive"

                results.append(
                    {
                        "candidate_id": cid,
                        "child_label": child_label,
                        "budget": int(budget),
                        "child_puct_value_root": round_float(child_puct_value),
                        "child_exact_value_root": child_exact,
                        "child_puct_error": child_puct_error,
                        "child_selected_move": child_selected,
                        "child_puct_agrees_with_exact": child_puct_agrees,
                        "classification": child_classification,
                        "notes": "",
                    }
                )

    return results


# ── Diagnostic PUCT for counterfactuals ─────────────────────────────────────


class DiagnosticPUCT(PUCT):
    def __init__(
        self,
        *args,
        child_value_overrides: dict[int, float] | None = None,
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


# ── Step 6: Root counterfactual diagnostics ────────────────────────────────


def run_counterfactual_diagnostics(
    evaluator: ArtifactEvaluator,
    valid_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for row in valid_rows:
        cid = str(row["candidate_id"])
        classification = row.get("_root_puct_classification", "")
        if classification == "passes_at_high_budget":
            continue

        state = row["state"]
        root_player = KalahGame.from_state(state).current_player
        optimal_move = (
            row.get("_tb_optimal_moves", [None])[0]
            if row.get("_tb_optimal_moves")
            else None
        )
        baseline_1200 = next(
            (
                r
                for r in row.get("_baseline_results", [])
                if r["row_id"] == cid and r["budget"] == 1200
            ),
            None,
        )
        selected_move = baseline_1200["selected_move"] if baseline_1200 else None

        if optimal_move is None or selected_move is None:
            continue

        tb = EndgameTablebase()
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

        interventions = {
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
                "root_prior_override": (
                    make_equalize_optimal_selected_priors_override(
                        optimal_move, selected_move
                    )
                ),
                "child_value_overrides": None,
                "notes": "equalize optimal/selected priors",
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


# ── Step 7: Build clean split ──────────────────────────────────────────────


def build_clean_split(
    valid_rows: list[dict[str, Any]],
    exact_results: list[dict[str, Any]],
    baseline_results: list[dict[str, Any]],
    value_results: list[dict[str, Any]],
) -> dict[str, Any]:
    split_rows: list[dict[str, Any]] = []
    target_count = 0
    control_count = 0
    holdout_count = 0
    excluded_count = 0

    for row in valid_rows:
        cid = str(row["candidate_id"])
        exact = next((r for r in exact_results if r["candidate_id"] == cid), {})
        signal = exact.get("exact_signal_class", "")
        value = next((r for r in value_results if r["candidate_id"] == cid), {})
        puct_classification = row.get("_root_puct_classification", "")

        if signal in (
            "tablebase_error",
            "tablebase_not_unique",
            "exact_unique_tiny_margin",
        ):
            excluded_count += 1
            split_rows.append(
                {
                    "candidate_id": cid,
                    "assigned_role": "exclude",
                    "reason": f"exact signal: {signal}" if signal else "excluded",
                    "exact_signal_class": signal,
                    "puct_failure_status": puct_classification,
                    "value_error_class": value.get("classification", ""),
                    "notes": "",
                }
            )
            continue

        optimal_move = exact.get("tablebase_optimal_move")
        if optimal_move is None:
            excluded_count += 1
            split_rows.append(
                {
                    "candidate_id": cid,
                    "assigned_role": "exclude",
                    "reason": "no unique optimal move",
                    "exact_signal_class": signal,
                    "puct_failure_status": puct_classification,
                    "value_error_class": value.get("classification", ""),
                    "notes": "",
                }
            )
            continue

        fails_at_384 = not row.get("_puct_pass_384", False)
        fails_at_1200 = not row.get("_puct_pass_1200", False)

        if fails_at_384 or fails_at_1200:
            role = "target_candidate"
            target_count += 1
        else:
            passes_at_2400 = row.get("_puct_pass_2400", False) or row.get(
                "_puct_pass_5000", False
            )
            if passes_at_2400:
                if control_count < 2:
                    role = "preservation_control"
                    control_count += 1
                else:
                    role = "holdout_candidate"
                    holdout_count += 1
            else:
                role = "holdout_candidate"
                holdout_count += 1

        split_rows.append(
            {
                "candidate_id": cid,
                "assigned_role": role,
                "reason": (
                    "PUCT fails at search budgets"
                    if role == "target_candidate"
                    else "preserves under PUCT"
                    if role == "preservation_control"
                    else "holdout"
                ),
                "exact_signal_class": signal,
                "puct_failure_status": puct_classification,
                "value_error_class": value.get("classification", ""),
                "notes": "",
            }
        )

    return {
        "split_rows": split_rows,
        "target_candidate_count": target_count,
        "preservation_control_count": control_count,
        "holdout_count": holdout_count,
        "excluded_count": excluded_count,
    }


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load rows
    evaluator = ArtifactEvaluator(CURRENT_ARTIFACT)
    selected_rows = load_jsonl(SELECTED_ROWS_PATH)

    print(f"Loaded {len(selected_rows)} selected rows from {SELECTED_ROWS_PATH}")

    # Step 1: Validate
    print("\n=== Step 1: Validation ===")
    validation = validate_selected_rows(selected_rows)
    valid_count = sum(1 for v in validation if v["valid"])
    invalid_count = sum(1 for v in validation if not v["valid"])
    print(f"  Valid: {valid_count}, Invalid: {invalid_count}")

    valid_rows = [row for row, v in zip(selected_rows, validation) if v["valid"]]

    # Attach _tb_optimal_moves and state
    for row in valid_rows:
        game = KalahGame.from_state(row["state"])
        legal = game.possible_moves()
        row["_legal_moves"] = legal
        if "_tb_child_values" not in row or "_tb_optimal_moves" not in row:
            tb_inner = EndgameTablebase()
            offset = game.current_player * 6
            child_values: dict[int, float] = {}
            optimal: list[int] = []
            best_val = -float("inf")
            for move in legal:
                child_game = game.clone()
                child_game.move(offset + move)
                c_wr = tb_inner.lookup(child_game, game.current_player)
                if c_wr is not None:
                    cv = (2.0 * float(c_wr)) - 1.0
                    child_values[move] = round_float(cv)
                    if cv > best_val:
                        best_val = cv
            if best_val > -float("inf"):
                optimal = sorted(
                    m for m, v in child_values.items() if abs(v - best_val) < EPS
                )
            row["_tb_child_values"] = child_values
            row["_tb_optimal_moves"] = optimal

    # Step 2: Exact tablebase enumeration
    print("\n=== Step 2: Exact Tablebase Enumeration ===")
    exact_results = exact_tablebase_enumeration(valid_rows)
    for r in exact_results:
        print(
            f"  {r['candidate_id']}: {r['exact_signal_class']} "
            f"optimal={r['tablebase_optimal_move']} "
            f"best-second={r['best_minus_second_best']}"
        )

    # Step 3: Current PUCT baseline
    print("\n=== Step 3: Current PUCT Baseline ===")
    baseline_results = run_baseline(evaluator, valid_rows)
    for row in valid_rows:
        cid = str(row["candidate_id"])
        row["_baseline_results"] = [r for r in baseline_results if r["row_id"] == cid]

    for row in valid_rows:
        cid = str(row["candidate_id"])
        cls = row.get("_root_puct_classification", "")
        print(
            f"  {cid}: {cls} "
            f"(128={row.get('_puct_pass_128', '?')}, "
            f"384={row.get('_puct_pass_384', '?')}, "
            f"1200={row.get('_puct_pass_1200', '?')}, "
            f"2400={row.get('_puct_pass_2400', '?')}, "
            f"5000={row.get('_puct_pass_5000', '?')})"
        )

    # Step 4: Neural value audit
    print("\n=== Step 4: Neural Value Audit ===")
    value_results = neural_value_audit(evaluator, valid_rows)
    for r in value_results:
        print(
            f"  {r['candidate_id']}: {r['classification']} "
            f"ve={r['root_value_error']} "
            f"rank_err={r['value_rank_error']} "
            f"sign_err={r['sign_error']}"
        )

    # Step 5: Child PUCT audit
    print("\n=== Step 5: Child PUCT Audit ===")
    child_puct_results = child_puct_audit(evaluator, valid_rows)
    for r in child_puct_results:
        print(
            f"  {r['candidate_id']} {r['child_label']}@{r['budget']}: "
            f"{r['classification']} err={r['child_puct_error']}"
        )

    # Step 6: Root counterfactual diagnostics
    print("\n=== Step 6: Root Counterfactual Diagnostics ===")
    counterfactual_results = run_counterfactual_diagnostics(evaluator, valid_rows)
    for r in counterfactual_results:
        print(
            f"  {r['candidate_id']} {r['intervention']}@{r['budget']}: "
            f"sel={r['selected_move']} optimal={r['selected_is_optimal']} "
            f"flipped={r['flipped_to_optimal']}"
        )

    # Step 7: Build clean split
    print("\n=== Step 7: Clean Split ===")
    split = build_clean_split(
        valid_rows, exact_results, baseline_results, value_results
    )
    print(
        f"  Target: {split['target_candidate_count']}, "
        f"Control: {split['preservation_control_count']}, "
        f"Holdout: {split['holdout_count']}, "
        f"Excluded: {split['excluded_count']}"
    )

    # Step 8: Decision
    print("\n=== Step 8: Decision ===")
    meaningful_value_rank_errors = sum(
        1 for r in value_results if r.get("value_rank_error")
    )
    meaningful_value_sign_errors = sum(1 for r in value_results if r.get("sign_error"))
    value_err_count = meaningful_value_rank_errors + meaningful_value_sign_errors

    failing_at_384 = sum(
        1 for row in valid_rows if not row.get("_puct_pass_384", False)
    )
    failing_at_1200 = sum(
        1 for row in valid_rows if not row.get("_puct_pass_1200", False)
    )
    persistent_failures = sum(
        1
        for row in valid_rows
        if row.get("_root_puct_classification") == "persistent_failure"
    )
    low_budget_only = sum(
        1
        for row in valid_rows
        if row.get("_root_puct_classification") == "low_budget_only_failure"
    )

    target_candidates = split["target_candidate_count"]
    controls = split["preservation_control_count"]
    holdouts = split["holdout_count"]
    excluded = split["excluded_count"]

    # Decision logic
    excluded_or_invalid = invalid_count + excluded
    if excluded_or_invalid >= len(selected_rows) * 0.5:
        decision = "tablebase_unique_not_clean_after_validation"
        next_action = "tighten fresh mining filters and rerun"
        dominant = "validation_error"
    elif target_candidates < 5:
        decision = "tablebase_unique_too_small"
        next_action = "mine more fresh exact-tablebase unique rows"
        dominant = "too_few_targets"
    elif low_budget_only >= target_candidates * 0.5:
        decision = "tablebase_unique_mostly_low_budget_noise"
        next_action = "do not train; use as low-budget search diagnostic only"
        dominant = "low_budget_noise"
    elif value_err_count >= min(3, target_candidates) and target_candidates >= 3:
        decision = "tablebase_unique_value_diagnostic_ready"
        next_action = "build a tiny exact-tablebase value-calibration diagnostic artifact; no arena until local exact-value metrics improve"
        dominant = "value_error"
    elif persistent_failures >= target_candidates * 0.5 and target_candidates >= 3:
        decision = "tablebase_unique_search_diagnostic_ready"
        next_action = "run root cpuct/prior/value-override diagnostics on selected rows"
        dominant = "search_selection"
    elif target_candidates >= 3:
        decision = "tablebase_unique_policy_artifact_ready"
        next_action = "build a tiny train-only tablebase policy/value diagnostic artifact with controls"
        dominant = "policy_artifact"
    else:
        decision = "tablebase_unique_too_small"
        next_action = "mine more fresh exact-tablebase unique rows"
        dominant = "too_few_targets"

    print(f"  Decision: {decision}")
    print(f"  Next action: {next_action}")

    # ── Build row-level diagnostics JSONL ──────────────────────────────
    row_diagnostics: list[dict[str, Any]] = []
    for row in valid_rows:
        cid = str(row["candidate_id"])
        exact = next((r for r in exact_results if r["candidate_id"] == cid), {})
        value = next((r for r in value_results if r["candidate_id"] == cid), {})
        baselines = [r for r in baseline_results if r["row_id"] == cid]
        child_audits = [r for r in child_puct_results if r["candidate_id"] == cid]
        cfs = [r for r in counterfactual_results if r["candidate_id"] == cid]
        split_info = next(
            (r for r in split["split_rows"] if r["candidate_id"] == cid), {}
        )

        row_diagnostics.append(
            {
                "candidate_id": cid,
                "provisional_family": FAMILY,
                "canonical_state_hash": str(row.get("canonical_state_hash", "")),
                "state": row.get("state"),
                "legal_moves": row.get("_legal_moves", []),
                "validation": next(
                    (v for v in validation if v["candidate_id"] == cid), {}
                ),
                "exact_tablebase": exact,
                "puct_baseline": baselines,
                "neural_value_audit": value,
                "child_puct_audit": child_audits,
                "counterfactual": cfs,
                "assigned_role": split_info.get("assigned_role", "unassigned"),
                "root_puct_classification": row.get("_root_puct_classification", ""),
            }
        )
    # Also include invalid rows in diagnostics
    for row, v in zip(selected_rows, validation):
        if not v["valid"]:
            row_diagnostics.append(
                {
                    "candidate_id": str(row.get("candidate_id", "")),
                    "provisional_family": str(row.get("provisional_family", "")),
                    "canonical_state_hash": str(row.get("canonical_state_hash", "")),
                    "state": row.get("state"),
                    "legal_moves": KalahGame.from_state(row["state"]).possible_moves()
                    if row.get("state")
                    else [],
                    "validation": v,
                    "exact_tablebase": {},
                    "puct_baseline": [],
                    "neural_value_audit": {},
                    "child_puct_audit": [],
                    "counterfactual": [],
                    "assigned_role": "excluded",
                    "root_puct_classification": "invalid",
                }
            )

    write_jsonl(
        OUTPUT_DIR / "fresh_endgame_tablebase_unique_row_diagnostics.jsonl",
        row_diagnostics,
    )

    # ── Build summary JSON ─────────────────────────────────────────────
    summary = {
        "schema": "azlite_fresh_endgame_tablebase_unique_diagnostics_v1",
        "family": FAMILY,
        "description": (
            "Tablebase-backed local search/value diagnostics for "
            "fresh_endgame_tablebase_unique. Determines whether the family "
            "is a clean exact-target family suitable for a later tiny "
            "diagnostic artifact."
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
        "row_validation": {
            "total_selected_rows": len(selected_rows),
            "valid_rows": valid_count,
            "invalid_rows": invalid_count,
        },
        "validation_table": validation,
        "exact_tablebase_table": exact_results,
        "puct_baseline_table": baseline_results,
        "neural_value_audit_table": value_results,
        "child_puct_audit_table": child_puct_results,
        "counterfactual_table": counterfactual_results,
        "clean_split": split,
        "decision": {
            "classification": decision,
            "next_action": next_action,
            "dominant_mechanism": dominant,
            "counts": {
                "target_candidates": target_candidates,
                "preservation_controls": controls,
                "holdouts": holdouts,
                "excluded": excluded,
                "invalid": invalid_count,
                "total_selected": len(selected_rows),
                "failing_at_384": failing_at_384,
                "failing_at_1200": failing_at_1200,
                "persistent_failures": persistent_failures,
                "low_budget_only_failures": low_budget_only,
                "value_rank_errors": meaningful_value_rank_errors,
                "value_sign_errors": meaningful_value_sign_errors,
                "meaningful_value_errors": value_err_count,
            },
        },
    }

    write_json(
        OUTPUT_DIR / "fresh_endgame_tablebase_unique_diagnostics_summary.json",
        summary,
    )

    # ── Write clean split JSON ─────────────────────────────────────────
    clean_split = {
        "schema": "azlite_fresh_endgame_tablebase_unique_clean_split_v1",
        "family": FAMILY,
        "split_rows": split["split_rows"],
        "counts": {
            "target_candidates": split["target_candidate_count"],
            "preservation_controls": split["preservation_control_count"],
            "holdouts": split["holdout_count"],
            "excluded": split["excluded_count"],
        },
        "decision": {
            "classification": decision,
            "next_action": next_action,
        },
    }
    write_json(
        OUTPUT_DIR / "fresh_endgame_tablebase_unique_clean_split.json",
        clean_split,
    )

    print(f"\nWrote all outputs to {OUTPUT_DIR}/")
    print(
        f"  Summary: {OUTPUT_DIR / 'fresh_endgame_tablebase_unique_diagnostics_summary.json'}"
    )
    print(
        f"  Row diagnostics: {OUTPUT_DIR / 'fresh_endgame_tablebase_unique_row_diagnostics.jsonl'}"
    )
    print(
        f"  Clean split: {OUTPUT_DIR / 'fresh_endgame_tablebase_unique_clean_split.json'}"
    )

    # Console summary
    print(f"\n{'=' * 90}")
    print("FRESH ENDGAME TABLEBASE UNIQUE — DIAGNOSTICS")
    print(f"{'=' * 90}")
    print(f"\n{'Validation':30s} {valid_count} valid / {len(validation)} total")
    print(f"{'Target candidates':30s} {target_candidates}")
    print(f"{'Preservation controls':30s} {controls}")
    print(f"{'Holdouts':30s} {holdouts}")
    print(f"{'Excluded':30s} {excluded}")
    print(f"\n{'Classification':30s} {decision}")
    print(f"{'Dominant mechanism':30s} {dominant}")
    print(f"{'Next action':30s} {next_action}")
    print(f"\n{'=' * 90}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
