#!/usr/bin/env python3
"""Build larger train-only exact-tablebase diagnostic artifacts.

Reuses PR #74/PR #75 clean split rows and generates additional adversarial
endgame candidates to reach target row counts:
  - production candidates: >= 30
  - value-only candidates: >= 40
  - preservation controls: >= 8
  - holdouts: >= 50

Does not run arena, does not promote, does not touch storage/ai/alphazero_lite/current.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import (
    build_eval_search_options,
    encode_state,
)

FAMILY = "harder_fresh_endgame_tablebase"
PITS_PER_PLAYER = 6
EPS = 1e-9
INPUT_ENCODING = "kalah_v3"
POLICY_SIZE = 6
OPTIMAL_POLICY_MASS = 0.85
C_PUCT = 1.25
SEED = 17
SEARCH_OPTIONS = build_eval_search_options(
    fpu_mode="parent_q",
    reuse_subtree=True,
    normalize_values=True,
    root_policy_mode="deterministic",
    tactical_root_bias=0.1,
)

CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
REFERENCE_PATH = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")

EXISTING_CLEAN_SPLIT_PATH = Path(
    "/tmp/azlite_harder_endgame_tablebase_local_diagnostics/"
    "harder_endgame_tablebase_clean_split.json"
)
EXISTING_ROW_DIAGNOSTICS_PATH = Path(
    "/tmp/azlite_harder_endgame_tablebase_local_diagnostics/"
    "harder_endgame_tablebase_local_row_diagnostics.jsonl"
)
OUTPUT_DIR = Path("/tmp/azlite_larger_harder_endgame_tablebase_diagnostic")

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


def is_exhausted_row_id(row_id: str) -> bool:
    for prefix in EXHAUSTED_ROW_ID_PREFIXES:
        if row_id.startswith(prefix):
            return True
    if row_id in CORRECTED_GUARD_ROW_IDS:
        return True
    return False


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


def total_seeds_remaining(state: dict[str, Any]) -> int:
    return sum(state.get("player_pits", [])) + sum(state.get("opponent_pits", []))


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


def build_policy_target(optimal_move: int, legal_moves: list[int]) -> list[float]:
    policy = [0.0] * POLICY_SIZE
    if optimal_move not in legal_moves:
        return policy
    other_legal = [m for m in legal_moves if m != optimal_move]
    policy[optimal_move] = OPTIMAL_POLICY_MASS
    if other_legal:
        remaining = 1.0 - OPTIMAL_POLICY_MASS
        per_move = remaining / len(other_legal)
        for m in other_legal:
            policy[m] = per_move
    total = sum(policy)
    if total > 0:
        policy = [p / total for p in policy]
    return policy


# ── Step 1: Generate additional adversarial candidates ──────────────────────


def generate_adversarial_candidates(
    *,
    samples: int,
    seeds: list[int],
    min_seeds: int = 2,
    max_seeds: int = 14,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    tb = EndgameTablebase()
    rng = random.Random(seeds[0] * 7)

    attrs = 0
    while attrs < samples:
        seeds_rem = rng.randint(min_seeds, min(max_seeds, 12))
        pits_list = [0] * 12
        remaining = seeds_rem
        for i in range(12):
            if i == 11:
                pits_list[i] = remaining
            else:
                take = rng.randint(0, remaining)
                pits_list[i] = take
                remaining -= take
        rng.shuffle(pits_list)
        current_player = rng.randint(0, 1)
        captured = [rng.randint(0, 8), rng.randint(0, 8)]

        offset = current_player * 6
        if all(p == 0 for p in pits_list[offset : offset + 6]):
            continue

        player_seeds = sum(pits_list[offset : offset + 6])
        opp_seeds = sum(pits_list[6 - offset : 12 - offset])
        if player_seeds == 0 or opp_seeds == 0:
            continue

        state = {
            "player_pits": pits_list[:6],
            "opponent_pits": pits_list[6:],
            "player_store": captured[0],
            "opponent_store": captured[1],
            "current_player": current_player,
        }
        game = KalahGame.from_state(state)

        tb_wr = tb.lookup(game, current_player)
        if tb_wr is None:
            continue

        legal = game.possible_moves()
        offset_i = current_player * 6
        best_val = -float("inf")
        second_val = -float("inf")
        child_vals: dict[int, float] = {}
        optimal_moves: list[int] = []
        for move in legal:
            child_game = game.clone()
            child_game.move(offset_i + move)
            c_wr = tb.lookup(child_game, current_player)
            if c_wr is None:
                continue
            cv = (2.0 * float(c_wr)) - 1.0
            child_vals[move] = cv
            if cv > best_val:
                second_val = best_val
                best_val = cv
            elif cv > second_val and cv < best_val:
                second_val = cv

        if best_val > -float("inf"):
            optimal_moves = sorted(
                m for m, v in child_vals.items() if abs(v - best_val) < EPS
            )

        if len(optimal_moves) != 1:
            continue

        gap = round_float(best_val - second_val) if second_val > -float("inf") else None
        if gap is None or gap < EPS:
            continue

        all_vals = set(round_float(v, 6) for v in child_vals.values())
        if len(all_vals) == 1:
            continue

        c_hash = canonical_state_key(state)
        candidates.append(
            {
                "state": state,
                "canonical_state_hash": c_hash,
                "ply": None,
                "source_game_seed": seeds[0] + attrs + 10000,
                "source": "adversarial_near_threshold",
                "remaining_seeds": seeds_rem,
                "tb_optimal_move": optimal_moves[0],
                "gap": gap,
            }
        )
        attrs += 1

    return candidates


# ── Step 2: Dedup against existing data and suites ──────────────────────────


def deduplicate_candidates(
    candidates: list[dict[str, Any]],
    existing_hashes: set[str],
    suite_canon: set[str],
    suite_rows: dict[str, dict],
) -> list[dict[str, Any]]:
    seen_hashes: set[str] = set(existing_hashes)
    clean: list[dict[str, Any]] = []
    tb = EndgameTablebase()

    for cand in candidates:
        c_hash = cand["canonical_state_hash"]

        if c_hash in seen_hashes:
            continue
        seen_hashes.add(c_hash)

        if c_hash in suite_canon:
            overlaps_exhausted = False
            for row_id, row in suite_rows.items():
                row_cs = row.get("canonical_state") or canonical_state_key(
                    row.get("state", {})
                )
                if row_cs == c_hash:
                    if is_exhausted_row_id(row_id):
                        overlaps_exhausted = True
                        break
                    bucket = row.get("bucket", "")
                    if bucket in EXHAUSTED_BUCKETS:
                        overlaps_exhausted = True
                        break
                    break
            if overlaps_exhausted:
                continue

        state = cand["state"]
        game = KalahGame.from_state(state)
        tb_wr = tb.lookup(game, game.current_player)
        if tb_wr is None:
            continue

        legal = game.possible_moves()
        offset_i = game.current_player * 6
        child_vals: dict[int, float] = {}
        best_val = -float("inf")
        for move in legal:
            child_game = game.clone()
            child_game.move(offset_i + move)
            c_wr = tb.lookup(child_game, game.current_player)
            if c_wr is not None:
                cv = (2.0 * float(c_wr)) - 1.0
                child_vals[move] = cv
                if cv > best_val:
                    best_val = cv

        optimal_moves = sorted(
            m for m, v in child_vals.items() if abs(v - best_val) < EPS
        )
        if len(optimal_moves) == 0:
            continue
        if len(optimal_moves) > 1:
            continue

        all_vals = set(round_float(v, 6) for v in child_vals.values())
        if len(all_vals) == 1:
            continue

        unique_optimal = optimal_moves[0]
        sorted_vals = sorted([v for v in child_vals.values()], reverse=True)
        second_best = sorted_vals[1] if len(sorted_vals) >= 2 else None
        gap = round_float(best_val - second_best) if second_best is not None else None

        cand["_tb_root_value"] = round_float((2.0 * float(tb_wr)) - 1.0)
        cand["_tb_child_values"] = child_vals
        cand["_tb_optimal_moves"] = optimal_moves
        cand["_tb_unique_optimal"] = unique_optimal
        cand["_tb_best_minus_second"] = gap
        cand["_tb_all_equiv"] = False
        cand["_tb_second_best"] = (
            round_float(second_best) if second_best is not None else None
        )

        clean.append(cand)

    return clean


# ── Step 3: PUCT baseline on new candidates ─────────────────────────────────


def run_single_puct(
    evaluator: ArtifactEvaluator,
    state: dict[str, Any],
    budget: int,
    seed: int,
) -> dict[str, Any]:
    return evaluate_artifact_position(
        artifact_path=None,
        evaluator=evaluator,
        state=state,
        simulations=int(budget),
        seed=int(seed),
        c_puct=C_PUCT,
        search_options=dict(SEARCH_OPTIONS),
        ablation_mode="full",
    )


def run_puct_baseline(
    evaluator: ArtifactEvaluator,
    candidates: list[dict[str, Any]],
) -> None:
    budgets = (64, 384, 1200)

    for idx, cand in enumerate(candidates):
        if idx > 0 and idx % 25 == 0:
            print(f"    PUCT baseline: {idx}/{len(candidates)}")
        state = cand["state"]
        optimal_move = cand.get("_tb_unique_optimal")
        if optimal_move is None:
            continue

        row_results = []
        for budget in budgets:
            r = run_single_puct(evaluator, state, budget, SEED)
            selected_move = (
                None if r.get("selected_move") is None else int(r["selected_move"])
            )
            selection_map = selection_entry_map(r)
            opt_entry = (
                selection_map.get(optimal_move) if optimal_move is not None else {}
            )
            sel_entry = (
                selection_map.get(selected_move) if selected_move is not None else {}
            )
            selected_is_optimal = (
                selected_move == optimal_move if optimal_move is not None else None
            )
            visits_list = [float(v) for v in r.get("visits", [])]
            policy_list = [float(p) for p in r.get("policy", [])]

            row_results.append(
                {
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
                    "optimal_policy_rank": policy_rank_of_move(
                        policy_list, optimal_move
                    )
                    if optimal_move is not None
                    else None,
                    "pass_status": "pass" if selected_is_optimal else "fail",
                }
            )

        cand["_puct_baseline_results"] = row_results

        fails_384 = any(
            r["budget"] == 384 and not r["selected_is_optimal"] for r in row_results
        )
        fails_1200 = any(
            r["budget"] == 1200 and not r["selected_is_optimal"] for r in row_results
        )
        passes_64 = any(
            r["budget"] == 64 and r["selected_is_optimal"] for r in row_results
        )

        if fails_1200 or (
            fails_384
            and not any(
                r["budget"] == 1200 and r["selected_is_optimal"] for r in row_results
            )
        ):
            cand["_puct_class"] = "persistent_exact_failure"
            for budget in (2400,):
                r = run_single_puct(evaluator, state, budget, SEED + 1)
                selected_move = (
                    None if r.get("selected_move") is None else int(r["selected_move"])
                )
                selection_map = selection_entry_map(r)
                opt_entry = (
                    selection_map.get(optimal_move) if optimal_move is not None else {}
                )
                selected_is_optimal = (
                    selected_move == optimal_move if optimal_move is not None else None
                )
                visits_list = [float(v) for v in r.get("visits", [])]
                row_results.append(
                    {
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
                        "pass_status": "pass" if selected_is_optimal else "fail",
                    }
                )
        elif fails_384:
            cand["_puct_class"] = "medium_budget_exact_failure"
        elif not passes_64:
            cand["_puct_class"] = "low_budget_exact_failure"
        else:
            cand["_puct_class"] = "exact_clean_control"

        cand["_puct_pass_384"] = any(
            r["budget"] == 384 and r["selected_is_optimal"] for r in row_results
        )
        cand["_puct_pass_1200"] = any(
            r["budget"] == 1200 and r["selected_is_optimal"] for r in row_results
        )


# ── Step 4: Neural value rank scan on new candidates ────────────────────────


def run_neural_value_rank_scan(
    evaluator: ArtifactEvaluator,
    candidates: list[dict[str, Any]],
) -> None:
    tb = EndgameTablebase()
    for cand in candidates:
        state = cand["state"]
        root_player = KalahGame.from_state(state).current_player
        optimal_move = cand.get("_tb_unique_optimal")
        if optimal_move is None:
            continue

        legal_moves = KalahGame.from_state(state).possible_moves()
        neural_child_values: dict[int, float | None] = {}
        exact_child_values: dict[int, float | None] = {}

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

        neural_clean = {m: v for m, v in neural_child_values.items() if v is not None}
        if not neural_clean:
            continue
        neural_best_move = max(neural_clean, key=lambda m: neural_clean[m])
        neural_best_is_exact_optimal = neural_best_move == optimal_move

        value_rank_error = False
        if not neural_best_is_exact_optimal and optimal_move in neural_child_values:
            value_rank_error = True

        _, raw_root_nv = evaluator.evaluate(KalahGame.from_state(state))
        neural_root = float(raw_root_nv)
        tb_wr = tb.lookup(KalahGame.from_state(state), root_player)
        exact_root = (2.0 * float(tb_wr)) - 1.0 if tb_wr is not None else None
        sign_error = False
        if exact_root is not None and abs(exact_root) > EPS:
            exact_sign = math.copysign(1.0, exact_root)
            neural_sign = math.copysign(1.0, neural_root)
            sign_error = exact_sign != neural_sign

        cand["_neural_rank_error"] = value_rank_error
        cand["_neural_sign_error"] = sign_error
        cand["_neural_best_move"] = neural_best_move
        cand["_neural_best_is_optimal"] = neural_best_is_exact_optimal


# ── Step 5: Classify candidates ─────────────────────────────────────────────


def classify_candidates(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    split_rows: list[dict[str, Any]] = []
    target_count = 0
    control_count = 0
    holdout_count = 0
    low_budget_count = 0
    excluded_count = 0

    for cand in candidates:
        c_hash = cand["canonical_state_hash"]
        puct_class = cand.get("_puct_class", "")
        value_rank_error = cand.get("_neural_rank_error", False)

        failure_budget = None
        baselines = cand.get("_puct_baseline_results", [])
        for r in sorted(baselines, key=lambda x: x["budget"]):
            if not r.get("selected_is_optimal", True):
                failure_budget = r["budget"]
                break
        if failure_budget is None and value_rank_error:
            failure_budget = "value_only"

        reason = ""
        if puct_class == "persistent_exact_failure":
            role = "target_candidate"
            target_count += 1
            reason = "PUCT fails persistently at mid-high budget"
        elif puct_class == "medium_budget_exact_failure":
            role = "target_candidate"
            target_count += 1
            reason = "PUCT fails at medium budget, recovers"
        elif puct_class == "low_budget_exact_failure":
            role = "low_budget_diagnostic_only"
            low_budget_count += 1
            reason = "low-budget only failure"
        elif puct_class == "exact_clean_control":
            if value_rank_error:
                role = "target_candidate"
                target_count += 1
                reason = "neural value rank error even though PUCT passes"
            elif control_count < 8:
                role = "preservation_control"
                control_count += 1
                reason = "preserves under PUCT"
            else:
                role = "holdout_candidate"
                holdout_count += 1
                reason = "holdout"
        else:
            role = "holdout_candidate"
            holdout_count += 1
            reason = "holdout"

        split_rows.append(
            {
                "candidate_hash": c_hash,
                "assigned_role": role,
                "failure_class": puct_class,
                "exact_signal_class": "exact_unique_clear_margin",
                "puct_failure_budget": failure_budget,
                "value_rank_error": value_rank_error,
                "reason": reason,
                "notes": "",
            }
        )

    return {
        "split_rows": split_rows,
        "target_candidate_count": target_count,
        "preservation_control_count": control_count,
        "holdout_count": holdout_count,
        "low_budget_diagnostic_count": low_budget_count,
        "excluded_count": excluded_count,
    }


# ── Step 6: Build refined clean split ───────────────────────────────────────


def build_refined_clean_split(
    existing_split_rows: list[dict[str, Any]],
    existing_diag_by_cid: dict[str, dict],
    new_candidates: list[dict[str, Any]],
    new_split: dict[str, Any],
    evaluator: ArtifactEvaluator,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {
        "production_candidate_later": 0,
        "value_only_candidate": 0,
        "preservation_control": 0,
        "holdout_candidate": 0,
        "search_diagnostic_only": 0,
        "exclude": 0,
    }

    for er in existing_split_rows:
        bucket = er.get("bucket", "")
        if bucket not in (
            "production_candidate_later",
            "value_only_candidate",
            "preservation_control",
            "holdout_candidate",
        ):
            continue
        counts[bucket] = counts.get(bucket, 0) + 1
        cid = er.get("candidate_id", "")
        diag = existing_diag_by_cid.get(cid, {})
        if diag:
            tb_info = diag.get("exact_tablebase", {})
            nv_info = diag.get("neural_value_audit", {})
            er["_state"] = diag.get("state", {})
            er["_legal_moves"] = diag.get("legal_moves", [])
            er["_canonical_state_hash"] = diag.get("canonical_state_hash", "")
            er["_optimal_move"] = tb_info.get("optimal_move")
            er["_root_value"] = tb_info.get("root_value")
            er["_best_minus_second_best"] = tb_info.get("best_minus_second_best")
            if "value_rank_error" not in er:
                er["value_rank_error"] = nv_info.get("neural_value_rank_error", False)
            if "sign_error" not in er:
                er["sign_error"] = nv_info.get("neural_value_sign_error", False)
            puct_list = diag.get("puct_baseline", [])
            if isinstance(puct_list, list):
                er["passes_384"] = any(
                    p.get("budget") == 384 and p.get("selected_is_optimal")
                    for p in puct_list
                )
                er["passes_1200"] = any(
                    p.get("budget") == 1200 and p.get("selected_is_optimal")
                    for p in puct_list
                )
        output_rows.append(er)

    for cand in new_candidates:
        c_hash = cand["canonical_state_hash"]
        state = cand["state"]
        optimal_move = cand.get("_tb_unique_optimal")
        puct_class = cand.get("_puct_class", "")
        value_rank_error = cand.get("_neural_rank_error", False)

        sr = next(
            (s for s in new_split["split_rows"] if s["candidate_hash"] == c_hash),
            None,
        )
        if sr is None:
            continue

        role = sr["assigned_role"]
        if role == "target_candidate":
            if value_rank_error and puct_class == "exact_clean_control":
                bucket = "value_only_candidate"
                mechanism = "value_error_but_search_compensates"
            else:
                bucket = "production_candidate_later"
                puct_results = cand.get("_puct_baseline_results", [])
                fails_384 = not any(
                    r["budget"] == 384 and r["selected_is_optimal"]
                    for r in puct_results
                )
                fails_1200 = not any(
                    r["budget"] == 1200 and r["selected_is_optimal"]
                    for r in puct_results
                )
                if not fails_384 and not fails_1200:
                    mechanism = "value_rank_error_drives_failure"
                elif fails_384 and not fails_1200:
                    mechanism = "root_selection_pressure"
                else:
                    mechanism = "child_puct_value_error"
        elif role == "preservation_control":
            bucket = "preservation_control"
            mechanism = "inconclusive"
        elif role == "holdout_candidate":
            bucket = "holdout_candidate"
            mechanism = "inconclusive"
        elif role == "low_budget_diagnostic_only":
            bucket = "search_diagnostic_only"
            mechanism = "low_budget_search_noise"
        else:
            bucket = "exclude"
            mechanism = "inconclusive"

        if bucket in counts:
            counts[bucket] += 1

        game = KalahGame.from_state(state)
        legal_moves = game.possible_moves()
        root_value = cand.get("_tb_root_value")
        best_minus_second = cand.get("_tb_best_minus_second")
        sign_error = cand.get("_neural_sign_error", False)

        passes_384 = any(
            r["budget"] == 384 and r["selected_is_optimal"]
            for r in cand.get("_puct_baseline_results", [])
        )
        passes_1200 = any(
            r["budget"] == 1200 and r["selected_is_optimal"]
            for r in cand.get("_puct_baseline_results", [])
        )

        output_rows.append(
            {
                "bucket": bucket,
                "candidate_id": f"harder_{cand.get('source_game_seed', '?')}_{cand.get('ply', '?')}",
                "exact_signal_class": "exact_unique_clear_margin",
                "mechanism": mechanism,
                "notes": "",
                "passes_1200": passes_1200,
                "passes_384": passes_384,
                "reason": sr["reason"],
                "sign_error": sign_error,
                "value_rank_error": value_rank_error,
                "_state": state,
                "_legal_moves": legal_moves,
                "_optimal_move": optimal_move,
                "_root_value": root_value,
                "_best_minus_second_best": best_minus_second,
                "_canonical_state_hash": c_hash,
            }
        )

    return output_rows, counts


# ── Step 7: Build diagnostic artifacts ──────────────────────────────────────


def validate_artifacts(
    policy_artifact: list[dict],
    value_artifact: list[dict],
    controls_artifact: list[dict],
) -> list[str]:
    errors: list[str] = []

    for label, artifact in [
        ("policy_value", policy_artifact),
        ("controls", controls_artifact),
    ]:
        for idx, row in enumerate(artifact):
            policy = row.get("policy", [])
            if abs(sum(policy) - 1.0) > 1e-6:
                errors.append(f"{label}[{idx}]: policy sum={sum(policy):.6f} != 1.0")
            value = float(row.get("value", 0.0))
            if value < -1.0 or value > 1.0:
                errors.append(f"{label}[{idx}]: value={value} out of range")
            if "exact_optimal_move" not in row:
                errors.append(f"{label}[{idx}]: missing exact_optimal_move")
            if "exact_root_value" not in row:
                errors.append(f"{label}[{idx}]: missing exact_root_value")
            cid = row.get("candidate_id", f"row_{idx}")
            if is_exhausted_row_id(cid):
                errors.append(f"{label}[{idx}]: exhausted row id {cid}")

    for idx, row in enumerate(policy_artifact):
        policy = row.get("policy", [])
        optimal_move = row.get("exact_optimal_move")
        if optimal_move is not None and optimal_move < len(policy):
            optimal_mass = policy[optimal_move]
            for m, mass in enumerate(policy):
                if m != optimal_move and mass > optimal_mass:
                    errors.append(
                        f"policy_value[{idx}]: move {m} has {mass:.4f} > "
                        f"optimal {optimal_mass:.4f}"
                    )
                    break

    seen_states: dict[str, str] = {}
    for label, artifact in [
        ("policy_value", policy_artifact),
        ("value_only", value_artifact),
        ("controls", controls_artifact),
    ]:
        for idx, row in enumerate(artifact):
            c_hash = row.get("canonical_state_hash", "")
            if c_hash:
                if c_hash in seen_states:
                    prev_label = seen_states[c_hash]
                    if prev_label != label:
                        v1 = next(
                            (
                                r.get("value")
                                for r in artifact
                                if r.get("canonical_state_hash") == c_hash
                            ),
                            None,
                        )
                        v2 = row.get("value")
                        if (
                            v1 is not None
                            and v2 is not None
                            and abs(float(v1) - float(v2)) > 1e-6
                        ):
                            errors.append(
                                f"state {c_hash}: conflicting value targets "
                                f"({v1} vs {v2}) across {prev_label}/{label}"
                            )
                else:
                    seen_states[c_hash] = label

    return errors


def build_artifacts(
    split_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    policy_value_rows: list[dict] = []
    value_only_rows: list[dict] = []
    controls_rows: list[dict] = []
    holdout_candidate_ids: list[str] = []

    for sr in split_rows:
        cid = sr.get("candidate_id", "")
        bucket = sr.get("bucket", "")
        mechanism = sr.get("mechanism", "")
        state = sr.get("_state")
        if not state:
            continue
        optimal_move = sr.get("_optimal_move")
        legal_moves = sr.get("_legal_moves", [])
        root_value = sr.get("_root_value")
        best_minus_second = sr.get("_best_minus_second_best")
        c_hash = sr.get("_canonical_state_hash", "")

        if root_value is None or optimal_move is None:
            continue
        if optimal_move not in legal_moves:
            continue

        root_value_f = float(root_value)
        encoded_state = encode_state(state, input_encoding=INPUT_ENCODING)

        if bucket == "holdout_candidate":
            holdout_candidate_ids.append(cid)
            continue

        if bucket == "production_candidate_later":
            policy = build_policy_target(optimal_move, legal_moves)
            row_data = {
                "candidate_id": cid,
                "canonical_state_hash": c_hash,
                "state": encoded_state,
                "raw_state": state,
                "policy": policy,
                "value": root_value_f,
                "legal_moves": legal_moves,
                "source": FAMILY,
                "label_source": "exact_tablebase",
                "role": "production_candidate_later",
                "train_only": True,
                "exclude_from_validation": True,
                "exact_optimal_move": optimal_move,
                "exact_root_value": root_value_f,
                "best_minus_second_best": best_minus_second,
                "row_mechanism": mechanism,
                "replay_role": "exact_tablebase_diagnostic",
                "bucket": "harder_fresh_endgame_tablebase",
                "family": FAMILY,
            }
            policy_value_rows.append(row_data)

        elif bucket == "value_only_candidate":
            row_data = {
                "candidate_id": cid,
                "canonical_state_hash": c_hash,
                "state": encoded_state,
                "raw_state": state,
                "value": root_value_f,
                "legal_moves": legal_moves,
                "source": FAMILY,
                "label_source": "exact_tablebase",
                "role": "value_only_candidate",
                "train_only": True,
                "exclude_from_validation": True,
                "exact_optimal_move": optimal_move,
                "exact_root_value": root_value_f,
                "best_minus_second_best": best_minus_second,
                "replay_role": "exact_tablebase_diagnostic_value_only",
                "policy_target_allowed": False,
                "reason": "tablebase exact value label, not policy target",
                "bucket": "harder_fresh_endgame_tablebase",
                "family": FAMILY,
            }
            value_only_rows.append(row_data)

        elif bucket == "preservation_control":
            policy = build_policy_target(optimal_move, legal_moves)
            row_data = {
                "candidate_id": cid,
                "canonical_state_hash": c_hash,
                "state": encoded_state,
                "raw_state": state,
                "policy": policy,
                "value": root_value_f,
                "legal_moves": legal_moves,
                "source": FAMILY,
                "label_source": "exact_tablebase",
                "role": "preservation_control",
                "train_only": True,
                "exclude_from_validation": True,
                "exact_optimal_move": optimal_move,
                "exact_root_value": root_value_f,
                "best_minus_second_best": best_minus_second,
                "replay_role": "exact_tablebase_diagnostic",
                "bucket": "harder_fresh_endgame_tablebase",
                "family": FAMILY,
            }
            controls_rows.append(row_data)

    summary = {
        "schema": "azlite_larger_harder_endgame_tablebase_diagnostic_artifact_v1",
        "family": FAMILY,
        "description": "Larger train-only exact-tablebase diagnostic artifact built from adversarial mining + PR #74 clean split.",
        "guardrails": {
            "mutated_active_fixture": False,
            "ran_training": False,
            "ran_arena": False,
            "promoted_model": False,
            "exhausted_family_excluded": True,
        },
        "input_encoding": INPUT_ENCODING,
        "artifacts": {
            "policy_value": {
                "path": str(OUTPUT_DIR / "exact_tablebase_policy_value_artifact.jsonl"),
                "row_count": len(policy_value_rows),
                "roles": ["production_candidate_later"],
                "target_types": ["policy", "value"],
                "policy_target_mass": OPTIMAL_POLICY_MASS,
                "value_source": "exact_tablebase",
            },
            "value_only": {
                "path": str(OUTPUT_DIR / "exact_tablebase_value_only_artifact.jsonl"),
                "row_count": len(value_only_rows),
                "roles": ["value_only_candidate"],
                "target_types": ["value"],
                "policy_target_allowed": False,
                "value_source": "exact_tablebase",
            },
            "controls": {
                "path": str(OUTPUT_DIR / "exact_tablebase_controls_artifact.jsonl"),
                "row_count": len(controls_rows),
                "roles": ["preservation_control"],
                "target_types": ["policy", "value"],
                "policy_target_mass": OPTIMAL_POLICY_MASS,
                "value_source": "exact_tablebase",
            },
        },
        "holdout_candidate_ids": holdout_candidate_ids,
        "counts": {
            "policy_value_rows": len(policy_value_rows),
            "value_only_rows": len(value_only_rows),
            "controls_rows": len(controls_rows),
            "holdout_count": len(holdout_candidate_ids),
        },
    }

    write_jsonl(
        OUTPUT_DIR / "exact_tablebase_policy_value_artifact.jsonl", policy_value_rows
    )
    write_jsonl(
        OUTPUT_DIR / "exact_tablebase_value_only_artifact.jsonl", value_only_rows
    )
    write_jsonl(OUTPUT_DIR / "exact_tablebase_controls_artifact.jsonl", controls_rows)

    errors = validate_artifacts(policy_value_rows, value_only_rows, controls_rows)
    if errors:
        print(f"\nValidation ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
        print("Artifact validation FAILED.")
        summary["validation"] = "FAILED"
        summary["validation_errors"] = errors
    else:
        print("\nStatic validation PASSED.")
        summary["validation"] = "PASSED"

    write_json(OUTPUT_DIR / "artifact_summary.json", summary)
    return summary


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("LARGER HARDER ENDGAME TABLEBASE DIAGNOSTIC ARTIFACT BUILD")
    print("=" * 70)

    # Load existing clean split and diagnostics
    print("\nLoading existing clean split...")
    existing_clean_split = json.loads(
        EXISTING_CLEAN_SPLIT_PATH.read_text(encoding="utf-8")
    )
    existing_split_rows = existing_clean_split.get("split_rows", [])
    existing_counts = existing_clean_split.get("counts", {})
    print(f"  Existing rows: {len(existing_split_rows)}")
    print(f"  Existing counts: {existing_counts}")

    print("\nLoading existing row diagnostics...")
    existing_diag_rows = load_jsonl(EXISTING_ROW_DIAGNOSTICS_PATH)
    existing_diag_by_cid = {r.get("candidate_id", ""): r for r in existing_diag_rows}
    print(f"  Existing diagnostic rows: {len(existing_diag_rows)}")

    # Collect existing canonical hashes for dedup
    existing_hashes: set[str] = set()
    for dr in existing_diag_rows:
        h = dr.get("canonical_state_hash", "")
        if h:
            existing_hashes.add(h)
    print(f"  Existing unique hashes: {len(existing_hashes)}")

    # Load suite inventory for dedup
    print("\nLoading fixture inventory...")
    suite_rows = load_suite(SUITE_PATH)
    suite_canon = suite_canonical_states(suite_rows)
    print(f"  Suite rows: {len(suite_rows)}")

    # Load artifact evaluator
    print("\nLoading artifact evaluator...")
    evaluator = ArtifactEvaluator(CURRENT_ARTIFACT)

    # ── Generate additional adversarial candidates ──
    print("\n=== Generating additional adversarial candidates ===")
    seeds = [SEED + i for i in range(500)]
    new_candidates_raw = generate_adversarial_candidates(samples=500, seeds=seeds)
    print(f"  Raw adversarial candidates: {len(new_candidates_raw)}")

    # Dedup against existing
    new_candidates = deduplicate_candidates(
        new_candidates_raw, existing_hashes, suite_canon, suite_rows
    )
    print(f"  New clean candidates after dedup: {len(new_candidates)}")

    if len(new_candidates) < 50:
        print(
            f"  WARNING: only {len(new_candidates)} new candidates, may not reach targets"
        )

    # ── PUCT baseline on new candidates ──
    print("\n=== PUCT baseline on new candidates ===")
    run_puct_baseline(evaluator, new_candidates)
    puct_classes = {}
    for c in new_candidates:
        pc = c.get("_puct_class", "")
        puct_classes[pc] = puct_classes.get(pc, 0) + 1
    print(f"  PUCT class distribution: {puct_classes}")

    # ── Neural value rank scan on new candidates ──
    print("\n=== Neural value rank scan on new candidates ===")
    run_neural_value_rank_scan(evaluator, new_candidates)
    nre = sum(1 for c in new_candidates if c.get("_neural_rank_error"))
    print(f"  Value rank errors: {nre}")

    # ── Classify new candidates ──
    print("\n=== Classifying new candidates ===")
    new_split = classify_candidates(new_candidates)
    print(f"  Targets: {new_split['target_candidate_count']}")
    print(f"  Controls: {new_split['preservation_control_count']}")
    print(f"  Holdouts: {new_split['holdout_count']}")
    print(f"  Low-budget: {new_split['low_budget_diagnostic_count']}")
    print(f"  Excluded: {new_split['excluded_count']}")

    # ── Build refined clean split ──
    print("\n=== Building refined clean split ===")
    refined_rows, refined_counts = build_refined_clean_split(
        existing_split_rows, existing_diag_by_cid, new_candidates, new_split, evaluator
    )
    print(f"  Refined split counts: {refined_counts}")

    # Check if we meet targets
    prod_count = refined_counts.get("production_candidate_later", 0)
    ctrl_count = refined_counts.get("preservation_control", 0)

    if prod_count < 30:
        print(f"\n  WARNING: production candidates ({prod_count}) < 30 target")
    if ctrl_count < 8:
        print(f"\n  WARNING: controls ({ctrl_count}) < 8 target")

    # ── Build artifacts ──
    print("\n=== Building artifacts ===")
    summary = build_artifacts(refined_rows)

    art = summary.get("artifacts", {})
    print("\nArtifact summary:")
    print(f"  Policy/value: {art.get('policy_value', {}).get('row_count', 0)} rows")
    print(f"  Value-only: {art.get('value_only', {}).get('row_count', 0)} rows")
    print(f"  Controls: {art.get('controls', {}).get('row_count', 0)} rows")
    print(f"  Holdouts: {len(summary.get('holdout_candidate_ids', []))}")

    decision_note = ""
    if prod_count < 30 or ctrl_count < 8:
        decision_note = "larger_exact_tablebase_artifact_not_enough_signal"
        print(f"\n  Classification: {decision_note}")
    else:
        print("\n  Artifacts ready for diagnostic traces")
        decision_note = "larger_exact_tablebase_artifact_ready"

    # Save refined clean split for trace runner
    refined_clean_split = {
        "schema": "azlite_larger_harder_endgame_tablebase_clean_split_v1",
        "family": FAMILY,
        "counts": refined_counts,
        "split_rows": refined_rows,
        "guardrails": {
            "mutated_active_fixture": False,
            "ran_training": False,
            "ran_arena": False,
            "promoted_model": False,
        },
    }
    write_json(
        OUTPUT_DIR / "harder_endgame_tablebase_clean_split.json", refined_clean_split
    )

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("  artifact_summary.json")
    print("  exact_tablebase_policy_value_artifact.jsonl")
    print("  exact_tablebase_value_only_artifact.jsonl")
    print("  exact_tablebase_controls_artifact.jsonl")
    print("  harder_endgame_tablebase_clean_split.json")
    print(f"\nDecision: {decision_note}")

    if prod_count < 5:
        print("ERROR: Fewer than 5 production candidates. Cannot proceed.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
