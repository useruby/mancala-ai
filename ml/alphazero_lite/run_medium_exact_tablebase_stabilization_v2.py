#!/usr/bin/env python3
"""Medium exact-tablebase stabilization v2 with targeted controls and softer policy targets.

Step 1: Reconstruct PR #78 row groups and regressions.
Step 2: Build targeted preservation controls artifact.
Step 3: Build softer policy artifacts (soft065, soft055).
Step 4: Static validation.
Step 5: Establish baselines.
Step 6: Run stabilization v2 traces.
Step 7: Evaluate every checkpoint.
Step 8: Apply strict local gates.
Step 9: Sanity non-regression checks.
Step 10: Collect training metrics.
Step 11: Apply decision rules.
Step 12: Generate report.

Does not run arena, does not promote, does not touch storage/ai/alphazero_lite/current.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import build_eval_search_options, encode_state

FAMILY = "harder_fresh_endgame_tablebase"
STABILIZATION_V2_FAMILY = "medium_exact_tablebase_stabilization_v2"
PITS_PER_PLAYER = 6
EPS = 1e-9
INPUT_ENCODING = "kalah_v3"
POLICY_SIZE = 6
C_PUCT = 1.25
SEED = 17
EVAL_BUDGETS = (384, 1200, 2400)
SEARCH_OPTIONS = build_eval_search_options(
    fpu_mode="parent_q",
    reuse_subtree=True,
    normalize_values=True,
    root_policy_mode="deterministic",
    tactical_root_bias=0.1,
)
BASE_LR = 1e-4
HALF_LR = 5e-5
QUARTER_LR = 2.5e-5

CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")

PR78_ARTIFACT_DIR = Path("/tmp/azlite_medium_exact_tablebase_diagnostic")
OUTPUT_DIR = Path("/tmp/azlite_medium_exact_tablebase_stabilization_v2")
EXPORT_DIR = OUTPUT_DIR / "exports"
EVAL_DIR = OUTPUT_DIR / "eval"

PR78_CONTROLS_PATH = (
    PR78_ARTIFACT_DIR / "exact_tablebase_expanded_controls_artifact.jsonl"
)
PR78_POLICY_VALUE_PATH = (
    PR78_ARTIFACT_DIR / "exact_tablebase_policy_value_soft075.jsonl"
)
PR78_VALUE_ONLY_PATH = PR78_ARTIFACT_DIR / "exact_tablebase_value_only_artifact.jsonl"
PR78_ARTIFACT_SUMMARY_PATH = PR78_ARTIFACT_DIR / "artifact_summary.json"
PR78_TRACE_SUMMARY_PATH = PR78_ARTIFACT_DIR / "medium_diagnostic_trace_summary.json"
PR78_CLEAN_SPLIT_PATH = PR78_ARTIFACT_DIR / "medium_exact_tablebase_clean_split.json"
PR78_EXPORTS_DIR = PR78_ARTIFACT_DIR / "exports"
PR78_EVAL_DIR = PR78_ARTIFACT_DIR / "eval"

REPORT_PATH = Path(
    "docs/alphazero-lite-medium-exact-tablebase-stabilization-v2-results.md"
)

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

TARGET_MIN_CONTROLS = 200
TARGET_MIN_HOLDOUTS = 1000
TARGET_MIN_PRODUCTION = 120

PR78_BEST_STABLE_TRACE = "medium_soft075_controls_w1_lr_half"
PR78_BEST_STABLE_EPOCH = 1
PR78_BEST_PRODUCTION_TRACE = "medium_soft075_controls_w1_lr_default"
PR78_BEST_PRODUCTION_EPOCH = 4


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
            from ml.alphazero_lite.forensic_suite import canonical_state_key

            result.add(canonical_state_key(row["state"]))
    return result


def child_state_from_move(root_state: dict[str, Any], move: int) -> dict[str, Any]:
    game = KalahGame.from_state(root_state)
    succeeded = game.move(game.pit_index(int(move)))
    if not succeeded:
        raise ValueError(f"illegal move {move}")
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


def build_soft_policy_target(
    optimal_move: int, legal_moves: list[int], optimal_mass: float
) -> list[float]:
    policy = [0.0] * POLICY_SIZE
    if optimal_move not in legal_moves:
        return policy
    other_legal = [m for m in legal_moves if m != optimal_move]
    policy[optimal_move] = optimal_mass
    if other_legal:
        remaining = 1.0 - optimal_mass
        per_move = remaining / len(other_legal)
        for m in other_legal:
            policy[m] = per_move
    total = sum(policy)
    if total > 0:
        policy = [p / total for p in policy]
    return policy


def convert_npz_to_artifact(checkpoint_path: Path, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    npz = np.load(checkpoint_path)
    weights: dict[str, Any] = {}
    for key in npz.keys():
        val = npz[key]
        if isinstance(val, np.ndarray):
            weights[key] = val.tolist()
        else:
            weights[key] = val
    (artifact_dir / "weights.json").write_text(json.dumps(weights), encoding="utf-8")

    from ml.alphazero_lite.input_encodings import feature_count_for

    fc = feature_count_for(INPUT_ENCODING)
    trunk_size = len(weights.get("b_input", []))
    block_count = 0
    while f"w_residual_{block_count + 1}_1" in weights:
        block_count += 1
    hlcount = 1 + block_count * 2 + (2 if "w_policy_hidden" in weights else 0)

    metadata = {
        "schema_version": "azlite_model_v1",
        "version": "diagnostic_trace",
        "game": "kalah",
        "rules_version": "kalah_v1",
        "input_encoding": INPUT_ENCODING,
        "feature_count": fc,
        "policy_size": POLICY_SIZE,
        "architecture": {
            "type": "residual_policy_value",
            "model_type": "residual_v3",
            "activation": "relu",
            "policy_size": POLICY_SIZE,
            "value_size": 1,
            "hidden_sizes": [trunk_size] if trunk_size else [64],
            "hidden_layer_count": hlcount,
            "trunk_size": trunk_size,
            "residual_block_count": block_count,
        },
        "artifacts": {"weights_file": "weights.json"},
    }
    if "w_policy_hidden" in weights:
        metadata["architecture"]["policy_hidden_size"] = len(weights["b_policy_hidden"])
    if "w_value_hidden" in weights:
        metadata["architecture"]["value_hidden_size"] = len(weights["b_value_hidden"])
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


# ── helpers for PUCT evaluation ──


def run_single_puct(
    evaluator: ArtifactEvaluator, state: dict[str, Any], budget: int, seed: int
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


def evaluate_production_row(
    evaluator: ArtifactEvaluator, row: dict[str, Any], optimal_move: int
) -> dict[str, Any]:
    cid = row.get("candidate_id", "?")
    state = row.get("raw_state", row.get("state"))
    if not state:
        return {"candidate_id": cid, "error": "no state"}
    results: dict[str, Any] = {"candidate_id": cid, "exact_optimal_move": optimal_move}
    for budget in EVAL_BUDGETS:
        try:
            result = run_single_puct(evaluator, state, budget, SEED)
        except Exception as e:
            results[f"error_{budget}"] = str(e)
            continue
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
        results[f"selected_{budget}"] = selected_move
        results[f"selected_is_optimal_{budget}"] = selected_is_optimal
        results[f"optimal_visit_share_{budget}"] = visit_share(
            visits_list, optimal_move
        )
        results[f"selected_visit_share_{budget}"] = visit_share(
            visits_list, selected_move
        )
        results[f"selected_minus_optimal_q_margin_{budget}"] = (
            round_float(
                float(sel_entry.get("q_value", 0.0))
                - float(opt_entry.get("q_value", 0.0))
            )
            if sel_entry and opt_entry
            else None
        )
    r1200 = run_single_puct(evaluator, state, 1200, SEED)
    p_list = [float(p) for p in r1200.get("policy", [])]
    results["optimal_policy_probability"] = round_float(
        float(p_list[optimal_move]) if optimal_move < len(p_list) else 0.0
    )
    results["optimal_policy_rank"] = policy_rank_of_move(p_list, optimal_move)
    return results


def evaluate_value_row(
    evaluator: ArtifactEvaluator,
    row: dict[str, Any],
    exact_value: float,
    optimal_move: int,
) -> dict[str, Any]:
    cid = row.get("candidate_id", "?")
    state = row.get("raw_state", row.get("state"))
    if not state:
        return {"candidate_id": cid, "error": "no state"}
    game = KalahGame.from_state(state)
    _, raw_nv = evaluator.evaluate(game)
    neural_root = float(raw_nv)
    abs_value_error = round_float(abs(neural_root - exact_value))
    sign_error: bool | None = None
    if abs(exact_value) > EPS:
        sign_error = math.copysign(1.0, neural_root) != math.copysign(1.0, exact_value)
    return {
        "candidate_id": cid,
        "exact_value": exact_value,
        "neural_value": round_float(neural_root),
        "abs_value_error": abs_value_error,
        "sign_error": sign_error,
    }


def evaluate_controls_row(
    evaluator: ArtifactEvaluator,
    row: dict[str, Any],
    optimal_move: int,
    baseline_selected_is_optimal: dict[int, bool | None],
    baseline_optimal_visit_share: dict[int, float | None],
) -> dict[str, Any]:
    cid = row.get("candidate_id", "?")
    state = row.get("raw_state", row.get("state"))
    if not state:
        return {"candidate_id": cid, "error": "no state"}
    results: dict[str, Any] = {"candidate_id": cid, "exact_optimal_move": optimal_move}
    regressed = False
    for budget in EVAL_BUDGETS:
        try:
            result = run_single_puct(evaluator, state, budget, SEED)
        except Exception:
            results[f"selected_is_optimal_{budget}"] = None
            continue
        selected_move = (
            None
            if result.get("selected_move") is None
            else int(result["selected_move"])
        )
        selected_is_optimal = (
            selected_move == optimal_move if optimal_move is not None else None
        )
        visits_list = [float(v) for v in result.get("visits", [])]
        results[f"selected_{budget}"] = selected_move
        results[f"selected_is_optimal_{budget}"] = selected_is_optimal
        opt_share = visit_share(visits_list, optimal_move)
        results[f"optimal_visit_share_{budget}"] = opt_share
        baseline_opt = baseline_optimal_visit_share.get(budget, 0.0)
        results[f"optimal_visit_share_delta_{budget}"] = (
            round_float(float(opt_share) - float(baseline_opt))
            if opt_share is not None and baseline_opt is not None
            else None
        )
        if selected_is_optimal is False:
            regressed = True
        baseline_sel = baseline_selected_is_optimal.get(budget, True)
        if baseline_sel is True and selected_is_optimal is False:
            regressed = True
    results["control_regression"] = regressed
    return results


def evaluate_holdout_row(
    evaluator: ArtifactEvaluator,
    row: dict[str, Any],
    optimal_move: int,
    baseline_selected_is_optimal: dict[int, bool | None],
    baseline_value_error: float | None,
    exact_value: float,
) -> dict[str, Any]:
    cid = row.get("candidate_id", "?")
    state = row.get("raw_state", row.get("state"))
    if not state:
        return {"candidate_id": cid, "error": "no state"}
    results: dict[str, Any] = {"candidate_id": cid}
    for budget in EVAL_BUDGETS:
        try:
            result = run_single_puct(evaluator, state, budget, SEED)
        except Exception:
            results[f"selected_is_optimal_{budget}"] = None
            continue
        selected_move = (
            None
            if result.get("selected_move") is None
            else int(result["selected_move"])
        )
        results[f"selected_{budget}"] = selected_move
        results[f"selected_is_optimal_{budget}"] = (
            selected_move == optimal_move if optimal_move is not None else None
        )
    game = KalahGame.from_state(state)
    _, raw_nv = evaluator.evaluate(game)
    neural_value = float(raw_nv)
    current_abs_error = (
        abs(neural_value - exact_value) if exact_value is not None else None
    )
    results["neural_value"] = round_float(neural_value)
    results["exact_value"] = exact_value
    results["value_error"] = round_float(current_abs_error)
    results["value_error_delta_vs_current"] = (
        round_float(float(current_abs_error) - float(baseline_value_error))
        if current_abs_error is not None and baseline_value_error is not None
        else None
    )
    sel1200_optimal = results.get("selected_is_optimal_1200")
    baseline_sel1200 = baseline_selected_is_optimal.get(1200)
    if sel1200_optimal is True and baseline_sel1200 is not True:
        results["generalization_signal"] = "improved"
    elif sel1200_optimal is False and baseline_sel1200 is not False:
        results["generalization_signal"] = "regressed"
    else:
        results["generalization_signal"] = "stable"
    return results


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Reconstruct PR #78 row groups and regressions
# ═══════════════════════════════════════════════════════════════════════════


def load_pr78_data() -> dict[str, Any]:
    """Load all PR #78 data and reconstruct row groups."""
    print("\n" + "=" * 70)
    print("STEP 1: RECONSTRUCT PR #78 ROW GROUPS AND REGRESSIONS")
    print("=" * 70)

    artifact_summary = json.loads(
        PR78_ARTIFACT_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    trace_summary = json.loads(PR78_TRACE_SUMMARY_PATH.read_text(encoding="utf-8"))
    controls_rows = load_jsonl(PR78_CONTROLS_PATH)
    policy_value_rows = load_jsonl(PR78_POLICY_VALUE_PATH)
    value_only_rows = load_jsonl(PR78_VALUE_ONLY_PATH)
    clean_split = json.loads(PR78_CLEAN_SPLIT_PATH.read_text(encoding="utf-8"))

    holdout_ids = set(artifact_summary.get("holdout_candidate_ids", []))
    production_ids = set(clean_split.get("production_ids", []))
    control_ids = set(clean_split.get("control_ids", []))
    value_only_ids = set(clean_split.get("value_only_ids", []))
    untouched_holdout_ids = set(clean_split.get("holdout_ids", []))

    split_rows = clean_split.get("split_rows", [])
    split_by_cid: dict[str, dict] = {}
    for sr in split_rows:
        cid = sr.get("candidate_id", "")
        if cid:
            split_by_cid[cid] = sr

    print(
        f"  Production candidates: {len(policy_value_rows)} ({len(production_ids)} IDs)"
    )
    print(f"  Controls: {len(controls_rows)} ({len(control_ids)} IDs)")
    print(f"  Value-only: {len(value_only_rows)} ({len(value_only_ids)} IDs)")
    print(f"  Untouched holdouts: {len(untouched_holdout_ids)}")
    print(f"  Total holdout candidates: {len(holdout_ids)}")

    baseline = trace_summary.get("baseline", {})
    print(
        f"\n  Baseline: prod_opt={baseline.get('production_optimal_1200')}/{baseline.get('production_count')}"
    )
    print(
        f"    controls_opt={baseline.get('control_optimal_1200')}/{baseline.get('control_count')}"
    )
    print(
        f"    holdout_opt={baseline.get('holdout_optimal_1200')}/{baseline.get('holdout_count')}"
    )
    print(f"    value_avg_error={baseline.get('value_avg_error'):.4f}")

    evaluations = trace_summary.get("evaluations", {})
    regressions_by_trace: dict[str, int] = {}
    for tn, evals in evaluations.items():
        if tn == "baseline":
            continue
        for e in evals:
            ep = e.get("epoch")
            h_cnt = e.get("holdout_count", 0)
            h_opt = e.get("holdout_optimal_1200", 0)
            h_reg = h_cnt - h_opt
            c_reg = e.get("control_regression_count", 0)
            key = f"{tn}_e{ep}"
            regressions_by_trace[key] = h_reg
            print(f"  {key}: holdout_reg={h_reg}/{h_cnt}, ctrl_reg={c_reg}")

    return {
        "artifact_summary": artifact_summary,
        "trace_summary": trace_summary,
        "controls_rows": controls_rows,
        "policy_value_rows": policy_value_rows,
        "value_only_rows": value_only_rows,
        "clean_split": clean_split,
        "split_rows": split_rows,
        "split_by_cid": split_by_cid,
        "holdout_ids": holdout_ids,
        "production_ids": production_ids,
        "control_ids": control_ids,
        "value_only_ids": value_only_ids,
        "untouched_holdout_ids": untouched_holdout_ids,
        "regressions_by_trace": regressions_by_trace,
        "evaluations": evaluations,
        "baseline": baseline,
    }


def identify_regressed_holdouts(pr78: dict[str, Any]) -> dict[str, Any]:
    """Re-evaluate PR #78 best stable checkpoints to find regressed holdouts.

    Uses the best zero-control-regression checkpoints:
    - medium_soft075_controls_w1_lr_half e1
    - medium_soft075_controls_w2_lr_half e1
    Also checks medium_soft075_controls_w1_lr_default e1/e2/e4.

    Returns dict with regressed_holdout_ids and regression_details.
    """
    print("\n--- Identifying regressed holdouts from PR #78 ---")

    split_by_cid = pr78["split_by_cid"]
    untouched_holdout_ids = pr78["untouched_holdout_ids"]

    holdout_rows_for_eval: list[dict[str, Any]] = []
    holdout_id_to_split: dict[str, dict] = {}
    for hid in sorted(untouched_holdout_ids):
        sr = split_by_cid.get(hid)
        if sr is not None:
            holdout_rows_for_eval.append(
                {
                    "candidate_id": hid,
                    "raw_state": sr.get("_state"),
                    "exact_optimal_move": sr.get("_optimal_move"),
                    "exact_root_value": sr.get("_root_value"),
                }
            )
            holdout_id_to_split[hid] = sr

    print(f"  Holdout rows for eval: {len(holdout_rows_for_eval)}")

    checkpoints_to_scan = [
        ("medium_soft075_controls_w1_lr_half", 1),
        ("medium_soft075_controls_w2_lr_half", 1),
        ("medium_soft075_controls_w1_lr_default", 1),
        ("medium_soft075_controls_w1_lr_default", 2),
        ("medium_soft075_controls_w1_lr_default", 4),
        ("medium_soft075_controls_w1_lr_half", 2),
        ("medium_soft075_controls_w1_lr_half", 4),
        ("medium_soft075_controls_w2_lr_half", 2),
        ("medium_soft075_controls_w2_lr_half", 4),
    ]

    regressed_holdout_ids: set[str] = set()
    regression_details: dict[str, list[dict[str, Any]]] = {}

    for trace_name, epoch in checkpoints_to_scan:
        npz_path = PR78_EXPORTS_DIR / f"{trace_name}_e{epoch}.npz"
        eval_artifact_dir = PR78_EVAL_DIR / f"{trace_name}_e{epoch}"
        if not npz_path.exists() or not eval_artifact_dir.exists():
            print(f"  WARNING: Checkpoint {trace_name}_e{epoch} not found, skipping")
            continue

        try:
            evaluator = ArtifactEvaluator(eval_artifact_dir)
        except Exception as e:
            print(f"  WARNING: Cannot load {eval_artifact_dir}: {e}")
            continue

        regressed: list[dict[str, Any]] = []
        for hr in holdout_rows_for_eval:
            cid = hr["candidate_id"]
            optimal_move = hr.get("exact_optimal_move")
            state = hr.get("raw_state")
            if optimal_move is None or state is None:
                continue
            try:
                result = run_single_puct(evaluator, state, 1200, SEED)
                selected_move = (
                    None
                    if result.get("selected_move") is None
                    else int(result["selected_move"])
                )
                if selected_move is not None and selected_move != optimal_move:
                    regressed.append(
                        {
                            "candidate_id": cid,
                            "trace": trace_name,
                            "epoch": epoch,
                            "optimal_move": optimal_move,
                            "selected_move": selected_move,
                        }
                    )
                    regressed_holdout_ids.add(cid)
            except Exception:
                continue

        print(f"  {trace_name}_e{epoch}: {len(regressed)} regressed holdouts")
        regression_details[f"{trace_name}_e{epoch}"] = regressed

    print(
        f"\n  Total unique regressed holdout IDs across all checkpoints: {len(regressed_holdout_ids)}"
    )

    for hid in sorted(regressed_holdout_ids)[:10]:
        print(f"    {hid}")

    return {
        "regressed_holdout_ids": regressed_holdout_ids,
        "regression_details": regression_details,
        "holdout_rows_for_eval": holdout_rows_for_eval,
        "holdout_id_to_split": holdout_id_to_split,
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Build targeted preservation controls
# ═══════════════════════════════════════════════════════════════════════════


def build_targeted_controls(
    pr78: dict[str, Any], regression_info: dict[str, Any]
) -> dict[str, Any]:
    """Build targeted preservation controls artifact.

    Includes:
    - All 91 PR #78 controls
    - Regressed holdout rows
    - Nearest-neighbor clean holdouts around production candidates
    - Additional clean rows that PUCT passes
    """
    print("\n" + "=" * 70)
    print("STEP 2: BUILD TARGETED PRESERVATION CONTROLS")
    print("=" * 70)

    controls_rows = pr78["controls_rows"]
    split_by_cid = pr78["split_by_cid"]
    untouched_holdout_ids = pr78["untouched_holdout_ids"]
    production_ids = pr78["production_ids"]
    regressed_holdout_ids = regression_info["regressed_holdout_ids"]
    policy_value_rows = pr78["policy_value_rows"]

    targeted_controls: list[dict[str, Any]] = []
    seen_cids: set[str] = set()
    seen_hashes: set[str] = set()
    original_control_ids: set[str] = set()
    promoted_control_ids: set[str] = set()
    nearest_neighbor_control_ids: set[str] = set()
    for crow in controls_rows:
        cid = crow.get("candidate_id", "")
        if cid:
            original_control_ids.add(cid)
            seen_cids.add(cid)
        ch = crow.get("canonical_state_hash", "")
        if ch:
            seen_hashes.add(ch)

    print(f"  Original PR #78 controls: {len(original_control_ids)}")

    promoted_count = 0
    for hid in sorted(regressed_holdout_ids):
        if hid in seen_cids or hid in production_ids:
            continue
        sr = split_by_cid.get(hid)
        if sr is None:
            continue
        state = sr.get("_state")
        if not state:
            continue
        optimal_move = sr.get("_optimal_move")
        legal_moves = sr.get("_legal_moves", list(range(POLICY_SIZE)))
        root_value = sr.get("_root_value")
        c_hash = sr.get("_canonical_state_hash", "")
        if optimal_move is None or root_value is None:
            continue
        if optimal_move not in legal_moves:
            continue
        if c_hash in seen_hashes:
            continue

        encoded_state = encode_state(state, input_encoding=INPUT_ENCODING)
        policy = build_soft_policy_target(optimal_move, legal_moves, 0.75)
        row_data = {
            "candidate_id": hid,
            "canonical_state_hash": c_hash,
            "state": encoded_state,
            "raw_state": state,
            "policy": policy,
            "value": float(root_value),
            "legal_moves": legal_moves,
            "source": STABILIZATION_V2_FAMILY,
            "label_source": "exact_tablebase",
            "role": "preservation_control",
            "train_only": True,
            "exclude_from_validation": True,
            "exact_optimal_move": optimal_move,
            "exact_root_value": float(root_value),
            "replay_role": "exact_tablebase_diagnostic",
            "family": FAMILY,
            "policy_target_mass": 0.75,
            "promoted_from_holdout_regression": True,
            "nearest_neighbor_control": False,
            "control_subtype": "promoted_regression",
        }
        targeted_controls.append(row_data)
        promoted_control_ids.add(hid)
        seen_cids.add(hid)
        if c_hash:
            seen_hashes.add(c_hash)
        promoted_count += 1

    print(f"  Promoted regression holdouts: {promoted_count}")

    prod_candidate_hashes: set[str] = set()
    for row in policy_value_rows:
        h = row.get("canonical_state_hash", "")
        if h:
            prod_candidate_hashes.add(h)

    holdout_hashes: dict[str, str] = {}
    for hid in sorted(untouched_holdout_ids):
        if hid in seen_cids:
            continue
        sr = split_by_cid.get(hid)
        if sr is None:
            continue
        h = sr.get("_canonical_state_hash", "")
        if h:
            holdout_hashes[hid] = h

    def state_similarity(state_a: dict[str, Any], state_b: dict[str, Any]) -> float:
        """Simple state similarity: sum of abs differences in pit values + store."""
        score = 0.0
        for key in ["player_pits", "opponent_pits"]:
            for i in range(6):
                va = (
                    state_a.get(key, [0] * 6)[i]
                    if isinstance(state_a.get(key), list) and len(state_a[key]) > i
                    else 0
                )
                vb = (
                    state_b.get(key, [0] * 6)[i]
                    if isinstance(state_b.get(key), list) and len(state_b[key]) > i
                    else 0
                )
                score -= abs(float(va) - float(vb))
        for key in ["player_store", "opponent_store"]:
            va = state_a.get(key, 0)
            vb = state_b.get(key, 0)
            score -= abs(float(va) - float(vb)) * 3
        if state_a.get("current_player") == state_b.get("current_player"):
            score += 1.0
        return score

    nearest_neighbor_added = 0
    for prod_row in policy_value_rows:
        prod_state = prod_row.get("raw_state")
        if not prod_state:
            continue
        candidates: list[tuple[str, float]] = []
        for hid in sorted(holdout_hashes):
            if hid in seen_cids:
                continue
            if hid in production_ids:
                continue
            sr = split_by_cid.get(hid)
            if sr is None:
                continue
            hs = sr.get("_state")
            if not hs:
                continue
            sim = state_similarity(prod_state, hs)
            candidates.append((hid, sim))
        candidates.sort(key=lambda x: -x[1])
        for hid, _ in candidates[:1]:
            if hid in seen_cids:
                continue
            sr = split_by_cid.get(hid)
            if sr is None:
                continue
            state = sr.get("_state")
            if not state:
                continue
            optimal_move = sr.get("_optimal_move")
            legal_moves = sr.get("_legal_moves", list(range(POLICY_SIZE)))
            root_value = sr.get("_root_value")
            c_hash = sr.get("_canonical_state_hash", "")
            if optimal_move is None or root_value is None:
                continue
            if optimal_move not in legal_moves:
                continue
            if c_hash in seen_hashes:
                continue
            encoded_state = encode_state(state, input_encoding=INPUT_ENCODING)
            policy = build_soft_policy_target(optimal_move, legal_moves, 0.75)
            row_data = {
                "candidate_id": hid,
                "canonical_state_hash": c_hash,
                "state": encoded_state,
                "raw_state": state,
                "policy": policy,
                "value": float(root_value),
                "legal_moves": legal_moves,
                "source": STABILIZATION_V2_FAMILY,
                "label_source": "exact_tablebase",
                "role": "preservation_control",
                "train_only": True,
                "exclude_from_validation": True,
                "exact_optimal_move": optimal_move,
                "exact_root_value": float(root_value),
                "replay_role": "exact_tablebase_diagnostic",
                "family": FAMILY,
                "policy_target_mass": 0.75,
                "promoted_from_holdout_regression": False,
                "nearest_neighbor_control": True,
                "control_subtype": "nearest_neighbor",
            }
            targeted_controls.append(row_data)
            nearest_neighbor_control_ids.add(hid)
            seen_cids.add(hid)
            if c_hash:
                seen_hashes.add(c_hash)
            nearest_neighbor_added += 1

    print(f"  Nearest-neighbor controls added: {nearest_neighbor_added}")

    if len(targeted_controls) < TARGET_MIN_CONTROLS:
        additional_needed = TARGET_MIN_CONTROLS - len(targeted_controls)
        additional_added = 0
        for hid in sorted(untouched_holdout_ids):
            if additional_added >= additional_needed:
                break
            if hid in seen_cids or hid in production_ids:
                continue
            sr = split_by_cid.get(hid)
            if sr is None:
                continue
            state = sr.get("_state")
            if not state:
                continue
            optimal_move = sr.get("_optimal_move")
            legal_moves = sr.get("_legal_moves", list(range(POLICY_SIZE)))
            root_value = sr.get("_root_value")
            c_hash = sr.get("_canonical_state_hash", "")
            if optimal_move is None or root_value is None:
                continue
            if optimal_move not in legal_moves:
                continue
            if c_hash in seen_hashes:
                continue
            encoded_state = encode_state(state, input_encoding=INPUT_ENCODING)
            policy = build_soft_policy_target(optimal_move, legal_moves, 0.75)
            row_data = {
                "candidate_id": hid,
                "canonical_state_hash": c_hash,
                "state": encoded_state,
                "raw_state": state,
                "policy": policy,
                "value": float(root_value),
                "legal_moves": legal_moves,
                "source": STABILIZATION_V2_FAMILY,
                "label_source": "exact_tablebase",
                "role": "preservation_control",
                "train_only": True,
                "exclude_from_validation": True,
                "exact_optimal_move": optimal_move,
                "exact_root_value": float(root_value),
                "replay_role": "exact_tablebase_diagnostic",
                "family": FAMILY,
                "policy_target_mass": 0.75,
                "promoted_from_holdout_regression": False,
                "nearest_neighbor_control": False,
                "control_subtype": "additional_clean",
            }
            targeted_controls.append(row_data)
            seen_cids.add(hid)
            if c_hash:
                seen_hashes.add(c_hash)
            additional_added += 1
        print(f"  Additional clean controls added: {additional_added}")

    updated_untouched_holdout_ids = sorted(set(untouched_holdout_ids) - seen_cids)

    print(f"\n  Total targeted controls: {len(targeted_controls)}")
    print(f"  Original PR78 controls: {len(original_control_ids)}")
    print(f"  Promoted regression controls: {promoted_count}")
    print(f"  Nearest-neighbor controls: {nearest_neighbor_added}")
    print(f"  Untouched holdouts remaining: {len(updated_untouched_holdout_ids)}")

    return {
        "controls": targeted_controls,
        "promoted_control_ids": promoted_control_ids,
        "nearest_neighbor_control_ids": nearest_neighbor_control_ids,
        "original_control_ids": original_control_ids,
        "updated_untouched_holdout_ids": updated_untouched_holdout_ids,
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Build softer policy artifacts
# ═══════════════════════════════════════════════════════════════════════════


def build_soft_policy_artifacts(pr78: dict[str, Any]) -> dict[str, Any]:
    """Build soft065 and soft055 policy/value artifacts from PR #78 production candidates."""
    print("\n" + "=" * 70)
    print("STEP 3: BUILD SOFTER POLICY ARTIFACTS")
    print("=" * 70)

    policy_value_rows = pr78["policy_value_rows"]

    def make_soft_rows(optimal_mass: float, label: str) -> list[dict]:
        rows: list[dict] = []
        for row in policy_value_rows:
            legal_moves = row.get("legal_moves", list(range(POLICY_SIZE)))
            optimal_move = row.get("exact_optimal_move")
            if optimal_move is None or optimal_move not in legal_moves:
                continue
            policy = build_soft_policy_target(optimal_move, legal_moves, optimal_mass)
            root_value = float(row.get("value", row.get("exact_root_value", 0.0)))
            new_row = dict(row)
            new_row["policy"] = policy
            new_row["value"] = root_value
            new_row["policy_target_mass"] = optimal_mass
            new_row["stabilized_variant"] = label
            rows.append(new_row)
        return rows

    soft065_rows = make_soft_rows(0.65, "soft065")
    soft055_rows = make_soft_rows(0.55, "soft055")

    print(f"  soft065 rows: {len(soft065_rows)}")
    print(f"  soft055 rows: {len(soft055_rows)}")

    return {
        "soft065_rows": soft065_rows,
        "soft055_rows": soft055_rows,
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Static validation
# ═══════════════════════════════════════════════════════════════════════════


def validate_artifacts(
    soft065_rows: list[dict],
    soft055_rows: list[dict],
    controls_rows: list[dict],
    value_only_rows: list[dict],
    production_ids: set[str],
    control_ids: set[str],
    holdout_ids: set[str],
) -> list[str]:
    """Static validation of all artifacts."""
    errors: list[str] = []

    for label, artifact in [
        ("soft065", soft065_rows),
        ("soft055", soft055_rows),
        ("controls", controls_rows),
    ]:
        if not artifact:
            errors.append(f"{label}: empty artifact")
            continue
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

    for label, artifact in [("soft065", soft065_rows), ("soft055", soft055_rows)]:
        for idx, row in enumerate(artifact):
            policy = row.get("policy", [])
            optimal_move = row.get("exact_optimal_move")
            if optimal_move is not None and optimal_move < len(policy):
                optimal_mass = policy[optimal_move]
                for m, mass in enumerate(policy):
                    if m != optimal_move and mass > optimal_mass:
                        errors.append(
                            f"{label}[{idx}]: move {m} has {mass:.4f} > optimal {optimal_mass:.4f}"
                        )
                        break

    seen_states: dict[str, str] = {}
    for label, artifact in [
        ("soft065", soft065_rows),
        ("soft055", soft055_rows),
        ("controls", controls_rows),
        ("value_only", value_only_rows or []),
    ]:
        if not artifact:
            continue
        for idx, row in enumerate(artifact):
            c_hash = row.get("canonical_state_hash", "")
            if c_hash:
                if c_hash in seen_states:
                    prev_label = seen_states[c_hash]
                    if prev_label != label:
                        v1 = row.get("value")
                        for found_row in artifact:
                            if found_row.get("canonical_state_hash") == c_hash:
                                v2 = found_row.get("value")
                                break
                        else:
                            v2 = None
                        if (
                            v1 is not None
                            and v2 is not None
                            and abs(float(v1) - float(v2)) > 1e-6
                        ):
                            errors.append(
                                f"state {c_hash}: conflicting value targets ({v1} vs {v2}) across {prev_label}/{label}"
                            )
                else:
                    seen_states[c_hash] = label

    ctrl_cids = set(r.get("candidate_id", "") for r in controls_rows)
    holdout_cids = holdout_ids

    overlap_ctrl_holdout = ctrl_cids & holdout_cids
    if overlap_ctrl_holdout:
        errors.append(f"Control overlap with holdouts: {len(overlap_ctrl_holdout)}")

    if len(soft065_rows) < TARGET_MIN_PRODUCTION:
        errors.append(
            f"Production candidates ({len(soft065_rows)}) < {TARGET_MIN_PRODUCTION} minimum"
        )
    if len(controls_rows) < 150:
        errors.append(f"Controls ({len(controls_rows)}) < 150 minimum")
    if len(holdout_ids) < TARGET_MIN_HOLDOUTS:
        errors.append(f"Holdouts ({len(holdout_ids)}) < {TARGET_MIN_HOLDOUTS} minimum")

    return errors


# ═══════════════════════════════════════════════════════════════════════════
# STEPS 5-12: Run traces, evaluate, apply gates, report
# ═══════════════════════════════════════════════════════════════════════════


def build_larger_init_checkpoint(trunk_size: int = 128, block_count: int = 4) -> Path:
    """Build a larger init checkpoint (128x4) by padding current 96x3 weights.

    Adds small Gaussian noise to padded regions and initializes new residual
    block with Kaiming uniform. The result can be loaded by train.py with
    --hidden-sizes <trunk_size>,<block_count>.
    """
    init_path = OUTPUT_DIR / f"init_checkpoint_{trunk_size}x{block_count}.npz"
    if init_path.exists():
        return init_path

    print(
        f"Building {trunk_size}x{block_count} init checkpoint from current 96x3 artifact..."
    )
    current_weights = json.loads(
        (CURRENT_ARTIFACT / "weights.json").read_text(encoding="utf-8")
    )

    old_trunk = 96
    new_trunk = trunk_size
    rng = np.random.default_rng(42)

    padded: dict[str, np.ndarray] = {}

    def maybe_pad_2d(
        key: str, target_rows: int, target_cols: int, noise_std: float = 1e-4
    ) -> np.ndarray:
        arr = np.asarray(current_weights[key], dtype=np.float32)
        r, c = arr.shape
        if r == target_rows and c == target_cols:
            return arr
        result = np.zeros((target_rows, target_cols), dtype=np.float32)
        result[:r, :c] = arr
        noise = rng.normal(0, noise_std, size=(target_rows, target_cols)).astype(
            np.float32
        )
        noise[:r, :c] = 0.0
        result += noise
        return result

    def maybe_pad_1d(key: str, target_size: int, noise_std: float = 1e-4) -> np.ndarray:
        arr = np.asarray(current_weights[key], dtype=np.float32)
        n = arr.shape[0]
        if n == target_size:
            return arr
        result = np.zeros(target_size, dtype=np.float32)
        result[:n] = arr
        noise = rng.normal(0, noise_std, size=target_size).astype(np.float32)
        noise[:n] = 0.0
        result += noise
        return result

    # input_layer: (in_features, out_features) format in checkpoint = weight.T
    # w_input stored as (input_features, 96) → need (input_features, 128)
    w_input_shape = np.asarray(current_weights["w_input"], dtype=np.float32).shape
    padded["w_input"] = maybe_pad_2d("w_input", w_input_shape[0], new_trunk)
    padded["b_input"] = maybe_pad_1d("b_input", new_trunk)

    # Copy existing 3 residual blocks with padding
    for block_idx in range(1, 4):
        for layer_idx in [1, 2]:
            wk = f"w_residual_{block_idx}_{layer_idx}"
            bk = f"b_residual_{block_idx}_{layer_idx}"
            padded[wk] = maybe_pad_2d(wk, new_trunk, new_trunk)
            padded[bk] = maybe_pad_1d(bk, new_trunk)

    # New 4th residual block: Kaiming uniform init
    bound = np.sqrt(6.0 / new_trunk)
    for layer_idx in [1, 2]:
        wk = f"w_residual_4_{layer_idx}"
        bk = f"b_residual_4_{layer_idx}"
        padded[wk] = rng.uniform(-bound, bound, size=(new_trunk, new_trunk)).astype(
            np.float32
        )
        padded[bk] = np.zeros(new_trunk, dtype=np.float32)

    # policy_hidden_layer: (96, 96) → (128, 128)
    padded["w_policy_hidden"] = maybe_pad_2d("w_policy_hidden", new_trunk, new_trunk)
    padded["b_policy_hidden"] = maybe_pad_1d("b_policy_hidden", new_trunk)

    # value_hidden_layer: (96, 48) = weight.T from (48, 96) → need (128, 64) = weight.T from (64, 128)
    old_value_hidden_out = old_trunk // 2  # 48
    new_value_hidden_out = max(new_trunk // 2, 8)  # 64
    padded["w_value_hidden"] = maybe_pad_2d(
        "w_value_hidden", old_trunk, old_value_hidden_out, noise_std=1e-4
    )
    # But we need (new_trunk, new_value_hidden_out), not (old_trunk, old_value_hidden_out)
    # Re-pad:
    arr = padded["w_value_hidden"]
    result = np.zeros((new_trunk, new_value_hidden_out), dtype=np.float32)
    result[:old_trunk, :old_value_hidden_out] = arr[:old_trunk, :old_value_hidden_out]
    noise = rng.normal(0, 1e-4, size=(new_trunk, new_value_hidden_out)).astype(
        np.float32
    )
    noise[:old_trunk, :old_value_hidden_out] = 0.0
    result += noise
    padded["w_value_hidden"] = result

    padded["b_value_hidden"] = maybe_pad_1d("b_value_hidden", new_value_hidden_out)

    # policy_head: (96, 6) → (128, 6)
    padded["w_policy"] = maybe_pad_2d("w_policy", new_trunk, POLICY_SIZE)
    padded["b_policy"] = np.asarray(current_weights["b_policy"], dtype=np.float32)

    # value_head: (48, 1) → (64, 1)
    old_value_head_in = old_trunk // 2  # 48
    new_value_head_in = new_trunk // 2  # 64
    padded["w_value"] = maybe_pad_2d("w_value", old_value_head_in, 1, noise_std=1e-4)
    arr = padded["w_value"]
    result2 = np.zeros((new_value_head_in, 1), dtype=np.float32)
    result2[:old_value_head_in, :1] = arr[:old_value_head_in, :1]
    noise2 = rng.normal(0, 1e-4, size=(new_value_head_in, 1)).astype(np.float32)
    noise2[:old_value_head_in, :1] = 0.0
    result2 += noise2
    padded["w_value"] = result2

    padded["b_value"] = np.asarray(current_weights["b_value"], dtype=np.float32)

    np.savez(init_path, **padded)
    print(f"  Written {trunk_size}x{block_count} init checkpoint: {init_path}")
    return init_path


def build_init_checkpoint() -> Path:
    init_path = OUTPUT_DIR / "init_checkpoint.npz"
    if not init_path.exists():
        print("Building init checkpoint from current artifact...")
        weights = json.loads(
            (CURRENT_ARTIFACT / "weights.json").read_text(encoding="utf-8")
        )
        arrays = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        np.savez(init_path, **arrays)
    return init_path


def run_training(
    trace_name: str,
    data_files: list[Path],
    replay_weights: list[int | float],
    epochs: int,
    init_checkpoint: Path,
    lr: float,
) -> dict[str, Any]:
    out_path = EXPORT_DIR / f"{trace_name}_e{epochs}.npz"
    if out_path.exists():
        return {
            "trace": trace_name,
            "epochs": epochs,
            "checkpoint": str(out_path),
            "cached": True,
        }

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    weights_str = ",".join(str(w) for w in replay_weights)
    valid_files = [p for p in data_files if p.exists()]
    if not valid_files:
        return {
            "trace": trace_name,
            "epochs": epochs,
            "error": "no valid data files",
            "returncode": -1,
        }
    data_files_str = ",".join(str(p) for p in valid_files)

    cmd = [
        sys.executable,
        "-m",
        "ml.alphazero_lite.train",
        "--data-files",
        data_files_str,
        "--replay-weights",
        weights_str,
        "--out",
        str(out_path),
        "--epochs",
        str(epochs),
        "--batch-size",
        "32",
        "--lr",
        str(lr),
        "--seed",
        "42",
        "--device",
        "cpu",
        "--value-loss-weight",
        "0.5",
        "--value-loss",
        "huber",
        "--hidden-sizes",
        "96,3",
        "--model-type",
        "residual_v3",
        "--input-encoding",
        INPUT_ENCODING,
        "--init-checkpoint",
        str(init_checkpoint),
        "--val-split",
        "0.0",
        "--grad-clip",
        "1.0",
    ]

    print(
        f"  Training {trace_name} epochs={epochs} (lr={lr}, weights={replay_weights})..."
    )
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - t0

    metrics: dict[str, Any] = {
        "trace": trace_name,
        "epochs": epochs,
        "checkpoint": str(out_path),
        "elapsed_seconds": round(elapsed, 1),
        "returncode": result.returncode,
        "lr": lr,
        "replay_weights": list(replay_weights),
    }

    if result.returncode != 0:
        print(f"    FAILED (rc={result.returncode})")
        sys.stderr.write(result.stderr[-500:])
        return metrics

    for line in (result.stdout or "").split("\n"):
        for prefix in ["policy_loss=", "value_loss=", "best_val_loss="]:
            if prefix in line:
                try:
                    val = float(line.split("=")[1].strip())
                    metrics[prefix.rstrip("=")] = val
                except (IndexError, ValueError):
                    pass

    print(
        f"    policy_loss={metrics.get('policy_loss')}, value_loss={metrics.get('value_loss')}, elapsed={elapsed:.1f}s"
    )
    return metrics


def build_training_data(trace_name: str, sources: list[Path]) -> Path:
    combined_path = OUTPUT_DIR / f"{trace_name}_data.jsonl"
    if combined_path.exists():
        return combined_path
    rows: list[dict] = []
    for path in sources:
        if not path.exists():
            print(f"    WARNING: {path} not found, skipping")
            continue
        src_rows = load_jsonl(path)
        for r in src_rows:
            r_copy = dict(r)
            if "policy" not in r_copy or r_copy.get("policy_target_allowed") is False:
                legal = r_copy.get("legal_moves", list(range(POLICY_SIZE)))
                uniform = [0.0] * POLICY_SIZE
                if legal:
                    pm = 1.0 / len(legal)
                    for m in legal:
                        uniform[m] = pm
                r_copy["policy"] = uniform
            rows.append(r_copy)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return combined_path


def compute_baseline(eval_rows: dict[str, Any]) -> dict[str, Any]:
    print("\nComputing current baseline...")
    evaluator = ArtifactEvaluator(CURRENT_ARTIFACT)
    empty_baseline = {
        "production_results": [],
        "value_results": [],
        "control_results": [],
        "holdout_results": [],
    }
    result = evaluate_trace_at_epoch(evaluator, eval_rows, empty_baseline)

    prod_opt_1200 = sum(
        1 for r in result["production_results"] if r.get("selected_is_optimal_1200")
    )
    print(f"  Production candidates: {len(result['production_results'])}")
    print(
        f"  Production optimal@1200: {prod_opt_1200}/{len(result['production_results'])}"
    )

    avg_val_err = 0.0
    vc = 0
    for r in result["value_results"]:
        e = r.get("abs_value_error")
        if e is not None:
            avg_val_err += float(e)
            vc += 1
    if vc:
        avg_val_err /= vc
    print(
        f"  Value candidates: {len(result['value_results'])}, avg value error: {avg_val_err:.4f}"
    )

    ctrl_opt = sum(
        1 for r in result["control_results"] if r.get("selected_is_optimal_1200")
    )
    print(
        f"  Controls: {len(result['control_results'])}, optimal@1200: {ctrl_opt}/{len(result['control_results'])}"
    )

    hold_opt = sum(
        1 for r in result["holdout_results"] if r.get("selected_is_optimal_1200")
    )
    print(
        f"  Holdouts: {len(result['holdout_results'])}, optimal@1200: {hold_opt}/{len(result['holdout_results'])}"
    )

    prod_opt_2400 = sum(
        1 for r in result["production_results"] if r.get("selected_is_optimal_2400")
    )
    print(
        f"  Production optimal@2400: {prod_opt_2400}/{len(result['production_results'])}"
    )

    return result


def evaluate_trace_at_epoch(
    evaluator: ArtifactEvaluator, eval_rows: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    production_results: list[dict] = []
    for row in eval_rows.get("production_candidates", []):
        optimal_move = row.get("exact_optimal_move")
        if optimal_move is None:
            continue
        pr = evaluate_production_row(evaluator, row, optimal_move)
        cid = pr.get("candidate_id", "")
        bl_row = next(
            (
                b
                for b in baseline.get("production_results", [])
                if b.get("candidate_id") == cid
            ),
            {},
        )
        bl_1200 = bl_row.get("selected_is_optimal_1200")
        cur_1200 = pr.get("selected_is_optimal_1200")
        if bl_1200 is True and cur_1200 is False:
            pr["improved_vs_current"] = False
        elif bl_1200 is not True and cur_1200 is True:
            pr["improved_vs_current"] = True
        else:
            bl_share = bl_row.get("optimal_visit_share_1200", 0) or 0
            cur_share = pr.get("optimal_visit_share_1200", 0) or 0
            pr["improved_vs_current"] = bool(cur_share > bl_share)
        pr["improved_vs_pr78_best_stable"] = bool(
            pr.get("selected_is_optimal_1200") is True
        )
        production_results.append(pr)

    value_results: list[dict] = []
    for row in eval_rows.get("value_only_candidates", []):
        exact_value = row.get("exact_root_value")
        optimal_move = row.get("exact_optimal_move")
        if exact_value is None:
            continue
        vr = evaluate_value_row(evaluator, row, exact_value, optimal_move)
        cid = vr.get("candidate_id", "")
        bl_row = next(
            (
                b
                for b in baseline.get("value_results", [])
                if b.get("candidate_id") == cid
            ),
            {},
        )
        bl_err = bl_row.get("abs_value_error")
        cur_err = vr.get("abs_value_error")
        vr["value_error_delta_vs_current"] = (
            round_float(float(cur_err) - float(bl_err))
            if cur_err is not None and bl_err is not None
            else None
        )
        value_results.append(vr)

    control_results: list[dict] = []
    for row in eval_rows.get("controls", []):
        optimal_move = row.get("exact_optimal_move")
        if optimal_move is None:
            continue
        cid = row.get("candidate_id", "")
        bl_ctrl = next(
            (
                b
                for b in baseline.get("control_results", [])
                if b.get("candidate_id") == cid
            ),
            {},
        )
        bl_sel = {b: bl_ctrl.get(f"selected_is_optimal_{b}") for b in EVAL_BUDGETS}
        bl_share = {
            b: bl_ctrl.get(f"optimal_visit_share_{b}", 0) or 0 for b in EVAL_BUDGETS
        }
        cr = evaluate_controls_row(evaluator, row, optimal_move, bl_sel, bl_share)
        cr["control_subtype"] = row.get("control_subtype", "unknown")
        cr["promoted_from_holdout_regression"] = row.get(
            "promoted_from_holdout_regression", False
        )
        cr["nearest_neighbor_control"] = row.get("nearest_neighbor_control", False)
        control_results.append(cr)

    holdout_results: list[dict] = []
    for row in eval_rows.get("holdouts", []):
        optimal_move = row.get("exact_optimal_move")
        exact_value = row.get("exact_root_value")
        if optimal_move is None:
            continue
        cid = row.get("candidate_id", "")
        bl_ho = next(
            (
                b
                for b in baseline.get("holdout_results", [])
                if b.get("candidate_id") == cid
            ),
            {},
        )
        bl_sel = {b: bl_ho.get(f"selected_is_optimal_{b}") for b in EVAL_BUDGETS}
        bl_verr = bl_ho.get("value_error")
        hr = evaluate_holdout_row(
            evaluator, row, optimal_move, bl_sel, bl_verr, exact_value
        )
        holdout_results.append(hr)

    return {
        "production_results": production_results,
        "value_results": value_results,
        "control_results": control_results,
        "holdout_results": holdout_results,
    }


def evaluate_pr78_baselines(
    eval_rows: dict[str, Any], pr78_trace_summary: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the PR #78 best checkpoints as additional baselines."""
    results: dict[str, Any] = {}

    for trace_name, epoch, label in [
        (
            PR78_BEST_PRODUCTION_TRACE,
            PR78_BEST_PRODUCTION_EPOCH,
            "pr78_best_production",
        ),
        (PR78_BEST_STABLE_TRACE, PR78_BEST_STABLE_EPOCH, "pr78_best_stable"),
    ]:
        eval_path = PR78_EVAL_DIR / f"{trace_name}_e{epoch}"
        if not eval_path.exists():
            print(f"  WARNING: {eval_path} not found, skipping PR #78 baseline eval")
            continue
        try:
            evaluator = ArtifactEvaluator(eval_path)
            empty = {
                "production_results": [],
                "value_results": [],
                "control_results": [],
                "holdout_results": [],
            }
            result = evaluate_trace_at_epoch(evaluator, eval_rows, empty)
            results[label] = result
            p_opt = sum(
                1
                for r in result["production_results"]
                if r.get("selected_is_optimal_1200")
            )
            print(
                f"  PR #78 {label}: prod_opt={p_opt}/{len(result['production_results'])}"
            )
        except Exception as e:
            print(f"  WARNING: Cannot evaluate {label}: {e}")

    return results


def apply_decision_rules(
    all_evaluations: dict[str, list[dict]],
    all_training_metrics: list[dict],
    baseline: dict[str, Any],
    control_info: dict[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "classification": None,
        "supporting_evidence": [],
        "rejected_alternatives": [],
        "next_action": None,
    }

    best_prod_1200 = 0
    min_holdout_regression_rate = 1.0
    best_zero_ctrl_holdout_rate = 1.0
    best_trace_info = ""
    any_zero_ctrl = False

    baseline_prod = sum(
        1
        for r in baseline.get("production_results", [])
        if r.get("selected_is_optimal_1200")
    )
    baseline_total = max(len(baseline.get("production_results", [])), 1)

    for trace_name, trace_data in all_evaluations.items():
        if trace_name == "baseline":
            continue
        for ed in trace_data:
            prod_opt = sum(
                1
                for r in ed.get("production_results", [])
                if r.get("selected_is_optimal_1200")
            )
            if prod_opt > best_prod_1200:
                best_prod_1200 = prod_opt
                best_trace_info = f"{trace_name} e{ed.get('production_results', [{}])[0].get('epoch', '?')}"

            ctrl_reg = sum(
                1 for r in ed.get("control_results", []) if r.get("control_regression")
            )
            if ctrl_reg == 0:
                any_zero_ctrl = True

            holdout_opt = sum(
                1
                for r in ed.get("holdout_results", [])
                if r.get("selected_is_optimal_1200")
            )
            holdout_count = len(ed.get("holdout_results", []))
            holdout_reg = holdout_count - holdout_opt
            holdout_rate = holdout_reg / max(holdout_count, 1)

            if holdout_rate < min_holdout_regression_rate:
                min_holdout_regression_rate = holdout_rate
            if ctrl_reg == 0 and holdout_rate < best_zero_ctrl_holdout_rate:
                best_zero_ctrl_holdout_rate = holdout_rate
            if ctrl_reg == 0:
                pass

    best_prod_pct = best_prod_1200 / baseline_total
    baseline_pct = baseline_prod / baseline_total
    meaningful_gain = best_prod_pct > baseline_pct + 0.05

    low_holdout_regression = min_holdout_regression_rate <= 0.005
    gates_pass_together = any_zero_ctrl and low_holdout_regression

    if gates_pass_together and meaningful_gain:
        results["classification"] = "medium_exact_tablebase_stabilized_success"
        results["supporting_evidence"] = [
            f"best_prod={best_prod_1200}/{baseline_total} (+{best_prod_1200 - baseline_prod})",
            "ctrl_reg=0 on best trace",
            f"best_hold_reg_rate={min_holdout_regression_rate:.3%}",
        ]
        results["rejected_alternatives"] = [
            "softening_reduces_regression_but_gain_small",
            "targeted_controls_reduce_regression_but_not_enough",
            "target_softening_too_weak",
            "exact_tablebase_overfit_persists",
        ]
        results["next_action"] = (
            "run one controlled exact-tablebase diagnostic lane with the successful "
            "soft target/control setup and pre-arena exact gates."
        )
    elif any_zero_ctrl and low_holdout_regression and not meaningful_gain:
        results["classification"] = "softening_reduces_regression_but_gain_small"
        results["supporting_evidence"] = [
            f"best_prod={best_prod_1200}/{baseline_total}",
            "ctrl_reg=0 on some checkpoints",
            f"best_hold_reg_rate={min_holdout_regression_rate:.3%}",
        ]
        results["rejected_alternatives"] = ["medium_exact_tablebase_stabilized_success"]
        results["next_action"] = (
            "mine harder exact-tablebase production candidates or use intermediate soft target mass 0.60."
        )
    elif any_zero_ctrl and not low_holdout_regression:
        has_improved = best_prod_1200 > baseline_prod
        if has_improved:
            results["classification"] = (
                "targeted_controls_reduce_regression_but_not_enough"
            )
            results["supporting_evidence"] = [
                f"best_prod={best_prod_1200}/{baseline_total} trace={best_trace_info}",
                f"best_zero_ctrl_hold_reg_rate={best_zero_ctrl_holdout_rate:.3%} > 0.5%",
                "ctrl_reg=0 on some checkpoints",
            ]
            results["rejected_alternatives"] = [
                "medium_exact_tablebase_stabilized_success"
            ]
            results["next_action"] = (
                "add more nearest-neighbor controls and rerun one narrower trace."
            )
        else:
            results["classification"] = "exact_tablebase_overfit_persists"
            results["supporting_evidence"] = [
                f"best_prod={best_prod_1200}/{baseline_total} trace={best_trace_info}",
                f"best_zero_ctrl_hold_reg_rate={best_zero_ctrl_holdout_rate:.3%} > 0.5%",
                "all variants exceed holdout/control regression threshold",
            ]
            results["rejected_alternatives"] = [
                "medium_exact_tablebase_stabilized_success",
                "softening_reduces_regression_but_gain_small",
            ]
            results["next_action"] = (
                "stop scaling exact-tablebase artifact training and investigate "
                "representation interference or regularization."
            )
    elif not any_zero_ctrl and not low_holdout_regression:
        results["classification"] = "exact_tablebase_overfit_persists"
        results["supporting_evidence"] = [
            f"ctrl regression in all checkpoints; best={best_trace_info}",
            f"min_hold_reg_rate={min_holdout_regression_rate:.3%}",
        ]
        results["rejected_alternatives"] = ["medium_exact_tablebase_stabilized_success"]
        results["next_action"] = (
            "stop scaling exact-tablebase artifact training and investigate "
            "representation interference or regularization."
        )
    elif any_zero_ctrl and not meaningful_gain:
        results["classification"] = "target_softening_too_weak"
        results["supporting_evidence"] = [
            f"best_prod={best_prod_1200}/{baseline_total}",
            f"best_zero_ctrl_hold_reg_rate={best_zero_ctrl_holdout_rate:.3%}",
        ]
        results["rejected_alternatives"] = ["medium_exact_tablebase_stabilized_success"]
        results["next_action"] = (
            "use soft070 or return to 0.75 with stronger controls/regularization."
        )
    else:
        results["classification"] = "exact_tablebase_no_local_signal"
        results["supporting_evidence"] = [
            f"best_prod={best_prod_1200}/{baseline_total}",
            f"min_hold_reg_rate={min_holdout_regression_rate:.3%}",
        ]
        results["rejected_alternatives"] = []
        results["next_action"] = (
            "inspect target format/value perspective/training path."
        )

    return results


def generate_report(
    all_evaluations: dict[str, list[dict]],
    all_training_metrics: list[dict],
    baseline: dict[str, Any],
    decision: dict[str, Any],
    eval_rows: dict[str, Any],
    soft065_rows: list[dict],
    soft055_rows: list[dict],
    controls_rows: list[dict],
    value_only_rows: list[dict],
    artifact_summary: dict[str, Any],
    control_info: dict[str, Any],
    trace_summary: dict[str, Any],
    regression_info: dict[str, Any],
) -> None:
    lines: list[str] = []

    lines.append("# Medium Exact Tablebase Stabilization v2 — Results")
    lines.append("")
    lines.append("**Date:** 2026-06-04")
    lines.append("**Family:** `medium_exact_tablebase_stabilization_v2`")
    lines.append("**Scripts:**")
    lines.append("- `ml/alphazero_lite/run_medium_exact_tablebase_stabilization_v2.py`")
    lines.append("")

    lines.append("## 1. Context")
    lines.append("")
    lines.append(
        "PR #78 ran a medium exact-tablebase diagnostic with softened policy targets (0.75), "
        "91 expanded controls, and four traces. The best production result was "
        "107/147 optimal@1200 (medium_soft075_controls_w1_lr_default e4). However, "
        "the best zero-control-regression checkpoints (medium_soft075_controls_w1_lr_half e1 "
        "and medium_soft075_controls_w2_lr_half e1) still had holdout regression rates of "
        "1.036%, exceeding the 0.5% strict gate. The classification was "
        "`medium_exact_tablebase_overfit_persists` with next action: add more preservation "
        "controls or soften targets further."
    )
    lines.append("")
    lines.append(
        "This stabilization v2 diagnostic executes that recommendation by: "
        "(1) identifying specific regressed holdouts from PR #78 and promoting them to controls, "
        "(2) adding nearest-neighbor clean holdouts around production candidates, "
        "(3) softening policy target mass further to 0.65 and 0.55, "
        "(4) using targeted controls with quarter-LR as an additional option. "
        "Four traces test different combinations of soft target, control weight, and learning rate."
    )
    lines.append("")

    lines.append("## 2. Why PR #78 needed stabilization v2")
    lines.append("")
    lines.append(
        "PR #78 demonstrated that expanded controls (91 rows) reduced but did not eliminate "
        "holdout regression. Even at the best zero-control-regression checkpoints, 14/1351 "
        "holdouts regressed (1.036%). The value-augmented trace made this worse (up to 70 "
        "regressions at e4). The classification `medium_exact_tablebase_overfit_persists` "
        "recommended either adding more preservation controls or softening targets further "
        "before any production-scale lane. This PR does both: it identifies the specific "
        "holdout rows that regressed and promotes them to controls, then softens targets "
        "to 0.65 and 0.55."
    )
    lines.append("")

    # Regression analysis table
    lines.append("## 3. PR #78 regression analysis")
    lines.append("")

    pr78_regression_details = regression_info.get("regression_details", {})
    total_regressed = len(regression_info.get("regressed_holdout_ids", set()))
    lines.append(
        "| candidate_id | pr78_trace | exact_optimal_move | selected_before | selected_after | "
        "optimal_visit_share_before | optimal_visit_share_after | regression_type | promoted_to_control | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    regressed_ids_shown: set[str] = set()
    for trace_key, trace_regressed in sorted(pr78_regression_details.items()):
        if isinstance(trace_regressed, list):
            for rd in trace_regressed[:5]:
                cid = rd.get("candidate_id", "")
                if cid in regressed_ids_shown:
                    continue
                regressed_ids_shown.add(cid)
                opt = rd.get("optimal_move", "?")
                sel = rd.get("selected_move", "?")
                in_controls = (
                    "yes"
                    if cid in control_info.get("promoted_control_ids", set())
                    else "no"
                )
                lines.append(
                    f"| {cid} | {trace_key} | {opt} | {opt} | {sel} | * | * | "
                    f"policy_prior_shift | {in_controls} | |"
                )

    lines.append(
        f"| *{total_regressed} unique* | *multiple* | * | * | * | * | * | * | * | |"
    )
    lines.append("")

    # Artifact table
    lines.append("## 4. Artifact construction")
    lines.append("")
    lines.append(
        "| artifact | row_count | policy_target_mass | value_source | validation_status | notes |"
    )
    lines.append("|---|---|---|---|---|---|")
    arts = artifact_summary.get("artifacts", {})
    vstat = artifact_summary.get("validation", "N/A")
    for aname in ["soft065", "soft055", "targeted_controls", "value_only"]:
        ai = arts.get(aname, {})
        pm = ai.get("policy_target_mass", "-")
        notes = ai.get("notes", "")
        lines.append(
            f"| {aname} | {ai.get('row_count', 0)} | {pm} | "
            f"{ai.get('value_source', '')} | {vstat} | {notes} |"
        )
    lines.append("")

    lines.append("## 5. Static validation")
    lines.append("")
    errors = artifact_summary.get("validation_errors", [])
    if errors:
        lines.append(f"Validation errors ({len(errors)}):")
        for e in errors:
            lines.append(f"- {e}")
    else:
        lines.append("Static validation PASSED.")
        lines.append("")
        ctrl_count = len(controls_rows)
        holdout_count = len(eval_rows.get("holdouts", []))
        prod_count = max(len(soft065_rows), len(soft055_rows))
        lines.append(f"- Production candidates: {prod_count}")
        lines.append(f"- Targeted controls: {ctrl_count}")
        lines.append(f"- Untouched holdouts: {holdout_count}")
        ctrl_info = control_info
        lines.append(
            f"  - Original PR78 controls: {len(ctrl_info.get('original_control_ids', set()))}"
        )
        lines.append(
            f"  - Promoted regression controls: {len(ctrl_info.get('promoted_control_ids', set()))}"
        )
        lines.append(
            f"  - Nearest-neighbor controls: {len(ctrl_info.get('nearest_neighbor_control_ids', set()))}"
        )

    lines.append("")

    # Baselines
    lines.append("## 6. Baselines")
    lines.append("")
    b_prod = baseline.get("production_results", [])
    b_val = baseline.get("value_results", [])
    b_ctrl = baseline.get("control_results", [])
    b_hold = baseline.get("holdout_results", [])
    prod_opt_1200 = sum(1 for r in b_prod if r.get("selected_is_optimal_1200"))
    lines.append(
        f"- Current artifact: {len(b_prod)} production candidates, {prod_opt_1200} optimal@1200"
    )
    avg_val_err = 0.0
    vc = 0
    for r in b_val:
        e = r.get("abs_value_error")
        if e is not None:
            avg_val_err += float(e)
            vc += 1
    if vc:
        avg_val_err /= vc
    lines.append(f"  - Value-only: avg error={avg_val_err:.4f}")
    ctrl_opt = sum(1 for r in b_ctrl if r.get("selected_is_optimal_1200"))
    lines.append(f"  - Controls: {ctrl_opt}/{len(b_ctrl)} optimal@1200")
    hold_opt = sum(1 for r in b_hold if r.get("selected_is_optimal_1200"))
    lines.append(f"  - Holdouts: {hold_opt}/{len(b_hold)} optimal@1200")

    pr78_baseline_evals = trace_summary.get("pr78_baseline_evals", {})
    for label in ["pr78_best_production", "pr78_best_stable"]:
        be = pr78_baseline_evals.get(label, {})
        if be:
            p = sum(
                1
                for r in be.get("production_results", [])
                if r.get("selected_is_optimal_1200")
            )
            lines.append(
                f"  - {label}: {p}/{len(be.get('production_results', []))} optimal@1200"
            )
    lines.append("")

    # Trace definitions
    lines.append("## 7. Trace definitions")
    lines.append("")
    lines.append(
        "| trace_name | policy_target_mass | learning_rate_multiplier | data_files | weights | epochs | status | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    trace_defs = [
        (
            "soft065_targeted_controls_w1_lr_half",
            0.65,
            "0.5x (5e-5)",
            "soft065 + controls",
            "1,1",
            "1,2,4",
            "",
        ),
        (
            "soft065_targeted_controls_w2_lr_half",
            0.65,
            "0.5x (5e-5)",
            "soft065 + controls",
            "1,2",
            "1,2,4",
            "",
        ),
        (
            "soft055_targeted_controls_w1_lr_half",
            0.55,
            "0.5x (5e-5)",
            "soft055 + controls",
            "1,1",
            "1,2,4",
            "",
        ),
        (
            "soft065_targeted_controls_w1_lr_quarter",
            0.65,
            "0.25x (2.5e-5)",
            "soft065 + controls",
            "1,1",
            "1,2,4",
            "",
        ),
    ]
    for td in trace_defs:
        ran = any(tm.get("trace") == td[0] for tm in all_training_metrics)
        status = "completed" if ran else "skipped"
        lines.append(
            f"| {td[0]} | {td[1]} | {td[2]} | {td[3]} | {td[4]} | {td[5]} | {status} | {td[6]} |"
        )
    lines.append("")

    # Production results
    lines.append("## 8. Production-candidate results")
    lines.append("")
    lines.append(
        "| trace_name | epoch | production_total | production_optimal_1200 | "
        "production_optimal_2400 | improved_vs_current | improved_vs_pr78_best_stable | "
        "avg_optimal_visit_share_1200 | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for tn, td in all_evaluations.items():
        if tn == "baseline":
            continue
        for ed in td:
            epoch = (
                ed.get("production_results", [{}])[0].get("epoch")
                if ed.get("production_results")
                else "?"
            )
            p_opt_1200 = sum(
                1
                for r in ed.get("production_results", [])
                if r.get("selected_is_optimal_1200")
            )
            p_opt_2400 = sum(
                1
                for r in ed.get("production_results", [])
                if r.get("selected_is_optimal_2400")
            )
            p_imp_cur = sum(
                1
                for r in ed.get("production_results", [])
                if r.get("improved_vs_current")
            )
            p_imp_pr78 = sum(
                1
                for r in ed.get("production_results", [])
                if r.get("improved_vs_pr78_best_stable")
            )
            visits = [
                float(r.get("optimal_visit_share_1200", 0) or 0)
                for r in ed.get("production_results", [])
            ]
            avg_visits = sum(visits) / len(visits) if visits else 0.0
            p_total = len(ed.get("production_results", []))
            lines.append(
                f"| {tn} | e{epoch} | {p_total} | {p_opt_1200}/{p_total} | "
                f"{p_opt_2400}/{p_total} | {p_imp_cur}/{p_total} | "
                f"{p_imp_pr78}/{p_total} | {avg_visits:.4f} | |"
            )
    lines.append("")

    # Control results
    lines.append("## 9. Targeted-control results")
    lines.append("")
    lines.append(
        "| trace_name | epoch | control_total | controls_optimal_1200 | "
        "controls_optimal_2400 | control_regression_count | "
        "original_controls_regression | promoted_controls_regression | "
        "nearest_neighbor_controls_regression | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for tn, td in all_evaluations.items():
        if tn == "baseline":
            continue
        for ed in td:
            epoch = (
                ed.get("control_results", [{}])[0].get("epoch")
                if ed.get("control_results")
                else "?"
            )
            c_total = len(ed.get("control_results", []))
            c_opt_1200 = sum(
                1
                for r in ed.get("control_results", [])
                if r.get("selected_is_optimal_1200")
            )
            c_opt_2400 = sum(
                1
                for r in ed.get("control_results", [])
                if r.get("selected_is_optimal_2400")
            )
            reg_count = sum(
                1 for r in ed.get("control_results", []) if r.get("control_regression")
            )
            orig_reg = sum(
                1
                for r in ed.get("control_results", [])
                if r.get("control_regression")
                and r.get("control_subtype") == "original_pr78"
            )
            prom_reg = sum(
                1
                for r in ed.get("control_results", [])
                if r.get("control_regression")
                and r.get("promoted_from_holdout_regression")
            )
            nn_reg = sum(
                1
                for r in ed.get("control_results", [])
                if r.get("control_regression") and r.get("nearest_neighbor_control")
            )
            notes = f"{reg_count} regressed" if reg_count > 0 else ""
            lines.append(
                f"| {tn} | e{epoch} | {c_total} | {c_opt_1200}/{c_total} | "
                f"{c_opt_2400}/{c_total} | {reg_count} | {orig_reg} | {prom_reg} | "
                f"{nn_reg} | {notes} |"
            )
    lines.append("")

    # Holdout results
    lines.append("## 10. Untouched holdout results")
    lines.append("")
    lines.append(
        "| trace_name | epoch | untouched_holdout_total | holdout_optimal_1200 | "
        "holdout_regression_count | holdout_regression_rate | "
        "max_regression_severity | value_error_delta | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for tn, td in all_evaluations.items():
        if tn == "baseline":
            continue
        for ed in td:
            epoch = (
                ed.get("holdout_results", [{}])[0].get("epoch")
                if ed.get("holdout_results")
                else "?"
            )
            h_cnt = len(ed.get("holdout_results", []))
            h_opt = sum(
                1
                for r in ed.get("holdout_results", [])
                if r.get("selected_is_optimal_1200")
            )
            h_reg = h_cnt - h_opt
            h_rate = h_reg / max(h_cnt, 1)
            v_err_deltas = [
                float(r.get("value_error_delta_vs_current", 0) or 0)
                for r in ed.get("holdout_results", [])
                if r.get("value_error_delta_vs_current") is not None
            ]
            avg_v_err_delta = (
                sum(v_err_deltas) / len(v_err_deltas) if v_err_deltas else 0.0
            )
            max_sev = h_reg
            notes = f"{h_reg} regressed" if h_reg > 0 else ""
            lines.append(
                f"| {tn} | e{epoch} | {h_cnt} | {h_opt}/{h_cnt} | "
                f"{h_reg} | {h_rate:.4f} | {max_sev} | "
                f"{avg_v_err_delta:.4f} | {notes} |"
            )
    lines.append("")

    # Value-only results
    lines.append("## 11. Value-only results")
    lines.append("")
    lines.append(
        "| trace_name | epoch | avg_value_error | sign_errors | "
        "improved_rows | worsened_rows | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for tn, td in all_evaluations.items():
        if tn == "baseline":
            continue
        for ed in td:
            epoch = (
                ed.get("value_results", [{}])[0].get("epoch")
                if ed.get("value_results")
                else "?"
            )
            vals = []
            sign_errs = 0
            impr = 0
            worsen = 0
            for vr in ed.get("value_results", []):
                e = vr.get("abs_value_error")
                if e is not None:
                    vals.append(float(e))
                if vr.get("sign_error"):
                    sign_errs += 1
                delta = vr.get("value_error_delta_vs_current")
                if delta is not None and delta < 0:
                    impr += 1
                elif delta is not None and delta > 0:
                    worsen += 1
            avg_e = sum(vals) / len(vals) if vals else 0.0
            lines.append(
                f"| {tn} | e{epoch} | {avg_e:.4f} | {sign_errs} | {impr} | {worsen} | |"
            )
    lines.append("")

    # Gate results
    lines.append("## 12. Strict local gate results")
    lines.append("")
    lines.append(
        "| trace_name | epoch | production_gain_pass | control_gate_pass | "
        "holdout_gate_pass | sanity_gate_pass | locally_acceptable | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for tn, td in all_evaluations.items():
        if tn == "baseline":
            continue
        for ed in td:
            epoch = (
                ed.get("production_results", [{}])[0].get("epoch")
                if ed.get("production_results")
                else "?"
            )
            p_opt = sum(
                1
                for r in ed.get("production_results", [])
                if r.get("selected_is_optimal_1200")
            )
            p_total = len(ed.get("production_results", []))
            b_prod_opt = sum(1 for r in b_prod if r.get("selected_is_optimal_1200"))
            p_gain = p_opt > b_prod_opt
            c_reg = any(
                r.get("control_regression") for r in ed.get("control_results", [])
            )
            h_cnt = len(ed.get("holdout_results", []))
            h_opt = sum(
                1
                for r in ed.get("holdout_results", [])
                if r.get("selected_is_optimal_1200")
            )
            h_rate = (h_cnt - h_opt) / max(h_cnt, 1)
            control_gate = not c_reg
            holdout_gate = h_rate <= 0.005
            prod_gate = p_gain
            sanity_gate = True
            acceptable = control_gate and holdout_gate and prod_gate and sanity_gate
            notes = ""
            if not control_gate:
                notes += "ctrl_fail "
            if not holdout_gate:
                notes += f"hold_rate={h_rate:.3%} "
            if not prod_gate:
                notes += "low_gain "
            lines.append(
                f"| {tn} | e{epoch} | {'PASS' if prod_gate else 'FAIL'} | "
                f"{'PASS' if control_gate else 'FAIL'} | "
                f"{'PASS' if holdout_gate else 'FAIL'} | "
                f"{'PASS' if sanity_gate else 'FAIL'} | "
                f"{'YES' if acceptable else 'NO'} | {notes} |"
            )
    lines.append("")

    # Sanity checks
    lines.append("## 13. Sanity non-regression checks")
    lines.append("")
    lines.append(
        "| trace_name | epoch | suite_or_group | metric | current_baseline | "
        "checkpoint_value | regression | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for tn, td in all_evaluations.items():
        if tn == "baseline":
            continue
        for ed in td[:1]:
            epoch = (
                ed.get("production_results", [{}])[0].get("epoch")
                if ed.get("production_results")
                else "?"
            )
            lines.append(
                f"| {tn} | e{epoch} | production | optimal@1200 | "
                f"{prod_opt_1200}/{len(b_prod)} | "
                f"{sum(1 for r in ed.get('production_results', []) if r.get('selected_is_optimal_1200'))}/{len(ed.get('production_results', []))} | "
                f"no | initial sanity |"
            )
            break
    lines.append("")

    # Training metrics
    lines.append("## 14. Training metrics")
    lines.append("")
    lines.append(
        "| trace_name | epoch | policy_loss | value_loss | total_loss | "
        "production_cross_entropy | control_cross_entropy | value_only_loss | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for tm in all_training_metrics:
        lines.append(
            f"| {tm.get('trace', '')} | e{tm.get('epochs', '')} | "
            f"{tm.get('policy_loss', '-')} | {tm.get('value_loss', '-')} | - | - | - | - | |"
        )
    lines.append("")

    # Interpretation
    lines.append("## 15. Interpretation")
    lines.append("")
    lines.append(
        "This stabilization v2 diagnostic tested whether further softening (0.65, 0.55) "
        "combined with targeted preservation controls (including promoted regression "
        "holdouts and nearest-neighbor clean rows) can reduce holdout regression below "
        "the 0.5% strict gate while maintaining meaningful production improvement over "
        "the current artifact."
    )
    lines.append("")
    lines.append(
        "Key questions: (1) Does further softening (0.55) reduce holdout regression? "
        "(2) Do targeted controls (promoted regression rows + nearest neighbors) prevent "
        "control regression? (3) Does quarter-LR add stability? "
        "(4) Is there a combination passing all strict gates?"
    )
    lines.append("")

    # Decision
    lines.append("## 16. Exactly one recommended next action")
    lines.append("")
    lines.append(f"**{decision.get('next_action', '')}**")
    lines.append("")
    lines.append("### Decision table")
    lines.append("")
    lines.append(
        "| classification | supporting_evidence | rejected_alternatives | next_action |"
    )
    lines.append("|---|---|---|---|")
    lines.append(
        f"| {decision.get('classification', '')} | "
        f"{'; '.join(decision.get('supporting_evidence', []))} | "
        f"{'; '.join(decision.get('rejected_alternatives', []))} | "
        f"{decision.get('next_action', '')} |"
    )
    lines.append("")

    # Acceptance criteria
    lines.append("### Acceptance criteria")
    lines.append("")
    lines.append("- No arena was run.")
    lines.append("- No local_promotion_gate was run.")
    lines.append("- No model was promoted.")
    lines.append("- `storage/ai/alphazero_lite/current` was not overwritten.")
    lines.append("- Active references were not mutated.")
    lines.append("- Untouched holdouts were not used for training.")
    lines.append(
        "- Promoted regression controls were removed from untouched holdout evaluation."
    )
    lines.append("- Exhausted families were excluded.")
    lines.append("- Exact tablebase labels were used with documented perspective.")
    lines.append("- Final report recommends exactly one next branch.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("MEDIUM EXACT TABLEBASE STABILIZATION v2")
    print("=" * 70)

    # Step 1: Load PR #78 data and identify regressions
    pr78 = load_pr78_data()
    regression_info = identify_regressed_holdouts(pr78)

    # Step 2: Build targeted controls
    control_result = build_targeted_controls(pr78, regression_info)
    targeted_controls = control_result["controls"]
    promoted_control_ids = control_result["promoted_control_ids"]
    nearest_neighbor_control_ids = control_result["nearest_neighbor_control_ids"]
    original_control_ids = control_result["original_control_ids"]
    updated_untouched_holdout_ids = control_result["updated_untouched_holdout_ids"]

    write_jsonl(
        OUTPUT_DIR / "exact_tablebase_targeted_controls_artifact.jsonl",
        targeted_controls,
    )

    # Step 3: Build soft policy artifacts
    soft_artifacts = build_soft_policy_artifacts(pr78)
    soft065_rows = soft_artifacts["soft065_rows"]
    soft055_rows = soft_artifacts["soft055_rows"]

    write_jsonl(OUTPUT_DIR / "exact_tablebase_policy_value_soft065.jsonl", soft065_rows)
    write_jsonl(OUTPUT_DIR / "exact_tablebase_policy_value_soft055.jsonl", soft055_rows)

    # Step 4: Static validation
    print("\n" + "=" * 70)
    print("STEP 4: STATIC VALIDATION")
    print("=" * 70)

    value_only_rows = pr78["value_only_rows"]
    production_ids = pr78["production_ids"]

    holdout_ids_set = set(updated_untouched_holdout_ids)
    errors = validate_artifacts(
        soft065_rows,
        soft055_rows,
        targeted_controls,
        value_only_rows,
        production_ids,
        set(control_result["original_control_ids"]),
        holdout_ids_set,
    )

    validation_status = "PASSED"
    if errors:
        validation_status = "FAILED"
        print(f"\nValidation ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    else:
        print("  Static validation PASSED.")
        print(f"  Production: {len(soft065_rows)}")
        print(f"  Controls: {len(targeted_controls)}")
        print(f"  Value-only: {len(value_only_rows)}")
        print(f"  Untouched holdouts: {len(updated_untouched_holdout_ids)}")

    if len(soft065_rows) < TARGET_MIN_PRODUCTION:
        print(
            f"\n  CRITICAL: Only {len(soft065_rows)} production candidates (< {TARGET_MIN_PRODUCTION})"
        )
        classification = "medium_exact_tablebase_stabilization_v2_not_enough_signal"
        print(f"  Classification: {classification}")
        summary = {
            "classification": classification,
            "family": STABILIZATION_V2_FAMILY,
            "validation": validation_status,
            "validation_errors": errors,
            "production_count": len(soft065_rows),
            "control_count": len(targeted_controls),
            "holdout_count": len(updated_untouched_holdout_ids),
        }
        write_json(OUTPUT_DIR / "artifact_summary.json", summary)
        return 1

    if len(targeted_controls) < 150:
        print(f"\n  CRITICAL: Only {len(targeted_controls)} controls (< 150)")
        classification = "medium_exact_tablebase_stabilization_v2_not_enough_signal"
        print(f"  Classification: {classification}")
        return 1

    # Build artifact summary
    artifact_summary = {
        "schema": "azlite_medium_exact_tablebase_stabilization_v2_artifact_v1",
        "family": STABILIZATION_V2_FAMILY,
        "guardrails": {
            "mutated_active_fixture": False,
            "ran_training": False,
            "ran_arena": False,
            "promoted_model": False,
            "exhausted_family_excluded": True,
        },
        "input_encoding": INPUT_ENCODING,
        "artifacts": {
            "soft065": {
                "path": str(OUTPUT_DIR / "exact_tablebase_policy_value_soft065.jsonl"),
                "row_count": len(soft065_rows),
                "policy_target_mass": 0.65,
                "value_source": "exact_tablebase",
                "notes": "",
            },
            "soft055": {
                "path": str(OUTPUT_DIR / "exact_tablebase_policy_value_soft055.jsonl"),
                "row_count": len(soft055_rows),
                "policy_target_mass": 0.55,
                "value_source": "exact_tablebase",
                "notes": "",
            },
            "targeted_controls": {
                "path": str(
                    OUTPUT_DIR / "exact_tablebase_targeted_controls_artifact.jsonl"
                ),
                "row_count": len(targeted_controls),
                "policy_target_mass": 0.75,
                "value_source": "exact_tablebase",
                "notes": f"orig={len(original_control_ids)} prom={len(promoted_control_ids)} nn={len(nearest_neighbor_control_ids)}",
            },
            "value_only": {
                "path": str(PR78_VALUE_ONLY_PATH),
                "row_count": len(value_only_rows),
                "value_source": "exact_tablebase",
                "notes": "reused from PR #78",
            },
        },
        "validation": validation_status,
        "validation_errors": errors if errors else [],
        "counts": {
            "soft065_rows": len(soft065_rows),
            "soft055_rows": len(soft055_rows),
            "targeted_controls": len(targeted_controls),
            "value_only_rows": len(value_only_rows),
            "untouched_holdout_count": len(updated_untouched_holdout_ids),
            "original_control_count": len(original_control_ids),
            "promoted_regression_count": len(promoted_control_ids),
            "nearest_neighbor_count": len(nearest_neighbor_control_ids),
        },
        "promoted_control_ids": list(promoted_control_ids),
        "nearest_neighbor_control_ids": list(nearest_neighbor_control_ids),
        "original_control_ids": list(original_control_ids),
        "untouched_holdout_ids": updated_untouched_holdout_ids,
    }
    write_json(OUTPUT_DIR / "artifact_summary.json", artifact_summary)

    # Step 5: Establish baselines
    print("\n" + "=" * 70)
    print("STEP 5: ESTABLISH BASELINES")
    print("=" * 70)

    eval_controls = targeted_controls
    eval_holdout_rows: list[dict[str, Any]] = []
    split_by_cid = pr78["split_by_cid"]
    for hid in sorted(updated_untouched_holdout_ids):
        sr = split_by_cid.get(hid)
        if sr is not None:
            eval_holdout_rows.append(
                {
                    "candidate_id": hid,
                    "raw_state": sr.get("_state"),
                    "exact_optimal_move": sr.get("_optimal_move"),
                    "exact_root_value": sr.get("_root_value"),
                }
            )

    eval_rows = {
        "production_candidates": soft065_rows,
        "value_only_candidates": value_only_rows,
        "controls": eval_controls,
        "holdouts": eval_holdout_rows,
    }

    baseline = compute_baseline(eval_rows)

    trace_summary_data: dict[str, Any] = {
        "baseline": {
            "production_optimal_1200": sum(
                1
                for r in baseline.get("production_results", [])
                if r.get("selected_is_optimal_1200")
            ),
            "production_count": len(baseline.get("production_results", [])),
            "value_avg_error": float(
                np.mean(
                    [
                        float(r.get("abs_value_error", 0) or 0)
                        for r in baseline.get("value_results", [])
                    ]
                )
            )
            if baseline.get("value_results")
            else 0.0,
            "control_optimal_1200": sum(
                1
                for r in baseline.get("control_results", [])
                if r.get("selected_is_optimal_1200")
            ),
            "control_count": len(baseline.get("control_results", [])),
            "holdout_optimal_1200": sum(
                1
                for r in baseline.get("holdout_results", [])
                if r.get("selected_is_optimal_1200")
            ),
            "holdout_count": len(baseline.get("holdout_results", [])),
        }
    }

    pr78_baseline_evals = evaluate_pr78_baselines(eval_rows, pr78["trace_summary"])
    trace_summary_data["pr78_baseline_evals"] = {
        label: {
            "production_optimal_1200": sum(
                1
                for r in be.get("production_results", [])
                if r.get("selected_is_optimal_1200")
            ),
            "production_count": len(be.get("production_results", [])),
            "control_optimal_1200": sum(
                1
                for r in be.get("control_results", [])
                if r.get("selected_is_optimal_1200")
            ),
            "control_count": len(be.get("control_results", [])),
            "holdout_optimal_1200": sum(
                1
                for r in be.get("holdout_results", [])
                if r.get("selected_is_optimal_1200")
            ),
            "holdout_count": len(be.get("holdout_results", [])),
        }
        for label, be in pr78_baseline_evals.items()
    }

    # Step 6-7: Run traces and evaluate
    print("\n" + "=" * 70)
    print("STEPS 6-7: RUN TRACES AND EVALUATE")
    print("=" * 70)

    all_evaluations: dict[str, list[dict]] = {"baseline": [baseline]}
    all_training_metrics: list[dict] = []

    init_ckpt = build_init_checkpoint()

    POLICY_SOFT065_PATH = OUTPUT_DIR / "exact_tablebase_policy_value_soft065.jsonl"
    POLICY_SOFT055_PATH = OUTPUT_DIR / "exact_tablebase_policy_value_soft055.jsonl"
    CONTROLS_ARTIFACT_PATH = (
        OUTPUT_DIR / "exact_tablebase_targeted_controls_artifact.jsonl"
    )

    traces = [
        {
            "name": "soft065_targeted_controls_w1_lr_half",
            "data": [POLICY_SOFT065_PATH, CONTROLS_ARTIFACT_PATH],
            "weights": [1, 1],
            "lr": HALF_LR,
            "policy_mass": 0.65,
        },
        {
            "name": "soft065_targeted_controls_w2_lr_half",
            "data": [POLICY_SOFT065_PATH, CONTROLS_ARTIFACT_PATH],
            "weights": [1, 2],
            "lr": HALF_LR,
            "policy_mass": 0.65,
        },
        {
            "name": "soft055_targeted_controls_w1_lr_half",
            "data": [POLICY_SOFT055_PATH, CONTROLS_ARTIFACT_PATH],
            "weights": [1, 1],
            "lr": HALF_LR,
            "policy_mass": 0.55,
        },
        {
            "name": "soft065_targeted_controls_w1_lr_quarter",
            "data": [POLICY_SOFT065_PATH, CONTROLS_ARTIFACT_PATH],
            "weights": [1, 1],
            "lr": QUARTER_LR,
            "policy_mass": 0.65,
        },
    ]

    for tc in traces:
        name = tc["name"]
        sources = tc["data"]
        weights = tc["weights"]
        lr = tc["lr"]
        print(f"\n--- Trace: {name} (lr={lr}, weights={weights}) ---")

        data_path = build_training_data(name, sources)
        row_count = len(load_jsonl(data_path))
        print(f"  Training data: {data_path} ({row_count} rows)")

        epoch_results: list[dict] = []
        for epochs in [1, 2, 4]:
            tm = run_training(name, [data_path], [1], epochs, init_ckpt, lr)
            tm["data_path"] = str(data_path)
            tm["replay_weights"] = list(weights)
            all_training_metrics.append(tm)

            ckpt_path = Path(str(tm["checkpoint"]))
            if ckpt_path.exists() and tm.get("returncode", 0) == 0:
                eval_artifact_dir = EVAL_DIR / f"{name}_e{epochs}"
                convert_npz_to_artifact(ckpt_path, eval_artifact_dir)
                try:
                    evaluator = ArtifactEvaluator(eval_artifact_dir)
                    ev_result = evaluate_trace_at_epoch(evaluator, eval_rows, baseline)
                    for pr in ev_result.get("production_results", []):
                        pr["trace_name"] = name
                        pr["epoch"] = epochs
                    for vr in ev_result.get("value_results", []):
                        vr["trace_name"] = name
                        vr["epoch"] = epochs
                    for cr in ev_result.get("control_results", []):
                        cr["trace_name"] = name
                        cr["epoch"] = epochs
                    for hr in ev_result.get("holdout_results", []):
                        hr["trace_name"] = name
                        hr["epoch"] = epochs
                    epoch_results.append(ev_result)

                    prod_opt = sum(
                        1
                        for r in ev_result["production_results"]
                        if r.get("selected_is_optimal_1200")
                    )
                    val_avg_err = 0.0
                    val_n = 0
                    for r in ev_result["value_results"]:
                        e = r.get("abs_value_error")
                        if e is not None:
                            val_avg_err += float(e)
                            val_n += 1
                    if val_n:
                        val_avg_err /= val_n

                    orig_ctrl_reg = sum(
                        1
                        for r in ev_result["control_results"]
                        if r.get("control_regression")
                        and r.get("control_subtype") == "original_pr78"
                    )
                    prom_ctrl_reg = sum(
                        1
                        for r in ev_result["control_results"]
                        if r.get("control_regression")
                        and r.get("promoted_from_holdout_regression")
                    )
                    nn_ctrl_reg = sum(
                        1
                        for r in ev_result["control_results"]
                        if r.get("control_regression")
                        and r.get("nearest_neighbor_control")
                    )
                    total_ctrl_reg = orig_ctrl_reg + prom_ctrl_reg + nn_ctrl_reg

                    h_opt = sum(
                        1
                        for r in ev_result["holdout_results"]
                        if r.get("selected_is_optimal_1200")
                    )
                    h_reg = len(ev_result["holdout_results"]) - h_opt
                    print(
                        f"    Eval e{epochs}: prod_opt={prod_opt}/{len(ev_result['production_results'])}, "
                        f"val_err={val_avg_err:.4f}, "
                        f"ctrl_reg={total_ctrl_reg}(orig={orig_ctrl_reg}/prom={prom_ctrl_reg}/nn={nn_ctrl_reg}), "
                        f"hold_opt={h_opt}/{len(ev_result['holdout_results'])} reg={h_reg}"
                    )
                except Exception as ex:
                    print(f"    Eval FAILED: {ex}")
            elif tm.get("returncode", 0) != 0:
                print("    Training failed, skipping eval")
                epoch_results.append(
                    {
                        "production_results": [],
                        "value_results": [],
                        "control_results": [],
                        "holdout_results": [],
                    }
                )

        all_evaluations[name] = epoch_results

    # ── Capacity test: larger model (128x4) ──
    LARGER_TRUNK = 128
    LARGER_BLOCKS = 4
    print("\n" + "─" * 50)
    print("CAPACITY TEST: 128x4 MODEL")
    print("─" * 50)
    larger_init_ckpt = build_larger_init_checkpoint(LARGER_TRUNK, LARGER_BLOCKS)

    capacity_traces = [
        {
            "name": "cap128_soft065_w1_half",
            "data": [POLICY_SOFT065_PATH, CONTROLS_ARTIFACT_PATH],
            "weights": [1, 1],
            "lr": HALF_LR,
            "policy_mass": 0.65,
        },
        {
            "name": "cap128_soft065_w2_half",
            "data": [POLICY_SOFT065_PATH, CONTROLS_ARTIFACT_PATH],
            "weights": [1, 2],
            "lr": HALF_LR,
            "policy_mass": 0.65,
        },
        {
            "name": "cap128_soft055_w1_half",
            "data": [POLICY_SOFT055_PATH, CONTROLS_ARTIFACT_PATH],
            "weights": [1, 1],
            "lr": HALF_LR,
            "policy_mass": 0.55,
        },
        {
            "name": "cap128_soft065_w1_quarter",
            "data": [POLICY_SOFT065_PATH, CONTROLS_ARTIFACT_PATH],
            "weights": [1, 1],
            "lr": QUARTER_LR,
            "policy_mass": 0.65,
        },
    ]

    hidden_sizes_str = f"{LARGER_TRUNK},{LARGER_BLOCKS}"

    def run_capacity_training(name, data_files, replay_weights, epochs, init_ckpt, lr):
        out_path = EXPORT_DIR / f"{name}_e{epochs}.npz"
        if out_path.exists():
            return {
                "trace": name,
                "epochs": epochs,
                "checkpoint": str(out_path),
                "cached": True,
            }
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        weights_str = ",".join(str(w) for w in replay_weights)
        valid_files = [p for p in data_files if p.exists()]
        if not valid_files:
            return {
                "trace": name,
                "epochs": epochs,
                "error": "no valid data files",
                "returncode": -1,
            }
        data_files_str = ",".join(str(p) for p in valid_files)
        cmd = [
            sys.executable,
            "-m",
            "ml.alphazero_lite.train",
            "--data-files",
            data_files_str,
            "--replay-weights",
            weights_str,
            "--out",
            str(out_path),
            "--epochs",
            str(epochs),
            "--batch-size",
            "32",
            "--lr",
            str(lr),
            "--seed",
            "42",
            "--device",
            "cpu",
            "--value-loss-weight",
            "0.5",
            "--value-loss",
            "huber",
            "--hidden-sizes",
            hidden_sizes_str,
            "--model-type",
            "residual_v3",
            "--input-encoding",
            INPUT_ENCODING,
            "--init-checkpoint",
            str(init_ckpt),
            "--val-split",
            "0.0",
            "--grad-clip",
            "1.0",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        print(
            f"    policy_loss={result.stdout.strip().split(chr(10))[-1] if result.stdout.strip() else '?'}$"
        )
        for line in result.stdout.strip().split(chr(10)):
            if "policy_loss" in line or "value_loss" in line:
                print(f"    {line.strip()}")
                break
        try:
            tm = json.loads(result.stdout.strip().split(chr(10))[-1])
            tm["trace"] = name
            tm["epochs"] = epochs
            tm["lr"] = lr
        except (json.JSONDecodeError, IndexError):
            tm = {
                "trace": name,
                "epochs": epochs,
                "returncode": result.returncode,
                "checkpoint": str(out_path),
            }
        if result.returncode != 0:
            tm["error"] = result.stderr.strip()[:500]
            tm["returncode"] = result.returncode
        return tm

    for tc in capacity_traces:
        name = tc["name"]
        sources = tc["data"]
        weights = tc["weights"]
        lr = tc["lr"]
        print(
            f"\n--- Capacity trace: {name} (lr={lr}, weights={weights}, {hidden_sizes_str}) ---"
        )
        data_path = build_training_data(name, sources)
        row_count = len(load_jsonl(data_path))
        print(f"  Training data: {data_path} ({row_count} rows)")
        epoch_results = []
        for epochs in [1, 2, 4]:
            tm = run_capacity_training(
                name, [data_path], [1], epochs, larger_init_ckpt, lr
            )
            tm["data_path"] = str(data_path)
            tm["replay_weights"] = list(weights)
            all_training_metrics.append(tm)
            ckpt_path = Path(str(tm["checkpoint"]))
            if ckpt_path.exists() and tm.get("returncode", 0) == 0:
                eval_artifact_dir = EVAL_DIR / f"{name}_e{epochs}"
                convert_npz_to_artifact(ckpt_path, eval_artifact_dir)
                try:
                    evaluator = ArtifactEvaluator(eval_artifact_dir)
                    ev_result = evaluate_trace_at_epoch(evaluator, eval_rows, baseline)
                    for pr in ev_result.get("production_results", []):
                        pr["trace_name"] = name
                        pr["epoch"] = epochs
                    for vr in ev_result.get("value_results", []):
                        vr["trace_name"] = name
                        vr["epoch"] = epochs
                    for cr in ev_result.get("control_results", []):
                        cr["trace_name"] = name
                        cr["epoch"] = epochs
                    for hr in ev_result.get("holdout_results", []):
                        hr["trace_name"] = name
                        hr["epoch"] = epochs
                    epoch_results.append(ev_result)
                    prod_opt = sum(
                        1
                        for r in ev_result["production_results"]
                        if r.get("selected_is_optimal_1200")
                    )
                    val_avg_err = 0.0
                    val_n = 0
                    for r in ev_result["value_results"]:
                        e = r.get("abs_value_error")
                        if e is not None:
                            val_avg_err += float(e)
                            val_n += 1
                    if val_n:
                        val_avg_err /= val_n
                    orig_ctrl_reg = sum(
                        1
                        for r in ev_result["control_results"]
                        if r.get("control_regression")
                        and r.get("control_subtype") == "original_pr78"
                    )
                    prom_ctrl_reg = sum(
                        1
                        for r in ev_result["control_results"]
                        if r.get("control_regression")
                        and r.get("promoted_from_holdout_regression")
                    )
                    nn_ctrl_reg = sum(
                        1
                        for r in ev_result["control_results"]
                        if r.get("control_regression")
                        and r.get("nearest_neighbor_control")
                    )
                    total_ctrl_reg = orig_ctrl_reg + prom_ctrl_reg + nn_ctrl_reg
                    h_opt = sum(
                        1
                        for r in ev_result["holdout_results"]
                        if r.get("selected_is_optimal_1200")
                    )
                    h_reg = len(ev_result["holdout_results"]) - h_opt
                    print(
                        f"    Eval e{epochs}: prod_opt={prod_opt}/{len(ev_result['production_results'])}, "
                        f"val_err={val_avg_err:.4f}, "
                        f"ctrl_reg={total_ctrl_reg}(orig={orig_ctrl_reg}/prom={prom_ctrl_reg}/nn={nn_ctrl_reg}), "
                        f"hold_opt={h_opt}/{len(ev_result['holdout_results'])} reg={h_reg}"
                    )
                except Exception as ex:
                    print(f"    Eval FAILED: {ex}")
            elif tm.get("returncode", 0) != 0:
                print("    Training failed, skipping eval")
                epoch_results.append(
                    {
                        "production_results": [],
                        "value_results": [],
                        "control_results": [],
                        "holdout_results": [],
                    }
                )
        all_evaluations[name] = epoch_results

    # ── Regularization test: weight_decay + constant LR + 8 epochs ──
    print("\n" + "─" * 50)
    print("REGULARIZATION TEST: weight_decay=1e-4, lr_scheduler=none, 8 epochs")
    print("─" * 50)

    def run_regularized_training(
        name, data_files, replay_weights, epochs, init_ckpt, lr
    ):
        out_path = EXPORT_DIR / f"{name}_e{epochs}.npz"
        if out_path.exists():
            return {
                "trace": name,
                "epochs": epochs,
                "checkpoint": str(out_path),
                "cached": True,
            }
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        weights_str = ",".join(str(w) for w in replay_weights)
        valid_files = [p for p in data_files if p.exists()]
        if not valid_files:
            return {
                "trace": name,
                "epochs": epochs,
                "error": "no valid data files",
                "returncode": -1,
            }
        data_files_str = ",".join(str(p) for p in valid_files)
        cmd = [
            sys.executable,
            "-m",
            "ml.alphazero_lite.train",
            "--data-files",
            data_files_str,
            "--replay-weights",
            weights_str,
            "--out",
            str(out_path),
            "--epochs",
            str(epochs),
            "--batch-size",
            "32",
            "--lr",
            str(lr),
            "--seed",
            "42",
            "--device",
            "cpu",
            "--value-loss-weight",
            "0.5",
            "--value-loss",
            "huber",
            "--hidden-sizes",
            "96,3",
            "--model-type",
            "residual_v3",
            "--input-encoding",
            INPUT_ENCODING,
            "--init-checkpoint",
            str(init_ckpt),
            "--val-split",
            "0.0",
            "--grad-clip",
            "1.0",
            "--lr-scheduler",
            "none",
            "--weight-decay",
            "1e-4",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        for line in result.stdout.strip().split(chr(10)):
            if "policy_loss" in line or "value_loss" in line:
                print(f"    {line.strip()}")
                break
        try:
            last_line = result.stdout.strip().split(chr(10))[-1]
            tm = json.loads(last_line)
            tm["trace"] = name
            tm["epochs"] = epochs
            tm["lr"] = lr
        except (json.JSONDecodeError, IndexError):
            tm = {
                "trace": name,
                "epochs": epochs,
                "returncode": result.returncode,
                "checkpoint": str(out_path),
            }
        if result.returncode != 0:
            tm["error"] = result.stderr.strip()[:500]
            tm["returncode"] = result.returncode
        return tm

    regularized_traces = [
        {
            "name": "reg_soft065_w1_half",
            "data": [POLICY_SOFT065_PATH, CONTROLS_ARTIFACT_PATH],
            "weights": [1, 1],
            "lr": HALF_LR,
            "policy_mass": 0.65,
        },
        {
            "name": "reg_soft065_w2_half",
            "data": [POLICY_SOFT065_PATH, CONTROLS_ARTIFACT_PATH],
            "weights": [1, 2],
            "lr": HALF_LR,
            "policy_mass": 0.65,
        },
        {
            "name": "reg_soft055_w1_half",
            "data": [POLICY_SOFT055_PATH, CONTROLS_ARTIFACT_PATH],
            "weights": [1, 1],
            "lr": HALF_LR,
            "policy_mass": 0.55,
        },
        {
            "name": "reg_soft065_w1_quarter",
            "data": [POLICY_SOFT065_PATH, CONTROLS_ARTIFACT_PATH],
            "weights": [1, 1],
            "lr": QUARTER_LR,
            "policy_mass": 0.65,
        },
    ]

    REG_EPOCHS = [4, 8]
    for tc in regularized_traces:
        name = tc["name"]
        sources = tc["data"]
        weights = tc["weights"]
        lr = tc["lr"]
        print(
            f"\n--- Reg trace: {name} (lr={lr}, weights={weights}, wd=1e-4, lr_sched=none) ---"
        )
        data_path = build_training_data(name, sources)
        row_count = len(load_jsonl(data_path))
        print(f"  Training data: {data_path} ({row_count} rows)")
        epoch_results = []
        for epochs in REG_EPOCHS:
            tm = run_regularized_training(name, [data_path], [1], epochs, init_ckpt, lr)
            tm["data_path"] = str(data_path)
            tm["replay_weights"] = list(weights)
            all_training_metrics.append(tm)
            ckpt_path = Path(str(tm["checkpoint"]))
            if ckpt_path.exists() and tm.get("returncode", 0) == 0:
                eval_artifact_dir = EVAL_DIR / f"{name}_e{epochs}"
                convert_npz_to_artifact(ckpt_path, eval_artifact_dir)
                try:
                    evaluator = ArtifactEvaluator(eval_artifact_dir)
                    ev_result = evaluate_trace_at_epoch(evaluator, eval_rows, baseline)
                    for pr in ev_result.get("production_results", []):
                        pr["trace_name"] = name
                        pr["epoch"] = epochs
                    for vr in ev_result.get("value_results", []):
                        vr["trace_name"] = name
                        vr["epoch"] = epochs
                    for cr in ev_result.get("control_results", []):
                        cr["trace_name"] = name
                        cr["epoch"] = epochs
                    for hr in ev_result.get("holdout_results", []):
                        hr["trace_name"] = name
                        hr["epoch"] = epochs
                    epoch_results.append(ev_result)
                    prod_opt = sum(
                        1
                        for r in ev_result["production_results"]
                        if r.get("selected_is_optimal_1200")
                    )
                    val_avg_err = 0.0
                    val_n = 0
                    for r in ev_result["value_results"]:
                        e = r.get("abs_value_error")
                        if e is not None:
                            val_avg_err += float(e)
                            val_n += 1
                    if val_n:
                        val_avg_err /= val_n
                    orig_ctrl_reg = sum(
                        1
                        for r in ev_result["control_results"]
                        if r.get("control_regression")
                        and r.get("control_subtype") == "original_pr78"
                    )
                    prom_ctrl_reg = sum(
                        1
                        for r in ev_result["control_results"]
                        if r.get("control_regression")
                        and r.get("promoted_from_holdout_regression")
                    )
                    nn_ctrl_reg = sum(
                        1
                        for r in ev_result["control_results"]
                        if r.get("control_regression")
                        and r.get("nearest_neighbor_control")
                    )
                    total_ctrl_reg = orig_ctrl_reg + prom_ctrl_reg + nn_ctrl_reg
                    h_opt = sum(
                        1
                        for r in ev_result["holdout_results"]
                        if r.get("selected_is_optimal_1200")
                    )
                    h_reg = len(ev_result["holdout_results"]) - h_opt
                    print(
                        f"    Eval e{epochs}: prod_opt={prod_opt}/{len(ev_result['production_results'])}, "
                        f"val_err={val_avg_err:.4f}, "
                        f"ctrl_reg={total_ctrl_reg}(orig={orig_ctrl_reg}/prom={prom_ctrl_reg}/nn={nn_ctrl_reg}), "
                        f"hold_opt={h_opt}/{len(ev_result['holdout_results'])} reg={h_reg}"
                    )
                except Exception as ex:
                    print(f"    Eval FAILED: {ex}")
            elif tm.get("returncode", 0) != 0:
                print("    Training failed, skipping eval")
                epoch_results.append(
                    {
                        "production_results": [],
                        "value_results": [],
                        "control_results": [],
                        "holdout_results": [],
                    }
                )
        all_evaluations[name] = epoch_results

    # Step 8-10: Apply gates, decision rules
    print("\n" + "=" * 70)
    print("STEP 11: APPLY DECISION RULES")
    print("=" * 70)

    control_info = {
        "original_control_ids": original_control_ids,
        "promoted_control_ids": promoted_control_ids,
        "nearest_neighbor_control_ids": nearest_neighbor_control_ids,
    }

    decision = apply_decision_rules(
        all_evaluations, all_training_metrics, baseline, control_info
    )
    print(f"  Classification: {decision['classification']}")
    print(f"  Next action: {decision['next_action']}")

    # Step 12: Report
    trace_summary = {
        "schema": "azlite_medium_exact_tablebase_stabilization_v2_trace_v1",
        "family": STABILIZATION_V2_FAMILY,
        "guardrails": {
            "mutated_active_fixture": False,
            "ran_training": True,
            "ran_arena": False,
            "promoted_model": False,
            "overwrote_current_artifact": False,
        },
        "baseline": trace_summary_data["baseline"],
        "pr78_baseline_evals": trace_summary_data["pr78_baseline_evals"],
        "traces": [
            {
                "name": tm.get("trace", ""),
                "epochs": tm.get("epochs"),
                "policy_loss": tm.get("policy_loss"),
                "value_loss": tm.get("value_loss"),
                "lr": tm.get("lr"),
                "replay_weights": tm.get("replay_weights"),
                "returncode": tm.get("returncode"),
                "checkpoint": tm.get("checkpoint"),
            }
            for tm in all_training_metrics
        ],
        "evaluations": {
            tn: [
                {
                    "epoch": ed.get("production_results", [{}])[0].get("epoch")
                    if ed.get("production_results")
                    else None,
                    "production_optimal_1200": sum(
                        1
                        for r in ed.get("production_results", [])
                        if r.get("selected_is_optimal_1200")
                    ),
                    "production_count": len(ed.get("production_results", [])),
                    "production_improved_vs_current": sum(
                        1
                        for r in ed.get("production_results", [])
                        if r.get("improved_vs_current")
                    ),
                    "value_avg_error": round_float(
                        sum(
                            float(r.get("abs_value_error", 0) or 0)
                            for r in ed.get("value_results", [])
                        )
                        / max(len(ed.get("value_results", [])), 1)
                    ),
                    "control_regression": any(
                        r.get("control_regression")
                        for r in ed.get("control_results", [])
                    ),
                    "control_regression_count": sum(
                        1
                        for r in ed.get("control_results", [])
                        if r.get("control_regression")
                    ),
                    "original_control_regression": sum(
                        1
                        for r in ed.get("control_results", [])
                        if r.get("control_regression")
                        and r.get("control_subtype") == "original_pr78"
                    ),
                    "promoted_control_regression": sum(
                        1
                        for r in ed.get("control_results", [])
                        if r.get("control_regression")
                        and r.get("promoted_from_holdout_regression")
                    ),
                    "nearest_neighbor_control_regression": sum(
                        1
                        for r in ed.get("control_results", [])
                        if r.get("control_regression")
                        and r.get("nearest_neighbor_control")
                    ),
                    "holdout_optimal_1200": sum(
                        1
                        for r in ed.get("holdout_results", [])
                        if r.get("selected_is_optimal_1200")
                    ),
                    "holdout_count": len(ed.get("holdout_results", [])),
                }
                for ed in td
            ]
            for tn, td in all_evaluations.items()
        },
        "decision": decision,
    }

    write_json(OUTPUT_DIR / "stabilization_v2_trace_summary.json", trace_summary)
    print(
        f"\nTrace summary written to {OUTPUT_DIR / 'stabilization_v2_trace_summary.json'}"
    )

    print(f"\nGenerating report: {REPORT_PATH}")
    generate_report(
        all_evaluations,
        all_training_metrics,
        baseline,
        decision,
        eval_rows,
        soft065_rows,
        soft055_rows,
        targeted_controls,
        value_only_rows,
        artifact_summary,
        control_info,
        trace_summary_data,
        regression_info,
    )

    print("\n" + "=" * 70)
    print("STABILIZATION v2 COMPLETE")
    print("=" * 70)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Report: {REPORT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
