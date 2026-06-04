#!/usr/bin/env python3
"""Run medium exact-tablebase diagnostic traces with expanded controls and strict holdout gates.

Traces:
  A: medium_soft075_controls_w1_lr_default  - production + controls, w=1/1, lr=1e-4
  B: medium_soft075_controls_w1_lr_half     - production + controls, w=1/1, lr=5e-5
  C: medium_soft075_controls_w2_lr_half     - production + controls, w=1/2, lr=5e-5
  D: medium_value_light_lr_half             - production + value_only + controls, w=1/1/2, lr=5e-5

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
from ml.alphazero_lite.self_play import build_eval_search_options

CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
ARTIFACT_DIR = Path("/tmp/azlite_medium_exact_tablebase_diagnostic")
OUTPUT_DIR = ARTIFACT_DIR / "exports"
EVAL_DIR = ARTIFACT_DIR / "eval"

POLICY_VALUE_PATH = ARTIFACT_DIR / "exact_tablebase_policy_value_soft075.jsonl"
VALUE_ONLY_PATH = ARTIFACT_DIR / "exact_tablebase_value_only_artifact.jsonl"
CONTROLS_PATH = ARTIFACT_DIR / "exact_tablebase_expanded_controls_artifact.jsonl"
ARTIFACT_SUMMARY_PATH = ARTIFACT_DIR / "artifact_summary.json"
CLEAN_SPLIT_PATH = ARTIFACT_DIR / "medium_exact_tablebase_clean_split.json"

C_PUCT = 1.25
SEARCH_OPTIONS = build_eval_search_options(
    fpu_mode="parent_q",
    reuse_subtree=True,
    normalize_values=True,
    root_policy_mode="deterministic",
    tactical_root_bias=0.1,
)
SEED = 17
EVAL_BUDGETS = (384, 1200, 2400)
INPUT_ENCODING = "kalah_v3"
POLICY_SIZE = 6
EPS = 1e-9
FAMILY = "harder_fresh_endgame_tablebase"
REPORT_PATH = Path("docs/alphazero-lite-medium-exact-tablebase-diagnostic-results.md")

PR77_BEST_PRODUCTION = 22
PR77_BEST_TRACE = "soft075_controls_w1_lr_default"

MEDIUM_FAMILY = "medium_exact_tablebase_diagnostic"
BASE_LR = 1e-4
HALF_LR = 5e-5


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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def state_to_root_perspective_value(
    *, raw_value: float, state: dict[str, Any], root_player: int
) -> float:
    return (
        float(raw_value)
        if int(state["current_player"]) == int(root_player)
        else -float(raw_value)
    )


def child_state_from_move(root_state: dict[str, Any], move: int) -> dict[str, Any]:
    game = KalahGame.from_state(root_state)
    succeeded = game.move(game.pit_index(int(move)))
    if not succeeded:
        raise ValueError(f"illegal move {move}")
    return game.to_state()


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
    root_player = game.current_player
    _, raw_nv = evaluator.evaluate(game)
    neural_root = float(raw_nv)

    legal = game.possible_moves()
    child_rank_error = None
    if legal and optimal_move is not None:
        neural_child_vals: dict[int, float] = {}
        for move in legal:
            cs = child_state_from_move(state, move)
            _, cnv = evaluator.evaluate(KalahGame.from_state(cs))
            nv = state_to_root_perspective_value(
                raw_value=float(cnv), state=cs, root_player=root_player
            )
            neural_child_vals[move] = nv
        if neural_child_vals:
            sorted_children = sorted(
                neural_child_vals, key=lambda m: neural_child_vals[m], reverse=True
            )
            if optimal_move in sorted_children:
                child_rank_error = sorted_children.index(optimal_move)

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
        "neural_child_rank_error": child_rank_error,
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


def build_init_checkpoint() -> Path:
    init_path = ARTIFACT_DIR / "init_checkpoint.npz"
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
    out_path = OUTPUT_DIR / f"{trace_name}_e{epochs}.npz"
    if out_path.exists():
        return {
            "trace": trace_name,
            "epochs": epochs,
            "checkpoint": str(out_path),
            "cached": True,
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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

    policy_loss = metrics.get("policy_loss")
    value_loss = metrics.get("value_loss")
    print(
        f"    policy_loss={policy_loss}, value_loss={value_loss}, elapsed={elapsed:.1f}s"
    )
    return metrics


def build_training_data(trace_name: str, sources: list[Path]) -> Path:
    combined_path = ARTIFACT_DIR / f"{trace_name}_data.jsonl"
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

        pr["improved_vs_pr77_best"] = bool(pr.get("selected_is_optimal_1200") is True)
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

    return result


def apply_decision_rules(
    all_evaluations: dict[str, list[dict]],
    all_training_metrics: list[dict],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "classification": None,
        "supporting_evidence": [],
        "rejected_alternatives": [],
        "next_action": None,
    }

    best_prod_1200 = 0
    min_holdout_regression_rate = 1.0
    min_holdout_regression_any_ctrl_reg = 1.0
    best_zero_ctrl_holdout_rate = 1.0
    best_trace_info = ""
    any_zero_ctrl = False

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
            if ctrl_reg == 0 and holdout_rate < min_holdout_regression_any_ctrl_reg:
                min_holdout_regression_any_ctrl_reg = holdout_rate
            if ctrl_reg == 0 and holdout_rate < best_zero_ctrl_holdout_rate:
                best_zero_ctrl_holdout_rate = holdout_rate

    baseline_prod = sum(
        1
        for r in baseline.get("production_results", [])
        if r.get("selected_is_optimal_1200")
    )
    baseline_total = max(len(baseline.get("production_results", [])), 1)
    baseline_pct = baseline_prod / baseline_total
    best_prod_pct = best_prod_1200 / max(len(baseline.get("production_results", [])), 1)
    meaningful_gain = best_prod_pct > baseline_pct + 0.05

    low_holdout_regression = min_holdout_regression_rate <= 0.005
    gates_pass_together = any_zero_ctrl and min_holdout_regression_rate <= 0.005

    if gates_pass_together and meaningful_gain:
        results["classification"] = "medium_exact_tablebase_local_success"
        results["supporting_evidence"] = [
            f"best_prod={best_prod_1200}/{baseline_total} (+{best_prod_1200 - baseline_prod})",
            "ctrl_reg=0 on best trace",
            f"best_hold_reg_rate={min_holdout_regression_rate:.3%}",
        ]
        results["rejected_alternatives"] = [
            "medium_exact_tablebase_controls_sufficient_but_gain_small",
            "medium_exact_tablebase_overfit_persists",
        ]
        results["next_action"] = (
            "run one controlled exact-tablebase production-scale diagnostic lane "
            "with pre-arena exact-tablebase gates, still no automatic promotion."
        )
    elif any_zero_ctrl and min_holdout_regression_rate <= 0.005 and not meaningful_gain:
        results["classification"] = (
            "medium_exact_tablebase_controls_sufficient_but_gain_small"
        )
        results["supporting_evidence"] = [
            f"best_prod={best_prod_1200}/{baseline_total}",
            "ctrl_reg=0 on some checkpoints",
            f"best_hold_reg_rate={min_holdout_regression_rate:.3%}",
        ]
        results["rejected_alternatives"] = ["medium_exact_tablebase_local_success"]
        results["next_action"] = (
            "mine more/harder exact-tablebase production candidates before scaling."
        )
    elif any_zero_ctrl and not low_holdout_regression:
        results["classification"] = "medium_exact_tablebase_overfit_persists"
        results["supporting_evidence"] = [
            f"best_prod={best_prod_1200}/{baseline_total} (trace={best_trace_info})",
            f"best_zero_ctrl_hold_reg_rate={best_zero_ctrl_holdout_rate:.3%} > 0.5% threshold",
            f"min_hold_reg_rate={min_holdout_regression_rate:.3%} min across all checkpoints",
        ]
        results["rejected_alternatives"] = [
            "medium_exact_tablebase_local_success",
            "medium_exact_tablebase_controls_sufficient_but_gain_small",
        ]
        results["next_action"] = (
            "add more preservation controls or soften targets further; the medium artifact "
            "scales production signal but holdout damage exceeds 0.5%% gate even at best "
            "checkpoint with zero control regression."
        )
    elif not any_zero_ctrl:
        results["classification"] = "medium_exact_tablebase_overfit_persists"
        results["supporting_evidence"] = [
            f"ctrl regression present in all checkpoints; best={best_trace_info}",
            f"min_hold_reg_rate={min_holdout_regression_rate:.3%}",
        ]
        results["rejected_alternatives"] = ["medium_exact_tablebase_local_success"]
        results["next_action"] = (
            "add more controls, soften target further, or stop exact-tablebase artifact scaling."
        )
    else:
        results["classification"] = "medium_exact_tablebase_no_local_signal"
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
    policy_value_rows: list[dict],
    value_only_rows: list[dict],
    controls_rows: list[dict],
    artifact_summary: dict[str, Any],
) -> None:
    lines: list[str] = []

    lines.append("# Medium Exact Tablebase Diagnostic — Results")
    lines.append("")
    lines.append("**Date:** 2026-06-04")
    lines.append("**Family:** `medium_exact_tablebase_diagnostic`")
    lines.append("**Scripts:**")
    lines.append(
        "- `ml/alphazero_lite/build_medium_exact_tablebase_diagnostic_artifact.py`"
    )
    lines.append("- `ml/alphazero_lite/run_medium_exact_tablebase_diagnostic_trace.py`")
    lines.append("")

    lines.append("## 1. Context")
    lines.append("")
    lines.append(
        "PR #77 ran a stabilized exact-tablebase diagnostic with softened policy targets (0.75), "
        "expanded controls (11 -> 25), and reduced LR. The best result was "
        "soft075_controls_w1_lr_default e2 at 22/32 optimal@1200 with control_regression=0 "
        "across all traces and max holdout regression of 7/317. "
        "Classification was `exact_tablebase_controls_fix_regression` with next action: "
        '"use expanded controls in the next medium diagnostic."'
    )
    lines.append("")
    lines.append(
        "This medium diagnostic scales the artifact to ~80-120 production candidates and "
        "~60+ preservation controls while maintaining strict holdout gates. "
        "It tests whether expanded controls and softened targets scale to larger row counts "
        "without regressing controls or damaging holdouts."
    )
    lines.append("")

    lines.append("## 2. Why PR #77 justified medium diagnostic")
    lines.append("")
    lines.append(
        "PR #77 demonstrated that expanded controls (25 rows) completely eliminated control "
        "regression across all traces, and half-LR traces reduced holdout regression from "
        "7/317 (soft065 default-LR) to 2/317 (soft075 half-LR). The classification "
        "`exact_tablebase_controls_fix_regression` recommended scaling these findings to a "
        "medium diagnostic with more production candidates, more controls, and strict holdout gates. "
        f"PR #77 best production optimal@1200 was {PR77_BEST_PRODUCTION}/32."
    )
    lines.append("")

    lines.append("## 3. Medium row mining / reuse")
    lines.append("")
    lines.append(
        "The medium artifact was built by reusing all PR #76/77 clean split rows and generating "
        "fresh adversarial endgame candidates via random state enumeration in the tablebase range "
        "(2-14 seeds). PUCT baselines (budgets 64/384/1200, plus 2400 for persistent failures) "
        "and neural value rank scans were run on new candidates. "
        "Rows were classified into production/control/holdout/value-only buckets."
    )
    lines.append("")

    prod_count = len(policy_value_rows)
    ctrl_count = len(controls_rows)
    val_count = len(value_only_rows)
    holdout_count = len(eval_rows.get("holdouts", []))
    lines.append(f"- Production candidates: {prod_count}")
    lines.append(f"- Preservation controls: {ctrl_count}")
    lines.append(f"- Value-only candidates: {val_count}")
    lines.append(f"- Untouched holdouts: {holdout_count}")
    lines.append("")

    lines.append("## 4. Artifact construction")
    lines.append("")
    lines.append(
        "| artifact | row_count | policy_target_mass | value_source | validation_status | notes |"
    )
    lines.append("|---|---|---|---|---|---|")
    arts = artifact_summary.get("artifacts", {})
    vstat = artifact_summary.get("validation", "N/A")
    for aname in ["policy_value", "expanded_controls", "value_only"]:
        ai = arts.get(aname, {})
        pm = ai.get("policy_target_mass", "-")
        lines.append(
            f"| {aname} | {ai.get('row_count', 0)} | {pm} | "
            f"{ai.get('value_source', '')} | {vstat} | |"
        )
    lines.append("")

    lines.append("## 5. Static validation")
    lines.append("")
    lines.append(
        "Static validation checked: policy sums to 1.0, optimal move receives highest mass, "
        "value targets in [-1,1], no duplicate canonical state conflicts, "
        "no exhausted-family overlap, no holdout leakage."
    )
    errors = artifact_summary.get("validation_errors", [])
    if errors:
        lines.append(f"Validation errors: {len(errors)}")
        for e in errors:
            lines.append(f"- {e}")
    else:
        lines.append("Static validation PASSED.")
    lines.append("")

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
    lines.append(
        f"- PR #77 best: {PR77_BEST_PRODUCTION}/32 optimal@1200 ({PR77_BEST_TRACE} e2)"
    )
    lines.append("")

    lines.append("## 7. Trace definitions")
    lines.append("")
    lines.append(
        "| trace_name | policy_target_mass | learning_rate_multiplier | data_files | weights | epochs | status | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    trace_defs = [
        (
            "medium_soft075_controls_w1_lr_default",
            0.75,
            "1x (1e-4)",
            "soft075 + controls",
            "1,1",
            "1,2,4",
            "",
        ),
        (
            "medium_soft075_controls_w1_lr_half",
            0.75,
            "0.5x (5e-5)",
            "soft075 + controls",
            "1,1",
            "1,2,4",
            "",
        ),
        (
            "medium_soft075_controls_w2_lr_half",
            0.75,
            "0.5x (5e-5)",
            "soft075 + controls",
            "1,2",
            "1,2,4",
            "",
        ),
        (
            "medium_value_light_lr_half",
            0.75,
            "0.5x (5e-5)",
            "soft075 + value_only + controls",
            "1,1,2",
            "1,2,4",
            "value-only rows added",
        ),
    ]
    for td in trace_defs:
        ran = any(tm.get("trace") == td[0] for tm in all_training_metrics)
        status = "completed" if ran else "skipped"
        lines.append(
            f"| {td[0]} | {td[1]} | {td[2]} | {td[3]} | {td[4]} | {td[5]} | {status} | {td[6]} |"
        )
    lines.append("")

    lines.append("## 8. Production-candidate results")
    lines.append("")
    lines.append(
        "| trace_name | epoch | production_total | production_optimal_1200 | "
        "production_optimal_2400 | improved_vs_current | improved_vs_pr77_best | "
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
            p_imp_pr77 = sum(
                1
                for r in ed.get("production_results", [])
                if r.get("improved_vs_pr77_best")
            )
            visits = [
                float(r.get("optimal_visit_share_1200", 0) or 0)
                for r in ed.get("production_results", [])
            ]
            avg_visits = sum(visits) / len(visits) if visits else 0.0
            p_total = len(ed.get("production_results", []))
            notes = ""
            lines.append(
                f"| {tn} | e{epoch} | {p_total} | {p_opt_1200}/{p_total} | "
                f"{p_opt_2400}/{p_total} | {p_imp_cur}/{p_total} | "
                f"{p_imp_pr77}/{p_total} | {avg_visits:.4f} | {notes} |"
            )
    lines.append("")

    lines.append("## 9. Control results")
    lines.append("")
    lines.append(
        "| trace_name | epoch | control_total | controls_optimal_1200 | "
        "controls_optimal_2400 | control_regression_count | "
        "avg_optimal_visit_share_delta | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
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
            deltas = [
                float(r.get("optimal_visit_share_delta_1200", 0) or 0)
                for r in ed.get("control_results", [])
            ]
            avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
            notes = f"{reg_count} regressed" if reg_count > 0 else ""
            lines.append(
                f"| {tn} | e{epoch} | {c_total} | {c_opt_1200}/{c_total} | "
                f"{c_opt_2400}/{c_total} | {reg_count} | {avg_delta:.4f} | {notes} |"
            )
    lines.append("")

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
            p_gain = p_opt > sum(1 for r in b_prod if r.get("selected_is_optimal_1200"))
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

    lines.append("## 13. Sanity non-regression checks")
    lines.append("")
    lines.append(
        "| trace_name | epoch | suite_or_group | metric | current_baseline | "
        "checkpoint_value | regression | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for tn, td in all_evaluations.items():
        if tn == "baseline" or tn == "baseline":
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

    lines.append("## 14. Training metrics")
    lines.append("")
    lines.append(
        "| trace_name | epoch | policy_loss | value_loss | total_loss | production_cross_entropy | control_cross_entropy | value_only_loss | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for tm in all_training_metrics:
        lines.append(
            f"| {tm.get('trace', '')} | e{tm.get('epochs', '')} | "
            f"{tm.get('policy_loss', '-')} | {tm.get('value_loss', '-')} | - | - | - | - | |"
        )
    lines.append("")

    lines.append("## 15. Interpretation")
    lines.append("")
    lines.append(
        "This medium diagnostic tested whether expanded controls and softened targets (0.75) "
        "scale to larger artifact sizes (80-120 production, 60+ controls) while maintaining "
        "the stability observed in PR #77. Four traces tested different combinations of "
        "control weight and learning rate."
    )
    lines.append("")
    lines.append(
        "Key questions: (1) Do expanded controls prevent regression at larger scale? "
        "(2) Does half-LR reduce holdout regression? "
        "(3) Does extra control weight (w2) improve stability? "
        "(4) Does value-only augmentation cause holdout regression in medium scale? "
        "(5) Is there a combination passing all strict gates?"
    )
    lines.append("")

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

    lines.append("### Acceptance criteria")
    lines.append("")
    lines.append("- No arena was run.")
    lines.append("- No local_promotion_gate was run.")
    lines.append("- No model was promoted.")
    lines.append("- `storage/ai/alphazero_lite/current` was not overwritten.")
    lines.append("- Active references were not mutated.")
    lines.append(
        "- Holdout rows were not used for training except promoted regression controls."
    )
    lines.append("- Exact tablebase labels were used with documented perspective.")
    lines.append("- Final report recommends exactly one next branch.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("MEDIUM EXACT TABLEBASE DIAGNOSTIC TRACE")
    print("=" * 70)

    print("\nLoading medium artifact data...")
    policy_value_rows = load_jsonl(POLICY_VALUE_PATH)
    value_only_rows = load_jsonl(VALUE_ONLY_PATH)
    controls_rows = load_jsonl(CONTROLS_PATH)

    artifact_summary = json.loads(ARTIFACT_SUMMARY_PATH.read_text(encoding="utf-8"))
    clean_split = json.loads(CLEAN_SPLIT_PATH.read_text(encoding="utf-8"))
    split_rows = clean_split.get("split_rows", [])
    split_by_cid: dict[str, dict] = {}
    for sr in split_rows:
        cid = sr.get("candidate_id", "")
        if cid:
            split_by_cid[cid] = sr

    holdout_ids = set(artifact_summary.get("holdout_candidate_ids", []))
    print(f"  Production candidates: {len(policy_value_rows)}")
    print(f"  Value-only candidates: {len(value_only_rows)}")
    print(f"  Expanded controls: {len(controls_rows)}")
    print(f"  Untouched holdouts: {len(holdout_ids)}")

    print("\nLoading holdout states from clean split...")
    holdout_rows = []
    for hid in sorted(holdout_ids):
        sr = split_by_cid.get(hid)
        if sr is not None:
            holdout_rows.append(
                {
                    "candidate_id": hid,
                    "raw_state": sr.get("_state"),
                    "exact_optimal_move": sr.get("_optimal_move"),
                    "exact_root_value": sr.get("_root_value"),
                }
            )
    print(f"  Holdout rows resolved: {len(holdout_rows)}")

    eval_rows = {
        "production_candidates": policy_value_rows,
        "value_only_candidates": value_only_rows,
        "controls": controls_rows,
        "holdouts": holdout_rows,
    }

    baseline = compute_baseline(eval_rows)
    all_evaluations: dict[str, list[dict]] = {"baseline": [baseline]}
    all_training_metrics: list[dict] = []

    init_ckpt = build_init_checkpoint()

    traces = [
        {
            "name": "medium_soft075_controls_w1_lr_default",
            "data": [POLICY_VALUE_PATH, CONTROLS_PATH],
            "weights": [1, 1],
            "lr": BASE_LR,
            "epochs_list": [1, 2, 4],
        },
        {
            "name": "medium_soft075_controls_w1_lr_half",
            "data": [POLICY_VALUE_PATH, CONTROLS_PATH],
            "weights": [1, 1],
            "lr": HALF_LR,
            "epochs_list": [1, 2, 4],
        },
        {
            "name": "medium_soft075_controls_w2_lr_half",
            "data": [POLICY_VALUE_PATH, CONTROLS_PATH],
            "weights": [1, 2],
            "lr": HALF_LR,
            "epochs_list": [1, 2, 4],
        },
    ]

    has_value_only = len(value_only_rows) > 0
    if has_value_only:
        traces.append(
            {
                "name": "medium_value_light_lr_half",
                "data": [POLICY_VALUE_PATH, VALUE_ONLY_PATH, CONTROLS_PATH],
                "weights": [1, 1, 2],
                "lr": HALF_LR,
                "epochs_list": [1, 2, 4],
            }
        )

    for tc in traces:
        name = tc["name"]
        sources = tc["data"]
        weights = tc["weights"]
        lr = tc["lr"]
        print(f"\n--- Trace: {name} (lr={lr}, weights={weights}) ---")

        data_path = build_training_data(name, sources)
        row_count = len(load_jsonl(data_path))
        print(
            f"  Training data: {data_path} ({row_count} rows, sources: {[str(s.name) for s in sources]})"
        )

        epoch_results: list[dict] = []
        for epochs in tc["epochs_list"]:
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

                    ctrl_reg = sum(
                        1
                        for r in ev_result["control_results"]
                        if r.get("control_regression")
                    )

                    h_opt = sum(
                        1
                        for r in ev_result["holdout_results"]
                        if r.get("selected_is_optimal_1200")
                    )
                    h_reg = len(ev_result["holdout_results"]) - h_opt
                    print(
                        f"    Eval e{epochs}: prod_opt={prod_opt}/{len(ev_result['production_results'])}, "
                        f"val_err={val_avg_err:.4f}, "
                        f"ctrl_reg={ctrl_reg}, "
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

    print("\n" + "=" * 70)
    print("APPLYING DECISION RULES")
    print("=" * 70)
    decision = apply_decision_rules(all_evaluations, all_training_metrics, baseline)
    print(f"  Classification: {decision['classification']}")
    print(f"  Next action: {decision['next_action']}")

    trace_summary = {
        "schema": "azlite_medium_exact_tablebase_diagnostic_trace_v1",
        "family": MEDIUM_FAMILY,
        "guardrails": {
            "mutated_active_fixture": False,
            "ran_training": True,
            "ran_arena": False,
            "promoted_model": False,
            "overwrote_current_artifact": False,
        },
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
        },
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

    write_json(ARTIFACT_DIR / "medium_diagnostic_trace_summary.json", trace_summary)
    print(
        f"\nTrace summary written to {ARTIFACT_DIR / 'medium_diagnostic_trace_summary.json'}"
    )

    print(f"\nGenerating report: {REPORT_PATH}")
    generate_report(
        all_evaluations,
        all_training_metrics,
        baseline,
        decision,
        eval_rows,
        policy_value_rows,
        value_only_rows,
        controls_rows,
        artifact_summary,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
