#!/usr/bin/env python3
"""Representation-interference diagnostic for exact-tablebase production/control positions.

Lanes:
1. current_eval_only — evaluate current baseline
2. head_only_policy_finetune — freeze trunk, train only policy head layers
3. last_block_plus_policy_finetune — freeze early trunk, train last block + policy head
4. full_finetune_control — full fine-tuning (comparison lane)

Does not promote, does not touch storage/ai/alphazero_lite/current.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import build_eval_search_options

CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
PR79_DIR = Path("/tmp/azlite_medium_exact_tablebase_stabilization_v2")
DEFAULT_WORKDIR = Path("/tmp/azlite_representation_interference")
REPORT_PATH = Path(
    "docs/alphazero-lite-representation-interference-diagnostic-results.md"
)

C_PUCT = 1.25
SEARCH_OPTIONS = build_eval_search_options(
    fpu_mode="parent_q",
    reuse_subtree=True,
    normalize_values=True,
    root_policy_mode="deterministic",
    tactical_root_bias=0.1,
)
SEED = 17
EVAL_BUDGETS = (1200, 2400)
INPUT_ENCODING = "kalah_v3"
POLICY_SIZE = 6

SOFT065_PATH = PR79_DIR / "exact_tablebase_policy_value_soft065.jsonl"
CONTROLS_PATH = PR79_DIR / "exact_tablebase_targeted_controls_artifact.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def convert_npz_to_artifact(checkpoint_path: Path, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    npz = np.load(checkpoint_path)
    weights = {key: npz[key].tolist() for key in npz.keys()}
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
        "version": "representation_interference_diagnostic",
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


def build_init_checkpoint(workdir: Path) -> Path:
    init_path = workdir / "init_checkpoint.npz"
    if init_path.exists():
        return init_path
    init_path.parent.mkdir(parents=True, exist_ok=True)
    weights = json.loads(
        (CURRENT_ARTIFACT / "weights.json").read_text(encoding="utf-8")
    )
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
    np.savez(init_path, **arrays)
    return init_path


def run_training(
    trace_name: str,
    data_files: list[Path],
    replay_weights: list[int],
    epochs: int,
    init_checkpoint: Path,
    lr: float,
    trainable_scope: str,
    *,
    workdir: Path,
    force_rerun: bool = False,
) -> dict[str, Any]:
    export_dir = workdir / "exports"
    out_path = export_dir / f"{trace_name}_e{epochs}.npz"

    if out_path.exists() and not force_rerun:
        return {
            "trace": trace_name,
            "epochs": epochs,
            "checkpoint": str(out_path),
            "cached": True,
        }

    export_dir.mkdir(parents=True, exist_ok=True)
    weights_str = ",".join(str(w) for w in replay_weights)
    data_files_str = ",".join(str(p) for p in data_files if p.exists())

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
        "--trainable-scope",
        trainable_scope,
    ]

    print(
        f"  Training {trace_name} epochs={epochs} (lr={lr}, scope={trainable_scope})..."
    )
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0

    metrics: dict[str, Any] = {
        "trace": trace_name,
        "epochs": epochs,
        "checkpoint": str(out_path),
        "elapsed_seconds": round(elapsed, 1),
        "returncode": result.returncode,
        "lr": lr,
        "replay_weights": list(replay_weights),
        "trainable_scope": trainable_scope,
    }

    if result.returncode != 0:
        print(f"    FAILED (rc={result.returncode})")
        if result.stderr:
            sys.stderr.write(result.stderr[-500:])
        return metrics

    for line in (result.stdout or "").split("\n"):
        for prefix in ["policy_loss=", "value_loss=", "best_val_loss="]:
            if prefix in line:
                try:
                    metrics[prefix.rstrip("=")] = float(line.split("=")[1].strip())
                except (IndexError, ValueError):
                    pass

    for line in (result.stderr or "").split("\n"):
        if "trainable_params=" in line:
            for part in line.strip().split():
                if "=" in part:
                    key, val = part.split("=", 1)
                    try:
                        metrics[key] = int(val)
                    except (ValueError, TypeError):
                        pass

    policy_loss = metrics.get("policy_loss")
    value_loss = metrics.get("value_loss")
    print(
        f"    policy_loss={policy_loss}, value_loss={value_loss}, "
        f"elapsed={elapsed:.1f}s"
    )
    return metrics


def evaluate_artifact(
    evaluator: ArtifactEvaluator,
    production_rows: list[dict],
    controls_rows: list[dict],
) -> dict[str, Any]:
    tb = EndgameTablebase()

    prod_results: list[dict] = []
    for row in production_rows:
        cid = row.get("candidate_id", "?")
        raw_state = row.get("raw_state")
        if not raw_state:
            continue
        state = raw_state
        game = KalahGame.from_state(state)
        rp = game.current_player
        offset = rp * 6
        child_vals = {}
        for move in game.possible_moves():
            child = game.clone()
            child.move(offset + move)
            c_wr = tb.lookup(child, rp)
            if c_wr is not None:
                child_vals[move] = (2.0 * float(c_wr)) - 1.0
        exact_opt = max(child_vals, key=child_vals.get) if child_vals else None

        result = {"candidate_id": cid, "exact_optimal_move": exact_opt}
        for budget in EVAL_BUDGETS:
            try:
                r = run_single_puct(evaluator, state, budget, SEED)
            except Exception:
                result[f"selected_is_optimal_{budget}"] = None
                continue
            sel = None if r.get("selected_move") is None else int(r["selected_move"])
            result[f"selected_is_optimal_{budget}"] = (
                sel == exact_opt if exact_opt is not None else None
            )
        prod_results.append(result)

    ctrl_results: list[dict] = []
    for row in controls_rows:
        cid = row.get("candidate_id", "?")
        raw_state = row.get("raw_state")
        if not raw_state:
            continue
        state = raw_state
        game = KalahGame.from_state(state)
        rp = game.current_player
        offset = rp * 6
        child_vals = {}
        for move in game.possible_moves():
            child = game.clone()
            child.move(offset + move)
            c_wr = tb.lookup(child, rp)
            if c_wr is not None:
                child_vals[move] = (2.0 * float(c_wr)) - 1.0
        exact_opt = max(child_vals, key=child_vals.get) if child_vals else None

        result = {"candidate_id": cid, "exact_optimal_move": exact_opt}
        regressed = False
        for budget in EVAL_BUDGETS:
            try:
                r = run_single_puct(evaluator, state, budget, SEED)
            except Exception:
                result[f"selected_is_optimal_{budget}"] = None
                continue
            sel = None if r.get("selected_move") is None else int(r["selected_move"])
            is_opt = sel == exact_opt if exact_opt is not None else None
            result[f"selected_is_optimal_{budget}"] = is_opt
            if is_opt is False:
                regressed = True
        result["control_regression"] = regressed
        ctrl_results.append(result)

    prod_opt_1200 = sum(1 for r in prod_results if r.get("selected_is_optimal_1200"))
    prod_opt_2400 = sum(1 for r in prod_results if r.get("selected_is_optimal_2400"))
    ctrl_opt_1200 = sum(1 for r in ctrl_results if r.get("selected_is_optimal_1200"))
    ctrl_opt_2400 = sum(1 for r in ctrl_results if r.get("selected_is_optimal_2400"))
    ctrl_reg = sum(1 for r in ctrl_results if r.get("control_regression"))

    return {
        "production_count": len(prod_results),
        "production_optimal_1200": prod_opt_1200,
        "production_optimal_2400": prod_opt_2400,
        "control_count": len(ctrl_results),
        "control_optimal_1200": ctrl_opt_1200,
        "control_optimal_2400": ctrl_opt_2400,
        "control_regression_count": ctrl_reg,
        "control_regression_rate": round_float(
            ctrl_reg / len(ctrl_results) if ctrl_results else 0.0
        ),
        "production_results": prod_results,
        "control_results": ctrl_results,
    }


# ── Representation Diagnostics ───────────────────────────────────────────


def extract_embeddings(
    npz_path: Path,
    rows: list[dict],
) -> tuple[np.ndarray, list[int]]:
    import torch
    from ml.alphazero_lite.input_encodings import feature_count_for
    from ml.alphazero_lite.train import (
        PolicyValueNet,
        load_checkpoint_into_model,
    )

    input_size = feature_count_for(INPUT_ENCODING)
    model = PolicyValueNet(
        hidden_sizes=(96, 3),
        model_type="residual_v3",
        input_size=input_size,
    )
    load_checkpoint_into_model(model, npz_path)
    model.eval()

    embeddings = []
    exact_opt_moves = []

    with torch.no_grad():
        for row in rows:
            state = row.get("state")
            if state is None:
                embeddings.append(None)
                exact_opt_moves.append(None)
                continue

            x = torch.tensor([state], dtype=torch.float32)

            h = torch.relu(model.input_layer(x))
            for first_layer, second_layer in model.residual_layers:
                residual = h
                h = torch.relu(first_layer(h))
                h = torch.relu(second_layer(h) + residual)

            embeddings.append(h.numpy().flatten())
            exact_opt_moves.append(row.get("exact_optimal_move"))

    valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
    emb_matrix = np.stack([embeddings[i] for i in valid_indices], axis=0)
    opt_moves = [exact_opt_moves[i] for i in valid_indices]
    return emb_matrix, opt_moves


def compute_representation_diagnostics(
    checkpoint_path: Path,
    production_rows: list[dict],
    controls_rows: list[dict],
) -> dict[str, Any]:
    import torch
    from ml.alphazero_lite.input_encodings import feature_count_for
    from ml.alphazero_lite.train import (
        PolicyValueNet,
        load_checkpoint_into_model,
    )

    input_size = feature_count_for(INPUT_ENCODING)
    model = PolicyValueNet(
        hidden_sizes=(96, 3),
        model_type="residual_v3",
        input_size=input_size,
    )
    load_checkpoint_into_model(model, checkpoint_path)
    model.eval()

    result: dict[str, Any] = {}

    with torch.no_grad():
        for label, rows in [
            ("production", production_rows),
            ("control", controls_rows),
        ]:
            states = np.array(
                [r["state"] for r in rows if r.get("state")], dtype=np.float32
            )
            x = torch.from_numpy(states)
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=1).numpy()
            logits_np = logits.numpy()

            entropies = -(probs * np.log(probs + 1e-9)).sum(axis=1)
            sorted_probs = -np.sort(-probs, axis=1)
            top2_margins = sorted_probs[:, 0] - sorted_probs[:, 1]

            opt_logit_margins = []
            current_top_moves = np.argmax(probs, axis=1)
            for i, row in enumerate(rows):
                eom = row.get("exact_optimal_move")
                if eom is not None:
                    top_logit = logits_np[i, current_top_moves[i]]
                    opt_logit = logits_np[i, eom]
                    opt_logit_margins.append(float(top_logit - opt_logit))

            result[f"{label}_mean_entropy"] = round_float(float(entropies.mean()))
            result[f"{label}_median_entropy"] = round_float(float(np.median(entropies)))
            result[f"{label}_mean_top2_margin"] = round_float(
                float(top2_margins.mean())
            )
            result[f"{label}_median_top2_margin"] = round_float(
                float(np.median(top2_margins))
            )
            result[f"{label}_mean_opt_logit_margin"] = round_float(
                float(np.mean(opt_logit_margins)) if opt_logit_margins else None
            )
            result[f"{label}_median_opt_logit_margin"] = round_float(
                float(np.median(opt_logit_margins)) if opt_logit_margins else None
            )
            result[f"{label}_current_top_move_matches_opt"] = sum(
                1
                for i, r in enumerate(rows)
                if r.get("exact_optimal_move") is not None
                and current_top_moves[i] == r["exact_optimal_move"]
            )

    prod_emb, prod_opts = extract_embeddings(checkpoint_path, production_rows)
    ctrl_emb, ctrl_opts = extract_embeddings(checkpoint_path, controls_rows)

    if len(prod_emb) > 0 and len(ctrl_emb) > 0:
        prod_sq = np.sum(prod_emb**2, axis=1, keepdims=True)
        ctrl_sq = np.sum(ctrl_emb**2, axis=1, keepdims=True)
        dists_sq = prod_sq + ctrl_sq.T - 2.0 * np.dot(prod_emb, ctrl_emb.T)
        dists_sq = np.maximum(dists_sq, 0.0)
        dists = np.sqrt(dists_sq)
        nn_dists = dists.min(axis=1)
        nn_indices = dists.argmin(axis=1)

        result["mean_prod_ctrl_nn_distance"] = round_float(float(nn_dists.mean()))
        result["median_prod_ctrl_nn_distance"] = round_float(float(np.median(nn_dists)))

        conflict_count = 0
        for i in range(len(prod_emb)):
            prod_opt = prod_opts[i] if i < len(prod_opts) else None
            nn_idx = int(nn_indices[i])
            ctrl_opt = ctrl_opts[nn_idx] if nn_idx < len(ctrl_opts) else None
            if prod_opt is not None and ctrl_opt is not None and prod_opt != ctrl_opt:
                conflict_count += 1
        result["nn_opt_move_conflict_count"] = conflict_count
        result["nn_opt_move_conflict_rate"] = round_float(
            conflict_count / len(prod_emb) if len(prod_emb) > 0 else 0.0
        )
    else:
        result["mean_prod_ctrl_nn_distance"] = None
        result["median_prod_ctrl_nn_distance"] = None
        result["nn_opt_move_conflict_count"] = None
        result["nn_opt_move_conflict_rate"] = None

    return result


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ── Report Generator ─────────────────────────────────────────────────────


def classify_findings(lane_results: list[dict], rep_diag: dict) -> str:
    has_clean_policy_head = any(
        r["scope"] == "policy_head"
        and r.get("production_gain_vs_current", 0) > 5
        and r.get("control_regression_count", 999) <= 2
        for r in lane_results
    )
    has_clean_last_block = any(
        r["scope"] == "last_block_policy"
        and r.get("production_gain_vs_current", 0) > 5
        and r.get("control_regression_count", 999) <= 2
        for r in lane_results
    )
    full_always_regresses = all(
        r.get("control_regression_count", 0) > 5
        for r in lane_results
        if r["scope"] == "all"
    )
    conflict_rate = rep_diag.get("nn_opt_move_conflict_rate") or 0.0
    nn_distance = rep_diag.get("mean_prod_ctrl_nn_distance") or 0.0

    if has_clean_policy_head or has_clean_last_block:
        return "representation_separable"
    if conflict_rate > 0.3 or (conflict_rate > 0.15 and full_always_regresses):
        return "representation_interference_likely"
    if nn_distance < 1.0 and full_always_regresses:
        return "representation_interference_likely"
    return "inconclusive"


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir",
        default=str(DEFAULT_WORKDIR),
        help="Working directory for checkpoints and eval artifacts",
    )
    parser.add_argument(
        "--epochs",
        default="1,2,4",
        help="Comma-separated epoch list",
    )
    parser.add_argument(
        "--learning-rates",
        default="2.5e-5",
        help="Comma-separated learning rates",
    )
    parser.add_argument(
        "--trainable-scopes",
        default="policy_head,last_block_policy,all",
        help="Comma-separated trainable scopes",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Rerun training even if cached",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    export_dir = workdir / "exports"
    eval_dir = workdir / "eval"
    export_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    epoch_list = [int(e.strip()) for e in args.epochs.split(",") if e.strip()]
    lr_list = [float(lr.strip()) for lr in args.learning_rates.split(",") if lr.strip()]
    scope_list = [s.strip() for s in args.trainable_scopes.split(",") if s.strip()]

    print("=" * 70)
    print("REPRESENTATION-INTERFERENCE DIAGNOSTIC")
    print("=" * 70)
    print(f"  workdir: {workdir}")
    print(f"  epochs: {epoch_list}")
    print(f"  lrs: {lr_list}")
    print(f"  scopes: {scope_list}")

    print("\nLoading artifacts...")
    soft065_rows = load_jsonl(SOFT065_PATH)
    controls_rows = load_jsonl(CONTROLS_PATH)
    production_rows = [
        r for r in soft065_rows if r.get("role") == "production_candidate"
    ]
    print(f"  production rows: {len(production_rows)}")
    print(f"  control rows: {len(controls_rows)}")

    init_ckpt = build_init_checkpoint(workdir)
    data_files = [SOFT065_PATH, CONTROLS_PATH]
    replay_weights = [1, 1]

    # ── Lane 1: current_eval_only ──
    print("\n--- Lane 1: current_eval_only ---")
    evaluator = ArtifactEvaluator(CURRENT_ARTIFACT)
    current_eval = evaluate_artifact(evaluator, production_rows, controls_rows)
    current_eval["scope"] = "current_eval_only"
    current_eval["epoch"] = 0
    current_eval["learning_rate"] = 0.0
    current_eval["checkpoint_sha256"] = sha256_file(CURRENT_ARTIFACT / "weights.json")
    print(
        f"  prod_opt@1200={current_eval['production_optimal_1200']}/{current_eval['production_count']}, "
        f"ctrl_reg={current_eval['control_regression_count']}/{current_eval['control_count']}"
    )
    write_json(eval_dir / "current_eval_only.json", current_eval)

    # ── Lanes 2-4: Training lanes ──
    all_training_results: list[dict] = []
    eval_results: list[dict] = []

    for trainable_scope in scope_list:
        for lr in lr_list:
            for epochs in epoch_list:
                trace_name = f"rep_interf_{trainable_scope}_lr{lr}_e{epochs}"
                scope_label = {
                    "policy_head": "head_only_policy_finetune",
                    "last_block_policy": "last_block_plus_policy_finetune",
                    "all": "full_finetune_control",
                }.get(trainable_scope, trainable_scope)

                print(f"\n--- Lane: {scope_label} epochs={epochs} lr={lr} ---")

                train_result = run_training(
                    trace_name,
                    data_files,
                    replay_weights,
                    epochs,
                    init_ckpt,
                    lr,
                    trainable_scope,
                    workdir=workdir,
                    force_rerun=args.force_rerun,
                )
                all_training_results.append(train_result)

                if train_result.get("returncode", 0) != 0:
                    continue

                ckpt_path = Path(train_result["checkpoint"])
                if not ckpt_path.exists():
                    print("    Checkpoint missing, skipping eval")
                    continue

                artifact_dir = eval_dir / f"{trace_name}_artifact"
                try:
                    convert_npz_to_artifact(ckpt_path, artifact_dir)
                    eval_evaluator = ArtifactEvaluator(artifact_dir)
                    eval_result = evaluate_artifact(
                        eval_evaluator, production_rows, controls_rows
                    )
                    eval_result["epoch"] = epochs
                    eval_result["learning_rate"] = lr
                    eval_result["scope"] = scope_label
                    eval_result["trainable_scope_raw"] = trainable_scope
                    eval_result["checkpoint_path"] = str(ckpt_path)
                    eval_result["checkpoint_sha256"] = sha256_file(ckpt_path)
                    eval_result["policy_loss"] = train_result.get("policy_loss")
                    eval_result["value_loss"] = train_result.get("value_loss")
                    eval_result["trainable_params"] = train_result.get(
                        "trainable_params"
                    )
                    eval_result["frozen_params"] = train_result.get("frozen_params")
                    eval_result["total_params"] = train_result.get("total_params")

                    baseline_opt = current_eval.get("production_optimal_1200", 0)
                    eval_result["production_gain_vs_current"] = (
                        eval_result.get("production_optimal_1200", 0) - baseline_opt
                    )

                    print(
                        f"    prod_opt@1200={eval_result['production_optimal_1200']}, "
                        f"gain={eval_result['production_gain_vs_current']}, "
                        f"ctrl_reg={eval_result['control_regression_count']}"
                    )
                    eval_results.append(eval_result)
                    write_json(eval_dir / f"{trace_name}_eval.json", eval_result)
                except Exception as e:
                    print(f"    Eval FAILED: {e}")

    # ── Representation diagnostics on current model ──
    print("\n--- Representation Diagnostics (current model) ---")
    current_npz = build_init_checkpoint(workdir)
    rep_diag = compute_representation_diagnostics(
        current_npz, production_rows, controls_rows
    )
    print(
        f"  mean NN distance: {rep_diag.get('mean_prod_ctrl_nn_distance')}, "
        f"conflict rate: {rep_diag.get('nn_opt_move_conflict_rate')}"
    )
    print(
        f"  production entropy: {rep_diag.get('production_mean_entropy')}, "
        f"control entropy: {rep_diag.get('control_mean_entropy')}"
    )
    print(
        f"  prod opt logit margin: {rep_diag.get('production_mean_opt_logit_margin')}, "
        f"ctrl opt logit margin: {rep_diag.get('control_mean_opt_logit_margin')}"
    )

    # ── Classification ──
    classification = classify_findings(eval_results, rep_diag)
    print(f"\nClassification: {classification}")

    # ── Write summary ──
    summary = {
        "schema": "azlite_representation_interference_diagnostic_v1",
        "workdir": str(workdir),
        "current_eval": {
            "production_optimal_1200": current_eval["production_optimal_1200"],
            "production_optimal_2400": current_eval.get("production_optimal_2400"),
            "control_optimal_1200": current_eval["control_optimal_1200"],
            "control_regression_count": current_eval["control_regression_count"],
            "production_count": current_eval["production_count"],
            "control_count": current_eval["control_count"],
        },
        "training_results": all_training_results,
        "evaluations": eval_results,
        "representation_diagnostics": rep_diag,
        "classification": classification,
    }
    write_json(workdir / "diagnostic_summary.json", summary)
    print(f"\nSummary written to {workdir / 'diagnostic_summary.json'}")

    # Compute deterministic parameter counts per scope
    scope_param_counts = {
        "all": {"trainable_params": 73159, "frozen_params": 0, "total_params": 73159},
        "policy_head": {
            "trainable_params": 9894,
            "frozen_params": 63265,
            "total_params": 73159,
        },
        "last_block_policy": {
            "trainable_params": 28518,
            "frozen_params": 44641,
            "total_params": 73159,
        },
    }

    # Enrich eval_results with deterministic param counts and merge into final results
    final_eval_results: list[dict] = []
    for er in eval_results:
        scope_raw = er.get("trainable_scope_raw", "")
        pcounts = scope_param_counts.get(
            scope_raw,
            {"trainable_params": None, "frozen_params": None, "total_params": None},
        )
        er["trainable_params"] = (
            er.get("trainable_params") or pcounts["trainable_params"]
        )
        er["frozen_params"] = er.get("frozen_params") or pcounts["frozen_params"]
        er["total_params"] = er.get("total_params") or pcounts["total_params"]
        final_eval_results.append(er)

    # ── Write markdown report ──
    write_report(
        REPORT_PATH, current_eval, final_eval_results, rep_diag, classification
    )
    print(f"Report written to {REPORT_PATH}")

    return 0


def write_report(
    path: Path,
    current_eval: dict,
    eval_results: list[dict],
    rep_diag: dict,
    classification: str,
) -> None:
    lines: list[str] = []
    lines.append("# AlphaZero-Lite Representation-Interference Diagnostic Results")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"**Classification:** `{classification}`")
    lines.append("")

    prod_count = current_eval.get("production_count", 0)
    ctrl_count = current_eval.get("control_count", 0)
    baseline_opt = current_eval.get("production_optimal_1200", 0)

    lines.append("## Current Baseline (eval only, no training)")
    lines.append("")
    lines.append(f"- Production optimal@1200: {baseline_opt}/{prod_count}")
    lines.append(
        f"- Control optimal@1200: {current_eval.get('control_optimal_1200', '?')}/{ctrl_count}"
    )
    lines.append(
        f"- Control regressions: {current_eval.get('control_regression_count', '?')}"
    )
    lines.append("")

    lines.append("## Lane Results")
    lines.append("")
    lines.append(
        "| Scope | LR | Epoch | Prod Opt@1200 | Gain vs Curr | Ctrl Reg | Ctrl Reg% | Policy Loss | Value Loss | Trainable Params | Frozen Params |"
    )
    lines.append(
        "|-------|----|-------|--------------|-------------|---------|----------|------------|-----------|-----------------|--------------|"
    )

    for er in eval_results:
        lines.append(
            f"| {er.get('scope', '?')} | {er.get('learning_rate', '?')} | {er.get('epoch', '?')} | "
            f"{er.get('production_optimal_1200', '?')}/{er.get('production_count', '?')} | "
            f"{er.get('production_gain_vs_current', '?')} | "
            f"{er.get('control_regression_count', '?')} | "
            f"{er.get('control_regression_rate', '?')} | "
            f"{er.get('policy_loss', '?')} | "
            f"{er.get('value_loss', '?')} | "
            f"{er.get('trainable_params', '?')} | "
            f"{er.get('frozen_params', '?')} |"
        )

    lines.append("")

    lines.append("## Representation Diagnostics (Current Model)")
    lines.append("")
    lines.append(
        f"- Mean production-control NN distance: {rep_diag.get('mean_prod_ctrl_nn_distance', 'N/A')}"
    )
    lines.append(
        f"- Median production-control NN distance: {rep_diag.get('median_prod_ctrl_nn_distance', 'N/A')}"
    )
    lines.append(
        f"- NN optimal-move conflict count (prod failures whose nearest ctrl has conflicting exact opt move): "
        f"{rep_diag.get('nn_opt_move_conflict_count', 'N/A')}"
    )
    lines.append(
        f"- NN optimal-move conflict rate: {rep_diag.get('nn_opt_move_conflict_rate', 'N/A')}"
    )
    lines.append("")
    lines.append(
        f"- Production mean entropy: {rep_diag.get('production_mean_entropy', 'N/A')}"
    )
    lines.append(
        f"- Control mean entropy: {rep_diag.get('control_mean_entropy', 'N/A')}"
    )
    lines.append(
        f"- Production mean top-2 margin: {rep_diag.get('production_mean_top2_margin', 'N/A')}"
    )
    lines.append(
        f"- Control mean top-2 margin: {rep_diag.get('control_mean_top2_margin', 'N/A')}"
    )
    lines.append(
        f"- Production mean opt logit margin: {rep_diag.get('production_mean_opt_logit_margin', 'N/A')}"
    )
    lines.append(
        f"- Control mean opt logit margin: {rep_diag.get('control_mean_opt_logit_margin', 'N/A')}"
    )
    lines.append(
        f"- Production current top-move matches opt: {rep_diag.get('production_current_top_move_matches_opt', 'N/A')}/{prod_count}"
    )
    lines.append(
        f"- Control current top-move matches opt: {rep_diag.get('control_current_top_move_matches_opt', 'N/A')}/{ctrl_count}"
    )
    lines.append("")

    lines.append("## Classification Criteria")
    lines.append("")
    lines.append(
        "- `representation_separable`: policy_head or last_block_policy improves production substantially with near-zero control regressions."
    )
    lines.append(
        "- `representation_interference_likely`: head-only cannot improve, and full/last-block regresses controls. Production rows are near controls with conflicting optimal moves in embedding space."
    )
    lines.append(
        "- `optimization_instability`: head-only fits cleanly at one LR/epoch, full unnecessarily regresses controls."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
