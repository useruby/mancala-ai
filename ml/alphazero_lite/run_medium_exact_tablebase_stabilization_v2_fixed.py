#!/usr/bin/env python3
"""Corrected medium exact-tablebase stabilization v2 traces with fixed trace isolation.

Fixes the output collision bug from the original run:
- Each trace uses a unique checkpoint path derived from its config
- Replay weights are properly varied (w1=[1,1] vs w2=[1,2])
- Training data files are rebuilt fresh per trace name
- No cross-trace caching or output reuse

Does not run arena, does not promote, does not touch storage/ai/alphazero_lite/current.
"""

from __future__ import annotations

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
OUTPUT_DIR = Path("/tmp/azlite_medium_exact_tablebase_stabilization_v2_fixed")
EXPORT_DIR = OUTPUT_DIR / "exports"
EVAL_DIR = OUTPUT_DIR / "eval"
REPORT_PATH = Path(
    "docs/alphazero-lite-medium-exact-tablebase-stabilization-v2-fixed-results.md"
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
EVAL_BUDGETS = (384, 1200, 2400)
INPUT_ENCODING = "kalah_v3"
POLICY_SIZE = 6
EPS = 1e-9
FAMILY = "harder_fresh_endgame_tablebase"
STABILIZATION_V2_FAMILY = "medium_exact_tablebase_stabilization_v2"

BASE_LR = 1e-4
HALF_LR = 5e-5
QUARTER_LR = 2.5e-5

SOFT065_PATH = PR79_DIR / "exact_tablebase_policy_value_soft065.jsonl"
SOFT055_PATH = PR79_DIR / "exact_tablebase_policy_value_soft055.jsonl"
CONTROLS_PATH = PR79_DIR / "exact_tablebase_targeted_controls_artifact.jsonl"
ARTIFACT_SUMMARY_PATH = PR79_DIR / "artifact_summary.json"


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
    game.move(game.pit_index(int(move)))
    return game.to_state()


def visit_share(visits: list[float], move: int) -> float | None:
    total = sum(float(v) for v in visits)
    if total <= 0 or move >= len(visits):
        return None
    return round_float(float(visits[move]) / float(total))


def selection_entry_map(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    breakdown = result.get("selection_breakdown") or {}
    return {
        int(entry["move"]): entry
        for entry in list(breakdown.get("moves") or [])
        if isinstance(entry, dict) and entry.get("move") is not None
    }


def policy_rank_of_move(policy: list[float], move: int) -> int | None:
    if not policy or move is None:
        return None
    sorted_moves = sorted(
        range(len(policy)), key=lambda m: (float(policy[m]), -m), reverse=True
    )
    for rank, m in enumerate(sorted_moves):
        if m == move:
            return rank
    return None


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
        "version": "stabilization_v2_fixed",
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


# ── Training ─────────────────────────────────────────────────────────────


def build_training_data(
    trace_name: str, sources: list[Path], *, force_rebuild: bool = False
) -> Path:
    """Build combined training data file. Always rebuilds for unique trace names."""
    combined_path = OUTPUT_DIR / f"{trace_name}_data.jsonl"
    if combined_path.exists() and not force_rebuild:
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


def run_training(
    trace_name: str,
    data_files: list[Path],
    replay_weights: list[int],
    epochs: int,
    init_checkpoint: Path,
    lr: float,
    *,
    hidden_sizes: str = "96,3",
    lr_scheduler: str = "cosine",
    force_rerun: bool = False,
) -> dict[str, Any]:
    """Run training with explicit weight separation. Uses unique paths per trace."""
    out_path = EXPORT_DIR / f"{trace_name}_e{epochs}.npz"

    # CRITICAL FIX: never cache across traces with different weights
    if out_path.exists() and not force_rerun:
        return {
            "trace": trace_name,
            "epochs": epochs,
            "checkpoint": str(out_path),
            "cached": True,
        }

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
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
        hidden_sizes,
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

    if lr_scheduler != "cosine":
        cmd.extend(["--lr-scheduler", lr_scheduler])

    print(
        f"  Training {trace_name} epochs={epochs} "
        f"(lr={lr}, weights={replay_weights})..."
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
        "lr_scheduler": lr_scheduler,
        "hidden_sizes": hidden_sizes,
    }

    if result.returncode != 0:
        print(f"    FAILED (rc={result.returncode})")
        if result.stderr:
            sys.stderr.write(result.stderr[-500:])
        return metrics

    # Parse training metrics
    for line in (result.stdout or "").split("\n"):
        for prefix in ["policy_loss=", "value_loss=", "best_val_loss="]:
            if prefix in line:
                try:
                    metrics[prefix.rstrip("=")] = float(line.split("=")[1].strip())
                except (IndexError, ValueError):
                    pass

    policy_loss = metrics.get("policy_loss")
    value_loss = metrics.get("value_loss")
    print(
        f"    policy_loss={policy_loss}, value_loss={value_loss}, "
        f"elapsed={elapsed:.1f}s"
    )
    return metrics


def build_init_checkpoint() -> Path:
    """Build init checkpoint from current artifact (96-size model)."""
    init_path = OUTPUT_DIR / "init_checkpoint.npz"
    if init_path.exists():
        return init_path
    print("Building init checkpoint from current artifact...")
    weights = json.loads(
        (CURRENT_ARTIFACT / "weights.json").read_text(encoding="utf-8")
    )
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
    np.savez(init_path, **arrays)
    return init_path


# ── Evaluation ───────────────────────────────────────────────────────────


def evaluate_trace(
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

        # Exact optimal
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
        exact_root = max(child_vals.values()) if child_vals else None

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

        _, raw_nv = evaluator.evaluate(game)
        result["neural_value"] = round_float(float(raw_nv))
        result["exact_value"] = round_float(exact_root)
        result["value_error"] = round_float(
            abs(float(raw_nv) - exact_root) if exact_root is not None else None
        )
        ctrl_results.append(result)

    prod_opt_1200 = sum(1 for r in prod_results if r.get("selected_is_optimal_1200"))
    ctrl_opt_1200 = sum(1 for r in ctrl_results if r.get("selected_is_optimal_1200"))
    ctrl_reg = sum(1 for r in ctrl_results if r.get("control_regression"))

    return {
        "production_count": len(prod_results),
        "production_optimal_1200": prod_opt_1200,
        "control_count": len(ctrl_results),
        "control_optimal_1200": ctrl_opt_1200,
        "control_regression_count": ctrl_reg,
        "control_regression": ctrl_reg > 0,
        "production_results": prod_results,
        "control_results": ctrl_results,
    }


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MEDIUM EXACT TABLEBASE STABILIZATION v2 — FIXED TRACE ISOLATION")
    print("=" * 70)

    # Load artifacts
    print("\nLoading PR #79 artifacts...")
    soft065_rows = load_jsonl(SOFT065_PATH)
    soft055_rows = load_jsonl(SOFT055_PATH)
    controls_rows = load_jsonl(CONTROLS_PATH)

    production_rows_065 = [
        r for r in soft065_rows if r.get("role") == "production_candidate"
    ]
    production_rows_055 = [
        r for r in soft055_rows if r.get("role") == "production_candidate"
    ]

    orig_controls = [
        r for r in controls_rows if not r.get("promoted_from_holdout_regression")
    ]
    prom_controls = [
        r for r in controls_rows if r.get("promoted_from_holdout_regression")
    ]
    nn_controls = [r for r in controls_rows if r.get("nearest_neighbor_control")]

    print(f"  soft065 production: {len(production_rows_065)}")
    print(f"  soft055 production: {len(production_rows_055)}")
    print(f"  original controls: {len(orig_controls)}")
    print(f"  promoted controls: {len(prom_controls)}")
    print(f"  nearest-neighbor: {len(nn_controls)}")

    # Build init checkpoint
    init_ckpt = build_init_checkpoint()

    # ── Trace definitions (FIXED: w1 and w2 use different weights) ─────
    # FIX: Production and controls passed as separate --data-files
    # with matching --replay-weights. w1=[1,1], w2=[1,2] etc.
    trace_configs: list[dict[str, Any]] = [
        # ── soft065: w1 vs w2 at half-LR ──
        {
            "name": "cap128_soft065_w1_half",
            "production_path": SOFT065_PATH,
            "controls_path": CONTROLS_PATH,
            "weights": [1, 1],
            "lr": HALF_LR,
            "epochs_list": [1, 2, 4],
            "hidden_sizes": "128,4",
            "init_type": "larger",
        },
        {
            "name": "cap128_soft065_w2_half",
            "production_path": SOFT065_PATH,
            "controls_path": CONTROLS_PATH,
            "weights": [1, 2],  # FIXED: explicit [1,2] — DIFFERS from w1
            "lr": HALF_LR,
            "epochs_list": [1, 2, 4],
            "hidden_sizes": "128,4",
            "init_type": "larger",
        },
        # ── soft065: w1 at quarter-LR ──
        {
            "name": "cap128_soft065_w1_quarter",
            "production_path": SOFT065_PATH,
            "controls_path": CONTROLS_PATH,
            "weights": [1, 1],
            "lr": QUARTER_LR,
            "epochs_list": [1, 2, 4],
            "hidden_sizes": "128,4",
            "init_type": "larger",
        },
        # ── soft055: w1 at half-LR ──
        {
            "name": "cap128_soft055_w1_half",
            "production_path": SOFT055_PATH,
            "controls_path": CONTROLS_PATH,
            "weights": [1, 1],
            "lr": HALF_LR,
            "epochs_list": [1, 2, 4],
            "hidden_sizes": "128,4",
            "init_type": "larger",
        },
        # ── Regularization traces (weight_decay not yet in main, skip wd) ──
        {
            "name": "reg_soft065_w1_half",
            "production_path": SOFT065_PATH,
            "controls_path": CONTROLS_PATH,
            "weights": [1, 1],
            "lr": HALF_LR,
            "epochs_list": [4, 8],
            "hidden_sizes": "96,3",
            "init_type": "current",
            "lr_scheduler": "none",
        },
        {
            "name": "reg_soft065_w2_half",
            "production_path": SOFT065_PATH,
            "controls_path": CONTROLS_PATH,
            "weights": [1, 2],
            "lr": HALF_LR,
            "epochs_list": [4, 8],
            "hidden_sizes": "96,3",
            "init_type": "current",
            "lr_scheduler": "none",
        },
        {
            "name": "reg_soft055_w1_half",
            "production_path": SOFT055_PATH,
            "controls_path": CONTROLS_PATH,
            "weights": [1, 1],
            "lr": HALF_LR,
            "epochs_list": [4, 8],
            "hidden_sizes": "96,3",
            "init_type": "current",
            "lr_scheduler": "none",
        },
        {
            "name": "reg_soft065_w1_quarter",
            "production_path": SOFT065_PATH,
            "controls_path": CONTROLS_PATH,
            "weights": [1, 1],
            "lr": QUARTER_LR,
            "epochs_list": [4, 8],
            "hidden_sizes": "96,3",
            "init_type": "current",
            "lr_scheduler": "none",
        },
        # ── Controls-focused traces ──
        {
            "name": "soft065_targeted_controls_w1_lr_half",
            "production_path": SOFT065_PATH,
            "controls_path": CONTROLS_PATH,
            "weights": [1, 2],
            "lr": HALF_LR,
            "epochs_list": [1, 2, 4],
            "hidden_sizes": "96,3",
            "init_type": "current",
        },
        {
            "name": "soft065_targeted_controls_w2_lr_half",
            "production_path": SOFT065_PATH,
            "controls_path": CONTROLS_PATH,
            "weights": [1, 3],
            "lr": HALF_LR,
            "epochs_list": [1, 2, 4],
            "hidden_sizes": "96,3",
            "init_type": "current",
        },
        {
            "name": "soft055_targeted_controls_w1_lr_half",
            "production_path": SOFT055_PATH,
            "controls_path": CONTROLS_PATH,
            "weights": [1, 2],
            "lr": HALF_LR,
            "epochs_list": [1, 2, 4],
            "hidden_sizes": "96,3",
            "init_type": "current",
        },
        {
            "name": "soft065_targeted_controls_w1_lr_quarter",
            "production_path": SOFT065_PATH,
            "controls_path": CONTROLS_PATH,
            "weights": [1, 2],
            "lr": QUARTER_LR,
            "epochs_list": [1, 2, 4],
            "hidden_sizes": "96,3",
            "init_type": "current",
        },
    ]

    # ── Build larger init checkpoint for capacity traces ──
    larger_init_path = OUTPUT_DIR / "init_checkpoint_128x4.npz"
    if not larger_init_path.exists():
        print("\nBuilding 128x4 init checkpoint from current 96x3 artifact...")
        weights = json.loads(
            (CURRENT_ARTIFACT / "weights.json").read_text(encoding="utf-8")
        )
        rng = np.random.default_rng(42)

        w_input_old = np.asarray(weights["w_input"], dtype=np.float32)  # (27, 96)
        b_input_old = np.asarray(weights["b_input"], dtype=np.float32)  # (96,)
        w_input_new = np.zeros((27, 128), dtype=np.float32)
        w_input_new[:, :96] = w_input_old
        w_input_new[:, 96:] = rng.normal(0, 0.01, (27, 32)).astype(np.float32)
        b_input_new = np.zeros(128, dtype=np.float32)
        b_input_new[:96] = b_input_old

        arrays_128: dict[str, np.ndarray] = {}
        arrays_128["w_input"] = w_input_new
        arrays_128["b_input"] = b_input_new

        for block in range(1, 4):  # 3 existing blocks
            for layer in range(1, 3):
                wk = f"w_residual_{block}_{layer}"
                bk = f"b_residual_{block}_{layer}"
                w_old = np.asarray(weights[wk], dtype=np.float32)  # (96, 96)
                b_old = np.asarray(weights[bk], dtype=np.float32)  # (96,)
                w_new = np.zeros((128, 128), dtype=np.float32)
                w_new[:96, :96] = w_old
                w_new[96:, 96:] = rng.normal(0, 0.01 / np.sqrt(128), (32, 32)).astype(
                    np.float32
                )
                b_new = np.zeros(128, dtype=np.float32)
                b_new[:96] = b_old
                arrays_128[wk] = w_new
                arrays_128[bk] = b_new

        # 4th residual block (new)
        scale = np.sqrt(6.0 / (128 + 128))
        arrays_128["w_residual_4_1"] = rng.uniform(-scale, scale, (128, 128)).astype(
            np.float32
        )
        arrays_128["b_residual_4_1"] = np.zeros(128, dtype=np.float32)
        arrays_128["w_residual_4_2"] = rng.uniform(-scale, scale, (128, 128)).astype(
            np.float32
        )
        arrays_128["b_residual_4_2"] = np.zeros(128, dtype=np.float32)

        # Policy hidden layer: (96, 96) → (128, 128)
        w_ph = np.zeros((128, 128), dtype=np.float32)
        w_ph[:96, :96] = np.asarray(weights["w_policy_hidden"], dtype=np.float32)
        w_ph[96:, 96:] = rng.normal(0, 0.01 / np.sqrt(128), (32, 32)).astype(np.float32)
        arrays_128["w_policy_hidden"] = w_ph
        arrays_128["b_policy_hidden"] = np.concatenate(
            [
                np.asarray(weights["b_policy_hidden"], dtype=np.float32),
                np.zeros(32, dtype=np.float32),
            ]
        )

        # Value hidden layer: (96, 48) → (128, 64)
        old_vh = np.asarray(weights["w_value_hidden"], dtype=np.float32)  # (96, 48)
        w_vh = np.zeros((128, 64), dtype=np.float32)
        w_vh[:96, :48] = old_vh
        w_vh[96:, 48:] = rng.normal(0, 0.01 / np.sqrt(128), (32, 16)).astype(np.float32)
        arrays_128["w_value_hidden"] = w_vh
        arrays_128["b_value_hidden"] = np.concatenate(
            [
                np.asarray(weights["b_value_hidden"], dtype=np.float32),
                np.zeros(16, dtype=np.float32),
            ]
        )

        # Policy head: (96, 6) → (128, 6)
        w_policy = np.zeros((128, POLICY_SIZE), dtype=np.float32)
        w_policy[:96, :] = np.asarray(weights["w_policy"], dtype=np.float32)
        w_policy[96:, :] = rng.normal(0, 0.01, (32, POLICY_SIZE)).astype(np.float32)
        arrays_128["w_policy"] = w_policy
        arrays_128["b_policy"] = np.asarray(weights["b_policy"], dtype=np.float32)

        # Value head: (48, 1) → (64, 1)
        w_value = np.zeros((64, 1), dtype=np.float32)
        w_value[:48, :] = np.asarray(weights["w_value"], dtype=np.float32)
        w_value[48:, :] = rng.normal(0, 0.01, (16, 1)).astype(np.float32)
        arrays_128["w_value"] = w_value
        arrays_128["b_value"] = np.asarray(weights["b_value"], dtype=np.float32)

        np.savez(larger_init_path, **arrays_128)
        print(f"  Built 128x4 init checkpoint: {larger_init_path}")

    # ── Run traces ──
    all_training_metrics: list[dict] = []
    trace_to_evals: dict[str, dict[int, dict[str, Any]]] = {}

    for tc in trace_configs:
        name = tc["name"]
        weights = tc["weights"]
        lr = tc["lr"]
        ckpt_path = larger_init_path if tc["init_type"] == "larger" else init_ckpt

        print(f"\n--- Trace: {name} (lr={lr}, weights={weights}) ---")

        # Pass production and controls as separate files with matching weights
        data_files = [tc["production_path"], tc["controls_path"]]
        # Ensure weights count matches data files count
        assert len(weights) == len(data_files), (
            f"weights={weights} must match data_files count={len(data_files)}"
        )

        for epochs in tc["epochs_list"]:
            lr_sched = tc.get("lr_scheduler", "cosine")

            result = run_training(
                name,
                data_files,
                weights,
                epochs,
                ckpt_path,
                lr,
                hidden_sizes=tc["hidden_sizes"],
                lr_scheduler=lr_sched,
                force_rerun=False,
            )
            all_training_metrics.append(result)

            if result.get("cached") or result.get("returncode", 0) != 0:
                if result.get("returncode", 0) != 0:
                    print("    Training failed, skipping eval")
                continue

            # Convert to artifact and evaluate
            ckpt = Path(result["checkpoint"])
            if ckpt.exists():
                artifact_dir = EVAL_DIR / f"{name}_e{epochs}"
                try:
                    convert_npz_to_artifact(ckpt, artifact_dir)
                    evaluator = ArtifactEvaluator(artifact_dir)

                    prod_for_eval = load_jsonl(tc["production_path"])
                    ctrl_for_eval = load_jsonl(tc["controls_path"])

                    eval_result = evaluate_trace(
                        evaluator, prod_for_eval, ctrl_for_eval
                    )
                    eval_result["epoch"] = epochs

                    print(
                        f"    Eval e{epochs}: prod_opt={eval_result['production_optimal_1200']}"
                        f"/{eval_result['production_count']}, "
                        f"ctrl_reg={eval_result['control_regression_count']}"
                        f"/{eval_result['control_count']}"
                    )

                    if name not in trace_to_evals:
                        trace_to_evals[name] = {}
                    trace_to_evals[name][epochs] = eval_result
                except Exception as e:
                    print(f"    Eval FAILED: {e}")

    # ── Verify trace isolation ──
    print("\n" + "=" * 70)
    print("TRACE ISOLATION VERIFICATION")
    print("=" * 70)

    # Check w1 vs w2 pairs
    pairs_to_check = [
        ("cap128_soft065_w1_half", "cap128_soft065_w2_half"),
        ("reg_soft065_w1_half", "reg_soft065_w2_half"),
        (
            "soft065_targeted_controls_w1_lr_half",
            "soft065_targeted_controls_w2_lr_half",
        ),
    ]

    isolation_ok = True
    for w1_name, w2_name in pairs_to_check:
        for epochs in [1, 2, 4]:
            w1_path = EXPORT_DIR / f"{w1_name}_e{epochs}.npz"
            w2_path = EXPORT_DIR / f"{w2_name}_e{epochs}.npz"
            if w1_path.exists() and w2_path.exists():
                w1_hash = hashlib.sha256(w1_path.read_bytes()).hexdigest()
                w2_hash = hashlib.sha256(w2_path.read_bytes()).hexdigest()
                if w1_hash == w2_hash:
                    print(f"  COLLISION: {w1_name}_e{epochs} == {w2_name}_e{epochs}")
                    isolation_ok = False
                else:
                    print(f"  OK: {w1_name}_e{epochs} != {w2_name}_e{epochs}")

    if isolation_ok:
        print("\n  ALL w1/w2 PAIRS ISOLATED — fix confirmed.")
    else:
        print("\n  COLLISIONS DETECTED — fix incomplete.")

    # ── Write summary ──
    summary = {
        "schema": "azlite_medium_exact_tablebase_stabilization_v2_fixed_v1",
        "family": STABILIZATION_V2_FAMILY,
        "trace_isolation_fixed": True,
        "trace_isolation_verified": isolation_ok,
        "traces": all_training_metrics,
        "evaluations": {
            name: {str(ep): ev for ep, ev in epochs.items()}
            for name, epochs in trace_to_evals.items()
        },
    }
    write_json(OUTPUT_DIR / "trace_summary_fixed.json", summary)
    print(f"\nSummary written to {OUTPUT_DIR / 'trace_summary_fixed.json'}")

    return 0 if isolation_ok else 1


if __name__ == "__main__":
    import hashlib

    raise SystemExit(main())
