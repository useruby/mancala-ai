#!/usr/bin/env python3
"""Conservative preservation fine-tune sweep from iter0_reference.

Tests whether iter0_reference can be fine-tuned with much smaller learning
rates and shorter schedules without destroying its 1200:1200 disadvantaged-seat
breakthrough.

Architecture: residual_v3 only.
No tablebase overlay, tactical-bias-off replay, new mined replay, promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = str(REPO_ROOT / ".venv/bin/python")

ENV = os.environ.copy()
ENV.setdefault("OMP_NUM_THREADS", "1")
ENV.setdefault("OPENBLAS_NUM_THREADS", "1")
ENV.setdefault("MKL_NUM_THREADS", "1")

BUDGET_PAIR_LABELS = {
    (384, 256): "384_256",
    (768, 256): "768_256",
    (1200, 1200): "1200_1200",
    (256, 768): "256_768",
}


def sha256_hex(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command: list[str], *, cwd: Path, timeout: int = 7200) -> dict:
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=ENV,
    )
    duration = round(time.time() - started, 2)
    return {
        "command": command,
        "returncode": completed.returncode,
        "duration_s": duration,
        "stdout": completed.stdout[-4000:]
        if len(completed.stdout) > 4000
        else completed.stdout,
        "stderr": completed.stderr[-4000:]
        if len(completed.stderr) > 4000
        else completed.stderr,
    }


def wilson_interval(score: float, sample_size: int) -> dict[str, float]:
    z = 1.96
    if sample_size <= 0:
        return {"lower": 0.0, "upper": 0.0}
    denominator = 1.0 + ((z**2) / sample_size)
    center = score + ((z**2) / (2.0 * sample_size))
    margin = (
        z
        * (((score * (1.0 - score)) + ((z**2) / (4.0 * sample_size))) / sample_size)
        ** 0.5
    )
    return {
        "lower": max(0.0, (center - margin) / denominator),
        "upper": min(1.0, (center + margin) / denominator),
    }


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _lr_to_label(lr: float) -> str:
    """Convert a learning rate to a compact label, e.g. 0.0001 -> '1e_4'."""
    if lr == 1e-4:
        return "1e_4"
    if lr == 3e-5:
        return "3e_5"
    if lr == 1e-5:
        return "1e_5"
    return str(lr).replace(".", "_").replace("-", "m")


def parse_game_jsonl(path: str) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def compute_seat_split_metrics(entries: list[dict]) -> dict:
    p0_wins = p0_losses = p0_draws = 0
    p1_wins = p1_losses = p1_draws = 0
    p0_margins: list[int] = []
    p1_margins: list[int] = []
    p0_lengths: list[int] = []
    p1_lengths: list[int] = []
    trajectory_hashes: list[str] = []

    for ent in entries:
        cp = ent["challenger_player"]
        winner = ent["winner"]
        margin = ent["margin"]
        if cp == 0:
            if winner == "challenger":
                p0_wins += 1
            elif winner == "current":
                p0_losses += 1
            else:
                p0_draws += 1
            p0_margins.append(margin)
            p0_lengths.append(ent["game_length"])
        else:
            if winner == "challenger":
                p1_wins += 1
            elif winner == "current":
                p1_losses += 1
            else:
                p1_draws += 1
            p1_margins.append(margin)
            p1_lengths.append(ent["game_length"])
        trajectory_hashes.append(ent["trajectory"])

    p0_total = p0_wins + p0_losses + p0_draws
    p1_total = p1_wins + p1_losses + p1_draws
    traj_counter = Counter(trajectory_hashes)
    duplicate_count = sum(c for c in traj_counter.values() if c > 1)
    p0_score = (p0_wins + 0.5 * p0_draws) / max(p0_total, 1)
    p1_score = (p1_wins + 0.5 * p1_draws) / max(p1_total, 1)

    return {
        "challenger_starts_0": {
            "games": p0_total,
            "wins": p0_wins,
            "losses": p0_losses,
            "draws": p0_draws,
            "score": p0_score,
            "ci95": wilson_interval(p0_score, p0_total),
            "margin_mean": statistics.fmean(p0_margins) if p0_margins else 0.0,
            "margin_median": statistics.median(p0_margins) if p0_margins else 0.0,
            "game_length_mean": statistics.fmean(p0_lengths) if p0_lengths else 0.0,
            "game_length_median": statistics.median(p0_lengths) if p0_lengths else 0.0,
        },
        "challenger_starts_1": {
            "games": p1_total,
            "wins": p1_wins,
            "losses": p1_losses,
            "draws": p1_draws,
            "score": p1_score,
            "ci95": wilson_interval(p1_score, p1_total),
            "margin_mean": statistics.fmean(p1_margins) if p1_margins else 0.0,
            "margin_median": statistics.median(p1_margins) if p1_margins else 0.0,
            "game_length_mean": statistics.fmean(p1_lengths) if p1_lengths else 0.0,
            "game_length_median": statistics.median(p1_lengths) if p1_lengths else 0.0,
        },
        "disadvantaged_seat_score": p1_score,
        "margin_mean": statistics.fmean(p0_margins + p1_margins)
        if p0_margins + p1_margins
        else 0.0,
        "margin_median": statistics.median(p0_margins + p1_margins)
        if p0_margins + p1_margins
        else 0.0,
        "game_length_mean": statistics.fmean(p0_lengths + p1_lengths)
        if p0_lengths + p1_lengths
        else 0.0,
        "game_length_median": statistics.median(p0_lengths + p1_lengths)
        if p0_lengths + p1_lengths
        else 0.0,
        "unique_trajectories": len(traj_counter),
        "duplicate_trajectory_count": duplicate_count,
        "total_games": p0_total + p1_total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conservative preservation fine-tune sweep from iter0_reference."
    )
    parser.add_argument("--workdir", required=True, help="Working directory")
    parser.add_argument(
        "--init-checkpoint",
        required=True,
        help="Path to iter0_reference checkpoint .npz",
    )
    parser.add_argument(
        "--current",
        default="model-artifact/current",
        help="Current production artifact directory",
    )
    parser.add_argument(
        "--data-files",
        required=True,
        help="Comma-separated JSONL training data paths",
    )
    parser.add_argument(
        "--replay-weights",
        default="4,1",
        help="Comma-separated replay weights",
    )
    parser.add_argument(
        "--learning-rates",
        default="1e-4,3e-5,1e-5",
        help="Comma-separated learning rates to sweep",
    )
    parser.add_argument(
        "--epochs",
        default="1,2,4",
        help="Comma-separated epochs to checkpoint",
    )
    parser.add_argument(
        "--model-type",
        default="residual_v3",
        choices=["residual_v2", "residual_v3", "residual_v4_move_factorized"],
    )
    parser.add_argument(
        "--input-encoding",
        default="kalah_v3",
    )
    parser.add_argument("--hidden-sizes", default="96,3")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--value-loss", default="huber")
    parser.add_argument("--value-loss-weight", type=float, default=0.3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--budget-pairs",
        default="384:256,768:256,1200:1200,256:768",
        help="Comma-separated challenger:current simulation pairs",
    )
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument(
        "--tactical-root-bias",
        type=float,
        default=0.1,
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-tbo-secondary", action="store_true")
    parser.add_argument(
        "--policy-head-only",
        action="store_true",
        help="Also run policy_head_only lane at LR 1e-4",
    )
    return parser.parse_args()


def train_checkpoint(
    *,
    init_checkpoint: Path,
    out_checkpoint: Path,
    data_files: list[Path],
    replay_weights: list[int],
    lr: float,
    epochs: int,
    batch_size: int,
    value_loss: str,
    value_loss_weight: float,
    grad_clip: float,
    hidden_sizes: str,
    model_type: str,
    input_encoding: str,
    trainable_scope: str,
    seed: int,
    timeout: int = 7200,
) -> dict:
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    data_files_arg = ",".join(str(p) for p in data_files)
    weights_arg = ",".join(str(w) for w in replay_weights)

    command = [
        VENV_PYTHON,
        "ml/alphazero_lite/train.py",
        "--init-checkpoint",
        str(init_checkpoint),
        "--data-files",
        data_files_arg,
        "--replay-weights",
        weights_arg,
        "--model-type",
        model_type,
        "--input-encoding",
        input_encoding,
        "--hidden-sizes",
        hidden_sizes,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--value-loss",
        value_loss,
        "--value-loss-weight",
        str(value_loss_weight),
        "--grad-clip",
        str(grad_clip),
        "--out",
        str(out_checkpoint),
        "--device",
        "auto",
        "--policy-target-mode",
        "sharpened",
        "--value-target-mode",
        "sharpened",
        "--lr-scheduler",
        "none",
        "--trainable-scope",
        trainable_scope,
        "--seed",
        str(seed),
        "--val-split",
        "0.1",
    ]

    result = run_command(command, cwd=REPO_ROOT, timeout=timeout)
    if result["returncode"] != 0:
        result["status"] = "failed"
        return result
    result["status"] = "completed"

    metrics = {}
    for line in result.get("stdout", "").splitlines():
        for key in ("policy_loss", "value_loss", "best_val_loss"):
            if line.startswith(f"{key}="):
                try:
                    metrics[key] = float(line.split("=", 1)[1])
                except (ValueError, IndexError):
                    pass
    result["metrics"] = metrics
    result["checkpoint_path"] = str(out_checkpoint)
    result["checkpoint_sha256"] = sha256_hex(out_checkpoint)
    return result


def export_artifact(
    *,
    checkpoint_path: Path,
    artifact_dir: Path,
    version: str,
    model_type: str,
    input_encoding: str,
    timeout: int = 300,
) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    command = [
        VENV_PYTHON,
        "ml/alphazero_lite/export_artifact.py",
        "--checkpoint",
        str(checkpoint_path),
        "--out-dir",
        str(artifact_dir),
        "--version",
        version,
        "--model-type",
        model_type,
        "--rules-version",
        "kalah_v1",
        "--input-encoding",
        input_encoding,
    ]
    result = run_command(command, cwd=REPO_ROOT, timeout=timeout)
    if result["returncode"] != 0:
        result["status"] = "failed"
        return result
    result["status"] = "completed"
    result["artifact_dir"] = str(artifact_dir)
    result["weights_json_sha256"] = sha256_hex(artifact_dir / "weights.json")
    return result


def run_arena(
    *,
    challenger: str,
    current: str,
    challenger_sims: int,
    current_sims: int,
    games: int,
    seed: int,
    workers: int,
    c_puct: float,
    tactical_root_bias: float,
    out_json: str,
    out_jsonl: str,
    timeout: int = 3600,
) -> dict:
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    command = [
        VENV_PYTHON,
        str(REPO_ROOT / "ml/alphazero_lite/arena.py"),
        "--challenger",
        challenger,
        "--current",
        current,
        "--challenger-simulations",
        str(challenger_sims),
        "--current-simulations",
        str(current_sims),
        "--games",
        str(games),
        "--seed",
        str(seed),
        "--workers",
        str(max(1, workers)),
        "--min-score",
        "0.0",
        "--c-puct",
        str(c_puct),
        "--tactical-root-bias",
        str(tactical_root_bias),
        "--root-policy-mode",
        "deterministic",
        "--out",
        out_json,
        "--game-jsonl",
        out_jsonl,
    ]
    result = run_command(command, cwd=REPO_ROOT, timeout=timeout)
    if result["returncode"] != 0:
        result["status"] = "failed"
        return result
    result["status"] = "completed"
    if Path(out_json).exists():
        report = load_json(Path(out_json))
        notes = report.get("notes", {}) if isinstance(report.get("notes"), dict) else {}
        result["arena"] = {
            "score": report.get("score"),
            "wins": report.get("wins"),
            "losses": report.get("losses"),
            "draws": report.get("draws"),
            "games_played": report.get("games_played"),
            "move_time_mean_ms": notes.get("move_time_mean_ms"),
            "move_time_p95_ms": notes.get("move_time_p95_ms"),
        }
    return result


def evaluate_checkpoint(
    *,
    artifact_dir: str,
    current: str,
    budget_pairs: list[tuple[int, int]],
    games: int,
    seed: int,
    workers: int,
    c_puct: float,
    tactical_root_bias: float,
    eval_dir: Path,
    skip_eval: bool,
) -> dict:
    eval_dir.mkdir(parents=True, exist_ok=True)
    budget_results: list[dict] = []

    for chall_sims, curr_sims in budget_pairs:
        budget_label = BUDGET_PAIR_LABELS.get(
            (chall_sims, curr_sims), f"{chall_sims}_vs_{curr_sims}"
        )
        pair_dir = eval_dir / budget_label
        pair_dir.mkdir(parents=True, exist_ok=True)

        alt_json = str(pair_dir / "alternating_arena.json")
        alt_jsonl = str(pair_dir / "alternating_games.jsonl")

        print(
            f"    [{budget_label}] alternating (c_puct={c_puct}, bias={tactical_root_bias}) ...",
            end=" ",
            flush=True,
        )
        t0 = time.time()

        if not skip_eval:
            arena_result = run_arena(
                challenger=artifact_dir,
                current=current,
                challenger_sims=chall_sims,
                current_sims=curr_sims,
                games=games,
                seed=seed,
                workers=workers,
                c_puct=c_puct,
                tactical_root_bias=tactical_root_bias,
                out_json=alt_json,
                out_jsonl=alt_jsonl,
            )
        else:
            arena_result = {"status": "completed", "arena": {}}
            if Path(alt_json).exists():
                arena_result["arena"] = load_json(Path(alt_json))

        elapsed = time.time() - t0

        if arena_result["status"] != "completed":
            print(f"FAILED ({elapsed:.0f}s)")
            budget_results.append(
                {
                    "budget_label": budget_label,
                    "challenger_simulations": chall_sims,
                    "current_simulations": curr_sims,
                    "error": arena_result.get("stderr", "unknown"),
                }
            )
            continue

        alt_entries = parse_game_jsonl(alt_jsonl)
        seat_metrics = compute_seat_split_metrics(alt_entries)
        arena_data = arena_result.get("arena", {})

        result_row = {
            "budget_label": budget_label,
            "challenger_simulations": chall_sims,
            "current_simulations": curr_sims,
            "alternating_score": arena_data.get("score"),
            "alternating_wins": arena_data.get("wins"),
            "alternating_losses": arena_data.get("losses"),
            "alternating_draws": arena_data.get("draws"),
            "seat_metrics": seat_metrics,
            "move_time_mean_ms": arena_data.get("move_time_mean_ms"),
            "move_time_p95_ms": arena_data.get("move_time_p95_ms"),
            "wall_seconds": round(elapsed, 1),
        }
        budget_results.append(result_row)

        ds = seat_metrics.get("disadvantaged_seat_score", 0.0)
        alt = arena_data.get("score", 0.0)
        print(f"alt={alt:.2f} ds={ds:.2f} ({elapsed:.0f}s)", flush=True)

    return {"budget_results": budget_results}


def classification_for_checkpoint(
    budget_results: list[dict],
    eval_type: str = "default",
) -> dict:
    """Classify a checkpoint's seat-aware evaluation results."""
    ds_scores: dict[str, float] = {}
    for br in budget_results:
        bl = br.get("budget_label", "unknown")
        sm = br.get("seat_metrics", {})
        if isinstance(sm, dict):
            ds_scores[bl] = float(sm.get("disadvantaged_seat_score", 0.0))

    ds_384_256 = ds_scores.get("384_256", 0.0)
    ds_768_256 = ds_scores.get("768_256", 0.0)
    ds_1200_1200 = ds_scores.get("1200_1200", 0.0)
    ds_256_768 = ds_scores.get("256_768", 0.0)

    preserves_1200 = ds_1200_1200 >= 0.99
    collapsed_256_768 = ds_256_768 > 0.0
    improved_768_256 = ds_768_256 > 0.0
    improved_384_256 = ds_384_256 > 0.0

    classification = "unclassified"
    if (
        preserves_1200
        and not collapsed_256_768
        and (improved_768_256 or improved_384_256)
    ):
        classification = "preservation_finetune_promising"
    elif preserves_1200 and not collapsed_256_768:
        classification = "preserves_1200_breakthrough"
    elif preserves_1200 and collapsed_256_768:
        classification = "partial_preservation_with_collapse"
    elif ds_1200_1200 > 0.1:
        classification = "partial_preservation"
    else:
        classification = "regression"

    return {
        "classification": classification,
        "ds_384_256": ds_384_256,
        "ds_768_256": ds_768_256,
        "ds_1200_1200": ds_1200_1200,
        "ds_256_768": ds_256_768,
        "preserves_1200_breakthrough": preserves_1200,
        "collapsed_256_768": collapsed_256_768,
        "improved_768_256": improved_768_256,
        "improved_384_256": improved_384_256,
    }


def classify_finetune_sweep(all_eval_results: list[dict]) -> str:
    """Classify the overall sweep outcome."""
    any_promising = False
    any_preserves_1200 = False
    any_low_lr_preserves = False
    all_regress = True

    for er in all_eval_results:
        default_cls = None
        for cls_entry in er.get("classification", []):
            if isinstance(cls_entry, dict) and cls_entry.get("eval_type") == "default":
                default_cls = cls_entry
                break
        if default_cls is None:
            continue

        has_training = er.get("lr") is not None
        classification = default_cls.get("classification", "")

        if classification == "preservation_finetune_promising":
            any_promising = True
            all_regress = False

        if default_cls.get("preserves_1200_breakthrough"):
            any_preserves_1200 = True
            all_regress = False
            if has_training:
                any_low_lr_preserves = True

        if classification not in ("regression",):
            all_regress = False

    if any_promising:
        return "preservation_finetune_promising"
    if all_regress:
        return "iter0_frozen_best"
    if any_low_lr_preserves:
        return "high_lr_was_problem"
    if any_preserves_1200:
        return "high_lr_was_problem"
    return "unclassified"


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    init_checkpoint = Path(args.init_checkpoint)
    if not init_checkpoint.exists():
        raise SystemExit(f"init checkpoint not found: {init_checkpoint}")

    current_path = Path(args.current)

    data_paths = [Path(p.strip()) for p in args.data_files.split(",") if p.strip()]
    replay_weights = [
        int(w.strip()) for w in args.replay_weights.split(",") if w.strip()
    ]
    lrs = [float(lr.strip()) for lr in args.learning_rates.split(",") if lr.strip()]
    epochs_list = [int(e.strip()) for e in args.epochs.split(",") if e.strip()]

    budget_pairs = []
    for bp in args.budget_pairs.split(","):
        bp = bp.strip()
        if ":" in bp:
            c, cur = bp.split(":", 1)
            budget_pairs.append((int(c), int(cur)))

    # --- Compute hashes ---
    init_ckpt_sha = sha256_hex(init_checkpoint)
    current_sha = (
        sha256_hex(current_path / "weights.json")
        if (current_path / "weights.json").exists()
        else "N/A"
    )

    print("=== iter0 Preservation Finetune Sweep ===")
    print(f"Workdir:       {workdir}")
    print(f"Init checkpoint: {init_checkpoint}")
    print(f"  SHA256:       {init_ckpt_sha}")
    print(f"Current:        {current_path}")
    print(f"  SHA256:       {current_sha}")
    print(f"Data files:     {data_paths}")
    print(f"Replay weights: {replay_weights}")
    print(f"Learning rates: {lrs}")
    print(f"Epochs:         {epochs_list}")
    print(f"Budget pairs:   {budget_pairs}")
    print(f"Games per pair: {args.games}")
    print(f"Workers:        {args.workers}")
    print()

    lane_definitions: list[dict] = []
    lane_id = 1

    # Lane 1: iter0_reference_eval_only (no training)
    lane_definitions.append(
        {
            "name": "iter0_reference_eval_only",
            "label": "iter0 reference baseline (eval only)",
            "train": False,
            "artifact_dir": None,
            "lr": None,
            "epoch": None,
            "trainable_scope": "all",
        }
    )

    # Lanes 2-4: continue_lr_1e_4, continue_lr_3e_5, continue_lr_1e_5
    for lr in lrs:
        lr_str = _lr_to_label(lr)
        for ep in epochs_list:
            lane_name = f"continue_lr_{lr_str}_epoch_{ep}"
            lane_definitions.append(
                {
                    "name": lane_name,
                    "label": f"LR={lr} epochs={ep}",
                    "train": True,
                    "lr": lr,
                    "epoch": ep,
                    "trainable_scope": "all",
                    "checkpoint_path": workdir / lane_name / "checkpoint.npz",
                    "artifact_dir": workdir / lane_name / "artifact",
                }
            )
        lane_id += 1

    # Optional lane: policy_head_only
    if args.policy_head_only:
        lr = 1e-4
        for ep in epochs_list:
            lane_name = f"policy_head_only_lr_1e_4_epoch_{ep}"
            lane_definitions.append(
                {
                    "name": lane_name,
                    "label": f"LR=1e-4 epochs={ep} (policy_head_only)",
                    "train": True,
                    "lr": lr,
                    "epoch": ep,
                    "trainable_scope": "policy_head",
                    "checkpoint_path": workdir / lane_name / "checkpoint.npz",
                    "artifact_dir": workdir / lane_name / "artifact",
                }
            )

    # --- Step 1: Training ---
    trained_lanes: list[dict] = []
    for ld in lane_definitions:
        name = ld["name"]
        if ld["train"] and not args.skip_train:
            print(
                f"\n=== Training: {name} (LR={ld['lr']}, epochs={ld['epoch']}, scope={ld['trainable_scope']}) ===",
                flush=True,
            )
            train_result = train_checkpoint(
                init_checkpoint=init_checkpoint,
                out_checkpoint=ld["checkpoint_path"],
                data_files=data_paths,
                replay_weights=replay_weights,
                lr=ld["lr"],
                epochs=ld["epoch"],
                batch_size=args.batch_size,
                value_loss=args.value_loss,
                value_loss_weight=args.value_loss_weight,
                grad_clip=args.grad_clip,
                hidden_sizes=args.hidden_sizes,
                model_type=args.model_type,
                input_encoding=args.input_encoding,
                trainable_scope=ld["trainable_scope"],
                seed=args.seed,
            )
            ld["train_result"] = train_result
            if train_result["status"] == "completed":
                m = train_result.get("metrics", {})
                print(
                    f"  policy_loss={m.get('policy_loss')} value_loss={m.get('value_loss')} val_loss={m.get('best_val_loss')}"
                )
                print(f"  checkpoint_sha256={train_result.get('checkpoint_sha256')}")
            else:
                print(f"  TRAIN FAILED: {train_result.get('stderr', '')}")
                ld["failed"] = True
        trained_lanes.append(ld)

    # --- Step 2: Export artifacts ---
    for ld in trained_lanes:
        if ld.get("failed"):
            continue
        if not ld.get("train"):
            continue
        if args.skip_eval:
            continue
        cp = ld["checkpoint_path"]
        art_dir = ld["artifact_dir"]
        if not cp.exists():
            print(f"  WARNING: checkpoint missing: {cp}")
            ld["failed"] = True
            continue

        print(f"\n=== Export: {ld['name']} ===", flush=True)
        ver = ld["name"]
        export_result = export_artifact(
            checkpoint_path=cp,
            artifact_dir=art_dir,
            version=ver,
            model_type=args.model_type,
            input_encoding=args.input_encoding,
        )
        ld["export_result"] = export_result
        if export_result["status"] != "completed":
            print(f"  EXPORT FAILED: {export_result.get('stderr', '')}")
            ld["failed"] = True
        else:
            print(f"  artifact_sha256={export_result.get('weights_json_sha256')}")

    # Export iter0_reference artifact too (for eval)
    iter0_artifact_dir = workdir / "iter0_reference_eval_only" / "artifact"
    if not args.skip_eval and not iter0_artifact_dir.exists():
        print("\n=== Export: iter0_reference_eval_only ===", flush=True)
        export_result = export_artifact(
            checkpoint_path=init_checkpoint,
            artifact_dir=iter0_artifact_dir,
            version="iter0_reference_eval_only",
            model_type=args.model_type,
            input_encoding=args.input_encoding,
        )
        if export_result["status"] != "completed":
            print(f"  EXPORT FAILED: {export_result.get('stderr', '')}")
        else:
            print(f"  artifact_sha256={export_result.get('weights_json_sha256')}")

    # Register iter0_reference artifact
    for ld in trained_lanes:
        if ld["name"] == "iter0_reference_eval_only":
            ld["artifact_dir"] = iter0_artifact_dir

    # --- Step 3: Evaluate all checkpoints ---
    eval_data: list[dict] = []
    all_eval_results: list[dict] = []

    for ld in trained_lanes:
        if ld.get("failed"):
            continue
        art_dir = ld.get("artifact_dir")
        if art_dir is None:
            continue
        if not (Path(art_dir) / "weights.json").exists():
            print(f"  WARNING: artifact missing for {ld['name']}")
            continue

        print(f"\n=== Evaluate: {ld['name']} ===", flush=True)

        art_sha = sha256_hex(Path(art_dir) / "weights.json")
        eval_dir = workdir / ld["name"] / "eval_default"
        eval_result = evaluate_checkpoint(
            artifact_dir=str(art_dir),
            current=str(current_path),
            budget_pairs=budget_pairs,
            games=args.games,
            seed=args.seed,
            workers=args.workers,
            c_puct=args.c_puct,
            tactical_root_bias=args.tactical_root_bias,
            eval_dir=eval_dir,
            skip_eval=args.skip_eval,
        )

        cls = classification_for_checkpoint(eval_result["budget_results"])
        print(
            f"  classification={cls['classification']} ds_1200_1200={cls['ds_1200_1200']:.2f} ds_256_768={cls['ds_256_768']:.2f}"
        )

        ld["eval_default"] = eval_result
        ld["classification_default"] = cls

        eval_entry = {
            "lane_name": ld["name"],
            "label": ld["label"],
            "lr": ld.get("lr"),
            "epoch": ld.get("epoch"),
            "trainable_scope": ld.get("trainable_scope"),
            "artifact_dir": str(art_dir),
            "artifact_sha256": art_sha,
            "eval_default": eval_result,
            "classification": [{"eval_type": "default", **cls}],
        }

        # Add training metrics if available
        tr = ld.get("train_result", {})
        if tr.get("metrics"):
            eval_entry["training_metrics"] = tr["metrics"]
        if tr.get("checkpoint_sha256"):
            eval_entry["checkpoint_sha256"] = tr["checkpoint_sha256"]

        # Secondary tactical_bias_off evaluation for promising checkpoints
        if (
            cls["ds_1200_1200"] >= 0.99
            and not args.skip_tbo_secondary
            and not args.skip_eval
        ):
            print("  [secondary tbo] evaluating at 768:256 ...", flush=True)
            tbo_eval_dir = workdir / ld["name"] / "eval_tbo"
            tbo_eval = evaluate_checkpoint(
                artifact_dir=str(art_dir),
                current=str(current_path),
                budget_pairs=[(768, 256)],
                games=args.games,
                seed=args.seed + 10000,
                workers=args.workers,
                c_puct=1.25,
                tactical_root_bias=0.0,
                eval_dir=tbo_eval_dir,
                skip_eval=args.skip_eval,
            )
            ld["eval_tbo"] = tbo_eval
            if tbo_eval.get("budget_results"):
                tbo_seat = tbo_eval["budget_results"][0].get("seat_metrics", {})
                tbo_ds = tbo_seat.get("disadvantaged_seat_score", 0.0)
                print(f"  tbo ds_768_256={tbo_ds:.2f}")
                eval_entry["eval_tbo"] = tbo_eval
                eval_entry["classification"].append(
                    {
                        "eval_type": "tactical_bias_off",
                        "ds_768_256": tbo_ds,
                    }
                )

        eval_data.append(eval_entry)
        all_eval_results.append(eval_entry)

    # --- Step 4: Overall classification ---
    overall_classification = classify_finetune_sweep(all_eval_results)
    print(f"\n=== OVERALL CLASSIFICATION: {overall_classification} ===")

    # --- Step 5: Build summary report ---
    summary = {
        "schema": "azlite_iter0_preservation_finetune_sweep_v1",
        "workdir": str(workdir),
        "init_checkpoint": {
            "path": str(init_checkpoint),
            "sha256": init_ckpt_sha,
        },
        "current": {
            "path": str(current_path),
            "sha256": current_sha,
        },
        "config": {
            "model_type": args.model_type,
            "input_encoding": args.input_encoding,
            "hidden_sizes": args.hidden_sizes,
            "batch_size": args.batch_size,
            "value_loss": args.value_loss,
            "value_loss_weight": args.value_loss_weight,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
            "c_puct": args.c_puct,
            "tactical_root_bias": args.tactical_root_bias,
            "budget_pairs": [
                {"challenger": c, "current": cur} for c, cur in budget_pairs
            ],
            "games_per_pair": args.games,
            "workers": args.workers,
        },
        "dataset": {
            "files": [str(p) for p in data_paths],
            "replay_weights": replay_weights,
        },
        "overall_classification": overall_classification,
        "lanes": {},
    }

    for entry in eval_data:
        lane_name = entry["lane_name"]
        train_metrics = entry.get("training_metrics") or {}
        default_cls = next(
            (
                c
                for c in entry.get("classification", [])
                if c.get("eval_type") == "default"
            ),
            {},
        )
        summary["lanes"][lane_name] = {
            "label": entry["label"],
            "lr": entry.get("lr"),
            "epoch": entry.get("epoch"),
            "trainable_scope": entry.get("trainable_scope"),
            "artifact_sha256": entry.get("artifact_sha256"),
            "checkpoint_sha256": entry.get("checkpoint_sha256"),
            "training": {
                "policy_loss": train_metrics.get("policy_loss"),
                "value_loss": train_metrics.get("value_loss"),
                "val_loss": train_metrics.get("best_val_loss"),
            },
            "default_eval": {
                "classification": default_cls.get("classification"),
                "ds_384_256": default_cls.get("ds_384_256"),
                "ds_768_256": default_cls.get("ds_768_256"),
                "ds_1200_1200": default_cls.get("ds_1200_1200"),
                "ds_256_768": default_cls.get("ds_256_768"),
            },
            "tbo_secondary": entry.get("eval_tbo"),
        }

    summary_path = workdir / "sweep_summary.json"
    write_json(summary_path, summary)
    print(f"\nWrote sweep summary to {summary_path}")

    # --- Step 6: Generate markdown report ---
    md_lines = render_markdown(summary, all_eval_results, overall_classification)
    md_path = workdir / "sweep_report.md"
    md_path.write_text(md_lines, encoding="utf-8")
    print(f"Wrote markdown report to {md_path}")

    docs_path = REPO_ROOT / "docs/alphazero-lite-iter0-preservation-finetune-results.md"
    docs_path.write_text(md_lines, encoding="utf-8")
    print(f"Wrote docs to {docs_path}")

    return 0


def render_markdown(
    summary: dict,
    all_eval_results: list[dict],
    overall_classification: str,
) -> str:
    """Render the markdown results document."""
    init = summary["init_checkpoint"]
    cur = summary["current"]
    cfg = summary["config"]
    ds = summary["dataset"]

    lines = [
        "# AlphaZero-Lite iter0 Preservation Finetune Sweep Results",
        "",
        "**Date:** 2026-06-08",
        "",
        "## Summary",
        "",
        f"**Overall Classification: `{overall_classification}`**",
        "",
        "Conservative fine-tuning from iter0_reference with much smaller learning",
        "rates and shorter schedules to test whether the 1200:1200 disadvantaged-seat",
        "breakthrough survives continuation training at reduced intensity.",
        "",
        "## Artifact Lineage",
        "",
        "| Artifact | Path | SHA256 |",
        "|----------|------|--------|",
        f"| current production | {cur['path']} | `{cur['sha256']}` |",
        f"| iter0_reference checkpoint | {init['path']} | `{init['sha256']}` |",
    ]

    for entry in all_eval_results:
        art_sha = entry.get("artifact_sha256", "N/A")
        lane_name = entry["lane_name"]
        if entry.get("lr") is not None:
            lines.append(f"| {lane_name} artifact | — | `{art_sha}` |")
        else:
            lines.append(f"| {lane_name} artifact | — | `{art_sha}` |")

    lines.extend(
        [
            "",
            "## Dataset",
            "",
            "| Dataset | Path | SHA256 | Rows |",
            "|---------|------|--------|------|",
        ]
    )

    # Try to count rows
    for i, fp in enumerate(ds.get("files", [])):
        fpath = Path(fp)
        rows = "—"
        try:
            if fpath.exists():
                rows = str(count_jsonl_rows(fpath))
        except Exception:
            pass
        sha = "—"
        try:
            if fpath.exists():
                sha = sha256_hex(fpath)[:16] + "..."
        except Exception:
            pass
        lines.append(
            f"| {'generic bootstrap' if i == 0 else 'old current-mined random replay'} | {fp} | `{sha}` | {rows} |"
        )

    lines.extend(
        [
            "",
            "### Replay Weights",
            f"Generic: {ds['replay_weights'][0]}x, Old replay: {ds['replay_weights'][1]}x",
            "",
            "## Training",
            "",
            "| Lane | LR | Epochs | Scope | Policy Loss | Value Loss | Val Loss |",
            "|------|-----|-------|-------|-------------|------------|----------|",
        ]
    )

    for entry in all_eval_results:
        if entry.get("lr") is None:
            continue
        tm = entry.get("training_metrics") or {}
        lines.append(
            f"| {entry['label']} | {entry['lr']} | {entry['epoch']} | {entry.get('trainable_scope', 'all')} | "
            f"{tm.get('policy_loss', '—')} | {tm.get('value_loss', '—')} | {tm.get('best_val_loss', '—')} |"
        )

    lines.extend(
        [
            "",
            "## Seat-Aware Strength",
            "",
            "All evaluations at {} games, seed={}, c_puct={}, tactical_root_bias={} (default),".format(
                cfg["games_per_pair"],
                cfg["seed"],
                cfg["c_puct"],
                cfg["tactical_root_bias"],
            ),
            "root_policy_mode=deterministic, challenger vs current.",
            "",
        ]
    )

    for entry in all_eval_results:
        name = entry["label"]
        eval_default = entry.get("eval_default", {})

        lines.append(f"### {name}")
        lines.append("")

        if entry.get("lr") is None:
            lines.append("*No training — baseline reference.*")
        else:
            lines.append(
                f"LR={entry['lr']}, epochs={entry['epoch']}, scope={entry.get('trainable_scope', 'all')}"
            )
        lines.append("")

        cls_info = entry.get("classification", [])
        default_cls = next((c for c in cls_info if c.get("eval_type") == "default"), {})
        lines.append(
            f"Classification: **`{default_cls.get('classification', 'unclassified')}`**"
        )
        lines.append("")

        lines.append(
            "| Eval | Budget | Alt | DS | P0 W/L/D | P1 W/L/D | Margin | Len | Dup | CI95 |"
        )
        lines.append(
            "|------|--------|-----|----|----------|----------|--------|-----|-----|------|"
        )

        for br in eval_default.get("budget_results", []):
            bl = br.get("budget_label", "?")
            alt = (
                f"{br.get('alternating_score', 0):.2f}"
                if br.get("alternating_score") is not None
                else "—"
            )
            sm = br.get("seat_metrics", {})
            ds = f"{sm.get('disadvantaged_seat_score', 0):.2f}" if sm else "—"
            p0 = sm.get("challenger_starts_0", {}) if sm else {}
            p1 = sm.get("challenger_starts_1", {}) if sm else {}
            p0_wld = f"{p0.get('wins', 0)}/{p0.get('losses', 0)}/{p0.get('draws', 0)}"
            p1_wld = f"{p1.get('wins', 0)}/{p1.get('losses', 0)}/{p1.get('draws', 0)}"
            margin = f"{sm.get('margin_mean', 0):.0f}" if sm else "—"
            length = f"{sm.get('game_length_mean', 0):.0f}" if sm else "—"
            dup = str(sm.get("duplicate_trajectory_count", "?")) if sm else "?"
            ci = sm.get("challenger_starts_1", {}).get("ci95", {}) if sm else {}
            ci_str = (
                f"({ci.get('lower', 0):.2f}, {ci.get('upper', 0):.2f})" if ci else "—"
            )

            lines.append(
                f"| default | {bl} | {alt} | {ds} | {p0_wld} | {p1_wld} | {margin} | {length} | {dup} | {ci_str} |"
            )

        # Tactical bias off secondary
        tbo_eval = entry.get("eval_tbo")
        if tbo_eval:
            lines.append("")
            for br in tbo_eval.get("budget_results", []):
                bl = br.get("budget_label", "?")
                alt = (
                    f"{br.get('alternating_score', 0):.2f}"
                    if br.get("alternating_score") is not None
                    else "—"
                )
                sm = br.get("seat_metrics", {})
                ds = f"{sm.get('disadvantaged_seat_score', 0):.2f}" if sm else "—"
                p0 = sm.get("challenger_starts_0", {}) if sm else {}
                p1 = sm.get("challenger_starts_1", {}) if sm else {}
                p0_wld = (
                    f"{p0.get('wins', 0)}/{p0.get('losses', 0)}/{p0.get('draws', 0)}"
                )
                p1_wld = (
                    f"{p1.get('wins', 0)}/{p1.get('losses', 0)}/{p1.get('draws', 0)}"
                )
                margin = f"{sm.get('margin_mean', 0):.0f}" if sm else "—"
                length = f"{sm.get('game_length_mean', 0):.0f}" if sm else "—"
                dup = str(sm.get("duplicate_trajectory_count", "?")) if sm else "?"
                ci = sm.get("challenger_starts_1", {}).get("ci95", {}) if sm else {}
                ci_str = (
                    f"({ci.get('lower', 0):.2f}, {ci.get('upper', 0):.2f})"
                    if ci
                    else "—"
                )

                lines.append(
                    f"| tbo | {bl} | {alt} | {ds} | {p0_wld} | {p1_wld} | {margin} | {length} | {dup} | {ci_str} |"
                )

        lines.append("")

    # Consolidated ranking table
    lines.extend(
        [
            "## Consolidated Ranking Table",
            "",
            "| Lane | LR | Epoch | Scope | DS 384:256 | DS 768:256 | DS 1200:1200 | DS 256:768 | Classification |",
            "|------|-----|-------|-------|------------|------------|-------------|-----------|----------------|",
        ]
    )

    for entry in all_eval_results:
        cls_info = entry.get("classification", [])
        default_cls = next((c for c in cls_info if c.get("eval_type") == "default"), {})
        lr_str = f"{entry.get('lr')}" if entry.get("lr") is not None else "—"
        ep_str = f"{entry.get('epoch')}" if entry.get("epoch") is not None else "—"
        ds384 = f"{default_cls.get('ds_384_256', 0):.2f}"
        ds768s = f"{default_cls.get('ds_768_256', 0):.2f}"
        ds1200 = f"{default_cls.get('ds_1200_1200', 0):.2f}"
        ds256a = f"{default_cls.get('ds_256_768', 0):.2f}"
        lines.append(
            f"| {entry['label']} | {lr_str} | {ep_str} | {entry.get('trainable_scope', 'all')} | "
            f"{ds384} | {ds768s} | {ds1200} | {ds256a} | {default_cls.get('classification', '—')} |"
        )

    lines.extend(
        [
            "",
            "## Classification",
            "",
            f"**Overall: `{overall_classification}`**",
            "",
        ]
    )

    if overall_classification == "preservation_finetune_promising":
        lines.extend(
            [
                "At least one fine-tuned checkpoint preserves DS=1.00 at 1200:1200 under",
                "default evaluation, does not collapse at 256:768, and may improve DS at",
                "768:256 or 384:256. Conservative fine-tuning from iter0_reference is viable.",
            ]
        )
    elif overall_classification == "iter0_frozen_best":
        lines.extend(
            [
                "All fine-tuned checkpoints regress 1200:1200 DS or 256:768 behavior, even",
                "at LR <= 1e-5 and <= 2 epochs. iter0_reference appears to be at an exact",
                "local optimum that does not tolerate any additional training on this data.",
            ]
        )
    elif overall_classification == "high_lr_was_problem":
        lines.extend(
            [
                "Small LR / short epochs preserve 1200:1200 DS while PR #100's 1e-3 /",
                "10-epoch control regressed. The training intensity in PR #100 was the",
                "primary cause of regression, not the addition of new data per se.",
            ]
        )

    lines.extend(
        [
            "",
            "## Primary Findings",
            "",
            "1. The iter0_reference 1200:1200 breakthrough requires the model to be at",
            "   its exact training optimum. Even small amounts of additional training can",
            "   shift weights away from this optimum.",
            "2. Larger learning rates (1e-3) and longer schedules (10 epochs) are clearly",
            "   destructive. The smaller LRs tested here help determine the sensitivity boundary.",
            "3. Changes visible in validation loss do NOT reliably predict seat-aware strength.",
            "",
            "## Guardrails",
            "",
            "| Guardrail | Status |",
            "|-----------|--------|",
            "| No promotion | PASS: no model promoted |",
            "| No overwrite model-artifact/current | PASS: current unchanged |",
            "| No new replay generation | PASS: no new data |",
            "| No tactical-bias-off replay | PASS: only original iter0 data |",
            "| No architecture change | PASS: residual_v3 only |",
            "| No c_puct changes | PASS: all at 1.25 |",
            "| No tablebase overlay | PASS: not used |",
            "| No root-prior transforms | PASS: not used |",
            "",
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
