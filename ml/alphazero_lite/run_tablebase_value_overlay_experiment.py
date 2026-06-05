#!/usr/bin/env python3
"""Run tablebase-value-overlay experiment: baseline, overwrite, blend lanes.

Each lane generates a dataset via classic-MCTS self-play, optionally
overwriting or blending value targets with exact tablebase values, then
trains a model and runs evaluation.

Important guardrails:
- Do NOT promote any model.
- Do NOT overwrite storage/ai/alphazero_lite/current.
- Do NOT change model architecture.
- Do NOT change policy target shaping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def venv_python() -> str:
    candidates = [
        REPO_ROOT / ".venv/bin/python",
        REPO_ROOT.parents[1] / ".venv/bin/python",
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return sys.executable


def run_step(name: str, command: list[str], *, timeout: int | None = None) -> dict:
    started = time.time()
    print(f"\n--- {name} ---", flush=True)
    print(f"  {' '.join(command)}", flush=True)
    result = subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    duration = round(time.time() - started, 2)
    success = result.returncode == 0
    if not success:
        print(f"  FAILED after {duration}s", flush=True)
        print(f"  stdout: {result.stdout[-2000:]}", flush=True)
        print(f"  stderr: {result.stderr[-2000:]}", flush=True)
    else:
        print(f"  OK ({duration}s)", flush=True)
        for line in result.stdout.strip().splitlines():
            print(f"    {line}")
    return {
        "name": name,
        "status": "ok" if success else "failed",
        "duration_s": duration,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": command,
    }


def _parse_kv_from_line(line: str, prefix: str) -> dict[str, str] | None:
    if not line.startswith(prefix):
        return None
    rest = line[len(prefix) :].strip()
    if not rest:
        return None
    parsed: dict[str, str] = {}
    for match in re.finditer(r"(\w+)=(\S+)", rest):
        parsed[match.group(1)] = match.group(2)
    return parsed if parsed else None


def parse_dataset_metrics(stdout: str) -> dict:
    metrics: dict = {}
    for line in stdout.splitlines():
        line = line.strip()
        kv = _parse_kv_from_line(line, "dataset_stats")
        if kv:
            for key in (
                "rows_written",
                "positions_visited",
                "positions_searched",
                "games_processed",
            ):
                if key in kv:
                    try:
                        metrics[key] = int(kv[key])
                    except (ValueError, TypeError):
                        metrics[key] = kv[key]
            continue
        kv = _parse_kv_from_line(line, "tablebase_value_overlay_summary")
        if kv:
            for key in (
                "overlay",
                "tablebase_rows",
                "coverage_rate",
                "mean_abs_value_delta",
                "max_abs_value_delta",
                "coverage_early",
                "coverage_mid",
                "coverage_late",
            ):
                if key in kv:
                    try:
                        metrics[key] = float(kv[key])
                    except (ValueError, TypeError):
                        metrics[key] = kv[key]
            if "tablebase_rows" in metrics and isinstance(
                metrics.get("tablebase_rows"), float
            ):
                metrics["tablebase_rows"] = int(metrics["tablebase_rows"])
            if "coverage_early" in metrics and isinstance(
                metrics.get("coverage_early"), float
            ):
                metrics["coverage_early"] = int(metrics["coverage_early"])
            if "coverage_mid" in metrics and isinstance(
                metrics.get("coverage_mid"), float
            ):
                metrics["coverage_mid"] = int(metrics["coverage_mid"])
            if "coverage_late" in metrics and isinstance(
                metrics.get("coverage_late"), float
            ):
                metrics["coverage_late"] = int(metrics["coverage_late"])
            continue
    return metrics


def parse_training_metrics(stdout: str) -> dict:
    metrics: dict = {}
    top_k_paths: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        m = re.match(r"^policy_loss=(\S+)", line)
        if m:
            metrics["policy_loss"] = float(m.group(1))
            continue
        m = re.match(r"^value_loss=(\S+)", line)
        if m:
            metrics["value_loss"] = float(m.group(1))
            continue
        m = re.match(r"^best_val_loss=(\S+)", line)
        if m:
            metrics["best_val_loss"] = float(m.group(1))
            continue
        m = re.match(r"^saved_top_k=(\d+)", line)
        if m:
            metrics["saved_top_k"] = int(m.group(1))
            continue
        m = re.match(r"^saved_top_checkpoint_(\d+)=(\S+)\s+val_loss=(\S+)", line)
        if m:
            top_k_paths.append(m.group(2))
            continue
    metrics["top_k_checkpoint_paths"] = top_k_paths
    return metrics


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compare_lane_datasets(baseline_path: Path, overwrite_path: Path) -> dict:
    if not baseline_path.is_file():
        return {"error": f"baseline dataset not found: {baseline_path}"}
    if not overwrite_path.is_file():
        return {"error": f"overwrite dataset not found: {overwrite_path}"}

    baseline_rows = []
    with baseline_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                baseline_rows.append(json.loads(line))

    overwrite_rows = []
    with overwrite_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                overwrite_rows.append(json.loads(line))

    if len(baseline_rows) != len(overwrite_rows):
        return {
            "error": (
                f"row count mismatch: baseline={len(baseline_rows)} "
                f"overwrite={len(overwrite_rows)}"
            ),
            "baseline_row_count": len(baseline_rows),
            "overwrite_row_count": len(overwrite_rows),
        }

    policy_diff_count = 0
    state_diff_count = 0
    player_diff_count = 0
    move_index_diff_count = 0
    value_diff_rows = 0
    value_match_rows = 0

    for i, (b_row, o_row) in enumerate(zip(baseline_rows, overwrite_rows)):
        b_state = b_row.get("state")
        o_state = o_row.get("state")
        if b_state != o_state:
            state_diff_count += 1

        b_policy = b_row.get("policy")
        o_policy = o_row.get("policy")
        if b_policy != o_policy:
            policy_diff_count += 1

        if b_row.get("player") != o_row.get("player"):
            player_diff_count += 1
        if b_row.get("move_index") != o_row.get("move_index"):
            move_index_diff_count += 1

        b_value = b_row.get("value")
        o_value = o_row.get("value")
        if b_value != o_value:
            value_diff_rows += 1
        else:
            value_match_rows += 1

    return {
        "total_rows": len(baseline_rows),
        "policy_diff_count": policy_diff_count,
        "state_diff_count": state_diff_count,
        "player_diff_count": player_diff_count,
        "move_index_diff_count": move_index_diff_count,
        "value_diff_rows": value_diff_rows,
        "value_match_rows": value_match_rows,
    }


def generate_production_report(
    results: dict,
    workdir: Path,
    args: argparse.Namespace,
    comparison: dict | None,
) -> str:
    lines: list[str] = []
    lines.append("# AlphaZero-Lite Tablebase Value Overlay — Production Results")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    lines.append(
        "**Experiment:** Production-scale tablebase value overlay strength evaluation"
    )
    lines.append("")
    lines.append("## 1. Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Games | {args.games} |")
    lines.append("| Engine | Classic MCTS |")
    lines.append(f"| Simulations | {args.simulations} |")
    lines.append(f"| Seed | {args.seed} |")
    lines.append(f"| Workers | {args.workers} |")
    lines.append(f"| Max positions per game | {args.max_positions_per_game} |")
    lines.append("| Input encoding | `kalah_v3` |")
    lines.append("| Policy target mode | `sharpened` |")
    lines.append("| Value target mode | `sharpened` |")
    lines.append("| Model type | `residual_v3` |")
    lines.append("| Hidden sizes | 96,3 |")
    lines.append(f"| Epochs | {args.epochs} |")
    lines.append(f"| Batch size | {args.batch_size} |")
    lines.append("| Value loss | huber, huber_delta=1.0 |")
    lines.append("| Value loss weight | 0.3 |")
    lines.append("| Val split | 0.1 |")
    lines.append("| Grad clip | 1.0 |")
    lines.append(f"| Arena games | {args.arena_games} |")
    lines.append(f"| MCTS games | {args.mcts_games} |")
    lines.append("")

    lanes_ran = [k for k in results]
    lines.append("## 2. Lanes")
    lines.append("")
    lines.append("| Lane | Tablebase value overlay |")
    lines.append("|------|------------------------|")
    for lane in lanes_ran:
        if lane == "baseline":
            lines.append(f"| {lane} | off |")
        elif lane == "overwrite":
            lines.append(f"| {lane} | overwrite |")
        elif lane.startswith("blend"):
            lines.append(f"| {lane} | blend |")
    lines.append("")

    lines.append("## 3. Dataset Metrics")
    lines.append("")
    ds_headers = [
        "Metric",
        *[lane for lane in lanes_ran if lane in results],
    ]
    lines.append("| " + " | ".join(ds_headers) + " |")
    lines.append("|" + "|".join(["---"] * len(ds_headers)) + "|")

    ds_metric_keys = [
        "rows_written",
        "positions_visited",
        "tablebase_rows",
        "tablebase_coverage_rate",
        "coverage_early",
        "coverage_mid",
        "coverage_late",
        "mean_abs_value_delta",
        "max_abs_value_delta",
    ]
    for key in ds_metric_keys:
        vals = []
        for lane in lanes_ran:
            lr = results.get(lane, {})
            ds = lr.get("dataset_metrics", {})
            v = ds.get(key)
            if v is None:
                vals.append("N/A")
            elif isinstance(v, float):
                if key in (
                    "tablebase_coverage_rate",
                    "mean_abs_value_delta",
                    "max_abs_value_delta",
                ):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(
                        f"{v:.4f}" if key == "tablebase_coverage_rate" else f"{v}"
                    )
            else:
                vals.append(str(v))
        lines.append("| " + key + " | " + " | ".join(vals) + " |")
    lines.append("")

    if comparison:
        lines.append("### Row-Level Comparison (baseline vs overwrite)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| total_rows | {comparison.get('total_rows', 'N/A')} |")
        lines.append(
            f"| policy_diff_count | {comparison.get('policy_diff_count', 'N/A')} |"
        )
        lines.append(
            f"| state_diff_count | {comparison.get('state_diff_count', 'N/A')} |"
        )
        lines.append(
            f"| player_diff_count | {comparison.get('player_diff_count', 'N/A')} |"
        )
        lines.append(
            f"| move_index_diff_count | {comparison.get('move_index_diff_count', 'N/A')} |"
        )
        lines.append(
            f"| value_diff_rows | {comparison.get('value_diff_rows', 'N/A')} |"
        )
        lines.append(
            f"| value_match_rows | {comparison.get('value_match_rows', 'N/A')} |"
        )
        lines.append("")

    lines.append("## 4. Training Metrics")
    lines.append("")
    train_headers = [
        "Metric",
        *[lane for lane in lanes_ran if lane in results],
    ]
    lines.append("| " + " | ".join(train_headers) + " |")
    lines.append("|" + "|".join(["---"] * len(train_headers)) + "|")

    train_metric_keys = ["policy_loss", "value_loss", "best_val_loss"]
    for key in train_metric_keys:
        vals = []
        for lane in lanes_ran:
            lr = results.get(lane, {})
            tm = lr.get("train_metrics", {})
            v = tm.get(key)
            vals.append(f"{v:.6f}" if isinstance(v, float) else "N/A")
        lines.append("| " + key + " | " + " | ".join(vals) + " |")

    # top-k checkpoints
    chk_vals = []
    for lane in lanes_ran:
        lr = results.get(lane, {})
        tm = lr.get("train_metrics", {})
        top_k_paths = tm.get("top_k_checkpoint_paths", [])
        hashes = lr.get("checkpoint_hashes", {})
        parts = []
        for p in top_k_paths:
            fname = Path(p).name if p else "N/A"
            h = hashes.get(p, "")
            if h:
                parts.append(f"{fname} ({h[:12]}...)")
            else:
                parts.append(fname)
        chk_vals.append(", ".join(parts) if parts else "N/A")
    lines.append("| top_k_checkpoints | " + " | ".join(chk_vals) + " |")
    lines.append("")

    lines.append("## 5. Strength Metrics")
    lines.append("")
    for lane in lanes_ran:
        lr = results.get(lane, {})
        gr = lr.get("gate_report", {})
        if gr:
            lines.append(f"### {lane}")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| gate_passed | {gr.get('passed')} |")
            lines.append(f"| arena_score | {gr.get('arena_score')} |")
            lines.append(f"| candidate_mcts_score | {gr.get('candidate_mcts_score')} |")
            lines.append(f"| current_mcts_score | {gr.get('current_mcts_score')} |")
            fr = gr.get("failure_reasons", [])
            lines.append(f"| failure_reasons | {fr if fr else 'none'} |")
            lines.append("")

    lines.append("## 6. Computation Cost")
    lines.append("")
    lines.append(
        "| Lane | Dataset generation (s) | Training (s) | Export (s) | Gate (s) |"
    )
    lines.append(
        "|------|------------------------|-------------|-----------|---------|"
    )
    for lane in lanes_ran:
        lr = results.get(lane, {})
        ds_dur = lr.get("dataset_generation", {}).get("duration_s", "N/A")
        tr_dur = lr.get("train", {}).get("duration_s", "N/A")
        ex_dur = lr.get("export", {}).get("duration_s", "N/A")
        gt_dur = lr.get("gate", {}).get("duration_s", "N/A")
        lines.append(f"| {lane} | {ds_dur} | {tr_dur} | {ex_dur} | {gt_dur} |")
    lines.append("")

    lines.append("## 7. Acceptance Criteria Assessment")
    lines.append("")
    lines.append("| Criterion | Status | Detail |")
    lines.append("|-----------|--------|--------|")
    if comparison:
        pol_diffs = comparison.get("policy_diff_count", -1)
        lines.append(
            f"| Policy targets unchanged | "
            f"{'PASS' if pol_diffs == 0 else 'FAIL'} | "
            f"{pol_diffs} policy diffs across {comparison.get('total_rows', '?')} rows |"
        )
        st_diffs = comparison.get("state_diff_count", -1)
        lines.append(
            f"| State encodings match | "
            f"{'PASS' if st_diffs == 0 else 'FAIL'} | "
            f"{st_diffs} state diffs |"
        )
    lanes_with_gate = [
        name for name in lanes_ran if results.get(name, {}).get("gate_report")
    ]
    for lane in lanes_with_gate:
        gr = results[lane]["gate_report"]
        passed = gr.get("passed")
        arena = gr.get("arena_score")
        mcts = gr.get("candidate_mcts_score")
        reasons = gr.get("failure_reasons", [])
        lines.append(
            f"| {lane} gate passed | {'PASS' if passed else 'FAIL'} | "
            f"arena={arena}, mcts={mcts}, failures={reasons if reasons else 'none'} |"
        )
    lines.append("")

    lines.append("## 8. Artifacts")
    lines.append("")
    lines.append("| Artifact | Path |")
    lines.append("|----------|------|")
    for lane in lanes_ran:
        ds_path = workdir / "datasets" / f"{lane}.jsonl"
        chk_path = workdir / "models" / f"{lane}_checkpoint.npz"
        exp_path = workdir / "models" / lane
        lines.append(f"| {lane} dataset | `{ds_path}` |")
        lines.append(f"| {lane} checkpoint | `{chk_path}` |")
        lines.append(f"| {lane} artifact | `{exp_path}` |")
    lines.append(f"| Experiment summary | `{workdir / 'experiment_summary.json'}` |")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tablebase value overlay experiment lanes"
    )
    parser.add_argument(
        "--workdir",
        default="/tmp/azlite_tb_value_experiment",
        help="Working directory for outputs",
    )
    parser.add_argument(
        "--lanes",
        default="baseline,overwrite",
        help="Comma-separated lanes: baseline, overwrite, blend",
    )
    parser.add_argument(
        "--games", type=int, default=200, help="Games per dataset generation"
    )
    parser.add_argument(
        "--simulations", type=int, default=1200, help="Simulations per MCTS search"
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--workers", type=int, default=6, help="Parallel workers for data generation"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Training batch size"
    )
    parser.add_argument(
        "--max-positions-per-game",
        type=int,
        default=24,
        help="Max positions retained per game",
    )
    parser.add_argument(
        "--skip-train", action="store_true", help="Skip training, only generate data"
    )
    parser.add_argument(
        "--skip-gate", action="store_true", help="Skip local_promotion_gate evaluation"
    )
    parser.add_argument(
        "--arena-games", type=int, default=120, help="Arena games for gate evaluation"
    )
    parser.add_argument(
        "--mcts-games", type=int, default=40, help="MCTS1200 games for gate evaluation"
    )
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    lanes = [lane.strip() for lane in args.lanes.split(",") if lane.strip()]

    if not lanes:
        raise SystemExit("at least one lane is required")

    python = venv_python()
    dataset_shard_dir = workdir / "datasets"
    dataset_shard_dir.mkdir(parents=True, exist_ok=True)
    model_dir = workdir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = workdir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    common_dataset_args = [
        "--teacher-mode",
        "classic_mcts",
        "--games",
        str(args.games),
        "--simulations",
        str(args.simulations),
        "--seed",
        str(args.seed),
        "--workers",
        str(args.workers),
        "--input-encoding",
        "kalah_v3",
        "--max-positions-per-game",
        str(args.max_positions_per_game),
        "--policy-target-mode",
        "sharpened",
        "--value-target-mode",
        "sharpened",
        "--tau",
        "1.1",
    ]

    common_train_args = [
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        "auto",
        "--hidden-sizes",
        "96,3",
        "--model-type",
        "residual_v3",
        "--input-encoding",
        "kalah_v3",
        "--value-loss",
        "huber",
        "--huber-delta",
        "1.0",
        "--value-loss-weight",
        "0.3",
        "--val-split",
        "0.1",
        "--grad-clip",
        "1.0",
        "--policy-target-mode",
        "sharpened",
        "--value-target-mode",
        "sharpened",
        "--save-top-k",
        "3",
    ]

    results: dict[str, dict] = {}

    for lane in lanes:
        lane_label = lane.replace(" ", "_")
        dataset_path = dataset_shard_dir / f"{lane_label}.jsonl"
        checkpoint_path = model_dir / f"{lane_label}_checkpoint.npz"

        overlay_flag = "off"
        blend_alpha = None
        if lane == "overwrite":
            overlay_flag = "overwrite"
        elif lane == "blend" or lane.startswith("blend_"):
            overlay_flag = "blend"
            if "_" in lane:
                blend_alpha = float(lane.split("_", 1)[1])

        dataset_cmd = [
            python,
            str(Path("ml/alphazero_lite/generate_bootstrap_dataset.py")),
            "--out",
            str(dataset_path),
            "--tablebase-value-overlay",
            overlay_flag,
            *common_dataset_args,
        ]
        if blend_alpha is not None:
            dataset_cmd.extend(["--tablebase-blend-alpha", str(blend_alpha)])

        result = run_step(f"generate_dataset_{lane_label}", dataset_cmd, timeout=3600)
        ds_metrics = (
            parse_dataset_metrics(result["stdout"]) if result["status"] == "ok" else {}
        )

        if result["status"] != "ok":
            results[lane_label] = {
                "dataset_generation": result,
                "dataset_metrics": ds_metrics,
            }
            continue

        lane_info: dict = {
            "dataset_generation": result,
            "dataset_metrics": ds_metrics,
        }

        if args.skip_train:
            results[lane_label] = lane_info
            continue

        train_cmd = [
            python,
            str(Path("ml/alphazero_lite/train.py")),
            "--data",
            str(dataset_path),
            "--out",
            str(checkpoint_path),
            "--seed",
            str(args.seed),
            *common_train_args,
        ]

        train_result = run_step(f"train_{lane_label}", train_cmd, timeout=3600)
        train_metrics = (
            parse_training_metrics(train_result["stdout"])
            if train_result["status"] == "ok"
            else {}
        )
        lane_info["train"] = train_result
        lane_info["train_metrics"] = train_metrics

        if train_result["status"] != "ok":
            results[lane_label] = lane_info
            continue

        checkpoint_hashes: dict[str, str] = {}
        for chk_candidate in sorted(
            Path(str(checkpoint_path.parent)).glob(f"{checkpoint_path.stem}*.npz")
        ):
            h = sha256_file(chk_candidate)
            if h:
                checkpoint_hashes[str(chk_candidate)] = h
        lane_info["checkpoint_hashes"] = checkpoint_hashes

        export_dir = model_dir / lane_label
        export_dir.mkdir(parents=True, exist_ok=True)
        export_cmd = [
            python,
            str(Path("ml/alphazero_lite/export_artifact.py")),
            "--checkpoint",
            str(checkpoint_path),
            "--out-dir",
            str(export_dir),
            "--version",
            f"tbv_overlay_{lane_label}",
            "--model-type",
            "residual_v3",
            "--rules-version",
            "kalah_v1",
            "--input-encoding",
            "kalah_v3",
        ]

        export_result = run_step(f"export_{lane_label}", export_cmd)
        lane_info["export"] = export_result

        if export_result["status"] != "ok":
            results[lane_label] = lane_info
            continue

        if not args.skip_gate:
            gate_out = eval_dir / f"gate_{lane_label}.json"
            gate_cmd = [
                python,
                str(Path("script/ai/local_promotion_gate")),
                "--candidate-path",
                str(export_dir),
                "--current-path",
                "storage/ai/alphazero_lite/current",
                "--hard-path",
                "model-artifact/current",
                "--arena-games",
                str(args.arena_games),
                "--mcts-games",
                str(args.mcts_games),
                "--min-arena-score",
                "0.55",
                "--out",
                str(gate_out),
            ]

            gate_result = run_step(f"gate_{lane_label}", gate_cmd, timeout=3600)
            lane_info["gate"] = gate_result
            if gate_out.exists():
                try:
                    gate_data = json.loads(gate_out.read_text(encoding="utf-8"))
                    lane_info["gate_report"] = {
                        "passed": gate_data.get("passed"),
                        "arena_score": gate_data.get("arena_score"),
                        "candidate_mcts_score": gate_data.get("candidate_mcts_score"),
                        "current_mcts_score": gate_data.get("current_mcts_score"),
                        "failure_reasons": [
                            r.get("code") for r in gate_data.get("failure_reasons", [])
                        ],
                    }
                except (json.JSONDecodeError, OSError):
                    pass

        results[lane_label] = lane_info

    comparison = None
    if "baseline" in results and "overwrite" in results:
        baseline_path = dataset_shard_dir / "baseline.jsonl"
        overwrite_path = dataset_shard_dir / "overwrite.jsonl"
        comparison = compare_lane_datasets(baseline_path, overwrite_path)
        if comparison and "error" in comparison:
            print(f"\nLane comparison error: {comparison['error']}")
        else:
            print(f"\nLane comparison: {json.dumps(comparison, indent=2)}")

    enriched_results: dict[str, object] = {**results}
    enriched_results["_comparison"] = comparison
    enriched_results["_config"] = {
        "games": args.games,
        "simulations": args.simulations,
        "seed": args.seed,
        "workers": args.workers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_positions_per_game": args.max_positions_per_game,
        "arena_games": args.arena_games,
        "mcts_games": args.mcts_games,
        "lanes": lanes,
    }

    summary_path = workdir / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(enriched_results, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nExperiment summary written to {summary_path}")

    report_md = generate_production_report(results, workdir, args, comparison)
    report_path = (
        REPO_ROOT
        / "docs"
        / "alphazero-lite-tablebase-value-overlay-production-results.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")
    print(f"Production report written to {report_path}")

    for lane_label, lane_result in results.items():
        print(f"\n--- {lane_label} ---")
        ds = lane_result.get("dataset_metrics", {})
        for k, v in ds.items():
            print(f"  dataset.{k}: {v}")
        tm = lane_result.get("train_metrics", {})
        for k, v in tm.items():
            if k != "top_k_checkpoint_paths":
                print(f"  train.{k}: {v}")
        for step_name in ("dataset_generation", "train", "export", "gate"):
            step_result = lane_result.get(step_name)
            if step_result:
                status = step_result.get("status", "unknown")
                duration = step_result.get("duration_s", "N/A")
                print(f"  {step_name}: {status} ({duration}s)")
        if "gate_report" in lane_result:
            gr = lane_result["gate_report"]
            print(f"  gate_passed: {gr.get('passed')}")
            print(f"  arena_score: {gr.get('arena_score')}")
            print(f"  candidate_mcts_score: {gr.get('candidate_mcts_score')}")
            print(f"  failure_reasons: {gr.get('failure_reasons')}")


if __name__ == "__main__":
    main()
