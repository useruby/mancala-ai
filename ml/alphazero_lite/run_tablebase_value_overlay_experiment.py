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
import json
import os
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
        if result["status"] != "ok":
            results[lane_label] = {"dataset_generation": result}
            continue

        if args.skip_train:
            results[lane_label] = {"dataset_generation": result}
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
        if train_result["status"] != "ok":
            results[lane_label] = {"dataset_generation": result, "train": train_result}
            continue

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
        if export_result["status"] != "ok":
            results[lane_label] = {
                "dataset_generation": result,
                "train": train_result,
                "export": export_result,
            }
            continue

        lane_result = {
            "dataset_generation": result,
            "train": train_result,
            "export": export_result,
        }

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
            lane_result["gate"] = gate_result
            if gate_out.exists():
                try:
                    gate_data = json.loads(gate_out.read_text(encoding="utf-8"))
                    lane_result["gate_report"] = {
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

        results[lane_label] = lane_result

    summary_path = workdir / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nExperiment summary written to {summary_path}")

    for lane_label, lane_result in results.items():
        print(f"\n--- {lane_label} ---")
        for step_name, step_result in lane_result.items():
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
