#!/usr/bin/env python3
"""Run control_ep2 seed-replication sweep from iter0_reference.

Lanes:
  1. control_ep2_eval_only (reference checkpoint, eval only)
  2. replicate_seed_<N> (exact 2-epoch continuation from iter0_reference)

Does not promote, does not overwrite current, and does not add data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = str(REPO_ROOT / ".venv/bin/python")

DEFAULT_BUDGET_PAIRS = "384:256,768:256,768:768,1200:1200,256:768"
LABEL_TO_BUDGET = {
    "standard": "384:256",
    "challenger_768_vs_256": "768:256",
    "equal_768": "768:768",
    "equal_high": "1200:1200",
    "current_high_asymmetry": "256:768",
}


def _python() -> str:
    if Path(VENV_PYTHON).is_file():
        return VENV_PYTHON
    return sys.executable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_param_delta_norm(
    npz_path: Path, reference_npz: Path
) -> tuple[float, float]:
    import numpy as np

    ckpt = np.load(npz_path)
    ref = np.load(reference_npz)

    total_sq = 0.0
    ref_sq = 0.0
    for key in sorted(ref.files):
        if key in ckpt:
            delta = ckpt[key] - ref[key]
            total_sq += float((delta**2).sum())
            ref_sq += float((ref[key] ** 2).sum())

    delta_norm = float(total_sq**0.5)
    ref_norm = float(ref_sq**0.5)
    rel_delta = (delta_norm / ref_norm * 100.0) if ref_norm > 0 else 0.0
    return delta_norm, rel_delta


def run_train(
    *,
    data_files: str,
    replay_weights: str,
    init_checkpoint: str,
    out: str,
    top_k_dir: str,
    lr: float,
    epochs: int,
    seed: int,
) -> dict:
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/train.py"),
        "--data-files",
        data_files,
        "--replay-weights",
        replay_weights,
        "--init-checkpoint",
        init_checkpoint,
        "--model-type",
        "residual_v3",
        "--input-encoding",
        "kalah_v3",
        "--hidden-sizes",
        "96,3",
        "--epochs",
        str(epochs),
        "--batch-size",
        "512",
        "--lr",
        str(lr),
        "--value-loss",
        "huber",
        "--value-loss-weight",
        "0.3",
        "--grad-clip",
        "1.0",
        "--save-top-k",
        "0",
        "--top-k-dir",
        top_k_dir,
        "--out",
        out,
        "--policy-target-mode",
        "sharpened",
        "--value-target-mode",
        "sharpened",
        "--lr-scheduler",
        "none",
        "--seed",
        str(seed),
        "--save-epochs",
        str(epochs),
    ]

    print(f"[train] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    elapsed = time.time() - t0
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0:
        print(f"[train] FAILED after {elapsed:.0f}s", flush=True)
        print(f"[train] stdout: {stdout[-2000:]}", flush=True)
        print(f"[train] stderr: {stderr[-2000:]}", flush=True)
        raise RuntimeError(f"train.py failed with return code {result.returncode}")

    metrics: dict[str, object] = {"training_elapsed_s": elapsed}
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("policy_loss="):
            metrics["policy_loss"] = float(line.split("=")[1])
        elif line.startswith("value_loss="):
            metrics["value_loss"] = float(line.split("=")[1])
        elif line.startswith("best_val_loss="):
            metrics["best_val_loss"] = float(line.split("=")[1])
        elif line.startswith("saved_epoch_checkpoint_epoch="):
            parts = line.split()
            epoch_part = parts[0].split("=")[1]
            path_part = parts[1].split("=")[1]
            metrics[f"epoch_{epoch_part}_path"] = path_part
    print(
        f"[train] done {elapsed:.0f}s policy_loss={metrics.get('policy_loss', '?'):.6f}",
        flush=True,
    )
    return metrics


def export_checkpoint(
    checkpoint_path: str,
    out_dir: str,
    version: str,
    policy_loss: float = 0.0,
    value_loss: float = 0.0,
) -> None:
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/export_artifact.py"),
        "--checkpoint",
        checkpoint_path,
        "--out-dir",
        out_dir,
        "--version",
        version,
        "--model-type",
        "residual_v3",
        "--input-encoding",
        "kalah_v3",
        "--rules-version",
        "kalah_v3",
        "--policy-loss",
        str(policy_loss),
        "--value-loss",
        str(value_loss),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"export_artifact failed: {result.stderr}")
    print(f"[export] {out_dir}", flush=True)


def run_opening_suite_benchmark(
    *,
    workdir: str,
    suite: str,
    current: str,
    candidates: str,
    budget_pairs: str,
    games_per_opening: int,
    seed: int,
    workers: int,
    timeout: int,
) -> dict:
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/run_opening_suite_seat_benchmark.py"),
        "--workdir",
        workdir,
        "--suite",
        suite,
        "--current",
        current,
        "--candidates",
        candidates,
        "--budget-pairs",
        budget_pairs,
        "--games-per-opening",
        str(games_per_opening),
        "--seed",
        str(seed),
        "--root-policy-mode",
        "deterministic",
        "--workers",
        str(workers),
        "--timeout",
        str(timeout),
    ]
    print(f"[eval] {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[eval] FAILED: {result.stderr[-2000:]}", flush=True)
        raise RuntimeError(f"benchmark failed: {result.stderr}")
    report_path = Path(workdir) / "temperature_benchmark_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    return {}


def run_default_gate(
    *,
    candidate_path: str,
    current_path: str,
    out: str,
    games: int,
    seed: int,
    workers: int,
    budget_pairs: str,
) -> dict:
    cmd = [
        _python(),
        str(REPO_ROOT / "script/ai/seat_aware_promotion_gate"),
        "--candidate-path",
        candidate_path,
        "--current-path",
        current_path,
        "--out",
        out,
        "--games",
        str(games),
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--budget-pairs",
        budget_pairs,
    ]
    print(f"[gate] {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    if result.returncode != 0:
        print(f"[gate] FAILED: {result.stderr[-2000:]}", flush=True)
        raise RuntimeError(f"gate failed: {result.stderr}")
    out_path = Path(out)
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    return {}


def find_candidate_report(report: dict, candidate: str) -> dict | None:
    for temperature_report in report.get("temperature_reports", []):
        for seed_report in temperature_report.get("seed_reports", []):
            for candidate_report in seed_report.get("candidate_reports", []):
                if candidate_report.get("candidate") == candidate:
                    return candidate_report
    return None


def budget_results_by_pair(candidate_report: dict) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for label, pair in LABEL_TO_BUDGET.items():
        budget_result = candidate_report.get("budget_results", {}).get(label)
        if budget_result:
            result[pair] = budget_result
    return result


def candidate_standard_ds(report: dict, candidate: str) -> float:
    candidate_report = find_candidate_report(report, candidate)
    if candidate_report is None:
        return float("-inf")
    standard = candidate_report.get("budget_results", {}).get("standard", {})
    ds = standard.get("ds")
    return float(ds) if ds is not None else float("-inf")


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="/tmp/azlite_control_ep2_seed_replication")
    parser.add_argument(
        "--init-checkpoint",
        default="/tmp/azlite_iterative_random_replay/iter0_candidate_checkpoint.npz",
    )
    parser.add_argument(
        "--reference-checkpoint",
        default="/tmp/azlite_control_continuation_sweep/lr3e5/checkpoint_epoch2.npz",
    )
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--data-files",
        default="/tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl,"
        "/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl",
    )
    parser.add_argument("--replay-weights", default="4,1")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--model-type", default="residual_v3")
    parser.add_argument("--input-encoding", default="kalah_v3")
    parser.add_argument("--hidden-sizes", default="96,3")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--value-loss", default="huber")
    parser.add_argument("--value-loss-weight", type=float, default=0.3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--medium-suite", default="/tmp/azlite_opening_suite/medium_eval.jsonl"
    )
    parser.add_argument(
        "--large-suite", default="/tmp/azlite_opening_suite/large_eval.jsonl"
    )
    parser.add_argument("--budget-pairs", default=DEFAULT_BUDGET_PAIRS)
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-eval-medium", action="store_true")
    parser.add_argument("--skip-eval-large", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--large-eval-top-n", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if args.lr != 3e-5:
        print("[error] guardrail violation: lr must remain 3e-5")
        return 1
    if args.epochs != 2:
        print("[error] guardrail violation: epochs must remain 2")
        return 1
    if args.model_type != "residual_v3":
        print("[error] guardrail violation: model_type must remain residual_v3")
        return 1
    if args.input_encoding != "kalah_v3":
        print("[error] guardrail violation: input_encoding must remain kalah_v3")
        return 1
    if args.hidden_sizes != "96,3":
        print("[error] guardrail violation: hidden_sizes must remain 96,3")
        return 1
    if args.batch_size != 512:
        print("[error] guardrail violation: batch_size must remain 512")
        return 1
    if args.value_loss != "huber":
        print("[error] guardrail violation: value_loss must remain huber")
        return 1
    if args.value_loss_weight != 0.3:
        print("[error] guardrail violation: value_loss_weight must remain 0.3")
        return 1
    if args.grad_clip != 1.0:
        print("[error] guardrail violation: grad_clip must remain 1.0")
        return 1

    init_checkpoint = Path(args.init_checkpoint)
    reference_checkpoint = Path(args.reference_checkpoint)
    current_artifact = Path(args.current)
    if not init_checkpoint.exists() or not reference_checkpoint.exists():
        print("[error] missing required checkpoint path")
        return 1

    seeds = parse_int_list(args.seeds)

    lanes: list[dict[str, object]] = []
    control_dir = workdir / "control_ep2_eval_only"
    control_dir.mkdir(parents=True, exist_ok=True)
    lanes.append(
        {
            "name": "control_ep2",
            "seed": None,
            "checkpoint_path": str(reference_checkpoint),
            "artifact_dir": str(control_dir / "artifact_control_ep2"),
            "source": "reference_eval_only",
        }
    )

    if not args.skip_training:
        for seed in seeds:
            lane_dir = workdir / f"replicate_seed_{seed}"
            lane_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_out = lane_dir / "checkpoint.npz"
            train_metrics = run_train(
                data_files=args.data_files,
                replay_weights=args.replay_weights,
                init_checkpoint=str(init_checkpoint),
                out=str(checkpoint_out),
                top_k_dir=str(lane_dir),
                lr=args.lr,
                epochs=args.epochs,
                seed=seed,
            )
            epoch_checkpoint = lane_dir / f"checkpoint_epoch{args.epochs}.npz"
            lanes.append(
                {
                    "name": f"replicate_seed_{seed}",
                    "seed": seed,
                    "checkpoint_path": str(epoch_checkpoint),
                    "artifact_dir": str(lane_dir / f"artifact_replicate_seed_{seed}"),
                    "source": "trained_replication",
                    "train_metrics": train_metrics,
                }
            )
    else:
        for seed in seeds:
            lane_dir = workdir / f"replicate_seed_{seed}"
            lanes.append(
                {
                    "name": f"replicate_seed_{seed}",
                    "seed": seed,
                    "checkpoint_path": str(
                        lane_dir / f"checkpoint_epoch{args.epochs}.npz"
                    ),
                    "artifact_dir": str(lane_dir / f"artifact_replicate_seed_{seed}"),
                    "source": "trained_replication",
                }
            )

    for lane in lanes:
        checkpoint_path = Path(str(lane["checkpoint_path"]))
        artifact_dir = Path(str(lane["artifact_dir"]))
        if not checkpoint_path.exists():
            raise RuntimeError(
                f"missing checkpoint for {lane['name']}: {checkpoint_path}"
            )
        if not (artifact_dir / "weights.json").exists():
            train_metrics = lane.get("train_metrics")
            if not isinstance(train_metrics, dict):
                train_metrics = {}
            export_checkpoint(
                checkpoint_path=str(checkpoint_path),
                out_dir=str(artifact_dir),
                version=str(lane["name"]),
                policy_loss=float(train_metrics.get("policy_loss", 0.0)),
                value_loss=float(train_metrics.get("value_loss", 0.0)),
            )

    candidates = ",".join(str(lane["artifact_dir"]) for lane in lanes)

    medium_report: dict = {}
    if not args.skip_eval_medium:
        medium_workdir = workdir / "eval_medium"
        medium_report = run_opening_suite_benchmark(
            workdir=str(medium_workdir),
            suite=args.medium_suite,
            current=str(current_artifact),
            candidates=candidates,
            budget_pairs=args.budget_pairs,
            games_per_opening=args.games_per_opening,
            seed=42,
            workers=args.workers,
            timeout=args.timeout,
        )

    large_report: dict = {}
    large_candidate_names: list[str] = []
    if not args.skip_eval_large:
        if not medium_report:
            medium_report_path = (
                workdir / "eval_medium" / "temperature_benchmark_report.json"
            )
            medium_report = json.loads(medium_report_path.read_text(encoding="utf-8"))

        ranked_names = sorted(
            [str(lane["name"]) for lane in lanes],
            key=lambda name: candidate_standard_ds(medium_report, name),
            reverse=True,
        )
        seed_ranked = [
            name for name in ranked_names if name.startswith("replicate_seed_")
        ]
        large_candidate_names = ["control_ep2"]
        for name in seed_ranked[: args.large_eval_top_n]:
            if name not in large_candidate_names:
                large_candidate_names.append(name)
        large_candidates = ",".join(
            str(lane["artifact_dir"])
            for lane in lanes
            if str(lane["name"]) in large_candidate_names
        )
        large_workdir = workdir / "eval_large"
        large_report = run_opening_suite_benchmark(
            workdir=str(large_workdir),
            suite=args.large_suite,
            current=str(current_artifact),
            candidates=large_candidates,
            budget_pairs=args.budget_pairs,
            games_per_opening=args.games_per_opening,
            seed=42,
            workers=args.workers,
            timeout=args.timeout,
        )

    gate_reports: dict[str, dict] = {}
    if not args.skip_gate:
        gate_dir = workdir / "eval_gate"
        gate_dir.mkdir(parents=True, exist_ok=True)

        ranked_names = large_candidate_names
        if not large_report:
            large_report_path = (
                workdir / "eval_large" / "temperature_benchmark_report.json"
            )
            if large_report_path.exists():
                large_report = json.loads(large_report_path.read_text(encoding="utf-8"))
        if large_report:
            ranked_names = sorted(
                [str(lane["name"]) for lane in lanes],
                key=lambda name: candidate_standard_ds(large_report, name),
                reverse=True,
            )
        if not ranked_names:
            if not medium_report:
                medium_report_path = (
                    workdir / "eval_medium" / "temperature_benchmark_report.json"
                )
                medium_report = json.loads(
                    medium_report_path.read_text(encoding="utf-8")
                )
            ranked_names = sorted(
                [str(lane["name"]) for lane in lanes],
                key=lambda name: candidate_standard_ds(medium_report, name),
                reverse=True,
            )
        best_replicate_name = next(
            (name for name in ranked_names if name.startswith("replicate_seed_")),
            None,
        )
        gate_targets = ["control_ep2"]
        if best_replicate_name is not None:
            gate_targets.append(best_replicate_name)

        for lane in lanes:
            lane_name = str(lane["name"])
            if lane_name not in gate_targets:
                continue
            gate_out = gate_dir / f"{lane_name}_default_gate.json"
            gate_reports[lane_name] = run_default_gate(
                candidate_path=str(lane["artifact_dir"]),
                current_path=str(current_artifact),
                out=str(gate_out),
                games=60,
                seed=42,
                workers=args.workers,
                budget_pairs=args.budget_pairs,
            )

    final_candidates: list[dict[str, object]] = []
    for lane in lanes:
        lane_name = str(lane["name"])
        checkpoint_path = Path(str(lane["checkpoint_path"]))
        artifact_dir = Path(str(lane["artifact_dir"]))
        checkpoint_sha = sha256_file(checkpoint_path)
        artifact_sha = sha256_file(artifact_dir / "weights.json")
        delta_vs_init, rel_vs_init = compute_param_delta_norm(
            checkpoint_path, init_checkpoint
        )
        delta_vs_ref, rel_vs_ref = compute_param_delta_norm(
            checkpoint_path, reference_checkpoint
        )

        row: dict[str, object] = {
            "candidate": lane_name,
            "seed": lane.get("seed"),
            "checkpoint_path": str(checkpoint_path),
            "artifact_dir": str(artifact_dir),
            "checkpoint_sha256": checkpoint_sha,
            "artifact_sha256": artifact_sha,
            "parameter_delta_vs_iter0_reference": delta_vs_init,
            "parameter_delta_vs_iter0_reference_pct": rel_vs_init,
            "parameter_delta_vs_control_ep2": delta_vs_ref,
            "parameter_delta_vs_control_ep2_pct": rel_vs_ref,
        }

        train_metrics = lane.get("train_metrics")
        if isinstance(train_metrics, dict):
            row["policy_loss"] = train_metrics.get("policy_loss")
            row["value_loss"] = train_metrics.get("value_loss")
            row["best_val_loss"] = train_metrics.get("best_val_loss")

        if medium_report:
            medium_candidate_report = find_candidate_report(medium_report, lane_name)
            if medium_candidate_report is not None:
                row["medium_budget_results"] = budget_results_by_pair(
                    medium_candidate_report
                )

        if large_report:
            large_candidate_report = find_candidate_report(large_report, lane_name)
            if large_candidate_report is not None:
                row["large_budget_results"] = budget_results_by_pair(
                    large_candidate_report
                )

        if lane_name in gate_reports:
            row["default_gate"] = gate_reports[lane_name]

        final_candidates.append(row)

    summary = {
        "schema": "azlite_control_ep2_seed_replication_v1",
        "init_checkpoint": str(init_checkpoint),
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "reference_checkpoint": str(reference_checkpoint),
        "reference_checkpoint_sha256": sha256_file(reference_checkpoint),
        "current_artifact": str(current_artifact),
        "current_artifact_sha256": sha256_file(current_artifact / "weights.json"),
        "dataset_files": args.data_files.split(","),
        "replay_weights": args.replay_weights,
        "lr": args.lr,
        "epochs": args.epochs,
        "seeds": seeds,
        "budget_pairs": args.budget_pairs.split(","),
        "large_eval_candidates": large_candidate_names,
        "candidates": final_candidates,
    }
    report_out = workdir / "control_ep2_seed_replication_report.json"
    report_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[report] {report_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
