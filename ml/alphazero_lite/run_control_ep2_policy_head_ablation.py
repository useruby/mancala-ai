#!/usr/bin/env python3
"""Run frozen-trunk control_ep2 policy-head ablation.

Lanes:
1. canonical_ref (no training)
2. orig_policy_head_e1/e2
3. added_policy_head_e1/e2

Does not promote and does not overwrite model-artifact/current.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = REPO_ROOT / ".venv/bin/python"
DEFAULT_BUDGET_PAIRS = "384:256,768:256,768:768,1200:1200,256:768"
LABEL_TO_BUDGET = {
    "standard": "384:256",
    "challenger_768_vs_256": "768:256",
    "equal_768": "768:768",
    "equal_high": "1200:1200",
    "current_high_asymmetry": "256:768",
}
EXPECTED_CONTROL_CHECKPOINT_SHA256 = (
    "619376dba87b968cc85bfc401cc230aca64e444dc3570216b0630c2c29a388f9"
)
EXPECTED_CONTROL_ARTIFACT_SHA256 = (
    "34b3697f95c3c63e092838d10668ba1361dab2a68a0bfe12305a294c45f765ad"
)
FULL_ADDED_DATA_BASELINE = {
    "candidate": "selfplay_e2",
    "large_384_256_ds": -0.1901,
    "classification": "regression_masked_by_seat",
}


def _python() -> str:
    if VENV_PYTHON.is_file():
        return str(VENV_PYTHON)
    return sys.executable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def require_existing_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")


def compute_param_delta_norm(
    npz_path: Path, reference_npz: Path
) -> tuple[float, float]:
    import numpy as np

    checkpoint = np.load(npz_path)
    reference = np.load(reference_npz)

    total_sq = 0.0
    ref_sq = 0.0
    for key in sorted(reference.files):
        if key not in checkpoint:
            continue
        delta = checkpoint[key] - reference[key]
        total_sq += float((delta**2).sum())
        ref_sq += float((reference[key] ** 2).sum())

    delta_norm = float(total_sq**0.5)
    ref_norm = float(ref_sq**0.5)
    relative_delta_pct = (delta_norm / ref_norm * 100.0) if ref_norm > 0 else 0.0
    return delta_norm, relative_delta_pct


def run_train(
    *,
    data_files: str,
    replay_weights: str,
    init_checkpoint: str,
    out: str,
    top_k_dir: str,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
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
        "1e-5",
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
        "--trainable-scope",
        "policy_head",
        "--save-epochs",
        str(epochs),
    ]

    print(f"[train] {' '.join(cmd)}", flush=True)
    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    elapsed = time.time() - start
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0:
        print(f"[train] FAILED after {elapsed:.0f}s", flush=True)
        print(f"[train] stdout: {stdout[-2000:]}", flush=True)
        print(f"[train] stderr: {stderr[-2000:]}", flush=True)
        raise RuntimeError(f"train.py failed with return code {result.returncode}")

    metrics: dict[str, Any] = {
        "training_elapsed_s": elapsed,
        "trainable_scope": "policy_head",
    }
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("policy_loss="):
            metrics["policy_loss"] = float(line.split("=", 1)[1])
        elif line.startswith("value_loss="):
            metrics["value_loss"] = float(line.split("=", 1)[1])
        elif line.startswith("best_val_loss="):
            metrics["best_val_loss"] = float(line.split("=", 1)[1])
        elif line.startswith("saved_epoch_checkpoint_epoch="):
            parts = line.split()
            epoch_number = parts[0].split("=", 1)[1]
            path_value = parts[1].split("=", 1)[1]
            metrics[f"epoch_{epoch_number}_path"] = path_value
    for line in stderr.splitlines():
        line = line.strip()
        for key in (
            "trainable_params",
            "frozen_params",
            "total_params",
        ):
            token = f"{key}="
            if token in line:
                for part in line.split():
                    if part.startswith(token):
                        metrics[key] = int(part.split("=", 1)[1])

    print(
        f"[train] done {elapsed:.0f}s policy_loss={float(metrics.get('policy_loss', 0.0)):.6f}",
        flush=True,
    )
    return metrics


def export_checkpoint(
    *,
    checkpoint_path: str,
    out_dir: str,
    version: str,
    policy_loss: float,
    value_loss: float,
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
) -> dict[str, Any]:
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
        timeout=timeout + 300,
    )
    if result.returncode != 0:
        print(f"[eval] FAILED: {result.stderr[-2000:]}", flush=True)
        raise RuntimeError(f"benchmark failed: {result.stderr}")

    report_path = Path(workdir) / "temperature_benchmark_report.json"
    return json.loads(report_path.read_text(encoding="utf-8"))


def run_default_gate(
    *,
    candidate_path: str,
    current_path: str,
    out: str,
    seed: int,
    workers: int,
) -> dict[str, Any]:
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
        "60",
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--budget-pairs",
        DEFAULT_BUDGET_PAIRS,
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
    return json.loads(out_path.read_text(encoding="utf-8"))


def find_candidate_report(
    report: dict[str, Any], candidate: str
) -> dict[str, Any] | None:
    for temperature_report in report.get("temperature_reports", []):
        for seed_report in temperature_report.get("seed_reports", []):
            for candidate_report in seed_report.get("candidate_reports", []):
                if candidate_report.get("candidate") == candidate:
                    return candidate_report
    return None


def candidate_standard_ds(report: dict[str, Any], candidate: str) -> float:
    candidate_report = find_candidate_report(report, candidate)
    if candidate_report is None:
        return float("-inf")
    standard = candidate_report.get("budget_results", {}).get("standard", {})
    ds = standard.get("ds")
    return float(ds) if ds is not None else float("-inf")


def budget_results_by_pair(
    candidate_report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for label, pair in LABEL_TO_BUDGET.items():
        budget_result = candidate_report.get("budget_results", {}).get(label)
        if not budget_result:
            continue
        results[pair] = {
            "ds": budget_result.get("ds"),
            "p0_score": budget_result.get("p0_score"),
            "p1_score": budget_result.get("p1_score"),
            "disadvantaged_seat_score": budget_result.get("disadvantaged_seat_score"),
            "margin_mean": budget_result.get("margin_mean"),
            "margin_median": budget_result.get("margin_median"),
            "duplicate_trajectory_count": budget_result.get(
                "duplicate_trajectory_count"
            ),
            "total_games": budget_result.get("total_games"),
        }
    return results


def select_large_eval_candidates(
    lanes: list[dict[str, Any]],
    medium_report: dict[str, Any],
    top_n: int,
) -> list[str]:
    ranked = sorted(
        lanes,
        key=lambda lane: candidate_standard_ds(
            medium_report, str(lane["report_candidate_name"])
        ),
        reverse=True,
    )
    selected = ["canonical_ref"]
    for lane in ranked:
        name = str(lane["name"])
        if name == "canonical_ref":
            continue
        selected.append(name)
        if len(selected) >= top_n + 1:
            break
    return selected


def build_input_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        summary["rows"] = count_jsonl_rows(path)
    return summary


def verify_expected_hash(
    path: Path, expected_hash: str | None, label: str
) -> dict[str, Any]:
    actual_hash = sha256_file(path)
    result = {
        "path": str(path),
        "actual_sha256": actual_hash,
        "expected_sha256": expected_hash,
        "matches_expected": expected_hash is None or actual_hash == expected_hash,
    }
    if expected_hash is not None and actual_hash != expected_hash:
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir",
        default="/tmp/azlite_control_ep2_policy_head_ablation",
    )
    parser.add_argument(
        "--control-checkpoint",
        default="/tmp/azlite_control_continuation_sweep/lr3e5/checkpoint_epoch2.npz",
    )
    parser.add_argument(
        "--control-artifact",
        default="/tmp/azlite_control_ep2_seed_harvest/control_ep2_eval_only/artifact_control_ep2",
    )
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--generic-bootstrap",
        default="/tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl",
    )
    parser.add_argument(
        "--random-teacher",
        default="/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl",
    )
    parser.add_argument(
        "--added-classic-mcts",
        default="/tmp/azlite_control_ep2_multi_iter/control_ep2_classic_mcts_selfplay.jsonl",
    )
    parser.add_argument(
        "--medium-suite",
        default="/tmp/azlite_opening_suite/medium_eval.jsonl",
    )
    parser.add_argument(
        "--large-suite",
        default="/tmp/azlite_opening_suite/large_eval.jsonl",
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--budget-pairs", default=DEFAULT_BUDGET_PAIRS)
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-eval-medium", action="store_true")
    parser.add_argument("--skip-eval-large", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument(
        "--expected-control-checkpoint-sha256",
        default=EXPECTED_CONTROL_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--expected-control-artifact-sha256",
        default=EXPECTED_CONTROL_ARTIFACT_SHA256,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    control_checkpoint = Path(args.control_checkpoint)
    control_artifact = Path(args.control_artifact)
    current_artifact = Path(args.current)
    generic_bootstrap = Path(args.generic_bootstrap)
    random_teacher = Path(args.random_teacher)
    added_classic_mcts = Path(args.added_classic_mcts)
    medium_suite = Path(args.medium_suite)
    large_suite = Path(args.large_suite)

    require_existing_file(control_checkpoint, "control checkpoint")
    require_existing_file(control_artifact / "weights.json", "control artifact weights")
    require_existing_file(current_artifact / "weights.json", "current artifact weights")
    require_existing_file(generic_bootstrap, "generic bootstrap replay")
    require_existing_file(random_teacher, "random teacher replay")
    require_existing_file(added_classic_mcts, "added classic MCTS replay")
    require_existing_file(medium_suite, "medium suite")
    require_existing_file(large_suite, "large suite")

    input_summary = {
        "control_checkpoint": verify_expected_hash(
            control_checkpoint,
            args.expected_control_checkpoint_sha256,
            "control checkpoint",
        ),
        "control_artifact_weights": verify_expected_hash(
            control_artifact / "weights.json",
            args.expected_control_artifact_sha256,
            "control artifact",
        ),
        "current_artifact_weights": build_input_summary(
            current_artifact / "weights.json"
        ),
        "generic_bootstrap": build_input_summary(generic_bootstrap),
        "random_teacher": build_input_summary(random_teacher),
        "added_classic_mcts": build_input_summary(added_classic_mcts),
        "medium_suite": build_input_summary(medium_suite),
        "large_suite": build_input_summary(large_suite),
    }

    lane_specs = [
        {
            "name": "orig_policy_head_e1",
            "epochs": 1,
            "data_files": f"{generic_bootstrap},{random_teacher}",
            "replay_weights": "4,1",
            "source": "orig_policy_head",
        },
        {
            "name": "orig_policy_head_e2",
            "epochs": 2,
            "data_files": f"{generic_bootstrap},{random_teacher}",
            "replay_weights": "4,1",
            "source": "orig_policy_head",
        },
        {
            "name": "added_policy_head_e1",
            "epochs": 1,
            "data_files": f"{generic_bootstrap},{random_teacher},{added_classic_mcts}",
            "replay_weights": "4,1,1",
            "source": "added_policy_head",
        },
        {
            "name": "added_policy_head_e2",
            "epochs": 2,
            "data_files": f"{generic_bootstrap},{random_teacher},{added_classic_mcts}",
            "replay_weights": "4,1,1",
            "source": "added_policy_head",
        },
    ]

    lanes: list[dict[str, Any]] = [
        {
            "name": "canonical_ref",
            "report_candidate_name": "control_ep2",
            "epochs": 0,
            "checkpoint_path": str(control_checkpoint),
            "artifact_dir": str(control_artifact),
            "source": "canonical_ref",
            "trainable_scope": "none",
        }
    ]

    for lane_spec in lane_specs:
        lane_dir = workdir / lane_spec["name"]
        lane_dir.mkdir(parents=True, exist_ok=True)
        epoch_checkpoint_path = lane_dir / f"checkpoint_epoch{lane_spec['epochs']}.npz"
        export_dir = lane_dir / f"artifact_{lane_spec['name']}"
        train_metrics: dict[str, Any] | None = None

        if not args.skip_training:
            checkpoint_out = lane_dir / "checkpoint.npz"
            train_metrics = run_train(
                data_files=str(lane_spec["data_files"]),
                replay_weights=str(lane_spec["replay_weights"]),
                init_checkpoint=str(control_checkpoint),
                out=str(checkpoint_out),
                top_k_dir=str(lane_dir),
                epochs=int(lane_spec["epochs"]),
                seed=args.seed,
            )
            export_checkpoint(
                checkpoint_path=str(epoch_checkpoint_path),
                out_dir=str(export_dir),
                version=str(lane_spec["name"]),
                policy_loss=float(train_metrics.get("policy_loss", 0.0)),
                value_loss=float(train_metrics.get("value_loss", 0.0)),
            )
        else:
            require_existing_file(
                epoch_checkpoint_path, f"checkpoint for {lane_spec['name']}"
            )
            require_existing_file(
                export_dir / "weights.json", f"artifact for {lane_spec['name']}"
            )

        lanes.append(
            {
                "name": lane_spec["name"],
                "report_candidate_name": export_dir.name,
                "epochs": lane_spec["epochs"],
                "checkpoint_path": str(epoch_checkpoint_path),
                "artifact_dir": str(export_dir),
                "source": lane_spec["source"],
                "replay_weights": lane_spec["replay_weights"],
                "data_files": str(lane_spec["data_files"]).split(","),
                "trainable_scope": "policy_head",
                "train_metrics": train_metrics,
            }
        )

    candidate_paths = ",".join(str(lane["artifact_dir"]) for lane in lanes)
    medium_report_path = workdir / "eval_medium" / "temperature_benchmark_report.json"
    if args.skip_eval_medium:
        medium_report = json.loads(medium_report_path.read_text(encoding="utf-8"))
    else:
        medium_report = run_opening_suite_benchmark(
            workdir=str(workdir / "eval_medium"),
            suite=str(medium_suite),
            current=str(current_artifact),
            candidates=candidate_paths,
            budget_pairs=args.budget_pairs,
            games_per_opening=args.games_per_opening,
            seed=args.seed,
            workers=args.workers,
            timeout=args.timeout,
        )

    large_candidate_names = select_large_eval_candidates(lanes, medium_report, top_n=3)
    large_candidate_paths = ",".join(
        str(lane["artifact_dir"])
        for lane in lanes
        if str(lane["name"]) in large_candidate_names
    )
    large_report_path = workdir / "eval_large" / "temperature_benchmark_report.json"
    if args.skip_eval_large:
        large_report = json.loads(large_report_path.read_text(encoding="utf-8"))
    else:
        large_report = run_opening_suite_benchmark(
            workdir=str(workdir / "eval_large"),
            suite=str(large_suite),
            current=str(current_artifact),
            candidates=large_candidate_paths,
            budget_pairs=args.budget_pairs,
            games_per_opening=args.games_per_opening,
            seed=args.seed,
            workers=args.workers,
            timeout=args.timeout,
        )

    canonical_large_384_256 = candidate_standard_ds(large_report, "control_ep2")
    gate_target_names = ["canonical_ref"]
    for name in large_candidate_names:
        if name == "canonical_ref":
            continue
        lane = next(
            candidate_lane
            for candidate_lane in lanes
            if str(candidate_lane["name"]) == name
        )
        if (
            candidate_standard_ds(large_report, str(lane["report_candidate_name"]))
            >= canonical_large_384_256
        ):
            gate_target_names.append(name)

    gate_reports: dict[str, dict[str, Any]] = {}
    gate_dir = workdir / "eval_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_gate:
        for lane_name in gate_target_names:
            gate_path = gate_dir / f"{lane_name}_default_gate.json"
            gate_reports[lane_name] = json.loads(gate_path.read_text(encoding="utf-8"))
    else:
        for lane in lanes:
            lane_name = str(lane["name"])
            if lane_name not in gate_target_names:
                continue
            gate_reports[lane_name] = run_default_gate(
                candidate_path=str(lane["artifact_dir"]),
                current_path=str(current_artifact),
                out=str(gate_dir / f"{lane_name}_default_gate.json"),
                seed=args.seed,
                workers=args.workers,
            )

    summary_candidates: list[dict[str, Any]] = []
    for lane in lanes:
        lane_name = str(lane["name"])
        checkpoint_path = Path(str(lane["checkpoint_path"]))
        artifact_dir = Path(str(lane["artifact_dir"]))
        checkpoint_sha = sha256_file(checkpoint_path)
        artifact_sha = sha256_file(artifact_dir / "weights.json")
        delta_norm, relative_delta_pct = compute_param_delta_norm(
            checkpoint_path, control_checkpoint
        )

        row: dict[str, Any] = {
            "candidate": lane_name,
            "report_candidate_name": lane.get("report_candidate_name"),
            "source": lane.get("source"),
            "epochs": lane.get("epochs"),
            "trainable_scope": lane.get("trainable_scope"),
            "checkpoint_path": str(checkpoint_path),
            "artifact_dir": str(artifact_dir),
            "checkpoint_sha256": checkpoint_sha,
            "artifact_weights_sha256": artifact_sha,
            "delta_norm_vs_canonical_control_ep2": delta_norm,
            "relative_delta_pct_vs_canonical_control_ep2": relative_delta_pct,
        }

        train_metrics = lane.get("train_metrics")
        if isinstance(train_metrics, dict):
            row["policy_loss"] = train_metrics.get("policy_loss")
            row["value_loss"] = train_metrics.get("value_loss")
            row["validation_loss"] = train_metrics.get("best_val_loss")
            row["training_elapsed_s"] = train_metrics.get("training_elapsed_s")
            row["trainable_params"] = train_metrics.get("trainable_params")
            row["frozen_params"] = train_metrics.get("frozen_params")
            row["total_params"] = train_metrics.get("total_params")

        medium_candidate_report = find_candidate_report(
            medium_report, str(lane["report_candidate_name"])
        )
        if medium_candidate_report is not None:
            row["medium_budget_results"] = budget_results_by_pair(
                medium_candidate_report
            )

        large_candidate_report = find_candidate_report(
            large_report, str(lane["report_candidate_name"])
        )
        if large_candidate_report is not None:
            row["large_budget_results"] = budget_results_by_pair(large_candidate_report)

        gate_report = gate_reports.get(lane_name)
        if gate_report is not None:
            row["default_gate"] = {
                "classification": gate_report.get("classification"),
                "standard_alternating_score": gate_report.get(
                    "standard_alternating_score"
                ),
                "budget_results": {
                    LABEL_TO_BUDGET.get(label, label): {
                        "disadvantaged_seat_score": budget_result.get(
                            "disadvantaged_seat_score"
                        )
                    }
                    for label, budget_result in gate_report.get(
                        "budget_results", {}
                    ).items()
                },
            }
            row["high_search_breakthrough_preserved"] = (
                gate_report.get("classification") == "high_search_breakthrough"
            )
        else:
            row["high_search_breakthrough_preserved"] = None

        summary_candidates.append(row)

    added_candidates = [
        row
        for row in summary_candidates
        if str(row.get("candidate", "")).startswith("added_policy_head_")
    ]
    best_added_candidate = max(
        added_candidates,
        key=lambda row: float(
            row.get("large_budget_results", {})
            .get("384:256", {})
            .get("ds", float("-inf"))
        ),
    )
    best_added_large_384_256 = float(
        best_added_candidate.get("large_budget_results", {})
        .get("384:256", {})
        .get("ds", float("-inf"))
    )
    best_added_signature_preserved = best_added_candidate.get(
        "high_search_breakthrough_preserved"
    )
    delta_vs_canonical = best_added_large_384_256 - canonical_large_384_256

    if (
        best_added_large_384_256 >= canonical_large_384_256
        and best_added_signature_preserved is True
    ):
        added_data_decision = "keep_for_followup"
        experiment_classification = "trunk_value_drift_confirmed"
    elif delta_vs_canonical < -0.01 or best_added_signature_preserved is False:
        added_data_decision = "reject_added_classic_mcts_data"
        experiment_classification = "added_data_not_useful_confirmed"
    else:
        added_data_decision = "borderline_no_followup"
        experiment_classification = "inconclusive"

    summary = {
        "schema": "azlite_control_ep2_policy_head_ablation_v1",
        "workdir": str(workdir),
        "seed": args.seed,
        "workers": args.workers,
        "games_per_opening": args.games_per_opening,
        "budget_pairs": args.budget_pairs.split(","),
        "inputs": input_summary,
        "guardrails": {
            "model_type": "residual_v3",
            "input_encoding": "kalah_v3",
            "hidden_sizes": "96,3",
            "batch_size": 512,
            "seed": args.seed,
            "lr": 1e-5,
            "lr_scheduler": "none",
            "value_loss": "huber",
            "value_loss_weight": 0.3,
            "grad_clip": 1.0,
            "policy_target_mode": "sharpened",
            "value_target_mode": "sharpened",
            "trainable_scope": "policy_head",
            "promotion": "disabled",
            "overwrite_current": False,
        },
        "large_eval_candidates": large_candidate_names,
        "gate_targets": gate_target_names,
        "pr115_full_added_data_baseline": FULL_ADDED_DATA_BASELINE,
        "canonical_large_384_256_ds": canonical_large_384_256,
        "best_added_policy_head_candidate": best_added_candidate.get("candidate"),
        "best_added_large_384_256_ds": best_added_large_384_256,
        "best_added_delta_vs_canonical_large_384_256_ds": delta_vs_canonical,
        "best_added_high_search_breakthrough_preserved": best_added_signature_preserved,
        "added_data_decision": added_data_decision,
        "experiment_classification": experiment_classification,
        "candidates": summary_candidates,
    }

    summary_path = workdir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[report] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
