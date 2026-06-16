#!/usr/bin/env python3
"""Run control_ep2 PUCT trainable-scope ablation.

Does not generate new self-play, promote, or overwrite model-artifact/current.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
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
EXPECTED_PUCT_REPLAY_SHA256 = (
    "045287417b1878662ba51092bf9c770c66f9751a686b2bdcec4456ad4f521393"
)


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def require_existing_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")


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
    if path.suffix == ".jsonl":
        result["rows"] = count_jsonl_rows(path)
    return result


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
    trainable_scope: str,
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
        trainable_scope,
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
        timeout=7200,
    )
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(f"train.py failed: {result.stderr[-2000:]}")
    metrics: dict[str, Any] = {
        "training_elapsed_s": elapsed,
        "trainable_scope": trainable_scope,
    }
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("policy_loss="):
            metrics["policy_loss"] = float(line.split("=", 1)[1])
        elif line.startswith("value_loss="):
            metrics["value_loss"] = float(line.split("=", 1)[1])
        elif line.startswith("best_val_loss="):
            metrics["best_val_loss"] = float(line.split("=", 1)[1])
        elif line.startswith("saved_epoch_checkpoint_epoch="):
            parts = line.split()
            metrics[f"epoch_{parts[0].split('=', 1)[1]}_path"] = parts[1].split("=", 1)[
                1
            ]
    for line in result.stderr.splitlines():
        for key in ("trainable_params", "frozen_params", "total_params"):
            token = f"{key}="
            if token in line:
                for part in line.split():
                    if part.startswith(token):
                        metrics[key] = int(part.split("=", 1)[1])
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
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"export_artifact failed: {result.stderr[-2000:]}")


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
        raise RuntimeError(f"benchmark failed: {result.stderr[-2000:]}")
    return json.loads(
        (Path(workdir) / "temperature_benchmark_report.json").read_text(
            encoding="utf-8"
        )
    )


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
        raise RuntimeError(f"gate failed: {result.stderr[-2000:]}")
    return json.loads(Path(out).read_text(encoding="utf-8"))


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


def ensure_policy_head_ref(
    *,
    artifact_dir: Path,
    checkpoint_path: Path,
    version: str,
) -> tuple[Path, Path, str]:
    artifact_weights = artifact_dir / "weights.json"
    if artifact_weights.is_file():
        checkpoint_source = (
            checkpoint_path if checkpoint_path.is_file() else artifact_dir / "model.npz"
        )
        require_existing_file(
            checkpoint_source, "puct policy-head reference checkpoint"
        )
        return checkpoint_source, artifact_dir, "reused_artifact"

    require_existing_file(checkpoint_path, "puct policy-head reference checkpoint")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    export_checkpoint(
        checkpoint_path=str(checkpoint_path),
        out_dir=str(artifact_dir),
        version=version,
        policy_loss=0.0,
        value_loss=0.0,
    )
    return checkpoint_path, artifact_dir, "re_exported_from_checkpoint"


def lane_scope_family(lane_name: str) -> str:
    if lane_name.startswith("puct_all_"):
        return "all"
    if lane_name.startswith("puct_last_block_policy_"):
        return "last_block_policy"
    if lane_name.startswith("puct_policy_head_"):
        return "policy_head"
    return "reference"


def large_metric(row: dict[str, Any], budget: str) -> float:
    return float(
        row.get("large_budget_results", {}).get(budget, {}).get("ds", float("-inf"))
    )


def classify_experiment(
    summary_candidates: list[dict[str, Any]],
    baseline_row: dict[str, Any],
) -> str:
    baseline_large_384 = large_metric(baseline_row, "384:256")
    baseline_large_1200 = large_metric(baseline_row, "1200:1200")

    def is_safe_update(row: dict[str, Any]) -> bool:
        return (
            large_metric(row, "384:256") >= baseline_large_384 - 0.01
            and large_metric(row, "1200:1200") >= baseline_large_1200 + 0.03
            and row.get("high_search_breakthrough_preserved") is True
        )

    all_candidates = [
        row
        for row in summary_candidates
        if lane_scope_family(str(row.get("candidate"))) == "all"
    ]
    last_block_candidates = [
        row
        for row in summary_candidates
        if lane_scope_family(str(row.get("candidate"))) == "last_block_policy"
    ]

    if any(is_safe_update(row) for row in all_candidates):
        return "full_network_safe_update"
    if any(is_safe_update(row) for row in last_block_candidates):
        return "last_block_policy_safe_update"

    broader_candidates = [*all_candidates, *last_block_candidates]
    for row in broader_candidates:
        if large_metric(row, "384:256") < baseline_large_384 - 0.03:
            return "broader_scope_destructive"
        if row.get("high_search_breakthrough_preserved") is False:
            return "broader_scope_destructive"

    return "policy_head_sufficient"


def replay_duplicate_trajectory_count(path: Path) -> int:
    rows = read_jsonl(path)
    per_game_hash: dict[int, str] = {}
    for row in rows:
        game_index = row.get("game_index")
        trajectory_hash = row.get("trajectory_hash")
        if not isinstance(game_index, int):
            continue
        if not isinstance(trajectory_hash, str) or not trajectory_hash:
            continue
        per_game_hash.setdefault(game_index, trajectory_hash)
    counter = Counter(per_game_hash.values())
    return sum(count for count in counter.values() if count > 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir",
        default="/tmp/azlite_control_ep2_puct_scope_ablation",
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
        "--puct-replay",
        default="/tmp/azlite_control_ep2_puct_smoke/control_ep2_puct_selfplay.jsonl",
    )
    parser.add_argument(
        "--puct-policy-head-e2-artifact",
        default="/tmp/azlite_control_ep2_puct_smoke/artifacts/puct_policy_head_e2",
    )
    parser.add_argument(
        "--puct-policy-head-e2-checkpoint",
        default="/tmp/azlite_control_ep2_puct_smoke/puct_policy_head_e2/checkpoint_epoch2.npz",
    )
    parser.add_argument(
        "--generic-bootstrap",
        default="/tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl",
    )
    parser.add_argument(
        "--random-teacher",
        default="/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl",
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
    parser.add_argument("--timeout", type=int, default=14400)
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
    parser.add_argument(
        "--expected-puct-replay-sha256",
        default=EXPECTED_PUCT_REPLAY_SHA256,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    control_checkpoint = Path(args.control_checkpoint)
    control_artifact = Path(args.control_artifact)
    current_artifact = Path(args.current)
    puct_replay = Path(args.puct_replay)
    puct_policy_head_e2_artifact = Path(args.puct_policy_head_e2_artifact)
    puct_policy_head_e2_checkpoint = Path(args.puct_policy_head_e2_checkpoint)
    generic_bootstrap = Path(args.generic_bootstrap)
    random_teacher = Path(args.random_teacher)
    medium_suite = Path(args.medium_suite)
    large_suite = Path(args.large_suite)

    require_existing_file(control_checkpoint, "control checkpoint")
    require_existing_file(control_artifact / "weights.json", "control artifact weights")
    require_existing_file(current_artifact / "weights.json", "current artifact weights")
    require_existing_file(puct_replay, "puct replay")
    require_existing_file(generic_bootstrap, "generic bootstrap replay")
    require_existing_file(random_teacher, "random teacher replay")
    require_existing_file(medium_suite, "medium suite")
    require_existing_file(large_suite, "large suite")

    puct_ref_checkpoint_path, puct_ref_artifact_dir, puct_ref_source = (
        ensure_policy_head_ref(
            artifact_dir=puct_policy_head_e2_artifact,
            checkpoint_path=puct_policy_head_e2_checkpoint,
            version="puct_policy_head_e2_ref",
        )
    )

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
        "puct_replay": verify_expected_hash(
            puct_replay,
            args.expected_puct_replay_sha256,
            "puct replay",
        ),
        "puct_policy_head_e2_artifact_weights": build_input_summary(
            puct_ref_artifact_dir / "weights.json"
        ),
        "puct_policy_head_e2_checkpoint": build_input_summary(puct_ref_checkpoint_path),
        "generic_bootstrap": build_input_summary(generic_bootstrap),
        "random_teacher": build_input_summary(random_teacher),
        "medium_suite": build_input_summary(medium_suite),
        "large_suite": build_input_summary(large_suite),
    }

    lane_specs = [
        {
            "name": "puct_last_block_policy_e1",
            "epochs": 1,
            "trainable_scope": "last_block_policy",
        },
        {
            "name": "puct_last_block_policy_e2",
            "epochs": 2,
            "trainable_scope": "last_block_policy",
        },
        {
            "name": "puct_all_e1",
            "epochs": 1,
            "trainable_scope": "all",
        },
        {
            "name": "puct_all_e2",
            "epochs": 2,
            "trainable_scope": "all",
        },
    ]
    replay_data_files = f"{generic_bootstrap},{random_teacher},{puct_replay}"
    replay_weights = "4,1,1"

    lanes: list[dict[str, Any]] = [
        {
            "name": "canonical_ref",
            "report_candidate_name": "control_ep2",
            "epochs": 0,
            "checkpoint_path": str(control_checkpoint),
            "artifact_dir": str(control_artifact),
            "source": "canonical_ref",
            "trainable_scope": "none",
        },
        {
            "name": "puct_policy_head_e2_ref",
            "report_candidate_name": puct_ref_artifact_dir.name,
            "epochs": 2,
            "checkpoint_path": str(puct_ref_checkpoint_path),
            "artifact_dir": str(puct_ref_artifact_dir),
            "source": puct_ref_source,
            "trainable_scope": "policy_head",
            "data_files": [
                str(generic_bootstrap),
                str(random_teacher),
                str(puct_replay),
            ],
            "replay_weights": replay_weights,
        },
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
                data_files=replay_data_files,
                replay_weights=replay_weights,
                init_checkpoint=str(control_checkpoint),
                out=str(checkpoint_out),
                top_k_dir=str(lane_dir),
                epochs=int(lane_spec["epochs"]),
                seed=args.seed,
                trainable_scope=str(lane_spec["trainable_scope"]),
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
                "source": "puct_scope_ablation",
                "replay_weights": replay_weights,
                "data_files": replay_data_files.split(","),
                "trainable_scope": lane_spec["trainable_scope"],
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

    large_report_path = workdir / "eval_large" / "temperature_benchmark_report.json"
    if args.skip_eval_large:
        large_report = json.loads(large_report_path.read_text(encoding="utf-8"))
    else:
        large_report = run_opening_suite_benchmark(
            workdir=str(workdir / "eval_large"),
            suite=str(large_suite),
            current=str(current_artifact),
            candidates=candidate_paths,
            budget_pairs=args.budget_pairs,
            games_per_opening=args.games_per_opening,
            seed=args.seed,
            workers=args.workers,
            timeout=args.timeout,
        )

    puct_policy_head_large_384_256 = candidate_standard_ds(
        large_report, puct_ref_artifact_dir.name
    )
    gate_target_names = ["canonical_ref", "puct_policy_head_e2_ref"]
    for lane in lanes:
        lane_name = str(lane["name"])
        if lane_name in gate_target_names:
            continue
        candidate_score = candidate_standard_ds(
            large_report, str(lane["report_candidate_name"])
        )
        if candidate_score >= puct_policy_head_large_384_256 - 0.01:
            gate_target_names.append(lane_name)

    gate_reports: dict[str, dict[str, Any]] = {}
    gate_dir = workdir / "eval_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_gate:
        for lane_name in gate_target_names:
            gate_reports[lane_name] = json.loads(
                (gate_dir / f"{lane_name}_default_gate.json").read_text(
                    encoding="utf-8"
                )
            )
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
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "artifact_weights_sha256": sha256_file(artifact_dir / "weights.json"),
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
            row["primary_large_384_256_ds"] = (
                row["large_budget_results"].get("384:256", {}).get("ds")
            )
            row["large_1200_1200_ds"] = (
                row["large_budget_results"].get("1200:1200", {}).get("ds")
            )
            row["large_256_768_ds"] = (
                row["large_budget_results"].get("256:768", {}).get("ds")
            )
            row["large_384_256_p0_score"] = (
                row["large_budget_results"].get("384:256", {}).get("p0_score")
            )
            row["large_384_256_p1_score"] = (
                row["large_budget_results"].get("384:256", {}).get("p1_score")
            )
            row["large_384_256_duplicate_trajectory_count"] = (
                row["large_budget_results"]
                .get("384:256", {})
                .get("duplicate_trajectory_count")
            )
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

    baseline_row = next(
        row
        for row in summary_candidates
        if row["candidate"] == "puct_policy_head_e2_ref"
    )
    classification = classify_experiment(summary_candidates, baseline_row)
    canonical_row = next(
        row for row in summary_candidates if row["candidate"] == "canonical_ref"
    )

    summary = {
        "schema": "azlite_control_ep2_puct_scope_ablation_v1",
        "status": "completed",
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
            "replay_mix": [
                str(generic_bootstrap),
                str(random_teacher),
                str(puct_replay),
            ],
            "replay_weights": [4, 1, 1],
            "promotion": "disabled",
            "overwrite_current": False,
            "new_self_play_generation": False,
        },
        "puct_policy_head_e2_reference_source": puct_ref_source,
        "gate_targets": gate_target_names,
        "classification": classification,
        "canonical_large_384_256_ds": large_metric(canonical_row, "384:256"),
        "puct_policy_head_e2_ref_large_384_256_ds": large_metric(
            baseline_row, "384:256"
        ),
        "puct_policy_head_e2_ref_large_1200_1200_ds": large_metric(
            baseline_row, "1200:1200"
        ),
        "replay_duplicate_trajectories": replay_duplicate_trajectory_count(puct_replay),
        "candidates": summary_candidates,
    }
    summary_path = workdir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[report] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
