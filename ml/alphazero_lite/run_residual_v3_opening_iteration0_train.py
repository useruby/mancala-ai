#!/usr/bin/env python3
"""Materialize and evaluate residual_v3 opening iteration-0 candidates.

Consumes the deterministic preflight manifest from the selection-only PR,
builds `train.py`-ready supervised rows from the selected positions, trains a
small conservative residual_v3 sweep from `model-artifact/current`, and
evaluates the candidates against current on the deterministic opening-suite
seat-aware benchmark.

This runner does not promote or overwrite `model-artifact/current`.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import train as train_lib  # noqa: E402
from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint  # noqa: E402
from ml.alphazero_lite.run_residual_v3_opening_iteration0_preflight import (  # noqa: E402
    EXPECTED_CURRENT_WEIGHTS_SHA256,
    PREFLIGHT_SCHEMA,
    build_promoted_search_profile,
    canonical_state_hash,
    distribute,
    phase_bucket,
    search_profile_hash,
    sha256_file,
    stable_float,
    validate_guardrails,
    write_json,
    write_jsonl,
)
from ml.alphazero_lite.self_play import encode_state  # noqa: E402


DEFAULT_TARGET_LANE_LABEL = "sim768_equal_override"
TRAINING_SCHEMA = "azlite_residual_v3_opening_iteration0_training_v1"
TARGET_ROWS_FILENAME = "iteration0_train_targets.jsonl"
TARGET_MANIFEST_FILENAME = "iteration0_target_manifest.json"
SUMMARY_FILENAME = "iteration0_training_summary.json"
DEFAULT_BUDGET_PAIRS = "384:256,768:768,1200:1200,1200:256"
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 128
DEFAULT_VALUE_LOSS_WEIGHT = 0.3
LANE_SPECS = (
    {"name": "low_lr_all", "lr": 3e-5, "trainable_scope": "all"},
    {"name": "very_low_lr_all", "lr": 1e-5, "trainable_scope": "all"},
    {
        "name": "very_low_lr_policy_head",
        "lr": 1e-5,
        "trainable_scope": "policy_head",
    },
)

ENV = os.environ.copy()
ENV.setdefault("OMP_NUM_THREADS", "1")
ENV.setdefault("OPENBLAS_NUM_THREADS", "1")
ENV.setdefault("MKL_NUM_THREADS", "1")

BENCHMARK_BUDGET_LABELS = {
    "384:256": "standard",
    "768:768": "equal_768",
    "1200:1200": "equal_high",
    "1200:256": "1200_vs_256",
}


def _python() -> str:
    candidate = REPO_ROOT / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clamp_unit_interval(value: float) -> float:
    return max(-1.0, min(1.0, stable_float(float(value))))


def require_existing_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")


def verify_preflight_manifest(
    *, manifest_path: Path, selected_rows_path: Path | None = None
) -> tuple[dict[str, Any], Path]:
    require_existing_file(manifest_path, "iteration-0 training manifest")
    manifest = load_json(manifest_path)
    if manifest.get("schema") != PREFLIGHT_SCHEMA:
        raise RuntimeError(
            f"unexpected preflight schema: {manifest.get('schema')} != {PREFLIGHT_SCHEMA}"
        )
    expected_profile = build_promoted_search_profile()
    search_profile = manifest.get("search_profile", {})
    validate_guardrails(
        search_profile=search_profile,
        model_type=str(search_profile.get("model_type", "")),
    )
    if search_profile_hash(search_profile) != search_profile_hash(expected_profile):
        raise RuntimeError(
            "preflight search profile hash mismatch against promoted schedule"
        )

    training_data = manifest.get("training_data", {})
    materialized_rows_path = (
        selected_rows_path
        if selected_rows_path is not None
        else Path(training_data["path"])
    )
    require_existing_file(materialized_rows_path, "selected positions JSONL")
    actual_sha = sha256_file(materialized_rows_path)
    expected_sha = str(training_data.get("sha256", ""))
    if actual_sha != expected_sha:
        raise RuntimeError(
            "selected positions hash mismatch: "
            f"expected {expected_sha}, got {actual_sha}"
        )
    expected_rows = int(training_data.get("rows", 0))
    actual_rows = count_jsonl_rows(materialized_rows_path)
    if actual_rows != expected_rows:
        raise RuntimeError(
            f"selected positions row count mismatch: expected {expected_rows}, got {actual_rows}"
        )
    return manifest, materialized_rows_path


def verify_current_artifact(
    *,
    current_path: Path,
    expected_weights_sha256: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    weights_path = current_path / "weights.json"
    metadata_path = current_path / "metadata.json"
    require_existing_file(weights_path, "current artifact weights")
    require_existing_file(metadata_path, "current artifact metadata")

    actual_sha = sha256_file(weights_path)
    if actual_sha != expected_weights_sha256:
        raise RuntimeError(
            "current artifact weights hash mismatch: "
            f"expected {expected_weights_sha256}, got {actual_sha}"
        )
    manifest_expected_sha = str(
        manifest.get("current_artifact", {}).get("weights_sha256", "")
    )
    if actual_sha != manifest_expected_sha:
        raise RuntimeError(
            "current artifact hash does not match preflight manifest: "
            f"expected {manifest_expected_sha}, got {actual_sha}"
        )

    metadata = load_json(metadata_path)
    architecture = metadata.get("architecture", {})
    model_type = str(architecture.get("model_type", ""))
    validate_guardrails(
        search_profile=manifest["search_profile"],
        model_type=model_type,
    )
    if str(metadata.get("input_encoding", "")) != "kalah_v3":
        raise RuntimeError(
            "guardrail violation: current artifact input_encoding must be kalah_v3"
        )
    return metadata


def hidden_sizes_arg_from_metadata(metadata: dict[str, Any]) -> str:
    architecture = metadata.get("architecture", {})
    trunk_size = int(architecture.get("trunk_size", 0))
    residual_block_count = int(architecture.get("residual_block_count", 0))
    if trunk_size <= 0 or residual_block_count <= 0:
        raise RuntimeError(
            "current artifact metadata is missing residual_v3 trunk/block sizes"
        )
    return f"{trunk_size},{residual_block_count}"


def select_target_search_row(
    search_results: list[dict[str, Any]], *, preferred_label: str
) -> dict[str, Any]:
    by_label = {str(row["label"]): row for row in search_results}
    if preferred_label in by_label:
        return by_label[preferred_label]
    strongest = sorted(
        search_results,
        key=lambda row: (
            -int(row.get("simulations", 0)),
            str(row.get("budget_pair", "")) != "768:768",
            str(row.get("label", "")),
        ),
    )
    if not strongest:
        raise RuntimeError("selected row is missing preflight search results")
    return strongest[0]


def normalize_policy(policy: list[float], legal_moves: list[int]) -> list[float]:
    normalized = [0.0] * 6
    if not legal_moves:
        return normalized
    total = sum(float(policy[move]) for move in legal_moves)
    if total <= 0.0:
        uniform = 1.0 / len(legal_moves)
        for move in legal_moves:
            normalized[move] = stable_float(uniform)
        return normalized
    for move in legal_moves:
        normalized[move] = stable_float(float(policy[move]) / total)
    residual = 1.0 - sum(normalized)
    normalized[max(legal_moves)] = stable_float(normalized[max(legal_moves)] + residual)
    return normalized


def build_target_row(
    *,
    selected_row: dict[str, Any],
    input_encoding: str,
    preferred_target_lane_label: str,
    search_profile_hash_value: str,
) -> dict[str, Any]:
    raw_state = dict(selected_row["state"])
    state_hash = str(selected_row.get("state_hash") or canonical_state_hash(raw_state))
    target_search_row = select_target_search_row(
        list(selected_row.get("search_results", [])),
        preferred_label=preferred_target_lane_label,
    )
    legal_moves = [int(move) for move in target_search_row.get("legal_moves", [])]
    policy = normalize_policy(list(target_search_row["search_policy"]), legal_moves)
    value = clamp_unit_interval(float(target_search_row["root_value"]))
    encoded_state = encode_state(raw_state, input_encoding=input_encoding)
    return {
        "state": encoded_state,
        "policy": policy,
        "value": value,
        "policy_target_mode": "default",
        "value_target_mode": "default",
        "input_encoding": input_encoding,
        "source_state_hash": state_hash,
        "source_prefix_hash": str(selected_row["prefix_hash"]),
        "source_prefix_moves": [
            int(move) for move in selected_row.get("prefix_moves", [])
        ],
        "selection_rank": int(selected_row["selection_rank"]),
        "selection_tags": [str(tag) for tag in selected_row.get("selection_tags", [])],
        "search_profile_hash": search_profile_hash_value,
        "target_lane_label": str(target_search_row["label"]),
        "target_budget_pair": str(target_search_row["budget_pair"]),
        "target_simulations": int(target_search_row["simulations"]),
        "target_c_puct": stable_float(target_search_row["c_puct"]),
        "target_root_value_source": "selected_rows.search_results[].root_value",
        "target_policy_source": "selected_rows.search_results[].search_policy",
        "target_top_share": stable_float(target_search_row["top_share"]),
        "target_margin": stable_float(target_search_row["margin"]),
        "target_entropy": stable_float(target_search_row["entropy"]),
        "target_legal_moves": legal_moves,
        "target_role_context": [
            str(value) for value in target_search_row.get("role_context", [])
        ],
        "source_phase_bucket": str(
            selected_row.get("phase_bucket", phase_bucket(selected_row))
        ),
        "source_side_to_move": int(
            selected_row.get("side_to_move", raw_state["current_player"])
        ),
        "source_first_move_family": str(selected_row.get("first_move_family", "none")),
        "selection_metrics": dict(selected_row.get("selection_metrics", {})),
    }


def materialize_target_rows(
    *,
    selected_rows: list[dict[str, Any]],
    input_encoding: str,
    preferred_target_lane_label: str,
    search_profile_hash_value: str,
) -> list[dict[str, Any]]:
    ordered_rows = sorted(selected_rows, key=lambda row: int(row["selection_rank"]))
    return [
        build_target_row(
            selected_row=row,
            input_encoding=input_encoding,
            preferred_target_lane_label=preferred_target_lane_label,
            search_profile_hash_value=search_profile_hash_value,
        )
        for row in ordered_rows
    ]


def verify_train_compatibility(target_rows_path: Path) -> dict[str, Any]:
    x, p, v = train_lib.load_jsonl(target_rows_path)
    return {
        "rows": int(x.shape[0]),
        "feature_count": int(x.shape[1]) if x.ndim == 2 and x.shape[0] > 0 else 0,
        "policy_size": int(p.shape[1]) if p.ndim == 2 and p.shape[0] > 0 else 0,
        "value_count": int(v.shape[0]),
    }


def build_target_manifest(
    *,
    preflight_manifest_path: Path,
    preflight_manifest: dict[str, Any],
    selected_rows_path: Path,
    selected_rows: list[dict[str, Any]],
    target_rows_path: Path,
    target_rows: list[dict[str, Any]],
    preferred_target_lane_label: str,
    current_path: Path,
    current_metadata: dict[str, Any],
    train_compatibility: dict[str, Any],
) -> dict[str, Any]:
    target_values = [float(row["value"]) for row in target_rows]
    lane_distribution = Counter(str(row["target_lane_label"]) for row in target_rows)
    return {
        "schema": TRAINING_SCHEMA,
        "classification": "residual_v3_opening_iteration0_target_materialized",
        "preflight_manifest": {
            "path": str(preflight_manifest_path),
            "sha256": sha256_file(preflight_manifest_path),
            "schema": preflight_manifest["schema"],
        },
        "selected_positions": {
            "path": str(selected_rows_path),
            "sha256": sha256_file(selected_rows_path),
            "rows": len(selected_rows),
        },
        "current_artifact": {
            "path": str(current_path),
            "weights_sha256": sha256_file(current_path / "weights.json"),
            "model_type": str(current_metadata["architecture"]["model_type"]),
            "input_encoding": str(current_metadata["input_encoding"]),
        },
        "target_generation": {
            "target_policy_source": "deterministic searched policy from selected_rows.search_results",
            "target_value_source": "deterministic searched root value from the same lane selected_rows.search_results[].root_value",
            "preferred_target_lane_label": preferred_target_lane_label,
            "search_profile": preflight_manifest["search_profile"],
            "search_profile_hash": preflight_manifest["search_profile_hash"],
            "tablebase_overlay": False,
            "root_prior_transform": None,
            "value_transform": None,
        },
        "target_dataset": {
            "path": str(target_rows_path),
            "sha256": sha256_file(target_rows_path),
            "rows": len(target_rows),
            "train_py_compatible": True,
            "compatibility": train_compatibility,
        },
        "distribution": {
            "phase": distribute(target_rows, "source_phase_bucket"),
            "side_to_move": distribute(target_rows, "source_side_to_move"),
            "first_move_family": distribute(target_rows, "source_first_move_family"),
            "target_lane": {
                key: int(lane_distribution[key]) for key in sorted(lane_distribution)
            },
        },
        "value_summary": {
            "min": stable_float(min(target_values)) if target_values else 0.0,
            "max": stable_float(max(target_values)) if target_values else 0.0,
            "mean": stable_float(statistics.fmean(target_values))
            if target_values
            else 0.0,
        },
    }


def run_command(command: list[str], *, cwd: Path, timeout: int) -> dict[str, Any]:
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
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "duration_s": round(time.time() - started, 2),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def parse_train_metrics(stdout: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        for key in (
            "policy_loss",
            "value_loss",
            "best_val_loss",
            "total_loss",
            "saved_epoch_checkpoint_epoch",
        ):
            token = f"{key}="
            if not line.startswith(token):
                continue
            if key == "saved_epoch_checkpoint_epoch":
                continue
            try:
                metrics[key] = float(line.split("=", 1)[1])
            except ValueError:
                continue
    return metrics


def export_artifact(
    *,
    checkpoint_path: Path,
    artifact_dir: Path,
    version: str,
    input_encoding: str,
    timeout: int,
) -> dict[str, Any]:
    command = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/export_artifact.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--out-dir",
        str(artifact_dir),
        "--version",
        version,
        "--model-type",
        "residual_v3",
        "--rules-version",
        "kalah_v3",
        "--input-encoding",
        input_encoding,
    ]
    result = run_command(command, cwd=REPO_ROOT, timeout=timeout)
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"][-2000:])
    return {
        "artifact_dir": str(artifact_dir),
        "weights_sha256": sha256_file(artifact_dir / "weights.json"),
        "metadata_path": str(artifact_dir / "metadata.json"),
        "command": command,
    }


def augment_candidate_metadata(
    *,
    artifact_dir: Path,
    lane_name: str,
    lane_spec: dict[str, Any],
    current_path: Path,
    current_sha256: str,
    target_manifest_path: Path,
    target_manifest: dict[str, Any],
) -> None:
    metadata_path = artifact_dir / "metadata.json"
    metadata = load_json(metadata_path)
    metadata["parent_artifact_path"] = str(current_path)
    metadata["parent_artifact_weights_sha256"] = current_sha256
    metadata["source_runner"] = (
        "ml/alphazero_lite/run_residual_v3_opening_iteration0_train.py"
    )
    metadata["source_experiment"] = (
        "docs/alphazero-lite-residual-v3-opening-iteration0-training-results.md"
    )
    metadata["selected_lane"] = lane_name
    metadata["trainable_scope"] = lane_spec["trainable_scope"]
    metadata["opening_iteration0_target_manifest"] = {
        "path": str(target_manifest_path),
        "sha256": sha256_file(target_manifest_path),
        "target_dataset_sha256": target_manifest["target_dataset"]["sha256"],
    }
    metadata["architecture_change"] = "none"
    write_json(metadata_path, metadata)


def train_candidate_lane(
    *,
    workdir: Path,
    lane_spec: dict[str, Any],
    init_checkpoint: Path,
    target_rows_path: Path,
    hidden_sizes_arg: str,
    input_encoding: str,
    seed: int,
    epochs: int,
    batch_size: int,
    value_loss_weight: float,
    timeout: int,
    current_path: Path,
    current_sha256: str,
    target_manifest_path: Path,
    target_manifest: dict[str, Any],
) -> dict[str, Any]:
    lane_dir = workdir / "candidates" / lane_spec["name"]
    lane_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = lane_dir / f"checkpoint_epoch{epochs}.npz"
    command = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/train.py"),
        "--data",
        str(target_rows_path),
        "--out",
        str(checkpoint_path),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lane_spec["lr"]),
        "--seed",
        str(seed),
        "--device",
        "auto",
        "--value-loss-weight",
        str(value_loss_weight),
        "--value-loss",
        "huber",
        "--val-split",
        "0.0",
        "--grad-clip",
        "1.0",
        "--hidden-sizes",
        hidden_sizes_arg,
        "--model-type",
        "residual_v3",
        "--input-encoding",
        input_encoding,
        "--policy-target-mode",
        "default",
        "--value-target-mode",
        "default",
        "--save-top-k",
        "0",
        "--save-epochs",
        str(epochs),
        "--lr-scheduler",
        "none",
        "--trainable-scope",
        lane_spec["trainable_scope"],
        "--init-checkpoint",
        str(init_checkpoint),
    ]
    train_result = run_command(command, cwd=REPO_ROOT, timeout=timeout)
    if train_result["returncode"] != 0:
        raise RuntimeError(train_result["stderr"][-4000:])
    if not checkpoint_path.is_file():
        raise RuntimeError(f"train.py did not produce checkpoint: {checkpoint_path}")

    artifact_dir = workdir / "artifacts" / lane_spec["name"]
    export_result = export_artifact(
        checkpoint_path=checkpoint_path,
        artifact_dir=artifact_dir,
        version=f"azlite-{lane_spec['name']}",
        input_encoding=input_encoding,
        timeout=min(timeout, 600),
    )
    augment_candidate_metadata(
        artifact_dir=artifact_dir,
        lane_name=lane_spec["name"],
        lane_spec=lane_spec,
        current_path=current_path,
        current_sha256=current_sha256,
        target_manifest_path=target_manifest_path,
        target_manifest=target_manifest,
    )
    return {
        "lane": lane_spec["name"],
        "learning_rate": lane_spec["lr"],
        "trainable_scope": lane_spec["trainable_scope"],
        "epochs": epochs,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "artifact_dir": str(artifact_dir),
        "artifact_weights_sha256": sha256_file(artifact_dir / "weights.json"),
        "train_command": command,
        "export_command": export_result["command"],
        "train_metrics": parse_train_metrics(train_result["stdout"]),
    }


def run_opening_suite_benchmark(
    *,
    workdir: Path,
    eval_suite: Path,
    current_path: Path,
    candidate_dirs: list[Path],
    budget_pairs: str,
    games_per_opening: int,
    seed: int,
    workers: int,
    timeout: int,
) -> dict[str, Any]:
    command = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/run_opening_suite_seat_benchmark.py"),
        "--workdir",
        str(workdir),
        "--suite",
        str(eval_suite),
        "--current",
        str(current_path),
        "--candidates",
        ",".join(str(path) for path in candidate_dirs),
        "--budget-pairs",
        budget_pairs,
        "--games-per-opening",
        str(games_per_opening),
        "--seed",
        str(seed),
        "--root-policy-mode",
        "deterministic",
        "--workers",
        str(max(1, workers)),
        "--timeout",
        str(timeout),
        "--c-puct",
        "1.25",
        "--c-puct-schedule-json",
        json.dumps({"768:768": 0.90}, sort_keys=True),
        "--tactical-root-bias",
        "0.0",
    ]
    result = run_command(command, cwd=REPO_ROOT, timeout=timeout + 300)
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"][-4000:])
    report_path = workdir / "temperature_benchmark_report.json"
    require_existing_file(report_path, "opening-suite benchmark report")
    report = load_json(report_path)
    report["command"] = command
    return report


def summarize_candidate_budgets(candidate_report: dict[str, Any]) -> dict[str, Any]:
    budget_results = dict(candidate_report.get("budget_results", {}))
    summary: dict[str, Any] = {}
    for budget_pair, budget_label in BENCHMARK_BUDGET_LABELS.items():
        row = dict(budget_results.get(budget_label, {}))
        summary[budget_pair] = {
            "label": budget_label,
            "ds": stable_float(float(row.get("ds", 0.0))),
            "p0_score": stable_float(float(row.get("p0_score", 0.0))),
            "p1_score": stable_float(float(row.get("p1_score", 0.0))),
            "disadvantaged_seat_score": stable_float(
                float(row.get("disadvantaged_seat_score", 0.0))
            ),
            "duplicate_trajectory_rate": stable_float(
                float(row.get("duplicate_trajectory_rate", 0.0))
            ),
            "total_games": int(row.get("total_games", 0)),
        }
    return summary


def candidate_gate_summary(candidate_report: dict[str, Any]) -> dict[str, Any]:
    budgets = summarize_candidate_budgets(candidate_report)
    std_ds = float(budgets["384:256"]["ds"])
    eq_768_ds = float(budgets["768:768"]["ds"])
    eq_hi_ds = float(budgets["1200:1200"]["ds"])
    hi_asym_ds = float(budgets["1200:256"]["ds"])
    primary_384 = std_ds > 0.0
    primary_768 = eq_768_ds >= 0.0
    nonreg_1200 = eq_hi_ds >= 0.0
    nonreg_1200_256 = hi_asym_ds >= 0.0
    follow_up = primary_384 and primary_768 and nonreg_1200 and nonreg_1200_256
    return {
        "budgets": budgets,
        "primary_384_256_improved": primary_384,
        "primary_768_768_non_regressed": primary_768,
        "non_regression_1200_1200": nonreg_1200,
        "non_regression_1200_256": nonreg_1200_256,
        "follow_up_promotion_gate_candidate": follow_up,
    }


def classify_results(candidate_summaries: list[dict[str, Any]]) -> str:
    if any(
        summary["gates"]["follow_up_promotion_gate_candidate"]
        for summary in candidate_summaries
    ):
        return "residual_v3_opening_iteration0_candidate_positive"
    if any(
        summary["gates"]["primary_384_256_improved"]
        or summary["gates"]["primary_768_768_non_regressed"]
        for summary in candidate_summaries
    ):
        return "residual_v3_opening_iteration0_candidate_mixed"
    return "residual_v3_opening_iteration0_rejected"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Residual_v3 opening iteration-0 target materialization and candidate training."
    )
    parser.add_argument("--workdir", required=True)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Preflight iteration0_training_manifest.json path",
    )
    parser.add_argument(
        "--selected-rows",
        default=None,
        help="Optional override path for iteration0_selected_positions.jsonl",
    )
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--expected-current-sha256",
        default=EXPECTED_CURRENT_WEIGHTS_SHA256,
    )
    parser.add_argument(
        "--target-lane-label",
        default=DEFAULT_TARGET_LANE_LABEL,
    )
    parser.add_argument(
        "--eval-suite",
        default=None,
        help="Optional opening-suite JSONL override for evaluation",
    )
    parser.add_argument("--budget-pairs", default=DEFAULT_BUDGET_PAIRS)
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--value-loss-weight", type=float, default=DEFAULT_VALUE_LOSS_WEIGHT
    )
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument(
        "--skip-policy-head-lane",
        action="store_true",
        help="Do not include the clean policy_head conservative lane.",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    selected_rows_override = Path(args.selected_rows) if args.selected_rows else None
    current_path = Path(args.current)

    manifest, selected_rows_path = verify_preflight_manifest(
        manifest_path=manifest_path,
        selected_rows_path=selected_rows_override,
    )
    current_metadata = verify_current_artifact(
        current_path=current_path,
        expected_weights_sha256=str(args.expected_current_sha256),
        manifest=manifest,
    )

    selected_rows = parse_jsonl(selected_rows_path)
    target_rows = materialize_target_rows(
        selected_rows=selected_rows,
        input_encoding=str(current_metadata["input_encoding"]),
        preferred_target_lane_label=str(args.target_lane_label),
        search_profile_hash_value=str(manifest["search_profile_hash"]),
    )
    targets_dir = workdir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    target_rows_path = targets_dir / TARGET_ROWS_FILENAME
    write_jsonl(target_rows_path, target_rows)
    train_compatibility = verify_train_compatibility(target_rows_path)
    target_manifest = build_target_manifest(
        preflight_manifest_path=manifest_path,
        preflight_manifest=manifest,
        selected_rows_path=selected_rows_path,
        selected_rows=selected_rows,
        target_rows_path=target_rows_path,
        target_rows=target_rows,
        preferred_target_lane_label=str(args.target_lane_label),
        current_path=current_path,
        current_metadata=current_metadata,
        train_compatibility=train_compatibility,
    )
    target_manifest_path = targets_dir / TARGET_MANIFEST_FILENAME
    write_json(target_manifest_path, target_manifest)

    current_checkpoint = workdir / "inputs" / "current_from_weights_json.npz"
    current_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    materialize_weights_json_checkpoint(
        weights_path=current_path / "weights.json",
        out_path=current_checkpoint,
    )

    lane_specs = [dict(spec) for spec in LANE_SPECS]
    if args.skip_policy_head_lane:
        lane_specs = [
            spec for spec in lane_specs if spec["trainable_scope"] != "policy_head"
        ]

    hidden_sizes_arg = hidden_sizes_arg_from_metadata(current_metadata)
    current_sha256 = sha256_file(current_path / "weights.json")
    trained_candidates: list[dict[str, Any]] = []
    if not args.skip_train:
        for lane_spec in lane_specs:
            trained_candidates.append(
                train_candidate_lane(
                    workdir=workdir,
                    lane_spec=lane_spec,
                    init_checkpoint=current_checkpoint,
                    target_rows_path=target_rows_path,
                    hidden_sizes_arg=hidden_sizes_arg,
                    input_encoding=str(current_metadata["input_encoding"]),
                    seed=int(args.seed),
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    value_loss_weight=float(args.value_loss_weight),
                    timeout=int(args.timeout),
                    current_path=current_path,
                    current_sha256=current_sha256,
                    target_manifest_path=target_manifest_path,
                    target_manifest=target_manifest,
                )
            )

    eval_suite = (
        Path(args.eval_suite)
        if args.eval_suite
        else Path(manifest["input_suites"][0]["path"])
    )
    require_existing_file(eval_suite, "evaluation opening suite")
    benchmark_report: dict[str, Any] | None = None
    candidate_summaries: list[dict[str, Any]] = []
    if not args.skip_eval and trained_candidates:
        benchmark_report = run_opening_suite_benchmark(
            workdir=workdir / "evaluation",
            eval_suite=eval_suite,
            current_path=current_path,
            candidate_dirs=[
                Path(candidate["artifact_dir"]) for candidate in trained_candidates
            ],
            budget_pairs=str(args.budget_pairs),
            games_per_opening=int(args.games_per_opening),
            seed=int(args.seed),
            workers=int(args.workers),
            timeout=int(args.timeout),
        )
        temperature_report = benchmark_report["temperature_reports"][0]
        seed_report = temperature_report["seed_reports"][0]
        for candidate_report in seed_report["candidate_reports"]:
            summary = {
                "candidate": str(candidate_report["candidate"]),
                "candidate_sha256": str(candidate_report["candidate_sha256"]),
                "gates": candidate_gate_summary(candidate_report),
            }
            candidate_summaries.append(summary)

    summary = {
        "schema": TRAINING_SCHEMA,
        "classification": classify_results(candidate_summaries),
        "preflight_manifest_path": str(manifest_path),
        "preflight_manifest_sha256": sha256_file(manifest_path),
        "selected_rows_path": str(selected_rows_path),
        "selected_rows_sha256": sha256_file(selected_rows_path),
        "target_manifest_path": str(target_manifest_path),
        "target_manifest_sha256": sha256_file(target_manifest_path),
        "target_dataset_path": str(target_rows_path),
        "target_dataset_sha256": sha256_file(target_rows_path),
        "eval_suite_path": str(eval_suite),
        "trained_candidates": trained_candidates,
        "evaluation": {
            "benchmark_report_path": None
            if benchmark_report is None
            else str(workdir / "evaluation" / "temperature_benchmark_report.json"),
            "candidate_summaries": candidate_summaries,
        },
    }
    summary_path = workdir / SUMMARY_FILENAME
    write_json(summary_path, summary)

    print(f"[iteration0-train] target_manifest={target_manifest_path}")
    print(f"[iteration0-train] target_rows={target_rows_path}")
    print(f"[iteration0-train] summary={summary_path}")
    print(f"[iteration0-train] classification={summary['classification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
