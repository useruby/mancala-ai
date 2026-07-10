#!/usr/bin/env python3
"""Search-teacher distillation value-target ablation for residual_v3_96x3.

Audits DS orientation, audits teacher value targets, trains same-size ablation
lanes, probes them against teacher search behavior, runs deterministic opening
suite evaluation, and writes JSON + markdown reports without promotion.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position  # noqa: E402
from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    parse_cpuct_schedule_json,
    schedule_definition,
)
from ml.alphazero_lite.export_artifact import sha256_file  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    benchmark_budget_results,
    bootstrap_ci,
    candidate_gate_preserved,
    find_candidate_report,
    pooled_per_opening_differences,
    run_default_gate,
    run_opening_suite_benchmark,
)
from ml.alphazero_lite.run_promoted_current_opening_puct_disagreement import (  # noqa: E402
    policy_entropy,
    read_jsonl,
    top_policy_move,
)
from ml.alphazero_lite.run_search_teacher_student_preflight import (  # noqa: E402
    DEFAULT_GATE_BUDGETS,
    DEFAULT_SUITE_EVAL_BUDGETS,
    DEFAULT_VALUE_LOSS,
    DEFAULT_HUBER_DELTA,
    StudentSpec,
    build_probe_rows,
    current_checkpoint_path,
    export_checkpoint_artifact,
    kl_divergence,
    load_json,
    masked_cross_entropy,
    masked_policy,
    verify_expected_hash,
    write_json,
)
from ml.alphazero_lite.self_play import build_eval_search_options  # noqa: E402
from ml.alphazero_lite.train import (  # noqa: E402
    PolicyValueNet,
    apply_trainable_scope,
    checkpoint_from_model,
    compute_value_loss_vector,
    input_size_for_encoding,
    load_checkpoint_into_model,
    select_device,
    set_seed,
)


SUMMARY_SCHEMA = "azlite_search_teacher_distillation_value_ablation_v1"
REPORT_PATH = (
    REPO_ROOT
    / "docs/alphazero-lite-search-teacher-distillation-value-ablation-results.md"
)
SUMMARY_FILENAME = "summary_metrics.json"
DS_AUDIT_FILENAME = "ds_orientation_audit.json"
VALUE_AUDIT_FILENAME = "value_target_audit.json"
INPUT_ENCODING = "kalah_v3"
MODEL_TYPE = "residual_v3"
PRIMARY_BUDGETS = ("384:256", "768:768", "1200:1200", "1200:256")
PROBE_BUDGETS = ("384:256", "768:768", "1200:1200", "1200:256")
SUITE_BUDGETS = ("384:256", "768:256", "768:768", "1200:1200", "1200:256", "256:768")
PROBE_ENTROPY_FLOOR = 1.0
BOOTSTRAP_SAMPLES = 10000
ROOT_VALUE_SAMPLE_ROWS = 4096


@dataclass(frozen=True)
class LaneSpec:
    name: str
    epochs: int
    value_target_mode: str
    trainable_scope: str = "all"
    init_checkpoint_mode: str | None = None


LANE_SPECS = {
    "outcome_repro": LaneSpec(
        name="scratch_96x3_outcome_value_repro",
        epochs=2,
        value_target_mode="stored_outcome",
    ),
    "teacher_root_e2": LaneSpec(
        name="scratch_96x3_teacher_root_value_e2",
        epochs=2,
        value_target_mode="teacher_root",
    ),
    "teacher_root_e4": LaneSpec(
        name="scratch_96x3_teacher_root_value_e4",
        epochs=4,
        value_target_mode="teacher_root",
    ),
    "blend_root75_outcome25": LaneSpec(
        name="scratch_96x3_blend_root75_outcome25_e2",
        epochs=2,
        value_target_mode="blend_root75_outcome25",
    ),
    "current_init_policy_only": LaneSpec(
        name="current_init_96x3_policy_only_e2",
        epochs=2,
        value_target_mode="stored_outcome",
        trainable_scope="policy_head",
        init_checkpoint_mode="current",
    ),
}


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def log_progress(message: str) -> None:
    print(f"[value-ablation] {message}", flush=True)


def require_existing_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")


def parse_csv_paths(text: str) -> list[Path]:
    return [Path(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_csv_values(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    if np.std(x_arr) <= 1e-12 or np.std(y_arr) <= 1e-12:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def summarize_distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "max": None,
            "negative": 0,
            "zero": 0,
            "positive": 0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25.0)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75.0)),
        "max": float(np.max(arr)),
        "negative": int(np.sum(arr < 0.0)),
        "zero": int(np.sum(arr == 0.0)),
        "positive": int(np.sum(arr > 0.0)),
    }


def orientation_bundle(diffs: list[float], *, seed: int) -> dict[str, Any]:
    return bootstrap_ci(diffs, seed=seed, samples=BOOTSTRAP_SAMPLES)


def discover_pr151_student(
    summary: dict[str, Any], pr151_workdir: Path
) -> dict[str, Any]:
    candidates = summary.get("candidates", {})
    candidate = candidates.get("residual_v3_96x3")
    if not isinstance(candidate, dict):
        raise RuntimeError(
            "PR #151 summary is missing residual_v3_96x3 candidate entry"
        )
    artifact_dir = Path(str(candidate.get("artifact_dir", "")))
    checkpoint_path = None
    for checkpoint in candidate.get("checkpoints", []):
        if int(checkpoint.get("epoch", 0)) == 2:
            checkpoint_path = Path(str(checkpoint.get("checkpoint_path")))
            break
    if checkpoint_path is None:
        lane_dir = pr151_workdir / "residual_v3_96x3"
        fallback = lane_dir / "checkpoint_epoch2.npz"
        checkpoint_path = fallback if fallback.is_file() else None
    require_existing_file(artifact_dir / "weights.json", "PR #151 artifact weights")
    return {
        "name": "pr151_student_ref",
        "artifact_dir": str(artifact_dir),
        "artifact_weights_sha256": sha256_file(artifact_dir / "weights.json"),
        "checkpoint_path": str(checkpoint_path)
        if checkpoint_path is not None
        else None,
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path else None,
        "report_name": artifact_dir.name,
    }


def build_probe_rows_cached(
    *,
    workdir: Path,
    current_artifact: Path,
    medium_suite: Path,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
) -> list[dict[str, Any]]:
    probe_path = workdir / "medium_probe_rows.jsonl"
    if probe_path.is_file():
        return read_jsonl(probe_path)
    evaluator = ArtifactEvaluator(current_artifact)
    rows = build_probe_rows(
        evaluator=evaluator,
        suite_paths=[medium_suite],
        current_artifact=current_artifact,
        default_c_puct=default_c_puct,
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=tactical_root_bias,
        max_trajectory_plies=12,
        seed=seed,
    )
    probe_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return rows


def infer_candidate_report_name(artifact_path: Path) -> str:
    return artifact_path.name


def suite_eval_candidates_csv(candidate_artifacts: list[tuple[str, Path]]) -> str:
    return ",".join(str(path) for _name, path in candidate_artifacts)


def matching_candidate_report(
    *,
    report: dict[str, Any],
    candidate_name: str,
    artifact_path: Path,
    current_artifact: Path,
) -> dict[str, Any] | None:
    expected_sha = sha256_file(artifact_path / "weights.json")
    lookup_names = [
        "current"
        if candidate_name == "current_ref"
        else infer_candidate_report_name(artifact_path),
        candidate_name,
        artifact_path.name,
    ]
    for lookup_name in lookup_names:
        candidate_report = find_candidate_report(report, lookup_name)
        if candidate_report is None:
            continue
        candidate_sha = str(candidate_report.get("candidate_sha256", ""))
        candidate_path = str(candidate_report.get("candidate_path", ""))
        if candidate_sha and candidate_sha != expected_sha:
            continue
        if candidate_path and candidate_path not in {
            str(artifact_path),
            str(current_artifact),
        }:
            continue
        return candidate_report
    return None


def run_suite_evaluations(
    *,
    workdir: Path,
    current_artifact: Path,
    candidate_artifacts: list[tuple[str, Path]],
    suite_paths: list[Path],
    budget_pairs: str,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    suite_rows: dict[str, Any] = {}
    candidates_csv = suite_eval_candidates_csv(candidate_artifacts)
    for suite_path in suite_paths:
        suite_name = suite_path.stem
        suite_workdir = workdir / "suite_eval" / suite_name
        suite_workdir.mkdir(parents=True, exist_ok=True)
        report_path = suite_workdir / "temperature_benchmark_report.json"
        report = None
        if report_path.is_file():
            cached = load_json(report_path)
            all_present = True
            for candidate_name, artifact_path in candidate_artifacts:
                if (
                    matching_candidate_report(
                        report=cached,
                        candidate_name=candidate_name,
                        artifact_path=artifact_path,
                        current_artifact=current_artifact,
                    )
                    is None
                ):
                    all_present = False
                    break
            if all_present:
                report = cached
                log_progress(f"reusing suite benchmark suite={suite_name}")
        if report is None:
            log_progress(f"running suite benchmark suite={suite_name}")
            report = run_opening_suite_benchmark(
                workdir=str(suite_workdir),
                suite=str(suite_path),
                current=str(current_artifact),
                candidates=candidates_csv,
                budget_pairs=budget_pairs,
                games_per_opening=2,
                seed=seed,
                workers=workers,
                timeout=7200,
            )
        suite_rows[suite_name] = {
            "suite_name": suite_name,
            "suite_path": str(suite_path),
            "suite_sha256": sha256_file(suite_path),
            "suite_size": sum(
                1
                for line in suite_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ),
            "candidates": {},
        }
        for candidate_name, artifact_path in candidate_artifacts:
            candidate_report = matching_candidate_report(
                report=report,
                candidate_name=candidate_name,
                artifact_path=artifact_path,
                current_artifact=current_artifact,
            )
            if candidate_report is None:
                raise RuntimeError(
                    f"missing benchmark report for {candidate_name} in {suite_name}"
                )
            suite_rows[suite_name]["candidates"][candidate_name] = {
                "candidate_path": str(artifact_path),
                "candidate_sha256": candidate_report.get("candidate_sha256"),
                "budget_results": benchmark_budget_results(candidate_report),
            }
    return suite_rows


def aggregate_suite_summary(
    suite_rows: dict[str, Any], candidate_names: list[str]
) -> dict[str, Any]:
    heldout_names = [name for name in suite_rows if name.startswith("heldout_seed")]
    summary: dict[str, Any] = {}
    for candidate_name in candidate_names:
        candidate_summary = {
            "fixed_large": {},
            "heldout": {},
            "p0_p1_384_256": {},
            "duplicate_trajectory_count": 0.0,
        }
        fixed_large = suite_rows["large_eval"]["candidates"][candidate_name][
            "budget_results"
        ]
        candidate_summary["fixed_large"] = fixed_large
        duplicates = []
        p0_values = []
        p1_values = []
        for budget in PRIMARY_BUDGETS:
            values = [
                float(
                    suite_rows[suite_name]["candidates"][candidate_name][
                        "budget_results"
                    ][budget]["ds"]
                )
                for suite_name in heldout_names
            ]
            candidate_summary["heldout"][budget] = {
                "mean_ds": float(statistics.fmean(values)) if values else 0.0,
                "worst_suite_ds": min(values) if values else 0.0,
            }
        for suite_name in heldout_names:
            budget_result = suite_rows[suite_name]["candidates"][candidate_name][
                "budget_results"
            ]["384:256"]
            p0_values.append(float(budget_result["p0_score"]))
            p1_values.append(float(budget_result["p1_score"]))
            duplicates.append(
                float(budget_result.get("duplicate_trajectory_count") or 0.0)
            )
        candidate_summary["p0_p1_384_256"] = {
            "mean_p0": float(statistics.fmean(p0_values)) if p0_values else 0.0,
            "mean_p1": float(statistics.fmean(p1_values)) if p1_values else 0.0,
            "gap": (
                float(statistics.fmean(p1_values)) - float(statistics.fmean(p0_values))
            )
            if p0_values
            else 0.0,
        }
        candidate_summary["duplicate_trajectory_count"] = (
            float(statistics.fmean(duplicates)) if duplicates else 0.0
        )
        summary[candidate_name] = candidate_summary
    return summary


def build_ds_orientation_audit(
    *,
    suite_rows: dict[str, Any],
    summary_candidate_name: str,
    report_doc_path: Path,
    seed: int,
) -> dict[str, Any]:
    heldout_only = {
        name: row for name, row in suite_rows.items() if name.startswith("heldout_seed")
    }
    fixed_large = suite_rows["large_eval"]
    audit: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "candidate": summary_candidate_name,
        "fixed_large": {},
        "heldout": {},
        "report_consistency": {},
        "passed": True,
        "warnings": [],
    }
    for budget in PRIMARY_BUDGETS:
        candidate_fixed = float(
            fixed_large["candidates"][summary_candidate_name]["budget_results"][budget][
                "ds"
            ]
        )
        current_fixed = float(
            fixed_large["candidates"]["current_ref"]["budget_results"][budget]["ds"]
        )
        fixed_candidate_minus_current = candidate_fixed - current_fixed
        fixed_current_minus_candidate = current_fixed - candidate_fixed
        ds_diffs = pooled_per_opening_differences(
            suite_rows=heldout_only,
            candidate_a=summary_candidate_name,
            candidate_b="current_ref",
            budget_pair=budget,
            metric_key="ds",
        )
        score_diffs = pooled_per_opening_differences(
            suite_rows=heldout_only,
            candidate_a=summary_candidate_name,
            candidate_b="current_ref",
            budget_pair=budget,
            metric_key="disadvantaged_seat_score",
        )
        audit["fixed_large"][budget] = {
            "current_ref_ds": current_fixed,
            "candidate_ds": candidate_fixed,
            "candidate_minus_current": fixed_candidate_minus_current,
            "current_minus_candidate": fixed_current_minus_candidate,
        }
        audit["heldout"][budget] = {
            "candidate_minus_current": orientation_bundle(ds_diffs, seed=seed),
            "current_minus_candidate": orientation_bundle(
                [-value for value in ds_diffs], seed=seed
            ),
            "candidate_minus_current_disadvantaged_score": orientation_bundle(
                score_diffs, seed=seed
            ),
            "current_minus_candidate_disadvantaged_score": orientation_bundle(
                [-value for value in score_diffs], seed=seed
            ),
        }

    if report_doc_path.is_file():
        report_text = report_doc_path.read_text(encoding="utf-8")
        heldout_delta = audit["heldout"]["384:256"]["candidate_minus_current"]["mean"]
        disadvantaged_ci = audit["heldout"]["384:256"][
            "candidate_minus_current_disadvantaged_score"
        ]["mean"]
        uses_negative_ds_ci = (
            "residual_v3_96x3_minus_current_ref_384_256 | -0." in report_text
        )
        mentions_positive_delta = (
            "| residual_v3_96x3 | -0.0026 | +0.3025 |" in report_text
        )
        inferred = "unknown"
        if uses_negative_ds_ci and mentions_positive_delta:
            inferred = "mixed_metrics_or_inverted_labels"
        elif str(round(heldout_delta, 4)) in report_text:
            inferred = "candidate_minus_current_ds"
        elif str(round(-heldout_delta, 4)) in report_text:
            inferred = "current_minus_candidate_ds"
        elif str(round(disadvantaged_ci, 4)) in report_text:
            inferred = "candidate_minus_current_disadvantaged_score"
        elif str(round(-disadvantaged_ci, 4)) in report_text:
            inferred = "current_minus_candidate_disadvantaged_score"
        audit["report_consistency"] = {
            "source": str(report_doc_path),
            "inferred_existing_report_orientation": inferred,
            "doc_contains_positive_heldout_delta_row": mentions_positive_delta,
            "doc_contains_negative_ci_row": uses_negative_ds_ci,
        }
        if inferred == "mixed_metrics_or_inverted_labels":
            audit["passed"] = False
            audit["warnings"].append(
                "Existing PR #151 report mixes DS deltas with disadvantaged-seat bootstrap signs; do not use it for promotion reasoning."
            )
    return audit


def build_value_target_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stored_values = [float(row.get("value_target", 0.0)) for row in rows]
    root_values = [
        float(row.get("teacher_root_value", 0.0))
        for row in rows
        if row.get("teacher_root_value") is not None
    ]
    final_outcomes = [
        float(row.get("final_outcome", 0.0))
        for row in rows
        if row.get("final_outcome") is not None
    ]
    by_phase: dict[str, dict[str, Any]] = {}
    by_budget: dict[str, dict[str, Any]] = {}
    by_seat: dict[str, dict[str, Any]] = {}
    for key_name, target in (
        ("phase", by_phase),
        ("budget_context", by_budget),
        ("seat_context", by_seat),
    ):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str(row.get(key_name, "unknown"))].append(row)
        for bucket_name, bucket_rows in sorted(buckets.items()):
            bucket_stored = [float(row.get("value_target", 0.0)) for row in bucket_rows]
            bucket_root = [
                float(row.get("teacher_root_value", 0.0)) for row in bucket_rows
            ]
            bucket_outcome = [
                float(row.get("final_outcome", 0.0))
                for row in bucket_rows
                if row.get("final_outcome") is not None
            ]
            target[bucket_name] = {
                "rows": len(bucket_rows),
                "stored_value_target": summarize_distribution(bucket_stored),
                "teacher_root_value": summarize_distribution(bucket_root),
                "final_outcome": summarize_distribution(bucket_outcome),
                "corr_stored_vs_root": correlation(bucket_stored, bucket_root),
                "corr_stored_vs_outcome": correlation(bucket_stored, bucket_outcome)
                if len(bucket_stored) == len(bucket_outcome)
                else None,
            }
    return {
        "schema": SUMMARY_SCHEMA,
        "row_count": len(rows),
        "stored_value_target_distribution": summarize_distribution(stored_values),
        "teacher_root_value_distribution": summarize_distribution(root_values),
        "final_outcome_distribution": summarize_distribution(final_outcomes),
        "corr_stored_value_target_vs_teacher_root_value": correlation(
            stored_values, root_values
        ),
        "corr_stored_value_target_vs_final_outcome": correlation(
            stored_values, final_outcomes
        )
        if len(stored_values) == len(final_outcomes)
        else None,
        "teacher_root_value_recompute": {
            "stored_in_dataset": True,
            "sample_rows": min(ROOT_VALUE_SAMPLE_ROWS, len(rows)),
            "recompute_required": False,
        },
        "by_phase": by_phase,
        "by_budget_context": by_budget,
        "by_seat": by_seat,
    }


def clone_rows_with_value_targets(
    rows: list[dict[str, Any]], mode: str
) -> list[dict[str, Any]]:
    cloned: list[dict[str, Any]] = []
    for row in rows:
        cloned_row = dict(row)
        stored_value = float(row.get("value_target", 0.0))
        root_value = float(row.get("teacher_root_value", 0.0))
        if mode == "stored_outcome":
            target = stored_value
            source = "stored_outcome_value"
        elif mode == "teacher_root":
            target = root_value
            source = "teacher_root_value"
        elif mode == "blend_root75_outcome25":
            target = (0.75 * root_value) + (0.25 * stored_value)
            source = "blend_root75_outcome25"
        else:
            raise ValueError(f"unsupported value target mode: {mode}")
        cloned_row["value_target"] = float(max(-1.0, min(1.0, target)))
        cloned_row["value_target_mode"] = mode
        cloned_row["value_target_source"] = source
        cloned.append(cloned_row)
    return cloned


def train_student_candidate(
    *,
    spec: StudentSpec,
    candidate_name: str,
    train_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    workdir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    grad_clip: float,
    seed: int,
    device: torch.device,
    init_checkpoint: Path | None,
    trainable_scope: str,
) -> dict[str, Any]:
    lane_dir = workdir / candidate_name
    lane_dir.mkdir(parents=True, exist_ok=True)
    model = PolicyValueNet(
        hidden_sizes=spec.hidden_sizes,
        model_type=spec.model_type,
        input_size=input_size_for_encoding(INPUT_ENCODING),
    ).to(device)
    if init_checkpoint is not None:
        load_checkpoint_into_model(model, init_checkpoint)
    apply_trainable_scope(model, trainable_scope)
    optimizer = torch.optim.Adam(
        (param for param in model.parameters() if param.requires_grad), lr=lr
    )
    x = torch.from_numpy(
        np.asarray([row["state"] for row in train_rows], dtype=np.float32)
    ).to(device)
    p = torch.from_numpy(
        np.asarray([row["teacher_puct_policy"] for row in train_rows], dtype=np.float32)
    ).to(device)
    v = torch.from_numpy(
        np.asarray([[row["value_target"]] for row in train_rows], dtype=np.float32)
    ).to(device)
    legal_mask = torch.from_numpy(
        np.asarray([row["legal_mask"] for row in train_rows], dtype=np.float32)
    ).to(device)
    training_metrics: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    best_probe_metrics: dict[str, Any] | None = None
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(x.size(0), device=device)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        total_losses: list[float] = []
        grad_norms: list[float] = []
        model.train()
        for start in range(0, x.size(0), batch_size):
            idx = perm[start : start + batch_size]
            xb = x[idx]
            pb = p[idx]
            vb = v[idx]
            maskb = legal_mask[idx]
            logits, values = model(xb)
            policy_loss = masked_cross_entropy(logits, pb, maskb).mean()
            value_loss = compute_value_loss_vector(
                values,
                vb,
                value_loss=DEFAULT_VALUE_LOSS,
                huber_delta=DEFAULT_HUBER_DELTA,
            ).mean()
            total_loss = policy_loss + value_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_sq = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    grad_sq += float(torch.sum(parameter.grad.detach() ** 2).item())
            grad_norms.append(float(math.sqrt(grad_sq)) if grad_sq > 0.0 else 0.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            policy_losses.append(float(policy_loss.detach().cpu().item()))
            value_losses.append(float(value_loss.detach().cpu().item()))
            total_losses.append(float(total_loss.detach().cpu().item()))
        checkpoint_path = lane_dir / f"checkpoint_epoch{epoch}.npz"
        np.savez(checkpoint_path, **checkpoint_from_model(model))
        artifact_dir = lane_dir / f"artifact_epoch{epoch}"
        export_checkpoint_artifact(
            checkpoint_path=checkpoint_path,
            export_dir=artifact_dir,
            version=f"{candidate_name}_epoch{epoch}",
            model_type=spec.model_type,
            input_encoding=INPUT_ENCODING,
            policy_loss=float(statistics.fmean(policy_losses))
            if policy_losses
            else 0.0,
            value_loss=float(statistics.fmean(value_losses)) if value_losses else 0.0,
        )
        probe_metrics = evaluate_probe_candidate_enhanced(
            artifact_path=artifact_dir,
            candidate_name=candidate_name,
            probe_rows=probe_rows,
        )
        best_probe_metrics = probe_metrics
        training_metrics.append(
            {
                "epoch": epoch,
                "policy_loss": float(statistics.fmean(policy_losses))
                if policy_losses
                else 0.0,
                "value_loss": float(statistics.fmean(value_losses))
                if value_losses
                else 0.0,
                "total_loss": float(statistics.fmean(total_losses))
                if total_losses
                else 0.0,
                "gradient_norm": float(statistics.fmean(grad_norms))
                if grad_norms
                else 0.0,
                "probe_top1_agreement": probe_metrics["top1_agreement"],
                "probe_policy_kl": probe_metrics["policy_kl"],
            }
        )
        checkpoints.append(
            {
                "epoch": epoch,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "artifact_dir": str(artifact_dir),
                "artifact_weights_sha256": sha256_file(artifact_dir / "weights.json"),
            }
        )
        log_progress(
            f"student {candidate_name} epoch={epoch} policy_loss={training_metrics[-1]['policy_loss']:.4f} "
            f"value_loss={training_metrics[-1]['value_loss']:.4f} probe_top1={probe_metrics['top1_agreement']:.4f}"
        )
    final_artifact = Path(checkpoints[-1]["artifact_dir"])
    return {
        "name": candidate_name,
        "kind": "student",
        "architecture": {
            "model_type": spec.model_type,
            "trunk_size": spec.trunk_size,
            "residual_block_count": spec.residual_block_count,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "grad_clip": grad_clip,
            "rows": len(train_rows),
            "seed": seed,
            "trainable_scope": trainable_scope,
            "init_checkpoint": str(init_checkpoint)
            if init_checkpoint is not None
            else None,
        },
        "training_metrics": training_metrics,
        "checkpoints": checkpoints,
        "artifact_dir": str(final_artifact),
        "artifact_weights_sha256": sha256_file(final_artifact / "weights.json"),
        "probe_metrics": best_probe_metrics,
    }


def _init_metric_bucket() -> dict[str, float]:
    return {
        "rows": 0.0,
        "top1_matches": 0.0,
        "search_matches": 0.0,
        "legal_failures": 0.0,
        "policy_kl_total": 0.0,
        "entropy_total": 0.0,
        "outcome_mae_total": 0.0,
        "root_mae_total": 0.0,
        "outcome_sign_total": 0.0,
        "root_sign_total": 0.0,
    }


def _finalize_metric_bucket(bucket: dict[str, float]) -> dict[str, Any]:
    rows = max(int(bucket["rows"]), 1)
    return {
        "rows": int(bucket["rows"]),
        "top1_agreement": bucket["top1_matches"] / rows,
        "search_selected_move_agreement": bucket["search_matches"] / rows,
        "policy_kl": bucket["policy_kl_total"] / rows,
        "entropy": bucket["entropy_total"] / rows,
        "value_mae_vs_stored_outcome": bucket["outcome_mae_total"] / rows,
        "value_mae_vs_teacher_root": bucket["root_mae_total"] / rows,
        "value_sign_accuracy_vs_stored_outcome": bucket["outcome_sign_total"] / rows,
        "value_sign_accuracy_vs_teacher_root": bucket["root_sign_total"] / rows,
        "legal_failures": int(bucket["legal_failures"]),
    }


def evaluate_probe_candidate_enhanced(
    *, artifact_path: Path, candidate_name: str, probe_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    evaluator = ArtifactEvaluator(artifact_path)
    totals = _init_metric_bucket()
    budget_buckets: dict[str, dict[str, float]] = defaultdict(_init_metric_bucket)
    phase_buckets: dict[str, dict[str, float]] = defaultdict(_init_metric_bucket)
    seat_buckets: dict[str, dict[str, float]] = defaultdict(_init_metric_bucket)
    for row in probe_rows:
        game = KalahGame.from_state(row["raw_state"])
        legal_moves = [
            move for move, flag in enumerate(row["legal_mask"]) if int(flag) == 1
        ]
        policy, value = evaluator.evaluate(game)
        student_policy = masked_policy(policy.tolist(), legal_moves)
        legal_failure = int(
            any(
                student_policy[move] > 1e-7
                for move in range(6)
                if move not in legal_moves
            )
        )
        teacher_top = top_policy_move(row["teacher_puct_policy"], legal_moves)
        student_top = top_policy_move(student_policy, legal_moves)
        search_result = evaluate_artifact_position(
            evaluator=evaluator,
            state=game.to_state(),
            simulations=int(row["challenger_simulations"]),
            seed=int(row["challenger_simulations"])
            + int(row["current_simulations"])
            + int(row.get("absolute_ply", 0)),
            c_puct=float(row["effective_c_puct"]),
            search_options=build_eval_search_options(
                root_policy_mode="deterministic", tactical_root_bias=0.0
            ),
            ablation_mode="full",
        )
        search_selected_move = search_result.get("selected_move")
        kl = kl_divergence(row["teacher_puct_policy"], student_policy, legal_moves)
        entropy = policy_entropy(student_policy, legal_moves)
        stored_outcome = float(row.get("value_target", 0.0))
        teacher_root = float(row.get("teacher_root_value", 0.0))
        outcome_sign_match = float((float(value) >= 0.0) == (stored_outcome >= 0.0))
        root_sign_match = float((float(value) >= 0.0) == (teacher_root >= 0.0))
        for bucket in (
            totals,
            budget_buckets[str(row["budget_context"])],
            phase_buckets[str(row["phase"])],
            seat_buckets[str(row["seat_context"])],
        ):
            bucket["rows"] += 1.0
            bucket["top1_matches"] += float(teacher_top == student_top)
            bucket["search_matches"] += float(
                search_selected_move == row.get("teacher_selected_move")
            )
            bucket["legal_failures"] += float(legal_failure)
            bucket["policy_kl_total"] += float(kl)
            bucket["entropy_total"] += float(entropy)
            bucket["outcome_mae_total"] += abs(float(value) - stored_outcome)
            bucket["root_mae_total"] += abs(float(value) - teacher_root)
            bucket["outcome_sign_total"] += outcome_sign_match
            bucket["root_sign_total"] += root_sign_match
    metrics = _finalize_metric_bucket(totals)
    metrics.update(
        {
            "candidate": candidate_name,
            "policy_kl": metrics["policy_kl"],
            "mean_output_entropy": metrics["entropy"],
            "top1_agreement": metrics["top1_agreement"],
            "legal_mask_failures": metrics["legal_failures"],
            "value_mae": metrics["value_mae_vs_stored_outcome"],
            "value_sign_accuracy": metrics["value_sign_accuracy_vs_stored_outcome"],
            "value_root_mae": metrics["value_mae_vs_teacher_root"],
            "value_root_sign_accuracy": metrics["value_sign_accuracy_vs_teacher_root"],
            "search_selected_move_agreement": metrics["search_selected_move_agreement"],
            "budget_metrics": {
                key: _finalize_metric_bucket(bucket)
                for key, bucket in sorted(budget_buckets.items())
            },
            "phase_metrics": {
                key: _finalize_metric_bucket(bucket)
                for key, bucket in sorted(phase_buckets.items())
            },
            "seat_metrics": {
                key: _finalize_metric_bucket(bucket)
                for key, bucket in sorted(seat_buckets.items())
            },
        }
    )
    return metrics


def probe_abort_reasons(
    *,
    metrics: dict[str, Any],
    current_probe: dict[str, Any],
    pr151_probe: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if int(metrics.get("legal_mask_failures", 0)) > 0:
        reasons.append("legal failures > 0")
    if float(metrics.get("top1_agreement", 0.0)) < 0.60:
        reasons.append("teacher top-1 agreement < 0.60")
    if float(metrics.get("policy_kl", 0.0)) > float(pr151_probe.get("policy_kl", 0.0)):
        reasons.append("KL is worse than PR #151 student")
    current_root_mae = float(
        current_probe.get(
            "value_root_mae", current_probe.get("value_mae_vs_teacher_root", 0.0)
        )
    )
    if float(metrics.get("value_root_mae", 0.0)) > (current_root_mae * 1.10):
        reasons.append("value root MAE is worse than current_ref by more than 10%")
    if float(metrics.get("mean_output_entropy", 0.0)) < PROBE_ENTROPY_FLOOR:
        reasons.append("entropy collapses below 1.0")
    for budget in PRIMARY_BUDGETS:
        budget_metrics = metrics.get("budget_metrics", {}).get(budget, {})
        if budget_metrics and float(budget_metrics.get("top1_agreement", 1.0)) < 0.50:
            reasons.append(f"{budget} top-1 agreement < 0.50")
    return reasons


def should_run_gate(
    candidate_name: str, suite_summary: dict[str, Any]
) -> tuple[bool, str | None]:
    if candidate_name == "current_ref":
        return False, "n/a"
    candidate = suite_summary[candidate_name]["heldout"]
    current = suite_summary["current_ref"]["heldout"]
    delta_384 = float(candidate["384:256"]["mean_ds"]) - float(
        current["384:256"]["mean_ds"]
    )
    delta_768 = float(candidate["768:768"]["mean_ds"]) - float(
        current["768:768"]["mean_ds"]
    )
    delta_1200 = float(candidate["1200:1200"]["mean_ds"]) - float(
        current["1200:1200"]["mean_ds"]
    )
    delta_1200_256 = float(candidate["1200:256"]["mean_ds"]) - float(
        current["1200:256"]["mean_ds"]
    )
    if (
        delta_384 >= 0.05
        and delta_768 >= -0.05
        and delta_1200 >= -0.03
        and delta_1200_256 >= -0.03
    ):
        return True, None
    return False, "did not clear audited held-out robustness gate for explicit gate run"


def run_gate_if_eligible(
    *,
    workdir: Path,
    current_artifact: Path,
    candidate_name: str,
    candidate_path: Path,
    suite_summary: dict[str, Any],
    workers: int,
    seed: int,
) -> dict[str, Any]:
    eligible, reason = should_run_gate(candidate_name, suite_summary)
    if not eligible:
        return {"ran": False, "reason": reason}
    out_path = workdir / "gate" / f"{candidate_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = run_default_gate(
        candidate_path=str(candidate_path),
        current_path=str(current_artifact),
        out=str(out_path),
        seed=seed,
        workers=workers,
        games=120,
        budget_pairs=DEFAULT_GATE_BUDGETS,
    )
    return {
        "ran": True,
        "report_path": str(out_path),
        "classification": report.get("classification"),
        "preserved": candidate_gate_preserved(report),
    }


def compute_bootstrap_tables(
    suite_rows: dict[str, Any], candidate_names: list[str], seed: int
) -> dict[str, Any]:
    heldout_only = {
        name: row for name, row in suite_rows.items() if name.startswith("heldout_seed")
    }
    tables: dict[str, Any] = {}
    for candidate_name in candidate_names:
        if candidate_name == "current_ref":
            continue
        per_budget: dict[str, Any] = {}
        for budget in PRIMARY_BUDGETS:
            ds_diffs = pooled_per_opening_differences(
                suite_rows=heldout_only,
                candidate_a=candidate_name,
                candidate_b="current_ref",
                budget_pair=budget,
                metric_key="ds",
            )
            per_budget[budget] = {
                "candidate_minus_current": bootstrap_ci(
                    ds_diffs, seed=seed, samples=BOOTSTRAP_SAMPLES
                ),
                "current_minus_candidate": bootstrap_ci(
                    [-value for value in ds_diffs], seed=seed, samples=BOOTSTRAP_SAMPLES
                ),
            }
        tables[candidate_name] = per_budget
    return tables


def final_classification(
    *,
    ds_audit: dict[str, Any],
    suite_summary: dict[str, Any],
    bootstrap_tables: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    gate_results: dict[str, Any],
) -> str:
    if not ds_audit.get("passed", False):
        return "metric_reporting_bug"
    for candidate_name in (
        "scratch_96x3_teacher_root_value_e2",
        "scratch_96x3_teacher_root_value_e4",
        "scratch_96x3_blend_root75_outcome25_e2",
    ):
        if candidate_name not in suite_summary:
            continue
        current_heldout = suite_summary["current_ref"]["heldout"]
        candidate_heldout = suite_summary[candidate_name]["heldout"]
        if (
            float(candidate_heldout["384:256"]["mean_ds"])
            - float(current_heldout["384:256"]["mean_ds"])
            >= 0.05
            and float(candidate_heldout["768:768"]["mean_ds"])
            - float(current_heldout["768:768"]["mean_ds"])
            >= -0.05
            and float(candidate_heldout["1200:1200"]["mean_ds"])
            - float(current_heldout["1200:1200"]["mean_ds"])
            >= -0.03
            and float(candidate_heldout["1200:256"]["mean_ds"])
            - float(current_heldout["1200:256"]["mean_ds"])
            >= -0.03
        ):
            return "value_target_was_primary_issue"
    for candidate_name, per_budget in bootstrap_tables.items():
        heldout = suite_summary[candidate_name]["heldout"]
        current = suite_summary["current_ref"]["heldout"]
        if (
            float(heldout["384:256"]["mean_ds"]) - float(current["384:256"]["mean_ds"])
            >= 0.05
            and float(per_budget["384:256"]["candidate_minus_current"]["lower"]) > 0.01
            and float(heldout["768:768"]["mean_ds"])
            - float(current["768:768"]["mean_ds"])
            >= -0.05
            and float(heldout["1200:1200"]["mean_ds"])
            - float(current["1200:1200"]["mean_ds"])
            >= -0.03
            and float(heldout["1200:256"]["mean_ds"])
            - float(current["1200:256"]["mean_ds"])
            >= -0.03
            and gate_results.get(candidate_name, {}).get("preserved") is True
        ):
            return "policy_distillation_runtime_candidate"
    gains = []
    regressions = []
    for candidate_name, entry in suite_summary.items():
        if candidate_name == "current_ref":
            continue
        current = suite_summary["current_ref"]["heldout"]
        gains.append(
            float(entry["heldout"]["384:256"]["mean_ds"])
            - float(current["heldout"]["384:256"]["mean_ds"])
        )
        regressions.append(
            min(
                float(entry["heldout"]["768:768"]["mean_ds"])
                - float(current["heldout"]["768:768"]["mean_ds"]),
                float(entry["heldout"]["1200:1200"]["mean_ds"])
                - float(current["heldout"]["1200:1200"]["mean_ds"]),
                float(entry["heldout"]["1200:256"]["mean_ds"])
                - float(current["heldout"]["1200:256"]["mean_ds"]),
            )
        )
    if (
        gains
        and any(gain >= 0.05 for gain in gains)
        and any(reg < -0.03 for reg in regressions)
    ):
        return "distillation_tradeoff_confirmed"
    return "policy_teacher_objective_not_enough"


def build_markdown_report(summary: dict[str, Any]) -> str:
    ds_audit = summary["ds_orientation_audit"]
    candidates = summary["candidates"]
    suite_summary = summary.get("suite_summary", {})
    bootstrap_tables = summary.get("bootstrap_tables", {})
    training_rows = []
    lane_rows = []
    probe_rows = []
    aborted_rows = []
    fixed_large_rows = []
    heldout_rows = []
    p0p1_rows = []
    bootstrap_rows = []
    for candidate_name, candidate in candidates.items():
        lane_rows.append(
            [
                candidate_name,
                candidate.get("kind"),
                candidate.get("architecture", {}).get("model_type"),
                candidate.get("architecture", {}).get("trunk_size"),
                candidate.get("architecture", {}).get("residual_block_count"),
                candidate.get("artifact_weights_sha256"),
            ]
        )
        for metric in candidate.get("training_metrics", []):
            training_rows.append(
                [
                    candidate_name,
                    metric.get("epoch"),
                    fmt(metric.get("policy_loss")),
                    fmt(metric.get("value_loss")),
                    fmt(metric.get("total_loss")),
                    fmt(metric.get("gradient_norm")),
                ]
            )
        probe = candidate.get("probe_metrics", {})
        if probe:
            probe_rows.append(
                [
                    candidate_name,
                    fmt(probe.get("top1_agreement")),
                    fmt(probe.get("policy_kl")),
                    fmt(probe.get("mean_output_entropy")),
                    fmt(
                        probe.get("value_mae_vs_stored_outcome", probe.get("value_mae"))
                    ),
                    fmt(
                        probe.get(
                            "value_mae_vs_teacher_root", probe.get("value_root_mae")
                        )
                    ),
                    fmt(probe.get("search_selected_move_agreement")),
                    int(probe.get("legal_mask_failures", 0)),
                ]
            )
        if candidate.get("aborted"):
            aborted_rows.append(
                [candidate_name, "; ".join(candidate.get("abort_reasons", []))]
            )
    for candidate_name, candidate_summary in suite_summary.items():
        fixed = candidate_summary["fixed_large"]
        fixed_large_rows.append(
            [candidate_name, *[fmt(fixed[budget]["ds"]) for budget in SUITE_BUDGETS]]
        )
        heldout = candidate_summary["heldout"]
        heldout_rows.append(
            [
                candidate_name,
                fmt(heldout["384:256"]["mean_ds"]),
                fmt(heldout["384:256"]["worst_suite_ds"]),
                fmt(heldout["768:768"]["mean_ds"]),
                fmt(heldout["1200:1200"]["mean_ds"]),
                fmt(heldout["1200:256"]["mean_ds"]),
            ]
        )
        p0p1 = candidate_summary["p0_p1_384_256"]
        p0p1_rows.append(
            [
                candidate_name,
                fmt(p0p1["mean_p0"]),
                fmt(p0p1["mean_p1"]),
                fmt(p0p1["gap"]),
            ]
        )
    for candidate_name, budgets in bootstrap_tables.items():
        for budget in PRIMARY_BUDGETS:
            bootstrap_rows.append(
                [
                    f"{candidate_name} {budget}",
                    fmt(budgets[budget]["candidate_minus_current"]["mean"]),
                    fmt(budgets[budget]["candidate_minus_current"]["lower"]),
                    fmt(budgets[budget]["candidate_minus_current"]["upper"]),
                    fmt(budgets[budget]["current_minus_candidate"]["mean"]),
                    fmt(budgets[budget]["current_minus_candidate"]["lower"]),
                    fmt(budgets[budget]["current_minus_candidate"]["upper"]),
                ]
            )
    lines = [
        "# AlphaZero-Lite Search-Teacher Distillation Value Ablation Results",
        "",
        f"**Classification**: `{summary['classification']}`",
        "",
        "## Current Artifact Hash",
        "",
        f"- current weights SHA256: `{summary['current_hash']['actual_sha256']}`",
        "",
        "## PR #151 Artifact And Dataset Hashes",
        "",
        f"- PR #151 student artifact SHA256: `{summary['pr151_student']['artifact_weights_sha256']}`",
        f"- teacher dataset SHA256: `{summary['teacher_dataset_sha256']}`",
        f"- teacher dataset audit SHA256: `{summary['teacher_dataset_audit_sha256']}`",
        "",
        "## DS Orientation Audit",
        "",
        f"- audit passed: `{ds_audit['passed']}`",
        f"- inferred existing report orientation: `{ds_audit.get('report_consistency', {}).get('inferred_existing_report_orientation', 'unknown')}`",
        *[f"- warning: `{warning}`" for warning in ds_audit.get("warnings", [])],
        "",
        "## Value Target Audit",
        "",
        f"- stored value target distribution: `{json.dumps(summary['value_target_audit']['stored_value_target_distribution'], sort_keys=True)}`",
        f"- teacher root value distribution: `{json.dumps(summary['value_target_audit']['teacher_root_value_distribution'], sort_keys=True)}`",
        f"- final outcome distribution: `{json.dumps(summary['value_target_audit']['final_outcome_distribution'], sort_keys=True)}`",
        f"- corr(stored, root): `{fmt(summary['value_target_audit']['corr_stored_value_target_vs_teacher_root_value'])}`",
        f"- corr(stored, outcome): `{fmt(summary['value_target_audit']['corr_stored_value_target_vs_final_outcome'])}`",
        "",
        "## Candidate Lane Table",
        "",
        markdown_table(
            ["Candidate", "Kind", "Model", "Trunk", "Blocks", "Weights SHA256"],
            lane_rows,
        ),
        "",
        "## Training Loss Table",
        "",
        markdown_table(
            [
                "Candidate",
                "Epoch",
                "Policy loss",
                "Value loss",
                "Total loss",
                "Grad norm",
            ],
            training_rows,
        ),
        "",
        "## Probe Metrics Table",
        "",
        markdown_table(
            [
                "Candidate",
                "Top-1",
                "KL",
                "Entropy",
                "Outcome MAE",
                "Root MAE",
                "Search agree",
                "Legal fails",
            ],
            probe_rows,
        ),
        "",
        "## Aborted-Candidate Table",
        "",
        markdown_table(["Candidate", "Reasons"], aborted_rows or [["none", ""]]),
        "",
        "## Fixed Large DS Table",
        "",
        markdown_table(["Candidate", *SUITE_BUDGETS], fixed_large_rows),
        "",
        "## Held-Out Mean/Worst-Suite DS Table",
        "",
        markdown_table(
            [
                "Candidate",
                "Mean 384:256",
                "Worst 384:256",
                "Mean 768:768",
                "Mean 1200:1200",
                "Mean 1200:256",
            ],
            heldout_rows,
        ),
        "",
        "## Bootstrap CIs",
        "",
        markdown_table(
            [
                "Comparison",
                "Cand-Cur mean",
                "Lower",
                "Upper",
                "Cur-Cand mean",
                "Lower",
                "Upper",
            ],
            bootstrap_rows,
        ),
        "",
        "## P0/P1 Split For 384:256",
        "",
        markdown_table(["Candidate", "Mean P0", "Mean P1", "Gap"], p0p1_rows),
        "",
        "## Duplicate Trajectory Count",
        "",
        markdown_table(
            ["Candidate", "Mean duplicates"],
            [
                [name, fmt(entry["duplicate_trajectory_count"])]
                for name, entry in suite_summary.items()
            ],
        ),
        "",
        "## Gate Classification If Run",
        "",
        *[
            f"- {name}: `{result.get('classification', 'not_run') if result.get('ran') else 'not_run'}`"
            + (f" reason=`{result.get('reason')}`" if result.get("reason") else "")
            for name, result in sorted(summary.get("gate_results", {}).items())
        ],
        "",
        "## Final Recommendation",
        "",
        f"- result: `{summary['classification']}`",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--pr151-workdir", required=True)
    parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--teacher-dataset-audit", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--teacher-default-c-puct", type=float, default=1.25)
    parser.add_argument("--teacher-cpuct-schedule", default='{"768:768":0.90}')
    parser.add_argument("--teacher-tactical-root-bias", type=float, default=0.0)
    parser.add_argument(
        "--lanes",
        default="outcome_repro,teacher_root_e2,teacher_root_e4,blend_root75_outcome25,current_init_policy_only",
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    current_artifact = Path(args.current)
    pr151_workdir = Path(args.pr151_workdir)
    teacher_dataset = Path(args.teacher_dataset)
    teacher_dataset_audit = Path(args.teacher_dataset_audit)
    medium_suite = Path(args.medium_suite)
    fixed_large_suite = Path(args.fixed_large_suite)
    heldout_suites = parse_csv_paths(args.heldout_suites)
    lane_tokens = parse_csv_values(args.lanes)
    cpuct_schedule = parse_cpuct_schedule_json(args.teacher_cpuct_schedule)
    for path, label in (
        (teacher_dataset, "teacher dataset"),
        (teacher_dataset_audit, "teacher dataset audit"),
        (medium_suite, "medium suite"),
        (fixed_large_suite, "fixed large suite"),
    ):
        require_existing_file(path, label)
    for suite_path in heldout_suites:
        require_existing_file(suite_path, f"heldout suite {suite_path.name}")
    current_hash = verify_expected_hash(
        current_artifact / "weights.json",
        args.expected_current_weights_sha256,
        "current artifact weights",
    )
    pr151_summary = load_json(pr151_workdir / "summary_metrics.json")
    pr151_student = discover_pr151_student(pr151_summary, pr151_workdir)
    dataset_rows = read_jsonl(teacher_dataset)
    value_audit = build_value_target_audit(dataset_rows)
    write_json(workdir / VALUE_AUDIT_FILENAME, value_audit)
    audit_suite_paths = [fixed_large_suite, *heldout_suites]
    audit_candidate_artifacts = [
        ("current_ref", current_artifact),
        ("pr151_student_ref", Path(pr151_student["artifact_dir"])),
    ]
    ds_suite_rows = run_suite_evaluations(
        workdir=workdir / "phase_a_audit",
        current_artifact=current_artifact,
        candidate_artifacts=audit_candidate_artifacts,
        suite_paths=audit_suite_paths,
        budget_pairs=",".join(PRIMARY_BUDGETS),
        workers=args.workers,
        seed=args.seed,
    )
    ds_audit = build_ds_orientation_audit(
        suite_rows=ds_suite_rows,
        summary_candidate_name="pr151_student_ref",
        report_doc_path=REPO_ROOT
        / "docs/alphazero-lite-search-teacher-student-preflight-results.md",
        seed=args.seed,
    )
    write_json(workdir / DS_AUDIT_FILENAME, ds_audit)
    probe_rows = build_probe_rows_cached(
        workdir=workdir,
        current_artifact=current_artifact,
        medium_suite=medium_suite,
        default_c_puct=args.teacher_default_c_puct,
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=args.teacher_tactical_root_bias,
        seed=args.seed,
    )
    device = select_device("auto")
    set_seed(args.seed)
    spec = StudentSpec(
        name="residual_v3_96x3",
        model_type=MODEL_TYPE,
        trunk_size=96,
        residual_block_count=3,
    )
    candidates: dict[str, dict[str, Any]] = {
        "current_ref": {
            "name": "current_ref",
            "kind": "reference",
            "architecture": {
                "model_type": MODEL_TYPE,
                "trunk_size": 96,
                "residual_block_count": 3,
            },
            "artifact_dir": str(current_artifact),
            "artifact_weights_sha256": current_hash["actual_sha256"],
            "training_metrics": [],
            "checkpoints": [],
            "probe_metrics": evaluate_probe_candidate_enhanced(
                artifact_path=current_artifact,
                candidate_name="current_ref",
                probe_rows=probe_rows,
            ),
        },
        "pr151_student_ref": {
            "name": "pr151_student_ref",
            "kind": "reference",
            "architecture": {
                "model_type": MODEL_TYPE,
                "trunk_size": 96,
                "residual_block_count": 3,
            },
            "artifact_dir": str(pr151_student["artifact_dir"]),
            "artifact_weights_sha256": pr151_student["artifact_weights_sha256"],
            "training_metrics": [],
            "checkpoints": [],
            "probe_metrics": evaluate_probe_candidate_enhanced(
                artifact_path=Path(pr151_student["artifact_dir"]),
                candidate_name="pr151_student_ref",
                probe_rows=probe_rows,
            ),
        },
    }
    current_probe = candidates["current_ref"]["probe_metrics"]
    pr151_probe = candidates["pr151_student_ref"]["probe_metrics"]
    current_checkpoint = current_checkpoint_path(current_artifact, workdir)
    for lane_token in lane_tokens:
        lane = LANE_SPECS[lane_token]
        if lane_token == "teacher_root_e4":
            e2_candidate = candidates.get("scratch_96x3_teacher_root_value_e2")
            e2_probe = e2_candidate.get("probe_metrics") if e2_candidate else None
            if (
                not e2_probe
                or float(e2_probe.get("top1_agreement", 0.0)) < 0.60
                or float(e2_probe.get("mean_output_entropy", 0.0)) < 1.0
            ):
                log_progress(
                    "skipping teacher_root_e4 because teacher_root_e2 collapsed"
                )
                continue
        lane_rows = clone_rows_with_value_targets(dataset_rows, lane.value_target_mode)
        init_checkpoint = (
            current_checkpoint if lane.init_checkpoint_mode == "current" else None
        )
        candidate = train_student_candidate(
            spec=spec,
            candidate_name=lane.name,
            train_rows=lane_rows,
            probe_rows=probe_rows,
            workdir=workdir,
            epochs=lane.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            grad_clip=args.grad_clip,
            seed=args.seed,
            device=device,
            init_checkpoint=init_checkpoint,
            trainable_scope=lane.trainable_scope,
        )
        abort_reasons = probe_abort_reasons(
            metrics=candidate["probe_metrics"],
            current_probe=current_probe,
            pr151_probe=pr151_probe,
        )
        candidate["abort_reasons"] = abort_reasons
        candidate["aborted"] = bool(abort_reasons)
        candidates[candidate["name"]] = candidate
    candidate_artifacts = [
        (name, Path(entry["artifact_dir"]))
        for name, entry in candidates.items()
        if name in {"current_ref", "pr151_student_ref"}
        or not entry.get("aborted", False)
    ]
    suite_rows = run_suite_evaluations(
        workdir=workdir,
        current_artifact=current_artifact,
        candidate_artifacts=candidate_artifacts,
        suite_paths=[medium_suite, fixed_large_suite, *heldout_suites],
        budget_pairs=DEFAULT_SUITE_EVAL_BUDGETS,
        workers=args.workers,
        seed=args.seed,
    )
    suite_summary = aggregate_suite_summary(
        suite_rows, [name for name, _path in candidate_artifacts]
    )
    bootstrap_tables = compute_bootstrap_tables(
        suite_rows, [name for name, _path in candidate_artifacts], args.seed
    )
    gate_results = {
        name: run_gate_if_eligible(
            workdir=workdir,
            current_artifact=current_artifact,
            candidate_name=name,
            candidate_path=Path(candidates[name]["artifact_dir"]),
            suite_summary=suite_summary,
            workers=args.workers,
            seed=args.seed,
        )
        for name, _path in candidate_artifacts
        if name != "current_ref"
    }
    classification = final_classification(
        ds_audit=ds_audit,
        suite_summary=suite_summary,
        bootstrap_tables=bootstrap_tables,
        candidates=candidates,
        gate_results=gate_results,
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "inputs": {
            "workdir": str(workdir),
            "current": str(current_artifact),
            "pr151_workdir": str(pr151_workdir),
            "teacher_runtime_profile": {
                "artifact": str(current_artifact),
                "tactical_root_bias": float(args.teacher_tactical_root_bias),
                "default_c_puct": float(args.teacher_default_c_puct),
                "c_puct_schedule": schedule_definition(
                    default_c_puct=float(args.teacher_default_c_puct),
                    schedule=cpuct_schedule,
                ),
                "root_policy_mode": "deterministic",
                "root_prior_transform": None,
                "value_transform": None,
            },
            "workers": args.workers,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "grad_clip": args.grad_clip,
            "lanes": lane_tokens,
        },
        "current_hash": current_hash,
        "pr151_student": pr151_student,
        "teacher_dataset_sha256": sha256_file(teacher_dataset),
        "teacher_dataset_audit_sha256": sha256_file(teacher_dataset_audit),
        "ds_orientation_audit": ds_audit,
        "value_target_audit": value_audit,
        "candidates": candidates,
        "suite_rows": suite_rows,
        "suite_summary": suite_summary,
        "bootstrap_tables": bootstrap_tables,
        "gate_results": gate_results,
        "classification": classification,
    }
    write_json(workdir / SUMMARY_FILENAME, summary)
    report = build_markdown_report(summary)
    REPORT_PATH.write_text(report, encoding="utf-8")
    log_progress(f"wrote summary={workdir / SUMMARY_FILENAME}")
    log_progress(f"wrote report={REPORT_PATH}")


if __name__ == "__main__":
    main()
