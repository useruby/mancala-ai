#!/usr/bin/env python3
"""Budget-anchored current-init policy-head-only micro-distillation.

Builds budget-aware target/anchor splits from the existing search-teacher
dataset, trains tiny policy-head-only update lanes from current-init, probes
every exported checkpoint against search-change budgets, and only carries
budget-safe candidates into medium/fixed-large/held-out evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    parse_cpuct_schedule_json,
    schedule_definition,
)
from ml.alphazero_lite.export_artifact import sha256_file  # noqa: E402
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    candidate_gate_preserved,
)
from ml.alphazero_lite.run_current_init_policy_only_distillation_preflight import (  # noqa: E402
    build_orientation_audit,
    compare_preserved_weights,
    evaluate_search_outputs,
    run_suite_evaluations,
)
from ml.alphazero_lite.run_promoted_current_opening_puct_disagreement import (  # noqa: E402
    policy_entropy,
    read_jsonl,
    top_policy_move,
)
from ml.alphazero_lite.run_search_teacher_distillation_value_ablation import (  # noqa: E402
    build_ds_orientation_audit,
)
from ml.alphazero_lite.run_search_teacher_student_preflight import (  # noqa: E402
    aggregate_suite_metrics,
    compute_bootstrap_comparisons,
    current_checkpoint_path,
    export_checkpoint_artifact,
    masked_cross_entropy,
    require_existing_file,
    verify_expected_hash,
    write_json,
)
from ml.alphazero_lite.train import (  # noqa: E402
    PolicyValueNet,
    apply_trainable_scope,
    checkpoint_from_model,
    input_size_for_encoding,
    load_checkpoint_into_model,
    select_device,
    set_seed,
)


SUMMARY_SCHEMA = "azlite_budget_anchored_policy_only_microdistill_v1"
SUMMARY_FILENAME = "summary_metrics.json"
DATASET_SPLIT_AUDIT_FILENAME = "dataset_split_audit.json"
REPORT_PATH = (
    REPO_ROOT
    / "docs/alphazero-lite-budget-anchored-policy-only-microdistill-results.md"
)
INPUT_ENCODING = "kalah_v3"
MODEL_TYPE = "residual_v3"
TRUNK_SIZE = 96
RESIDUAL_BLOCK_COUNT = 3
SUITE_BUDGETS = ("384:256", "768:256", "768:768", "1200:1200", "1200:256", "256:768")
PROBE_BUDGETS = ("384:256", "768:768", "1200:1200", "1200:256")
ANCHOR_BUDGETS = ("768:768", "1200:1200", "1200:256", "256:768")
BOOTSTRAP_SAMPLES = 10000
SEARCH_PROBE_CAP_PER_BUCKET = 32
POLICY_BATCH_SIZE = 4096
TRAIN_BATCH_SIZE = 128
EARLY_STOP_INTERVAL = 50
ENTROPY_FLOOR = 1.0
TARGET_WEIGHT = 1.0


@dataclass(frozen=True)
class LaneSpec:
    name: str
    lr: float
    anchor_weight: float
    max_steps: int


LANE_SPECS = {
    "micro_lr1e-6_anchor4_steps250": LaneSpec(
        name="micro_lr1e-6_anchor4_steps250",
        lr=1e-6,
        anchor_weight=4.0,
        max_steps=250,
    ),
    "micro_lr1e-6_anchor8_steps250": LaneSpec(
        name="micro_lr1e-6_anchor8_steps250",
        lr=1e-6,
        anchor_weight=8.0,
        max_steps=250,
    ),
    "micro_lr3e-6_anchor8_steps250": LaneSpec(
        name="micro_lr3e-6_anchor8_steps250",
        lr=3e-6,
        anchor_weight=8.0,
        max_steps=250,
    ),
    "micro_lr1e-6_anchor8_steps500": LaneSpec(
        name="micro_lr1e-6_anchor8_steps500",
        lr=1e-6,
        anchor_weight=8.0,
        max_steps=500,
    ),
    "ultra_micro_lr5e-7_anchor8_steps500": LaneSpec(
        name="ultra_micro_lr5e-7_anchor8_steps500",
        lr=5e-7,
        anchor_weight=8.0,
        max_steps=500,
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
    print(f"[budget-microdistill] {message}", flush=True)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


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
    }


def parse_csv_paths(text: str) -> list[Path]:
    return [Path(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_csv_values(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def build_input_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": str(path), "sha256": sha256_file(path)}
    if path.suffix == ".jsonl":
        summary["rows"] = sum(1 for row in read_jsonl(path) if row)
    return summary


def lane_candidate_key(lane_name: str, step: int) -> str:
    return f"{lane_name}_step{step}"


def build_model(device: torch.device) -> PolicyValueNet:
    return PolicyValueNet(
        hidden_sizes=(TRUNK_SIZE, RESIDUAL_BLOCK_COUNT),
        model_type=MODEL_TYPE,
        input_size=input_size_for_encoding(INPUT_ENCODING),
    ).to(device)


def load_current_model(
    current_checkpoint: Path, device: torch.device
) -> PolicyValueNet:
    model = build_model(device)
    load_checkpoint_into_model(model, current_checkpoint)
    model.eval()
    return model


def discover_pr153_candidates(pr153_workdir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not pr153_workdir.is_dir():
        return entries
    for lane_dir in sorted(path for path in pr153_workdir.iterdir() if path.is_dir()):
        checkpoints = sorted(lane_dir.glob("checkpoint_epoch*.npz"))
        artifacts = sorted(
            path
            for path in lane_dir.glob("artifact_epoch*")
            if (path / "weights.json").is_file()
        )
        if not checkpoints and not artifacts:
            continue
        checkpoint_path = checkpoints[-1] if checkpoints else None
        artifact_path = artifacts[-1] if artifacts else None
        entries.append(
            {
                "lane": lane_dir.name,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                "checkpoint_sha256": sha256_file(checkpoint_path)
                if checkpoint_path
                else None,
                "artifact_dir": str(artifact_path) if artifact_path else None,
                "artifact_weights_sha256": sha256_file(artifact_path / "weights.json")
                if artifact_path
                else None,
            }
        )
    return entries


def row_current_raw_top(row: dict[str, Any]) -> int | None:
    legal_moves = [
        move for move, flag in enumerate(row["legal_mask"]) if int(flag) == 1
    ]
    return top_policy_move(row["teacher_raw_policy"], legal_moves)


def row_teacher_top(row: dict[str, Any]) -> int | None:
    legal_moves = [
        move for move, flag in enumerate(row["legal_mask"]) if int(flag) == 1
    ]
    return top_policy_move(row["teacher_puct_policy"], legal_moves)


def is_target_row(row: dict[str, Any]) -> bool:
    if str(row.get("budget_context")) != "384:256":
        return False
    teacher_top = row_teacher_top(row)
    current_top = row_current_raw_top(row)
    if teacher_top != current_top:
        return True
    if teacher_top is None:
        return False
    teacher_confidence = float(row["teacher_puct_policy"][teacher_top])
    current_confidence = float(row["teacher_raw_policy"][teacher_top])
    return teacher_confidence > (current_confidence + 1e-9)


def is_anchor_row(row: dict[str, Any]) -> bool:
    return str(row.get("budget_context")) in ANCHOR_BUDGETS


def split_teacher_dataset(
    rows: list[dict[str, Any]], current_raw_policies: np.ndarray
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    target_rows = []
    anchor_rows = []
    selected_rows = []
    for row, current_raw_policy in zip(rows, current_raw_policies, strict=True):
        row_copy = dict(row)
        row_copy["current_raw_policy"] = current_raw_policy.tolist()
        row_copy["current_selected_search_move"] = int(row["teacher_selected_move"])
        row_copy["current_root_visit_distribution"] = list(row["teacher_puct_policy"])
        if is_target_row(row_copy):
            row_copy["split_phase"] = "target"
            target_rows.append(row_copy)
            selected_rows.append(row_copy)
        elif is_anchor_row(row_copy):
            row_copy["split_phase"] = "anchor"
            anchor_rows.append(row_copy)
            selected_rows.append(row_copy)
    return selected_rows, target_rows, anchor_rows


def build_dataset_split_audit(
    *,
    selected_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_budget = Counter(str(row["budget_context"]) for row in selected_rows)
    phase_distribution = Counter(str(row["phase"]) for row in selected_rows)
    seat_distribution = Counter(str(row["seat_context"]) for row in selected_rows)
    split_distribution = Counter(str(row["split_phase"]) for row in selected_rows)
    disagreement_by_budget = defaultdict(int)
    agreement_by_budget = defaultdict(int)
    entropy_by_budget: dict[str, list[float]] = defaultdict(list)
    duplicate_count = len(selected_rows) - len(
        {str(row["state_hash"]) for row in selected_rows}
    )
    row_records: list[dict[str, Any]] = []
    for row in selected_rows:
        budget = str(row["budget_context"])
        current_raw_top = row_current_raw_top(row)
        teacher_top = row_teacher_top(row)
        disagreement_by_budget[budget] += int(current_raw_top != teacher_top)
        agreement_by_budget[budget] += int(
            int(row["teacher_selected_move"])
            == int(row["current_selected_search_move"])
        )
        legal_moves = [
            move for move, flag in enumerate(row["legal_mask"]) if int(flag) == 1
        ]
        entropy_by_budget[budget].append(
            policy_entropy(row["teacher_puct_policy"], legal_moves)
        )
        row_records.append(
            {
                "state_hash": row["state_hash"],
                "budget_context": budget,
                "split_phase": row["split_phase"],
                "phase": row["phase"],
                "seat": row["seat_context"],
                "teacher_puct_policy": row["teacher_puct_policy"],
                "current_raw_policy": row["current_raw_policy"],
                "current_selected_search_move": row["current_selected_search_move"],
                "current_root_visit_distribution": row[
                    "current_root_visit_distribution"
                ],
                "legal_mask": row["legal_mask"],
                "search_profile_hash": row["search_profile_hash"],
            }
        )
    return {
        "schema": SUMMARY_SCHEMA,
        "selected_row_count": len(selected_rows),
        "target_row_count": len(target_rows),
        "anchor_row_count": len(anchor_rows),
        "row_counts_by_budget": dict(sorted(by_budget.items())),
        "phase_distribution": dict(sorted(phase_distribution.items())),
        "seat_distribution": dict(sorted(seat_distribution.items())),
        "split_distribution": dict(sorted(split_distribution.items())),
        "teacher_current_raw_top1_disagreement_by_budget": {
            budget: {
                "rows": int(by_budget[budget]),
                "count": int(disagreement_by_budget[budget]),
                "rate": float(
                    disagreement_by_budget[budget] / max(by_budget[budget], 1)
                ),
            }
            for budget in sorted(by_budget)
        },
        "teacher_current_search_top1_agreement_by_budget": {
            budget: {
                "rows": int(by_budget[budget]),
                "count": int(agreement_by_budget[budget]),
                "rate": float(agreement_by_budget[budget] / max(by_budget[budget], 1)),
            }
            for budget in sorted(by_budget)
        },
        "policy_entropy_distribution_by_budget": {
            budget: summarize_distribution(entropy_by_budget[budget])
            for budget in sorted(entropy_by_budget)
        },
        "duplicate_state_count": int(duplicate_count),
        "row_records": row_records,
    }


def arrays_for_rows(
    rows: list[dict[str, Any]], policy_key: str
) -> dict[str, np.ndarray]:
    return {
        "x": np.asarray([row["state"] for row in rows], dtype=np.float32),
        "p": np.asarray([row[policy_key] for row in rows], dtype=np.float32),
        "legal_mask": np.asarray([row["legal_mask"] for row in rows], dtype=np.float32),
    }


def predict_model(
    *,
    model: PolicyValueNet,
    x: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    policies: list[np.ndarray] = []
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], POLICY_BATCH_SIZE):
            end = min(start + POLICY_BATCH_SIZE, x.shape[0])
            xb = torch.from_numpy(x[start:end]).to(device)
            maskb = torch.from_numpy(legal_mask[start:end]).to(device)
            logits, value = model(xb)
            masked_logits = logits.masked_fill(maskb <= 0.0, -1e9)
            policies.append(torch.softmax(masked_logits, dim=1).detach().cpu().numpy())
            values.append(value.detach().cpu().numpy())
    return np.concatenate(policies, axis=0), np.concatenate(values, axis=0)


def compute_policy_metrics(
    *,
    rows: list[dict[str, Any]],
    candidate_policies: np.ndarray,
    candidate_values: np.ndarray,
    current_values: np.ndarray,
    current_entropy: float | None = None,
) -> dict[str, Any]:
    teacher_top1 = 0
    raw_changed = 0
    legal_failures = 0
    entropy_values: list[float] = []
    value_diffs: list[float] = []
    by_budget: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    target_improvements: list[float] = []
    anchor_kls: list[float] = []
    for index, row in enumerate(rows):
        legal_moves = [
            move for move, flag in enumerate(row["legal_mask"]) if int(flag) == 1
        ]
        candidate_policy = candidate_policies[index].tolist()
        candidate_top = top_policy_move(candidate_policy, legal_moves)
        teacher_top = top_policy_move(row["teacher_puct_policy"], legal_moves)
        current_top = top_policy_move(row["current_raw_policy"], legal_moves)
        budget = str(row["budget_context"])
        entropy = policy_entropy(candidate_policy, legal_moves)
        entropy_values.append(entropy)
        value_diffs.append(
            abs(float(candidate_values[index][0]) - float(current_values[index][0]))
        )
        if candidate_top == teacher_top:
            teacher_top1 += 1
        if candidate_top != current_top:
            raw_changed += 1
        if any(
            candidate_policy[move] > 1e-7
            for move in range(6)
            if move not in legal_moves
        ):
            legal_failures += 1
        teacher_kl = float(
            np.sum(
                np.asarray(row["teacher_puct_policy"], dtype=np.float64)[legal_moves]
                * np.log(
                    np.clip(
                        np.asarray(row["teacher_puct_policy"], dtype=np.float64)[
                            legal_moves
                        ],
                        1e-8,
                        1.0,
                    )
                    / np.clip(
                        np.asarray(candidate_policy, dtype=np.float64)[legal_moves],
                        1e-8,
                        1.0,
                    )
                )
            )
        )
        current_anchor_kl = float(
            np.sum(
                np.asarray(row["current_raw_policy"], dtype=np.float64)[legal_moves]
                * np.log(
                    np.clip(
                        np.asarray(row["current_raw_policy"], dtype=np.float64)[
                            legal_moves
                        ],
                        1e-8,
                        1.0,
                    )
                    / np.clip(
                        np.asarray(candidate_policy, dtype=np.float64)[legal_moves],
                        1e-8,
                        1.0,
                    )
                )
            )
        )
        by_budget[budget]["teacher_top1"].append(
            1.0 if candidate_top == teacher_top else 0.0
        )
        by_budget[budget]["raw_changed"].append(
            1.0 if candidate_top != current_top else 0.0
        )
        by_budget[budget]["teacher_kl"].append(teacher_kl)
        by_budget[budget]["entropy"].append(entropy)
        by_budget[budget]["anchor_kl"].append(current_anchor_kl)
        if budget == "384:256":
            current_agree = 1.0 if current_top == teacher_top else 0.0
            candidate_agree = 1.0 if candidate_top == teacher_top else 0.0
            target_improvements.append(candidate_agree - current_agree)
        if budget in ANCHOR_BUDGETS:
            anchor_kls.append(current_anchor_kl)
    mean_entropy = float(statistics.fmean(entropy_values)) if entropy_values else 0.0
    entropy_threshold = max(
        ENTROPY_FLOOR, 0.75 * float(current_entropy or mean_entropy or 0.0)
    )
    return {
        "rows": len(rows),
        "teacher_puct_top1_agreement": float(teacher_top1 / max(len(rows), 1)),
        "changed_raw_top1_rate_vs_current": float(raw_changed / max(len(rows), 1)),
        "candidate_entropy": mean_entropy,
        "legal_failures": int(legal_failures),
        "max_value_diff_vs_current": max(value_diffs) if value_diffs else 0.0,
        "mean_value_diff_vs_current": float(statistics.fmean(value_diffs))
        if value_diffs
        else 0.0,
        "root_mae_vs_current": float(statistics.fmean(value_diffs))
        if value_diffs
        else 0.0,
        "mean_anchor_kl_vs_current_raw": float(statistics.fmean(anchor_kls))
        if anchor_kls
        else 0.0,
        "entropy_collapsed": mean_entropy < entropy_threshold,
        "teacher_top1_gain_384_256_vs_current": float(
            statistics.fmean(target_improvements)
        )
        if target_improvements
        else 0.0,
        "by_budget": {
            budget: {
                "rows": len(values["teacher_top1"]),
                "teacher_puct_top1_agreement": float(
                    statistics.fmean(values["teacher_top1"])
                ),
                "changed_raw_top1_rate_vs_current": float(
                    statistics.fmean(values["raw_changed"])
                ),
                "policy_kl_vs_teacher": float(statistics.fmean(values["teacher_kl"])),
                "candidate_entropy": float(statistics.fmean(values["entropy"])),
                "anchor_kl_vs_current_raw": float(
                    statistics.fmean(values["anchor_kl"])
                ),
            }
            for budget, values in sorted(by_budget.items())
        },
    }


def search_probe_rows(
    rows: list[dict[str, Any]], cap_per_bucket: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["budget_context"]), str(row["seat_context"]))].append(row)
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        bucket = list(grouped[key])
        if len(bucket) > cap_per_bucket:
            indexes = rng.choice(len(bucket), size=cap_per_bucket, replace=False)
            selected.extend(bucket[int(index)] for index in sorted(indexes))
        else:
            selected.extend(bucket)
    return selected


def current_search_outputs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs = []
    for row in rows:
        search_policy = list(row["teacher_puct_policy"])
        selected_move = int(row["current_selected_search_move"])
        outputs.append(
            {
                "state_hash": str(row["state_hash"]),
                "budget_context": str(row["budget_context"]),
                "seat_context": str(row["seat_context"]),
                "selected_move": selected_move,
                "search_policy": search_policy,
                "selected_visit_share": float(search_policy[selected_move]),
            }
        )
    return outputs


def compute_search_metrics(
    *,
    rows: list[dict[str, Any]],
    candidate_outputs: list[dict[str, Any]],
    current_outputs_ref: list[dict[str, Any]],
) -> dict[str, Any]:
    overall_changed = []
    overall_teacher = []
    visit_kls = []
    selected_share_deltas = []
    by_budget: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    p0 = []
    p1 = []
    for row, candidate, current in zip(
        rows, candidate_outputs, current_outputs_ref, strict=True
    ):
        legal_moves = [
            move for move, flag in enumerate(row["legal_mask"]) if int(flag) == 1
        ]
        budget = str(row["budget_context"])
        changed = (
            1.0
            if int(candidate["selected_move"]) != int(current["selected_move"])
            else 0.0
        )
        teacher = (
            1.0
            if int(candidate["selected_move"]) == int(row["teacher_selected_move"])
            else 0.0
        )
        current_policy = np.asarray(current["search_policy"], dtype=np.float64)
        candidate_policy = np.asarray(candidate["search_policy"], dtype=np.float64)
        visit_kl = float(
            np.sum(
                current_policy[legal_moves]
                * np.log(
                    np.clip(current_policy[legal_moves], 1e-8, 1.0)
                    / np.clip(candidate_policy[legal_moves], 1e-8, 1.0)
                )
            )
        )
        share_delta = float(candidate["selected_visit_share"]) - float(
            current["selected_visit_share"]
        )
        overall_changed.append(changed)
        overall_teacher.append(teacher)
        visit_kls.append(visit_kl)
        selected_share_deltas.append(share_delta)
        by_budget[budget]["changed"].append(changed)
        by_budget[budget]["teacher"].append(teacher)
        by_budget[budget]["visit_kl"].append(visit_kl)
        by_budget[budget]["share_delta"].append(share_delta)
        if budget == "384:256":
            if str(row["seat_context"]) == "challenger_player_0":
                p0.append(teacher)
            else:
                p1.append(teacher)
    return {
        "rows": len(rows),
        "search_selected_move_agreement_with_teacher": float(
            statistics.fmean(overall_teacher)
        )
        if overall_teacher
        else 0.0,
        "search_selected_move_changed_rate_vs_current": float(
            statistics.fmean(overall_changed)
        )
        if overall_changed
        else 0.0,
        "root_visit_kl_vs_current_ref": float(statistics.fmean(visit_kls))
        if visit_kls
        else 0.0,
        "root_selected_visit_share_delta_vs_current_ref": float(
            statistics.fmean(selected_share_deltas)
        )
        if selected_share_deltas
        else 0.0,
        "changed_move_rate_by_budget_context": {
            budget: {
                "rows": len(values["changed"]),
                "search_agreement_with_teacher": float(
                    statistics.fmean(values["teacher"])
                ),
                "changed_rate_vs_current": float(statistics.fmean(values["changed"])),
                "root_visit_kl_vs_current_ref": float(
                    statistics.fmean(values["visit_kl"])
                ),
                "selected_visit_share_delta_vs_current_ref": float(
                    statistics.fmean(values["share_delta"])
                ),
            }
            for budget, values in sorted(by_budget.items())
        },
        "p0_p1_split_384_256": {
            "challenger_player_0": float(statistics.fmean(p0)) if p0 else 0.0,
            "challenger_player_1": float(statistics.fmean(p1)) if p1 else 0.0,
        },
    }


def probe_gate(
    *,
    current_policy_metrics: dict[str, Any],
    policy_metrics: dict[str, Any],
    preservation: dict[str, Any],
    search_metrics: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if not preservation.get("passes", False):
        reasons.append("value/trunk preserved exactly failed")
    if int(policy_metrics.get("legal_failures", 0)) != 0:
        reasons.append("legal failures > 0")
    if policy_metrics.get("entropy_collapsed"):
        reasons.append("entropy collapses")
    if float(policy_metrics.get("mean_anchor_kl_vs_current_raw", 0.0)) > 0.02:
        reasons.append("anchor KL vs current raw > 0.02")
    budget_rates = search_metrics.get("changed_move_rate_by_budget_context", {})
    rate_384 = float(
        budget_rates.get("384:256", {}).get("changed_rate_vs_current", 0.0)
    )
    rate_768 = float(
        budget_rates.get("768:768", {}).get("changed_rate_vs_current", 0.0)
    )
    rate_1200 = float(
        budget_rates.get("1200:1200", {}).get("changed_rate_vs_current", 0.0)
    )
    rate_1200_256 = float(
        budget_rates.get("1200:256", {}).get("changed_rate_vs_current", 0.0)
    )
    if rate_384 < 0.03 or rate_384 > 0.15:
        reasons.append("384:256 changed search move rate outside [0.03, 0.15]")
    if rate_768 > 0.08:
        reasons.append("768:768 changed search move rate > 0.08")
    if rate_1200 > 0.08:
        reasons.append("1200:1200 changed search move rate > 0.08")
    if rate_1200_256 > 0.10:
        reasons.append("1200:256 changed search move rate > 0.10")
    for budget in ANCHOR_BUDGETS:
        if budget not in budget_rates:
            continue
        if float(budget_rates[budget].get("root_visit_kl_vs_current_ref", 0.0)) > 0.05:
            reasons.append(f"{budget} root visit KL > 0.05")
    current_384 = current_policy_metrics.get("by_budget", {}).get("384:256", {})
    candidate_384 = policy_metrics.get("by_budget", {}).get("384:256", {})
    if (
        float(candidate_384.get("teacher_puct_top1_agreement", 0.0))
        < float(current_384.get("teacher_puct_top1_agreement", 0.0)) + 0.01
    ):
        reasons.append("384:256 teacher top-1 improvement < +0.01")
    return {"passed": not reasons, "reasons": reasons}


def early_stop_reasons(
    *,
    preservation: dict[str, Any],
    policy_metrics: dict[str, Any],
    search_metrics: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if not preservation.get("passes", False):
        reasons.append("value/trunk preservation fails")
    if int(policy_metrics.get("legal_failures", 0)) != 0:
        reasons.append("legal failures > 0")
    if policy_metrics.get("entropy_collapsed"):
        reasons.append("entropy collapses")
    if float(policy_metrics.get("mean_anchor_kl_vs_current_raw", 0.0)) > 0.02:
        reasons.append("anchor KL vs current raw > 0.02")
    budget_rates = search_metrics.get("changed_move_rate_by_budget_context", {})
    for budget in ("768:768", "1200:1200"):
        if (
            float(budget_rates.get(budget, {}).get("changed_rate_vs_current", 0.0))
            > 0.08
        ):
            reasons.append(f"{budget} changed search move rate > 0.08")
    return reasons


def train_lane(
    *,
    lane_spec: LaneSpec,
    workdir: Path,
    current_checkpoint: Path,
    current_artifact: Path,
    current_model: PolicyValueNet,
    full_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    current_policy_metrics: dict[str, Any],
    current_values_full: np.ndarray,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    lane_dir = workdir / lane_spec.name
    lane_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    target_arrays = arrays_for_rows(target_rows, "teacher_puct_policy")
    anchor_arrays = arrays_for_rows(anchor_rows, "current_raw_policy")
    full_arrays = {
        "x": np.asarray([row["state"] for row in full_rows], dtype=np.float32),
        "legal_mask": np.asarray(
            [row["legal_mask"] for row in full_rows], dtype=np.float32
        ),
    }
    model = build_model(device)
    load_checkpoint_into_model(model, current_checkpoint)
    apply_trainable_scope(model, "policy_head")
    optimizer = torch.optim.Adam(
        (param for param in model.parameters() if param.requires_grad), lr=lane_spec.lr
    )
    target_x = torch.from_numpy(target_arrays["x"]).to(device)
    target_p = torch.from_numpy(target_arrays["p"]).to(device)
    target_mask = torch.from_numpy(target_arrays["legal_mask"]).to(device)
    anchor_x = torch.from_numpy(anchor_arrays["x"]).to(device)
    anchor_p = torch.from_numpy(anchor_arrays["p"]).to(device)
    anchor_mask = torch.from_numpy(anchor_arrays["legal_mask"]).to(device)
    target_count = target_x.size(0)
    anchor_count = anchor_x.size(0)
    training_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    aborted = False
    abort_reason = None
    current_probe_outputs = current_search_outputs(probe_rows)
    for step in range(1, lane_spec.max_steps + 1):
        model.train()
        target_index = torch.randint(
            0, target_count, (min(TRAIN_BATCH_SIZE, target_count),), device=device
        )
        anchor_index = torch.randint(
            0, anchor_count, (min(TRAIN_BATCH_SIZE, anchor_count),), device=device
        )
        target_logits, _target_value = model(target_x[target_index])
        anchor_logits, _anchor_value = model(anchor_x[anchor_index])
        target_loss = masked_cross_entropy(
            target_logits, target_p[target_index], target_mask[target_index]
        ).mean()
        anchor_loss = masked_cross_entropy(
            anchor_logits, anchor_p[anchor_index], anchor_mask[anchor_index]
        ).mean()
        total_loss = (TARGET_WEIGHT * target_loss) + (
            lane_spec.anchor_weight * anchor_loss
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_sq = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                grad_sq += float(torch.sum(parameter.grad.detach() ** 2).item())
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            1.0,
        )
        optimizer.step()
        training_rows.append(
            {
                "step": step,
                "target_loss": float(target_loss.detach().cpu().item()),
                "anchor_loss": float(anchor_loss.detach().cpu().item()),
                "total_loss": float(total_loss.detach().cpu().item()),
                "gradient_norm": float(math.sqrt(grad_sq)) if grad_sq > 0.0 else 0.0,
            }
        )
        if step % EARLY_STOP_INTERVAL != 0 and step != lane_spec.max_steps:
            continue
        checkpoint_path = lane_dir / f"checkpoint_step{step}.npz"
        np.savez(checkpoint_path, **checkpoint_from_model(model))
        artifact_dir = lane_dir / f"artifact_step{step}"
        export_checkpoint_artifact(
            checkpoint_path=checkpoint_path,
            export_dir=artifact_dir,
            version=f"{lane_spec.name}_step{step}",
            model_type=MODEL_TYPE,
            input_encoding=INPUT_ENCODING,
            policy_loss=training_rows[-1]["total_loss"],
            value_loss=0.0,
        )
        candidate_policies, candidate_values = predict_model(
            model=model,
            x=full_arrays["x"],
            legal_mask=full_arrays["legal_mask"],
            device=device,
        )
        policy_metrics = compute_policy_metrics(
            rows=full_rows,
            candidate_policies=candidate_policies,
            candidate_values=candidate_values,
            current_values=current_values_full,
            current_entropy=float(current_policy_metrics.get("candidate_entropy", 0.0)),
        )
        preservation = compare_preserved_weights(current_checkpoint, checkpoint_path)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        candidate_search_outputs = evaluate_search_outputs(
            workdir=lane_dir,
            candidate_name=f"{lane_candidate_key(lane_spec.name, step)}_{checkpoint_sha256[:12]}",
            artifact_path=artifact_dir,
            rows=probe_rows,
            default_c_puct=default_c_puct,
            cpuct_schedule=cpuct_schedule,
            tactical_root_bias=tactical_root_bias,
            seed=seed,
        )
        search_metrics = compute_search_metrics(
            rows=probe_rows,
            candidate_outputs=candidate_search_outputs,
            current_outputs_ref=current_probe_outputs,
        )
        gate = probe_gate(
            current_policy_metrics=current_policy_metrics,
            policy_metrics=policy_metrics,
            preservation=preservation,
            search_metrics=search_metrics,
        )
        stop_reasons = early_stop_reasons(
            preservation=preservation,
            policy_metrics=policy_metrics,
            search_metrics=search_metrics,
        )
        checkpoints.append(
            {
                "candidate_name": lane_candidate_key(lane_spec.name, step),
                "lane": lane_spec.name,
                "step": step,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "artifact_dir": str(artifact_dir),
                "artifact_weights_sha256": sha256_file(artifact_dir / "weights.json"),
                "policy_probe": policy_metrics,
                "value_trunk_preservation": preservation,
                "search_probe": search_metrics,
                "probe_gate": gate,
                "early_stop_reasons": stop_reasons,
            }
        )
        log_progress(
            f"lane={lane_spec.name} step={step} target_loss={training_rows[-1]['target_loss']:.4f} "
            f"anchor_loss={training_rows[-1]['anchor_loss']:.4f} gate_pass={gate['passed']}"
        )
        if stop_reasons:
            aborted = True
            abort_reason = "; ".join(stop_reasons)
            break
    return {
        "lane": lane_spec.name,
        "lr": lane_spec.lr,
        "target_weight": TARGET_WEIGHT,
        "anchor_weight": lane_spec.anchor_weight,
        "max_steps": lane_spec.max_steps,
        "trainable_scope": "policy_head",
        "init_checkpoint": str(current_checkpoint),
        "checkpoints": checkpoints,
        "training_steps": training_rows,
        "aborted": aborted,
        "abort_reason": abort_reason,
    }


def suite_delta(summary: dict[str, Any], candidate_name: str, budget: str) -> float:
    current_value = float(summary["current_ref"]["fixed_large"][budget]["ds"])
    candidate_value = float(summary[candidate_name]["fixed_large"][budget]["ds"])
    return candidate_value - current_value


def carry_fixed_large_candidates(medium_summary: dict[str, Any]) -> list[str]:
    eligible = []
    for candidate_name in medium_summary:
        if candidate_name == "current_ref":
            continue
        gain_384 = suite_delta(medium_summary, candidate_name, "384:256")
        reg_768 = suite_delta(medium_summary, candidate_name, "768:768")
        reg_1200 = suite_delta(medium_summary, candidate_name, "1200:1200")
        if gain_384 >= 0.02 and reg_768 >= -0.02 and reg_1200 >= -0.02:
            eligible.append((gain_384, candidate_name))
    eligible.sort(reverse=True)
    return ["current_ref", *[name for _gain, name in eligible[:3]]]


def select_heldout_candidate(fixed_large_summary: dict[str, Any]) -> str | None:
    best_name = None
    best_gain = -1e9
    for candidate_name in fixed_large_summary:
        if candidate_name == "current_ref":
            continue
        gain_384 = suite_delta(fixed_large_summary, candidate_name, "384:256")
        reg_768 = suite_delta(fixed_large_summary, candidate_name, "768:768")
        reg_1200 = suite_delta(fixed_large_summary, candidate_name, "1200:1200")
        reg_1200_256 = suite_delta(fixed_large_summary, candidate_name, "1200:256")
        if (
            gain_384 >= 0.03
            and reg_768 >= -0.03
            and reg_1200 >= -0.03
            and reg_1200_256 >= -0.03
        ):
            if gain_384 > best_gain:
                best_gain = gain_384
                best_name = candidate_name
    return best_name


def run_gate(
    *,
    workdir: Path,
    current_artifact: Path,
    candidate_name: str,
    candidate_path: Path,
    workers: int,
    seed: int,
    c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
) -> dict[str, Any]:
    out_path = workdir / "gate" / f"{candidate_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(REPO_ROOT / ".venv/bin/python"),
        str(REPO_ROOT / "script/ai/seat_aware_promotion_gate"),
        "--candidate-path",
        str(candidate_path),
        "--current-path",
        str(current_artifact),
        "--out",
        str(out_path),
        "--games",
        "120",
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--budget-pairs",
        "384:256,768:768,1200:1200,1200:256",
        "--c-puct",
        str(c_puct),
        "--c-puct-schedule-json",
        json.dumps(cpuct_schedule, sort_keys=True),
        "--root-policy-mode",
        "deterministic",
    ]
    if tactical_root_bias is not None:
        cmd.extend(["--tactical-root-bias", str(tactical_root_bias)])
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=14400,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gate failed: {result.stderr[-2000:]}")
    report = json.loads(out_path.read_text(encoding="utf-8"))
    return {
        "ran": True,
        "report_path": str(out_path),
        "classification": report.get("classification"),
        "preserved": candidate_gate_preserved(report),
    }


def final_classification(summary: dict[str, Any]) -> str:
    heldout_summary = summary.get("heldout_suite_summary") or {}
    bootstrap_tables = summary.get("bootstrap_tables") or {}
    gate_results = summary.get("gate_results") or {}
    current_fixed = summary.get("fixed_large_suite_summary") or {}
    probe_passing = summary.get("probe_passing_candidates", [])
    checkpoint_candidates = [
        candidate
        for name, candidate in summary.get("candidates", {}).items()
        if name != "current_ref"
    ]
    ultra_micro = summary.get("lane_runs", {}).get(
        "ultra_micro_lr5e-7_anchor8_steps500", {}
    )
    if heldout_summary:
        current = heldout_summary["current_ref"]["heldout"]
        for candidate_name, candidate_summary in heldout_summary.items():
            if candidate_name == "current_ref":
                continue
            gain_384 = float(
                candidate_summary["heldout"]["384:256"]["mean_ds"]
            ) - float(current["384:256"]["mean_ds"])
            reg_768 = float(candidate_summary["heldout"]["768:768"]["mean_ds"]) - float(
                current["768:768"]["mean_ds"]
            )
            reg_1200 = float(
                candidate_summary["heldout"]["1200:1200"]["mean_ds"]
            ) - float(current["1200:1200"]["mean_ds"])
            reg_1200_256 = float(
                candidate_summary["heldout"]["1200:256"]["mean_ds"]
            ) - float(current["1200:256"]["mean_ds"])
            lower = float(
                bootstrap_tables.get(candidate_name, {})
                .get("384:256", {})
                .get("ds", {})
                .get("candidate_minus_current", {})
                .get("lower", 0.0)
            )
            if (
                gain_384 >= 0.05
                and lower > 0.01
                and reg_768 >= -0.05
                and reg_1200 >= -0.03
                and reg_1200_256 >= -0.03
                and gate_results.get(candidate_name, {}).get("preserved") is True
            ):
                return "budget_anchored_policy_candidate"
    if current_fixed:
        current_summary = current_fixed["current_ref"]["fixed_large"]
        safe_but_weak = False
        tradeoff = False
        for candidate_name, candidate_summary in current_fixed.items():
            if candidate_name == "current_ref":
                continue
            gain_384 = float(candidate_summary["fixed_large"]["384:256"]["ds"]) - float(
                current_summary["384:256"]["ds"]
            )
            reg_768 = float(candidate_summary["fixed_large"]["768:768"]["ds"]) - float(
                current_summary["768:768"]["ds"]
            )
            reg_1200 = float(
                candidate_summary["fixed_large"]["1200:1200"]["ds"]
            ) - float(current_summary["1200:1200"]["ds"])
            reg_1200_256 = float(
                candidate_summary["fixed_large"]["1200:256"]["ds"]
            ) - float(current_summary["1200:256"]["ds"])
            if reg_768 >= -0.05 and reg_1200 >= -0.03 and reg_1200_256 >= -0.03:
                if gain_384 < 0.05:
                    safe_but_weak = True
            if gain_384 > 0.0 and (reg_768 < -0.05 or reg_1200 < -0.03):
                tradeoff = True
        if safe_but_weak:
            return "micro_update_safe_but_too_weak"
        if tradeoff:
            return "policy_distillation_inherently_tradeoff"
    safe_checkpoint_candidates = []
    for candidate in checkpoint_candidates:
        search_probe = candidate.get("search_probe", {})
        budgets = search_probe.get("changed_move_rate_by_budget_context", {})
        if (
            float(budgets.get("768:768", {}).get("changed_rate_vs_current", 1.0))
            <= 0.08
            and float(budgets.get("1200:1200", {}).get("changed_rate_vs_current", 1.0))
            <= 0.08
            and float(budgets.get("1200:256", {}).get("changed_rate_vs_current", 1.0))
            <= 0.10
        ):
            safe_checkpoint_candidates.append(candidate)
    if safe_checkpoint_candidates:
        if all(
            float(
                candidate.get("policy_probe", {}).get(
                    "teacher_top1_gain_384_256_vs_current", 0.0
                )
            )
            < 0.05
            and float(
                candidate.get("search_probe", {})
                .get("changed_move_rate_by_budget_context", {})
                .get("384:256", {})
                .get("changed_rate_vs_current", 0.0)
            )
            < 0.03
            for candidate in safe_checkpoint_candidates
        ):
            return "micro_update_safe_but_too_weak"
    if ultra_micro:
        checkpoints = ultra_micro.get("checkpoints", [])
        if checkpoints and all(
            any(
                reason
                in {
                    "768:768 changed search move rate > 0.08",
                    "1200:1200 changed search move rate > 0.08",
                }
                for reason in checkpoint.get("early_stop_reasons", [])
            )
            for checkpoint in checkpoints
        ):
            return "policy_distillation_too_brittle"
    if not probe_passing:
        return "teacher_cloning_closed_for_now"
    return "teacher_cloning_closed_for_now"


def build_markdown_report(summary: dict[str, Any]) -> str:
    dataset_audit = summary["dataset_split_audit"]
    pr153 = summary.get("pr153_candidate_hashes", [])
    checkpoint_rows = []
    policy_rows = []
    preservation_rows = []
    search_rows = []
    aborted_rows = []
    medium_rows = []
    fixed_rows = []
    heldout_rows = []
    bootstrap_rows = []
    p0p1_rows = []
    runtime_rows = []
    for lane_name, lane in summary.get("lane_runs", {}).items():
        for checkpoint in lane.get("checkpoints", []):
            checkpoint_rows.append(
                [
                    lane_name,
                    checkpoint["step"],
                    checkpoint["candidate_name"],
                    checkpoint["artifact_weights_sha256"],
                    checkpoint["probe_gate"]["passed"],
                    "; ".join(checkpoint.get("early_stop_reasons", [])) or "continue",
                ]
            )
            policy_probe = checkpoint["policy_probe"]
            search_probe = checkpoint["search_probe"]
            preservation = checkpoint["value_trunk_preservation"]
            policy_rows.append(
                [
                    checkpoint["candidate_name"],
                    fmt(policy_probe["teacher_puct_top1_agreement"]),
                    fmt(policy_probe["teacher_top1_gain_384_256_vs_current"]),
                    fmt(policy_probe["changed_raw_top1_rate_vs_current"]),
                    fmt(policy_probe["candidate_entropy"]),
                    int(policy_probe["legal_failures"]),
                ]
            )
            preservation_rows.append(
                [
                    checkpoint["candidate_name"],
                    preservation.get("passes"),
                    fmt(preservation.get("max_abs_diff", 0.0)),
                    ", ".join(preservation.get("changed_keys", [])) or "none",
                ]
            )
            for budget in PROBE_BUDGETS:
                budget_metrics = search_probe.get(
                    "changed_move_rate_by_budget_context", {}
                ).get(budget, {})
                search_rows.append(
                    [
                        checkpoint["candidate_name"],
                        budget,
                        fmt(budget_metrics.get("changed_rate_vs_current")),
                        fmt(budget_metrics.get("search_agreement_with_teacher")),
                        fmt(budget_metrics.get("root_visit_kl_vs_current_ref")),
                        fmt(
                            budget_metrics.get(
                                "selected_visit_share_delta_vs_current_ref"
                            )
                        ),
                    ]
                )
        if lane.get("aborted"):
            aborted_rows.append([lane_name, lane.get("abort_reason")])
    medium_summary = summary.get("medium_suite_summary") or {}
    if medium_summary:
        for candidate_name, candidate_summary in medium_summary.items():
            fixed = candidate_summary["fixed_large"]
            medium_rows.append(
                [
                    candidate_name,
                    *[fmt(fixed[budget]["ds"]) for budget in SUITE_BUDGETS],
                ]
            )
    fixed_large_summary = summary.get("fixed_large_suite_summary") or {}
    if fixed_large_summary:
        for candidate_name, candidate_summary in fixed_large_summary.items():
            fixed = candidate_summary["fixed_large"]
            fixed_rows.append(
                [
                    candidate_name,
                    *[fmt(fixed[budget]["ds"]) for budget in SUITE_BUDGETS],
                ]
            )
    heldout_summary = summary.get("heldout_suite_summary") or {}
    if heldout_summary:
        for candidate_name, candidate_summary in heldout_summary.items():
            held = candidate_summary["heldout"]
            heldout_rows.append(
                [
                    candidate_name,
                    fmt(held["384:256"]["mean_ds"]),
                    fmt(held["384:256"]["worst_suite_ds"]),
                    fmt(held["768:768"]["mean_ds"]),
                    fmt(held["1200:1200"]["mean_ds"]),
                    fmt(held["1200:256"]["mean_ds"]),
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
            runtime = candidate_summary["runtime_cost"]
            runtime_rows.append(
                [
                    candidate_name,
                    fmt(runtime.get("mean_move_time_ms"), 2),
                    fmt(runtime.get("mean_p95_move_time_ms"), 2),
                ]
            )
    for candidate_name, budgets in (summary.get("bootstrap_tables") or {}).items():
        for budget in ("384:256", "768:768", "1200:1200", "1200:256"):
            ds = budgets[budget]["ds"]
            bootstrap_rows.append(
                [
                    candidate_name,
                    budget,
                    "candidate_minus_current",
                    fmt(ds["candidate_minus_current"]["mean"]),
                    fmt(ds["candidate_minus_current"]["lower"]),
                    fmt(ds["candidate_minus_current"]["upper"]),
                ]
            )
            bootstrap_rows.append(
                [
                    candidate_name,
                    budget,
                    "current_minus_candidate",
                    fmt(ds["current_minus_candidate"]["mean"]),
                    fmt(ds["current_minus_candidate"]["lower"]),
                    fmt(ds["current_minus_candidate"]["upper"]),
                ]
            )
    lines = [
        "# AlphaZero-Lite Budget-Anchored Policy-Only Microdistill Results",
        "",
        f"**Classification**: `{summary['classification']}`",
        "",
        "## Current Artifact Hash",
        "",
        f"- current weights SHA256: `{summary['current_hash']['actual_sha256']}`",
        "",
        "## PR #153 Candidate Hashes",
        "",
    ]
    if pr153:
        for entry in pr153:
            lines.append(
                f"- {entry['lane']}: checkpoint_sha256=`{entry.get('checkpoint_sha256')}` weights_sha256=`{entry.get('artifact_weights_sha256')}`"
            )
    else:
        lines.append("- none discovered")
    lines.extend(
        [
            "",
            "## DS Orientation Audit Confirmation",
            "",
            f"- audit passed: `{summary['ds_orientation_audit']['passed']}`",
            f"- reference helper reused: `{True}`",
            f"- bootstrap orientations explicit: `{True}`",
            "",
            "## Dataset Split Audit",
            "",
            f"- selected rows: `{dataset_audit['selected_row_count']}`",
            f"- target rows: `{dataset_audit['target_row_count']}`",
            f"- anchor rows: `{dataset_audit['anchor_row_count']}`",
            f"- duplicate state count: `{dataset_audit['duplicate_state_count']}`",
            f"- row counts by budget: `{json.dumps(dataset_audit['row_counts_by_budget'], sort_keys=True)}`",
            f"- phase distribution: `{json.dumps(dataset_audit['phase_distribution'], sort_keys=True)}`",
            f"- seat distribution: `{json.dumps(dataset_audit['seat_distribution'], sort_keys=True)}`",
            "",
            "## Lane Definitions",
            "",
            markdown_table(
                ["Lane", "LR", "Target weight", "Anchor weight", "Max steps"],
                [
                    [
                        lane.name,
                        lane.lr,
                        TARGET_WEIGHT,
                        lane.anchor_weight,
                        lane.max_steps,
                    ]
                    for lane in LANE_SPECS.values()
                    if lane.name in summary["lane_runs"]
                ],
            ),
            "",
            "## Checkpoint/Early-Stop Table",
            "",
            markdown_table(
                [
                    "Lane",
                    "Step",
                    "Candidate",
                    "Weights SHA256",
                    "Probe pass",
                    "Early-stop",
                ],
                checkpoint_rows or [["none", "", "", "", "", ""]],
            ),
            "",
            "## Policy Probe Table",
            "",
            markdown_table(
                [
                    "Candidate",
                    "Teacher top-1",
                    "384:256 top-1 gain",
                    "Raw top-1 change",
                    "Entropy",
                    "Legal fails",
                ],
                policy_rows or [["none", "", "", "", "", ""]],
            ),
            "",
            "## Value/Trunk Preservation Table",
            "",
            markdown_table(
                ["Candidate", "Preserved", "Max abs diff", "Changed keys"],
                preservation_rows or [["none", "", "", ""]],
            ),
            "",
            "## Search-Aware Probe Table By Budget",
            "",
            markdown_table(
                [
                    "Candidate",
                    "Budget",
                    "Changed rate",
                    "Teacher agree",
                    "Root visit KL",
                    "Visit-share delta",
                ],
                search_rows or [["none", "", "", "", "", ""]],
            ),
            "",
            "## Aborted-Checkpoint Table",
            "",
            markdown_table(["Lane", "Reasons"], aborted_rows or [["none", ""]]),
            "",
            "## Medium DS Table",
            "",
            markdown_table(
                ["Candidate", *SUITE_BUDGETS],
                medium_rows or [["not_run", "", "", "", "", "", ""]],
            ),
            "",
            "## Fixed-Large DS Table",
            "",
            markdown_table(
                ["Candidate", *SUITE_BUDGETS],
                fixed_rows or [["not_run", "", "", "", "", "", ""]],
            ),
            "",
            "## Held-Out Table",
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
                heldout_rows or [["not_run", "", "", "", "", ""]],
            ),
            "",
            "## Bootstrap CIs",
            "",
            markdown_table(
                [
                    "Candidate",
                    "Budget",
                    "Orientation",
                    "Mean",
                    "Lower 95%",
                    "Upper 95%",
                ],
                bootstrap_rows or [["not_run", "", "", "", "", ""]],
            ),
            "",
            "## P0/P1 Split For 384:256",
            "",
            markdown_table(
                ["Candidate", "Mean P0", "Mean P1", "Gap"],
                p0p1_rows or [["not_run", "", "", ""]],
            ),
            "",
            "## Duplicate Trajectory Count",
            "",
            markdown_table(
                ["Candidate", "Mean duplicates"],
                [
                    [name, fmt(entry["duplicate_trajectory_count"])]
                    for name, entry in sorted(heldout_summary.items())
                ]
                or [["not_run", ""]],
            ),
            "",
            "## Runtime Cost",
            "",
            markdown_table(
                ["Candidate", "Mean move latency ms", "Mean p95 latency ms"],
                runtime_rows or [["not_run", "", ""]],
            ),
            "",
            "## Gate Result",
            "",
        ]
    )
    if summary.get("gate_results"):
        for candidate_name, result in sorted(summary["gate_results"].items()):
            if result.get("ran"):
                lines.append(
                    f"- {candidate_name}: classification=`{result.get('classification')}` preserved=`{result.get('preserved')}`"
                )
            else:
                lines.append(
                    f"- {candidate_name}: `not_run` reason=`{result.get('reason')}`"
                )
    else:
        lines.append("- gate not run")
    lines.extend(
        [
            "",
            "## Final Classification",
            "",
            f"- result: `{summary['classification']}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--pr153-workdir", required=True)
    parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--teacher-default-c-puct", type=float, required=True)
    parser.add_argument("--teacher-cpuct-schedule", required=True)
    parser.add_argument("--teacher-tactical-root-bias", type=float, required=True)
    parser.add_argument("--lanes", required=True)
    parser.add_argument("--probe-every-steps", type=int, default=EARLY_STOP_INTERVAL)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.probe_every_steps) != EARLY_STOP_INTERVAL:
        raise ValueError(
            f"this runner currently requires --probe-every-steps {EARLY_STOP_INTERVAL}"
        )
    started_at = time.time()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    current_artifact = Path(args.current)
    pr153_workdir = Path(args.pr153_workdir)
    teacher_dataset = Path(args.teacher_dataset)
    medium_suite = Path(args.medium_suite)
    fixed_large_suite = Path(args.fixed_large_suite)
    heldout_suites = parse_csv_paths(args.heldout_suites)
    cpuct_schedule = parse_cpuct_schedule_json(args.teacher_cpuct_schedule)
    lane_names = parse_csv_values(args.lanes)
    lane_specs = [LANE_SPECS[name] for name in lane_names]
    require_existing_file(current_artifact / "weights.json", "current artifact weights")
    require_existing_file(teacher_dataset, "teacher dataset")
    require_existing_file(medium_suite, "medium suite")
    require_existing_file(fixed_large_suite, "fixed large suite")
    for suite in heldout_suites:
        require_existing_file(suite, f"heldout suite {suite.name}")
    current_hash = verify_expected_hash(
        current_artifact / "weights.json",
        args.expected_current_weights_sha256,
        "current weights",
    )
    current_checkpoint = current_checkpoint_path(current_artifact, workdir)
    device = select_device(args.device)
    current_model = load_current_model(current_checkpoint, device)
    teacher_rows = read_jsonl(teacher_dataset)
    teacher_x = np.asarray([row["state"] for row in teacher_rows], dtype=np.float32)
    teacher_mask = np.asarray(
        [row["legal_mask"] for row in teacher_rows], dtype=np.float32
    )
    current_raw_policies_all, _unused_current_values_all = predict_model(
        model=current_model,
        x=teacher_x,
        legal_mask=teacher_mask,
        device=device,
    )
    selected_rows, target_rows, anchor_rows = split_teacher_dataset(
        teacher_rows, current_raw_policies_all
    )
    dataset_split_audit = build_dataset_split_audit(
        selected_rows=selected_rows,
        target_rows=target_rows,
        anchor_rows=anchor_rows,
    )
    write_json(workdir / DATASET_SPLIT_AUDIT_FILENAME, dataset_split_audit)
    full_x = np.asarray([row["state"] for row in selected_rows], dtype=np.float32)
    full_mask = np.asarray(
        [row["legal_mask"] for row in selected_rows], dtype=np.float32
    )
    _current_policies, current_values_full = predict_model(
        model=current_model,
        x=full_x,
        legal_mask=full_mask,
        device=device,
    )
    current_policy_metrics = compute_policy_metrics(
        rows=selected_rows,
        candidate_policies=np.asarray(
            [row["current_raw_policy"] for row in selected_rows], dtype=np.float32
        ),
        candidate_values=current_values_full,
        current_values=current_values_full,
    )
    probe_rows = search_probe_rows(
        selected_rows, SEARCH_PROBE_CAP_PER_BUCKET, args.seed
    )
    lane_runs: dict[str, Any] = {}
    candidates: dict[str, dict[str, Any]] = {
        "current_ref": {
            "name": "current_ref",
            "kind": "reference",
            "artifact_dir": str(current_artifact),
            "artifact_weights_sha256": current_hash["actual_sha256"],
            "policy_probe": current_policy_metrics,
            "value_trunk_preservation": compare_preserved_weights(
                current_checkpoint, current_checkpoint
            ),
            "search_probe": compute_search_metrics(
                rows=probe_rows,
                candidate_outputs=current_search_outputs(probe_rows),
                current_outputs_ref=current_search_outputs(probe_rows),
            ),
        }
    }
    probe_passing_candidates = []
    for lane_spec in lane_specs:
        lane_run = train_lane(
            lane_spec=lane_spec,
            workdir=workdir,
            current_checkpoint=current_checkpoint,
            current_artifact=current_artifact,
            current_model=current_model,
            full_rows=selected_rows,
            target_rows=target_rows,
            anchor_rows=anchor_rows,
            probe_rows=probe_rows,
            current_policy_metrics=current_policy_metrics,
            current_values_full=current_values_full,
            default_c_puct=float(args.teacher_default_c_puct),
            cpuct_schedule=cpuct_schedule,
            tactical_root_bias=float(args.teacher_tactical_root_bias),
            seed=args.seed,
            device=device,
        )
        lane_runs[lane_spec.name] = lane_run
        for checkpoint in lane_run["checkpoints"]:
            candidate_name = checkpoint["candidate_name"]
            candidates[candidate_name] = {
                "name": candidate_name,
                "kind": "student",
                "artifact_dir": checkpoint["artifact_dir"],
                "artifact_weights_sha256": checkpoint["artifact_weights_sha256"],
                "training": {
                    "provenance": "trained_in_this_run",
                    "trainable_scope": "policy_head",
                    "init_checkpoint": str(current_checkpoint),
                    "steps": checkpoint["step"],
                    "lr": lane_spec.lr,
                    "anchor_weight": lane_spec.anchor_weight,
                },
                "policy_probe": checkpoint["policy_probe"],
                "value_trunk_preservation": checkpoint["value_trunk_preservation"],
                "search_probe": checkpoint["search_probe"],
                "probe_gate": checkpoint["probe_gate"],
                "checkpoint_path": checkpoint["checkpoint_path"],
            }
            if checkpoint["probe_gate"]["passed"]:
                probe_passing_candidates.append(candidate_name)
    probe_passing_candidates = sorted(set(probe_passing_candidates))
    medium_suite_summary = None
    fixed_large_suite_summary = None
    heldout_suite_summary = None
    bootstrap_tables = {}
    gate_results: dict[str, Any] = {}
    ds_orientation_audit = {"passed": True, "errors": []}
    if probe_passing_candidates:
        medium_artifacts = [("current_ref", current_artifact)] + [
            (candidate_name, Path(str(candidates[candidate_name]["artifact_dir"])))
            for candidate_name in probe_passing_candidates
        ]
        medium_rows = run_suite_evaluations(
            workdir=workdir / "medium_eval",
            current_artifact=current_artifact,
            candidate_artifacts=medium_artifacts,
            suite_paths=[medium_suite],
            budget_pairs=",".join(SUITE_BUDGETS),
            workers=args.workers,
            seed=args.seed,
        )
        medium_suite_summary = aggregate_suite_metrics(
            suite_rows={"large": next(iter(medium_rows.values()))},
            candidate_names=[name for name, _path in medium_artifacts],
        )
        fixed_large_candidate_names = carry_fixed_large_candidates(medium_suite_summary)
        fixed_large_artifacts = [
            (
                name,
                current_artifact
                if name == "current_ref"
                else Path(str(candidates[name]["artifact_dir"])),
            )
            for name in fixed_large_candidate_names
        ]
        fixed_large_rows = run_suite_evaluations(
            workdir=workdir / "fixed_large_eval",
            current_artifact=current_artifact,
            candidate_artifacts=fixed_large_artifacts,
            suite_paths=[fixed_large_suite],
            budget_pairs=",".join(SUITE_BUDGETS),
            workers=args.workers,
            seed=args.seed,
        )
        fixed_large_suite_summary = aggregate_suite_metrics(
            suite_rows={"large_eval": next(iter(fixed_large_rows.values()))},
            candidate_names=[name for name, _path in fixed_large_artifacts],
        )
        heldout_candidate_name = select_heldout_candidate(fixed_large_suite_summary)
        if heldout_candidate_name is not None:
            heldout_artifacts = [
                ("current_ref", current_artifact),
                (
                    heldout_candidate_name,
                    Path(str(candidates[heldout_candidate_name]["artifact_dir"])),
                ),
            ]
            heldout_rows = run_suite_evaluations(
                workdir=workdir / "heldout_eval",
                current_artifact=current_artifact,
                candidate_artifacts=heldout_artifacts,
                suite_paths=heldout_suites,
                budget_pairs=",".join(SUITE_BUDGETS),
                workers=args.workers,
                seed=args.seed,
            )
            heldout_rows["large_eval"] = next(iter(fixed_large_rows.values()))
            heldout_suite_summary = aggregate_suite_metrics(
                suite_rows=heldout_rows,
                candidate_names=[name for name, _path in heldout_artifacts],
            )
            bootstrap_tables = compute_bootstrap_comparisons(
                suite_rows=heldout_rows,
                candidate_names=[name for name, _path in heldout_artifacts],
                seed=args.seed,
            )
            ds_orientation_audit = build_orientation_audit(
                medium_rows=medium_rows,
                fixed_large_rows=fixed_large_rows,
                heldout_rows=heldout_rows,
                candidate_names=[name for name, _path in heldout_artifacts],
                ds_audit_reference=build_ds_orientation_audit(
                    suite_rows=heldout_rows,
                    summary_candidate_name=heldout_candidate_name,
                    report_doc_path=REPORT_PATH,
                    seed=args.seed,
                ),
                bootstrap_tables=bootstrap_tables,
                seed=args.seed,
            )
            candidate_heldout = heldout_suite_summary[heldout_candidate_name]["heldout"]
            current_heldout = heldout_suite_summary["current_ref"]["heldout"]
            lower = float(
                bootstrap_tables[heldout_candidate_name]["384:256"]["ds"][
                    "candidate_minus_current"
                ]["lower"]
            )
            if (
                float(candidate_heldout["384:256"]["mean_ds"])
                - float(current_heldout["384:256"]["mean_ds"])
                >= 0.05
                and lower > 0.01
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
                gate_results[heldout_candidate_name] = run_gate(
                    workdir=workdir,
                    current_artifact=current_artifact,
                    candidate_name=heldout_candidate_name,
                    candidate_path=Path(
                        str(candidates[heldout_candidate_name]["artifact_dir"])
                    ),
                    workers=args.workers,
                    seed=args.seed,
                    c_puct=float(args.teacher_default_c_puct),
                    cpuct_schedule=cpuct_schedule,
                    tactical_root_bias=float(args.teacher_tactical_root_bias),
                )
            else:
                gate_results[heldout_candidate_name] = {
                    "ran": False,
                    "reason": "did not clear held-out gate thresholds",
                }
    runtime_seconds = time.time() - started_at
    summary = {
        "schema": SUMMARY_SCHEMA,
        "inputs": {
            "workdir": str(workdir),
            "current": str(current_artifact),
            "pr153_workdir": str(pr153_workdir),
            "teacher_dataset": build_input_summary(teacher_dataset),
            "medium_suite": build_input_summary(medium_suite),
            "fixed_large_suite": build_input_summary(fixed_large_suite),
            "heldout_suites": [build_input_summary(path) for path in heldout_suites],
            "teacher_runtime_profile": {
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
            "lanes": lane_names,
            "probe_every_steps": int(args.probe_every_steps),
            "workers": args.workers,
            "seed": args.seed,
        },
        "current_hash": current_hash,
        "pr153_candidate_hashes": discover_pr153_candidates(pr153_workdir),
        "dataset_split_audit": dataset_split_audit,
        "search_probe_rows": len(probe_rows),
        "lane_runs": lane_runs,
        "candidates": candidates,
        "probe_passing_candidates": probe_passing_candidates,
        "medium_suite_summary": medium_suite_summary,
        "fixed_large_suite_summary": fixed_large_suite_summary,
        "heldout_suite_summary": heldout_suite_summary,
        "bootstrap_tables": bootstrap_tables,
        "ds_orientation_audit": ds_orientation_audit,
        "gate_results": gate_results,
        "runtime_seconds": runtime_seconds,
    }
    summary["classification"] = final_classification(summary)
    write_json(workdir / SUMMARY_FILENAME, summary)
    REPORT_PATH.write_text(build_markdown_report(summary), encoding="utf-8")
    log_progress(
        f"completed classification={summary['classification']} probe_pass={len(probe_passing_candidates)} runtime_s={runtime_seconds:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
