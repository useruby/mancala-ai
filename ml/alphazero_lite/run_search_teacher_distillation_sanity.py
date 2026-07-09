#!/usr/bin/env python3
"""Search-teacher distillation sanity audit.

Validates the PR #150 search-teacher dataset, checks checkpoint/export
consistency, runs tiny overfit lanes, recalibrates probe metrics, and writes
summary/report artifacts without promoting or overwriting current.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator  # noqa: E402
from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    parse_cpuct_schedule_json,
    resolve_budget_cpuct,
    schedule_definition,
)
from ml.alphazero_lite.export_artifact import sha256_file  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    benchmark_budget_results,
    find_candidate_report,
    run_opening_suite_benchmark,
)
from ml.alphazero_lite.run_promoted_current_opening_puct_disagreement import (  # noqa: E402
    artifact_forward_details,
    policy_entropy,
    read_jsonl,
    top_policy_move,
    write_jsonl,
)
from ml.alphazero_lite.run_search_teacher_student_preflight import (  # noqa: E402
    build_probe_rows,
    current_checkpoint_path,
    evaluate_probe_candidate,
    evaluate_teacher_state,
    export_checkpoint_artifact,
    kl_divergence,
    load_json,
    verify_expected_hash,
    write_json,
)
from ml.alphazero_lite.self_play import encode_state  # noqa: E402
from ml.alphazero_lite.train import (  # noqa: E402
    PolicyValueNet,
    apply_trainable_scope,
    checkpoint_from_model,
    input_size_for_encoding,
    load_checkpoint_into_model,
    select_device,
    set_seed,
    train_one_epoch,
)


SUMMARY_SCHEMA = "azlite_search_teacher_distillation_sanity_v1"
REPORT_PATH = (
    REPO_ROOT / "docs/alphazero-lite-search-teacher-distillation-sanity-results.md"
)
SUMMARY_FILENAME = "summary_metrics.json"
INPUT_ENCODING = "kalah_v3"
MODEL_TYPE = "residual_v3"
POLICY_SIZE = 6
DATASET_AUDIT_SAMPLE_ROWS = 2048
DATASET_RECOMPUTE_ROWS = 64
CHECKPOINT_ARTIFACT_SAMPLE_ROWS = 512
KL_EPSILON = 1e-8
HUBER_DELTA = 1.0
MAX_POLICY_DIFF_FOR_MATCH = 1e-5
MAX_VALUE_DIFF_FOR_MATCH = 1e-5
TARGET_RECOMPUTE_MIN_TOP1 = 0.95
TARGET_RECOMPUTE_MAX_MEAN_KL = 0.05
SMOKE_SUITE_ROWS = 8
SMOKE_GAMES_PER_OPENING = 1


@dataclass(frozen=True)
class LaneSpec:
    name: str
    trunk_size: int
    residual_block_count: int
    train_rows: int
    max_epochs: int
    target_train_top1: float
    lr: float
    batch_size: int
    init_checkpoint: Path | None = None
    trainable_scope: str = "all"

    @property
    def hidden_sizes(self) -> tuple[int, int]:
        return (self.trunk_size, self.residual_block_count)


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


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def log_progress(message: str) -> None:
    print(f"[distillation-sanity] {message}", flush=True)


def require_existing_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "max": None,
        }
    ordered = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(ordered)),
        "p25": percentile(values, 25.0),
        "median": float(np.median(ordered)),
        "mean": float(np.mean(ordered)),
        "p75": percentile(values, 75.0),
        "max": float(np.max(ordered)),
    }


def current_raw_policy_top1_agreement(
    evaluator: ArtifactEvaluator, rows: list[dict[str, Any]]
) -> float:
    matches = 0
    for row in rows:
        game = KalahGame.from_state(row["raw_state"])
        raw_policy, _raw_logits, _raw_value = artifact_forward_details(evaluator, game)
        legal_moves = [
            move for move, flag in enumerate(row["legal_mask"]) if int(flag) == 1
        ]
        teacher_top = top_policy_move(row["teacher_puct_policy"], legal_moves)
        current_top = top_policy_move(raw_policy, legal_moves)
        if teacher_top == current_top:
            matches += 1
    return matches / max(len(rows), 1)


def dataset_row_seed(base_seed: int, row: dict[str, Any]) -> int:
    return (
        int(base_seed)
        + int(row["challenger_simulations"])
        + int(row["current_simulations"])
        + int(row.get("absolute_ply", 0))
    )


def read_or_build_probe_rows(
    *,
    workdir: Path,
    current_artifact: Path,
    medium_suite: Path,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
) -> list[dict[str, Any]]:
    probe_rows_path = workdir / "medium_probe_rows.jsonl"
    if probe_rows_path.is_file():
        return read_jsonl(probe_rows_path)
    log_progress("building medium-suite probe rows")
    evaluator = ArtifactEvaluator(current_artifact)
    probe_rows = build_probe_rows(
        evaluator=evaluator,
        suite_paths=[medium_suite],
        current_artifact=current_artifact,
        default_c_puct=default_c_puct,
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=tactical_root_bias,
        max_trajectory_plies=12,
        seed=seed,
    )
    write_jsonl(probe_rows_path, probe_rows)
    return probe_rows


def sample_rows(
    rows: list[dict[str, Any]], count: int, *, seed: int
) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return list(rows)
    rng = random.Random(seed)
    indexes = sorted(rng.sample(range(len(rows)), count))
    return [rows[index] for index in indexes]


def verify_dataset_rows(
    *,
    rows: list[dict[str, Any]],
    current_evaluator: ArtifactEvaluator,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
) -> dict[str, Any]:
    sampled_rows = sample_rows(rows, DATASET_AUDIT_SAMPLE_ROWS, seed=seed)
    recompute_rows = sample_rows(sampled_rows, DATASET_RECOMPUTE_ROWS, seed=seed + 1)
    invalid_legal_mask_count = 0
    invalid_policy_count = 0
    illegal_target_top_move_count = 0
    state_encoding_mismatch_count = 0
    target_entropies: list[float] = []
    value_targets: list[float] = []
    top_moves = Counter()
    unique_states = set()
    stored_vs_recomputed_top1_matches = 0
    stored_vs_recomputed_policy_kls: list[float] = []
    stored_vs_recomputed_selected_move_mismatches = 0
    stored_vs_recomputed_value_abs_errors: list[float] = []

    for row in sampled_rows:
        state = row["raw_state"]
        game = KalahGame.from_state(state)
        legal_moves = game.possible_moves()
        legal_mask = [1 if move in legal_moves else 0 for move in range(POLICY_SIZE)]
        stored_legal_mask = [int(flag) for flag in row["legal_mask"]]
        if legal_mask != stored_legal_mask:
            invalid_legal_mask_count += 1
        encoded_state = encode_state(state, input_encoding=INPUT_ENCODING)
        if not np.allclose(
            np.asarray(encoded_state), np.asarray(row["state"]), atol=1e-6
        ):
            state_encoding_mismatch_count += 1
        policy = np.asarray(row["teacher_puct_policy"], dtype=np.float64)
        policy_sum = float(np.sum(policy[legal_moves])) if legal_moves else 0.0
        illegal_mass = float(
            np.sum(
                policy[[move for move in range(POLICY_SIZE) if move not in legal_moves]]
            )
        )
        if (
            not legal_moves
            or not np.isclose(policy_sum, 1.0, atol=1e-6)
            or illegal_mass > 1e-6
            or np.any(policy < -1e-8)
        ):
            invalid_policy_count += 1
        selected_move = row.get("teacher_selected_move")
        if selected_move is None or int(selected_move) not in legal_moves:
            illegal_target_top_move_count += 1
        target_entropies.append(policy_entropy(policy, legal_moves))
        value_targets.append(float(row["value_target"]))
        teacher_top = top_policy_move(policy, legal_moves)
        if teacher_top is not None:
            top_moves[str(teacher_top)] += 1
        unique_states.add(str(row["state_hash"]))

    current_raw_top1 = current_raw_policy_top1_agreement(
        current_evaluator, sampled_rows
    )

    for row in recompute_rows:
        effective_c_puct = resolve_budget_cpuct(
            schedule=cpuct_schedule,
            challenger_simulations=int(row["challenger_simulations"]),
            current_simulations=int(row["current_simulations"]),
            default_c_puct=default_c_puct,
        )
        recomputed = evaluate_teacher_state(
            evaluator=current_evaluator,
            game=KalahGame.from_state(row["raw_state"]),
            challenger_sims=int(row["challenger_simulations"]),
            current_sims=int(row["current_simulations"]),
            effective_c_puct=effective_c_puct,
            tactical_root_bias=tactical_root_bias,
            seed=dataset_row_seed(seed, row),
        )
        legal_moves = recomputed["legal_moves"]
        stored_top = top_policy_move(row["teacher_puct_policy"], legal_moves)
        recomputed_top = top_policy_move(recomputed["teacher_puct_policy"], legal_moves)
        if stored_top == recomputed_top:
            stored_vs_recomputed_top1_matches += 1
        stored_vs_recomputed_policy_kls.append(
            kl_divergence(
                row["teacher_puct_policy"],
                recomputed["teacher_puct_policy"],
                legal_moves,
            )
        )
        if int(row["teacher_selected_move"]) != int(
            recomputed["teacher_selected_move"]
        ):
            stored_vs_recomputed_selected_move_mismatches += 1
        stored_vs_recomputed_value_abs_errors.append(
            abs(
                float(row["teacher_root_value"])
                - float(recomputed["teacher_root_value"])
            )
        )

    recomputed_rows = len(recompute_rows)
    stored_vs_recomputed_top1 = stored_vs_recomputed_top1_matches / max(
        recomputed_rows, 1
    )
    stored_vs_recomputed_mean_kl = (
        float(statistics.fmean(stored_vs_recomputed_policy_kls))
        if stored_vs_recomputed_policy_kls
        else 0.0
    )

    return {
        "schema": SUMMARY_SCHEMA,
        "sampled_rows": len(sampled_rows),
        "row_count": len(rows),
        "unique_state_count": len(unique_states),
        "invalid_legal_mask_count": invalid_legal_mask_count,
        "invalid_policy_count": invalid_policy_count,
        "illegal_target_top_move_count": illegal_target_top_move_count,
        "state_encoding_mismatch_count": state_encoding_mismatch_count,
        "target_entropy_distribution": summarize_distribution(target_entropies),
        "top_move_distribution": dict(sorted(top_moves.items())),
        "value_target_distribution": {
            "mean": float(statistics.fmean(value_targets)) if value_targets else 0.0,
            "positive": sum(1 for value in value_targets if value > 0.0),
            "zero": sum(1 for value in value_targets if value == 0.0),
            "negative": sum(1 for value in value_targets if value < 0.0),
        },
        "current_raw_vs_teacher_top1_agreement": current_raw_top1,
        "stored_vs_recomputed_teacher": {
            "checked_rows": recomputed_rows,
            "top1_agreement": stored_vs_recomputed_top1,
            "selected_move_mismatches": stored_vs_recomputed_selected_move_mismatches,
            "mean_policy_kl": stored_vs_recomputed_mean_kl,
            "max_policy_kl": max(stored_vs_recomputed_policy_kls)
            if stored_vs_recomputed_policy_kls
            else 0.0,
            "mean_root_value_abs_error": float(
                statistics.fmean(stored_vs_recomputed_value_abs_errors)
            )
            if stored_vs_recomputed_value_abs_errors
            else 0.0,
            "max_root_value_abs_error": max(stored_vs_recomputed_value_abs_errors)
            if stored_vs_recomputed_value_abs_errors
            else 0.0,
        },
    }


def dataset_invalid(audit: dict[str, Any]) -> bool:
    recomputed = audit["stored_vs_recomputed_teacher"]
    return any(
        [
            int(audit["invalid_legal_mask_count"]) > 0,
            int(audit["invalid_policy_count"]) > 0,
            int(audit["illegal_target_top_move_count"]) > 0,
            int(audit["state_encoding_mismatch_count"]) > 0,
            float(recomputed["top1_agreement"]) < TARGET_RECOMPUTE_MIN_TOP1,
            float(recomputed["mean_policy_kl"]) > TARGET_RECOMPUTE_MAX_MEAN_KL,
        ]
    )


def build_model(spec: LaneSpec | tuple[int, int]) -> PolicyValueNet:
    hidden_sizes = spec.hidden_sizes if isinstance(spec, LaneSpec) else spec
    return PolicyValueNet(
        hidden_sizes=hidden_sizes,
        model_type=MODEL_TYPE,
        input_size=input_size_for_encoding(INPUT_ENCODING),
    )


def artifact_spec_from_dir(artifact_dir: Path) -> tuple[int, int]:
    metadata = load_json(artifact_dir / "metadata.json")
    architecture = metadata["architecture"]
    return int(architecture["trunk_size"]), int(architecture["residual_block_count"])


def model_predict_runtime_style(
    model: PolicyValueNet,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row["state"] for row in rows], dtype=np.float32)
    legal_mask = np.asarray([row["legal_mask"] for row in rows], dtype=np.float32)
    model.eval()
    policies: list[np.ndarray] = []
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            xb = torch.from_numpy(x[start:end]).to(device)
            logits, value = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            mask = legal_mask[start:end].astype(np.float32)
            masked = probs * mask
            totals = np.sum(masked, axis=1, keepdims=True)
            zero_rows = totals[:, 0] <= 0.0
            if np.any(zero_rows):
                for row_index in np.flatnonzero(zero_rows):
                    legal_moves = np.flatnonzero(mask[row_index] > 0.0)
                    if legal_moves.size > 0:
                        masked[row_index, legal_moves] = 1.0 / float(legal_moves.size)
                        totals[row_index, 0] = 1.0
            policies.append(masked / np.clip(totals, 1e-12, None))
            values.append(value.cpu().numpy().reshape(-1, 1).astype(np.float32))
    return np.concatenate(policies, axis=0), np.concatenate(values, axis=0)


def artifact_predict(
    artifact_dir: Path, rows: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    evaluator = ArtifactEvaluator(artifact_dir)
    policies: list[np.ndarray] = []
    values: list[float] = []
    for row in rows:
        policy, value = evaluator.evaluate(KalahGame.from_state(row["raw_state"]))
        policies.append(policy.astype(np.float32))
        values.append(float(value))
    return np.asarray(policies, dtype=np.float32), np.asarray(
        values, dtype=np.float32
    ).reshape(-1, 1)


def compare_checkpoint_and_artifact(
    *,
    checkpoint_path: Path,
    artifact_dir: Path,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    trunk_size, residual_block_count = artifact_spec_from_dir(artifact_dir)
    model = build_model((trunk_size, residual_block_count)).to(device)
    load_checkpoint_into_model(model, checkpoint_path)
    checkpoint_policies, checkpoint_values = model_predict_runtime_style(
        model, rows, device
    )
    artifact_policies, artifact_values = artifact_predict(artifact_dir, rows)
    legal_moves_list = [
        [move for move, flag in enumerate(row["legal_mask"]) if int(flag) == 1]
        for row in rows
    ]
    top1_matches = 0
    checkpoint_legal_failures = 0
    artifact_legal_failures = 0
    max_policy_diff = 0.0
    mean_policy_diff: list[float] = []
    for index, legal_moves in enumerate(legal_moves_list):
        cp = checkpoint_policies[index]
        ap = artifact_policies[index]
        max_policy_diff = max(max_policy_diff, float(np.max(np.abs(cp - ap))))
        mean_policy_diff.append(float(np.mean(np.abs(cp - ap))))
        if top_policy_move(cp, legal_moves) == top_policy_move(ap, legal_moves):
            top1_matches += 1
        if any(
            cp[move] > 1e-7 for move in range(POLICY_SIZE) if move not in legal_moves
        ):
            checkpoint_legal_failures += 1
        if any(
            ap[move] > 1e-7 for move in range(POLICY_SIZE) if move not in legal_moves
        ):
            artifact_legal_failures += 1
    value_abs_diff = np.abs(checkpoint_values - artifact_values)
    return {
        "checkpoint_path": str(checkpoint_path),
        "artifact_dir": str(artifact_dir),
        "rows": len(rows),
        "max_abs_policy_diff": max_policy_diff,
        "mean_abs_policy_diff": float(statistics.fmean(mean_policy_diff))
        if mean_policy_diff
        else 0.0,
        "top1_agreement": top1_matches / max(len(rows), 1),
        "max_abs_value_diff": float(np.max(value_abs_diff))
        if value_abs_diff.size
        else 0.0,
        "mean_abs_value_diff": float(np.mean(value_abs_diff))
        if value_abs_diff.size
        else 0.0,
        "checkpoint_legal_failures": checkpoint_legal_failures,
        "artifact_legal_failures": artifact_legal_failures,
        "material_mismatch": any(
            [
                max_policy_diff > MAX_POLICY_DIFF_FOR_MATCH,
                (top1_matches / max(len(rows), 1)) < 1.0,
                float(np.max(value_abs_diff)) > MAX_VALUE_DIFF_FOR_MATCH
                if value_abs_diff.size
                else False,
                checkpoint_legal_failures > 0,
                artifact_legal_failures > 0,
            ]
        ),
    }


def latest_epoch_artifact(student_dir: Path) -> tuple[Path, Path] | None:
    epoch = 0
    best_checkpoint = None
    best_artifact = None
    while True:
        candidate_epoch = epoch + 1
        checkpoint = student_dir / f"checkpoint_epoch{candidate_epoch}.npz"
        artifact = student_dir / f"artifact_epoch{candidate_epoch}"
        if not checkpoint.is_file() or not (artifact / "weights.json").is_file():
            break
        best_checkpoint = checkpoint
        best_artifact = artifact
        epoch = candidate_epoch
    if best_checkpoint is None or best_artifact is None:
        return None
    return best_checkpoint, best_artifact


def create_fixed_subsets(
    *, workdir: Path, rows: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    rng = random.Random(seed)
    indexes = list(range(len(rows)))
    rng.shuffle(indexes)
    shuffled = [rows[index] for index in indexes]
    heldout_rows = shuffled[:2048]
    train_pool = shuffled[2048:]
    overfit_2048 = train_pool[:2048]
    overfit_512 = overfit_2048[:512]
    overfit_128 = overfit_2048[:128]
    subset_dir = workdir / "subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "overfit_128_rows": subset_dir / "overfit_128_rows.jsonl",
        "overfit_512_rows": subset_dir / "overfit_512_rows.jsonl",
        "overfit_2048_rows": subset_dir / "overfit_2048_rows.jsonl",
        "heldout_probe_2048_rows": subset_dir / "heldout_probe_2048_rows.jsonl",
    }
    write_jsonl(paths["overfit_128_rows"], overfit_128)
    write_jsonl(paths["overfit_512_rows"], overfit_512)
    write_jsonl(paths["overfit_2048_rows"], overfit_2048)
    write_jsonl(paths["heldout_probe_2048_rows"], heldout_rows)
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "rows": {
            "overfit_128_rows": len(overfit_128),
            "overfit_512_rows": len(overfit_512),
            "overfit_2048_rows": len(overfit_2048),
            "heldout_probe_2048_rows": len(heldout_rows),
        },
    }


def evaluate_policy_value_metrics(
    *, rows: list[dict[str, Any]], policies: np.ndarray, values: np.ndarray
) -> dict[str, Any]:
    top1_matches = 0
    legal_failures = 0
    kls: list[float] = []
    ces: list[float] = []
    value_abs_errors: list[float] = []
    sign_matches = 0
    entropies: list[float] = []
    for index, row in enumerate(rows):
        legal_moves = [
            move for move, flag in enumerate(row["legal_mask"]) if int(flag) == 1
        ]
        policy = policies[index]
        target = row["teacher_puct_policy"]
        if any(
            policy[move] > 1e-7
            for move in range(POLICY_SIZE)
            if move not in legal_moves
        ):
            legal_failures += 1
        if top_policy_move(policy, legal_moves) == top_policy_move(target, legal_moves):
            top1_matches += 1
        kls.append(kl_divergence(target, policy.tolist(), legal_moves))
        ce = 0.0
        for move in legal_moves:
            probability = max(float(policy[move]), KL_EPSILON)
            ce -= float(target[move]) * math.log(probability)
        ces.append(ce)
        predicted_value = float(values[index][0])
        target_value = float(row["value_target"])
        value_abs_errors.append(abs(predicted_value - target_value))
        if (predicted_value >= 0.0) == (target_value >= 0.0):
            sign_matches += 1
        entropies.append(policy_entropy(policy, legal_moves))
    return {
        "rows": len(rows),
        "policy_ce": float(statistics.fmean(ces)) if ces else 0.0,
        "policy_kl": float(statistics.fmean(kls)) if kls else 0.0,
        "top1_agreement": top1_matches / max(len(rows), 1),
        "legal_mask_failures": legal_failures,
        "value_mae": float(statistics.fmean(value_abs_errors))
        if value_abs_errors
        else 0.0,
        "value_sign_accuracy": sign_matches / max(len(rows), 1),
        "mean_output_entropy": float(statistics.fmean(entropies)) if entropies else 0.0,
    }


def save_checkpoint(path: Path, model: PolicyValueNet) -> None:
    np.savez(path, **checkpoint_from_model(model))


def run_overfit_lane(
    *,
    workdir: Path,
    lane: LaneSpec,
    train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    lane_dir = workdir / lane.name
    lane_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model = build_model(lane).to(device)
    if lane.init_checkpoint is not None:
        load_checkpoint_into_model(model, lane.init_checkpoint)
    apply_trainable_scope(model, lane.trainable_scope)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lane.lr,
    )
    x = np.asarray([row["state"] for row in train_rows], dtype=np.float32)
    p = np.asarray([row["teacher_puct_policy"] for row in train_rows], dtype=np.float32)
    v = np.asarray([[row["value_target"]] for row in train_rows], dtype=np.float32)
    replay_indexes = np.arange(len(train_rows), dtype=np.int64)
    effective_batch_size = min(lane.batch_size, len(train_rows))
    steps_per_epoch = math.ceil(len(train_rows) / max(effective_batch_size, 1))
    epochs: list[dict[str, Any]] = []
    best_epoch: dict[str, Any] | None = None

    for epoch in range(1, lane.max_epochs + 1):
        train_result = train_one_epoch(
            model=model,
            optimizer=optimizer,
            compact_x=x,
            compact_p=p,
            compact_v=v,
            replay_indexes=replay_indexes,
            batch_size=effective_batch_size,
            device=device,
            value_loss_weight=1.0,
            value_loss="huber",
            huber_delta=HUBER_DELTA,
            grad_clip=1.0,
        )
        checkpoint_path = lane_dir / f"checkpoint_epoch{epoch}.npz"
        save_checkpoint(checkpoint_path, model)
        artifact_dir = lane_dir / f"artifact_epoch{epoch}"
        export_checkpoint_artifact(
            checkpoint_path=checkpoint_path,
            export_dir=artifact_dir,
            version=f"{lane.name}_epoch{epoch}",
            model_type=MODEL_TYPE,
            input_encoding=INPUT_ENCODING,
            policy_loss=float(train_result["policy_loss"] or 0.0),
            value_loss=float(train_result["value_loss"] or 0.0),
        )
        train_policies, train_values = model_predict_runtime_style(
            model, train_rows, device
        )
        heldout_policies, heldout_values = model_predict_runtime_style(
            model, heldout_rows, device
        )
        train_metrics = evaluate_policy_value_metrics(
            rows=train_rows, policies=train_policies, values=train_values
        )
        heldout_metrics = evaluate_policy_value_metrics(
            rows=heldout_rows, policies=heldout_policies, values=heldout_values
        )
        epoch_metrics = {
            "epoch": epoch,
            "optimizer_updates": epoch * steps_per_epoch,
            "train_policy_ce": train_metrics["policy_ce"],
            "train_policy_kl": train_metrics["policy_kl"],
            "train_top1_agreement": train_metrics["top1_agreement"],
            "train_legal_mask_failures": train_metrics["legal_mask_failures"],
            "train_value_mae": train_metrics["value_mae"],
            "train_value_sign_accuracy": train_metrics["value_sign_accuracy"],
            "heldout_top1_agreement": heldout_metrics["top1_agreement"],
            "heldout_kl": heldout_metrics["policy_kl"],
            "heldout_value_mae": heldout_metrics["value_mae"],
            "heldout_value_sign_accuracy": heldout_metrics["value_sign_accuracy"],
            "gradient_norm": float(train_result["gradient_norm"] or 0.0),
            "training_loop_policy_loss": float(train_result["policy_loss"] or 0.0),
            "training_loop_value_loss": float(train_result["value_loss"] or 0.0),
            "training_loop_total_loss": float(train_result["total_loss"] or 0.0),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "artifact_dir": str(artifact_dir),
            "artifact_weights_sha256": sha256_file(artifact_dir / "weights.json"),
        }
        epochs.append(epoch_metrics)
        if best_epoch is None or (
            epoch_metrics["train_top1_agreement"] > best_epoch["train_top1_agreement"]
            or (
                math.isclose(
                    epoch_metrics["train_top1_agreement"],
                    best_epoch["train_top1_agreement"],
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and epoch_metrics["heldout_kl"] < best_epoch["heldout_kl"]
            )
        ):
            best_epoch = epoch_metrics
        log_progress(
            f"lane={lane.name} epoch={epoch} train_top1={epoch_metrics['train_top1_agreement']:.4f} "
            f"heldout_top1={epoch_metrics['heldout_top1_agreement']:.4f} heldout_kl={epoch_metrics['heldout_kl']:.4f}"
        )
        if epoch_metrics["train_top1_agreement"] >= lane.target_train_top1:
            break

    assert best_epoch is not None
    best_artifact_dir = Path(best_epoch["artifact_dir"])
    return {
        "name": lane.name,
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "model_type": MODEL_TYPE,
        "hidden_sizes": list(lane.hidden_sizes),
        "seed": seed,
        "batch_size": effective_batch_size,
        "lr": lane.lr,
        "steps_per_epoch": steps_per_epoch,
        "trainable_scope": lane.trainable_scope,
        "init_checkpoint": str(lane.init_checkpoint)
        if lane.init_checkpoint is not None
        else None,
        "max_epochs": lane.max_epochs,
        "target_train_top1": lane.target_train_top1,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_artifact_dir": str(best_artifact_dir),
        "best_checkpoint_path": best_epoch["checkpoint_path"],
        "passed_overfit": bool(
            best_epoch["train_top1_agreement"] >= lane.target_train_top1
        ),
    }


def build_probe_recalibration(
    *,
    current_artifact: Path,
    probe_rows: list[dict[str, Any]],
    candidates: list[tuple[str, Path]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    current_metrics = evaluate_probe_candidate(
        artifact_path=current_artifact,
        candidate_name="current_ref",
        probe_rows=probe_rows,
    )
    results["current_ref"] = current_metrics
    for name, artifact_dir in candidates:
        results[name] = evaluate_probe_candidate(
            artifact_path=artifact_dir,
            candidate_name=name,
            probe_rows=probe_rows,
        )
    return results


def build_smoke_suite_subset(workdir: Path, medium_suite: Path) -> Path:
    entries = read_jsonl(medium_suite)[:SMOKE_SUITE_ROWS]
    out_path = workdir / "smoke_medium_subset.jsonl"
    write_jsonl(out_path, entries)
    return out_path


def run_smoke_eval(
    *,
    workdir: Path,
    suite_path: Path,
    current_artifact: Path,
    candidate_artifact: Path,
    seed: int,
) -> dict[str, Any]:
    report = run_opening_suite_benchmark(
        workdir=str(workdir),
        suite=str(suite_path),
        current=str(current_artifact),
        candidates=f"{current_artifact},{candidate_artifact}",
        budget_pairs="384:256,768:768,1200:1200",
        games_per_opening=SMOKE_GAMES_PER_OPENING,
        seed=seed,
        workers=4,
        timeout=7200,
    )
    results: dict[str, Any] = {}
    for label, path in (
        ("current_ref", current_artifact),
        (candidate_artifact.name, candidate_artifact),
    ):
        candidate_report = find_candidate_report(
            report, path.name
        ) or find_candidate_report(report, label)
        if candidate_report is None:
            continue
        results[label] = benchmark_budget_results(candidate_report)
    return {"suite_path": str(suite_path), "report": report, "results": results}


def classify_run(
    *,
    dataset_audit: dict[str, Any],
    checkpoint_artifact_consistency: dict[str, Any],
    lanes: dict[str, Any],
    probe_recalibration: dict[str, Any],
) -> str:
    if dataset_invalid(dataset_audit):
        return "target_dataset_invalid"
    if any(
        entry.get("material_mismatch")
        for entry in checkpoint_artifact_consistency.values()
    ):
        return "export_or_probe_mismatch"
    lane_128 = lanes.get("scratch_96x3_128rows")
    if lane_128 is None or not lane_128.get("passed_overfit"):
        return "training_loop_broken"
    lane_512 = lanes.get("scratch_96x3_512rows")
    current_probe = probe_recalibration["current_ref"]
    best_lane_probe = None
    for lane_name in (
        "scratch_96x3_512rows",
        "current_init_96x3_512rows",
        "scratch_128x4_512rows",
    ):
        probe = probe_recalibration.get(lane_name)
        if probe is None:
            continue
        if (
            best_lane_probe is None
            or probe["top1_agreement"] > best_lane_probe["top1_agreement"]
        ):
            best_lane_probe = probe
    if lane_512 is not None and lane_512.get("passed_overfit"):
        if best_lane_probe is not None and (
            best_lane_probe["top1_agreement"] > current_probe["top1_agreement"]
            or best_lane_probe["policy_kl"] < current_probe["policy_kl"]
        ):
            return "student_preflight_retry_ready"
        return "objective_hard_but_pipeline_valid"
    return "architecture_change_not_ready"


def render_report(summary: dict[str, Any]) -> str:
    dataset_audit = summary["dataset_integrity_audit"]
    stored_recomputed = dataset_audit["stored_vs_recomputed_teacher"]
    checkpoint_rows = []
    for name, entry in sorted(summary["checkpoint_vs_artifact_consistency"].items()):
        checkpoint_rows.append(
            [
                name,
                fmt(entry["max_abs_policy_diff"], 6),
                fmt(entry["top1_agreement"], 4),
                fmt(entry["max_abs_value_diff"], 6),
                entry["checkpoint_legal_failures"],
                entry["artifact_legal_failures"],
                entry["material_mismatch"],
            ]
        )
    lane_rows: list[list[Any]] = []
    for lane_name, lane in summary["overfit_lanes"].items():
        best = lane["best_epoch"]
        lane_rows.append(
            [
                lane_name,
                best["epoch"],
                best["optimizer_updates"],
                fmt(best["train_policy_ce"]),
                fmt(best["train_policy_kl"]),
                fmt(best["train_top1_agreement"]),
                fmt(best["heldout_top1_agreement"]),
                fmt(best["heldout_kl"]),
                fmt(best["gradient_norm"]),
                lane["passed_overfit"],
            ]
        )
    probe_rows = []
    for name, metrics in summary["probe_gate_recalibration"].items():
        probe_rows.append(
            [
                name,
                fmt(metrics["top1_agreement"]),
                fmt(metrics["policy_kl"]),
                fmt(metrics["mean_output_entropy"]),
                fmt(metrics["value_mae"]),
                fmt(metrics["value_sign_accuracy"]),
                metrics["legal_mask_failures"],
                fmt(
                    metrics["top1_agreement"]
                    - summary["probe_gate_recalibration"]["current_ref"][
                        "top1_agreement"
                    ]
                ),
            ]
        )
    lines = [
        "# AlphaZero-Lite Search-Teacher Distillation Sanity Results",
        "",
        f"**Classification**: `{summary['classification']}`",
        "",
        "## PR #150 Artifact And Dataset Hashes",
        "",
        f"- current weights SHA256: `{summary['current_hash']['actual_sha256']}`",
        f"- teacher dataset SHA256: `{summary['inputs']['teacher_dataset']['sha256']}`",
        f"- teacher dataset audit SHA256: `{summary['inputs']['teacher_dataset_audit']['sha256']}`",
        f"- medium suite SHA256: `{summary['inputs']['medium_suite']['sha256']}`",
        "",
        "## Dataset Integrity Audit",
        "",
        f"- sampled rows: `{dataset_audit['sampled_rows']}`",
        f"- unique state count: `{dataset_audit['unique_state_count']}`",
        f"- invalid legal-mask count: `{dataset_audit['invalid_legal_mask_count']}`",
        f"- invalid policy count: `{dataset_audit['invalid_policy_count']}`",
        f"- illegal target top-move count: `{dataset_audit['illegal_target_top_move_count']}`",
        f"- state encoding mismatch count: `{dataset_audit['state_encoding_mismatch_count']}`",
        f"- target entropy distribution: `{json.dumps(dataset_audit['target_entropy_distribution'], sort_keys=True)}`",
        f"- top-move distribution: `{json.dumps(dataset_audit['top_move_distribution'], sort_keys=True)}`",
        f"- value target distribution: `{json.dumps(dataset_audit['value_target_distribution'], sort_keys=True)}`",
        f"- current raw vs teacher top-1 agreement: `{dataset_audit['current_raw_vs_teacher_top1_agreement']:.4f}`",
        "",
        "## Stored-Vs-Recomputed Teacher Check",
        "",
        f"- checked rows: `{stored_recomputed['checked_rows']}`",
        f"- top-1 agreement: `{stored_recomputed['top1_agreement']:.4f}`",
        f"- selected-move mismatches: `{stored_recomputed['selected_move_mismatches']}`",
        f"- mean policy KL: `{stored_recomputed['mean_policy_kl']:.4f}`",
        f"- max policy KL: `{stored_recomputed['max_policy_kl']:.4f}`",
        f"- mean root-value abs error: `{stored_recomputed['mean_root_value_abs_error']:.4f}`",
        "",
        "## Checkpoint-Vs-Artifact Consistency",
        "",
        markdown_table(
            [
                "Candidate",
                "Max |policy diff|",
                "Top-1 agreement",
                "Max |value diff|",
                "Checkpoint legal fails",
                "Artifact legal fails",
                "Mismatch",
            ],
            checkpoint_rows,
        ),
        "",
        "## Overfit Training Curves",
        "",
        "- overfit lanes use explicit per-lane learning-rate and batch-size settings; optimizer updates are reported to avoid misleading epoch-only comparisons",
        markdown_table(
            [
                "Lane",
                "Best epoch",
                "Best updates",
                "Train CE",
                "Train KL",
                "Train top-1",
                "Heldout top-1",
                "Heldout KL",
                "Grad norm",
                "Passed",
            ],
            lane_rows,
        ),
        "",
        "## Probe-Gate Recalibration Table",
        "",
        markdown_table(
            [
                "Candidate",
                "Top-1",
                "KL",
                "Entropy",
                "Value MAE",
                "Sign acc",
                "Legal fails",
                "Delta vs current",
            ],
            probe_rows,
        ),
        "",
        "## Optional Tiny Suite Smoke",
        "",
    ]
    smoke = summary.get("optional_tiny_medium_suite_smoke")
    if smoke is None:
        lines.append("- not run")
    else:
        lines.append(f"- suite: `{smoke['suite_path']}`")
        lines.append(f"- candidates: `{json.dumps(smoke['results'], sort_keys=True)}`")
    lines.extend(
        [
            "",
            "## Final Classification",
            "",
            f"- result: `{summary['classification']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--pr150-workdir", required=True)
    parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--teacher-dataset-audit", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--default-c-puct", type=float, default=1.25)
    parser.add_argument("--cpuct-schedule", default="{}")
    parser.add_argument("--tactical-root-bias", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    current_artifact = Path(args.current)
    pr150_workdir = Path(args.pr150_workdir)
    teacher_dataset = Path(args.teacher_dataset)
    teacher_dataset_audit = Path(args.teacher_dataset_audit)
    medium_suite = Path(args.medium_suite)
    require_existing_file(current_artifact / "weights.json", "current artifact weights")
    require_existing_file(teacher_dataset, "teacher dataset")
    require_existing_file(teacher_dataset_audit, "teacher dataset audit")
    require_existing_file(medium_suite, "medium suite")
    device = select_device(args.device)
    cpuct_schedule = parse_cpuct_schedule_json(args.cpuct_schedule)
    current_hash = verify_expected_hash(
        current_artifact / "weights.json",
        args.expected_current_weights_sha256,
        "current weights",
    )
    dataset_rows = read_jsonl(teacher_dataset)
    current_evaluator = ArtifactEvaluator(current_artifact)

    log_progress("phase A dataset integrity audit")
    dataset_integrity_audit = verify_dataset_rows(
        rows=dataset_rows,
        current_evaluator=current_evaluator,
        default_c_puct=float(args.default_c_puct),
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=float(args.tactical_root_bias),
        seed=int(args.seed),
    )
    if dataset_invalid(dataset_integrity_audit):
        classification = "target_dataset_invalid"
        summary = {
            "schema": SUMMARY_SCHEMA,
            "inputs": {
                "workdir": str(workdir),
                "current": str(current_artifact),
                "pr150_workdir": str(pr150_workdir),
                "teacher_dataset": {
                    "path": str(teacher_dataset),
                    "sha256": sha256_file(teacher_dataset),
                    "rows": count_jsonl_rows(teacher_dataset),
                },
                "teacher_dataset_audit": {
                    "path": str(teacher_dataset_audit),
                    "sha256": sha256_file(teacher_dataset_audit),
                },
                "medium_suite": {
                    "path": str(medium_suite),
                    "sha256": sha256_file(medium_suite),
                    "rows": count_jsonl_rows(medium_suite),
                },
                "runtime_profile": {
                    "tactical_root_bias": float(args.tactical_root_bias),
                    "default_c_puct": float(args.default_c_puct),
                    "c_puct_schedule": schedule_definition(
                        default_c_puct=float(args.default_c_puct),
                        schedule=cpuct_schedule,
                    ),
                    "root_policy_mode": "deterministic",
                    "root_prior_transform": None,
                    "value_transform": None,
                },
                "seed": int(args.seed),
            },
            "current_hash": current_hash,
            "dataset_integrity_audit": dataset_integrity_audit,
            "checkpoint_vs_artifact_consistency": {},
            "subsets": {},
            "overfit_lanes": {},
            "probe_gate_recalibration": {},
            "optional_tiny_medium_suite_smoke": None,
            "classification": classification,
        }
        write_json(workdir / SUMMARY_FILENAME, summary)
        REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
        return

    log_progress("phase B checkpoint/export consistency")
    probe_subset_rows = sample_rows(
        dataset_rows, CHECKPOINT_ARTIFACT_SAMPLE_ROWS, seed=args.seed + 2
    )
    checkpoint_vs_artifact_consistency: dict[str, Any] = {}
    for candidate_name in ("residual_v3_96x3", "residual_v3_128x4"):
        latest = latest_epoch_artifact(pr150_workdir / candidate_name)
        if latest is None:
            continue
        checkpoint_path, artifact_dir = latest
        checkpoint_vs_artifact_consistency[candidate_name] = (
            compare_checkpoint_and_artifact(
                checkpoint_path=checkpoint_path,
                artifact_dir=artifact_dir,
                rows=probe_subset_rows,
                device=device,
            )
        )

    log_progress("phase C fixed subsets and overfit lanes")
    subsets = create_fixed_subsets(workdir=workdir, rows=dataset_rows, seed=args.seed)
    overfit_128_rows = read_jsonl(Path(subsets["paths"]["overfit_128_rows"]))
    overfit_512_rows = read_jsonl(Path(subsets["paths"]["overfit_512_rows"]))
    heldout_probe_rows = read_jsonl(Path(subsets["paths"]["heldout_probe_2048_rows"]))
    lanes: dict[str, Any] = {}
    scratch_96x3_128 = LaneSpec(
        name="scratch_96x3_128rows",
        trunk_size=96,
        residual_block_count=3,
        train_rows=128,
        max_epochs=100,
        target_train_top1=0.95,
        lr=3e-3,
        batch_size=128,
    )
    lanes[scratch_96x3_128.name] = run_overfit_lane(
        workdir=workdir,
        lane=scratch_96x3_128,
        train_rows=overfit_128_rows,
        heldout_rows=heldout_probe_rows,
        device=device,
        seed=args.seed,
    )
    scratch_96x3_512 = LaneSpec(
        name="scratch_96x3_512rows",
        trunk_size=96,
        residual_block_count=3,
        train_rows=512,
        max_epochs=50,
        target_train_top1=0.90,
        lr=1e-2,
        batch_size=128,
    )
    lanes[scratch_96x3_512.name] = run_overfit_lane(
        workdir=workdir,
        lane=scratch_96x3_512,
        train_rows=overfit_512_rows,
        heldout_rows=heldout_probe_rows,
        device=device,
        seed=args.seed,
    )
    current_checkpoint = current_checkpoint_path(current_artifact, workdir)
    current_init_lane = LaneSpec(
        name="current_init_96x3_512rows",
        trunk_size=96,
        residual_block_count=3,
        train_rows=512,
        max_epochs=20,
        target_train_top1=0.90,
        lr=3e-3,
        batch_size=128,
        init_checkpoint=current_checkpoint,
        trainable_scope="all",
    )
    lanes[current_init_lane.name] = run_overfit_lane(
        workdir=workdir,
        lane=current_init_lane,
        train_rows=overfit_512_rows,
        heldout_rows=heldout_probe_rows,
        device=device,
        seed=args.seed,
    )
    if lanes[scratch_96x3_128.name]["passed_overfit"]:
        scratch_128x4_512 = LaneSpec(
            name="scratch_128x4_512rows",
            trunk_size=128,
            residual_block_count=4,
            train_rows=512,
            max_epochs=50,
            target_train_top1=0.90,
            lr=1e-2,
            batch_size=128,
        )
        lanes[scratch_128x4_512.name] = run_overfit_lane(
            workdir=workdir,
            lane=scratch_128x4_512,
            train_rows=overfit_512_rows,
            heldout_rows=heldout_probe_rows,
            device=device,
            seed=args.seed,
        )

    log_progress("phase D probe-gate recalibration")
    probe_rows = read_or_build_probe_rows(
        workdir=workdir,
        current_artifact=current_artifact,
        medium_suite=medium_suite,
        default_c_puct=float(args.default_c_puct),
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=float(args.tactical_root_bias),
        seed=int(args.seed),
    )
    candidate_probe_artifacts: list[tuple[str, Path]] = []
    for lane_name, lane_result in lanes.items():
        candidate_probe_artifacts.append(
            (lane_name, Path(lane_result["best_artifact_dir"]))
        )
    for candidate_name in ("residual_v3_96x3", "residual_v3_128x4"):
        latest = latest_epoch_artifact(pr150_workdir / candidate_name)
        if latest is not None:
            _checkpoint_path, artifact_dir = latest
            candidate_probe_artifacts.append((candidate_name, artifact_dir))
    probe_gate_recalibration = build_probe_recalibration(
        current_artifact=current_artifact,
        probe_rows=probe_rows,
        candidates=candidate_probe_artifacts,
    )

    smoke_result = None
    overfit_ok = all(
        lanes[name]["passed_overfit"]
        for name in ("scratch_96x3_128rows", "scratch_96x3_512rows")
        if name in lanes
    )
    if overfit_ok and not any(
        entry.get("material_mismatch")
        for entry in checkpoint_vs_artifact_consistency.values()
    ):
        log_progress("phase E tiny medium-suite smoke")
        smoke_suite = build_smoke_suite_subset(workdir, medium_suite)
        smoke_candidate = Path(lanes["scratch_96x3_512rows"]["best_artifact_dir"])
        smoke_result = run_smoke_eval(
            workdir=workdir / "smoke_eval",
            suite_path=smoke_suite,
            current_artifact=current_artifact,
            candidate_artifact=smoke_candidate,
            seed=args.seed,
        )

    classification = classify_run(
        dataset_audit=dataset_integrity_audit,
        checkpoint_artifact_consistency=checkpoint_vs_artifact_consistency,
        lanes=lanes,
        probe_recalibration=probe_gate_recalibration,
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "inputs": {
            "workdir": str(workdir),
            "current": str(current_artifact),
            "pr150_workdir": str(pr150_workdir),
            "teacher_dataset": {
                "path": str(teacher_dataset),
                "sha256": sha256_file(teacher_dataset),
                "rows": count_jsonl_rows(teacher_dataset),
            },
            "teacher_dataset_audit": {
                "path": str(teacher_dataset_audit),
                "sha256": sha256_file(teacher_dataset_audit),
            },
            "medium_suite": {
                "path": str(medium_suite),
                "sha256": sha256_file(medium_suite),
                "rows": count_jsonl_rows(medium_suite),
            },
            "runtime_profile": {
                "tactical_root_bias": float(args.tactical_root_bias),
                "default_c_puct": float(args.default_c_puct),
                "c_puct_schedule": schedule_definition(
                    default_c_puct=float(args.default_c_puct), schedule=cpuct_schedule
                ),
                "root_policy_mode": "deterministic",
                "root_prior_transform": None,
                "value_transform": None,
            },
            "seed": int(args.seed),
        },
        "current_hash": current_hash,
        "dataset_integrity_audit": dataset_integrity_audit,
        "checkpoint_vs_artifact_consistency": checkpoint_vs_artifact_consistency,
        "subsets": subsets,
        "overfit_lanes": lanes,
        "probe_gate_recalibration": probe_gate_recalibration,
        "optional_tiny_medium_suite_smoke": smoke_result,
        "classification": classification,
    }
    summary_path = workdir / SUMMARY_FILENAME
    write_json(summary_path, summary)
    REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    log_progress(f"wrote summary={summary_path} report={REPORT_PATH}")


if __name__ == "__main__":
    main()
