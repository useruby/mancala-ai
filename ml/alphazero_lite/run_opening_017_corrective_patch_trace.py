#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.build_opening_017_corrective_artifact import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SELF_PLAY_PATH,
    build_artifact,
    top_policy_move,
)
from ml.alphazero_lite.corrected_guard_kill_gate import (
    DEFAULT_FALLBACK_REFERENCE_ARTIFACT,
    DEFAULT_REFERENCE_ARTIFACT,
    GUARD_ROW_IDS,
    run_corrected_guard_kill_gate,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_guard_safe_opening_full_pipeline_interaction_trace import (
    build_train_spec,
    load_json,
)
from ml.alphazero_lite.run_guard_safe_opening_low_epoch_drift_trace import (
    artifact_row_id,
    choose_device,
    compute_policy_cross_entropy,
    compute_value_loss_vector,
    current_checkpoint_path,
    export_checkpoint_artifact,
    metric_rank,
    policy_entropy,
    read_jsonl,
    train_one_epoch,
    write_json,
)
from ml.alphazero_lite.train import (
    PolicyValueNet,
    checkpoint_from_model,
    input_size_for_encoding,
    load_checkpoint_into_model,
    load_jsonl_replay,
    set_seed,
    split_replay_positions_by_source_row,
)


DEFAULT_CONFIG_PATH = Path(
    "ml/alphazero_lite/configs/exp_v3_guard_safe_opening_selected_w1.json"
)
DEFAULT_CURRENT_PATH = Path("storage/ai/alphazero_lite/current")
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_ROOT / "opening_017_corrective_patch_summary.json"
DEFAULT_REPORT_PATH = Path(
    "docs/alphazero-lite-opening-017-corrective-patch-results.md"
)
SCHEMA = "azlite_opening_017_corrective_patch_trace_v1"
TARGET_EPOCHS = (1, 2, 4)
DEFAULT_SEED = 42
POLICY_ROW_IDS = (
    "opening_plies_1_8-017",
    "capture_available-002",
    "capture_available-003",
    "capture_available-007",
    "capture_available-006",
    "capture_available-008",
)


@dataclass(frozen=True)
class TraceSpec:
    name: str
    data_files: tuple[Path, ...]
    replay_weights: tuple[int, ...]
    epochs: tuple[int, ...]
    status: str = "planned"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--current-path", type=Path, default=DEFAULT_CURRENT_PATH)
    parser.add_argument("--self-play", type=Path, default=DEFAULT_SELF_PLAY_PATH)
    parser.add_argument(
        "--reference-artifact", type=Path, default=DEFAULT_REFERENCE_ARTIFACT
    )
    parser.add_argument(
        "--fallback-reference-artifact",
        type=Path,
        default=DEFAULT_FALLBACK_REFERENCE_ARTIFACT,
    )
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def merged_reference_rows(
    reference_artifact_path: Path,
    fallback_reference_artifact_path: Path,
) -> dict[str, dict[str, Any]]:
    primary = load_json(reference_artifact_path)
    fallback = load_json(fallback_reference_artifact_path)
    rows: dict[str, dict[str, Any]] = {}
    for row in list(fallback.get("rows") or []):
        row_id = row.get("id")
        if isinstance(row_id, str) and row_id:
            rows[row_id] = dict(row)
    for row in list(primary.get("rows") or []):
        row_id = row.get("id")
        if isinstance(row_id, str) and row_id:
            rows[row_id] = dict(row)
    return rows


def build_trace_specs(
    self_play_path: Path, corrective_artifact_path: Path
) -> list[TraceSpec]:
    return [
        TraceSpec(
            name="self_play_only_reproduction",
            data_files=(self_play_path,),
            replay_weights=(1,),
            epochs=TARGET_EPOCHS,
        ),
        TraceSpec(
            name="self_play_plus_corrective_patch_w1",
            data_files=(self_play_path, corrective_artifact_path),
            replay_weights=(1, 1),
            epochs=TARGET_EPOCHS,
        ),
        TraceSpec(
            name="self_play_plus_corrective_patch_w2",
            data_files=(self_play_path, corrective_artifact_path),
            replay_weights=(1, 2),
            epochs=TARGET_EPOCHS,
        ),
    ]


def build_row_infos(
    paths: tuple[Path, ...], corrective_artifact_path: Path
) -> list[dict[str, Any]]:
    row_infos: list[dict[str, Any]] = []
    for path in paths:
        source_kind = (
            "corrective_patch" if path == corrective_artifact_path else "self_play"
        )
        for row in read_jsonl(path):
            row_infos.append(
                {
                    "row_id": artifact_row_id(row),
                    "source_kind": source_kind,
                }
            )
    return row_infos


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(np.mean(values)), 6)


def evaluate_training_snapshot(
    *,
    model: PolicyValueNet,
    compact_x: np.ndarray,
    compact_p: np.ndarray,
    compact_v: np.ndarray,
    row_infos: list[dict[str, Any]],
    val_replay_indexes: np.ndarray,
    device: torch.device,
    train_spec: Any,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(compact_x).to(device)
        p_tensor = torch.from_numpy(compact_p).to(device)
        v_tensor = torch.from_numpy(compact_v).to(device)
        logits, value_pred = model(x_tensor)
        policy_ce = (
            compute_policy_cross_entropy(logits, p_tensor).detach().cpu().numpy()
        )
        value_losses = (
            compute_value_loss_vector(
                value_pred,
                v_tensor,
                value_loss=train_spec.value_loss,
                huber_delta=train_spec.huber_delta,
            )
            .detach()
            .cpu()
            .numpy()
        )
        if val_replay_indexes.size > 0:
            index_tensor = torch.from_numpy(val_replay_indexes).to(device)
            val_logits, val_value_pred = model(x_tensor[index_tensor])
            val_policy_ce = compute_policy_cross_entropy(
                val_logits, p_tensor[index_tensor]
            )
            val_value_losses = compute_value_loss_vector(
                val_value_pred,
                v_tensor[index_tensor],
                value_loss=train_spec.value_loss,
                huber_delta=train_spec.huber_delta,
            )
            val_total = val_policy_ce + (
                train_spec.value_loss_weight * val_value_losses
            )
            val_loss = round(float(torch.mean(val_total).detach().cpu().item()), 6)
        else:
            val_loss = None
    total_losses = policy_ce + (train_spec.value_loss_weight * value_losses)
    self_play_losses = [
        float(policy_ce[index])
        for index, info in enumerate(row_infos)
        if info.get("source_kind") == "self_play"
    ]
    corrective_patch_losses = [
        float(policy_ce[index])
        for index, info in enumerate(row_infos)
        if info.get("source_kind") == "corrective_patch"
    ]
    guard_losses = [
        float(policy_ce[index])
        for index, info in enumerate(row_infos)
        if info.get("row_id") in GUARD_ROW_IDS
    ]
    return {
        "policy_loss": round(float(np.mean(policy_ce)), 6),
        "value_loss": round(float(np.mean(value_losses)), 6),
        "total_loss": round(float(np.mean(total_losses)), 6),
        "self_play_cross_entropy": mean_or_none(self_play_losses),
        "corrective_patch_cross_entropy": mean_or_none(corrective_patch_losses),
        "guard_cross_entropy": mean_or_none(guard_losses),
        "val_loss": val_loss,
    }


def evaluate_policy_rows(
    *,
    artifact_path: Path,
    reference_rows: dict[str, dict[str, Any]],
    old_self_play_top_moves: dict[str, int | None],
) -> list[dict[str, Any]]:
    evaluator = ArtifactEvaluator(artifact_path)
    rows: list[dict[str, Any]] = []
    for row_id in POLICY_ROW_IDS:
        reference_row = reference_rows[row_id]
        legal_moves = [
            int(move)
            for move in KalahGame.from_state(
                dict(reference_row["state"])
            ).possible_moves()
        ]
        policy, _value = evaluator.evaluate(
            KalahGame.from_state(dict(reference_row["state"]))
        )
        distribution = {move: float(policy[move]) for move in legal_moves}
        reference_move = int(reference_row["reference_move"])
        old_top_move = old_self_play_top_moves.get(row_id)
        old_top_probability = (
            round(float(distribution[old_top_move]), 4)
            if old_top_move is not None and old_top_move in distribution
            else None
        )
        reference_probability = round(float(distribution.get(reference_move, 0.0)), 4)
        margin = (
            None
            if old_top_probability is None
            else round(reference_probability - old_top_probability, 4)
        )
        rows.append(
            {
                "row_id": row_id,
                "corrected_reference_move": reference_move,
                "policy_top_move": top_policy_move(list(policy), legal_moves),
                "reference_policy_probability": reference_probability,
                "reference_policy_rank": metric_rank(distribution, reference_move),
                "entropy": policy_entropy(list(policy), legal_moves),
                "old_self_play_top_move": old_top_move,
                "old_top_policy_probability": old_top_probability,
                "reference_minus_old_top_margin": margin,
                "notes": "no_self_play_match"
                if old_top_move is None
                else (
                    "reference_policy_top"
                    if top_policy_move(list(policy), legal_moves) == reference_move
                    else "policy_top_not_reference"
                ),
            }
        )
    return rows


def run_trace(
    *,
    spec: TraceSpec,
    init_checkpoint: Path,
    current_metadata: dict[str, Any],
    train_spec: Any,
    output_root: Path,
    seed: int,
    device: torch.device,
    corrective_artifact_path: Path,
    reference_artifact_path: Path,
    fallback_reference_artifact_path: Path,
    reference_rows: dict[str, dict[str, Any]],
    old_self_play_top_moves: dict[str, int | None],
) -> dict[str, Any]:
    set_seed(seed)
    compact_x, compact_p, compact_v, replay_indexes = load_jsonl_replay(
        list(spec.data_files),
        list(spec.replay_weights),
        policy_target_mode=train_spec.policy_target_mode,
        value_target_mode=train_spec.value_target_mode,
    )
    compact_v = compact_v.astype(np.float32)
    row_infos = build_row_infos(spec.data_files, corrective_artifact_path)
    model = PolicyValueNet(
        hidden_sizes=train_spec.hidden_sizes,
        model_type=train_spec.model_type,
        input_size=input_size_for_encoding(train_spec.input_encoding),
    )
    load_checkpoint_into_model(model, init_checkpoint)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_spec.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, max(spec.epochs))
    )
    train_positions, val_positions = split_replay_positions_by_source_row(
        replay_indexes, val_split=train_spec.val_split
    )
    weighted_train_indexes = replay_indexes[train_positions]
    weighted_val_indexes = replay_indexes[val_positions]
    checkpoints_root = output_root / "checkpoints" / spec.name
    exports_root = output_root / "exports" / spec.name
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    exports_root.mkdir(parents=True, exist_ok=True)

    guard_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for epoch in range(1, max(spec.epochs) + 1):
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            compact_x=compact_x,
            compact_p=compact_p,
            compact_v=compact_v,
            replay_indexes=weighted_train_indexes,
            batch_size=train_spec.batch_size,
            device=device,
            value_loss_weight=train_spec.value_loss_weight,
            value_loss=train_spec.value_loss,
            huber_delta=train_spec.huber_delta,
            grad_clip=train_spec.grad_clip,
        )
        scheduler.step()
        if epoch not in spec.epochs:
            continue
        checkpoint_path = checkpoints_root / f"epoch_{epoch}.npz"
        np.savez(checkpoint_path, **checkpoint_from_model(model))
        export_dir = export_checkpoint_artifact(
            checkpoint_path=checkpoint_path,
            export_dir=exports_root / f"epoch_{epoch}",
            current_metadata=current_metadata,
            version=f"{spec.name}-epoch-{epoch}",
        )
        guard_payload = run_corrected_guard_kill_gate(
            candidate_path=export_dir,
            reference_artifact=reference_artifact_path,
            fallback_reference_artifact=fallback_reference_artifact_path,
            seed=seed + epoch,
        )
        for row in list(guard_payload["rows"]):
            guard_rows.append(
                {
                    "trace_name": spec.name,
                    "epoch": epoch,
                    "row_id": row["row_id"],
                    "corrected_reference_move": row["corrected_reference_move"],
                    "selected_move_384": row["selected_move_384"],
                    "selected_move_1200": row["selected_move_1200"],
                    "reference_visit_share_384": row.get("reference_visit_share_384"),
                    "reference_visit_share_1200": row.get("reference_visit_share_1200"),
                    "selected_minus_reference_q_margin_384": row.get(
                        "selected_minus_reference_q_margin_384"
                    ),
                    "selected_minus_reference_q_margin_1200": row.get(
                        "selected_minus_reference_q_margin_1200"
                    ),
                    "row_pass": bool(row["pass"]),
                    "gate_pass": bool(guard_payload["pass"]),
                    "notes": row.get("notes", "ok"),
                }
            )
        for row in evaluate_policy_rows(
            artifact_path=export_dir,
            reference_rows=reference_rows,
            old_self_play_top_moves=old_self_play_top_moves,
        ):
            policy_rows.append({"trace_name": spec.name, "epoch": epoch, **row})
        metrics = evaluate_training_snapshot(
            model=model,
            compact_x=compact_x,
            compact_p=compact_p,
            compact_v=compact_v,
            row_infos=row_infos,
            val_replay_indexes=weighted_val_indexes,
            device=device,
            train_spec=train_spec,
        )
        metric_rows.append(
            {
                "trace_name": spec.name,
                "epoch": epoch,
                **metrics,
                "notes": "full_dataset_snapshot",
            }
        )
    return {
        "trace_name": spec.name,
        "data_files": [str(path) for path in spec.data_files],
        "replay_weights": list(spec.replay_weights),
        "epochs": list(spec.epochs),
        "status": "completed",
        "guard_rows": guard_rows,
        "policy_rows": policy_rows,
        "metric_rows": metric_rows,
    }


def epoch_gate_pass(
    guard_rows: list[dict[str, Any]], trace_name: str, epoch: int
) -> bool:
    matches = [
        row
        for row in guard_rows
        if row["trace_name"] == trace_name and int(row["epoch"]) == epoch
    ]
    return bool(matches) and all(bool(row["row_pass"]) for row in matches)


def epoch_row_pass_map(
    guard_rows: list[dict[str, Any]], trace_name: str, epoch: int
) -> dict[str, bool]:
    return {
        row["row_id"]: bool(row["row_pass"])
        for row in guard_rows
        if row["trace_name"] == trace_name and int(row["epoch"]) == epoch
    }


def weak_policy_rows(
    policy_rows: list[dict[str, Any]], trace_name: str, epoch: int
) -> list[str]:
    weak: list[str] = []
    for row in policy_rows:
        if row["trace_name"] != trace_name or int(row["epoch"]) != epoch:
            continue
        if float(row["reference_policy_probability"]) < 0.5:
            weak.append(str(row["row_id"]))
    return weak


def classify_result(
    *,
    guard_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    trace_names: list[str],
) -> dict[str, Any]:
    epoch = 4
    partial_fix_traces: list[str] = []
    for trace_name in trace_names:
        row_passes = epoch_row_pass_map(guard_rows, trace_name, epoch)
        if not row_passes:
            continue
        if all(
            row_passes.get(row_id, False)
            for row_id in ("capture_available-002", "capture_available-003")
        ) and not all(
            row_passes.get(row_id, False)
            for row_id in (
                "capture_available-006",
                "capture_available-007",
                "capture_available-008",
            )
        ):
            partial_fix_traces.append(trace_name)
    if partial_fix_traces:
        return {
            "classification": "patch_overfit_or_guard_tradeoff",
            "next_action": "expand corrective artifact to include sibling preservation rows before production training.",
            "notes": "002/003 recovered but at least one control row still failed at epoch 4.",
        }

    for trace_name, classification, next_action in (
        (
            "self_play_plus_corrective_patch_w1",
            "small_corrective_patch_sufficient",
            "run one controlled production-scale lane with PR #36 self-play plus corrective patch w1, with corrected guard kill gate before arena.",
        ),
        (
            "self_play_plus_corrective_patch_w2",
            "corrective_patch_weight_needed",
            "run one controlled production-scale lane with corrective patch w2, no broader replay sweep.",
        ),
        (
            "self_play_plus_corrective_patch_w4",
            "corrective_patch_heavy_anchor_needed",
            "do not production-train yet; inspect whether the self-play target generator is too noisy around opening_plies_1_8-017.",
        ),
    ):
        if trace_name not in trace_names or not epoch_gate_pass(
            guard_rows, trace_name, epoch
        ):
            continue
        weak_rows = weak_policy_rows(policy_rows, trace_name, epoch)
        if weak_rows:
            return {
                "classification": "search_recovered_policy_still_weak",
                "next_action": "do not train full lane yet; refine target construction or add policy-only regularization.",
                "notes": f"guard passed at epoch 4, but corrected-reference policy stayed below 0.5 on {', '.join(sorted(weak_rows))}.",
            }
        return {
            "classification": classification,
            "next_action": next_action,
            "notes": f"{trace_name} passed the corrected guard kill gate at epoch 4.",
        }

    return {
        "classification": "self_play_target_generation_broken",
        "next_action": "audit PUCT target generation for opening_plies_1_8-017 and descendants, not replay anchoring.",
        "notes": "No corrective replay weight passed the corrected guard kill gate at epoch 4.",
    }


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-lite Opening 017 Corrective Patch Results",
        "",
        "## 1. Context",
        "",
        "- PR #37 classified the merged failure chain as `self_play_background_causes_guard_regression`.",
        "- This follow-up builds a minimal train-only corrective artifact around `opening_plies_1_8-017` and descendant guard rows, then reruns low-cost guarded training traces from `storage/ai/alphazero_lite/current`.",
        "- This run did not use arena, did not run any production MCTS1200 lane, did not promote, and did not overwrite `storage/ai/alphazero_lite/current`.",
        "",
        "## 2. Why PR #37 Shifted Blame To Self-Play Targets",
        "",
        "- `self_play_only` reproduced the PR #36-style corrected guard regression without the selected replay artifact.",
        "- Follow-up inspection showed `opening_plies_1_8-017` often under-targeted corrected move `2`, while descendants `capture_available-002` and `capture_available-007` drifted toward move `1`.",
        "- That made a narrow corrective target patch the next lowest-cost diagnostic.",
        "",
        "## 3. Failure-Chain Extraction",
        "",
        "| row_id | self_play_count | corrected_reference_move | averaged_self_play_top_move | corrected_reference_mass | top_move_mass | teacher_source | diagnosis |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["failure_chain_rows"]:
        lines.append(
            f"| {row['row_id']} | {row['self_play_count']} | {row['corrected_reference_move']} | {row['averaged_self_play_top_move'] if row['averaged_self_play_top_move'] is not None else '-'} | {format_float(row['corrected_reference_mass'])} | {format_float(row['top_move_mass'])} | {row['teacher_source'] or '-'} | {row['diagnosis']} |"
        )
    lines.extend(
        [
            "",
            "## 4. Corrective Artifact Construction",
            "",
            "| row_id | reason | corrected_reference_move | target_reference_mass | legal_moves | value_source | train_only | status |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["corrective_rows"]:
        lines.append(
            f"| {row['source_runs'][0]['id']} | {row['reason']} | {row['corrected_reference_move']} | {format_float(row['corrected_policy_mass'])} | `{json.dumps(row['legal_moves'])}` | {row['value_source']} | {str(bool(row['train_only'])).lower()} | ok |"
        )
    validation = summary["artifact_validation"]
    lines.extend(
        [
            "",
            "## 5. Static Validation",
            "",
            f"- Status: `{validation['status']}`.",
            f"- All required rows present: `{str(bool(validation['all_required_rows_present'])).lower()}`.",
            f"- Duplicate conflicts: `{validation['duplicate_conflicts']}`.",
            f"- Stale reference conflicts: `{validation['stale_reference_conflicts']}`.",
            f"- Notes: `{', '.join(validation['notes'])}`.",
            "",
            "## 6. Diagnostic Patch Traces",
            "",
        ]
    )
    for trace in summary["traces"]:
        lines.append(
            f"- `{trace['trace_name']}` used data `{json.dumps(trace['data_files'])}` with replay weights `{json.dumps(trace['replay_weights'])}` and epochs `{json.dumps(trace['epochs'])}`."
        )
    lines.extend(
        [
            "",
            "## 7. Corrected Guard Kill-Gate Results",
            "",
            "| trace_name | epoch | row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | reference_visit_share_384 | reference_visit_share_1200 | selected_minus_reference_q_margin_384 | selected_minus_reference_q_margin_1200 | row_pass | gate_pass | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["guard_rows"]:
        lines.append(
            f"| {row['trace_name']} | {row['epoch']} | {row['row_id']} | {row['corrected_reference_move']} | {row['selected_move_384'] if row['selected_move_384'] is not None else '-'} | {row['selected_move_1200'] if row['selected_move_1200'] is not None else '-'} | {format_float(row['reference_visit_share_384'])} | {format_float(row['reference_visit_share_1200'])} | {format_float(row['selected_minus_reference_q_margin_384'])} | {format_float(row['selected_minus_reference_q_margin_1200'])} | {str(bool(row['row_pass'])).lower()} | {str(bool(row['gate_pass'])).lower()} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 8. Policy-Target Movement",
            "",
            "| trace_name | epoch | row_id | corrected_reference_move | policy_top_move | reference_policy_probability | reference_policy_rank | old_self_play_top_move | old_top_policy_probability | reference_minus_old_top_margin | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["policy_rows"]:
        lines.append(
            f"| {row['trace_name']} | {row['epoch']} | {row['row_id']} | {row['corrected_reference_move']} | {row['policy_top_move'] if row['policy_top_move'] is not None else '-'} | {format_float(row['reference_policy_probability'])} | {row['reference_policy_rank'] if row['reference_policy_rank'] is not None else '-'} | {row['old_self_play_top_move'] if row['old_self_play_top_move'] is not None else '-'} | {format_float(row['old_top_policy_probability'])} | {format_float(row['reference_minus_old_top_margin'])} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 9. Training Metrics",
            "",
            "| trace_name | epoch | policy_loss | value_loss | total_loss | self_play_cross_entropy | corrective_patch_cross_entropy | guard_cross_entropy | val_loss | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["metric_rows"]:
        lines.append(
            f"| {row['trace_name']} | {row['epoch']} | {format_float(row['policy_loss'])} | {format_float(row['value_loss'])} | {format_float(row['total_loss'])} | {format_float(row['self_play_cross_entropy'])} | {format_float(row['corrective_patch_cross_entropy'])} | {format_float(row['guard_cross_entropy'])} | {format_float(row['val_loss'])} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 10. Interpretation",
            "",
            f"- Classification: `{summary['decision']['classification']}`.",
            f"- Evidence: {summary['decision']['notes']}",
            f"- Trace D executed: `{str(bool(summary['trace_d_run'])).lower()}`.",
            "",
            "## 11. Exactly One Recommended Next Action",
            "",
            f"Recommendation: **{summary['decision']['next_action']}**",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    output_root = (
        args.output_root if args.output_root.is_absolute() else root / args.output_root
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = (
        args.summary_path
        if args.summary_path.is_absolute()
        else root / args.summary_path
    )
    report_path = (
        args.report_path if args.report_path.is_absolute() else root / args.report_path
    )
    self_play_path = args.self_play
    if not self_play_path.exists():
        raise FileNotFoundError(f"missing PR #36 self-play artifact: {self_play_path}")

    corrective_summary = build_artifact(
        self_play_path=args.self_play,
        reference_artifact_path=args.reference_artifact,
        fallback_reference_artifact_path=args.fallback_reference_artifact,
        input_encoding="kalah_v3",
        policy_target_mode="sharpened",
        value_target_mode="sharpened",
    )
    write_json(args.artifact_path.with_suffix(".summary.json"), corrective_summary)
    args.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with args.artifact_path.open("w", encoding="utf-8") as handle:
        for row in corrective_summary["corrective_rows"]:
            handle.write(json.dumps(row) + "\n")

    config = load_json(root / args.config)
    train_spec = build_train_spec(config)
    current_path = root / args.current_path
    current_metadata = load_json(current_path / "metadata.json")
    init_checkpoint = current_checkpoint_path(current_path, output_root)
    device = choose_device(args.device)
    reference_rows = merged_reference_rows(
        args.reference_artifact,
        args.fallback_reference_artifact,
    )
    old_self_play_top_moves = {
        row["row_id"]: row["averaged_self_play_top_move"]
        for row in corrective_summary["failure_chain_rows"]
    }

    traces = build_trace_specs(args.self_play, args.artifact_path)
    executed_traces: list[dict[str, Any]] = []
    guard_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for spec in traces:
        trace = run_trace(
            spec=spec,
            init_checkpoint=init_checkpoint,
            current_metadata=current_metadata,
            train_spec=train_spec,
            output_root=output_root,
            seed=int(args.seed),
            device=device,
            corrective_artifact_path=args.artifact_path,
            reference_artifact_path=args.reference_artifact,
            fallback_reference_artifact_path=args.fallback_reference_artifact,
            reference_rows=reference_rows,
            old_self_play_top_moves=old_self_play_top_moves,
        )
        executed_traces.append(
            {
                key: value
                for key, value in trace.items()
                if key not in {"guard_rows", "policy_rows", "metric_rows"}
            }
        )
        guard_rows.extend(trace["guard_rows"])
        policy_rows.extend(trace["policy_rows"])
        metric_rows.extend(trace["metric_rows"])

    trace_d_run = False
    if not epoch_gate_pass(
        guard_rows, "self_play_plus_corrective_patch_w1", 4
    ) and not epoch_gate_pass(guard_rows, "self_play_plus_corrective_patch_w2", 4):
        trace_d_run = True
        trace_d = TraceSpec(
            name="self_play_plus_corrective_patch_w4",
            data_files=(args.self_play, args.artifact_path),
            replay_weights=(1, 4),
            epochs=TARGET_EPOCHS,
        )
        trace = run_trace(
            spec=trace_d,
            init_checkpoint=init_checkpoint,
            current_metadata=current_metadata,
            train_spec=train_spec,
            output_root=output_root,
            seed=int(args.seed),
            device=device,
            corrective_artifact_path=args.artifact_path,
            reference_artifact_path=args.reference_artifact,
            fallback_reference_artifact_path=args.fallback_reference_artifact,
            reference_rows=reference_rows,
            old_self_play_top_moves=old_self_play_top_moves,
        )
        executed_traces.append(
            {
                key: value
                for key, value in trace.items()
                if key not in {"guard_rows", "policy_rows", "metric_rows"}
            }
        )
        guard_rows.extend(trace["guard_rows"])
        policy_rows.extend(trace["policy_rows"])
        metric_rows.extend(trace["metric_rows"])

    decision = classify_result(
        guard_rows=guard_rows,
        policy_rows=policy_rows,
        trace_names=[trace["trace_name"] for trace in executed_traces],
    )
    summary = {
        "schema": SCHEMA,
        "artifact_path": str(args.artifact_path),
        "failure_chain_rows": corrective_summary["failure_chain_rows"],
        "corrective_rows": corrective_summary["corrective_rows"],
        "artifact_validation": corrective_summary["validation"],
        "traces": executed_traces,
        "guard_rows": guard_rows,
        "policy_rows": policy_rows,
        "metric_rows": metric_rows,
        "decision": decision,
        "trace_d_run": trace_d_run,
        "constraints": {
            "arena_run": False,
            "mcts1200_lane_run": False,
            "promotion_run": False,
            "current_overwritten": False,
            "self_play_reused": True,
        },
    }
    write_json(summary_path, summary)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "report_path": str(report_path),
                "classification": decision["classification"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
