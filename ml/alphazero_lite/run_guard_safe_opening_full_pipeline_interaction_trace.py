#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.corrected_guard_kill_gate import (
    DEFAULT_FALLBACK_REFERENCE_ARTIFACT,
    DEFAULT_REFERENCE_ARTIFACT,
    GUARD_ROW_IDS,
    validate_reference_rows,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.pipeline import (
    build_step_command,
    render_command,
    resolve_step_command,
)
from ml.alphazero_lite.run_guard_safe_opening_low_epoch_drift_trace import (
    artifact_row_id,
    choose_device,
    compute_policy_cross_entropy,
    compute_value_loss_vector,
    current_checkpoint_path,
    display_path,
    export_checkpoint_artifact,
    metric_rank,
    policy_entropy,
    read_jsonl,
    resolve_path,
    top_policy_move,
    train_one_epoch,
    validate_selected_artifact,
    write_json,
)
from ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment import (
    load_json,
    row_map_from_reference,
)
from ml.alphazero_lite.self_play import build_eval_search_options
from ml.alphazero_lite.train import (
    PolicyValueNet,
    checkpoint_from_model,
    input_size_for_encoding,
    load_checkpoint_into_model,
    load_jsonl_replay,
    parse_hidden_sizes,
    resolve_hidden_sizes,
    set_seed,
    split_replay_positions_by_source_row,
)


DEFAULT_CONFIG_PATH = Path(
    "ml/alphazero_lite/configs/exp_v3_guard_safe_opening_selected_w1.json"
)
DEFAULT_CURRENT_PATH = Path("storage/ai/alphazero_lite/current")
DEFAULT_SELECTED_ARTIFACT = Path(
    "/tmp/azlite_guard_safe_opening_replay/"
    "family_leave_one_out_without_opening_extra_turn_overbias.jsonl"
)
DEFAULT_GUARD_CONTROLS_ARTIFACT = Path(
    "/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl"
)
DEFAULT_ITER_DIR = Path(
    "/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_versions/"
    "exp-v3-guard-safe-opening-selected-w1-iter1"
)
DEFAULT_OUTPUT_ROOT = Path("/tmp/azlite_guard_safe_opening_full_pipeline_trace")
DEFAULT_SUMMARY_PATH = (
    DEFAULT_OUTPUT_ROOT / "full_pipeline_interaction_trace_summary.json"
)
DEFAULT_REPORT_PATH = Path(
    "docs/alphazero-lite-guard-safe-opening-full-pipeline-interaction-trace-results.md"
)
SCHEMA = "azlite_guard_safe_opening_full_pipeline_interaction_trace_v1"
TARGET_EPOCHS = (1, 2, 4)
DEFAULT_LR = 1e-3
LOW_LR = 1e-4
DEFAULT_SEED = 42
DEFAULT_C_PUCT = 1.25
REFERENCE_FAILURE_ROWS = ("capture_available-002", "capture_available-003")


@dataclass(frozen=True)
class TrainSpec:
    model_type: str
    input_encoding: str
    hidden_sizes: tuple[int, ...]
    value_loss: str
    huber_delta: float
    value_loss_weight: float
    val_split: float
    grad_clip: float
    batch_size: int
    policy_target_mode: str
    value_target_mode: str
    lr: float


@dataclass(frozen=True)
class TraceSpec:
    name: str
    data_files: tuple[Path, ...]
    replay_weights: tuple[int, ...]
    lr: float | None
    epochs: tuple[int, ...]
    purpose: str
    status: str = "planned"
    notes: str = ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--reference-artifact", type=Path, default=DEFAULT_REFERENCE_ARTIFACT
    )
    parser.add_argument(
        "--fallback-reference-artifact",
        type=Path,
        default=DEFAULT_FALLBACK_REFERENCE_ARTIFACT,
    )
    parser.add_argument("--current-path", type=Path, default=DEFAULT_CURRENT_PATH)
    parser.add_argument(
        "--selected-artifact", type=Path, default=DEFAULT_SELECTED_ARTIFACT
    )
    parser.add_argument(
        "--guard-controls-artifact",
        type=Path,
        default=DEFAULT_GUARD_CONTROLS_ARTIFACT,
    )
    parser.add_argument("--iter-dir", type=Path, default=DEFAULT_ITER_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def path_size_or_rows(path: Path) -> str:
    if not path.exists():
        return "-"
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return str(sum(1 for _ in handle))
    if path.is_dir():
        return str(sum(1 for _ in path.iterdir()))
    return str(path.stat().st_size)


def train_step_from_config(config: dict[str, Any]) -> list[str]:
    for step in list(config.get("steps") or []):
        if isinstance(step, dict) and step.get("name") == "train":
            return [str(token) for token in list(step.get("command") or [])]
    raise ValueError("config is missing train step")


def self_play_step_from_config(config: dict[str, Any]) -> dict[str, Any]:
    for step in list(config.get("steps") or []):
        if isinstance(step, dict) and step.get("name") == "self_play":
            return dict(step)
    raise ValueError("config is missing self_play step")


def command_flag_value(command: list[str], flag: str, default: str) -> str:
    if flag not in command:
        return default
    index = command.index(flag)
    if index + 1 >= len(command):
        return default
    return str(command[index + 1])


def build_train_spec(config: dict[str, Any]) -> TrainSpec:
    command = train_step_from_config(config)
    hidden_sizes = resolve_hidden_sizes(
        command_flag_value(command, "--model-type", "residual_v3"),
        parse_hidden_sizes(command_flag_value(command, "--hidden-sizes", "96,3")),
    )
    return TrainSpec(
        model_type=command_flag_value(command, "--model-type", "residual_v3"),
        input_encoding=command_flag_value(command, "--input-encoding", "kalah_v3"),
        hidden_sizes=hidden_sizes,
        value_loss=command_flag_value(command, "--value-loss", "huber"),
        huber_delta=float(command_flag_value(command, "--huber-delta", "1.0")),
        value_loss_weight=float(
            command_flag_value(command, "--value-loss-weight", "0.3")
        ),
        val_split=float(command_flag_value(command, "--val-split", "0.1")),
        grad_clip=float(command_flag_value(command, "--grad-clip", "1.0")),
        batch_size=int(command_flag_value(command, "--batch-size", "512")),
        policy_target_mode=command_flag_value(
            command, "--policy-target-mode", "sharpened"
        ),
        value_target_mode=command_flag_value(
            command, "--value-target-mode", "sharpened"
        ),
        lr=float(command_flag_value(command, "--lr", str(DEFAULT_LR))),
    )


def build_trace_specs(
    *,
    self_play_path: Path,
    selected_artifact: Path,
    guard_controls_artifact: Path,
    train_spec: TrainSpec,
) -> list[TraceSpec]:
    specs = [
        TraceSpec(
            name="self_play_only",
            data_files=(self_play_path,),
            replay_weights=(1,),
            lr=train_spec.lr,
            epochs=TARGET_EPOCHS,
            purpose="test whether generated self-play/background data alone causes guard regression.",
        ),
        TraceSpec(
            name="selected_artifact_only_pipeline_hparams",
            data_files=(selected_artifact,),
            replay_weights=(1,),
            lr=train_spec.lr,
            epochs=TARGET_EPOCHS,
            purpose="compare PR #35 artifact-only conclusion under the exact PR #36 training settings.",
        ),
        TraceSpec(
            name="self_play_plus_selected_artifact_w1",
            data_files=(self_play_path, selected_artifact),
            replay_weights=(1, 1),
            lr=train_spec.lr,
            epochs=TARGET_EPOCHS,
            purpose="reproduce the PR #36 full-lane regression and identify when it appears.",
        ),
    ]
    if guard_controls_artifact.exists():
        specs.append(
            TraceSpec(
                name="self_play_plus_guard_controls",
                data_files=(self_play_path, selected_artifact, guard_controls_artifact),
                replay_weights=(1, 1, 2),
                lr=train_spec.lr,
                epochs=TARGET_EPOCHS,
                purpose="test whether stronger guard anchoring prevents the 384-sim 002/003 drift.",
            )
        )
    else:
        specs.append(
            TraceSpec(
                name="self_play_plus_guard_controls",
                data_files=(self_play_path, selected_artifact),
                replay_weights=(1, 1),
                lr=train_spec.lr,
                epochs=TARGET_EPOCHS,
                purpose="test whether stronger guard anchoring prevents the 384-sim 002/003 drift.",
                status="skipped",
                notes="guard_safe_controls_only artifact missing",
            )
        )
    if train_spec.lr != LOW_LR:
        specs.append(
            TraceSpec(
                name="self_play_plus_selected_artifact_low_lr",
                data_files=(self_play_path, selected_artifact),
                replay_weights=(1, 1),
                lr=LOW_LR,
                epochs=TARGET_EPOCHS,
                purpose="test update-size sensitivity in the full-data setting.",
            )
        )
    else:
        specs.append(
            TraceSpec(
                name="self_play_plus_selected_artifact_low_lr",
                data_files=(self_play_path, selected_artifact),
                replay_weights=(1, 1),
                lr=None,
                epochs=TARGET_EPOCHS,
                purpose="test update-size sensitivity in the full-data setting.",
                status="skipped",
                notes="train.py already uses low LR under this config",
            )
        )
    return specs


def source_kind_for_path(
    path: Path,
    *,
    self_play_path: Path,
    selected_artifact: Path,
    guard_controls_artifact: Path,
) -> str:
    if path == self_play_path:
        return "self_play"
    if path == selected_artifact:
        return "selected_artifact"
    if path == guard_controls_artifact:
        return "guard_controls"
    return "other"


def build_row_infos(
    paths: tuple[Path, ...],
    *,
    self_play_path: Path,
    selected_artifact: Path,
    guard_controls_artifact: Path,
) -> list[dict[str, Any]]:
    row_infos: list[dict[str, Any]] = []
    for path in paths:
        source_kind = source_kind_for_path(
            path,
            self_play_path=self_play_path,
            selected_artifact=selected_artifact,
            guard_controls_artifact=guard_controls_artifact,
        )
        for row in read_jsonl(path):
            row_infos.append(
                {
                    "row_id": artifact_row_id(row),
                    "source_kind": source_kind,
                    "source_path": str(path),
                }
            )
    return row_infos


def regenerate_self_play_if_needed(
    *,
    config: dict[str, Any],
    config_path: Path,
    iter_dir: Path,
    current_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    self_play_path = iter_dir / "self_play.jsonl"
    if self_play_path.exists():
        return {
            "path": self_play_path,
            "regenerated": False,
            "status": "reused_existing",
            "notes": "reused PR #36 self_play.jsonl",
        }

    root = repo_root()
    step = self_play_step_from_config(config)
    command = build_step_command(step)
    if not isinstance(command, list) or not command:
        raise ValueError("self_play step command must be a non-empty list")

    init_checkpoint = current_checkpoint_path(current_path, output_root)
    rendered = render_command(
        command,
        iteration=int(config.get("start_iteration", 1)),
        iter_dir=iter_dir,
        run_id=str(config.get("run_id", "exp-v3-guard-safe-opening-selected-w1")),
        versions_dir=Path(str(config.get("versions_dir", output_root))),
        current_path=str(current_path),
        parent_model_dir=current_path,
        parent_checkpoint=init_checkpoint,
        replay_data="",
        replay_weights="",
        hard_state_validation_path=str(config.get("hard_state_validation_path", "")),
    )
    rendered = resolve_step_command(rendered, repo_root=root)

    started = time.time()
    result = subprocess.run(
        rendered, cwd=root, capture_output=True, text=True, check=False
    )
    duration_s = round(time.time() - started, 4)
    log_path = output_root / "self_play_regeneration.log"
    log_path.write_text(
        f"command={json.dumps(rendered)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode != 0 or not self_play_path.exists():
        raise RuntimeError(
            "failed to regenerate missing PR #36 self_play step; see "
            f"{log_path} for details"
        )
    return {
        "path": self_play_path,
        "regenerated": True,
        "status": "regenerated_from_config",
        "notes": f"regenerated missing self_play step from {display_path(root, config_path)}",
        "duration_s": duration_s,
        "log_path": str(log_path),
    }


def evaluate_guard_candidate(
    *,
    artifact_path: Path,
    reference_rows: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    evaluator = ArtifactEvaluator(artifact_path)
    search_options = dict(build_eval_search_options())
    rows: list[dict[str, Any]] = []
    gate_pass = True

    for row_index, row_id in enumerate(GUARD_ROW_IDS):
        row = reference_rows[row_id]
        game_state = dict(row["state"])
        legal_moves = [int(move) for move in list(row["legal_moves"])]
        reference_move = int(row["corrected_reference_move"])
        policy, _root_value = evaluator.evaluate(KalahGame.from_state(game_state))
        policy_distribution = {move: float(policy[move]) for move in legal_moves}

        def probe_view(simulations: int) -> dict[str, Any]:
            probe = evaluate_artifact_position(
                artifact_path=artifact_path,
                evaluator=evaluator,
                state=game_state,
                simulations=simulations,
                seed=seed + row_index + simulations,
                c_puct=DEFAULT_C_PUCT,
                search_options=search_options,
                ablation_mode="full",
            )
            child_stats = {
                int(child["move"]): child
                for child in list(probe.get("child_stats") or [])
            }
            visits = list(probe.get("visits") or [])
            total_visits = sum(
                float(visits[move]) for move in legal_moves if move < len(visits)
            )
            selected_move = probe.get("selected_move")
            reference_visit_share = None
            if total_visits > 0.0 and reference_move < len(visits):
                reference_visit_share = round(
                    float(visits[reference_move]) / total_visits, 4
                )
            selected_q = None
            if selected_move is not None and int(selected_move) in child_stats:
                selected_q = float(child_stats[int(selected_move)].get("q_value", 0.0))
            reference_q = None
            if reference_move in child_stats:
                reference_q = float(child_stats[reference_move].get("q_value", 0.0))
            return {
                "selected_move": None if selected_move is None else int(selected_move),
                "selected_is_reference": selected_move == reference_move,
                "reference_visit_share": reference_visit_share,
                "selected_minus_reference_q_margin": None
                if selected_q is None or reference_q is None
                else round(selected_q - reference_q, 4),
            }

        probe_384 = probe_view(384)
        probe_1200 = probe_view(1200)
        row_pass = bool(
            probe_384["selected_is_reference"] and probe_1200["selected_is_reference"]
        )
        if not row_pass:
            gate_pass = False
        policy_top = top_policy_move(list(policy), legal_moves)
        rows.append(
            {
                "row_id": row_id,
                "corrected_reference_move": reference_move,
                "selected_move_384": probe_384["selected_move"],
                "selected_move_1200": probe_1200["selected_move"],
                "reference_visit_share_384": probe_384["reference_visit_share"],
                "reference_visit_share_1200": probe_1200["reference_visit_share"],
                "selected_minus_reference_q_margin_384": probe_384[
                    "selected_minus_reference_q_margin"
                ],
                "selected_minus_reference_q_margin_1200": probe_1200[
                    "selected_minus_reference_q_margin"
                ],
                "policy_top_move": policy_top,
                "reference_policy_probability": round(
                    float(policy_distribution.get(reference_move, 0.0)), 4
                ),
                "reference_policy_rank": metric_rank(
                    policy_distribution, reference_move
                ),
                "policy_entropy": policy_entropy(list(policy), legal_moves),
                "row_pass": row_pass,
                "notes": "pass_reference_selected"
                if row_pass
                else "selected_move_not_reference",
            }
        )
    return {
        "rows": rows,
        "gate_pass": gate_pass,
        "decision": "pass" if gate_pass else "reject_guard_regression",
    }


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
    train_spec: TrainSpec,
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
            val_index_tensor = torch.from_numpy(val_replay_indexes).to(device)
            val_logits, val_value_pred = model(x_tensor[val_index_tensor])
            val_policy_ce = compute_policy_cross_entropy(
                val_logits, p_tensor[val_index_tensor]
            )
            val_value_losses = compute_value_loss_vector(
                val_value_pred,
                v_tensor[val_index_tensor],
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
    guard_policy_losses = [
        float(policy_ce[index])
        for index, info in enumerate(row_infos)
        if info.get("row_id") in GUARD_ROW_IDS
    ]
    selected_artifact_policy_losses = [
        float(policy_ce[index])
        for index, info in enumerate(row_infos)
        if info.get("source_kind") == "selected_artifact"
    ]
    self_play_policy_losses = [
        float(policy_ce[index])
        for index, info in enumerate(row_infos)
        if info.get("source_kind") == "self_play"
    ]
    return {
        "policy_loss": round(float(np.mean(policy_ce)), 6),
        "value_loss": round(float(np.mean(value_losses)), 6),
        "total_loss": round(float(np.mean(total_losses)), 6),
        "val_loss": val_loss,
        "guard_cross_entropy": mean_or_none(guard_policy_losses),
        "selected_artifact_cross_entropy": mean_or_none(
            selected_artifact_policy_losses
        ),
        "self_play_cross_entropy": mean_or_none(self_play_policy_losses),
    }


def run_trace(
    *,
    spec: TraceSpec,
    init_checkpoint: Path,
    current_metadata: dict[str, Any],
    train_spec: TrainSpec,
    reference_rows: dict[str, dict[str, Any]],
    output_root: Path,
    seed: int,
    device: torch.device,
    self_play_path: Path,
    selected_artifact: Path,
    guard_controls_artifact: Path,
) -> dict[str, Any]:
    if spec.status == "skipped":
        return {
            "trace_name": spec.name,
            "data_files": [str(path) for path in spec.data_files],
            "replay_weights": list(spec.replay_weights),
            "lr": spec.lr,
            "epochs": list(spec.epochs),
            "init_checkpoint": str(init_checkpoint),
            "purpose": spec.purpose,
            "status": spec.status,
            "notes": spec.notes,
            "guard_drift_rows": [],
            "training_metric_rows": [],
        }

    set_seed(seed)
    compact_x, compact_p, compact_v, replay_indexes = load_jsonl_replay(
        list(spec.data_files),
        list(spec.replay_weights),
        policy_target_mode=train_spec.policy_target_mode,
        value_target_mode=train_spec.value_target_mode,
    )
    compact_v = compact_v.astype(np.float32)
    row_infos = build_row_infos(
        spec.data_files,
        self_play_path=self_play_path,
        selected_artifact=selected_artifact,
        guard_controls_artifact=guard_controls_artifact,
    )

    model = PolicyValueNet(
        hidden_sizes=train_spec.hidden_sizes,
        model_type=train_spec.model_type,
        input_size=input_size_for_encoding(train_spec.input_encoding),
    )
    load_checkpoint_into_model(model, init_checkpoint)
    model = model.to(device)
    lr = spec.lr
    if lr is None:
        raise ValueError(f"trace {spec.name} is missing lr")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, max(spec.epochs))
    )

    train_positions, val_positions = split_replay_positions_by_source_row(
        replay_indexes, val_split=train_spec.val_split
    )
    weighted_train_indexes = replay_indexes[train_positions]
    weighted_val_indexes = replay_indexes[val_positions]

    trace_root = output_root / "traces" / spec.name
    checkpoints_root = trace_root / "checkpoints"
    exports_root = output_root / "exports" / spec.name
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    exports_root.mkdir(parents=True, exist_ok=True)

    guard_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    target_epochs = set(spec.epochs)

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
        if epoch not in target_epochs:
            continue

        checkpoint_path = checkpoints_root / f"epoch_{epoch}.npz"
        np.savez(checkpoint_path, **checkpoint_from_model(model))
        export_dir = export_checkpoint_artifact(
            checkpoint_path=checkpoint_path,
            export_dir=exports_root / f"epoch_{epoch}",
            current_metadata=current_metadata,
            version=f"{spec.name}-epoch-{epoch}",
        )
        guard_eval = evaluate_guard_candidate(
            artifact_path=export_dir,
            reference_rows=reference_rows,
            seed=seed + epoch,
        )
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
        for row in guard_eval["rows"]:
            guard_rows.append(
                {
                    "trace_name": spec.name,
                    "epoch": epoch,
                    **row,
                    "gate_pass": bool(guard_eval["gate_pass"]),
                }
            )
        metric_rows.append(
            {
                "trace_name": spec.name,
                "epoch": epoch,
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "total_loss": metrics["total_loss"],
                "val_loss": metrics["val_loss"],
                "guard_cross_entropy": metrics["guard_cross_entropy"],
                "selected_artifact_cross_entropy": metrics[
                    "selected_artifact_cross_entropy"
                ],
                "self_play_cross_entropy": metrics["self_play_cross_entropy"],
                "notes": "full_dataset_snapshot",
            }
        )

    return {
        "trace_name": spec.name,
        "data_files": [str(path) for path in spec.data_files],
        "replay_weights": list(spec.replay_weights),
        "lr": spec.lr,
        "epochs": list(spec.epochs),
        "init_checkpoint": str(init_checkpoint),
        "purpose": spec.purpose,
        "status": "completed",
        "notes": spec.notes or "ok",
        "guard_drift_rows": guard_rows,
        "training_metric_rows": metric_rows,
    }


def row_failed_384(row: dict[str, Any]) -> bool:
    return (
        row.get("row_id") in REFERENCE_FAILURE_ROWS
        and not bool(row.get("row_pass"))
        and row.get("selected_move_384") != row.get("corrected_reference_move")
    )


def trace_epoch_rows(
    guard_rows: list[dict[str, Any]], trace_name: str, epoch: int
) -> list[dict[str, Any]]:
    return [
        row
        for row in guard_rows
        if row["trace_name"] == trace_name and int(row["epoch"]) == epoch
    ]


def trace_reproduces_regression(
    guard_rows: list[dict[str, Any]], trace_name: str
) -> bool:
    epochs = sorted(
        {
            int(row["epoch"])
            for row in guard_rows
            if row["trace_name"] == trace_name
            and row["row_id"] in REFERENCE_FAILURE_ROWS
        }
    )
    for epoch in epochs:
        rows = trace_epoch_rows(guard_rows, trace_name, epoch)
        if not rows:
            continue
        failures = [row for row in rows if row_failed_384(row)]
        if len(failures) == len(REFERENCE_FAILURE_ROWS):
            return True
    return False


def trace_first_regression_epoch(
    guard_rows: list[dict[str, Any]], trace_name: str
) -> int | None:
    epochs = sorted(
        {
            int(row["epoch"])
            for row in guard_rows
            if row["trace_name"] == trace_name
            and row["row_id"] in REFERENCE_FAILURE_ROWS
        }
    )
    for epoch in epochs:
        rows = trace_epoch_rows(guard_rows, trace_name, epoch)
        failures = [row for row in rows if row_failed_384(row)]
        if len(failures) == len(REFERENCE_FAILURE_ROWS):
            return epoch
    return None


def classify_cause(summary: dict[str, Any]) -> dict[str, Any]:
    trace_status = {
        trace["trace_name"]: trace
        for trace in summary["traces"]
        if trace["status"] == "completed"
    }
    guard_rows = list(summary["guard_drift_rows"])
    reproduced = {
        name: trace_reproduces_regression(guard_rows, name) for name in trace_status
    }
    first_epochs = {
        name: trace_first_regression_epoch(guard_rows, name) for name in trace_status
    }

    completed_names = list(trace_status)
    if (
        completed_names
        and all(
            first_epochs.get(name) == 1
            for name in completed_names
            if reproduced.get(name)
        )
        and completed_names
        and all(reproduced.get(name, False) for name in completed_names)
    ):
        return {
            "classification": "inherited_or_gate_sensitivity",
            "supporting_evidence": "Every completed trace reproduced the 384-sim 002/003 failure at epoch 1.",
            "rejected_alternatives": "Did not isolate self-play-only, artifact-only, guard-anchor, LR, or mix-specific behavior.",
            "next_action": "audit 384-sim gate stability across seeds before using it as a hard kill gate.",
        }

    if reproduced.get("self_play_only", False):
        return {
            "classification": "self_play_background_causes_guard_regression",
            "supporting_evidence": "self_play_only reproduced the PR #36-style 384-sim 002/003 regression without any selected replay artifact.",
            "rejected_alternatives": "Selected artifact was not required to trigger the failure.",
            "next_action": "inspect PR #36 self-play rows near opening_plies_1_8 and mine conflicts from self-play targets.",
        }

    if reproduced.get("selected_artifact_only_pipeline_hparams", False):
        return {
            "classification": "selected_artifact_causes_guard_regression_under_pipeline_hparams",
            "supporting_evidence": "selected_artifact_only_pipeline_hparams reproduced the 384-sim 002/003 regression under PR #36 training settings.",
            "rejected_alternatives": "Self-play/background data was not required to trigger the failure.",
            "next_action": "abandon selected opening artifact or reduce to guard controls only.",
        }

    mix_regresses = reproduced.get("self_play_plus_selected_artifact_w1", False)
    guard_controls_regresses = reproduced.get("self_play_plus_guard_controls", False)
    low_lr_regresses = reproduced.get("self_play_plus_selected_artifact_low_lr", False)

    if (
        mix_regresses
        and not guard_controls_regresses
        and "self_play_plus_guard_controls" in trace_status
    ):
        return {
            "classification": "guard_anchor_needed",
            "supporting_evidence": "self_play_plus_selected_artifact_w1 reproduced the regression, while self_play_plus_guard_controls did not.",
            "rejected_alternatives": "self_play_only and selected_artifact_only did not independently reproduce the failure.",
            "next_action": "run one tiny controlled lane with selected artifact weight 1 + guard controls weight 2 and the same kill gate.",
        }

    if (
        mix_regresses
        and not low_lr_regresses
        and "self_play_plus_selected_artifact_low_lr" in trace_status
    ):
        return {
            "classification": "update_size_sensitive_full_pipeline",
            "supporting_evidence": "self_play_plus_selected_artifact_w1 reproduced the regression, while the low-LR variant did not.",
            "rejected_alternatives": "self_play_only and selected_artifact_only did not independently reproduce the failure.",
            "next_action": "run one tiny controlled lane with lower LR and guard kill gate.",
        }

    if (
        mix_regresses
        and not reproduced.get("self_play_only", False)
        and not reproduced.get("selected_artifact_only_pipeline_hparams", False)
    ):
        return {
            "classification": "data_mix_interaction",
            "supporting_evidence": "Only the self-play plus selected artifact mix reproduced the 384-sim 002/003 regression.",
            "rejected_alternatives": "Neither self-play-only nor selected-artifact-only reproduced the failure on their own.",
            "next_action": "test replay sampling/weighting or guard-anchor weighting, not more opening artifact filtering.",
        }

    if not any(reproduced.values()):
        return {
            "classification": "pipeline_non_determinism_or_checkpoint_selection_issue",
            "supporting_evidence": "None of the requested diagnostic traces reproduced the PR #36 384-sim 002/003 regression.",
            "rejected_alternatives": "No single component or traced interaction reproduced the merged-lane failure.",
            "next_action": "rerun PR #36 config with deterministic logging and compare train data/order/checkpoint hashes.",
        }

    return {
        "classification": "data_mix_interaction",
        "supporting_evidence": "The failure remained tied to full-pipeline interaction rather than a single isolated input source.",
        "rejected_alternatives": "No cleaner single-component explanation fit all completed trace outcomes.",
        "next_action": "test replay sampling/weighting or guard-anchor weighting, not more opening artifact filtering.",
    }


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def format_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return str(bool(value)).lower()


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-lite Guard-Safe Opening Full-Pipeline Interaction Trace Results",
        "",
        "## 1. Context",
        "",
        "- PR #36 trained the selected replay lane but every exported candidate failed the corrected guard kill gate on `capture_available-002` and `capture_available-003` at `384` simulations.",
        "- PR #35 already showed the selected artifact was statically guard-safe and did not regress under low-epoch artifact-only drift tracing.",
        "- This run stayed diagnostic-only: no arena, no MCTS1200 eval lane, no promotion, and no overwrite of `storage/ai/alphazero_lite/current`.",
        "",
        "## 2. Why PR #36 Stopped The Selected Lane",
        "",
        "- PR #36 stopped because rows `capture_available-002` and `capture_available-003` selected move `1` instead of corrected reference move `2` at `384` simulations, even though both rows recovered to move `2` at `1200` simulations.",
        "- That made the failure look like a full-pipeline interaction or search-sensitivity issue rather than a simple static replay-artifact conflict.",
        "",
        "## 3. Data/Input Verification",
        "",
        "| input | path | exists | rows_or_size | status | notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["input_verification"]:
        lines.append(
            f"| {row['input']} | `{row['path']}` | {format_bool(row['exists'])} | {row['rows_or_size']} | `{row['status']}` | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 4. Current Baseline Guard Result",
            "",
            "| trace_name | epoch | row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | reference_visit_share_384 | reference_visit_share_1200 | selected_minus_reference_q_margin_384 | selected_minus_reference_q_margin_1200 | row_pass | gate_pass | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["baseline_guard_result"]["rows"]:
        lines.append(
            f"| current_baseline | 0 | {row['row_id']} | {row['corrected_reference_move']} | {row['selected_move_384'] if row['selected_move_384'] is not None else '-'} | {row['selected_move_1200'] if row['selected_move_1200'] is not None else '-'} | {format_float(row['reference_visit_share_384'])} | {format_float(row['reference_visit_share_1200'])} | {format_float(row['selected_minus_reference_q_margin_384'])} | {format_float(row['selected_minus_reference_q_margin_1200'])} | {format_bool(row['row_pass'])} | {format_bool(summary['baseline_guard_result']['gate_pass'])} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 5. Trace Definitions",
            "",
            "| trace_name | data_files | replay_weights | lr | epochs | init_checkpoint | purpose | status |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["trace_definitions"]:
        lines.append(
            f"| {row['trace_name']} | `{json.dumps(row['data_files'])}` | `{json.dumps(row['replay_weights'])}` | {row['lr'] if row['lr'] is not None else '-'} | `{json.dumps(row['epochs'])}` | `{row['init_checkpoint']}` | {row['purpose']} | `{row['status']}` |"
        )
    lines.extend(
        [
            "",
            "## 6. Guard Drift Results",
            "",
            "| trace_name | epoch | row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | reference_visit_share_384 | reference_visit_share_1200 | selected_minus_reference_q_margin_384 | selected_minus_reference_q_margin_1200 | row_pass | gate_pass | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["guard_drift_rows"]:
        lines.append(
            f"| {row['trace_name']} | {row['epoch']} | {row['row_id']} | {row['corrected_reference_move']} | {row['selected_move_384'] if row['selected_move_384'] is not None else '-'} | {row['selected_move_1200'] if row['selected_move_1200'] is not None else '-'} | {format_float(row['reference_visit_share_384'])} | {format_float(row['reference_visit_share_1200'])} | {format_float(row['selected_minus_reference_q_margin_384'])} | {format_float(row['selected_minus_reference_q_margin_1200'])} | {format_bool(row['row_pass'])} | {format_bool(row['gate_pass'])} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 7. Training Metric Trace",
            "",
            "| trace_name | epoch | policy_loss | value_loss | total_loss | val_loss | guard_cross_entropy | selected_artifact_cross_entropy | self_play_cross_entropy | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["training_metric_rows"]:
        lines.append(
            f"| {row['trace_name']} | {row['epoch']} | {format_float(row['policy_loss'])} | {format_float(row['value_loss'])} | {format_float(row['total_loss'])} | {format_float(row['val_loss'])} | {format_float(row['guard_cross_entropy'])} | {format_float(row['selected_artifact_cross_entropy'])} | {format_float(row['self_play_cross_entropy'])} | {row['notes']} |"
        )
    decision = summary["decision"]
    lines.extend(
        [
            "",
            "## 8. Cause Classification",
            "",
            "| classification | supporting_evidence | rejected_alternatives | next_action |",
            "| --- | --- | --- | --- |",
            f"| `{decision['classification']}` | {decision['supporting_evidence']} | {decision['rejected_alternatives']} | {decision['next_action']} |",
            "",
            "## 9. Exactly One Recommended Next Action",
            "",
            f"Recommendation: **{decision['next_action']}**",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    config_path = resolve_path(root, args.config)
    reference_artifact = resolve_path(root, args.reference_artifact)
    fallback_reference_artifact = resolve_path(root, args.fallback_reference_artifact)
    current_path = resolve_path(root, args.current_path)
    selected_artifact = resolve_path(root, args.selected_artifact)
    guard_controls_artifact = resolve_path(root, args.guard_controls_artifact)
    iter_dir = resolve_path(root, args.iter_dir)
    output_root = resolve_path(root, args.output_root)
    summary_path = resolve_path(root, args.summary_path)
    report_path = resolve_path(root, args.report_path)
    output_root.mkdir(parents=True, exist_ok=True)

    config = load_json(config_path)
    self_play_status = regenerate_self_play_if_needed(
        config=config,
        config_path=config_path,
        iter_dir=iter_dir,
        current_path=current_path,
        output_root=output_root,
    )
    self_play_path = Path(self_play_status["path"])

    reference_validation, normalized_reference_rows = validate_reference_rows(
        reference_artifact,
        fallback_reference_artifact=fallback_reference_artifact,
    )
    if reference_validation["status"] != "ok":
        raise RuntimeError(
            "corrected guard references are invalid for this trace: "
            f"{reference_validation['notes']}"
        )

    merged_reference_rows = {
        **row_map_from_reference(load_json(fallback_reference_artifact)),
        **row_map_from_reference(load_json(reference_artifact)),
    }
    selected_validation, _selected_rows, _selected_rows_by_id = (
        validate_selected_artifact(
            artifact_path=selected_artifact,
            reference_rows=merged_reference_rows,
        )
    )

    current_metadata = load_json(current_path / "metadata.json")
    init_checkpoint = current_checkpoint_path(current_path, output_root)
    train_spec = build_train_spec(config)
    trace_specs = build_trace_specs(
        self_play_path=self_play_path,
        selected_artifact=selected_artifact,
        guard_controls_artifact=guard_controls_artifact,
        train_spec=train_spec,
    )
    device = choose_device(args.device)

    input_verification = [
        {
            "input": "pr36_self_play",
            "path": display_path(root, self_play_path),
            "exists": self_play_path.exists(),
            "rows_or_size": path_size_or_rows(self_play_path),
            "status": self_play_status["status"],
            "notes": self_play_status["notes"],
        },
        {
            "input": "selected_artifact",
            "path": display_path(root, selected_artifact),
            "exists": selected_artifact.exists(),
            "rows_or_size": path_size_or_rows(selected_artifact),
            "status": selected_validation["status"],
            "notes": selected_validation["notes"],
        },
        {
            "input": "guard_controls_artifact",
            "path": display_path(root, guard_controls_artifact),
            "exists": guard_controls_artifact.exists(),
            "rows_or_size": path_size_or_rows(guard_controls_artifact),
            "status": "ok" if guard_controls_artifact.exists() else "optional_missing",
            "notes": "available for Trace D"
            if guard_controls_artifact.exists()
            else "Trace D skipped if missing",
        },
        {
            "input": "corrected_references",
            "path": display_path(root, reference_artifact),
            "exists": reference_artifact.exists(),
            "rows_or_size": path_size_or_rows(reference_artifact),
            "status": reference_validation["status"],
            "notes": reference_validation["notes"],
        },
        {
            "input": "current_artifact",
            "path": display_path(root, current_path),
            "exists": current_path.exists(),
            "rows_or_size": path_size_or_rows(current_path),
            "status": "ok",
            "notes": "initializer artifact present; materialized checkpoint reused under /tmp",
        },
        {
            "input": "current_init_checkpoint",
            "path": display_path(root, init_checkpoint),
            "exists": init_checkpoint.exists(),
            "rows_or_size": path_size_or_rows(init_checkpoint),
            "status": "ok",
            "notes": "materialized from current weights.json"
            if init_checkpoint.name == "current_init_checkpoint.npz"
            else "existing checkpoint/model reused",
        },
    ]

    baseline_guard_result = evaluate_guard_candidate(
        artifact_path=current_path,
        reference_rows=normalized_reference_rows,
        seed=int(args.seed),
    )

    traces: list[dict[str, Any]] = []
    guard_drift_rows: list[dict[str, Any]] = []
    training_metric_rows: list[dict[str, Any]] = []
    for spec in trace_specs:
        trace = run_trace(
            spec=spec,
            init_checkpoint=init_checkpoint,
            current_metadata=current_metadata,
            train_spec=train_spec,
            reference_rows=normalized_reference_rows,
            output_root=output_root,
            seed=int(args.seed),
            device=device,
            self_play_path=self_play_path,
            selected_artifact=selected_artifact,
            guard_controls_artifact=guard_controls_artifact,
        )
        traces.append(
            {
                key: value
                for key, value in trace.items()
                if key not in {"guard_drift_rows", "training_metric_rows"}
            }
        )
        guard_drift_rows.extend(trace["guard_drift_rows"])
        training_metric_rows.extend(trace["training_metric_rows"])

    decision = classify_cause(
        {
            "traces": traces,
            "guard_drift_rows": guard_drift_rows,
        }
    )
    trace_definitions = [
        {
            "trace_name": trace["trace_name"],
            "data_files": trace["data_files"],
            "replay_weights": trace["replay_weights"],
            "lr": trace["lr"],
            "epochs": trace["epochs"],
            "init_checkpoint": trace["init_checkpoint"],
            "purpose": trace["purpose"],
            "status": trace["status"],
        }
        for trace in traces
    ]

    summary = {
        "schema": SCHEMA,
        "config_path": display_path(root, config_path),
        "current_path": display_path(root, current_path),
        "selected_artifact": display_path(root, selected_artifact),
        "iter_dir": display_path(root, iter_dir),
        "reference_artifact": display_path(root, reference_artifact),
        "fallback_reference_artifact": display_path(root, fallback_reference_artifact),
        "input_verification": input_verification,
        "baseline_guard_result": baseline_guard_result,
        "trace_definitions": trace_definitions,
        "traces": traces,
        "guard_drift_rows": guard_drift_rows,
        "training_metric_rows": training_metric_rows,
        "decision": decision,
        "constraints": {
            "arena_run": False,
            "mcts1200_lane_run": False,
            "promotion_run": False,
            "self_play_regenerated": bool(self_play_status["regenerated"]),
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
