#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.build_train_only_forensic_suite_from_selfplay import decode_state
from ml.alphazero_lite.corrected_guard_kill_gate import (
    DEFAULT_FALLBACK_REFERENCE_ARTIFACT,
    DEFAULT_REFERENCE_ARTIFACT,
    GUARD_ROW_IDS,
    run_corrected_guard_kill_gate,
)
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.pipeline import load_config
from ml.alphazero_lite.run_guard_safe_opening_full_pipeline_interaction_trace import (
    build_train_spec,
)
from ml.alphazero_lite.run_guard_safe_opening_low_epoch_drift_trace import (
    artifact_row_id,
    choose_device,
    current_checkpoint_path,
    export_checkpoint_artifact,
    train_one_epoch,
)
from ml.alphazero_lite.run_opening_017_puct_target_generation_audit import (
    merged_reference_rows,
    read_jsonl,
)
from ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment import (
    load_json,
)
from ml.alphazero_lite.self_play import build_eval_search_options
from ml.alphazero_lite.train import (
    PolicyValueNet,
    checkpoint_from_model,
    input_size_for_encoding,
    load_checkpoint_into_model,
    load_jsonl_replay,
    set_seed,
)


DEFAULT_CONFIG_PATH = Path(
    "ml/alphazero_lite/configs/"
    "exp_v3_guard_safe_opening_selected_w1_denoised_targets_opening_min_384.json"
)
DEFAULT_CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
DEFAULT_SELF_PLAY_PATH = Path(
    "/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_denoised_targets_opening_min_384_versions/"
    "exp-v3-guard-safe-opening-selected-w1-denoised-targets-opening-min-384-iter1/self_play.jsonl"
)
DEFAULT_SELECTED_ARTIFACT = Path(
    "/tmp/azlite_guard_safe_opening_replay/"
    "family_leave_one_out_without_opening_extra_turn_overbias.jsonl"
)
DEFAULT_GUARD_CONTROLS_ARTIFACT = Path(
    "/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path("/tmp/azlite_denoised_opening_min384_guard_tradeoff")
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_ROOT / "guard_tradeoff_summary.json"
DEFAULT_REPORT_PATH = Path(
    "docs/alphazero-lite-denoised-opening-min384-guard-tradeoff-results.md"
)
SCHEMA = "azlite_denoised_opening_min384_guard_tradeoff_v1"
TARGET_EPOCHS = (1, 2, 4)
TRACE_ROWS = (
    "capture_available-002",
    "capture_available-003",
    "capture_available-006",
    "capture_available-007",
    "capture_available-008",
)
TARGET_PRESSURE_ROWS = (
    "opening_plies_1_8-017",
    *TRACE_ROWS,
)
GUARD_BUDGETS = (384, 1200)
STABILITY_BUDGETS = (256, 384, 512, 768, 1200)


@dataclass(frozen=True)
class TraceSpec:
    name: str
    data_files: tuple[Path, ...]
    replay_weights: tuple[int | float, ...]
    status: str = "planned"
    notes: str = ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--current-artifact", type=Path, default=DEFAULT_CURRENT_ARTIFACT
    )
    parser.add_argument("--self-play", type=Path, default=DEFAULT_SELF_PLAY_PATH)
    parser.add_argument(
        "--selected-artifact", type=Path, default=DEFAULT_SELECTED_ARTIFACT
    )
    parser.add_argument(
        "--guard-controls-artifact",
        type=Path,
        default=DEFAULT_GUARD_CONTROLS_ARTIFACT,
    )
    parser.add_argument(
        "--reference-artifact", type=Path, default=DEFAULT_REFERENCE_ARTIFACT
    )
    parser.add_argument(
        "--fallback-reference-artifact",
        type=Path,
        default=DEFAULT_FALLBACK_REFERENCE_ARTIFACT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def format_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return str(bool(value)).lower()


def distribution_text(counter: dict[int, int]) -> str:
    if not counter:
        return "-"
    return (
        "{"
        + ", ".join(f"{move}:{count}" for move, count in sorted(counter.items()))
        + "}"
    )


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(sum(values) / len(values)), 6)


def policy_entropy(policy: list[float], legal_moves: list[int]) -> float:
    total = 0.0
    for move in legal_moves:
        probability = float(policy[move])
        if probability > 0.0:
            total -= probability * math.log(probability, 2)
    return round(total, 6)


def compute_policy_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1)


def compute_value_loss_vector(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    value_loss: str,
    huber_delta: float,
) -> torch.Tensor:
    if value_loss == "huber":
        return torch.nn.functional.smooth_l1_loss(
            predictions,
            targets,
            beta=huber_delta,
            reduction="none",
        ).reshape(-1)
    return torch.square(predictions - targets).reshape(-1)


def path_rows_or_size(path: Path) -> str:
    if not path.exists():
        return "-"
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return str(sum(1 for _ in handle))
    if path.is_dir():
        return str(sum(1 for _ in path.iterdir()))
    return str(path.stat().st_size)


def regenerate_self_play_if_needed(
    *, root: Path, config_path: Path, self_play_path: Path
) -> dict[str, Any]:
    if self_play_path.exists():
        return {
            "path": str(self_play_path),
            "status": "reused_existing",
            "regenerated": False,
            "notes": "reused PR39 denoised opening-min-384 self_play.jsonl",
        }

    command = [
        str(root / ".venv/bin/python"),
        "ml/alphazero_lite/pipeline.py",
        "--config",
        str(config_path),
        "--skip-step",
        "perspective_audit",
        "--skip-step",
        "train",
        "--skip-step",
        "export_artifact",
    ]
    result = subprocess.run(
        command, cwd=root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0 or not self_play_path.exists():
        raise RuntimeError(
            "failed to regenerate missing denoised opening-min-384 self-play; "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    return {
        "path": str(self_play_path),
        "status": "regenerated_self_play_only",
        "regenerated": True,
        "notes": "regenerated self-play only; train/export skipped",
    }


def artifact_rows_by_id(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_id: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return rows_by_id
    for row in read_jsonl(path):
        row_id = artifact_row_id(row)
        if row_id is None:
            continue
        rows_by_id.setdefault(row_id, []).append(row)
    return rows_by_id


def selected_artifact_exact_note(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    row = rows[0]
    policy = list(row.get("policy") or [])
    legal_moves = [int(move) for move in list(row.get("legal_moves") or [])]
    top_move = (
        min(legal_moves, key=lambda move: (-float(policy[move]), move))
        if legal_moves
        else None
    )
    if top_move is None:
        return None
    return (
        f"artifact_top={top_move}"
        f" move1={format_float(float(policy[1]) if len(policy) > 1 else None)}"
        f" move2={format_float(float(policy[2]) if len(policy) > 2 else None)}"
    )


def source_label(*, hit_count: int, in_selected: bool, in_guard_controls: bool) -> str:
    labels: list[str] = []
    if hit_count > 0:
        labels.append("self_play_exact")
    if in_selected:
        labels.append("selected_artifact")
    if in_guard_controls:
        labels.append("guard_controls")
    return " + ".join(labels) if labels else "none"


def build_target_pressure_rows(
    *,
    self_play_rows: list[dict[str, Any]],
    reference_rows: dict[str, dict[str, Any]],
    selected_rows_by_id: dict[str, list[dict[str, Any]]],
    guard_controls_rows_by_id: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical_to_row_id = {
        str(reference_rows[row_id]["canonical_state"]): row_id
        for row_id in TARGET_PRESSURE_ROWS
    }
    aggregates: dict[str, dict[str, Any]] = {
        row_id: {
            "hit_count": 0,
            "reference_mass": 0.0,
            "move_1_mass": 0.0,
            "move_2_mass": 0.0,
            "entropy": 0.0,
            "top_distribution": Counter(),
        }
        for row_id in TARGET_PRESSURE_ROWS
    }

    for row in self_play_rows:
        raw_state = decode_state(list(row["state"]))
        row_id = canonical_to_row_id.get(canonical_state_key(raw_state))
        if row_id is None:
            continue
        policy = list(row.get("policy") or [])
        if len(policy) != 6:
            continue
        legal_moves = [
            int(move) for move in KalahGame.from_state(raw_state).possible_moves()
        ]
        top_move = min(legal_moves, key=lambda move: (-float(policy[move]), move))
        reference_move = int(reference_rows[row_id]["reference_move"])
        aggregates[row_id]["hit_count"] += 1
        aggregates[row_id]["reference_mass"] += float(policy[reference_move])
        aggregates[row_id]["move_1_mass"] += float(policy[1])
        aggregates[row_id]["move_2_mass"] += float(policy[2])
        aggregates[row_id]["entropy"] += policy_entropy(policy, legal_moves)
        aggregates[row_id]["top_distribution"][top_move] += 1

    table_rows: list[dict[str, Any]] = []
    for row_id in TARGET_PRESSURE_ROWS:
        aggregate = aggregates[row_id]
        hit_count = int(aggregate["hit_count"])
        in_selected = row_id in selected_rows_by_id
        in_guard_controls = row_id in guard_controls_rows_by_id
        avg_reference = (
            None if hit_count <= 0 else aggregate["reference_mass"] / hit_count
        )
        avg_move_1 = None if hit_count <= 0 else aggregate["move_1_mass"] / hit_count
        avg_move_2 = None if hit_count <= 0 else aggregate["move_2_mass"] / hit_count
        avg_entropy = None if hit_count <= 0 else aggregate["entropy"] / hit_count
        notes: list[str] = []
        if hit_count <= 0:
            notes.append("no_exact_self_play_hits")
        if in_selected:
            selected_note = selected_artifact_exact_note(selected_rows_by_id[row_id])
            if selected_note:
                notes.append(f"selected_{selected_note}")
        if in_guard_controls:
            controls_note = selected_artifact_exact_note(
                guard_controls_rows_by_id[row_id]
            )
            if controls_note:
                notes.append(f"guard_controls_{controls_note}")
        if row_id == "opening_plies_1_8-017" and in_selected:
            notes.append("selected_artifact_adds_opening_anchor_row")
        if row_id == "capture_available-003" and hit_count <= 0 and in_selected:
            notes.append(
                "selected_artifact_supplies_exact_003_anchor_absent_from_self_play"
            )
        if row_id == "capture_available-002" and hit_count > 0 and in_selected:
            notes.append("self_play_and_selected_both_anchor_002_exactly")
        table_rows.append(
            {
                "row_id": row_id,
                "source": source_label(
                    hit_count=hit_count,
                    in_selected=in_selected,
                    in_guard_controls=in_guard_controls,
                ),
                "hit_count": hit_count,
                "corrected_reference_move": int(
                    reference_rows[row_id]["reference_move"]
                ),
                "average_reference_mass": None
                if avg_reference is None
                else round(float(avg_reference), 6),
                "move_1_mass": None
                if avg_move_1 is None
                else round(float(avg_move_1), 6),
                "move_2_mass": None
                if avg_move_2 is None
                else round(float(avg_move_2), 6),
                "top_target_distribution": dict(aggregate["top_distribution"]),
                "target_entropy": None
                if avg_entropy is None
                else round(float(avg_entropy), 6),
                "in_selected_artifact": in_selected,
                "in_guard_controls": in_guard_controls,
                "notes": ", ".join(notes) if notes else "ok",
            }
        )

    support = {
        "self_play_near_duplicate_support": False,
        "opening_family_metadata_in_self_play": False,
        "notes": "self_play.jsonl exposes exact canonical matches and search telemetry, but no row_id/opening_family labels for broader near-duplicate grouping",
    }
    return table_rows, support


def build_trace_specs(
    *,
    self_play_path: Path,
    selected_artifact: Path,
    guard_controls_artifact: Path,
) -> list[TraceSpec]:
    specs = [
        TraceSpec(
            name="denoised_opening_min384_self_play_only",
            data_files=(self_play_path,),
            replay_weights=(1,),
        ),
        TraceSpec(
            name="denoised_opening_min384_plus_selected_w1",
            data_files=(self_play_path, selected_artifact),
            replay_weights=(1, 1),
        ),
    ]
    if guard_controls_artifact.exists():
        specs.extend(
            [
                TraceSpec(
                    name="denoised_opening_min384_plus_guard_controls_w1",
                    data_files=(self_play_path, guard_controls_artifact),
                    replay_weights=(1, 1),
                ),
                TraceSpec(
                    name="denoised_opening_min384_plus_guard_controls_w2",
                    data_files=(self_play_path, guard_controls_artifact),
                    replay_weights=(1, 2),
                ),
                TraceSpec(
                    name="denoised_opening_min384_plus_selected_w1_guard_controls_w2",
                    data_files=(
                        self_play_path,
                        selected_artifact,
                        guard_controls_artifact,
                    ),
                    replay_weights=(1, 1, 2),
                ),
            ]
        )
    else:
        specs.extend(
            [
                TraceSpec(
                    name="denoised_opening_min384_plus_guard_controls_w1",
                    data_files=(self_play_path,),
                    replay_weights=(1,),
                    status="skipped",
                    notes="guard controls artifact missing",
                ),
                TraceSpec(
                    name="denoised_opening_min384_plus_guard_controls_w2",
                    data_files=(self_play_path,),
                    replay_weights=(1,),
                    status="skipped",
                    notes="guard controls artifact missing",
                ),
                TraceSpec(
                    name="denoised_opening_min384_plus_selected_w1_guard_controls_w2",
                    data_files=(self_play_path, selected_artifact),
                    replay_weights=(1, 1),
                    status="skipped",
                    notes="guard controls artifact missing",
                ),
            ]
        )
    specs.append(
        TraceSpec(
            name="denoised_opening_min384_plus_selected_half_if_supported",
            data_files=(self_play_path, selected_artifact),
            replay_weights=(1, 0.5),
            status="skipped",
            notes="fractional replay weights not supported cleanly by load_jsonl_replay",
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
    reference_rows: dict[str, dict[str, Any]],
    self_play_path: Path,
    selected_artifact: Path,
    guard_controls_artifact: Path,
) -> list[dict[str, Any]]:
    canonical_to_row_id = {
        str(reference_rows[row_id]["canonical_state"]): row_id
        for row_id in TARGET_PRESSURE_ROWS
    }
    row_infos: list[dict[str, Any]] = []
    for path in paths:
        source_kind = source_kind_for_path(
            path,
            self_play_path=self_play_path,
            selected_artifact=selected_artifact,
            guard_controls_artifact=guard_controls_artifact,
        )
        for row in read_jsonl(path):
            row_id = artifact_row_id(row)
            if row_id is None and source_kind == "self_play":
                raw_state = decode_state(list(row["state"]))
                row_id = canonical_to_row_id.get(canonical_state_key(raw_state))
            row_infos.append(
                {
                    "row_id": row_id,
                    "source_kind": source_kind,
                    "source_path": str(path),
                }
            )
    return row_infos


def evaluate_training_metrics(
    *,
    model: PolicyValueNet,
    compact_x: np.ndarray,
    compact_p: np.ndarray,
    compact_v: np.ndarray,
    row_infos: list[dict[str, Any]],
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
    total_losses = policy_ce + (train_spec.value_loss_weight * value_losses)

    def source_mean(source_kind: str) -> float | None:
        values = [
            float(policy_ce[index])
            for index, info in enumerate(row_infos)
            if info.get("source_kind") == source_kind
        ]
        return mean_or_none(values)

    guard_values = [
        float(policy_ce[index])
        for index, info in enumerate(row_infos)
        if info.get("row_id") in GUARD_ROW_IDS
    ]
    return {
        "policy_loss": round(float(np.mean(policy_ce)), 6),
        "value_loss": round(float(np.mean(value_losses)), 6),
        "total_loss": round(float(np.mean(total_losses)), 6),
        "self_play_cross_entropy": source_mean("self_play"),
        "selected_artifact_cross_entropy": source_mean("selected_artifact"),
        "guard_controls_cross_entropy": source_mean("guard_controls"),
        "guard_cross_entropy": mean_or_none(guard_values),
    }


def probe_budget_view(
    *,
    evaluator: ArtifactEvaluator,
    artifact_path: Path,
    state: dict[str, Any],
    legal_moves: list[int],
    reference_move: int,
    simulations: int,
    seed: int,
) -> dict[str, Any]:
    probe = evaluate_artifact_position(
        artifact_path=artifact_path,
        evaluator=evaluator,
        state=state,
        simulations=simulations,
        seed=seed + simulations,
        c_puct=1.25,
        search_options=dict(build_eval_search_options()),
        ablation_mode="full",
    )
    visits = list(probe.get("visits") or [])
    total_visits = sum(
        float(visits[move]) for move in legal_moves if move < len(visits)
    )
    child_stats = {
        int(child["move"]): child for child in list(probe.get("child_stats") or [])
    }
    selected_move = probe.get("selected_move")
    selected_q = None
    if selected_move is not None and int(selected_move) in child_stats:
        selected_q = float(child_stats[int(selected_move)].get("q_value", 0.0))
    reference_q = None
    if reference_move in child_stats:
        reference_q = float(child_stats[reference_move].get("q_value", 0.0))
    return {
        "selected_move": None if selected_move is None else int(selected_move),
        "selected_is_reference": selected_move == reference_move,
        "reference_visit_share": None
        if total_visits <= 0.0 or reference_move >= len(visits)
        else round(float(visits[reference_move]) / total_visits, 4),
        "selected_minus_reference_q_margin": None
        if selected_q is None or reference_q is None
        else round(selected_q - reference_q, 4),
    }


def diagnose_budget_stability(
    selected_by_budget: dict[int, int | None], reference_move: int
) -> tuple[int | None, str]:
    first_budget = next(
        (
            budget
            for budget in STABILITY_BUDGETS
            if selected_by_budget.get(budget) == reference_move
        ),
        None,
    )
    fail_384 = selected_by_budget.get(384) != reference_move
    pass_512 = selected_by_budget.get(512) == reference_move
    pass_768 = selected_by_budget.get(768) == reference_move
    pass_1200 = selected_by_budget.get(1200) == reference_move
    pass_256 = selected_by_budget.get(256) == reference_move
    if fail_384 and pass_256 and pass_512 and pass_768 and pass_1200:
        return first_budget, "384_outlier"
    if fail_384 and pass_512 and pass_768 and pass_1200:
        return first_budget, "fails_only_at_384_recovers_by_512"
    if fail_384 and not pass_512 and pass_768 and pass_1200:
        return first_budget, "fails_at_384_and_512_recovers_by_768"
    if fail_384 and not pass_512 and not pass_768 and pass_1200:
        return first_budget, "requires_1200"
    if first_budget is None:
        return None, "reference_never_selected"
    if first_budget == 256:
        return first_budget, "stable_from_256"
    return first_budget, f"recovers_by_{first_budget}"


def run_budget_stability_probe(
    *,
    trace_name: str,
    epoch: int,
    artifact_path: Path,
    reference_rows: dict[str, dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    evaluator = ArtifactEvaluator(artifact_path)
    rows: list[dict[str, Any]] = []
    for row_index, row_id in enumerate(TRACE_ROWS):
        row = reference_rows[row_id]
        legal_moves = [
            int(child["move"]) for child in list(row.get("child_stats") or [])
        ]
        reference_move = int(row["reference_move"])
        selected_by_budget: dict[int, int | None] = {}
        for budget in STABILITY_BUDGETS:
            view = probe_budget_view(
                evaluator=evaluator,
                artifact_path=artifact_path,
                state=dict(row["state"]),
                legal_moves=legal_moves,
                reference_move=reference_move,
                simulations=budget,
                seed=seed + row_index,
            )
            selected_by_budget[budget] = view["selected_move"]
        first_budget, diagnosis = diagnose_budget_stability(
            selected_by_budget, reference_move
        )
        notes = []
        if diagnosis == "384_outlier":
            notes.append("384_fails_while_neighbor_budgets_pass")
        if diagnosis.startswith("fails_only_at_384"):
            notes.append("512_or_higher_stable")
        rows.append(
            {
                "trace_name": trace_name,
                "epoch": epoch,
                "row_id": row_id,
                "selected_256": selected_by_budget.get(256),
                "selected_384": selected_by_budget.get(384),
                "selected_512": selected_by_budget.get(512),
                "selected_768": selected_by_budget.get(768),
                "selected_1200": selected_by_budget.get(1200),
                "first_budget_reference_selected": first_budget,
                "diagnosis": diagnosis,
                "notes": ", ".join(notes) if notes else "ok",
            }
        )
    return rows


def run_trace(
    *,
    spec: TraceSpec,
    init_checkpoint: Path,
    current_metadata: dict[str, Any],
    train_spec: Any,
    reference_artifact: Path,
    fallback_reference_artifact: Path,
    reference_rows: dict[str, dict[str, Any]],
    output_root: Path,
    device: torch.device,
    seed: int,
    self_play_path: Path,
    selected_artifact: Path,
    guard_controls_artifact: Path,
) -> dict[str, Any]:
    if spec.status == "skipped":
        return {
            "trace_name": spec.name,
            "data_files": [str(path) for path in spec.data_files],
            "replay_weights": list(spec.replay_weights),
            "epochs": list(TARGET_EPOCHS),
            "status": spec.status,
            "notes": spec.notes,
            "guard_rows": [],
            "training_metric_rows": [],
            "budget_stability_rows": [],
        }

    if any(not isinstance(weight, int) for weight in spec.replay_weights):
        return {
            "trace_name": spec.name,
            "data_files": [str(path) for path in spec.data_files],
            "replay_weights": list(spec.replay_weights),
            "epochs": list(TARGET_EPOCHS),
            "status": "skipped",
            "notes": "fractional replay weights not supported cleanly by load_jsonl_replay",
            "guard_rows": [],
            "training_metric_rows": [],
            "budget_stability_rows": [],
        }

    set_seed(seed)
    compact_x, compact_p, compact_v, replay_indexes = load_jsonl_replay(
        list(spec.data_files),
        [int(weight) for weight in spec.replay_weights],
        policy_target_mode=train_spec.policy_target_mode,
        value_target_mode=train_spec.value_target_mode,
    )
    compact_v = compact_v.astype(np.float32)
    row_infos = build_row_infos(
        spec.data_files,
        reference_rows=reference_rows,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=train_spec.lr)

    trace_root = output_root / "traces" / spec.name
    checkpoints_root = trace_root / "checkpoints"
    exports_root = output_root / "exports" / spec.name
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    exports_root.mkdir(parents=True, exist_ok=True)

    guard_rows: list[dict[str, Any]] = []
    training_metric_rows: list[dict[str, Any]] = []
    budget_stability_rows: list[dict[str, Any]] = []
    final_export_dir: Path | None = None

    for epoch in range(1, max(TARGET_EPOCHS) + 1):
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            compact_x=compact_x,
            compact_p=compact_p,
            compact_v=compact_v,
            replay_indexes=replay_indexes,
            batch_size=train_spec.batch_size,
            device=device,
            value_loss_weight=train_spec.value_loss_weight,
            value_loss=train_spec.value_loss,
            huber_delta=train_spec.huber_delta,
            grad_clip=train_spec.grad_clip,
        )
        if epoch not in TARGET_EPOCHS:
            continue

        checkpoint_path = checkpoints_root / f"epoch_{epoch}.npz"
        np.savez(checkpoint_path, **checkpoint_from_model(model))
        export_dir = export_checkpoint_artifact(
            checkpoint_path=checkpoint_path,
            export_dir=exports_root / f"epoch_{epoch}",
            current_metadata=current_metadata,
            version=f"{spec.name}-epoch-{epoch}",
        )
        final_export_dir = export_dir
        guard_eval = run_corrected_guard_kill_gate(
            candidate_path=export_dir,
            reference_artifact=reference_artifact,
            fallback_reference_artifact=fallback_reference_artifact,
            seed=seed + epoch,
        )
        metrics = evaluate_training_metrics(
            model=model,
            compact_x=compact_x,
            compact_p=compact_p,
            compact_v=compact_v,
            row_infos=row_infos,
            device=device,
            train_spec=train_spec,
        )
        for row in guard_eval["rows"]:
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
                    "gate_pass": bool(guard_eval["pass"]),
                    "notes": row.get("notes", "ok"),
                }
            )
        training_metric_rows.append(
            {
                "trace_name": spec.name,
                "epoch": epoch,
                **metrics,
                "notes": "full_dataset_snapshot",
            }
        )

    if final_export_dir is not None:
        budget_stability_rows = run_budget_stability_probe(
            trace_name=spec.name,
            epoch=max(TARGET_EPOCHS),
            artifact_path=final_export_dir,
            reference_rows=reference_rows,
            seed=seed + 1000,
        )

    return {
        "trace_name": spec.name,
        "data_files": [str(path) for path in spec.data_files],
        "replay_weights": list(spec.replay_weights),
        "epochs": list(TARGET_EPOCHS),
        "status": "completed",
        "notes": spec.notes or "ok",
        "guard_rows": guard_rows,
        "training_metric_rows": training_metric_rows,
        "budget_stability_rows": budget_stability_rows,
    }


def epoch_rows(
    guard_rows: list[dict[str, Any]], trace_name: str, epoch: int
) -> dict[str, dict[str, Any]]:
    return {
        row["row_id"]: row
        for row in guard_rows
        if row["trace_name"] == trace_name and int(row["epoch"]) == epoch
    }


def trace_gate_pass_at_epoch(
    guard_rows: list[dict[str, Any]], trace_name: str, epoch: int
) -> bool:
    rows = epoch_rows(guard_rows, trace_name, epoch)
    return bool(rows) and all(bool(rows[row_id]["row_pass"]) for row_id in TRACE_ROWS)


def all_final_rows_fail_only_at_384(budget_rows: list[dict[str, Any]]) -> bool:
    relevant = [row for row in budget_rows if int(row["epoch"]) == max(TARGET_EPOCHS)]
    if not relevant:
        return False
    for row in relevant:
        selected_384 = row.get("selected_384")
        selected_512 = row.get("selected_512")
        selected_768 = row.get("selected_768")
        selected_1200 = row.get("selected_1200")
        reference_move = 1 if row["row_id"] == "capture_available-008" else 2
        if selected_384 == reference_move:
            continue
        if not (
            selected_512 == reference_move
            and selected_768 == reference_move
            and selected_1200 == reference_move
        ):
            return False
    return True


def classify_results(
    *,
    trace_rows: list[dict[str, Any]],
    guard_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
) -> tuple[str, str, list[str]]:
    evidence: list[str] = []

    guard_anchor_trace_names = {
        "denoised_opening_min384_plus_guard_controls_w1",
        "denoised_opening_min384_plus_guard_controls_w2",
    }
    for trace_name in guard_anchor_trace_names:
        if trace_gate_pass_at_epoch(guard_rows, trace_name, 4):
            evidence.append(f"{trace_name} passes all five rows at epoch 4")
            return (
                "guard_anchor_sufficient",
                "run one controlled production-scale denoised opening-min-384 lane with guard controls and corrected guard kill gate before arena.",
                evidence,
            )

    if trace_gate_pass_at_epoch(
        guard_rows,
        "denoised_opening_min384_plus_selected_w1_guard_controls_w2",
        4,
    ):
        evidence.append(
            "selected w1 plus guard controls w2 passes all five rows at epoch 4"
        )
        evidence.append("pure guard-controls traces alone still miss 002 and/or 003")
        return (
            "guard_anchor_sufficient",
            "run one controlled production-scale denoised opening-min-384 lane with guard controls and corrected guard kill gate before arena.",
            evidence,
        )

    self_play_only = epoch_rows(guard_rows, "denoised_opening_min384_self_play_only", 4)
    selected_w1 = epoch_rows(guard_rows, "denoised_opening_min384_plus_selected_w1", 4)
    if self_play_only and selected_w1:
        improves_003 = not bool(
            self_play_only["capture_available-003"]["row_pass"]
        ) and bool(selected_w1["capture_available-003"]["row_pass"])
        hurts_002 = bool(
            self_play_only["capture_available-002"]["row_pass"]
        ) and not bool(selected_w1["capture_available-002"]["row_pass"])
        if improves_003 and hurts_002:
            evidence.append("selected w1 flips 003 to pass and 002 to fail at epoch 4")
            return (
                "selected_artifact_002_003_tradeoff",
                "drop selected artifact from the next lane; use denoised opening-min-384 self-play plus guard controls only.",
                evidence,
            )

    if all_final_rows_fail_only_at_384(budget_rows):
        evidence.append(
            "every final-checkpoint failure recovers by 512/768/1200 and is 384-only"
        )
        return (
            "guard_budget_too_brittle",
            "audit whether the corrected guard should use 768 or require reference visit-share threshold instead of strict 384 top-selection.",
            evidence,
        )

    any_both_pass = False
    for trace in trace_rows:
        if trace["status"] != "completed":
            continue
        rows = epoch_rows(guard_rows, trace["trace_name"], 4)
        if not rows:
            continue
        if bool(rows["capture_available-002"]["row_pass"]) and bool(
            rows["capture_available-003"]["row_pass"]
        ):
            any_both_pass = True
            break
    if not any_both_pass:
        evidence.append("no small mix makes both 002 and 003 pass together at epoch 4")
        return (
            "unresolved_guard_tradeoff",
            "inspect per-row target/Q traces for 002 and 003 under denoised opening-min-384 training.",
            evidence,
        )

    if trace_gate_pass_at_epoch(
        guard_rows, "denoised_opening_min384_self_play_only", 4
    ):
        evidence.append(
            "self-play-only rerun now passes the full five-row guard at epoch 4"
        )
        return (
            "nondeterministic_training_trace",
            "rerun with fixed data order/checkpoint hashes before any production lane.",
            evidence,
        )

    evidence.append("defaulted to unresolved tradeoff after higher-priority rules")
    return (
        "unresolved_guard_tradeoff",
        "inspect per-row target/Q traces for 002 and 003 under denoised opening-min-384 training.",
        evidence,
    )


def render_report(summary: dict[str, Any]) -> str:
    final_epoch = max(TARGET_EPOCHS)

    def final_guard_row(trace_name: str, row_id: str) -> dict[str, Any] | None:
        for row in summary["guard_rows"]:
            if (
                row["trace_name"] == trace_name
                and int(row["epoch"]) == final_epoch
                and row["row_id"] == row_id
            ):
                return row
        return None

    def final_stability_row(trace_name: str, row_id: str) -> dict[str, Any] | None:
        for row in summary["budget_stability_rows"]:
            if (
                row["trace_name"] == trace_name
                and int(row["epoch"]) == final_epoch
                and row["row_id"] == row_id
            ):
                return row
        return None

    lines = [
        "# AlphaZero-lite Denoised Opening-Min384 Guard Tradeoff Results",
        "",
        "## 1. Context",
        "",
        "- PR #39 added denoised self-play targets, optional root telemetry, and opening-min simulation controls without changing default production behavior.",
        "- PR #39 ended as `denoised_targets_partial_003_needs_more_sims`: no arena, no promotion, and no full corrected guard pass.",
        "- The remaining question is narrower than generic denoising: under denoised opening-min-384 training, self-play-only helps `capture_available-002` while selected replay helps `capture_available-003`.",
        "",
        "## 2. Why PR #39 Was Not Enough",
        "",
        "- `denoised_opening_min384_self_play_only` at epoch 4 previously passed `002/006/007/008` but still failed `003` at `384` sims.",
        "- `denoised_opening_min384_plus_selected_w1` at epoch 4 previously passed `003/006/007/008` but regressed `002` at `384` sims.",
        "- No prior trace achieved a full corrected guard pass, so the remaining issue is a `002` vs `003` tradeoff under the `384`-sim gate.",
        "",
        "## 3. Input and Target-Pressure Inspection",
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
            f"Near-duplicate scan support: {summary['target_pressure_support']['notes']}.",
            "",
            "| row_id | source | hit_count | corrected_reference_move | average_reference_mass | move_1_mass | move_2_mass | top_target_distribution | in_selected_artifact | in_guard_controls | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["target_pressure_rows"]:
        lines.append(
            f"| {row['row_id']} | {row['source']} | {row['hit_count']} | {row['corrected_reference_move']} | {format_float(row['average_reference_mass'])} | {format_float(row['move_1_mass'])} | {format_float(row['move_2_mass'])} | `{distribution_text(row['top_target_distribution'])}` | {format_bool(row['in_selected_artifact'])} | {format_bool(row['in_guard_controls'])} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 4. Trace Definitions",
            "",
            "| trace_name | data_files | replay_weights | epochs | status | notes |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["trace_rows"]:
        lines.append(
            f"| {row['trace_name']} | `{json.dumps(row['data_files'])}` | `{json.dumps(row['replay_weights'])}` | `{json.dumps(row['epochs'])}` | `{row['status']}` | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 5. Compact Decision Table",
            "",
            "Final epoch summary for the five corrected guard rows. Cells are `selected_move_384 / selected_move_1200`.",
            "",
            "| trace_name | capture_available-002 | capture_available-003 | capture_available-006 | capture_available-007 | capture_available-008 | gate_status |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for trace in summary["trace_rows"]:
        if trace["status"] != "completed":
            continue
        cell_values: list[str] = []
        gate_status = "pass"
        for row_id in TRACE_ROWS:
            row = final_guard_row(trace["trace_name"], row_id)
            if row is None:
                cell_values.append("`- / -`")
                gate_status = "fail"
                continue
            cell_values.append(
                f"`{row['selected_move_384'] if row['selected_move_384'] is not None else '-'} / {row['selected_move_1200'] if row['selected_move_1200'] is not None else '-'}`"
            )
            if not row["row_pass"]:
                gate_status = "fail"
        lines.append(
            f"| {trace['trace_name']} | {' | '.join(cell_values)} | {gate_status} |"
        )
    lines.extend(
        [
            "",
            "384-sim stability shorthand for the three rows that drove the tradeoff:",
            "",
            "| trace_name | capture_available-002 first_correct_budget | capture_available-003 first_correct_budget | capture_available-007 first_correct_budget | notes |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for trace in summary["trace_rows"]:
        if trace["status"] != "completed":
            continue
        stability_rows = [
            final_stability_row(trace["trace_name"], row_id)
            for row_id in (
                "capture_available-002",
                "capture_available-003",
                "capture_available-007",
            )
        ]
        notes: list[str] = []
        if (
            stability_rows[0] is not None
            and stability_rows[1] is not None
            and stability_rows[2] is not None
        ):
            row_002 = cast(dict[str, Any], stability_rows[0])
            row_003 = cast(dict[str, Any], stability_rows[1])
            row_007 = cast(dict[str, Any], stability_rows[2])
            if (
                row_002["first_budget_reference_selected"] == 1200
                and row_003["first_budget_reference_selected"] == 1200
                and row_007["first_budget_reference_selected"] == 1200
            ):
                notes.append("very brittle")
            if row_003["first_budget_reference_selected"] is None:
                notes.append("selected replay alone does not recover `003`")
            if (
                row_002["first_budget_reference_selected"] == 1200
                and row_003["first_budget_reference_selected"] == 1200
                and row_007["first_budget_reference_selected"] == 512
            ):
                notes.append("partial guard anchor")
            if (
                row_002["first_budget_reference_selected"] == 512
                and row_003["first_budget_reference_selected"] == 768
                and row_007["first_budget_reference_selected"] == 768
            ):
                notes.append("stronger guard anchor, still insufficient at `384`")
            if (
                row_002["first_budget_reference_selected"] == 384
                and row_003["first_budget_reference_selected"] == 384
                and row_007["first_budget_reference_selected"] == 256
            ):
                notes.append("only trace stable enough for the corrected gate")
        lines.append(
            "| "
            + trace["trace_name"]
            + " | "
            + " | ".join(
                f"`{row['first_budget_reference_selected'] if row is not None and row['first_budget_reference_selected'] is not None else '-'}`"
                for row in stability_rows
            )
            + " | "
            + ("; ".join(notes) if notes else "ok")
            + " |"
        )
    lines.extend(
        [
            "",
            "## 6. Corrected Guard Results",
            "",
            "| trace_name | epoch | row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | reference_visit_share_384 | reference_visit_share_1200 | row_pass | gate_pass | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["guard_rows"]:
        lines.append(
            f"| {row['trace_name']} | {row['epoch']} | {row['row_id']} | {row['corrected_reference_move']} | {row['selected_move_384'] if row['selected_move_384'] is not None else '-'} | {row['selected_move_1200'] if row['selected_move_1200'] is not None else '-'} | {format_float(row['reference_visit_share_384'])} | {format_float(row['reference_visit_share_1200'])} | {format_bool(row['row_pass'])} | {format_bool(row['gate_pass'])} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 7. 384-Sim Stability Probe",
            "",
            "| trace_name | epoch | row_id | selected_256 | selected_384 | selected_512 | selected_768 | selected_1200 | first_budget_reference_selected | diagnosis | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["budget_stability_rows"]:
        lines.append(
            f"| {row['trace_name']} | {row['epoch']} | {row['row_id']} | {row['selected_256'] if row['selected_256'] is not None else '-'} | {row['selected_384'] if row['selected_384'] is not None else '-'} | {row['selected_512'] if row['selected_512'] is not None else '-'} | {row['selected_768'] if row['selected_768'] is not None else '-'} | {row['selected_1200'] if row['selected_1200'] is not None else '-'} | {row['first_budget_reference_selected'] if row['first_budget_reference_selected'] is not None else '-'} | {row['diagnosis']} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 8. Training Metrics",
            "",
            "| trace_name | epoch | policy_loss | value_loss | total_loss | self_play_cross_entropy | selected_artifact_cross_entropy | guard_controls_cross_entropy | guard_cross_entropy | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["training_metric_rows"]:
        lines.append(
            f"| {row['trace_name']} | {row['epoch']} | {format_float(row['policy_loss'])} | {format_float(row['value_loss'])} | {format_float(row['total_loss'])} | {format_float(row['self_play_cross_entropy'])} | {format_float(row['selected_artifact_cross_entropy'])} | {format_float(row['guard_controls_cross_entropy'])} | {format_float(row['guard_cross_entropy'])} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 9. Interpretation",
            "",
            f"- Classification: `{summary['classification']}`.",
            f"- Evidence: {'; '.join(summary['classification_evidence'])}.",
            f"- Selected artifact summary: `{summary['artifact_summaries']['selected']['notes']}` with `{summary['artifact_summaries']['selected']['opening_row_count']}` opening rows.",
            f"- Guard controls summary: `{summary['artifact_summaries']['guard_controls']['notes']}` with `{summary['artifact_summaries']['guard_controls']['row_count']}` exact guard rows.",
            "",
            "## 10. Exactly One Recommended Next Action",
            "",
            f"Recommendation: **{summary['recommended_next_action']}**",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    config_path = resolve_path(root, args.config)
    current_artifact = resolve_path(root, args.current_artifact)
    self_play_path = resolve_path(root, args.self_play)
    selected_artifact = resolve_path(root, args.selected_artifact)
    guard_controls_artifact = resolve_path(root, args.guard_controls_artifact)
    reference_artifact = resolve_path(root, args.reference_artifact)
    fallback_reference_artifact = resolve_path(root, args.fallback_reference_artifact)
    output_root = resolve_path(root, args.output_root)
    summary_path = resolve_path(root, args.summary_path)
    report_path = resolve_path(root, args.report_path)
    output_root.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    self_play_status = regenerate_self_play_if_needed(
        root=root,
        config_path=config_path,
        self_play_path=self_play_path,
    )
    reference_rows = merged_reference_rows(
        reference_artifact, fallback_reference_artifact
    )
    self_play_rows = read_jsonl(self_play_path)
    selected_rows_by_id = artifact_rows_by_id(selected_artifact)
    guard_controls_rows_by_id = artifact_rows_by_id(guard_controls_artifact)
    target_pressure_rows, target_pressure_support = build_target_pressure_rows(
        self_play_rows=self_play_rows,
        reference_rows=reference_rows,
        selected_rows_by_id=selected_rows_by_id,
        guard_controls_rows_by_id=guard_controls_rows_by_id,
    )

    current_metadata = load_json(current_artifact / "metadata.json")
    init_checkpoint = current_checkpoint_path(current_artifact, output_root)
    train_spec = build_train_spec(config)
    device = choose_device(args.device)

    trace_specs = build_trace_specs(
        self_play_path=self_play_path,
        selected_artifact=selected_artifact,
        guard_controls_artifact=guard_controls_artifact,
    )
    trace_rows: list[dict[str, Any]] = []
    guard_rows: list[dict[str, Any]] = []
    budget_stability_rows: list[dict[str, Any]] = []
    training_metric_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(trace_specs):
        trace = run_trace(
            spec=spec,
            init_checkpoint=init_checkpoint,
            current_metadata=current_metadata,
            train_spec=train_spec,
            reference_artifact=reference_artifact,
            fallback_reference_artifact=fallback_reference_artifact,
            reference_rows=reference_rows,
            output_root=output_root,
            device=device,
            seed=42 + (index * 100),
            self_play_path=self_play_path,
            selected_artifact=selected_artifact,
            guard_controls_artifact=guard_controls_artifact,
        )
        trace_rows.append(
            {
                "trace_name": trace["trace_name"],
                "data_files": trace["data_files"],
                "replay_weights": trace["replay_weights"],
                "epochs": trace["epochs"],
                "status": trace["status"],
                "notes": trace["notes"],
            }
        )
        guard_rows.extend(trace["guard_rows"])
        budget_stability_rows.extend(trace["budget_stability_rows"])
        training_metric_rows.extend(trace["training_metric_rows"])

    selected_summary_path = selected_artifact.with_name(
        selected_artifact.stem + "_summary.json"
    )
    guard_controls_summary_path = guard_controls_artifact.with_name(
        guard_controls_artifact.stem + "_summary.json"
    )
    selected_summary = (
        load_json(selected_summary_path) if selected_summary_path.exists() else {}
    )
    guard_controls_summary = (
        load_json(guard_controls_summary_path)
        if guard_controls_summary_path.exists()
        else {}
    )

    classification, recommended_next_action, classification_evidence = classify_results(
        trace_rows=trace_rows,
        guard_rows=guard_rows,
        budget_rows=budget_stability_rows,
    )

    input_verification = [
        {
            "input": "denoised_opening_min384_self_play",
            "path": display_path(root, self_play_path),
            "exists": self_play_path.exists(),
            "rows_or_size": path_rows_or_size(self_play_path),
            "status": self_play_status["status"],
            "notes": self_play_status["notes"],
        },
        {
            "input": "selected_artifact",
            "path": display_path(root, selected_artifact),
            "exists": selected_artifact.exists(),
            "rows_or_size": path_rows_or_size(selected_artifact),
            "status": "ok" if selected_artifact.exists() else "missing",
            "notes": selected_summary.get("notes", "summary_missing"),
        },
        {
            "input": "guard_controls_artifact",
            "path": display_path(root, guard_controls_artifact),
            "exists": guard_controls_artifact.exists(),
            "rows_or_size": path_rows_or_size(guard_controls_artifact),
            "status": "ok" if guard_controls_artifact.exists() else "optional_missing",
            "notes": guard_controls_summary.get("notes", "summary_missing"),
        },
        {
            "input": "corrected_references",
            "path": display_path(root, reference_artifact),
            "exists": reference_artifact.exists(),
            "rows_or_size": path_rows_or_size(reference_artifact),
            "status": "ok" if reference_artifact.exists() else "missing",
            "notes": "corrected incumbent forensic references",
        },
        {
            "input": "current_artifact",
            "path": display_path(root, current_artifact),
            "exists": current_artifact.exists(),
            "rows_or_size": path_rows_or_size(current_artifact),
            "status": "ok" if current_artifact.exists() else "missing",
            "notes": "initializer only; never overwritten",
        },
    ]

    summary = {
        "schema": SCHEMA,
        "config_path": display_path(root, config_path),
        "paths": {
            "current_artifact": display_path(root, current_artifact),
            "self_play": display_path(root, self_play_path),
            "selected_artifact": display_path(root, selected_artifact),
            "guard_controls_artifact": display_path(root, guard_controls_artifact),
            "reference_artifact": display_path(root, reference_artifact),
        },
        "constraints": {
            "arena_run": False,
            "mcts1200_lane_run": False,
            "promotion_run": False,
            "overwrite_current": False,
            "production_scale_lane": False,
            "default_self_play_changed": False,
        },
        "input_verification": input_verification,
        "target_pressure_support": target_pressure_support,
        "target_pressure_rows": target_pressure_rows,
        "trace_rows": trace_rows,
        "guard_rows": guard_rows,
        "budget_stability_rows": budget_stability_rows,
        "training_metric_rows": training_metric_rows,
        "classification": classification,
        "classification_evidence": classification_evidence,
        "recommended_next_action": recommended_next_action,
        "artifact_summaries": {
            "selected": {
                "row_count": selected_summary.get("row_count"),
                "opening_row_count": selected_summary.get("opening_row_count"),
                "notes": selected_summary.get("notes", "summary_missing"),
            },
            "guard_controls": {
                "row_count": guard_controls_summary.get("row_count"),
                "opening_row_count": guard_controls_summary.get("opening_row_count"),
                "notes": guard_controls_summary.get("notes", "summary_missing"),
            },
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
                "classification": classification,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
