#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint
from ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment import (
    load_json,
    row_map_from_reference,
)
from ml.alphazero_lite.self_play import build_eval_search_options
from ml.alphazero_lite.train import (
    PolicyValueNet,
    checkpoint_from_model,
    compute_policy_cross_entropy,
    compute_value_loss_vector,
    input_size_for_encoding,
    load_checkpoint_into_model,
    load_jsonl_replay,
    resolve_hidden_sizes,
    set_seed,
    split_replay_positions_by_source_row,
    train_one_epoch as shared_train_one_epoch,
)


DEFAULT_REFERENCE_ARTIFACT = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_FALLBACK_REFERENCE_ARTIFACT = Path(
    "ml/alphazero_lite/fixtures/incumbent_train_only_forensic_references_v1.json"
)
DEFAULT_CURRENT_PATH = Path("storage/ai/alphazero_lite/current")
DEFAULT_SELECTED_ARTIFACT = Path(
    "/tmp/azlite_guard_safe_opening_replay/"
    "family_leave_one_out_without_opening_extra_turn_overbias.jsonl"
)
DEFAULT_GUARD_CONTROLS_ARTIFACT = Path(
    "/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl"
)
DEFAULT_MICROTRACE_ARTIFACTS = {
    "without_opening_edge_move_5_preference": Path(
        "/tmp/azlite_guard_safe_opening_replay/"
        "family_leave_one_out_without_opening_edge_move_5_preference.jsonl"
    ),
    "without_opening_missed_extra_turn_continuation": Path(
        "/tmp/azlite_guard_safe_opening_replay/"
        "family_leave_one_out_without_opening_missed_extra_turn_continuation.jsonl"
    ),
    "without_opening_extra_turn_overbias": DEFAULT_SELECTED_ARTIFACT,
}
DEFAULT_BASE_CONFIG = Path(
    "ml/alphazero_lite/configs/aggressive_v3_targeted_hard_state_replay.json"
)
DEFAULT_OUTPUT_ROOT = Path("/tmp/azlite_guard_safe_opening_drift_trace")
DEFAULT_SUMMARY_PATH = (
    DEFAULT_OUTPUT_ROOT / "guard_safe_opening_low_epoch_drift_trace_summary.json"
)
DEFAULT_REPORT_PATH = Path(
    "docs/alphazero-lite-guard-safe-opening-low-epoch-drift-trace-results.md"
)
SCHEMA = "azlite_guard_safe_opening_low_epoch_drift_trace_v1"
COMPATIBILITY_BASIS = "post_init_regression_relative_to_initializer"
EXPECTED_SELECTED_ROW_COUNT = 26
GUARD_ROW_IDS = (
    "capture_available-002",
    "capture_available-003",
    "capture_available-006",
    "capture_available-007",
    "capture_available-008",
)
REFERENCE_MOVES = {
    "capture_available-002": 2,
    "capture_available-003": 2,
    "capture_available-006": 2,
    "capture_available-007": 2,
    "capture_available-008": 1,
}
STALE_GUARD_SOURCE_MARKERS = (
    "train_only_forensic_references",
    "capture_002_003_rule_collision_diagnostic",
)
DEFAULT_SEED = 42
DEFAULT_C_PUCT = 1.25
DEFAULT_POLICY_TARGET_MODE = "sharpened"
DEFAULT_VALUE_TARGET_MODE = "sharpened"
DEFAULT_BATCH_SIZE = 32
CHECKPOINT_BUDGETS = (384, 1200)


@dataclass(frozen=True)
class TraceSpec:
    name: str
    data_files: tuple[Path, ...]
    replay_weights: tuple[int, ...]
    lr: float
    checkpoints: tuple[int, ...]
    batch_size: int
    notes: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path.resolve())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def artifact_row_id(row: dict[str, Any]) -> str | None:
    source_runs = list(row.get("source_runs") or [])
    if source_runs:
        row_id = source_runs[0].get("id")
        if isinstance(row_id, str) and row_id:
            return row_id
    row_id = row.get("id")
    if isinstance(row_id, str) and row_id:
        return row_id
    return None


def top_policy_move(policy: list[float], legal_moves: list[int]) -> int | None:
    if not legal_moves:
        return None
    return min(legal_moves, key=lambda move: (-float(policy[move]), move))


def policy_entropy(policy: list[float], legal_moves: list[int]) -> float:
    total = 0.0
    for move in legal_moves:
        probability = float(policy[move])
        if probability > 0.0:
            total -= probability * math.log(probability, 2)
    return round(total, 4)


def metric_rank(distribution: dict[int, float], move: int) -> int | None:
    if move not in distribution:
        return None
    ranked = sorted(distribution, key=lambda key: (-distribution[key], key))
    for index, candidate in enumerate(ranked, start=1):
        if candidate == move:
            return index
    return None


def current_checkpoint_path(current_path: Path, output_root: Path) -> Path:
    checkpoint_candidate = current_path / "checkpoint.npz"
    if checkpoint_candidate.exists():
        return checkpoint_candidate
    model_candidate = current_path / "model.npz"
    if model_candidate.exists():
        return model_candidate
    weights_candidate = current_path / "weights.json"
    if weights_candidate.exists():
        return materialize_weights_json_checkpoint(
            weights_path=weights_candidate,
            out_path=output_root / "current_init_checkpoint.npz",
        )
    raise FileNotFoundError(
        f"current artifact must contain checkpoint.npz, model.npz, or weights.json: {current_path}"
    )


def export_checkpoint_artifact(
    *,
    checkpoint_path: Path,
    export_dir: Path,
    current_metadata: dict[str, Any],
    version: str,
) -> Path:
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    model_path = export_dir / "model.npz"
    shutil.copy2(checkpoint_path, model_path)
    checkpoint = np.load(model_path)
    weights_payload = {key: checkpoint[key].tolist() for key in checkpoint.files}
    (export_dir / "weights.json").write_text(
        json.dumps(weights_payload), encoding="utf-8"
    )
    metadata = {
        "schema_version": current_metadata.get("schema_version", "azlite_model_v1"),
        "version": version,
        "game": current_metadata.get("game", "kalah"),
        "rules_version": current_metadata.get("rules_version", "kalah_v1"),
        "input_encoding": current_metadata["input_encoding"],
        "feature_count": current_metadata["feature_count"],
        "policy_size": current_metadata.get("policy_size", 6),
        "feature_order": current_metadata.get("feature_order", []),
        "architecture": current_metadata["architecture"],
        "normalization": current_metadata.get("normalization", {}),
        "training": {"self_play_games": 0},
        "metrics": {"policy_loss": 0.0, "value_loss": 0.0},
        "framework": current_metadata.get("framework", "numpy"),
    }
    (export_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return export_dir


def parameter_group(name: str) -> str:
    if name.startswith("policy_head") or name.startswith("policy_hidden_layer"):
        return "policy"
    if name.startswith("value_head") or name.startswith("value_hidden_layer"):
        return "value"
    return "trunk"


def delta_norms(
    current_state: dict[str, torch.Tensor], initial_state: dict[str, torch.Tensor]
) -> dict[str, float]:
    squared = {"policy": 0.0, "value": 0.0, "trunk": 0.0}
    for name, tensor in current_state.items():
        if name not in initial_state:
            continue
        delta = tensor.detach().cpu() - initial_state[name]
        squared[parameter_group(name)] += float(torch.sum(delta * delta).item())
    return {
        "policy": round(math.sqrt(squared["policy"]), 6),
        "value": round(math.sqrt(squared["value"]), 6),
        "trunk": round(math.sqrt(squared["trunk"]), 6),
    }


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(requested)


def evaluate_dataset_metrics(
    *,
    model: PolicyValueNet,
    compact_x: np.ndarray,
    compact_p: np.ndarray,
    compact_v: np.ndarray,
    row_infos: list[dict[str, Any]],
    device: torch.device,
    value_loss_weight: float,
    value_loss: str,
    huber_delta: float,
    initial_state: dict[str, torch.Tensor],
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
        value_loss_vector = (
            compute_value_loss_vector(
                value_pred,
                v_tensor,
                value_loss=value_loss,
                huber_delta=huber_delta,
            )
            .detach()
            .cpu()
            .numpy()
        )
    total_loss = policy_ce + (value_loss_weight * value_loss_vector)
    guard_indexes = [
        index
        for index, row in enumerate(row_infos)
        if row.get("row_id") in GUARD_ROW_IDS
    ]
    non_guard_indexes = [
        index
        for index, row in enumerate(row_infos)
        if row.get("row_id") not in GUARD_ROW_IDS
    ]
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    norms = delta_norms(state_dict, initial_state)

    def mean_or_none(indexes: list[int], values: np.ndarray) -> float | None:
        if not indexes:
            return None
        return round(float(np.mean(values[indexes])), 6)

    return {
        "policy_loss": round(float(np.mean(policy_ce)), 6),
        "value_loss": round(float(np.mean(value_loss_vector)), 6),
        "total_loss": round(float(np.mean(total_loss)), 6),
        "guard_cross_entropy": mean_or_none(guard_indexes, policy_ce),
        "non_guard_opening_cross_entropy": mean_or_none(non_guard_indexes, policy_ce),
        "policy_head_delta_norm": norms["policy"],
        "value_head_delta_norm": norms["value"],
        "trunk_delta_norm": norms["trunk"],
    }


def train_one_epoch(
    *,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    compact_x: np.ndarray,
    compact_p: np.ndarray,
    compact_v: np.ndarray,
    replay_indexes: np.ndarray,
    batch_size: int,
    device: torch.device,
    value_loss_weight: float,
    value_loss: str,
    huber_delta: float,
    grad_clip: float | None,
) -> dict[str, Any]:
    metrics = shared_train_one_epoch(
        model=model,
        optimizer=optimizer,
        compact_x=compact_x,
        compact_p=compact_p,
        compact_v=compact_v,
        replay_indexes=replay_indexes,
        batch_size=batch_size,
        device=device,
        value_loss_weight=value_loss_weight,
        value_loss=value_loss,
        huber_delta=huber_delta,
        grad_clip=grad_clip,
    )
    return {
        "policy_loss": round(float(metrics["policy_loss"] or 0.0), 6),
        "value_loss": round(float(metrics["value_loss"] or 0.0), 6),
        "total_loss": round(float(metrics["total_loss"] or 0.0), 6),
        "gradient_norm": round(float(metrics["gradient_norm"] or 0.0), 6)
        if metrics["gradient_norm"] is not None
        else None,
    }


def reference_row_payload(
    reference_rows: dict[str, dict[str, Any]], row_id: str
) -> dict[str, Any]:
    row = reference_rows[row_id]
    legal_moves = [int(child["move"]) for child in list(row.get("child_stats") or [])]
    return {
        "row_id": row_id,
        "state": dict(row["state"]),
        "legal_moves": legal_moves,
        "corrected_reference_move": int(row["reference_move"]),
    }


def reference_or_artifact_guard_row(
    *,
    row_id: str,
    reference_rows: dict[str, dict[str, Any]],
    artifact_rows_by_id: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if row_id in reference_rows:
        return reference_row_payload(reference_rows, row_id)
    artifact_rows = list(artifact_rows_by_id.get(row_id, []))
    if not artifact_rows:
        raise KeyError(row_id)
    corrected_rows = [
        row
        for row in artifact_rows
        if int(row.get("reference_move", -1)) == REFERENCE_MOVES[row_id]
    ]
    if not corrected_rows:
        raise KeyError(row_id)
    artifact_row = corrected_rows[0]
    return {
        "row_id": row_id,
        "state": dict(artifact_row["raw_state"]),
        "legal_moves": [
            int(move) for move in list(artifact_row.get("legal_moves") or [])
        ],
        "corrected_reference_move": REFERENCE_MOVES[row_id],
    }


def evaluate_guard_rows(
    *,
    artifact_path: Path,
    reference_rows: dict[str, dict[str, Any]],
    artifact_rows_by_id: dict[str, list[dict[str, Any]]],
    checkpoint_step: int,
    trace_name: str,
    run_1200: bool,
    seed: int,
) -> list[dict[str, Any]]:
    evaluator = ArtifactEvaluator(artifact_path)
    search_options = dict(build_eval_search_options())
    rows: list[dict[str, Any]] = []
    for row_id in GUARD_ROW_IDS:
        row = reference_or_artifact_guard_row(
            row_id=row_id,
            reference_rows=reference_rows,
            artifact_rows_by_id=artifact_rows_by_id,
        )
        game = KalahGame.from_state(row["state"])
        policy, root_value = evaluator.evaluate(game)
        policy_distribution = {
            move: round(float(policy[move]), 4) for move in row["legal_moves"]
        }
        reference_move = row["corrected_reference_move"]
        probe_384 = evaluate_artifact_position(
            artifact_path=artifact_path,
            evaluator=evaluator,
            state=row["state"],
            simulations=384,
            seed=seed,
            c_puct=DEFAULT_C_PUCT,
            search_options=search_options,
            ablation_mode="full",
        )
        probe_1200 = None
        if run_1200:
            probe_1200 = evaluate_artifact_position(
                artifact_path=artifact_path,
                evaluator=evaluator,
                state=row["state"],
                simulations=1200,
                seed=seed,
                c_puct=DEFAULT_C_PUCT,
                search_options=search_options,
                ablation_mode="full",
            )

        def probe_view(probe: dict[str, Any] | None) -> dict[str, Any]:
            if probe is None:
                return {
                    "selected_move": None,
                    "selected_is_reference": None,
                    "reference_visit_share": None,
                    "selected_visit_share": None,
                    "selected_minus_reference_q_margin": None,
                }
            child_stats = {
                int(child["move"]): child
                for child in list(probe.get("child_stats") or [])
            }
            visits = list(probe.get("visits") or [])
            total_visits = sum(
                float(visits[move]) for move in row["legal_moves"] if move < len(visits)
            )
            selected_move = probe.get("selected_move")
            reference_share = None
            selected_share = None
            if total_visits > 0.0 and reference_move < len(visits):
                reference_share = round(float(visits[reference_move]) / total_visits, 4)
            if (
                total_visits > 0.0
                and selected_move is not None
                and int(selected_move) < len(visits)
            ):
                selected_share = round(
                    float(visits[int(selected_move)]) / total_visits, 4
                )
            reference_q = None
            selected_q = None
            if reference_move in child_stats:
                reference_q = float(child_stats[reference_move].get("q_value", 0.0))
            if selected_move is not None and int(selected_move) in child_stats:
                selected_q = float(child_stats[int(selected_move)].get("q_value", 0.0))
            q_margin = None
            if reference_q is not None and selected_q is not None:
                q_margin = round(selected_q - reference_q, 4)
            return {
                "selected_move": None if selected_move is None else int(selected_move),
                "selected_is_reference": selected_move == reference_move,
                "reference_visit_share": reference_share,
                "selected_visit_share": selected_share,
                "selected_minus_reference_q_margin": q_margin,
            }

        view_384 = probe_view(probe_384)
        view_1200 = probe_view(probe_1200)
        top_move = top_policy_move(list(policy), row["legal_moves"])
        notes = []
        if top_move != reference_move:
            notes.append("policy_top_not_reference")
        if view_384["selected_is_reference"] is False:
            notes.append("puct_384_not_reference")
        if probe_1200 is not None and view_1200["selected_is_reference"] is False:
            notes.append("puct_1200_not_reference")
        classification = classify_checkpoint_drift(
            reference_move=reference_move,
            policy_top_move=top_move,
            puct_selected_384=view_384["selected_move"],
            q_margin_384=view_384["selected_minus_reference_q_margin"],
            puct_selected_1200=view_1200["selected_move"],
            q_margin_1200=view_1200["selected_minus_reference_q_margin"],
        )
        rows.append(
            {
                "trace_name": trace_name,
                "checkpoint_step": checkpoint_step,
                "row_id": row_id,
                "corrected_reference_move": reference_move,
                "policy_top_move": top_move,
                "reference_policy_probability": policy_distribution.get(reference_move),
                "reference_policy_rank": metric_rank(
                    policy_distribution, reference_move
                ),
                "policy_entropy": policy_entropy(list(policy), row["legal_moves"]),
                "puct_selected_move_384": view_384["selected_move"],
                "puct_selected_move_1200": view_1200["selected_move"],
                "selected_is_reference_384": view_384["selected_is_reference"],
                "selected_is_reference_1200": view_1200["selected_is_reference"],
                "reference_visit_share_384": view_384["reference_visit_share"],
                "reference_visit_share_1200": view_1200["reference_visit_share"],
                "selected_minus_reference_q_margin_384": view_384[
                    "selected_minus_reference_q_margin"
                ],
                "selected_minus_reference_q_margin_1200": view_1200[
                    "selected_minus_reference_q_margin"
                ],
                "value_prediction": round(float(root_value), 4),
                "drift_classification": classification,
                "notes": ",".join(notes) if notes else "pass_reference_selected",
            }
        )
    return rows


def classify_checkpoint_drift(
    *,
    reference_move: int,
    policy_top_move: int | None,
    puct_selected_384: int | None,
    q_margin_384: float | None,
    puct_selected_1200: int | None,
    q_margin_1200: float | None,
) -> str:
    if puct_selected_1200 is not None and puct_selected_1200 != reference_move:
        if q_margin_1200 is not None and q_margin_1200 > 0.0:
            return "value_q_drift"
        return "search_drift"
    if puct_selected_384 is not None and puct_selected_384 != reference_move:
        if q_margin_384 is not None and q_margin_384 > 0.0:
            return "value_q_drift"
        return "search_drift"
    if policy_top_move is not None and policy_top_move != reference_move:
        return "policy_drift_only"
    return "stable_guard"


def drift_severity(classification: str) -> int:
    if classification == "stable_guard":
        return 0
    if classification == "policy_drift_only":
        return 1
    return 2


def summarize_trace_row_outcomes(
    guard_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for row_id in GUARD_ROW_IDS:
        matching = sorted(
            [row for row in guard_rows if row["row_id"] == row_id],
            key=lambda row: int(row["checkpoint_step"]),
        )
        drift_rows = [
            row for row in matching if row["drift_classification"] != "stable_guard"
        ]
        if not drift_rows:
            summary[row_id] = {
                "classification": "stable_guard",
                "first_drift_step": None,
                "timing": None,
                "baseline_classification": "stable_guard",
                "baseline_step": 0 if matching else None,
                "has_regression": False,
                "first_regression_step": None,
                "regression_timing": None,
                "regression_classification": None,
            }
            continue
        first = min(drift_rows, key=lambda row: int(row["checkpoint_step"]))
        baseline = matching[0] if matching else None
        baseline_classification = (
            "stable_guard"
            if baseline is None
            else str(baseline["drift_classification"])
        )
        baseline_step = None if baseline is None else int(baseline["checkpoint_step"])
        baseline_severity = drift_severity(baseline_classification)
        regression_rows = [
            row
            for row in matching
            if baseline is not None
            and baseline_step is not None
            and int(row["checkpoint_step"]) > baseline_step
            and drift_severity(str(row["drift_classification"])) > baseline_severity
        ]
        first_regression = (
            None
            if not regression_rows
            else min(regression_rows, key=lambda row: int(row["checkpoint_step"]))
        )
        summary[row_id] = {
            "classification": first["drift_classification"],
            "first_drift_step": int(first["checkpoint_step"]),
            "timing": "immediate_drift"
            if int(first["checkpoint_step"]) <= 1
            else "late_drift",
            "baseline_classification": baseline_classification,
            "baseline_step": baseline_step,
            "has_regression": first_regression is not None,
            "first_regression_step": None
            if first_regression is None
            else int(first_regression["checkpoint_step"]),
            "regression_timing": None
            if first_regression is None
            else "immediate_regression"
            if int(first_regression["checkpoint_step"]) <= 1
            else "late_regression",
            "regression_classification": None
            if first_regression is None
            else str(first_regression["drift_classification"]),
        }
    return summary


def validate_selected_artifact(
    *,
    artifact_path: Path,
    reference_rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    if not artifact_path.exists():
        row = {
            "artifact_path": str(artifact_path),
            "row_count": 0,
            "guard_rows_present": False,
            "duplicate_conflicts": 0,
            "stale_reference_conflicts": 0,
            "status": "artifact_invalid",
            "notes": "selected_artifact_missing",
        }
        return row, [], {}

    rows = read_jsonl(artifact_path)
    by_row_id: dict[str, list[dict[str, Any]]] = {}
    duplicate_conflicts = 0
    stale_reference_conflicts = 0
    guard_presence = True
    notes: list[str] = []
    for row in rows:
        row_id = artifact_row_id(row)
        if row_id is None:
            continue
        by_row_id.setdefault(row_id, []).append(row)

    for row_id in GUARD_ROW_IDS:
        expected_reference_move = REFERENCE_MOVES[row_id]
        expected_canonical = (
            str(reference_rows[row_id]["canonical_state"])
            if row_id in reference_rows
            else None
        )
        matches = list(by_row_id.get(row_id, []))
        corrected_matches = [
            row
            for row in matches
            if (
                expected_canonical is None
                or str(row.get("canonical_state", "")) == expected_canonical
            )
            and int(row.get("reference_move", -1)) == expected_reference_move
        ]
        if not corrected_matches:
            guard_presence = False
            notes.append(f"missing_corrected_{row_id}")
        canonical_targets = {
            (
                str(row.get("canonical_state", "")),
                int(row.get("reference_move", -1)),
                top_policy_move(
                    [float(value) for value in list(row.get("policy") or [])],
                    [int(move) for move in list(row.get("legal_moves") or [])],
                ),
            )
            for row in matches
            if expected_canonical is None
            or str(row.get("canonical_state", "")) == expected_canonical
        }
        if len(canonical_targets) > 1:
            duplicate_conflicts += 1
        for row in matches:
            source_artifacts = [
                str(value) for value in list(row.get("source_artifacts") or [])
            ]
            if int(row.get("reference_move", -1)) != expected_reference_move or any(
                marker in source_path
                for marker in STALE_GUARD_SOURCE_MARKERS
                for source_path in source_artifacts
            ):
                stale_reference_conflicts += 1
                break

    if duplicate_conflicts > 0:
        notes.append("duplicate_canonical_guard_conflicts")
    if stale_reference_conflicts > 0:
        notes.append("stale_pre_pr31_guard_reference_detected")
    if len(rows) != EXPECTED_SELECTED_ROW_COUNT:
        notes.append(
            f"expected_row_count_{EXPECTED_SELECTED_ROW_COUNT}_got_{len(rows)}"
        )
    status = "ok"
    if not guard_presence or duplicate_conflicts > 0 or stale_reference_conflicts > 0:
        status = "artifact_invalid"
    return (
        {
            "artifact_path": str(artifact_path),
            "row_count": len(rows),
            "guard_rows_present": guard_presence,
            "duplicate_conflicts": duplicate_conflicts,
            "stale_reference_conflicts": stale_reference_conflicts,
            "status": status,
            "notes": ",".join(notes) if notes else "ok",
        },
        rows,
        by_row_id,
    )


def model_spec_from_current(
    *, current_metadata: dict[str, Any], base_config: dict[str, Any]
) -> dict[str, Any]:
    architecture = dict(current_metadata["architecture"])
    train_step = next(
        step
        for step in list(base_config.get("steps") or [])
        if step.get("name") == "train"
    )
    command = [str(token) for token in list(train_step.get("command") or [])]

    def flag_value(flag: str, default: str) -> str:
        return command[command.index(flag) + 1] if flag in command else default

    trunk_size = int(architecture.get("trunk_size", 96))
    residual_blocks = int(architecture.get("residual_block_count", 3))
    return {
        "model_type": str(architecture.get("model_type", "residual_v3")),
        "input_encoding": str(current_metadata.get("input_encoding", "kalah_v3")),
        "hidden_sizes": resolve_hidden_sizes(
            str(architecture.get("model_type", "residual_v3")),
            (trunk_size, residual_blocks),
        ),
        "value_loss": flag_value("--value-loss", "huber"),
        "value_loss_weight": float(flag_value("--value-loss-weight", "0.3")),
        "huber_delta": float(flag_value("--huber-delta", "1.0")),
        "grad_clip": float(flag_value("--grad-clip", "1.0")),
        "val_split": float(flag_value("--val-split", "0.1")),
    }


def build_trace_specs(
    *,
    selected_artifact: Path,
    guard_controls_artifact: Path,
    batch_size: int,
) -> list[TraceSpec]:
    specs = [
        TraceSpec(
            name="artifact_only_lr_1e-4",
            data_files=(selected_artifact,),
            replay_weights=(1,),
            lr=1e-4,
            checkpoints=(0, 1, 2, 4, 8),
            batch_size=batch_size,
            notes="artifact only diagnostic trace",
        ),
        TraceSpec(
            name="artifact_only_lr_1e-5",
            data_files=(selected_artifact,),
            replay_weights=(1,),
            lr=1e-5,
            checkpoints=(0, 1, 2, 4, 8),
            batch_size=batch_size,
            notes="artifact only lower learning rate trace",
        ),
    ]
    if guard_controls_artifact.exists():
        specs.append(
            TraceSpec(
                name="artifact_plus_guard_controls_lr_1e-4",
                data_files=(selected_artifact, guard_controls_artifact),
                replay_weights=(1, 2),
                lr=1e-4,
                checkpoints=(0, 1, 2, 4, 8),
                batch_size=batch_size,
                notes="selected artifact plus upweighted guard controls",
            )
        )
    for suffix, path in DEFAULT_MICROTRACE_ARTIFACTS.items():
        if not path.exists():
            continue
        specs.append(
            TraceSpec(
                name=f"family_leave_one_out_microtrace_{suffix}",
                data_files=(path,),
                replay_weights=(1,),
                lr=1e-4,
                checkpoints=(0, 1, 2),
                batch_size=batch_size,
                notes="two-epoch leave-one-out microtrace",
            )
        )
    return specs


def build_row_infos(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    row_infos: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            row_infos.append(
                {
                    "row_id": artifact_row_id(row),
                    "path": str(path),
                }
            )
    return row_infos


def run_trace(
    *,
    spec: TraceSpec,
    reference_rows: dict[str, dict[str, Any]],
    artifact_rows_by_id: dict[str, list[dict[str, Any]]],
    init_checkpoint: Path,
    current_metadata: dict[str, Any],
    model_spec: dict[str, Any],
    output_root: Path,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    compact_x, compact_p, compact_v, replay_indexes = load_jsonl_replay(
        list(spec.data_files),
        list(spec.replay_weights),
        policy_target_mode=DEFAULT_POLICY_TARGET_MODE,
        value_target_mode=DEFAULT_VALUE_TARGET_MODE,
    )
    row_infos = build_row_infos(spec.data_files)
    compact_v = compact_v.astype(np.float32)
    model = PolicyValueNet(
        hidden_sizes=tuple(model_spec["hidden_sizes"]),
        model_type=str(model_spec["model_type"]),
        input_size=input_size_for_encoding(str(model_spec["input_encoding"])),
    )
    load_checkpoint_into_model(model, init_checkpoint)
    model = model.to(device)
    initial_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.lr)
    trace_root = output_root / spec.name
    checkpoints_root = trace_root / "checkpoints"
    exports_root = output_root / "exports" / spec.name
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    exports_root.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    set_seed(seed)
    train_positions, _val_positions = split_replay_positions_by_source_row(
        replay_indexes, val_split=float(model_spec["val_split"])
    )
    weighted_train_indexes = replay_indexes[train_positions]
    checkpoint_steps = set(spec.checkpoints)
    max_step = max(spec.checkpoints)
    guard_drift_rows: list[dict[str, Any]] = []
    training_metric_rows: list[dict[str, Any]] = []

    def snapshot(step: int, gradient_norm: float | None, notes: str) -> None:
        checkpoint_path = checkpoints_root / f"step_{step}.npz"
        np.savez(checkpoint_path, **checkpoint_from_model(model))
        export_dir = export_checkpoint_artifact(
            checkpoint_path=checkpoint_path,
            export_dir=exports_root / f"step_{step}",
            current_metadata=current_metadata,
            version=f"{spec.name}-step-{step}",
        )
        metrics = evaluate_dataset_metrics(
            model=model,
            compact_x=compact_x,
            compact_p=compact_p,
            compact_v=compact_v,
            row_infos=row_infos,
            device=device,
            value_loss_weight=float(model_spec["value_loss_weight"]),
            value_loss=str(model_spec["value_loss"]),
            huber_delta=float(model_spec["huber_delta"]),
            initial_state=initial_state,
        )
        training_metric_rows.append(
            {
                "trace_name": spec.name,
                "checkpoint_step": step,
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "total_loss": metrics["total_loss"],
                "guard_cross_entropy": metrics["guard_cross_entropy"],
                "non_guard_opening_cross_entropy": metrics[
                    "non_guard_opening_cross_entropy"
                ],
                "policy_head_delta_norm": metrics["policy_head_delta_norm"],
                "value_head_delta_norm": metrics["value_head_delta_norm"],
                "trunk_delta_norm": metrics["trunk_delta_norm"],
                "gradient_norm": gradient_norm,
                "notes": notes,
            }
        )
        run_1200 = step in {0, max_step, 1, 2, 4, 8}
        guard_drift_rows.extend(
            evaluate_guard_rows(
                artifact_path=export_dir,
                reference_rows=reference_rows,
                artifact_rows_by_id=artifact_rows_by_id,
                checkpoint_step=step,
                trace_name=spec.name,
                run_1200=run_1200,
                seed=seed + step,
            )
        )

    snapshot(0, None, "initial_checkpoint")
    for step in range(1, max_step + 1):
        epoch_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            compact_x=compact_x,
            compact_p=compact_p,
            compact_v=compact_v,
            replay_indexes=weighted_train_indexes,
            batch_size=spec.batch_size,
            device=device,
            value_loss_weight=float(model_spec["value_loss_weight"]),
            value_loss=str(model_spec["value_loss"]),
            huber_delta=float(model_spec["huber_delta"]),
            grad_clip=float(model_spec["grad_clip"]),
        )
        if step in checkpoint_steps:
            snapshot(step, epoch_metrics["gradient_norm"], "post_epoch")

    row_outcomes = summarize_trace_row_outcomes(guard_drift_rows)
    return {
        "trace_name": spec.name,
        "data_files": [str(path) for path in spec.data_files],
        "replay_weights": list(spec.replay_weights),
        "lr": spec.lr,
        "epochs": list(spec.checkpoints),
        "batch_size": spec.batch_size,
        "init_checkpoint": str(init_checkpoint),
        "status": "completed",
        "notes": spec.notes,
        "row_outcomes": row_outcomes,
        "guard_drift_rows": guard_drift_rows,
        "training_metric_rows": training_metric_rows,
    }


def classify_overall_result(summary: dict[str, Any]) -> tuple[str, str, str]:
    validation = summary["artifact_validation"]
    if validation["status"] != "ok":
        return (
            "artifact_invalid",
            "rebuild the selected artifact with corrected 002/003/006/007/008 guard rows only, then rerun this exact low-epoch drift trace.",
            "Selected artifact failed pre-training validation, so optimization drift was not measured.",
        )

    traces = {
        trace["trace_name"]: trace
        for trace in summary["traces"]
        if trace["status"] == "completed"
    }

    def any_immediate_regression(trace_name: str) -> bool:
        trace = traces.get(trace_name)
        if trace is None:
            return False
        return any(
            row.get("regression_timing") == "immediate_regression"
            for row in trace.get("row_outcomes", {}).values()
        )

    def any_regression(trace_name: str) -> bool:
        trace = traces.get(trace_name)
        if trace is None:
            return False
        return any(
            bool(row.get("has_regression"))
            for row in trace.get("row_outcomes", {}).values()
        )

    def has_inherited_policy_only_drift(trace_name: str) -> bool:
        trace = traces.get(trace_name)
        if trace is None:
            return False
        return any(
            row.get("baseline_classification") == "policy_drift_only"
            for row in trace.get("row_outcomes", {}).values()
        )

    if any_immediate_regression("artifact_only_lr_1e-5"):
        return (
            "artifact_optimization_incompatible",
            "abandon opening replay training; mine different corrected non-opening failure families.",
            "Immediate post-init guard regression persisted even at lr 1e-5.",
        )
    if any_regression("artifact_only_lr_1e-4") and not any_regression(
        "artifact_only_lr_1e-5"
    ):
        return (
            "update_size_sensitive",
            "run one tiny controlled training lane with lower LR, guard kill gate before arena.",
            "Post-init guard regression appeared at lr 1e-4 but not at lr 1e-5.",
        )
    if any_regression("artifact_only_lr_1e-4") and not any_regression(
        "artifact_plus_guard_controls_lr_1e-4"
    ):
        return (
            "guard_anchor_needed",
            "run one tiny controlled training lane with selected artifact weight 1 and guard controls weight 2, with pre-arena corrected guard kill gate.",
            "Upweighted guard controls suppressed post-init regression seen in the artifact-only trace.",
        )

    microtrace_traces = [
        trace
        for name, trace in traces.items()
        if name.startswith("family_leave_one_out_microtrace_")
    ]
    stable_microtraces = [
        trace for trace in microtrace_traces if not any_regression(trace["trace_name"])
    ]
    drifting_microtraces = [
        trace for trace in microtrace_traces if any_regression(trace["trace_name"])
    ]
    if len(drifting_microtraces) == 1 and stable_microtraces:
        trace_name = drifting_microtraces[0]["trace_name"]
        suffix = trace_name.removeprefix("family_leave_one_out_microtrace_")
        return (
            "subfamily_optimization_poisoning",
            f"train only the safe subset excluding `{suffix}`, weight 1, with guard kill gate.",
            "Only one leave-one-out microtrace still drifted.",
        )

    loss_improves_with_drift = False
    for trace in traces.values():
        metrics = list(trace.get("training_metric_rows", []))
        if len(metrics) < 2:
            continue
        initial_total = metrics[0]["total_loss"]
        final_total = metrics[-1]["total_loss"]
        if final_total < initial_total and any_regression(trace["trace_name"]):
            loss_improves_with_drift = True
            break
    if loss_improves_with_drift:
        return (
            "objective_misalignment",
            "add explicit guard-preservation regularization or sampling constraint before any production training.",
            "Optimization loss improved while corrected guard behavior drifted.",
        )

    if not any(any_regression(trace_name) for trace_name in traces):
        if has_inherited_policy_only_drift("artifact_only_lr_1e-5"):
            return (
                "compatible_with_inherited_policy_drift",
                "use the selected artifact lane only with the existing corrected guard search gate; training did not worsen inherited guard policy mismatch.",
                "No post-init guard regression was observed; remaining policy-only mismatch was inherited from the initializer.",
            )
        return (
            "full_pipeline_interaction",
            "reproduce the prior regression with a low-cost pipeline trace including self-play/background replay mix, not artifact-only training.",
            "No post-init guard regression was reproduced by the diagnostic low-epoch trace.",
        )

    return (
        "search_drift_detected",
        "rerun only the lowest-cost trace that first reproduces drift and instrument per-update policy/value deltas more finely.",
        "At least one diagnostic trace drifted, but the stronger decision rules did not isolate a narrower cause.",
    )


def render_report(summary: dict[str, Any]) -> str:
    validation = summary["artifact_validation"]
    lines = [
        "# AlphaZero-lite Guard-Safe Opening Low-Epoch Drift Trace Results (Regression-Aware Compatibility)",
        "",
        "Compatibility basis: post-init regression relative to the initializer.",
        "",
        "## 1. Context",
        "",
        "- PR #34 selected a statically guard-safe opening replay artifact for a low-epoch optimization drift trace.",
        "- This run stayed diagnostic-only: no production training, no arena, no promotion, no artifact overwrite.",
        "- Compatibility here means training did not worsen corrected guard behavior relative to the initializer, even if the initializer already had raw-policy mismatch.",
        f"- Corrected references: `{summary['reference_artifact']}`.",
        f"- Current initialization artifact: `{summary['current_path']}`.",
        "",
        "## 2. Selected artifact validation",
        "",
        "| artifact_path | row_count | guard_rows_present | duplicate_conflicts | stale_reference_conflicts | status | notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| `{validation['artifact_path']}` | {validation['row_count']} | {str(bool(validation['guard_rows_present'])).lower()} | {validation['duplicate_conflicts']} | {validation['stale_reference_conflicts']} | `{validation['status']}` | {validation['notes']} |",
        "",
        "## 3. Trace variants",
        "",
        "| trace_name | data_files | replay_weights | lr | epochs | batch_size | init_checkpoint | status | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for trace in summary["traces"]:
        lines.append(
            f"| {trace['trace_name']} | `{json.dumps(trace['data_files'])}` | `{json.dumps(trace['replay_weights'])}` | {trace['lr']} | `{json.dumps(trace['epochs'])}` | {trace['batch_size']} | `{trace['init_checkpoint']}` | `{trace['status']}` | {trace['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 4. Guard policy drift results",
            "",
            "| trace_name | checkpoint_step | row_id | corrected_reference_move | policy_top_move | reference_policy_probability | reference_policy_rank | policy_entropy | puct_selected_move_384 | puct_selected_move_1200 | selected_is_reference_384 | selected_is_reference_1200 | reference_visit_share_384 | reference_visit_share_1200 | selected_minus_reference_q_margin_384 | selected_minus_reference_q_margin_1200 | drift_classification | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["guard_drift_rows"]:
        lines.append(
            f"| {row['trace_name']} | {row['checkpoint_step']} | {row['row_id']} | {row['corrected_reference_move']} | {row['policy_top_move']} | {format_float(row['reference_policy_probability'])} | {row['reference_policy_rank'] if row['reference_policy_rank'] is not None else '-'} | {format_float(row['policy_entropy'])} | {row['puct_selected_move_384'] if row['puct_selected_move_384'] is not None else '-'} | {row['puct_selected_move_1200'] if row['puct_selected_move_1200'] is not None else '-'} | {format_bool(row['selected_is_reference_384'])} | {format_bool(row['selected_is_reference_1200'])} | {format_float(row['reference_visit_share_384'])} | {format_float(row['reference_visit_share_1200'])} | {format_float(row['selected_minus_reference_q_margin_384'])} | {format_float(row['selected_minus_reference_q_margin_1200'])} | `{row['drift_classification']}` | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 5. Guard search drift results",
            "",
            f"- Evaluated rows: `{json.dumps(list(GUARD_ROW_IDS))}`.",
            "- Search settings matched the prior corrected-reference control baseline via `build_eval_search_options()` with no new root-prior intervention.",
            "",
            "## 6. Training metric trace",
            "",
            "| trace_name | checkpoint_step | policy_loss | value_loss | total_loss | guard_cross_entropy | non_guard_opening_cross_entropy | policy_head_delta_norm | value_head_delta_norm | trunk_delta_norm | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["training_metric_rows"]:
        lines.append(
            f"| {row['trace_name']} | {row['checkpoint_step']} | {format_float(row['policy_loss'])} | {format_float(row['value_loss'])} | {format_float(row['total_loss'])} | {format_float(row['guard_cross_entropy'])} | {format_float(row['non_guard_opening_cross_entropy'])} | {format_float(row['policy_head_delta_norm'])} | {format_float(row['value_head_delta_norm'])} | {format_float(row['trunk_delta_norm'])} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 7. Family/Subset Sensitivity",
            "",
            "| trace_name | row_outcomes | status | notes |",
            "| --- | --- | --- | --- |",
        ]
    )
    for trace in summary["traces"]:
        if not trace["trace_name"].startswith("family_leave_one_out_microtrace_"):
            continue
        lines.append(
            f"| {trace['trace_name']} | `{json.dumps(trace.get('row_outcomes', {}), sort_keys=True)}` | `{trace['status']}` | {trace['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 8. Interpretation",
            "",
            f"- Classification: `{summary['classification']}`.",
            f"- Interpretation: {summary['classification_notes']}",
            "- Primary question is answered only if artifact validation passed and drift rows were collected.",
            "",
            "## 9. Exactly one recommended next action",
            "",
            f"Recommendation: **{summary['recommended_next_action']}**",
        ]
    )
    return "\n".join(lines) + "\n"


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def format_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return str(bool(value)).lower()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    reference_artifact = resolve_path(root, args.reference_artifact)
    fallback_reference_artifact = resolve_path(root, args.fallback_reference_artifact)
    current_path = resolve_path(root, args.current_path)
    selected_artifact = resolve_path(root, args.selected_artifact)
    guard_controls_artifact = resolve_path(root, args.guard_controls_artifact)
    base_config_path = resolve_path(root, args.base_config)
    output_root = resolve_path(root, args.output_root)
    summary_path = resolve_path(root, args.summary_path)
    report_path = resolve_path(root, args.report_path)
    output_root.mkdir(parents=True, exist_ok=True)
    reference_artifact_display = display_path(root, reference_artifact)
    fallback_reference_artifact_display = display_path(
        root, fallback_reference_artifact
    )
    current_path_display = display_path(root, current_path)
    selected_artifact_display = display_path(root, selected_artifact)

    reference_rows = row_map_from_reference(load_json(reference_artifact))
    fallback_reference_rows = row_map_from_reference(
        load_json(fallback_reference_artifact)
    )
    merged_reference_rows = {**fallback_reference_rows, **reference_rows}
    validation_row, _selected_rows, artifact_rows_by_id = validate_selected_artifact(
        artifact_path=selected_artifact,
        reference_rows=merged_reference_rows,
    )

    planned_specs = build_trace_specs(
        selected_artifact=selected_artifact,
        guard_controls_artifact=guard_controls_artifact,
        batch_size=DEFAULT_BATCH_SIZE,
    )
    traces: list[dict[str, Any]] = []
    guard_drift_rows: list[dict[str, Any]] = []
    training_metric_rows: list[dict[str, Any]] = []

    if validation_row["status"] != "ok":
        for spec in planned_specs:
            traces.append(
                {
                    "trace_name": spec.name,
                    "data_files": [str(path) for path in spec.data_files],
                    "replay_weights": list(spec.replay_weights),
                    "lr": spec.lr,
                    "epochs": list(spec.checkpoints),
                    "batch_size": spec.batch_size,
                    "init_checkpoint": str(current_path),
                    "status": "skipped_artifact_invalid",
                    "notes": "artifact validation failed before training",
                }
            )
    else:
        current_metadata = load_json(current_path / "metadata.json")
        base_config = load_json(base_config_path)
        model_spec = model_spec_from_current(
            current_metadata=current_metadata,
            base_config=base_config,
        )
        init_checkpoint = current_checkpoint_path(current_path, output_root)
        device = choose_device(args.device)
        for spec in planned_specs:
            trace = run_trace(
                spec=spec,
                reference_rows=merged_reference_rows,
                artifact_rows_by_id=artifact_rows_by_id,
                init_checkpoint=init_checkpoint,
                current_metadata=current_metadata,
                model_spec=model_spec,
                output_root=output_root,
                seed=int(args.seed),
                device=device,
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

    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "compatibility_basis": COMPATIBILITY_BASIS,
        "reference_artifact": reference_artifact_display,
        "fallback_reference_artifact": fallback_reference_artifact_display,
        "current_path": current_path_display,
        "selected_artifact": selected_artifact_display,
        "artifact_validation": validation_row,
        "traces": traces,
        "guard_drift_rows": guard_drift_rows,
        "training_metric_rows": training_metric_rows,
    }
    classification, recommended_next_action, classification_notes = (
        classify_overall_result(summary)
    )
    summary["classification"] = classification
    summary["classification_notes"] = classification_notes
    summary["recommended_next_action"] = recommended_next_action
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
