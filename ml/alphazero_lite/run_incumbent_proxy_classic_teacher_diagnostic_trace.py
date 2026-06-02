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
from ml.alphazero_lite.build_incumbent_proxy_classic_teacher_diagnostic_artifact import (
    build_artifact as build_classic_artifact,
)
from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint
from ml.alphazero_lite.run_incumbent_proxy_disagreement_value_backup_audit import (
    load_json,
    write_json,
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
    set_seed,
    split_replay_positions_by_source_row,
    train_one_epoch as shared_train_one_epoch,
)


DEFAULT_REFERENCE_PATH = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
DEFAULT_CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
DEFAULT_CLASSIC_ROWS_PATH = Path(
    "/tmp/azlite_incumbent_proxy_teacher_bucket_split/classic_teacher_rows.jsonl"
)
DEFAULT_PUCT_ROWS_PATH = Path(
    "/tmp/azlite_incumbent_proxy_teacher_bucket_split/puct_teacher_rows.jsonl"
)
DEFAULT_EXCLUDED_ROWS_PATH = Path(
    "/tmp/azlite_incumbent_proxy_teacher_bucket_split/excluded_diagnostic_rows.jsonl"
)
DEFAULT_BUCKET_SUMMARY_PATH = Path(
    "/tmp/azlite_incumbent_proxy_teacher_bucket_split/teacher_bucket_split_summary.json"
)
DEFAULT_BUCKET_REPORT_PATH = Path(
    "docs/alphazero-lite-incumbent-proxy-disagreement-teacher-bucket-split-results.md"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/azlite_incumbent_proxy_classic_teacher_diagnostic")
DEFAULT_ARTIFACT_PATH = DEFAULT_OUTPUT_DIR / "classic_teacher_diagnostic_artifact.jsonl"
DEFAULT_ARTIFACT_SUMMARY_PATH = (
    DEFAULT_OUTPUT_DIR / "classic_teacher_diagnostic_artifact_summary.json"
)
DEFAULT_TARGET_ONLY_PATH = (
    DEFAULT_OUTPUT_DIR / "classic_teacher_target_candidates.jsonl"
)
DEFAULT_CONTROL_ONLY_PATH = (
    DEFAULT_OUTPUT_DIR / "classic_teacher_preservation_controls.jsonl"
)
DEFAULT_TRACE_SUMMARY_PATH = (
    DEFAULT_OUTPUT_DIR / "classic_teacher_diagnostic_trace_summary.json"
)
DEFAULT_EXPORT_ROOT = DEFAULT_OUTPUT_DIR / "exports"
DEFAULT_REPORT_PATH = Path(
    "docs/alphazero-lite-incumbent-proxy-classic-teacher-diagnostic-results.md"
)

SCHEMA = "azlite_incumbent_proxy_classic_teacher_diagnostic_trace_v1"
CLASSIC_ROW_IDS = (
    "incumbent_proxy_disagreement-008",
    "incumbent_proxy_disagreement-014",
    "incumbent_proxy_disagreement-022",
    "incumbent_proxy_disagreement-024",
    "incumbent_proxy_disagreement-025",
    "incumbent_proxy_disagreement-035",
)
PUCT_ROW_IDS = (
    "incumbent_proxy_disagreement-007",
    "incumbent_proxy_disagreement-009",
    "incumbent_proxy_disagreement-021",
    "incumbent_proxy_disagreement-026",
    "incumbent_proxy_disagreement-028",
    "incumbent_proxy_disagreement-032",
    "incumbent_proxy_disagreement-033",
)
EXCLUDED_ROW_IDS = (
    "incumbent_proxy_disagreement-003",
    "incumbent_proxy_disagreement-010",
    "incumbent_proxy_disagreement-012",
    "incumbent_proxy_disagreement-018",
    "incumbent_proxy_disagreement-020",
    "incumbent_proxy_disagreement-023",
    "incumbent_proxy_disagreement-027",
    "incumbent_proxy_disagreement-029",
)
CLASSIC_CONTROL_ROW_ID = "incumbent_proxy_disagreement-008"
TRACE_EPOCHS = (1, 2, 4)
SEARCH_BUDGETS = (384, 1200)
DEFAULT_SEED = 42
DEFAULT_C_PUCT = 1.25
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_VALUE_LOSS = "huber"
DEFAULT_VALUE_LOSS_WEIGHT = 0.3
DEFAULT_HUBER_DELTA = 1.0
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_GRAD_CLIP = 1.0
EVAL_SEARCH_OPTIONS = dict(
    build_eval_search_options(
        root_policy_mode="deterministic",
        tactical_root_bias=0.1,
        fpu_mode="parent_q",
        reuse_subtree=True,
        normalize_values=True,
    )
)


@dataclass(frozen=True)
class TraceSpec:
    name: str
    data_files: tuple[Path, ...]
    replay_weights: tuple[int, ...]
    epochs: tuple[int, ...]
    notes: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--suite-path", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument(
        "--current-artifact", type=Path, default=DEFAULT_CURRENT_ARTIFACT
    )
    parser.add_argument(
        "--classic-rows-path", type=Path, default=DEFAULT_CLASSIC_ROWS_PATH
    )
    parser.add_argument("--puct-rows-path", type=Path, default=DEFAULT_PUCT_ROWS_PATH)
    parser.add_argument(
        "--excluded-rows-path", type=Path, default=DEFAULT_EXCLUDED_ROWS_PATH
    )
    parser.add_argument(
        "--bucket-summary-path", type=Path, default=DEFAULT_BUCKET_SUMMARY_PATH
    )
    parser.add_argument(
        "--bucket-report-path", type=Path, default=DEFAULT_BUCKET_REPORT_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument(
        "--artifact-summary-path", type=Path, default=DEFAULT_ARTIFACT_SUMMARY_PATH
    )
    parser.add_argument(
        "--target-only-path", type=Path, default=DEFAULT_TARGET_ONLY_PATH
    )
    parser.add_argument(
        "--control-only-path", type=Path, default=DEFAULT_CONTROL_ONLY_PATH
    )
    parser.add_argument(
        "--trace-summary-path", type=Path, default=DEFAULT_TRACE_SUMMARY_PATH
    )
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def format_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return str(bool(value)).lower()


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(row) + " |")
    return output


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(requested)


def current_checkpoint_path(current_path: Path, output_root: Path) -> Path:
    for name in ("checkpoint.npz", "model.npz"):
        candidate = current_path / name
        if candidate.exists():
            return candidate
    weights_path = current_path / "weights.json"
    if weights_path.exists():
        return materialize_weights_json_checkpoint(
            weights_path=weights_path,
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
    with np.load(model_path) as checkpoint:
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


def top_policy_move(policy: list[float], legal_moves: list[int]) -> int | None:
    if not legal_moves:
        return None
    return min(legal_moves, key=lambda move: (-float(policy[move]), int(move)))


def metric_rank(distribution: dict[int, float], move: int) -> int | None:
    if move not in distribution:
        return None
    ranked = sorted(distribution, key=lambda key: (-distribution[key], key))
    for index, candidate in enumerate(ranked, start=1):
        if candidate == move:
            return index
    return None


def policy_entropy(policy: list[float], legal_moves: list[int]) -> float:
    entropy = 0.0
    for move in legal_moves:
        probability = float(policy[move])
        if probability > 0.0:
            entropy -= probability * math.log(probability, 2)
    return round(entropy, 4)


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


def artifact_row_id(row: dict[str, Any]) -> str | None:
    source_runs = list(row.get("source_runs") or [])
    if source_runs:
        row_id = source_runs[0].get("id")
        if isinstance(row_id, str) and row_id:
            return row_id
    row_id = row.get("row_id")
    if isinstance(row_id, str) and row_id:
        return row_id
    return None


def build_row_infos(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    row_infos: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            row_infos.append(
                {
                    "row_id": artifact_row_id(row),
                    "role": row.get("role"),
                    "path": str(path),
                }
            )
    return row_infos


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
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    norms = delta_norms(state_dict, initial_state)

    def indexes_for_role(role: str | None) -> list[int]:
        return [
            index
            for index, row_info in enumerate(row_infos)
            if row_info.get("role") == role
        ]

    def mean_or_none(indexes: list[int], values: np.ndarray) -> float | None:
        if not indexes:
            return None
        return round(float(np.mean(values[indexes])), 6)

    return {
        "policy_loss": round(float(np.mean(policy_ce)), 6),
        "value_loss": round(float(np.mean(value_loss_vector)), 6),
        "total_loss": round(float(np.mean(total_loss)), 6),
        "artifact_cross_entropy": round(float(np.mean(policy_ce)), 6),
        "target_candidate_cross_entropy": mean_or_none(
            indexes_for_role("target_candidate"), policy_ce
        ),
        "preservation_control_cross_entropy": mean_or_none(
            indexes_for_role("preservation_control"), policy_ce
        ),
        "policy_head_delta_norm": norms["policy"],
        "value_head_delta_norm": norms["value"],
        "trunk_delta_norm": norms["trunk"],
    }


def policy_distribution(
    policy: list[float], legal_moves: list[int]
) -> dict[int, float]:
    return {
        int(move): round(float(policy[move]), 4)
        for move in legal_moves
        if int(move) < len(policy)
    }


def evaluate_position_summary(
    *,
    artifact_path: Path,
    state: dict[str, Any],
    legal_moves: list[int],
    seed: int,
    evaluator_cache: dict[Path, ArtifactEvaluator],
) -> dict[str, Any]:
    cache_key = artifact_path.resolve()
    evaluator = evaluator_cache.get(cache_key)
    if evaluator is None:
        evaluator = ArtifactEvaluator(artifact_path)
        evaluator_cache[cache_key] = evaluator

    def run_budget(budget: int) -> dict[str, Any]:
        probe = evaluate_artifact_position(
            artifact_path=artifact_path,
            evaluator=evaluator,
            state=state,
            simulations=int(budget),
            seed=int(seed + budget),
            c_puct=DEFAULT_C_PUCT,
            search_options=EVAL_SEARCH_OPTIONS,
            ablation_mode="full",
        )
        child_stats = {
            int(child["move"]): child for child in list(probe.get("child_stats") or [])
        }
        selected_move = probe.get("selected_move")
        visits = [float(value) for value in list(probe.get("visits") or [])]
        total_visits = sum(
            float(visits[move]) for move in legal_moves if move < len(visits)
        )
        return {
            "selected_move": None if selected_move is None else int(selected_move),
            "visit_shares": {
                int(move): round(float(visits[move]) / total_visits, 4)
                if total_visits > 0.0 and move < len(visits)
                else None
                for move in legal_moves
            },
            "q_values": {
                int(move): round(float(child_stats[move].get("q_value", 0.0)), 4)
                for move in legal_moves
                if move in child_stats
            },
            "policy": [float(value) for value in list(probe.get("policy") or [])],
            "value": round(float(probe.get("value", 0.0)), 4),
        }

    result_384 = run_budget(384)
    result_1200 = run_budget(1200)
    return {
        384: result_384,
        1200: result_1200,
        "policy_distribution": policy_distribution(result_1200["policy"], legal_moves),
        "policy_top_move": top_policy_move(result_1200["policy"], legal_moves),
        "policy_entropy": policy_entropy(result_1200["policy"], legal_moves),
        "value_prediction": result_1200["value"],
    }


def selected_minus_reference_q_margin(
    budget_result: dict[str, Any], reference_move: int
) -> float | None:
    q_values = dict(budget_result.get("q_values") or {})
    selected_move = budget_result.get("selected_move")
    if (
        selected_move is None
        or reference_move not in q_values
        or selected_move not in q_values
    ):
        return None
    return round(float(q_values[selected_move]) - float(q_values[reference_move]), 4)


def strict_reference_pass(row: dict[str, Any]) -> bool:
    return bool(row["selected_is_reference_384"]) and bool(
        row["selected_is_reference_1200"]
    )


def classic_improved(
    *, baseline_row: dict[str, Any], current_row: dict[str, Any]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if (
        not baseline_row["selected_is_reference_384"]
        and current_row["selected_is_reference_384"]
    ):
        reasons.append("selection_improved_384")
    if (
        not baseline_row["selected_is_reference_1200"]
        and current_row["selected_is_reference_1200"]
    ):
        reasons.append("selection_improved_1200")
    for budget in SEARCH_BUDGETS:
        baseline_share = baseline_row.get(f"reference_visit_share_{budget}")
        current_share = current_row.get(f"reference_visit_share_{budget}")
        if (
            baseline_share is not None
            and current_share is not None
            and current_share > baseline_share + 1e-6
        ):
            reasons.append(f"reference_share_up_{budget}")
    return (bool(reasons), reasons)


def puct_regression(
    *, baseline_row: dict[str, Any], current_row: dict[str, Any]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if bool(baseline_row["selected_equals_puct_preferred_1200"]) and not bool(
        current_row["selected_equals_puct_preferred_1200"]
    ):
        reasons.append("lost_puct_selection_1200")
    if bool(baseline_row["selected_equals_puct_preferred_384"]) and not bool(
        current_row["selected_equals_puct_preferred_384"]
    ):
        reasons.append("lost_puct_selection_384")
    for budget in SEARCH_BUDGETS:
        baseline_share = baseline_row.get(f"puct_preferred_visit_share_{budget}")
        current_share = current_row.get(f"puct_preferred_visit_share_{budget}")
        if (
            baseline_share is not None
            and current_share is not None
            and (current_share - baseline_share) < -0.2
        ):
            reasons.append(f"large_preferred_share_drop_{budget}")
    return (bool(reasons), reasons)


def unexpected_excluded_drift(
    *, baseline_row: dict[str, Any], current_row: dict[str, Any]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    entropy_delta = abs(
        float(current_row.get("policy_entropy") or 0.0)
        - float(baseline_row.get("policy_entropy") or 0.0)
    )
    if (
        current_row["selected_move_1200"] != baseline_row["selected_move_1200"]
        and entropy_delta >= 0.5
    ):
        reasons.append("selection_changed_with_entropy_shift")
    if entropy_delta >= 1.0:
        reasons.append("large_entropy_shift")
    return (bool(reasons), reasons)


def build_classic_row_result(
    *,
    trace_name: str,
    epoch: int,
    row_spec: dict[str, Any],
    eval_summary: dict[str, Any],
    baseline_row: dict[str, Any],
) -> dict[str, Any]:
    reference_move = int(row_spec["active_reference_move"])
    policy_top = eval_summary["policy_top_move"]
    distribution = eval_summary["policy_distribution"]
    current_row = {
        "selected_is_reference_384": eval_summary[384]["selected_move"]
        == reference_move,
        "selected_is_reference_1200": eval_summary[1200]["selected_move"]
        == reference_move,
        "reference_visit_share_384": eval_summary[384]["visit_shares"].get(
            reference_move
        ),
        "reference_visit_share_1200": eval_summary[1200]["visit_shares"].get(
            reference_move
        ),
    }
    improved, improvement_notes = classic_improved(
        baseline_row=baseline_row, current_row={**baseline_row, **current_row}
    )
    strict_pass = (
        current_row["selected_is_reference_384"]
        and current_row["selected_is_reference_1200"]
    )
    notes = list(improvement_notes)
    if baseline_row.get("strict_pass") and not strict_pass:
        notes.append("regressed_from_prior_pass")
    return {
        "trace_name": trace_name,
        "epoch": epoch,
        "row_id": row_spec["row_id"],
        "role": row_spec["role"],
        "active_reference_move": reference_move,
        "selected_move_384": eval_summary[384]["selected_move"],
        "selected_move_1200": eval_summary[1200]["selected_move"],
        "selected_is_reference_384": current_row["selected_is_reference_384"],
        "selected_is_reference_1200": current_row["selected_is_reference_1200"],
        "reference_visit_share_384": current_row["reference_visit_share_384"],
        "reference_visit_share_1200": current_row["reference_visit_share_1200"],
        "selected_minus_reference_q_margin_384": selected_minus_reference_q_margin(
            eval_summary[384], reference_move
        ),
        "selected_minus_reference_q_margin_1200": selected_minus_reference_q_margin(
            eval_summary[1200], reference_move
        ),
        "reference_policy_probability": distribution.get(reference_move),
        "reference_policy_rank": metric_rank(distribution, reference_move),
        "policy_top_move": policy_top,
        "policy_entropy": eval_summary["policy_entropy"],
        "improved_vs_current": improved,
        "strict_pass": strict_pass,
        "notes": ",".join(notes) if notes else "no_change_vs_current",
    }


def build_puct_row_result(
    *,
    trace_name: str,
    epoch: int,
    row_spec: dict[str, Any],
    eval_summary: dict[str, Any],
    baseline_row: dict[str, Any],
) -> dict[str, Any]:
    puct_move = int(row_spec["preferred_move"])
    current_row = {
        "selected_equals_puct_preferred_384": eval_summary[384]["selected_move"]
        == puct_move,
        "selected_equals_puct_preferred_1200": eval_summary[1200]["selected_move"]
        == puct_move,
        "puct_preferred_visit_share_384": eval_summary[384]["visit_shares"].get(
            puct_move
        ),
        "puct_preferred_visit_share_1200": eval_summary[1200]["visit_shares"].get(
            puct_move
        ),
    }
    regressed, reasons = puct_regression(
        baseline_row=baseline_row, current_row={**baseline_row, **current_row}
    )
    return {
        "trace_name": trace_name,
        "epoch": epoch,
        "row_id": row_spec["row_id"],
        "puct_preferred_move": puct_move,
        "selected_move_384": eval_summary[384]["selected_move"],
        "selected_move_1200": eval_summary[1200]["selected_move"],
        "selected_equals_puct_preferred_384": current_row[
            "selected_equals_puct_preferred_384"
        ],
        "selected_equals_puct_preferred_1200": current_row[
            "selected_equals_puct_preferred_1200"
        ],
        "selected_equals_puct_preferred": current_row[
            "selected_equals_puct_preferred_384"
        ]
        and current_row["selected_equals_puct_preferred_1200"],
        "active_reference_move": row_spec["active_reference_move"],
        "active_reference_visit_share_384": eval_summary[384]["visit_shares"].get(
            int(row_spec["active_reference_move"])
        ),
        "active_reference_visit_share_1200": eval_summary[1200]["visit_shares"].get(
            int(row_spec["active_reference_move"])
        ),
        "puct_preferred_visit_share_384": current_row["puct_preferred_visit_share_384"],
        "puct_preferred_visit_share_1200": current_row[
            "puct_preferred_visit_share_1200"
        ],
        "cross_teacher_regression": regressed,
        "notes": ",".join(reasons) if reasons else "no_heavy_puct_regression",
    }


def build_excluded_row_result(
    *,
    trace_name: str,
    epoch: int,
    row_spec: dict[str, Any],
    eval_summary: dict[str, Any],
    baseline_row: dict[str, Any],
) -> dict[str, Any]:
    current_row = {
        "selected_move_1200": eval_summary[1200]["selected_move"],
        "policy_entropy": eval_summary["policy_entropy"],
    }
    drifted, reasons = unexpected_excluded_drift(
        baseline_row=baseline_row, current_row={**baseline_row, **current_row}
    )
    return {
        "trace_name": trace_name,
        "epoch": epoch,
        "row_id": row_spec["row_id"],
        "selected_move_1200": current_row["selected_move_1200"],
        "policy_entropy": current_row["policy_entropy"],
        "unexpected_drift": drifted,
        "notes": ",".join(reasons) if reasons else "no_large_unexpected_drift",
    }


def row_specs_from_bucket_rows(
    rows: list[dict[str, Any]], ids: tuple[str, ...]
) -> list[dict[str, Any]]:
    by_id = {str(row["row_id"]): row for row in rows if row.get("row_id")}
    missing = [row_id for row_id in ids if row_id not in by_id]
    if missing:
        raise ValueError("missing required bucket rows: " + ", ".join(sorted(missing)))
    normalized: list[dict[str, Any]] = []
    for row_id in ids:
        row = dict(by_id[row_id])
        if row.get("role") is None and row.get("recommended_role") is not None:
            row["role"] = row["recommended_role"]
        normalized.append(row)
    return normalized


def evaluate_row_set_baseline(
    *, artifact_path: Path, row_specs: list[dict[str, Any]], kind: str, seed: int
) -> list[dict[str, Any]]:
    baseline_rows: list[dict[str, Any]] = []
    evaluator_cache: dict[Path, ArtifactEvaluator] = {}
    for index, row_spec in enumerate(row_specs):
        summary = evaluate_position_summary(
            artifact_path=artifact_path,
            state=dict(row_spec["suite_state"]),
            legal_moves=[int(move) for move in row_spec["legal_moves"]],
            seed=seed + (index * 101),
            evaluator_cache=evaluator_cache,
        )
        if kind == "classic":
            reference_move = int(row_spec["active_reference_move"])
            baseline_rows.append(
                {
                    "row_id": row_spec["row_id"],
                    "selected_is_reference_384": summary[384]["selected_move"]
                    == reference_move,
                    "selected_is_reference_1200": summary[1200]["selected_move"]
                    == reference_move,
                    "reference_visit_share_384": summary[384]["visit_shares"].get(
                        reference_move
                    ),
                    "reference_visit_share_1200": summary[1200]["visit_shares"].get(
                        reference_move
                    ),
                    "strict_pass": summary[384]["selected_move"] == reference_move
                    and summary[1200]["selected_move"] == reference_move,
                }
            )
        elif kind == "puct":
            puct_move = int(row_spec["preferred_move"])
            baseline_rows.append(
                {
                    "row_id": row_spec["row_id"],
                    "selected_equals_puct_preferred_384": summary[384]["selected_move"]
                    == puct_move,
                    "selected_equals_puct_preferred_1200": summary[1200][
                        "selected_move"
                    ]
                    == puct_move,
                    "puct_preferred_visit_share_384": summary[384]["visit_shares"].get(
                        puct_move
                    ),
                    "puct_preferred_visit_share_1200": summary[1200][
                        "visit_shares"
                    ].get(puct_move),
                }
            )
        else:
            baseline_rows.append(
                {
                    "row_id": row_spec["row_id"],
                    "selected_move_1200": summary[1200]["selected_move"],
                    "policy_entropy": summary["policy_entropy"],
                }
            )
    return baseline_rows


def model_spec_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    architecture = dict(metadata["architecture"])
    return {
        "model_type": str(architecture.get("model_type", "residual_v3")),
        "hidden_sizes": (
            int(architecture.get("trunk_size", 96)),
            int(architecture.get("residual_block_count", 3)),
        ),
        "input_encoding": str(metadata.get("input_encoding", "kalah_v3")),
        "value_loss": DEFAULT_VALUE_LOSS,
        "value_loss_weight": DEFAULT_VALUE_LOSS_WEIGHT,
        "huber_delta": DEFAULT_HUBER_DELTA,
        "val_split": DEFAULT_VAL_SPLIT,
        "grad_clip": DEFAULT_GRAD_CLIP,
        "batch_size": DEFAULT_BATCH_SIZE,
        "lr": DEFAULT_LR,
    }


def run_trace(
    *,
    spec: TraceSpec,
    row_spec_by_id: dict[str, dict[str, Any]],
    puct_spec_by_id: dict[str, dict[str, Any]],
    excluded_spec_by_id: dict[str, dict[str, Any]],
    classic_baseline_by_id: dict[str, dict[str, Any]],
    puct_baseline_by_id: dict[str, dict[str, Any]],
    excluded_baseline_by_id: dict[str, dict[str, Any]],
    init_checkpoint: Path,
    current_metadata: dict[str, Any],
    model_spec: dict[str, Any],
    export_root: Path,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    compact_x, compact_p, compact_v, replay_indexes = load_jsonl_replay(
        list(spec.data_files),
        list(spec.replay_weights),
        policy_target_mode="default",
        value_target_mode="default",
    )
    row_infos = build_row_infos(spec.data_files)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=float(model_spec["lr"]))

    np.random.seed(seed)
    set_seed(seed)
    train_positions, _ = split_replay_positions_by_source_row(
        replay_indexes, val_split=float(model_spec["val_split"])
    )
    weighted_train_indexes = replay_indexes[train_positions]

    checkpoints_root = export_root / "checkpoints" / spec.name
    exports_root = export_root / spec.name
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    exports_root.mkdir(parents=True, exist_ok=True)

    classic_rows: list[dict[str, Any]] = []
    puct_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    training_metric_rows: list[dict[str, Any]] = []
    evaluator_cache: dict[Path, ArtifactEvaluator] = {}

    def snapshot(epoch: int, gradient_norm: float | None, notes: str) -> None:
        checkpoint_path = checkpoints_root / f"epoch_{epoch}.npz"
        np.savez(checkpoint_path, **checkpoint_from_model(model))
        export_dir = export_checkpoint_artifact(
            checkpoint_path=checkpoint_path,
            export_dir=exports_root / f"epoch_{epoch}",
            current_metadata=current_metadata,
            version=f"{spec.name}-epoch-{epoch}",
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
                "epoch": epoch,
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "total_loss": metrics["total_loss"],
                "artifact_cross_entropy": metrics["artifact_cross_entropy"],
                "target_candidate_cross_entropy": metrics[
                    "target_candidate_cross_entropy"
                ],
                "preservation_control_cross_entropy": metrics[
                    "preservation_control_cross_entropy"
                ],
                "policy_head_delta_norm": metrics["policy_head_delta_norm"],
                "value_head_delta_norm": metrics["value_head_delta_norm"],
                "trunk_delta_norm": metrics["trunk_delta_norm"],
                "gradient_norm": gradient_norm,
                "notes": notes,
            }
        )

        for row_id in CLASSIC_ROW_IDS:
            row_spec = row_spec_by_id[row_id]
            summary = evaluate_position_summary(
                artifact_path=export_dir,
                state=dict(row_spec["suite_state"]),
                legal_moves=[int(move) for move in row_spec["legal_moves"]],
                seed=seed + epoch + len(classic_rows),
                evaluator_cache=evaluator_cache,
            )
            classic_rows.append(
                build_classic_row_result(
                    trace_name=spec.name,
                    epoch=epoch,
                    row_spec=row_spec,
                    eval_summary=summary,
                    baseline_row=classic_baseline_by_id[row_id],
                )
            )

        for row_id in PUCT_ROW_IDS:
            row_spec = puct_spec_by_id[row_id]
            summary = evaluate_position_summary(
                artifact_path=export_dir,
                state=dict(row_spec["suite_state"]),
                legal_moves=[int(move) for move in row_spec["legal_moves"]],
                seed=seed + 500 + epoch + len(puct_rows),
                evaluator_cache=evaluator_cache,
            )
            puct_rows.append(
                build_puct_row_result(
                    trace_name=spec.name,
                    epoch=epoch,
                    row_spec=row_spec,
                    eval_summary=summary,
                    baseline_row=puct_baseline_by_id[row_id],
                )
            )

        for row_id in EXCLUDED_ROW_IDS:
            row_spec = excluded_spec_by_id[row_id]
            summary = evaluate_position_summary(
                artifact_path=export_dir,
                state=dict(row_spec["suite_state"]),
                legal_moves=[int(move) for move in row_spec["legal_moves"]],
                seed=seed + 900 + epoch + len(excluded_rows),
                evaluator_cache=evaluator_cache,
            )
            excluded_rows.append(
                build_excluded_row_result(
                    trace_name=spec.name,
                    epoch=epoch,
                    row_spec=row_spec,
                    eval_summary=summary,
                    baseline_row=excluded_baseline_by_id[row_id],
                )
            )

    snapshot(0, None, "initial_checkpoint")
    for epoch in range(1, max(spec.epochs) + 1):
        epoch_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            compact_x=compact_x,
            compact_p=compact_p,
            compact_v=compact_v,
            replay_indexes=weighted_train_indexes,
            batch_size=int(model_spec["batch_size"]),
            device=device,
            value_loss_weight=float(model_spec["value_loss_weight"]),
            value_loss=str(model_spec["value_loss"]),
            huber_delta=float(model_spec["huber_delta"]),
            grad_clip=float(model_spec["grad_clip"]),
        )
        if epoch in spec.epochs:
            snapshot(epoch, epoch_metrics["gradient_norm"], "post_epoch")

    return {
        "trace_name": spec.name,
        "data_files": [str(path) for path in spec.data_files],
        "replay_weights": list(spec.replay_weights),
        "epochs": list(spec.epochs),
        "status": "completed",
        "notes": spec.notes,
        "classic_local_rows": classic_rows,
        "puct_non_regression_rows": puct_rows,
        "excluded_sanity_rows": excluded_rows,
        "training_metric_rows": training_metric_rows,
    }


def classify_experiment(summary: dict[str, Any]) -> tuple[str, str]:
    traces = list(summary.get("traces") or [])
    improving_snapshots: list[dict[str, Any]] = []
    loss_improved = False

    for trace in traces:
        metrics_by_epoch = {
            int(row["epoch"]): row for row in trace.get("training_metric_rows", [])
        }
        init_loss = metrics_by_epoch.get(0, {}).get("total_loss")
        for metric_row in trace.get("training_metric_rows", []):
            if (
                init_loss is not None
                and int(metric_row["epoch"]) > 0
                and metric_row["total_loss"] < init_loss
            ):
                loss_improved = True

        for epoch in TRACE_EPOCHS:
            classic_rows = [
                row
                for row in trace.get("classic_local_rows", [])
                if int(row["epoch"]) == epoch
            ]
            puct_rows = [
                row
                for row in trace.get("puct_non_regression_rows", [])
                if int(row["epoch"]) == epoch
            ]
            target_rows = [
                row for row in classic_rows if row["role"] == "target_candidate"
            ]
            target_improved = sum(
                1 for row in target_rows if row["improved_vs_current"]
            )
            control_pass = any(
                row["row_id"] == CLASSIC_CONTROL_ROW_ID and row["strict_pass"]
                for row in classic_rows
            )
            puct_regressions = sum(
                1 for row in puct_rows if row["cross_teacher_regression"]
            )
            if target_improved > 0:
                improving_snapshots.append(
                    {
                        "trace_name": trace["trace_name"],
                        "epoch": epoch,
                        "target_improved": target_improved,
                        "control_pass": control_pass,
                        "puct_regressions": puct_regressions,
                    }
                )

    successful = [
        row
        for row in improving_snapshots
        if row["control_pass"] and row["puct_regressions"] == 0
    ]
    if successful:
        only_w2 = all(
            row["trace_name"] == "classic_artifact_only_w2" for row in successful
        )
        if only_w2:
            return (
                "classic_teacher_weight_needed",
                "run a production-scale lane with Classic artifact weight 2, not a sweep.",
            )
        return (
            "classic_teacher_diagnostic_viable",
            "build one small controlled Classic-teacher production-scale lane with pre-arena Classic-bucket gate and PUCT non-regression gate.",
        )

    if improving_snapshots and all(
        not row["control_pass"] for row in improving_snapshots
    ):
        return (
            "classic_bucket_overfit",
            "add same-bucket controls or reduce LR before any production lane.",
        )

    if improving_snapshots and all(
        row["puct_regressions"] > 0 for row in improving_snapshots
    ):
        return (
            "cross_teacher_interference",
            "keep Classic and PUCT branches isolated; do not production-train this artifact until interference is understood.",
        )

    if not improving_snapshots and loss_improved:
        return (
            "objective_local_misalignment",
            "inspect target construction / value source before training.",
        )

    return (
        "classic_teacher_artifact_no_signal",
        "abandon this Classic artifact and return to PUCT-reference branch or next non-opening family.",
    )


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-lite Incumbent Proxy Classic Teacher Diagnostic Results",
        "",
        "## 1. Context",
        "",
        "- No arena was run.",
        "- No model was promoted.",
        "- Active references stayed read-only at `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.",
        "- No production lane was trained.",
        "- The experiment used `storage/ai/alphazero_lite/current` only as the initializer.",
        "- Temporary exports were written only under `/tmp/azlite_incumbent_proxy_classic_teacher_diagnostic/exports/`.",
        "",
        "## 2. Why PR #47 selected the Classic diagnostic branch",
        "",
        "- PR #47 separated incompatible teacher policies into Classic, PUCT, and excluded buckets.",
        "- The Classic bucket offered 5 target candidates plus 1 preservation control without mutating active references.",
        "- The PUCT bucket still requires a separate reference-policy branch and was therefore kept out of training targets here.",
        "",
        "## 3. Artifact construction and validation",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "row_id",
                "role",
                "active_reference_move",
                "preferred_teacher",
                "preferred_move",
                "policy_target_reference_mass",
                "value_source",
                "status",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["role"],
                    str(row["active_reference_move"]),
                    row["preferred_teacher"],
                    str(row["preferred_move"]),
                    format_float(row["policy_target_reference_mass"]),
                    row["value_source"],
                    row["status"],
                    row["notes"],
                ]
                for row in summary["artifact_summary"]["artifact_rows"]
            ],
        )
    )
    lines.extend(
        [
            "",
            f"Artifact validation status: `{summary['artifact_summary']['validation']['status']}`.",
            f"Row count: `{summary['artifact_summary']['validation']['row_count']}`. Target candidates: `{summary['artifact_summary']['validation']['target_candidate_count']}`. Preservation controls: `{summary['artifact_summary']['validation']['preservation_control_count']}`.",
            "",
            "## 4. Trace definitions",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["trace_name", "data_files", "replay_weights", "epochs", "status", "notes"],
            [
                [
                    trace["trace_name"],
                    ", ".join(trace["data_files"]),
                    ", ".join(str(weight) for weight in trace["replay_weights"]),
                    ", ".join(str(epoch) for epoch in trace["epochs"]),
                    trace["status"],
                    trace["notes"],
                ]
                for trace in summary["traces"]
            ],
        )
    )
    lines.extend(["", "## 5. Classic-bucket local results", ""])
    classic_rows = [
        row
        for trace in summary["traces"]
        for row in trace["classic_local_rows"]
        if int(row["epoch"]) > 0
    ]
    lines.extend(
        markdown_table(
            [
                "trace_name",
                "epoch",
                "row_id",
                "role",
                "active_reference_move",
                "selected_move_384",
                "selected_move_1200",
                "reference_visit_share_384",
                "reference_visit_share_1200",
                "reference_policy_probability",
                "reference_policy_rank",
                "improved_vs_current",
                "strict_pass",
                "notes",
            ],
            [
                [
                    row["trace_name"],
                    str(row["epoch"]),
                    row["row_id"],
                    row["role"],
                    str(row["active_reference_move"]),
                    str(row["selected_move_384"]),
                    str(row["selected_move_1200"]),
                    format_float(row["reference_visit_share_384"]),
                    format_float(row["reference_visit_share_1200"]),
                    format_float(row["reference_policy_probability"]),
                    str(row["reference_policy_rank"])
                    if row["reference_policy_rank"] is not None
                    else "-",
                    format_bool(row["improved_vs_current"]),
                    format_bool(row["strict_pass"]),
                    row["notes"],
                ]
                for row in classic_rows
            ],
        )
    )
    lines.extend(["", "## 6. PUCT-bucket non-regression check", ""])
    puct_rows = [
        row
        for trace in summary["traces"]
        for row in trace["puct_non_regression_rows"]
        if int(row["epoch"]) > 0
    ]
    lines.extend(
        markdown_table(
            [
                "trace_name",
                "epoch",
                "row_id",
                "puct_preferred_move",
                "selected_move_384",
                "selected_move_1200",
                "selected_equals_puct_preferred",
                "puct_preferred_visit_share_384",
                "puct_preferred_visit_share_1200",
                "cross_teacher_regression",
                "notes",
            ],
            [
                [
                    row["trace_name"],
                    str(row["epoch"]),
                    row["row_id"],
                    str(row["puct_preferred_move"]),
                    str(row["selected_move_384"]),
                    str(row["selected_move_1200"]),
                    format_bool(row["selected_equals_puct_preferred"]),
                    format_float(row["puct_preferred_visit_share_384"]),
                    format_float(row["puct_preferred_visit_share_1200"]),
                    format_bool(row["cross_teacher_regression"]),
                    row["notes"],
                ]
                for row in puct_rows
            ],
        )
    )
    lines.extend(["", "## 7. Excluded diagnostic sanity check", ""])
    excluded_rows = [
        row
        for trace in summary["traces"]
        for row in trace["excluded_sanity_rows"]
        if int(row["epoch"]) > 0
    ]
    lines.extend(
        markdown_table(
            [
                "trace_name",
                "epoch",
                "row_id",
                "selected_move_1200",
                "policy_entropy",
                "unexpected_drift",
                "notes",
            ],
            [
                [
                    row["trace_name"],
                    str(row["epoch"]),
                    row["row_id"],
                    str(row["selected_move_1200"]),
                    format_float(row["policy_entropy"]),
                    format_bool(row["unexpected_drift"]),
                    row["notes"],
                ]
                for row in excluded_rows
            ],
        )
    )
    lines.extend(["", "## 8. Training metrics", ""])
    metric_rows = [
        row
        for trace in summary["traces"]
        for row in trace["training_metric_rows"]
        if int(row["epoch"]) > 0
    ]
    lines.extend(
        markdown_table(
            [
                "trace_name",
                "epoch",
                "policy_loss",
                "value_loss",
                "total_loss",
                "artifact_cross_entropy",
                "target_candidate_cross_entropy",
                "preservation_control_cross_entropy",
                "notes",
            ],
            [
                [
                    row["trace_name"],
                    str(row["epoch"]),
                    format_float(row["policy_loss"]),
                    format_float(row["value_loss"]),
                    format_float(row["total_loss"]),
                    format_float(row["artifact_cross_entropy"]),
                    format_float(row["target_candidate_cross_entropy"]),
                    format_float(row["preservation_control_cross_entropy"]),
                    row["notes"],
                ]
                for row in metric_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 9. Interpretation",
            "",
            f"- Final classification: `{summary['classification']}`.",
            "- Acceptance checks: arena_run=`false`, model_promoted=`false`, active_references_mutated=`false`, puct_rows_trained=`false`, excluded_rows_trained=`false`.",
            "- The local gate used 384 and 1200 search budgets only for per-row diagnostics; no arena or benchmark sweep was run.",
            "",
            "## 10. Exactly one recommended next action",
            "",
            f"Recommendation: **{summary['recommended_next_action']}**",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.export_root.mkdir(parents=True, exist_ok=True)

    artifact_build = build_classic_artifact(
        argparse.Namespace(
            reference_path=args.reference_path,
            suite_path=args.suite_path,
            current_artifact=args.current_artifact,
            classic_rows_path=args.classic_rows_path,
            puct_rows_path=args.puct_rows_path,
            excluded_rows_path=args.excluded_rows_path,
            bucket_summary_path=args.bucket_summary_path,
            report_path=args.bucket_report_path,
            output_dir=args.output_dir,
            artifact_out=args.artifact_path,
            summary_out=args.artifact_summary_path,
            target_only_out=args.target_only_path,
            control_only_out=args.control_only_path,
            input_encoding="kalah_v3",
            policy_target_mode="default",
            value_target_mode="default",
        )
    )
    write_jsonl(args.artifact_path, artifact_build["artifact_rows"])
    write_jsonl(args.target_only_path, artifact_build["target_rows"])
    write_jsonl(args.control_only_path, artifact_build["control_rows"])
    write_json(args.artifact_summary_path, artifact_build["summary"])

    artifact_summary = artifact_build["summary"]
    if artifact_summary["validation"]["target_candidate_count"] < 3:
        summary = {
            "schema": SCHEMA,
            "artifact_summary": artifact_summary,
            "classification": "artifact_not_enough_signal",
            "recommended_next_action": "stop before training trace; fewer than 3 Classic target candidates remained after validation",
            "traces": [],
        }
        write_json(args.trace_summary_path, summary)
        args.report_path.write_text(build_report(summary), encoding="utf-8")
        print(json.dumps({"classification": summary["classification"]}, sort_keys=True))
        return 0

    current_metadata = load_json(args.current_artifact / "metadata.json")
    model_spec = model_spec_from_metadata(current_metadata)
    init_checkpoint = current_checkpoint_path(args.current_artifact, args.output_dir)
    device = choose_device(args.device)

    classic_rows = read_jsonl(args.classic_rows_path)
    puct_rows = read_jsonl(args.puct_rows_path)
    excluded_rows = read_jsonl(args.excluded_rows_path)
    classic_specs = row_specs_from_bucket_rows(classic_rows, CLASSIC_ROW_IDS)
    puct_specs = row_specs_from_bucket_rows(puct_rows, PUCT_ROW_IDS)
    excluded_specs = row_specs_from_bucket_rows(excluded_rows, EXCLUDED_ROW_IDS)
    classic_spec_by_id = {row["row_id"]: row for row in classic_specs}
    puct_spec_by_id = {row["row_id"]: row for row in puct_specs}
    excluded_spec_by_id = {row["row_id"]: row for row in excluded_specs}

    classic_baseline_by_id = {
        row["row_id"]: row
        for row in evaluate_row_set_baseline(
            artifact_path=args.current_artifact,
            row_specs=classic_specs,
            kind="classic",
            seed=args.seed,
        )
    }
    puct_baseline_by_id = {
        row["row_id"]: row
        for row in evaluate_row_set_baseline(
            artifact_path=args.current_artifact,
            row_specs=puct_specs,
            kind="puct",
            seed=args.seed + 1000,
        )
    }
    excluded_baseline_by_id = {
        row["row_id"]: row
        for row in evaluate_row_set_baseline(
            artifact_path=args.current_artifact,
            row_specs=excluded_specs,
            kind="excluded",
            seed=args.seed + 2000,
        )
    }

    traces = [
        TraceSpec(
            name="classic_artifact_only_w1",
            data_files=(args.artifact_path,),
            replay_weights=(1,),
            epochs=TRACE_EPOCHS,
            notes="all 6 Classic rows only, replay weight 1",
        ),
        TraceSpec(
            name="classic_artifact_only_w2",
            data_files=(args.artifact_path,),
            replay_weights=(2,),
            epochs=TRACE_EPOCHS,
            notes="all 6 Classic rows only, replay weight 2",
        ),
    ]
    if args.target_only_path.exists() and args.control_only_path.exists():
        traces.append(
            TraceSpec(
                name="classic_artifact_plus_preservation_boost",
                data_files=(args.target_only_path, args.control_only_path),
                replay_weights=(1, 2),
                epochs=TRACE_EPOCHS,
                notes="target candidates weight 1 plus preservation control weight 2",
            )
        )

    trace_results = [
        run_trace(
            spec=trace,
            row_spec_by_id=classic_spec_by_id,
            puct_spec_by_id=puct_spec_by_id,
            excluded_spec_by_id=excluded_spec_by_id,
            classic_baseline_by_id=classic_baseline_by_id,
            puct_baseline_by_id=puct_baseline_by_id,
            excluded_baseline_by_id=excluded_baseline_by_id,
            init_checkpoint=init_checkpoint,
            current_metadata=current_metadata,
            model_spec=model_spec,
            export_root=args.export_root,
            seed=args.seed,
            device=device,
        )
        for trace in traces
    ]

    classification, next_action = classify_experiment(
        {"artifact_summary": artifact_summary, "traces": trace_results}
    )
    summary = {
        "schema": SCHEMA,
        "inputs": {
            "reference_path": str(args.reference_path),
            "suite_path": str(args.suite_path),
            "current_artifact": str(args.current_artifact),
            "artifact_path": str(args.artifact_path),
            "artifact_summary_path": str(args.artifact_summary_path),
            "classic_rows_path": str(args.classic_rows_path),
            "puct_rows_path": str(args.puct_rows_path),
            "excluded_rows_path": str(args.excluded_rows_path),
            "init_checkpoint": str(init_checkpoint),
        },
        "artifact_summary": artifact_summary,
        "classification": classification,
        "recommended_next_action": next_action,
        "acceptance": {
            "arena_run": False,
            "model_promoted": False,
            "active_references_mutated": False,
            "puct_rows_used_as_targets": False,
            "excluded_rows_used_as_targets": False,
            "broad_replay_sweep_run": False,
        },
        "traces": trace_results,
    }
    write_json(args.trace_summary_path, summary)
    args.report_path.write_text(build_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact_summary_path": str(args.artifact_summary_path),
                "trace_summary_path": str(args.trace_summary_path),
                "report_path": str(args.report_path),
                "classification": classification,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
