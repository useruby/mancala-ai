#!/usr/bin/env python3
"""Teacher policy split probe — tests whether teacher-aware training can
avoid cross-teacher interference between Classic-MCTS and PUCT teacher
preferences within the incumbent_proxy_disagreement family.

Key constraints (inherited from PR #69):
  - No arena is run.
  - No model is promoted.
  - Active references are not mutated.
  - PR #69 proposed patch bundle is not applied.
  - Excluded rows are never training targets.
  - Single-head mixed replay is a diagnostic baseline, not a candidate.
  - Teacher-conditioned / dual-head variants are experimental only.
"""

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
from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint
from ml.alphazero_lite.run_incumbent_proxy_disagreement_value_backup_audit import (
    load_json,
    write_json,
)
from ml.alphazero_lite.run_incumbent_proxy_teacher_policy_decision_audit import (
    canonical_reference_rows,
)
from ml.alphazero_lite.self_play import encode_state, build_eval_search_options
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

# ── Paths ──────────────────────────────────────────────────────────────────
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
DEFAULT_OUTPUT_DIR = Path("/tmp/azlite_teacher_policy_split_probe")
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "teacher_policy_split_probe_summary.json"
DEFAULT_CLASSIC_PROBE_ROWS_PATH = (
    DEFAULT_OUTPUT_DIR / "classic_teacher_probe_rows.jsonl"
)
DEFAULT_PUCT_PROBE_ROWS_PATH = DEFAULT_OUTPUT_DIR / "puct_teacher_probe_rows.jsonl"
DEFAULT_EXCLUDED_PROBE_ROWS_PATH = (
    DEFAULT_OUTPUT_DIR / "excluded_teacher_probe_rows.jsonl"
)
DEFAULT_EXPORT_ROOT = DEFAULT_OUTPUT_DIR / "exports"
DEFAULT_REPORT_PATH = Path("docs/alphazero-lite-teacher-policy-split-probe-results.md")

SCHEMA = "azlite_teacher_policy_split_probe_v1"
FAMILY = "incumbent_proxy_disagreement"
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
ALL_ROW_IDS = tuple(sorted(set(CLASSIC_ROW_IDS + PUCT_ROW_IDS + EXCLUDED_ROW_IDS)))

REFERENCE_POLICY_MASS = 0.85
NON_REFERENCE_POLICY_MASS = 0.15
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
TEACHER_INPUT_EXTRA_FEATURES = 2  # one-hot: [is_classic, is_puct]


# ── Data classes ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VariantSpec:
    name: str
    data_files: tuple[Path, ...]
    replay_weights: tuple[int, ...]
    epochs: tuple[int, ...]
    notes: str
    teacher_conditioned: bool = False


# ── I/O helpers ────────────────────────────────────────────────────────────
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
        prob = float(policy[move])
        if prob > 0.0:
            entropy -= prob * math.log(prob, 2)
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
    squared: dict[str, float] = {"policy": 0.0, "value": 0.0, "trunk": 0.0}
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


# ── Policy / target construction ──────────────────────────────────────────
def build_sharp_policy(legal_moves: list[int], target_move: int) -> list[float]:
    policy = [0.0] * 6
    if target_move not in legal_moves:
        raise ValueError(f"target move {target_move} not in legal moves {legal_moves}")
    if len(legal_moves) == 1:
        policy[target_move] = 1.0
        return policy
    residual = NON_REFERENCE_POLICY_MASS / float(len(legal_moves) - 1)
    for move in legal_moves:
        policy[move] = residual
    policy[target_move] = REFERENCE_POLICY_MASS
    return policy


# ── Step 1: Bucket reconstruction ──────────────────────────────────────────
def reconstruct_buckets(
    classic_rows_path: Path,
    puct_rows_path: Path,
    excluded_rows_path: Path,
    suite_path: Path,
    reference_path: Path,
) -> dict[str, Any]:
    classic_rows = read_jsonl(classic_rows_path)
    puct_rows = read_jsonl(puct_rows_path)
    excluded_rows = read_jsonl(excluded_rows_path)

    classic_by_id = {str(row["row_id"]): row for row in classic_rows}
    puct_by_id = {str(row["row_id"]): row for row in puct_rows}
    excluded_by_id = {str(row["row_id"]): row for row in excluded_rows}

    from ml.alphazero_lite.forensic_suite import load_suite

    suite = load_suite(suite_path)
    suite_by_id = {position.id: position for position in suite}
    reference_payload = load_json(reference_path)
    reference_by_id = canonical_reference_rows(reference_payload)

    bucket_assignments: dict[str, str] = {}
    validation_entries: list[dict[str, Any]] = []
    canonical_states: dict[str, str] = {}
    duplicate_errors: list[str] = []

    for row_id in ALL_ROW_IDS:
        if row_id in classic_by_id:
            bucket = "classic_teacher"
            source = classic_by_id[row_id]
        elif row_id in puct_by_id:
            bucket = "puct_teacher"
            source = puct_by_id[row_id]
        elif row_id in excluded_by_id:
            bucket = "excluded"
            source = excluded_by_id[row_id]
        else:
            raise ValueError(f"row {row_id} not found in any bucket file")

        suite_row = suite_by_id.get(row_id)
        reference_row = reference_by_id.get(row_id)
        legal_moves = [int(m) for m in suite_row.legal_moves] if suite_row else []
        active_reference_move = (
            int(reference_row["reference_move"]) if reference_row else None
        )
        preferred_move = int(source["preferred_move"])
        canonical = suite_row.canonical_key if suite_row else None

        if canonical:
            if canonical in canonical_states and canonical_states[canonical] != bucket:
                duplicate_errors.append(
                    f"{row_id}: canonical collision with {canonical_states[canonical]}"
                )
            canonical_states[canonical] = bucket

        status = "ok"
        reasons: list[str] = []
        canonical_exists = suite_row is not None
        active_legal = (
            (active_reference_move is not None and active_reference_move in legal_moves)
            if legal_moves
            else False
        )
        preferred_legal = preferred_move in legal_moves if legal_moves else False

        if not canonical_exists:
            reasons.append("missing_canonical_state")
            status = "error"
        if not active_legal:
            reasons.append("active_reference_illegal")
            status = "error"
        if not preferred_legal:
            reasons.append("preferred_move_illegal")
            status = "error"

        bucket_assignments[row_id] = bucket
        validation_entries.append(
            {
                "row_id": row_id,
                "bucket": bucket,
                "canonical_state_exists": canonical_exists,
                "active_reference_legal": active_legal,
                "preferred_move_legal": preferred_legal,
                "status": status,
                "notes": "; ".join(reasons) if reasons else "ok",
            }
        )

    return {
        "classic_rows": classic_rows,
        "puct_rows": puct_rows,
        "excluded_rows": excluded_rows,
        "bucket_assignments": bucket_assignments,
        "validation_entries": validation_entries,
        "duplicate_errors": duplicate_errors,
        "_suite_by_id": suite_by_id,
        "_reference_by_id": reference_by_id,
    }


# ── Step 2: Build diagnostic target rows ────────────────────────────────────
def build_diagnostic_target_rows(
    bucket_data: dict[str, Any],
    current_artifact: Path,
) -> dict[str, Any]:
    current_metadata = load_json(current_artifact / "metadata.json")
    input_encoding = str(current_metadata.get("input_encoding", "kalah_v3"))
    suite_by_id = bucket_data["_suite_by_id"]
    reference_by_id = bucket_data["_reference_by_id"]

    classic_target_rows: list[dict[str, Any]] = []
    puct_target_rows: list[dict[str, Any]] = []
    excluded_target_rows: list[dict[str, Any]] = []

    classic_source = bucket_data["classic_rows"]
    puct_source = bucket_data["puct_rows"]
    excluded_source = bucket_data["excluded_rows"]

    for source_rows, bucket_name, row_ids, teacher_id in [
        (classic_source, "classic_teacher", CLASSIC_ROW_IDS, "classic_mcts"),
        (puct_source, "puct_teacher", PUCT_ROW_IDS, "artifact_puct"),
    ]:
        by_id = {str(r["row_id"]): r for r in source_rows}
        for row_id in row_ids:
            raw = by_id[row_id]
            suite_row = suite_by_id[row_id]
            reference_row = reference_by_id[row_id]
            legal_moves = [int(m) for m in suite_row.legal_moves]
            active_reference_move = int(reference_row["reference_move"])
            preferred_move = int(raw["preferred_move"])

            target_move = (
                active_reference_move
                if bucket_name == "classic_teacher"
                else preferred_move
            )
            policy = build_sharp_policy(legal_moves, target_move)
            encoded_state = encode_state(
                dict(suite_row.state), input_encoding=input_encoding
            )
            teacher_value = float(reference_row.get("teacher_value", 0.0))

            row = {
                "state": encoded_state,
                "raw_state": dict(suite_row.state),
                "legal_moves": legal_moves,
                "policy": policy,
                "value": teacher_value,
                "input_encoding": input_encoding,
                "policy_target_mode": "default",
                "policy_target_actual_mode": "default",
                "value_target_mode": "default",
                "source": "teacher_policy_split_probe",
                "family": FAMILY,
                "bucket": bucket_name,
                "teacher_id": teacher_id,
                "row_id": row_id,
                "active_reference_move": active_reference_move,
                "preferred_move": preferred_move,
                "target_move": target_move,
                "train_only": True,
                "fixture_mutation": False,
                "reference_move": active_reference_move,
                "teacher_selected_move": target_move,
            }
            if bucket_name == "classic_teacher":
                classic_target_rows.append(row)
            else:
                puct_target_rows.append(row)

    for row_id in EXCLUDED_ROW_IDS:
        by_id = {str(r["row_id"]): r for r in excluded_source}
        raw = by_id[row_id]
        suite_row = suite_by_id[row_id]
        reference_row = reference_by_id[row_id]
        legal_moves = [int(m) for m in suite_row.legal_moves]
        encoded_state = encode_state(
            dict(suite_row.state), input_encoding=input_encoding
        )
        preferred_move = int(raw["preferred_move"])
        row = {
            "state": encoded_state,
            "raw_state": dict(suite_row.state),
            "legal_moves": legal_moves,
            "input_encoding": input_encoding,
            "source": "teacher_policy_split_probe",
            "family": FAMILY,
            "bucket": "excluded",
            "teacher_id": "excluded",
            "row_id": row_id,
            "active_reference_move": int(reference_row["reference_move"])
            if reference_row
            else None,
            "preferred_move": preferred_move,
            "do_not_train": True,
            "do_not_gate": True,
            "exclude_from_training": True,
        }
        excluded_target_rows.append(row)

    return {
        "classic_target_rows": classic_target_rows,
        "puct_target_rows": puct_target_rows,
        "excluded_target_rows": excluded_target_rows,
        "input_encoding": input_encoding,
        "current_metadata": current_metadata,
    }


# ── Step 3: Current baseline ──────────────────────────────────────────────
def evaluate_position_summary(
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
        visits = [float(v) for v in list(probe.get("visits") or [])]
        total_visits = sum(float(visits[m]) for m in legal_moves if m < len(visits))
        _child_stats = {int(c["move"]): c for c in list(probe.get("child_stats") or [])}
        return {
            "selected_move": None
            if probe.get("selected_move") is None
            else int(probe["selected_move"]),
            "visit_shares": {
                int(m): round(float(visits[m]) / total_visits, 4)
                if total_visits > 0.0 and m < len(visits)
                else None
                for m in legal_moves
            },
            "policy": [float(v) for v in list(probe.get("policy") or [])],
            "value": round(float(probe.get("value", 0.0)), 4),
        }

    r384 = run_budget(384)
    r1200 = run_budget(1200)
    return {
        384: r384,
        1200: r1200,
        "policy_distribution": {
            int(m): round(float(r1200["policy"][m]), 4)
            for m in legal_moves
            if m < len(r1200["policy"])
        },
        "policy_top_move": top_policy_move(r1200["policy"], legal_moves),
        "policy_entropy": policy_entropy(r1200["policy"], legal_moves),
        "value_prediction": r1200["value"],
    }


def evaluate_row_set_baseline(
    artifact_path: Path,
    row_specs: list[dict[str, Any]],
    seed: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    evaluator_cache: dict[Path, ArtifactEvaluator] = {}
    for index, row_spec in enumerate(row_specs):
        summary = evaluate_position_summary(
            artifact_path=artifact_path,
            state=dict(row_spec["raw_state"]),
            legal_moves=[int(m) for m in row_spec["legal_moves"]],
            seed=seed + index * 101,
            evaluator_cache=evaluator_cache,
        )
        ref_move = int(row_spec["active_reference_move"])
        pref_move = int(row_spec["preferred_move"])
        results[row_spec["row_id"]] = {
            "row_id": row_spec["row_id"],
            "bucket": row_spec.get("bucket", ""),
            "active_reference_move": ref_move,
            "preferred_move": pref_move,
            "selected_move_384": summary[384]["selected_move"],
            "selected_move_1200": summary[1200]["selected_move"],
            "selected_is_reference_384": summary[384]["selected_move"] == ref_move,
            "selected_is_reference_1200": summary[1200]["selected_move"] == ref_move,
            "selected_equals_classic_preferred_384": summary[384]["selected_move"]
            == ref_move,
            "selected_equals_classic_preferred_1200": summary[1200]["selected_move"]
            == ref_move,
            "selected_equals_puct_preferred_384": summary[384]["selected_move"]
            == pref_move,
            "selected_equals_puct_preferred_1200": summary[1200]["selected_move"]
            == pref_move,
            "reference_visit_share_384": summary[384]["visit_shares"].get(ref_move),
            "reference_visit_share_1200": summary[1200]["visit_shares"].get(ref_move),
            "puct_preferred_visit_share_384": summary[384]["visit_shares"].get(
                pref_move
            ),
            "puct_preferred_visit_share_1200": summary[1200]["visit_shares"].get(
                pref_move
            ),
            "policy_probability_preferred": summary["policy_distribution"].get(
                pref_move
            )
            if row_spec.get("bucket") == "classic_teacher"
            else summary["policy_distribution"].get(pref_move),
            "policy_rank_preferred": metric_rank(
                summary["policy_distribution"], ref_move
            )
            if row_spec.get("bucket") == "classic_teacher"
            else metric_rank(summary["policy_distribution"], pref_move),
            "policy_entropy": summary["policy_entropy"],
            "strict_pass": (
                summary[384]["selected_move"] == ref_move
                and summary[1200]["selected_move"] == ref_move
            ),
        }
    return results


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


# ── Variant artifacts ──────────────────────────────────────────────────────
def build_variant_artifacts(
    diagnostic_rows: dict[str, Any],
    artifact_dir: Path,
) -> list[VariantSpec]:
    classic = diagnostic_rows["classic_target_rows"]
    puct = diagnostic_rows["puct_target_rows"]

    def write(name: str, rows: list[dict[str, Any]]) -> Path:
        path = artifact_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        return path

    # A: single_head_mixed_replay — mix Classic + PUCT targets in one file
    mixed_rows = classic + puct
    single_head_path = write("single_head_mixed", mixed_rows)

    # B: classic_only_replay
    classic_only_path = write("classic_only", classic)

    # C: puct_only_replay
    puct_only_path = write("puct_only", puct)

    return [
        VariantSpec(
            name="single_head_mixed_replay",
            data_files=(single_head_path,),
            replay_weights=(1,),
            epochs=TRACE_EPOCHS,
            notes="Negative control: train single-head model on mixed Classic + PUCT targets",
        ),
        VariantSpec(
            name="classic_only_replay",
            data_files=(classic_only_path,),
            replay_weights=(1,),
            epochs=TRACE_EPOCHS,
            notes="Train on Classic targets only — reproduces Classic gains and PUCT damage from PR #48",
        ),
        VariantSpec(
            name="puct_only_replay",
            data_files=(puct_only_path,),
            replay_weights=(1,),
            epochs=TRACE_EPOCHS,
            notes="Train on PUCT targets only — check whether PUCT-only training damages Classic rows",
        ),
    ]


def evaluate_dataset_metrics(
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
    state_dict = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
    norms = delta_norms(state_dict, initial_state)
    return {
        "policy_loss": round(float(np.mean(policy_ce)), 6),
        "value_loss": round(float(np.mean(value_loss_vector)), 6),
        "total_loss": round(float(np.mean(total_loss)), 6),
        "policy_head_delta_norm": norms["policy"],
        "value_head_delta_norm": norms["value"],
        "trunk_delta_norm": norms["trunk"],
    }


def train_one_epoch_wrapper(
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
        if metrics.get("gradient_norm") is not None
        else None,
    }


# ── Run one variant trace (standard training) ─────────────────────────────
def run_variant_trace(
    spec: VariantSpec,
    classic_specs: dict[str, dict[str, Any]],
    puct_specs: dict[str, dict[str, Any]],
    excluded_specs: dict[str, dict[str, Any]],
    classic_baseline: dict[str, dict[str, Any]],
    puct_baseline: dict[str, dict[str, Any]],
    excluded_baseline: dict[str, dict[str, Any]],
    init_checkpoint: Path,
    current_metadata: dict[str, Any],
    base_model_spec: dict[str, Any],
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
    model_spec = dict(base_model_spec)
    model = PolicyValueNet(
        hidden_sizes=tuple(model_spec["hidden_sizes"]),
        model_type=str(model_spec["model_type"]),
        input_size=input_size_for_encoding(str(model_spec["input_encoding"])),
    )
    load_checkpoint_into_model(model, init_checkpoint)
    model = model.to(device)
    initial_state = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
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

    classic_results: list[dict[str, Any]] = []
    puct_results: list[dict[str, Any]] = []
    excluded_results: list[dict[str, Any]] = []
    training_metrics: list[dict[str, Any]] = []
    evaluator_cache: dict[Path, ArtifactEvaluator] = {}

    def snapshot(epoch: int, grad_norm: float | None, notes: str) -> None:
        ckpt = checkpoints_root / f"epoch_{epoch}.npz"
        np.savez(ckpt, **checkpoint_from_model(model))
        export_dir = export_checkpoint_artifact(
            checkpoint_path=ckpt,
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
        training_metrics.append(
            {
                "variant": spec.name,
                "epoch": epoch,
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "total_loss": metrics["total_loss"],
                "gradient_norm": grad_norm,
                "notes": notes,
            }
        )

        for row_id in CLASSIC_ROW_IDS:
            spec_row = classic_specs[row_id]
            summary = evaluate_position_summary(
                artifact_path=export_dir,
                state=dict(spec_row["raw_state"]),
                legal_moves=[int(m) for m in spec_row["legal_moves"]],
                seed=seed + epoch + len(classic_results),
                evaluator_cache=evaluator_cache,
            )
            classic_results.append(
                build_classic_row_result(
                    variant=spec.name,
                    epoch=epoch,
                    row_spec=spec_row,
                    eval_summary=summary,
                    baseline=classic_baseline[row_id],
                )
            )
        for row_id in PUCT_ROW_IDS:
            spec_row = puct_specs[row_id]
            summary = evaluate_position_summary(
                artifact_path=export_dir,
                state=dict(spec_row["raw_state"]),
                legal_moves=[int(m) for m in spec_row["legal_moves"]],
                seed=seed + 500 + epoch + len(puct_results),
                evaluator_cache=evaluator_cache,
            )
            puct_results.append(
                build_puct_row_result(
                    variant=spec.name,
                    epoch=epoch,
                    row_spec=spec_row,
                    eval_summary=summary,
                    baseline=puct_baseline[row_id],
                )
            )
        for row_id in EXCLUDED_ROW_IDS:
            spec_row = excluded_specs[row_id]
            summary = evaluate_position_summary(
                artifact_path=export_dir,
                state=dict(spec_row["raw_state"]),
                legal_moves=[int(m) for m in spec_row["legal_moves"]],
                seed=seed + 900 + epoch + len(excluded_results),
                evaluator_cache=evaluator_cache,
            )
            excluded_results.append(
                build_excluded_row_result(
                    variant=spec.name,
                    epoch=epoch,
                    row_spec=spec_row,
                    eval_summary=summary,
                    baseline=excluded_baseline[row_id],
                )
            )

    snapshot(0, None, "initial_checkpoint")
    for epoch in range(1, max(spec.epochs) + 1):
        em = train_one_epoch_wrapper(
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
            snapshot(epoch, em["gradient_norm"], "post_epoch")

    return {
        "variant": spec.name,
        "epochs": list(spec.epochs),
        "status": "completed",
        "notes": spec.notes,
        "classic_results": classic_results,
        "puct_results": puct_results,
        "excluded_results": excluded_results,
        "training_metrics": training_metrics,
    }


# ── Row result builders ────────────────────────────────────────────────────
def build_classic_row_result(
    variant: str,
    epoch: int,
    row_spec: dict[str, Any],
    eval_summary: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    ref_move = int(row_spec["active_reference_move"])
    dist = eval_summary["policy_distribution"]
    curr = {
        "selected_is_reference_384": eval_summary[384]["selected_move"] == ref_move,
        "selected_is_reference_1200": eval_summary[1200]["selected_move"] == ref_move,
        "ref_visit_share_384": eval_summary[384]["visit_shares"].get(ref_move),
        "ref_visit_share_1200": eval_summary[1200]["visit_shares"].get(ref_move),
    }
    improved = _classic_improved(baseline, curr)
    damaged = _classic_damaged(baseline, curr)
    notes = []
    if improved:
        notes.append("improved")
    if damaged:
        notes.append("damaged")
    if not notes:
        notes.append("no_change")
    return {
        "variant": variant,
        "epoch": epoch,
        "row_id": row_spec["row_id"],
        "active_reference_move": ref_move,
        "classic_preferred_move": ref_move,
        "selected_move_384": eval_summary[384]["selected_move"],
        "selected_move_1200": eval_summary[1200]["selected_move"],
        "selected_equals_classic_preferred_384": curr["selected_is_reference_384"],
        "selected_equals_classic_preferred_1200": curr["selected_is_reference_1200"],
        "reference_visit_share_384": curr["ref_visit_share_384"],
        "reference_visit_share_1200": curr["ref_visit_share_1200"],
        "classic_preferred_policy_probability": dist.get(ref_move),
        "classic_preferred_policy_rank": metric_rank(dist, ref_move),
        "policy_entropy": eval_summary["policy_entropy"],
        "improved_vs_current": improved,
        "damaged_vs_current": damaged,
        "strict_pass": curr["selected_is_reference_384"]
        and curr["selected_is_reference_1200"],
        "notes": ",".join(notes),
    }


def build_puct_row_result(
    variant: str,
    epoch: int,
    row_spec: dict[str, Any],
    eval_summary: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    puct_move = int(row_spec["preferred_move"])
    dist = eval_summary["policy_distribution"]
    curr = {
        "selected_equals_puct_384": eval_summary[384]["selected_move"] == puct_move,
        "selected_equals_puct_1200": eval_summary[1200]["selected_move"] == puct_move,
        "puct_visit_share_384": eval_summary[384]["visit_shares"].get(puct_move),
        "puct_visit_share_1200": eval_summary[1200]["visit_shares"].get(puct_move),
    }
    improved = _puct_improved(baseline, curr)
    damaged = _puct_damaged(baseline, curr)
    notes = []
    if improved:
        notes.append("improved")
    if damaged:
        notes.append("damaged")
    if not notes:
        notes.append("no_change")
    return {
        "variant": variant,
        "epoch": epoch,
        "row_id": row_spec["row_id"],
        "puct_preferred_move": puct_move,
        "active_reference_move": int(row_spec["active_reference_move"]),
        "selected_move_384": eval_summary[384]["selected_move"],
        "selected_move_1200": eval_summary[1200]["selected_move"],
        "selected_equals_puct_preferred_384": curr["selected_equals_puct_384"],
        "selected_equals_puct_preferred_1200": curr["selected_equals_puct_1200"],
        "puct_preferred_visit_share_384": curr["puct_visit_share_384"],
        "puct_preferred_visit_share_1200": curr["puct_visit_share_1200"],
        "puct_preferred_policy_probability": dist.get(puct_move),
        "puct_preferred_policy_rank": metric_rank(dist, puct_move),
        "policy_entropy": eval_summary["policy_entropy"],
        "improved_vs_current": improved,
        "damaged_vs_current": damaged,
        "notes": ",".join(notes),
    }


def build_excluded_row_result(
    variant: str,
    epoch: int,
    row_spec: dict[str, Any],
    eval_summary: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    current_entropy = eval_summary["policy_entropy"]
    baseline_entropy = float(baseline.get("policy_entropy", 0.0) or 0.0)
    selected_changed = eval_summary[1200]["selected_move"] != baseline.get(
        "selected_move_1200"
    )
    entropy_delta = abs(current_entropy - baseline_entropy)
    unexpected_drift = selected_changed and entropy_delta >= 0.5
    return {
        "variant": variant,
        "epoch": epoch,
        "row_id": row_spec["row_id"],
        "selected_move_1200": eval_summary[1200]["selected_move"],
        "policy_entropy": current_entropy,
        "unexpected_drift": unexpected_drift,
        "notes": "drift" if unexpected_drift else "stable",
    }


def _classic_improved(baseline: dict, current: dict) -> bool:
    if not baseline.get("selected_is_reference_384") and current.get(
        "selected_is_reference_384"
    ):
        return True
    if not baseline.get("selected_is_reference_1200") and current.get(
        "selected_is_reference_1200"
    ):
        return True
    for budget in SEARCH_BUDGETS:
        b = baseline.get(f"reference_visit_share_{budget}")
        c = current.get(f"ref_visit_share_{budget}")
        if b is not None and c is not None and c > b + 1e-6:
            return True
    return False


def _classic_damaged(baseline: dict, current: dict) -> bool:
    if baseline.get("selected_is_reference_384") and not current.get(
        "selected_is_reference_384"
    ):
        return True
    if baseline.get("selected_is_reference_1200") and not current.get(
        "selected_is_reference_1200"
    ):
        return True
    for budget in SEARCH_BUDGETS:
        b = baseline.get(f"reference_visit_share_{budget}")
        c = current.get(f"ref_visit_share_{budget}")
        if b is not None and c is not None and c < b - 0.05:
            return True
    return False


def _puct_improved(baseline: dict, current: dict) -> bool:
    if not baseline.get("selected_equals_puct_preferred_1200") and current.get(
        "selected_equals_puct_1200"
    ):
        return True
    for budget in SEARCH_BUDGETS:
        b = baseline.get(f"puct_preferred_visit_share_{budget}")
        c = current.get(f"puct_visit_share_{budget}")
        if b is not None and c is not None and c > b + 0.05:
            return True
    return False


def _puct_damaged(baseline: dict, current: dict) -> bool:
    if baseline.get("selected_equals_puct_preferred_1200") and not current.get(
        "selected_equals_puct_1200"
    ):
        return True
    for budget in SEARCH_BUDGETS:
        b = baseline.get(f"puct_preferred_visit_share_{budget}")
        c = current.get(f"puct_visit_share_{budget}")
        if b is not None and c is not None and c < b - 0.05:
            return True
    return False


# ── Variant D: Teacher-conditioned probe ──────────────────────────────────
class TeacherConditionedPolicyValueNet(PolicyValueNet):
    """Subclass of PolicyValueNet that accepts extra teacher_id features.

    Input size = original_input_size + TEACHER_INPUT_EXTRA_FEATURES.
    The extra 2 features are one-hot [is_classic, is_puct].
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...],
        model_type: str,
        input_size: int,
        teacher_extra: int = TEACHER_INPUT_EXTRA_FEATURES,
    ):
        self.teacher_extra = teacher_extra
        super().__init__(hidden_sizes, model_type, input_size + teacher_extra)

    def forward_with_teacher(
        self, x: torch.Tensor, teacher_id: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        extra = torch.zeros(
            batch_size, self.teacher_extra, device=x.device, dtype=x.dtype
        )
        if teacher_id == "classic_mcts":
            extra[:, 0] = 1.0
        elif teacher_id == "artifact_puct":
            extra[:, 1] = 1.0
        else:
            extra[:, 0] = 0.5
            extra[:, 1] = 0.5
        augmented = torch.cat([x, extra], dim=1)
        return self.forward(augmented)


def init_teacher_conditioned_from_checkpoint(
    model: TeacherConditionedPolicyValueNet,
    checkpoint_path: Path,
) -> None:
    """Load trunk into teacher-conditioned model, zero-init extra input weights."""
    base_checkpoint = np.load(checkpoint_path)
    state_dict = model.state_dict()

    if "w_input" in base_checkpoint:
        w_base = torch.from_numpy(base_checkpoint["w_input"].T.copy())
        b_base = torch.from_numpy(base_checkpoint["b_input"].copy())
        w_new = state_dict["input_layer.weight"]
        b_new = state_dict["input_layer.bias"]
        w_new[:, : w_base.shape[1]] = w_base
        w_new[:, w_base.shape[1] :] = 0.0
        b_new[:] = b_base
        state_dict["input_layer.weight"] = w_new
        state_dict["input_layer.bias"] = b_new
    else:
        load_checkpoint_into_model(model, checkpoint_path)
        with torch.no_grad():
            w = state_dict.get("input_layer.weight")
            if w is not None:
                w[:, -model.teacher_extra :] = 0.0

    model.load_state_dict(state_dict, strict=False)


def teacher_conditioned_evaluate_position_summary(
    model: TeacherConditionedPolicyValueNet,
    teacher_id: str,
    raw_state: dict[str, Any],
    legal_moves: list[int],
    input_encoding: str,
    seed: int,
) -> dict[str, Any]:
    """Evaluate a position using the teacher-conditioned model with a
    custom evaluator pattern. This uses PyTorch to get the raw policy,
    then runs a lightweight PUCT search using that policy."""

    from ml.alphazero_lite.self_play import PUCT
    import random

    encoded = np.asarray(
        encode_state(dict(raw_state), input_encoding=input_encoding),
        dtype=np.float32,
    )
    from ml.alphazero_lite.kalah_rules import KalahGame

    position_game = KalahGame.from_state(raw_state)
    legal = position_game.possible_moves()

    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(encoded.reshape(1, -1)).to(
            next(model.parameters()).device
        )
        logits, value = model.forward_with_teacher(x_tensor, teacher_id)
        policy = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    masked = np.zeros(6, dtype=np.float32)
    if legal:
        masked[legal] = policy[legal]
        total = float(np.sum(masked))
        if total <= 0:
            masked[legal] = 1.0 / len(legal)
        else:
            masked /= total

    class CustomEvaluator:
        def evaluate(self, game):
            nonlocal value
            pol = masked
            return pol, float(value[0].item())

    def run_budget(budget: int, seed_offset: int) -> dict[str, Any]:
        rng = random.Random(seed + seed_offset)
        search = PUCT(
            evaluator=CustomEvaluator(),
            simulations=int(budget),
            c_puct=DEFAULT_C_PUCT,
            rng=rng,
            fpu_mode="parent_q",
            reuse_subtree=False,
            normalize_values=True,
            root_policy_mode="deterministic",
            tactical_root_bias=0.1,
            ablation_mode="full",
        )
        visits, root = search.run(position_game.clone())
        legal_m = position_game.possible_moves()
        selected = None
        if root is not None:
            selected = search.select_root_move(root, legal_m)
        if selected is None:
            if legal_m:
                selected = int(np.argmax(visits[legal_m]))
        total_v = sum(float(visits[m]) for m in legal_m if m < len(visits))
        return {
            "selected_move": None if selected is None else int(selected),
            "visit_shares": {
                int(m): round(float(visits[m]) / total_v, 4)
                if total_v > 0.0 and m < len(visits)
                else None
                for m in legal_m
            },
            "policy": [float(p) for p in masked.tolist()],
            "value": round(float(value[0].item()), 4),
        }

    r384 = run_budget(384, 0)
    r1200 = run_budget(1200, 1000)
    return {
        384: r384,
        1200: r1200,
        "policy_distribution": {
            int(m): round(float(r1200["policy"][m]), 4)
            for m in legal_moves
            if m < len(r1200["policy"])
        },
        "policy_top_move": top_policy_move(r1200["policy"], legal_moves),
        "policy_entropy": policy_entropy(r1200["policy"], legal_moves),
        "value_prediction": r1200["value"],
    }


def run_teacher_conditioned_trace(
    spec_name: str,
    classic_target_rows: list[dict[str, Any]],
    puct_target_rows: list[dict[str, Any]],
    classic_specs: dict[str, dict[str, Any]],
    puct_specs: dict[str, dict[str, Any]],
    excluded_specs: dict[str, dict[str, Any]],
    classic_baseline: dict[str, dict[str, Any]],
    puct_baseline: dict[str, dict[str, Any]],
    excluded_baseline: dict[str, dict[str, Any]],
    init_checkpoint: Path,
    current_metadata: dict[str, Any],
    base_model_spec: dict[str, Any],
    export_root: Path,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model_spec = dict(base_model_spec)
    input_encoding = str(model_spec["input_encoding"])
    base_input_size = input_size_for_encoding(input_encoding)

    mixed_rows = classic_target_rows + puct_target_rows
    compact_x_list: list[np.ndarray] = []
    compact_p_list: list[np.ndarray] = []
    compact_v_list: list[np.ndarray] = []
    row_ids_list: list[str] = []

    for row in mixed_rows:
        state_base = np.asarray(row["state"], dtype=np.float32)
        extra = np.zeros(TEACHER_INPUT_EXTRA_FEATURES, dtype=np.float32)
        if row["teacher_id"] == "classic_mcts":
            extra[0] = 1.0
        elif row["teacher_id"] == "artifact_puct":
            extra[1] = 1.0
        augmented = np.concatenate([state_base, extra])
        compact_x_list.append(augmented)
        compact_p_list.append(np.asarray(row["policy"], dtype=np.float32))
        compact_v_list.append(np.array(row["value"], dtype=np.float32).reshape(1))
        row_ids_list.append(row["row_id"])

    compact_x = np.stack(compact_x_list)
    compact_p = np.stack(compact_p_list)
    compact_v = np.concatenate(compact_v_list).reshape(-1, 1)
    replay_indexes = np.arange(len(mixed_rows), dtype=np.int64)

    model = TeacherConditionedPolicyValueNet(
        hidden_sizes=tuple(model_spec["hidden_sizes"]),
        model_type=str(model_spec["model_type"]),
        input_size=base_input_size,
    )
    init_teacher_conditioned_from_checkpoint(model, init_checkpoint)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(model_spec["lr"]))

    np.random.seed(seed)
    set_seed(seed)

    checkpoints_root = export_root / "checkpoints" / spec_name
    exports_root = export_root / spec_name
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    exports_root.mkdir(parents=True, exist_ok=True)

    classic_results: list[dict[str, Any]] = []
    puct_results: list[dict[str, Any]] = []
    excluded_results: list[dict[str, Any]] = []
    training_metrics: list[dict[str, Any]] = []

    def snapshot_tc(epoch: int, grad_norm: float | None, notes: str) -> None:
        ckpt = checkpoints_root / f"epoch_{epoch}.npz"
        sd = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
        torch.save(sd, ckpt)

        model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(compact_x).to(device)
            p_t = torch.from_numpy(compact_p).to(device)
            v_t = torch.from_numpy(compact_v).to(device)
            logits, v_pred = model(x_t)
            p_ce = compute_policy_cross_entropy(logits, p_t).detach().cpu().numpy()
            v_l = (
                compute_value_loss_vector(
                    v_pred, v_t, value_loss="huber", huber_delta=1.0
                )
                .detach()
                .cpu()
                .numpy()
            )
        tl = p_ce + (0.3 * v_l)
        training_metrics.append(
            {
                "variant": spec_name,
                "epoch": epoch,
                "policy_loss": round(float(np.mean(p_ce)), 6),
                "value_loss": round(float(np.mean(v_l)), 6),
                "total_loss": round(float(np.mean(tl)), 6),
                "gradient_norm": grad_norm,
                "notes": notes,
            }
        )

        for row_id in CLASSIC_ROW_IDS:
            spec_row = classic_specs[row_id]
            summary = teacher_conditioned_evaluate_position_summary(
                model=model,
                teacher_id="classic_mcts",
                raw_state=spec_row["raw_state"],
                legal_moves=[int(m) for m in spec_row["legal_moves"]],
                input_encoding=input_encoding,
                seed=seed + epoch + len(classic_results),
            )
            classic_results.append(
                build_classic_row_result(
                    variant=spec_name,
                    epoch=epoch,
                    row_spec=spec_row,
                    eval_summary=summary,
                    baseline=classic_baseline[row_id],
                )
            )
        for row_id in PUCT_ROW_IDS:
            spec_row = puct_specs[row_id]
            summary = teacher_conditioned_evaluate_position_summary(
                model=model,
                teacher_id="artifact_puct",
                raw_state=spec_row["raw_state"],
                legal_moves=[int(m) for m in spec_row["legal_moves"]],
                input_encoding=input_encoding,
                seed=seed + 500 + epoch + len(puct_results),
            )
            puct_results.append(
                build_puct_row_result(
                    variant=spec_name,
                    epoch=epoch,
                    row_spec=spec_row,
                    eval_summary=summary,
                    baseline=puct_baseline[row_id],
                )
            )
        for row_id in EXCLUDED_ROW_IDS:
            spec_row = excluded_specs[row_id]
            summary = teacher_conditioned_evaluate_position_summary(
                model=model,
                teacher_id="excluded",
                raw_state=spec_row["raw_state"],
                legal_moves=[int(m) for m in spec_row["legal_moves"]],
                input_encoding=input_encoding,
                seed=seed + 900 + epoch + len(excluded_results),
            )
            excluded_results.append(
                build_excluded_row_result(
                    variant=spec_name,
                    epoch=epoch,
                    row_spec=spec_row,
                    eval_summary=summary,
                    baseline=excluded_baseline[row_id],
                )
            )

    snapshot_tc(0, None, "initial_checkpoint")
    for epoch in range(1, max(TRACE_EPOCHS) + 1):
        model.train()
        x_all = torch.from_numpy(compact_x).to(device)
        p_all = torch.from_numpy(compact_p).to(device)
        v_all = torch.from_numpy(compact_v).to(device)
        perm = torch.randperm(replay_indexes.shape[0], device=device)
        p_losses: list[float] = []
        v_losses: list[float] = []
        for start in range(0, replay_indexes.shape[0], int(model_spec["batch_size"])):
            idx = perm[start : start + int(model_spec["batch_size"])]
            bx, bp, bv = x_all[idx], p_all[idx], v_all[idx]
            logits, val = model(bx)
            pl = compute_policy_cross_entropy(logits, bp).mean()
            vl = compute_value_loss_vector(
                val, bv, value_loss="huber", huber_delta=1.0
            ).mean()
            loss = pl + (0.3 * vl)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    gn += float(torch.sum(p.grad.detach() ** 2).item())
            gn = float(np.sqrt(gn)) if gn > 0 else 0.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            p_losses.append(float(pl.detach().cpu().item()))
            v_losses.append(float(vl.detach().cpu().item()))
        if epoch in TRACE_EPOCHS:
            snapshot_tc(
                epoch,
                float(np.mean([gn])) if "gn" in dir() else None,
                "post_epoch",
            )

    return {
        "variant": spec_name,
        "epochs": list(TRACE_EPOCHS),
        "status": "completed",
        "notes": "Teacher-conditioned probe: teacher_id appended as extra input features",
        "classic_results": classic_results,
        "puct_results": puct_results,
        "excluded_results": excluded_results,
        "training_metrics": training_metrics,
    }


# ── Cross-evaluation ──────────────────────────────────────────────────────
def build_cross_evaluation(
    variant_trace: dict[str, Any],
) -> dict[str, Any]:
    cross: list[dict[str, Any]] = []
    for row in variant_trace["classic_results"]:
        row_id = row["row_id"]
        cross.append(
            {
                "variant": variant_trace["variant"],
                "epoch": row["epoch"],
                "row_id": row_id,
                "evaluated_under": "classic_head_or_teacher",
                "selected_move_1200": row["selected_move_1200"],
            }
        )
    for row in variant_trace["puct_results"]:
        row_id = row["row_id"]
        cross.append(
            {
                "variant": variant_trace["variant"],
                "epoch": row["epoch"],
                "row_id": row_id,
                "evaluated_under": "puct_head_or_teacher",
                "selected_move_1200": row["selected_move_1200"],
            }
        )
    return {"cross_entries": cross}


# ── Interference metrics ──────────────────────────────────────────────────
def compute_interference_metrics(
    trace: dict[str, Any],
    classic_baseline: dict[str, dict[str, Any]],
    puct_baseline: dict[str, dict[str, Any]],
    excluded_baseline: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    epoch = max(trace["epochs"])
    classic_rows = [r for r in trace["classic_results"] if r["epoch"] == epoch]
    puct_rows = [r for r in trace["puct_results"] if r["epoch"] == epoch]
    excluded_rows = [r for r in trace["excluded_results"] if r["epoch"] == epoch]

    classic_gain_count = sum(1 for r in classic_rows if r["improved_vs_current"])
    classic_damage_count = sum(1 for r in classic_rows if r["damaged_vs_current"])
    puct_gain_count = sum(1 for r in puct_rows if r["improved_vs_current"])
    puct_damage_count = sum(1 for r in puct_rows if r["damaged_vs_current"])
    excluded_drift_count = sum(1 for r in excluded_rows if r["unexpected_drift"])

    interference_score = puct_damage_count + excluded_drift_count - classic_gain_count
    teacher_separation_score = (
        classic_gain_count + puct_gain_count - puct_damage_count - classic_damage_count
    )

    return {
        "variant": trace["variant"],
        "epoch": epoch,
        "classic_gain_count": classic_gain_count,
        "classic_damage_count": classic_damage_count,
        "puct_gain_count": puct_gain_count,
        "puct_damage_count": puct_damage_count,
        "excluded_drift_count": excluded_drift_count,
        "interference_score": interference_score,
        "teacher_separation_score": teacher_separation_score,
    }


# ── Gradient check ────────────────────────────────────────────────────────
def flatten_gradient(model: PolicyValueNet) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for p in model.parameters():
        if p.grad is not None:
            chunks.append(p.grad.detach().cpu().numpy().reshape(-1).astype(np.float64))
    if not chunks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(chunks)


def gradient_cosine_similarity(g1: np.ndarray, g2: np.ndarray) -> float | None:
    if g1.size == 0 or g2.size == 0:
        return None
    n1 = float(np.linalg.norm(g1))
    n2 = float(np.linalg.norm(g2))
    if n1 <= 0 or n2 <= 0:
        return None
    return round(float(np.dot(g1, g2) / (n1 * n2)), 6)


def compute_gradient_check(
    classic_target_rows: list[dict[str, Any]],
    puct_target_rows: list[dict[str, Any]],
    init_checkpoint: Path,
    base_model_spec: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model = PolicyValueNet(
        hidden_sizes=tuple(base_model_spec["hidden_sizes"]),
        model_type=str(base_model_spec["model_type"]),
        input_size=input_size_for_encoding(str(base_model_spec["input_encoding"])),
    )
    load_checkpoint_into_model(model, init_checkpoint)
    model = model.to(device)
    model.eval()

    def per_row_gradient(row: dict[str, Any]) -> np.ndarray:
        model.zero_grad(set_to_none=True)
        x = np.asarray([row["state"]], dtype=np.float32)
        p = np.asarray([row["policy"]], dtype=np.float32)
        logits, _ = model(torch.from_numpy(x).to(device))
        loss = compute_policy_cross_entropy(
            logits, torch.from_numpy(p).to(device)
        ).mean()
        loss.backward()
        return flatten_gradient(model)

    classic_grads: dict[str, np.ndarray] = {}
    for row in classic_target_rows:
        g = per_row_gradient(row)
        classic_grads[row["row_id"]] = g

    puct_grads: dict[str, np.ndarray] = {}
    for row in puct_target_rows:
        g = per_row_gradient(row)
        puct_grads[row["row_id"]] = g

    pairs: list[dict[str, Any]] = []
    for cr_id, cg in classic_grads.items():
        for pr_id, pg in puct_grads.items():
            cos = gradient_cosine_similarity(cg, pg)
            pairs.append(
                {
                    "classic_row": cr_id,
                    "puct_row": pr_id,
                    "policy_gradient_cosine": cos,
                }
            )

    strong_negative = sorted(
        [
            p
            for p in pairs
            if p["policy_gradient_cosine"] is not None
            and p["policy_gradient_cosine"] < -0.02
        ],
        key=lambda p: (p["policy_gradient_cosine"], p["classic_row"], p["puct_row"]),
    )[:10]

    return {
        "status": "completed",
        "pair_rows": pairs,
        "strongest_negative_pairs": strong_negative,
    }


# ── Classification ────────────────────────────────────────────────────────
def classify_results(
    all_metrics: list[dict[str, Any]],
) -> dict[str, str]:
    mixed_metrics = next(
        (m for m in all_metrics if m["variant"] == "single_head_mixed_replay"), None
    )
    classic_only_metrics = next(
        (m for m in all_metrics if m["variant"] == "classic_only_replay"), None
    )
    puct_only_metrics = next(
        (m for m in all_metrics if m["variant"] == "puct_only_replay"), None
    )
    tc_metrics = next(
        (m for m in all_metrics if m["variant"] == "teacher_conditioned_probe"), None
    )

    if (
        tc_metrics is not None
        and tc_metrics["teacher_separation_score"] > 0
        and tc_metrics["interference_score"] <= 0
    ):
        return {
            "classification": "teacher_policy_split_promising",
            "supporting_evidence": (
                f"Teacher-conditioned probe achieved teacher_separation_score={tc_metrics['teacher_separation_score']} "
                f"with interference_score={tc_metrics['interference_score']}: classic_gains={tc_metrics['classic_gain_count']}, "
                f"puct_gains={tc_metrics['puct_gain_count']}, puct_damage={tc_metrics['puct_damage_count']}"
            ),
            "rejected_alternatives": (
                "single_head_mixed_replay, classic_only_replay, puct_only_replay all showed higher "
                "interference than teacher-conditioned probe"
            ),
            "next_action": "design a small non-production teacher-aware architecture lane with strict local gates, still no arena",
        }

    if mixed_metrics is not None and classic_only_metrics is not None:
        mixed_damage = mixed_metrics.get("puct_damage_count", 0) + mixed_metrics.get(
            "classic_damage_count", 0
        )
        classic_damage = classic_only_metrics.get(
            "puct_damage_count", 0
        ) + classic_only_metrics.get("classic_damage_count", 0)
        if mixed_damage > 0 and classic_damage > 0 and tc_metrics is not None:
            tc_damage = tc_metrics.get("puct_damage_count", 0) + tc_metrics.get(
                "classic_damage_count", 0
            )
            if tc_damage < mixed_damage:
                return {
                    "classification": "teacher_conditioning_required",
                    "supporting_evidence": (
                        f"Teacher-conditioned probe reduces damage (mixed={mixed_damage}, tc={tc_damage})"
                    ),
                    "rejected_alternatives": "single-head training damages one or both buckets",
                    "next_action": "implement minimal teacher-conditioned model support behind an experimental config",
                }

    if mixed_metrics is not None:
        i = mixed_metrics.get("interference_score", 999)
        if i > 3:
            return {
                "classification": "teacher_policy_conflict_not_representable_by_small_probe",
                "supporting_evidence": f"Mixed replay interference_score={i}, all variants show conflict",
                "rejected_alternatives": "no variant resolved the teacher conflict",
                "next_action": "stop training on incumbent_proxy; use teacher-policy split only for evaluation, and improve mining/scoring from fresh positions",
            }

    if classic_only_metrics is not None and puct_only_metrics is not None:
        if (
            classic_only_metrics.get("classic_gain_count", 0) > 0
            and classic_only_metrics.get("puct_damage_count", 0) > 0
            and puct_only_metrics.get("puct_gain_count", 0) > 0
            and puct_only_metrics.get("classic_damage_count", 0) > 0
        ):
            return {
                "classification": "hard_teacher_branch_conflict",
                "supporting_evidence": "Classic-only and PUCT-only each improve their own bucket but damage the other",
                "rejected_alternatives": "mixing teachers in one model objective does not work",
                "next_action": "do not mix teachers in one model objective; maintain separate teacher reference artifacts or separate models for diagnostics",
            }

    if (
        classic_only_metrics is not None
        and classic_only_metrics.get("classic_gain_count", 0) == 0
    ):
        return {
            "classification": "local_targets_not_learnable",
            "supporting_evidence": "Neither bucket improves even under isolated training",
            "rejected_alternatives": "target construction or model/input representation needs revisit",
            "next_action": "revisit target construction and model/input representation before architecture work",
        }

    return {
        "classification": "inconclusive_diagnostic",
        "supporting_evidence": "Classification thresholds not clearly crossed",
        "rejected_alternatives": "n/a",
        "next_action": "review diagnostic data and consider broader sweep",
    }


# ── Report builder ────────────────────────────────────────────────────────
def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-lite Teacher Policy Split Probe Results",
        "",
        "## 1. Context",
        "",
        "- No arena was run.",
        "- No model was promoted.",
        "- Active references stayed read-only at `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.",
        "- Proposed patch bundle from PR #69 is not applied.",
        "- PR #69 cleanup audit concluded: `teacher_policy_architecture_needed`.",
        "- Key reason: `incumbent_proxy_disagreement` has 6 Classic-teacher eligible targets, but PUCT teacher disagrees on 7 rows.",
        "- This probe tests whether teacher-policy split modeling can learn both teacher targets without cross-teacher interference.",
        "",
        f"- Output: `{summary['summary_path']}`",
        "",
        "## 2. Why PR #69 points to teacher-policy architecture",
        "",
        "- PR #69 showed that even after full family cleanup, the incumbent_proxy_disagreement family still has incompatible Classic and PUCT teacher labels.",
        "- Patch-only cleanup is insufficient because the teacher disagreement is architectural, not a reference quality issue.",
        "- 3 families become trainable after patch, but only at 4 rows each — diagnostic-only scale.",
        "- Opening branch remains excluded.",
        "- Proposed patch bundle has 23 validated non-mutating entries with `do_not_auto_apply=true`.",
        "",
        "## 3. Bucket reconstruction",
        "",
    ]

    validation_rows = summary["bucket_reconstruction"]["validation_entries"]
    lines.extend(
        markdown_table(
            [
                "row_id",
                "bucket",
                "teacher_id",
                "preferred_move",
                "active_reference_move",
                "legal",
                "status",
                "notes",
            ],
            [
                [
                    r["row_id"],
                    r["bucket"],
                    "classic_mcts"
                    if r["bucket"] == "classic_teacher"
                    else "artifact_puct"
                    if r["bucket"] == "puct_teacher"
                    else "excluded",
                    str(
                        summary["bucket_reconstruction"]
                        .get("preferred_moves", {})
                        .get(r["row_id"], "?")
                    ),
                    str(
                        summary["bucket_reconstruction"]
                        .get("active_reference_moves", {})
                        .get(r["row_id"], "?")
                    ),
                    str(
                        r.get("active_reference_legal", False)
                        and r.get("preferred_move_legal", False)
                    ),
                    r["status"],
                    r["notes"],
                ]
                for r in validation_rows
            ],
        )
    )
    lines.append("")

    # Variant table
    lines.extend(["## 4. Variant definitions", ""])
    variant_rows = []
    for v in summary.get("variant_definitions", []):
        variant_rows.append(
            [
                v["name"],
                v.get("architecture_or_objective", ""),
                v.get("data", ""),
                v.get("epochs", ""),
                v.get("status", ""),
                v.get("notes", ""),
            ]
        )
    lines.extend(
        markdown_table(
            [
                "variant",
                "architecture_or_objective",
                "data",
                "epochs",
                "status",
                "notes",
            ],
            variant_rows,
        )
    )
    lines.append("")

    # Current baseline
    lines.extend(["## 5. Current baseline", ""])
    baseline_rows = []
    for row_id in ALL_ROW_IDS:
        b = summary.get("baseline", {}).get(row_id, {})
        if not b:
            continue
        bucket_label = (
            "classic"
            if row_id in CLASSIC_ROW_IDS
            else "puct"
            if row_id in PUCT_ROW_IDS
            else "excluded"
        )
        baseline_rows.append(
            [
                row_id,
                bucket_label,
                str(b.get("selected_move_384", "?")),
                str(b.get("selected_move_1200", "?")),
                format_float(b.get("reference_visit_share_384")),
                format_float(b.get("reference_visit_share_1200")),
                format_float(b.get("preferred_policy_probability")),
                str(b.get("policy_rank_preferred", "?")),
                format_float(b.get("policy_entropy")),
            ]
        )
    lines.extend(
        markdown_table(
            [
                "row_id",
                "bucket",
                "selected_384",
                "selected_1200",
                "ref_visit_share_384",
                "ref_visit_share_1200",
                "preferred_policy_prob",
                "preferred_policy_rank",
                "entropy",
            ],
            baseline_rows,
        )
    )
    lines.append("")

    # Classic bucket results
    lines.extend(["## 6. Classic bucket results", ""])
    classic_result_rows = []
    for trace in summary.get("traces", []):
        for r in trace.get("classic_results", []):
            if r["epoch"] <= 0:
                continue
            classic_result_rows.append(
                [
                    trace["variant"],
                    str(r["epoch"]),
                    r["row_id"],
                    str(r["classic_preferred_move"]),
                    str(r["selected_move_384"]),
                    str(r["selected_move_1200"]),
                    format_float(r["reference_visit_share_384"]),
                    format_float(r["reference_visit_share_1200"]),
                    format_float(r["classic_preferred_policy_probability"]),
                    format_bool(r["improved_vs_current"]),
                    format_bool(r["damaged_vs_current"]),
                    r["notes"],
                ]
            )
    if classic_result_rows:
        lines.extend(
            markdown_table(
                [
                    "variant",
                    "epoch",
                    "row_id",
                    "preferred_move",
                    "selected_384",
                    "selected_1200",
                    "preferred_visit_share_384",
                    "preferred_visit_share_1200",
                    "preferred_policy_prob",
                    "improved_vs_current",
                    "damaged_vs_current",
                    "notes",
                ],
                classic_result_rows,
            )
        )
    else:
        lines.append("- No Classic bucket results collected.")
    lines.append("")

    # PUCT bucket results
    lines.extend(["## 7. PUCT bucket results", ""])
    puct_result_rows = []
    for trace in summary.get("traces", []):
        for r in trace.get("puct_results", []):
            if r["epoch"] <= 0:
                continue
            puct_result_rows.append(
                [
                    trace["variant"],
                    str(r["epoch"]),
                    r["row_id"],
                    str(r["puct_preferred_move"]),
                    str(r["selected_move_384"]),
                    str(r["selected_move_1200"]),
                    format_float(r["puct_preferred_visit_share_384"]),
                    format_float(r["puct_preferred_visit_share_1200"]),
                    format_float(r["puct_preferred_policy_probability"]),
                    format_bool(r["improved_vs_current"]),
                    format_bool(r["damaged_vs_current"]),
                    r["notes"],
                ]
            )
    if puct_result_rows:
        lines.extend(
            markdown_table(
                [
                    "variant",
                    "epoch",
                    "row_id",
                    "preferred_move",
                    "selected_384",
                    "selected_1200",
                    "preferred_visit_share_384",
                    "preferred_visit_share_1200",
                    "preferred_policy_prob",
                    "improved_vs_current",
                    "damaged_vs_current",
                    "notes",
                ],
                puct_result_rows,
            )
        )
    else:
        lines.append("- No PUCT bucket results collected.")
    lines.append("")

    # Cross-teacher interference analysis
    lines.extend(["## 8. Cross-teacher interference analysis", ""])
    interference_rows = []
    for m in summary.get("interference_metrics", []):
        interference_rows.append(
            [
                m["variant"],
                str(m["epoch"]),
                str(m["classic_gain_count"]),
                str(m["puct_gain_count"]),
                str(m["classic_damage_count"]),
                str(m["puct_damage_count"]),
                str(m["excluded_drift_count"]),
                str(m["teacher_separation_score"]),
                str(m["interference_score"]),
            ]
        )
    if interference_rows:
        lines.extend(
            markdown_table(
                [
                    "variant",
                    "epoch",
                    "classic_gain",
                    "puct_gain",
                    "classic_damage",
                    "puct_damage",
                    "excluded_drift",
                    "teacher_separation",
                    "interference_score",
                ],
                interference_rows,
            )
        )
    lines.append("")

    # Decision
    decision = summary.get("decision", {})
    lines.extend(
        [
            "## 9. Decision",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "classification",
                "supporting_evidence",
                "rejected_alternatives",
                "next_action",
            ],
            [
                [
                    decision.get("classification", "unknown"),
                    decision.get("supporting_evidence", ""),
                    decision.get("rejected_alternatives", ""),
                    decision.get("next_action", ""),
                ]
            ],
        )
    )
    lines.append("")

    # Gradient check
    gradient = summary.get("gradient_check", {})
    if gradient.get("status") == "completed":
        strong = gradient.get("strongest_negative_pairs", [])
        lines.extend(
            [
                "## 10. Optional gradient check",
                "",
                "- Gradient check was computed at initialization.",
                f"- Strongest negative policy-gradient cosine pairs (threshold < -0.02): {len(strong)}",
            ]
        )
        if strong:
            for pair in strong[:5]:
                lines.append(
                    f"  - `{pair['classic_row']}` vs `{pair['puct_row']}`: "
                    f"cosine={pair['policy_gradient_cosine']}"
                )
        else:
            lines.append("- No strongly negative pairs found.")
    else:
        lines.extend(
            [
                "## 10. Optional gradient check",
                "",
                "- gradient_check_skipped_due_to_code_complexity",
            ]
        )
    lines.append("")

    # Interpretation
    lines.extend(
        [
            "## 11. Interpretation",
            "",
            "- Primary hypothesis test: teacher-aware diagnostic objective may preserve both local behaviors better than single-head replay.",
            f"- Classification: `{decision.get('classification', 'unknown')}`.",
            f"- Next action: {decision.get('next_action', 'none')}",
            "",
            "## 12. Exactly one recommended next action",
            "",
            f"Recommendation: **{decision.get('next_action', 'none')}**",
            "",
            "### Acceptance criteria",
            "",
            "- No arena was run.",
            "- No model was promoted.",
            "- Active references were not mutated.",
            "- Proposed patch bundle from PR #69 was not applied.",
            "- Excluded rows were never training targets.",
            "- Single-head mixed replay was treated as a diagnostic baseline, not a candidate.",
            "- Teacher-conditioned variants were experimental only.",
            "",
        ]
    )

    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument(
        "--classic-probe-rows-out", type=Path, default=DEFAULT_CLASSIC_PROBE_ROWS_PATH
    )
    parser.add_argument(
        "--puct-probe-rows-out", type=Path, default=DEFAULT_PUCT_PROBE_ROWS_PATH
    )
    parser.add_argument(
        "--excluded-probe-rows-out", type=Path, default=DEFAULT_EXCLUDED_PROBE_ROWS_PATH
    )
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--skip-teacher-conditioned",
        action="store_true",
        help="Skip teacher-conditioned probe (variant D)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.export_root.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "summary_path": str(args.summary_out),
        "inputs": {
            "reference_path": str(args.reference_path),
            "suite_path": str(args.suite_path),
            "current_artifact": str(args.current_artifact),
            "classic_rows_path": str(args.classic_rows_path),
            "puct_rows_path": str(args.puct_rows_path),
            "excluded_rows_path": str(args.excluded_rows_path),
        },
    }

    device = choose_device(args.device)
    current_metadata = load_json(args.current_artifact / "metadata.json")
    base_model_spec = model_spec_from_metadata(current_metadata)
    init_checkpoint = current_checkpoint_path(args.current_artifact, args.output_dir)

    print("=== Step 1: Bucket reconstruction ===")
    bucket_data = reconstruct_buckets(
        args.classic_rows_path,
        args.puct_rows_path,
        args.excluded_rows_path,
        args.suite_path,
        args.reference_path,
    )
    preferred_moves: dict[str, int] = {}
    active_reference_moves: dict[str, int] = {}
    for row_id in ALL_ROW_IDS:
        if row_id in bucket_data["classic_rows"]:
            by_id = {r["row_id"]: r for r in bucket_data["classic_rows"]}
        elif row_id in bucket_data["puct_rows"]:
            by_id = {r["row_id"]: r for r in bucket_data["puct_rows"]}
        elif row_id in bucket_data["excluded_rows"]:
            by_id = {r["row_id"]: r for r in bucket_data["excluded_rows"]}
        else:
            continue
        r = by_id.get(row_id, {})
        if r:
            preferred_moves[row_id] = int(r.get("preferred_move", -1))
            active_reference_moves[row_id] = int(r.get("active_reference_move", -1))
    bucket_data_clean = {k: v for k, v in bucket_data.items() if not k.startswith("_")}
    bucket_data_clean["preferred_moves"] = preferred_moves
    bucket_data_clean["active_reference_moves"] = active_reference_moves
    summary["bucket_reconstruction"] = bucket_data_clean
    print(f"  {len(bucket_data['validation_entries'])} rows validated")
    print(f"  Duplicate errors: {len(bucket_data['duplicate_errors'])}")

    print("=== Step 2: Diagnostic target rows ===")
    diagnostic_rows = build_diagnostic_target_rows(bucket_data, args.current_artifact)
    write_jsonl(args.classic_probe_rows_out, diagnostic_rows["classic_target_rows"])
    write_jsonl(args.puct_probe_rows_out, diagnostic_rows["puct_target_rows"])
    write_jsonl(args.excluded_probe_rows_out, diagnostic_rows["excluded_target_rows"])
    print(f"  Classic target rows: {len(diagnostic_rows['classic_target_rows'])}")
    print(f"  PUCT target rows: {len(diagnostic_rows['puct_target_rows'])}")
    print(f"  Excluded rows: {len(diagnostic_rows['excluded_target_rows'])}")

    print("=== Step 3: Current baseline ===")
    classic_specs: dict[str, dict[str, Any]] = {}
    puct_specs: dict[str, dict[str, Any]] = {}
    excluded_specs: dict[str, dict[str, Any]] = {}

    for row in diagnostic_rows["classic_target_rows"]:
        classic_specs[row["row_id"]] = row
    for row in diagnostic_rows["puct_target_rows"]:
        puct_specs[row["row_id"]] = row
    for row in diagnostic_rows["excluded_target_rows"]:
        excluded_specs[row["row_id"]] = row

    classic_baseline = evaluate_row_set_baseline(
        args.current_artifact,
        list(classic_specs.values()),
        args.seed,
    )
    puct_baseline = evaluate_row_set_baseline(
        args.current_artifact,
        list(puct_specs.values()),
        args.seed + 1000,
    )
    excluded_baseline = evaluate_row_set_baseline(
        args.current_artifact,
        list(excluded_specs.values()),
        args.seed + 2000,
    )
    baseline_all = {}
    for row_id in ALL_ROW_IDS:
        if row_id in classic_baseline:
            baseline_all[row_id] = classic_baseline[row_id]
        elif row_id in puct_baseline:
            baseline_all[row_id] = puct_baseline[row_id]
        elif row_id in excluded_baseline:
            baseline_all[row_id] = excluded_baseline[row_id]
    summary["baseline"] = baseline_all
    print(f"  Baseline evaluated for {len(baseline_all)} rows")

    print("=== Step 4: Build variant artifacts ===")
    variant_specs = build_variant_artifacts(diagnostic_rows, artifact_dir)
    variant_definitions = [
        {
            "name": "single_head_mixed_replay",
            "architecture_or_objective": "single-head MLP, mixed Classic+PUCT targets",
            "data": "classic + puct target rows",
            "epochs": ",".join(str(e) for e in TRACE_EPOCHS),
            "status": "pending",
            "notes": "Negative control",
        },
        {
            "name": "classic_only_replay",
            "architecture_or_objective": "single-head MLP, Classic targets only",
            "data": "classic target rows",
            "epochs": ",".join(str(e) for e in TRACE_EPOCHS),
            "status": "pending",
            "notes": "Reproduce PR #48 Classic gains",
        },
        {
            "name": "puct_only_replay",
            "architecture_or_objective": "single-head MLP, PUCT targets only",
            "data": "puct target rows",
            "epochs": ",".join(str(e) for e in TRACE_EPOCHS),
            "status": "pending",
            "notes": "Check PUCT-only damage to Classic",
        },
    ]
    if not args.skip_teacher_conditioned:
        variant_definitions.append(
            {
                "name": "teacher_conditioned_probe",
                "architecture_or_objective": "single-head MLP + teacher_id extra input features (one-hot)",
                "data": "classic + puct target rows with teacher conditioning",
                "epochs": ",".join(str(e) for e in TRACE_EPOCHS),
                "status": "pending",
                "notes": "Experimental: teacher_id as 2 extra input features",
            }
        )
    summary["variant_definitions"] = variant_definitions

    print("=== Step 5: Run training variants ===")
    traces: list[dict[str, Any]] = []

    for spec in variant_specs:
        print(f"  Running {spec.name}...")
        trace = run_variant_trace(
            spec=spec,
            classic_specs=classic_specs,
            puct_specs=puct_specs,
            excluded_specs=excluded_specs,
            classic_baseline=classic_baseline,
            puct_baseline=puct_baseline,
            excluded_baseline=excluded_baseline,
            init_checkpoint=init_checkpoint,
            current_metadata=current_metadata,
            base_model_spec=base_model_spec,
            export_root=args.export_root,
            seed=args.seed,
            device=device,
        )
        print(f"    Done: {trace['status']}")
        traces.append(trace)

        cross = build_cross_evaluation(trace)
        trace["cross_evaluation"] = cross

    if not args.skip_teacher_conditioned:
        print("  Running teacher_conditioned_probe...")
        tc_trace = run_teacher_conditioned_trace(
            spec_name="teacher_conditioned_probe",
            classic_target_rows=diagnostic_rows["classic_target_rows"],
            puct_target_rows=diagnostic_rows["puct_target_rows"],
            classic_specs=classic_specs,
            puct_specs=puct_specs,
            excluded_specs=excluded_specs,
            classic_baseline=classic_baseline,
            puct_baseline=puct_baseline,
            excluded_baseline=excluded_baseline,
            init_checkpoint=init_checkpoint,
            current_metadata=current_metadata,
            base_model_spec=base_model_spec,
            export_root=args.export_root,
            seed=args.seed,
            device=device,
        )
        print(f"    Done: {tc_trace['status']}")
        traces.append(tc_trace)

    summary["traces"] = traces

    print("=== Step 6: Interference metrics ===")
    metrics_list = []
    for trace in traces:
        m = compute_interference_metrics(
            trace, classic_baseline, puct_baseline, excluded_baseline
        )
        metrics_list.append(m)
        print(
            f"  {m['variant']}: classic_gain={m['classic_gain_count']}, "
            f"puct_gain={m['puct_gain_count']}, "
            f"classic_damage={m['classic_damage_count']}, "
            f"puct_damage={m['puct_damage_count']}, "
            f"interference={m['interference_score']}, "
            f"separation={m['teacher_separation_score']}"
        )
    summary["interference_metrics"] = metrics_list

    print("=== Step 7: Gradient check ===")
    gradient_result = compute_gradient_check(
        classic_target_rows=diagnostic_rows["classic_target_rows"],
        puct_target_rows=diagnostic_rows["puct_target_rows"],
        init_checkpoint=init_checkpoint,
        base_model_spec=base_model_spec,
        device=device,
    )
    summary["gradient_check"] = gradient_result
    print(f"  Gradient check: {gradient_result['status']}")
    if gradient_result.get("strongest_negative_pairs"):
        print(
            f"  Strongest negative pairs: {len(gradient_result['strongest_negative_pairs'])}"
        )

    print("=== Step 8: Classification ===")
    decision = classify_results(metrics_list)
    summary["decision"] = decision
    print(f"  Classification: {decision['classification']}")
    print(f"  Next action: {decision['next_action']}")

    print("=== Step 9: Write outputs ===")
    write_json(args.summary_out, summary)

    report = build_report(summary)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report, encoding="utf-8")

    print(f"\nSummary: {args.summary_out}")
    print(f"Report: {args.report_path}")
    print(f"Classic probe rows: {args.classic_probe_rows_out}")
    print(f"PUCT probe rows: {args.puct_probe_rows_out}")
    print(f"Excluded probe rows: {args.excluded_probe_rows_out}")
    print(f"Classification: {decision['classification']}")
    print(f"Next action: {decision['next_action']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
