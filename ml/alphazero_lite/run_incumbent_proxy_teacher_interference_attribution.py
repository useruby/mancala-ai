#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.build_incumbent_proxy_classic_teacher_diagnostic_artifact import (
    build_artifact as build_classic_artifact,
)
from ml.alphazero_lite.forensic_suite import canonical_state_key, load_suite
from ml.alphazero_lite.run_incumbent_proxy_classic_teacher_diagnostic_trace import (
    build_classic_row_result,
    build_excluded_row_result,
    choose_device,
    current_checkpoint_path,
    evaluate_dataset_metrics,
    evaluate_position_summary,
    evaluate_row_set_baseline,
    format_bool,
    format_float,
    markdown_table,
    metric_rank,
    model_spec_from_metadata,
    row_specs_from_bucket_rows,
)
from ml.alphazero_lite.run_incumbent_proxy_disagreement_value_backup_audit import (
    load_json,
    write_json,
)
from ml.alphazero_lite.run_incumbent_proxy_teacher_policy_decision_audit import (
    canonical_reference_rows,
    row_consequences,
)
from ml.alphazero_lite.self_play import encode_state
from ml.alphazero_lite.train import (
    PolicyValueNet,
    checkpoint_from_model,
    compute_policy_cross_entropy,
    input_size_for_encoding,
    load_checkpoint_into_model,
    load_jsonl_replay,
    set_seed,
    split_replay_positions_by_source_row,
    train_one_epoch,
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
DEFAULT_BUCKET_REPORT_PATH = Path(
    "docs/alphazero-lite-incumbent-proxy-disagreement-teacher-bucket-split-results.md"
)
DEFAULT_PR48_ARTIFACT_PATH = Path(
    "/tmp/azlite_incumbent_proxy_classic_teacher_diagnostic/classic_teacher_diagnostic_artifact.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/tmp/azlite_incumbent_proxy_teacher_interference_attribution"
)
DEFAULT_ARTIFACT_DIR = DEFAULT_OUTPUT_DIR / "artifacts"
DEFAULT_EXPORT_ROOT = DEFAULT_OUTPUT_DIR / "exports"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "interference_attribution_summary.json"
DEFAULT_SIMILARITY_PATH = DEFAULT_OUTPUT_DIR / "state_similarity_matrix.json"
DEFAULT_REPORT_PATH = Path(
    "docs/alphazero-lite-incumbent-proxy-teacher-interference-attribution-results.md"
)

SCHEMA = "azlite_incumbent_proxy_teacher_interference_attribution_v1"
SOURCE_NAME = "incumbent_proxy_teacher_interference_attribution"
TRAIN_SOURCE = "incumbent_proxy_teacher_interference_attribution"
FAMILY = "incumbent_proxy_disagreement"
CLASSIC_TARGET_ROW_IDS = (
    "incumbent_proxy_disagreement-014",
    "incumbent_proxy_disagreement-022",
    "incumbent_proxy_disagreement-024",
    "incumbent_proxy_disagreement-025",
    "incumbent_proxy_disagreement-035",
)
CLASSIC_CONTROL_ROW_IDS = ("incumbent_proxy_disagreement-008",)
CLASSIC_ROW_IDS = (*CLASSIC_CONTROL_ROW_IDS, *CLASSIC_TARGET_ROW_IDS)
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
REFERENCE_POLICY_MASS = 0.85
NON_REFERENCE_POLICY_MASS = 0.15
LOW_LR = 2.5e-4
TRACE_PHASE1_EPOCHS = (1, 2)
TRACE_PHASE2_EPOCHS = (1, 2, 4)
SINGLE_ROW_ID_BY_VARIANT = {
    "classic_only_014": "incumbent_proxy_disagreement-014",
    "classic_only_022": "incumbent_proxy_disagreement-022",
    "classic_only_024": "incumbent_proxy_disagreement-024",
    "classic_only_025": "incumbent_proxy_disagreement-025",
    "classic_only_035": "incumbent_proxy_disagreement-035",
    "classic_only_008": "incumbent_proxy_disagreement-008",
}
LEAVE_ONE_OUT_ID_BY_VARIANT = {
    "classic_without_014": "incumbent_proxy_disagreement-014",
    "classic_without_022": "incumbent_proxy_disagreement-022",
    "classic_without_024": "incumbent_proxy_disagreement-024",
    "classic_without_025": "incumbent_proxy_disagreement-025",
    "classic_without_035": "incumbent_proxy_disagreement-035",
    "classic_without_008": "incumbent_proxy_disagreement-008",
}


@dataclass(frozen=True)
class VariantSpec:
    artifact_name: str
    included_row_ids: tuple[str, ...]
    notes: str
    epochs: tuple[int, ...] = TRACE_PHASE1_EPOCHS
    lr: float | None = None
    value_loss_weight: float | None = None
    artifact_path_override: Path | None = None

    @property
    def artifact_path(self) -> Path:
        if self.artifact_path_override is not None:
            return self.artifact_path_override
        return DEFAULT_ARTIFACT_DIR / f"{self.artifact_name}.jsonl"


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
        "--bucket-report-path", type=Path, default=DEFAULT_BUCKET_REPORT_PATH
    )
    parser.add_argument(
        "--pr48-artifact-path", type=Path, default=DEFAULT_PR48_ARTIFACT_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--similarity-out", type=Path, default=DEFAULT_SIMILARITY_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path} contains non-object jsonl")
                rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def top_policy_move(policy: list[float], legal_moves: list[int]) -> int | None:
    if not legal_moves:
        return None
    return min(legal_moves, key=lambda move: (-float(policy[move]), int(move)))


def build_policy(legal_moves: list[int], reference_move: int) -> list[float]:
    policy = [0.0] * 6
    if reference_move not in legal_moves:
        raise ValueError(f"reference move {reference_move} illegal")
    if len(legal_moves) == 1:
        policy[reference_move] = 1.0
        return policy
    residual = NON_REFERENCE_POLICY_MASS / float(len(legal_moves) - 1)
    for move in legal_moves:
        policy[int(move)] = residual
    policy[int(reference_move)] = REFERENCE_POLICY_MASS
    return policy


def flatten_policy_gradient(model: PolicyValueNet) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        chunks.append(
            parameter.grad.detach().cpu().numpy().reshape(-1).astype(np.float64)
        )
    if not chunks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(chunks)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or right.size == 0:
        return None
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return round(float(np.dot(left, right) / (left_norm * right_norm)), 6)


def clone_artifact_row(
    row: dict[str, Any], *, base_artifact_source_path: Path = DEFAULT_PR48_ARTIFACT_PATH
) -> dict[str, Any]:
    cloned = copy.deepcopy(row)
    cloned["source"] = TRAIN_SOURCE
    cloned["train_only"] = True
    cloned["exclude_from_validation"] = True
    cloned["preferred_teacher"] = "classic_mcts"
    cloned["do_not_mix_with_puct_teacher"] = True
    cloned["source_artifacts"] = list(cloned.get("source_artifacts") or [])
    if str(base_artifact_source_path) not in cloned["source_artifacts"]:
        cloned["source_artifacts"].append(str(base_artifact_source_path))
    source_runs = list(cloned.get("source_runs") or [])
    if source_runs:
        source_runs[0]["kind"] = TRAIN_SOURCE
    cloned["source_runs"] = source_runs
    reasons = list(cloned.get("selection_reasons") or [])
    if SOURCE_NAME not in reasons:
        reasons.append(SOURCE_NAME)
    cloned["selection_reasons"] = reasons
    return cloned


def removed_row_id_from_leave_one_out_variant(artifact_name: str) -> str:
    try:
        return LEAVE_ONE_OUT_ID_BY_VARIANT[artifact_name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported leave-one-out artifact name: {artifact_name}"
        ) from exc


def build_variant_catalog() -> list[VariantSpec]:
    variants = [
        VariantSpec(
            artifact_name="classic_all",
            included_row_ids=CLASSIC_ROW_IDS,
            notes="same rows as PR #48 full Classic artifact",
        ),
        VariantSpec(
            artifact_name="classic_without_014",
            included_row_ids=tuple(
                row_id
                for row_id in CLASSIC_ROW_IDS
                if row_id != "incumbent_proxy_disagreement-014"
            ),
            notes="leave out Classic target row 014",
        ),
        VariantSpec(
            artifact_name="classic_without_022",
            included_row_ids=tuple(
                row_id
                for row_id in CLASSIC_ROW_IDS
                if row_id != "incumbent_proxy_disagreement-022"
            ),
            notes="leave out Classic target row 022",
        ),
        VariantSpec(
            artifact_name="classic_without_024",
            included_row_ids=tuple(
                row_id
                for row_id in CLASSIC_ROW_IDS
                if row_id != "incumbent_proxy_disagreement-024"
            ),
            notes="leave out Classic target row 024",
        ),
        VariantSpec(
            artifact_name="classic_without_025",
            included_row_ids=tuple(
                row_id
                for row_id in CLASSIC_ROW_IDS
                if row_id != "incumbent_proxy_disagreement-025"
            ),
            notes="leave out Classic target row 025",
        ),
        VariantSpec(
            artifact_name="classic_without_035",
            included_row_ids=tuple(
                row_id
                for row_id in CLASSIC_ROW_IDS
                if row_id != "incumbent_proxy_disagreement-035"
            ),
            notes="leave out Classic target row 035",
        ),
        VariantSpec(
            artifact_name="classic_without_008",
            included_row_ids=tuple(
                row_id
                for row_id in CLASSIC_ROW_IDS
                if row_id != "incumbent_proxy_disagreement-008"
            ),
            notes="leave out the preservation control 008",
        ),
        VariantSpec(
            artifact_name="classic_only_014",
            included_row_ids=("incumbent_proxy_disagreement-014",),
            notes="single-row Classic target 014 only",
        ),
        VariantSpec(
            artifact_name="classic_only_022",
            included_row_ids=("incumbent_proxy_disagreement-022",),
            notes="single-row Classic target 022 only",
        ),
        VariantSpec(
            artifact_name="classic_only_024",
            included_row_ids=("incumbent_proxy_disagreement-024",),
            notes="single-row Classic target 024 only",
        ),
        VariantSpec(
            artifact_name="classic_only_025",
            included_row_ids=("incumbent_proxy_disagreement-025",),
            notes="single-row Classic target 025 only",
        ),
        VariantSpec(
            artifact_name="classic_only_035",
            included_row_ids=("incumbent_proxy_disagreement-035",),
            notes="single-row Classic target 035 only",
        ),
        VariantSpec(
            artifact_name="classic_only_008",
            included_row_ids=("incumbent_proxy_disagreement-008",),
            notes="preservation control 008 only",
        ),
        VariantSpec(
            artifact_name="target_only",
            included_row_ids=CLASSIC_TARGET_ROW_IDS,
            notes="all five Classic target rows without preservation control",
        ),
        VariantSpec(
            artifact_name="preservation_only",
            included_row_ids=CLASSIC_CONTROL_ROW_IDS,
            notes="preservation control only",
        ),
        VariantSpec(
            artifact_name="classic_all_low_lr",
            included_row_ids=CLASSIC_ROW_IDS,
            notes="same rows as classic_all with lower learning rate",
            lr=LOW_LR,
        ),
        VariantSpec(
            artifact_name="best_leave_one_out_low_lr",
            included_row_ids=(),
            notes="filled in after phase-1 selection using lower learning rate",
            lr=LOW_LR,
        ),
        VariantSpec(
            artifact_name="classic_all_policy_only",
            included_row_ids=CLASSIC_ROW_IDS,
            notes="same rows as classic_all with value loss weight forced to 0.0",
            value_loss_weight=0.0,
        ),
    ]
    return variants


def validate_bucket_rows(
    *,
    suite_path: Path,
    reference_path: Path,
    classic_rows: list[dict[str, Any]],
    puct_rows: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
    base_artifact_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    suite = load_suite(suite_path)
    suite_by_id = {position.id: position for position in suite}
    reference_payload = load_json(reference_path)
    reference_by_id = canonical_reference_rows(reference_payload)

    expected_ids = {
        "classic": set(CLASSIC_ROW_IDS),
        "puct": set(PUCT_ROW_IDS),
        "excluded": set(EXCLUDED_ROW_IDS),
    }
    actual_ids = {
        "classic": {str(row["row_id"]) for row in classic_rows},
        "puct": {str(row["row_id"]) for row in puct_rows},
        "excluded": {str(row["row_id"]) for row in excluded_rows},
    }
    mismatches = {
        bucket: sorted(expected_ids[bucket] ^ actual_ids[bucket])
        for bucket in expected_ids
        if expected_ids[bucket] != actual_ids[bucket]
    }
    if mismatches:
        raise ValueError(f"bucket row mismatch: {mismatches}")

    all_bucket_rows = [
        *[("classic_teacher", row) for row in classic_rows],
        *[("puct_teacher", row) for row in puct_rows],
        *[("excluded_diagnostic", row) for row in excluded_rows],
    ]
    validation_rows: list[dict[str, Any]] = []
    canonical_roles: dict[str, set[str]] = {}
    canonical_bucket_ids: dict[str, list[str]] = {}
    for bucket_name, row in all_bucket_rows:
        row_id = str(row["row_id"])
        suite_row = suite_by_id.get(row_id)
        reference_row = reference_by_id.get(row_id)
        canonical_state = (
            None if suite_row is None else canonical_state_key(suite_row.state)
        )
        active_reference_legal = False
        if suite_row is not None and reference_row is not None:
            active_reference_legal = int(reference_row["reference_move"]) in [
                int(move) for move in suite_row.legal_moves
            ]
        if canonical_state is not None:
            canonical_roles.setdefault(canonical_state, set()).add(bucket_name)
            canonical_bucket_ids.setdefault(canonical_state, []).append(row_id)
        validation_rows.append(
            {
                "row_id": row_id,
                "bucket": bucket_name,
                "exists_in_suite": suite_row is not None,
                "exists_in_reference": reference_row is not None,
                "active_reference_legal": active_reference_legal,
                "canonical_state": canonical_state,
                "status": "ok"
                if suite_row is not None
                and reference_row is not None
                and active_reference_legal
                else "invalid",
            }
        )

    conflicting_canonical_roles = {
        canonical: sorted(list(roles))
        for canonical, roles in canonical_roles.items()
        if len(roles) > 1
    }

    base_artifact_ids = {
        str((row.get("source_runs") or [{}])[0].get("id") or row.get("row_id"))
        for row in base_artifact_rows
    }
    puct_in_base = sorted(base_artifact_ids & set(PUCT_ROW_IDS))
    excluded_in_base = sorted(base_artifact_ids & set(EXCLUDED_ROW_IDS))
    status = "ok"
    if conflicting_canonical_roles or puct_in_base or excluded_in_base:
        status = "invalid"

    return {
        "status": status,
        "validation_rows": validation_rows,
        "conflicting_canonical_roles": conflicting_canonical_roles,
        "conflicting_canonical_row_ids": {
            canonical: sorted(ids)
            for canonical, ids in canonical_bucket_ids.items()
            if canonical in conflicting_canonical_roles
        },
        "puct_rows_in_classic_artifact": puct_in_base,
        "excluded_rows_in_classic_artifact": excluded_in_base,
    }


def consequence_signature(row_spec: dict[str, Any]) -> list[str]:
    legal_moves = [int(move) for move in row_spec["legal_moves"]]
    consequence_rows = row_consequences(dict(row_spec["suite_state"]), legal_moves)
    by_move = {int(item["move"]): item for item in consequence_rows}
    signature: list[str] = []
    for move in range(6):
        if move not in by_move:
            signature.append("illegal")
            continue
        item = by_move[move]
        signature.append(
            "|".join(
                [
                    str(int(item["gives_extra_turn"])),
                    str(int(item["produces_capture"])),
                    str(int(item["immediate_store_delta"])),
                    str(int(item["side_to_move_after"])),
                ]
            )
        )
    return signature


def similarity_components(
    classic_spec: dict[str, Any], puct_spec: dict[str, Any]
) -> dict[str, Any]:
    classic_encoded = np.asarray(
        encode_state(dict(classic_spec["suite_state"]), input_encoding="kalah_v3"),
        dtype=np.float64,
    )
    puct_encoded = np.asarray(
        encode_state(dict(puct_spec["suite_state"]), input_encoding="kalah_v3"),
        dtype=np.float64,
    )
    classic_seed_vector = np.asarray(
        [
            *classic_spec["suite_state"]["player_pits"],
            *classic_spec["suite_state"]["opponent_pits"],
            classic_spec["suite_state"]["player_store"],
            classic_spec["suite_state"]["opponent_store"],
        ],
        dtype=np.float64,
    )
    puct_seed_vector = np.asarray(
        [
            *puct_spec["suite_state"]["player_pits"],
            *puct_spec["suite_state"]["opponent_pits"],
            puct_spec["suite_state"]["player_store"],
            puct_spec["suite_state"]["opponent_store"],
        ],
        dtype=np.float64,
    )
    classic_legal = {int(move) for move in classic_spec["legal_moves"]}
    puct_legal = {int(move) for move in puct_spec["legal_moves"]}
    legal_match_count = sum(
        1 for move in range(6) if ((move in classic_legal) == (move in puct_legal))
    )
    classic_signature = consequence_signature(classic_spec)
    puct_signature = consequence_signature(puct_spec)
    consequence_match_count = sum(
        1
        for classic_item, puct_item in zip(classic_signature, puct_signature)
        if classic_item == puct_item
    )
    shared_legal_mask = round(legal_match_count / 6.0, 4)
    shared_consequence_pattern = round(consequence_match_count / 6.0, 4)
    encoded_distance = round(float(np.linalg.norm(classic_encoded - puct_encoded)), 6)
    seed_l1_distance = round(
        float(np.abs(classic_seed_vector - puct_seed_vector).sum()), 6
    )
    current_player_match = int(classic_spec["suite_state"]["current_player"]) == int(
        puct_spec["suite_state"]["current_player"]
    )
    combined_distance = round(
        encoded_distance
        + (1.0 - shared_legal_mask)
        + (1.0 - shared_consequence_pattern)
        + (0.0 if current_player_match else 0.5)
        + (seed_l1_distance / 100.0),
        6,
    )
    return {
        "encoded_distance": encoded_distance,
        "seed_l1_distance": seed_l1_distance,
        "current_player_match": current_player_match,
        "shared_legal_mask": shared_legal_mask,
        "shared_consequence_pattern": shared_consequence_pattern,
        "classic_consequence_signature": classic_signature,
        "puct_consequence_signature": puct_signature,
        "combined_distance": combined_distance,
    }


def build_similarity_matrix(
    *,
    classic_specs: list[dict[str, Any]],
    puct_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    pair_rows: list[dict[str, Any]] = []
    for classic_spec in classic_specs:
        for puct_spec in puct_specs:
            components = similarity_components(classic_spec, puct_spec)
            pair_rows.append(
                {
                    "classic_row": classic_spec["row_id"],
                    "puct_row": puct_spec["row_id"],
                    **components,
                }
            )

    nearest_puct_by_classic: dict[str, list[dict[str, Any]]] = {}
    nearest_classic_by_puct: dict[str, list[dict[str, Any]]] = {}
    for classic_row_id in CLASSIC_ROW_IDS:
        nearest_puct_by_classic[classic_row_id] = sorted(
            [row for row in pair_rows if row["classic_row"] == classic_row_id],
            key=lambda row: (row["combined_distance"], row["puct_row"]),
        )[:3]
    for puct_row_id in PUCT_ROW_IDS:
        nearest_classic_by_puct[puct_row_id] = sorted(
            [row for row in pair_rows if row["puct_row"] == puct_row_id],
            key=lambda row: (row["combined_distance"], row["classic_row"]),
        )[:3]

    return {
        "schema": "azlite_incumbent_proxy_teacher_interference_similarity_v1",
        "pair_rows": pair_rows,
        "nearest_puct_by_classic": nearest_puct_by_classic,
        "nearest_classic_by_puct": nearest_classic_by_puct,
    }


def load_or_build_base_artifact(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.pr48_artifact_path.exists():
        rows = read_jsonl(args.pr48_artifact_path)
    else:
        built = build_classic_artifact(
            argparse.Namespace(
                reference_path=args.reference_path,
                suite_path=args.suite_path,
                current_artifact=args.current_artifact,
                classic_rows_path=args.classic_rows_path,
                puct_rows_path=args.puct_rows_path,
                excluded_rows_path=args.excluded_rows_path,
                bucket_summary_path=Path(),
                report_path=args.bucket_report_path,
                output_dir=args.output_dir,
                artifact_out=args.output_dir / "rebuilt_pr48_artifact.jsonl",
                summary_out=args.output_dir / "rebuilt_pr48_artifact_summary.json",
                target_only_out=args.output_dir / "rebuilt_pr48_targets.jsonl",
                control_only_out=args.output_dir / "rebuilt_pr48_controls.jsonl",
                input_encoding="kalah_v3",
                policy_target_mode="default",
                value_target_mode="default",
            )
        )
        rows = built["artifact_rows"]

    row_ids = [
        str((row.get("source_runs") or [{}])[0].get("id") or row.get("row_id"))
        for row in rows
    ]
    if sorted(row_ids) != sorted(CLASSIC_ROW_IDS):
        raise ValueError(f"unexpected PR #48 artifact rows: {row_ids}")
    return rows


def materialize_variant_artifacts(
    *,
    base_artifact_rows: list[dict[str, Any]],
    base_artifact_source_path: Path,
    artifact_dir: Path,
    variant_specs: list[VariantSpec],
    selected_best_leave_one_out: str | None,
) -> list[dict[str, Any]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    base_by_id = {
        str((row.get("source_runs") or [{}])[0].get("id") or row.get("row_id")): row
        for row in base_artifact_rows
    }
    materialized: list[dict[str, Any]] = []
    for spec in variant_specs:
        included_ids = spec.included_row_ids
        notes = spec.notes
        if spec.artifact_name == "best_leave_one_out_low_lr":
            if not selected_best_leave_one_out:
                continue
            removed_row_id = removed_row_id_from_leave_one_out_variant(
                selected_best_leave_one_out
            )
            included_ids = tuple(
                row_id for row_id in CLASSIC_ROW_IDS if row_id != removed_row_id
            )
            notes = (
                f"same rows as {selected_best_leave_one_out} with lower learning rate"
            )
        path = (
            spec.artifact_path_override or artifact_dir / f"{spec.artifact_name}.jsonl"
        )
        rows = [
            clone_artifact_row(
                base_by_id[row_id], base_artifact_source_path=base_artifact_source_path
            )
            for row_id in included_ids
        ]
        write_jsonl(path, rows)
        materialized.append(
            {
                "artifact_name": spec.artifact_name,
                "artifact_path": str(path),
                "included_classic_rows": list(included_ids),
                "excluded_classic_rows": [
                    row_id
                    for row_id in CLASSIC_ROW_IDS
                    if row_id not in set(included_ids)
                ],
                "row_count": len(rows),
                "status": "completed",
                "notes": notes,
                "lr": spec.lr,
                "value_loss_weight": spec.value_loss_weight,
            }
        )
    materialized.sort(key=lambda row: row["artifact_name"])
    return materialized


def build_puct_row_result(
    *,
    artifact_name: str,
    epoch: int,
    row_spec: dict[str, Any],
    eval_summary: dict[str, Any],
    baseline_row: dict[str, Any],
) -> dict[str, Any]:
    puct_move = int(row_spec["preferred_move"])
    summary_384 = eval_summary[384]
    summary_1200 = eval_summary[1200]
    current_row = {
        "selected_equals_puct_preferred_384": summary_384["selected_move"] == puct_move,
        "selected_equals_puct_preferred_1200": summary_1200["selected_move"]
        == puct_move,
        "puct_preferred_visit_share_384": summary_384["visit_shares"].get(puct_move),
        "puct_preferred_visit_share_1200": summary_1200["visit_shares"].get(puct_move),
    }
    reasons: list[str] = []
    if bool(baseline_row["selected_equals_puct_preferred_1200"]) and not bool(
        current_row["selected_equals_puct_preferred_1200"]
    ):
        reasons.append("lost_puct_selection_1200")
    if bool(baseline_row["selected_equals_puct_preferred_384"]) and not bool(
        current_row["selected_equals_puct_preferred_384"]
    ):
        reasons.append("lost_puct_selection_384")
    for budget in (384, 1200):
        baseline_share = baseline_row.get(f"puct_preferred_visit_share_{budget}")
        current_share = current_row.get(f"puct_preferred_visit_share_{budget}")
        if (
            baseline_share is not None
            and current_share is not None
            and (current_share - baseline_share) < -0.2
        ):
            reasons.append(f"large_preferred_share_drop_{budget}")
    distribution = dict(eval_summary["policy_distribution"])
    return {
        "artifact_name": artifact_name,
        "trace_name": artifact_name,
        "epoch": epoch,
        "row_id": row_spec["row_id"],
        "puct_preferred_move": puct_move,
        "selected_move_384": summary_384["selected_move"],
        "selected_move_1200": summary_1200["selected_move"],
        "selected_equals_puct_preferred_384": current_row[
            "selected_equals_puct_preferred_384"
        ],
        "selected_equals_puct_preferred_1200": current_row[
            "selected_equals_puct_preferred_1200"
        ],
        "puct_preferred_visit_share_384": current_row["puct_preferred_visit_share_384"],
        "puct_preferred_visit_share_1200": current_row[
            "puct_preferred_visit_share_1200"
        ],
        "puct_preferred_policy_probability": distribution.get(puct_move),
        "puct_preferred_policy_rank": metric_rank(distribution, puct_move),
        "cross_teacher_regression": bool(reasons),
        "notes": ",".join(reasons) if reasons else "no_heavy_puct_regression",
    }


def run_variant_trace(
    *,
    spec: VariantSpec,
    classic_spec_by_id: dict[str, dict[str, Any]],
    puct_spec_by_id: dict[str, dict[str, Any]],
    excluded_spec_by_id: dict[str, dict[str, Any]],
    classic_baseline_by_id: dict[str, dict[str, Any]],
    puct_baseline_by_id: dict[str, dict[str, Any]],
    excluded_baseline_by_id: dict[str, dict[str, Any]],
    init_checkpoint: Path,
    current_metadata: dict[str, Any],
    base_model_spec: dict[str, Any],
    export_root: Path,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    compact_x, compact_p, compact_v, replay_indexes = load_jsonl_replay(
        [spec.artifact_path],
        [1],
        policy_target_mode="default",
        value_target_mode="default",
    )
    model_spec = dict(base_model_spec)
    if spec.lr is not None:
        model_spec["lr"] = float(spec.lr)
    if spec.value_loss_weight is not None:
        model_spec["value_loss_weight"] = float(spec.value_loss_weight)

    row_infos = []
    for row in read_jsonl(spec.artifact_path):
        row_id = str((row.get("source_runs") or [{}])[0].get("id") or row.get("row_id"))
        row_infos.append({"row_id": row_id, "role": row.get("role")})

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

    checkpoints_root = export_root / "checkpoints" / spec.artifact_name
    exports_root = export_root / spec.artifact_name
    if checkpoints_root.exists():
        shutil.rmtree(checkpoints_root)
    if exports_root.exists():
        shutil.rmtree(exports_root)
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    exports_root.mkdir(parents=True, exist_ok=True)

    classic_rows: list[dict[str, Any]] = []
    puct_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    training_metric_rows: list[dict[str, Any]] = []
    export_dirs: dict[int, str] = {}
    evaluator_cache: dict[Path, ArtifactEvaluator] = {}

    def snapshot(epoch: int, gradient_norm: float | None, notes: str) -> None:
        checkpoint_path = checkpoints_root / f"epoch_{epoch}.npz"
        np.savez(checkpoint_path, **checkpoint_from_model(model))
        export_dir = exports_root / f"epoch_{epoch}"
        export_dir.mkdir(parents=True, exist_ok=True)
        if (export_dir / "model.npz").exists():
            shutil.rmtree(export_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path, export_dir / "model.npz")
        with np.load(export_dir / "model.npz") as checkpoint:
            weights_payload = {
                key: checkpoint[key].tolist() for key in checkpoint.files
            }
        (export_dir / "weights.json").write_text(
            json.dumps(weights_payload), encoding="utf-8"
        )
        metadata = {
            "schema_version": current_metadata.get("schema_version", "azlite_model_v1"),
            "version": f"{spec.artifact_name}-epoch-{epoch}",
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
        export_dirs[epoch] = str(export_dir)

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
                "artifact_name": spec.artifact_name,
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
                "lr": float(model_spec["lr"]),
                "value_loss_weight": float(model_spec["value_loss_weight"]),
            }
        )

        for row_id in CLASSIC_ROW_IDS:
            row_spec = classic_spec_by_id[row_id]
            summary = evaluate_position_summary(
                artifact_path=export_dir,
                state=dict(row_spec["suite_state"]),
                legal_moves=[int(move) for move in row_spec["legal_moves"]],
                seed=seed + epoch + len(classic_rows),
                evaluator_cache=evaluator_cache,
            )
            row = build_classic_row_result(
                trace_name=spec.artifact_name,
                epoch=epoch,
                row_spec=row_spec,
                eval_summary=summary,
                baseline_row=classic_baseline_by_id[row_id],
            )
            row["artifact_name"] = spec.artifact_name
            classic_rows.append(row)

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
                    artifact_name=spec.artifact_name,
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
            row = build_excluded_row_result(
                trace_name=spec.artifact_name,
                epoch=epoch,
                row_spec=row_spec,
                eval_summary=summary,
                baseline_row=excluded_baseline_by_id[row_id],
            )
            row["artifact_name"] = spec.artifact_name
            excluded_rows.append(row)

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
        "artifact_name": spec.artifact_name,
        "artifact_path": str(spec.artifact_path),
        "epochs": list(spec.epochs),
        "status": "completed",
        "notes": spec.notes,
        "lr": float(model_spec["lr"]),
        "value_loss_weight": float(model_spec["value_loss_weight"]),
        "export_dirs": export_dirs,
        "classic_rows": classic_rows,
        "puct_rows": puct_rows,
        "excluded_rows": excluded_rows,
        "training_metric_rows": training_metric_rows,
    }


def latest_epoch(trace: dict[str, Any]) -> int:
    return max(int(epoch) for epoch in trace["epochs"])


def rows_for_epoch(rows: list[dict[str, Any]], epoch: int) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["epoch"]) == int(epoch)]


def attribution_row_for_trace(trace: dict[str, Any], epoch: int) -> dict[str, Any]:
    classic_rows = rows_for_epoch(trace["classic_rows"], epoch)
    puct_rows = rows_for_epoch(trace["puct_rows"], epoch)
    excluded_rows = rows_for_epoch(trace["excluded_rows"], epoch)
    target_rows = [
        row for row in classic_rows if row["row_id"] in CLASSIC_TARGET_ROW_IDS
    ]
    classic_gain_count = sum(
        1 for row in target_rows if bool(row["improved_vs_current"])
    )
    classic_strict_pass_count = sum(
        1 for row in classic_rows if bool(row["strict_pass"])
    )
    classic_visit_share_deltas = []
    for row in classic_rows:
        baseline_share = trace["classic_baseline_by_id"][row["row_id"]].get(
            "reference_visit_share_1200"
        )
        current_share = row.get("reference_visit_share_1200")
        if baseline_share is not None and current_share is not None:
            classic_visit_share_deltas.append(
                float(current_share) - float(baseline_share)
            )
    avg_classic_reference_visit_share_delta = (
        round(float(np.mean(classic_visit_share_deltas)), 4)
        if classic_visit_share_deltas
        else None
    )
    lost_384 = sum(
        1 for row in puct_rows if not bool(row["selected_equals_puct_preferred_384"])
    )
    lost_1200 = sum(
        1 for row in puct_rows if not bool(row["selected_equals_puct_preferred_1200"])
    )
    heavy_regressions = sum(
        1 for row in puct_rows if bool(row["cross_teacher_regression"])
    )
    puct_visit_deltas = [
        float(row["puct_preferred_visit_share_1200"])
        - float(
            trace["puct_baseline_by_id"][row["row_id"]][
                "puct_preferred_visit_share_1200"
            ]
        )
        for row in puct_rows
        if row.get("puct_preferred_visit_share_1200") is not None
        and trace["puct_baseline_by_id"][row["row_id"]].get(
            "puct_preferred_visit_share_1200"
        )
        is not None
    ]
    avg_puct_preferred_visit_share_delta = (
        round(float(np.mean(puct_visit_deltas)), 4) if puct_visit_deltas else None
    )
    excluded_drift_count = sum(
        1 for row in excluded_rows if bool(row["unexpected_drift"])
    )
    interference_score = int(
        lost_1200 + heavy_regressions + excluded_drift_count - classic_gain_count
    )
    return {
        "artifact_name": trace["artifact_name"],
        "epoch": epoch,
        "classic_gain_count": classic_gain_count,
        "avg_classic_reference_visit_share_delta": avg_classic_reference_visit_share_delta,
        "classic_strict_pass_count": classic_strict_pass_count,
        "puct_lost_selection_384_count": lost_384,
        "puct_lost_selection_1200_count": lost_1200,
        "avg_puct_preferred_visit_share_delta": avg_puct_preferred_visit_share_delta,
        "heavy_regression_count": heavy_regressions,
        "excluded_drift_count": excluded_drift_count,
        "interference_score": interference_score,
    }


def epoch2_attribution_map(
    traces: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        name: attribution_row_for_trace(trace, 2)
        for name, trace in traces.items()
        if 2 in set(int(epoch) for epoch in trace["epochs"])
    }


def select_best_leave_one_out(epoch2_rows: dict[str, dict[str, Any]]) -> str:
    candidates = [
        row for name, row in epoch2_rows.items() if name.startswith("classic_without_")
    ]
    if not candidates:
        raise ValueError("no leave-one-out candidates found")
    best = min(
        candidates,
        key=lambda row: (
            row["interference_score"],
            row["puct_lost_selection_1200_count"],
            row["excluded_drift_count"],
            -row["classic_gain_count"],
            row["artifact_name"],
        ),
    )
    return str(best["artifact_name"])


def select_worst_single_row(epoch2_rows: dict[str, dict[str, Any]]) -> str:
    candidates = [
        row for name, row in epoch2_rows.items() if name.startswith("classic_only_")
    ]
    if not candidates:
        raise ValueError("no single-row candidates found")
    worst = max(
        candidates,
        key=lambda row: (
            row["interference_score"],
            row["puct_lost_selection_1200_count"],
            row["heavy_regression_count"],
            -row["classic_gain_count"],
            row["artifact_name"],
        ),
    )
    return str(worst["artifact_name"])


def per_row_policy_gradient(
    *,
    model: PolicyValueNet,
    device: torch.device,
    row_spec: dict[str, Any],
    target_move: int,
) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    encoded = np.asarray(
        [encode_state(dict(row_spec["suite_state"]), input_encoding="kalah_v3")],
        dtype=np.float32,
    )
    policy = np.asarray(
        [
            build_policy(
                [int(move) for move in row_spec["legal_moves"]], int(target_move)
            )
        ],
        dtype=np.float32,
    )
    logits, _ = model(torch.from_numpy(encoded).to(device))
    loss = compute_policy_cross_entropy(
        logits, torch.from_numpy(policy).to(device)
    ).mean()
    loss.backward()
    return flatten_policy_gradient(model)


def compute_gradient_attribution(
    *,
    init_checkpoint: Path,
    current_metadata: dict[str, Any],
    device: torch.device,
    classic_spec_by_id: dict[str, dict[str, Any]],
    puct_spec_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    model_spec = model_spec_from_metadata(current_metadata)
    model = PolicyValueNet(
        hidden_sizes=tuple(model_spec["hidden_sizes"]),
        model_type=str(model_spec["model_type"]),
        input_size=input_size_for_encoding(str(model_spec["input_encoding"])),
    )
    load_checkpoint_into_model(model, init_checkpoint)
    model = model.to(device)
    model.eval()

    classic_gradients = {
        row_id: per_row_policy_gradient(
            model=model,
            device=device,
            row_spec=classic_spec_by_id[row_id],
            target_move=int(classic_spec_by_id[row_id]["active_reference_move"]),
        )
        for row_id in CLASSIC_TARGET_ROW_IDS
    }
    puct_preferred_gradients = {
        row_id: per_row_policy_gradient(
            model=model,
            device=device,
            row_spec=puct_spec_by_id[row_id],
            target_move=int(puct_spec_by_id[row_id]["preferred_move"]),
        )
        for row_id in PUCT_ROW_IDS
    }
    puct_reference_gradients = {
        row_id: per_row_policy_gradient(
            model=model,
            device=device,
            row_spec=puct_spec_by_id[row_id],
            target_move=int(puct_spec_by_id[row_id]["active_reference_move"]),
        )
        for row_id in PUCT_ROW_IDS
    }

    pair_rows: list[dict[str, Any]] = []
    for classic_row_id, classic_gradient in classic_gradients.items():
        for puct_row_id in PUCT_ROW_IDS:
            preferred_cosine = cosine_similarity(
                classic_gradient, puct_preferred_gradients[puct_row_id]
            )
            reference_cosine = cosine_similarity(
                classic_gradient, puct_reference_gradients[puct_row_id]
            )
            pair_rows.append(
                {
                    "classic_row": classic_row_id,
                    "puct_row": puct_row_id,
                    "preferred_policy_gradient_cosine": preferred_cosine,
                    "active_reference_gradient_cosine": reference_cosine,
                }
            )

    strongest_negative_pairs = sorted(
        [
            row
            for row in pair_rows
            if row["preferred_policy_gradient_cosine"] is not None
            and row["preferred_policy_gradient_cosine"] < -0.02
        ],
        key=lambda row: (
            row["preferred_policy_gradient_cosine"],
            row["classic_row"],
            row["puct_row"],
        ),
    )[:10]
    return {
        "status": "completed",
        "pair_rows": pair_rows,
        "strongest_negative_pairs": strongest_negative_pairs,
    }


def classify_results(
    *,
    traces: dict[str, dict[str, Any]],
    similarity: dict[str, Any],
    gradient: dict[str, Any],
) -> tuple[str, str, str, list[str], list[str]]:
    epoch2 = epoch2_attribution_map(traces)
    classic_all = epoch2["classic_all"]
    best_loo_name = select_best_leave_one_out(epoch2)
    best_loo = epoch2[best_loo_name]
    worst_single_name = select_worst_single_row(epoch2)
    worst_single = epoch2[worst_single_name]
    low_lr = epoch2.get("classic_all_low_lr")
    policy_only = epoch2.get("classic_all_policy_only")

    best_loo_removed_row = removed_row_id_from_leave_one_out_variant(best_loo_name)
    strongest_negative_pairs = list(gradient.get("strongest_negative_pairs") or [])
    nearest_classic_by_puct = similarity.get("nearest_classic_by_puct") or {}

    support: list[str] = []
    rejected: list[str] = []

    if (
        low_lr is not None
        and (classic_all["heavy_regression_count"] - low_lr["heavy_regression_count"])
        >= 2
        and low_lr["classic_gain_count"] >= classic_all["classic_gain_count"]
    ):
        support.append(
            f"classic_all_low_lr cut heavy PUCT regressions from {classic_all['heavy_regression_count']} to {low_lr['heavy_regression_count']} at epoch 2 without reducing Classic gains"
        )
        rejected.append("single-row removal was not needed to recover the PUCT bucket")
        rejected.append("policy-only was not the smallest sufficient explanation")
        return (
            "update_size_sensitive_interference",
            "run one small low-LR diagnostic lane with strict PUCT non-regression gate, no arena yet.",
            "update_size_sensitive_interference",
            support,
            rejected,
        )

    if (
        classic_all["heavy_regression_count"] > 0
        and (classic_all["heavy_regression_count"] - best_loo["heavy_regression_count"])
        >= 2
        and best_loo["classic_gain_count"]
        >= max(0, classic_all["classic_gain_count"] - 1)
        and worst_single["heavy_regression_count"]
        >= max(1, classic_all["heavy_regression_count"] - 1)
    ):
        support.append(
            f"removing {best_loo_removed_row} reduced heavy PUCT regressions from {classic_all['heavy_regression_count']} to {best_loo['heavy_regression_count']} at epoch 2 while preserving Classic gains ({classic_all['classic_gain_count']} -> {best_loo['classic_gain_count']})"
        )
        support.append(
            f"the matching single-row trace {worst_single_name} alone caused {worst_single['heavy_regression_count']} heavy PUCT regressions"
        )
        if strongest_negative_pairs:
            pair = strongest_negative_pairs[0]
            support.append(
                f"policy gradient cosine was negative for {pair['classic_row']} vs {pair['puct_row']} ({pair['preferred_policy_gradient_cosine']})"
            )
        rejected.append(
            "low LR did not explain the regression as directly as row removal"
        )
        rejected.append(
            "multiple single rows were not required to reproduce most of the damage"
        )
        return (
            "single_row_interference_driver",
            "build a Classic diagnostic artifact excluding that row and rerun the tiny trace; do not production-train yet.",
            best_loo_removed_row,
            support,
            rejected,
        )

    if policy_only is not None and policy_only["heavy_regression_count"] >= max(
        1, classic_all["heavy_regression_count"] - 1
    ):
        support.append(
            f"classic_all_policy_only preserved most of the PUCT damage ({policy_only['heavy_regression_count']} heavy regressions vs {classic_all['heavy_regression_count']} for classic_all) with value loss weight set to 0.0"
        )
        if strongest_negative_pairs:
            pair = strongest_negative_pairs[0]
            support.append(
                f"negative policy-gradient alignment appeared immediately at initialization ({pair['classic_row']} vs {pair['puct_row']} = {pair['preferred_policy_gradient_cosine']})"
            )
        rejected.append(
            "value-only attribution was skipped because policy loss cannot be disabled cleanly in train.py"
        )
        rejected.append(
            "no single leave-one-out row eliminated enough PUCT damage while keeping the same Classic signal"
        )
        return (
            "policy_target_interference",
            "test softer Classic policy targets or KL regularization against PUCT bucket behavior.",
            "policy_target_interference",
            support,
            rejected,
        )

    damaging_single_rows = [
        row
        for name, row in epoch2.items()
        if name.startswith("classic_only_") and row["heavy_regression_count"] >= 2
    ]
    if len(damaging_single_rows) >= 2:
        support.append(
            f"{len(damaging_single_rows)} independent single-row traces each caused at least two heavy PUCT regressions"
        )
        overlap_examples = []
        for puct_row_id, nearest_rows in nearest_classic_by_puct.items():
            top = nearest_rows[0]["classic_row"] if nearest_rows else None
            if top is not None:
                overlap_examples.append(f"{puct_row_id}->{top}")
        if overlap_examples:
            support.append(
                "nearest-neighbor overlap clustered around "
                + ", ".join(overlap_examples[:4])
            )
        rejected.append("no single row removal sharply resolved the conflict")
        rejected.append("low LR did not cleanly remove the conflict")
        return (
            "shared_representation_teacher_conflict",
            "do not train this family with ordinary replay; move to teacher-policy split architecture/objective design or select a different non-opening family.",
            "shared_representation_teacher_conflict",
            support,
            rejected,
        )

    support.append(
        "classic_all reproduced cross-teacher damage under the current main checkpoint"
    )
    rejected.append("no attribution branch crossed a clear decision threshold")
    return (
        "trace_nondeterminism_or_reproduction_gap",
        "reproduce PR #48 trace with fixed hashes and deterministic data order.",
        "trace_nondeterminism_or_reproduction_gap",
        support,
        rejected,
    )


def build_report(summary: dict[str, Any]) -> str:
    similarity_rows = []
    nearest_puct_by_classic = summary["state_similarity"]["nearest_puct_by_classic"]
    for classic_row_id in CLASSIC_ROW_IDS:
        nearest = nearest_puct_by_classic.get(classic_row_id, [])
        nearest_rows_text = ", ".join(
            f"{row['puct_row']} ({row['combined_distance']})" for row in nearest
        )
        first = nearest[0] if nearest else {}
        similarity_rows.append(
            [
                classic_row_id,
                nearest_rows_text or "-",
                str(first.get("combined_distance", "-")),
                format_float(first.get("shared_legal_mask")),
                format_float(first.get("shared_consequence_pattern")),
                "nearest_puct_neighbors",
            ]
        )

    artifact_rows = []
    for row in summary["artifact_variants"]:
        artifact_rows.append(
            [
                row["artifact_name"],
                ",".join(row["included_classic_rows"]),
                ",".join(row["excluded_classic_rows"]),
                str(row["row_count"]),
                row["status"],
                row["notes"],
            ]
        )

    classic_result_rows = []
    for trace in summary["traces"]:
        for row in trace["classic_rows"]:
            if int(row["epoch"]) <= 0:
                continue
            classic_result_rows.append(
                [
                    trace["artifact_name"],
                    str(row["epoch"]),
                    row["row_id"],
                    str(row["selected_move_384"]),
                    str(row["selected_move_1200"]),
                    str(row["active_reference_move"]),
                    format_float(row["reference_visit_share_384"]),
                    format_float(row["reference_visit_share_1200"]),
                    format_bool(row["improved_vs_current"]),
                    format_bool(row["strict_pass"]),
                    row["notes"],
                ]
            )

    puct_result_rows = []
    for trace in summary["traces"]:
        for row in trace["puct_rows"]:
            if int(row["epoch"]) <= 0:
                continue
            puct_result_rows.append(
                [
                    trace["artifact_name"],
                    str(row["epoch"]),
                    row["row_id"],
                    str(row["puct_preferred_move"]),
                    str(row["selected_move_384"]),
                    str(row["selected_move_1200"]),
                    format_float(row["puct_preferred_visit_share_384"]),
                    format_float(row["puct_preferred_visit_share_1200"]),
                    format_bool(row["selected_equals_puct_preferred_1200"]),
                    format_bool(row["cross_teacher_regression"]),
                    row["notes"],
                ]
            )

    attribution_rows = []
    for row in summary["attribution_rows"]:
        attribution_rows.append(
            [
                row["artifact_name"],
                str(row["classic_gain_count"]),
                str(row["heavy_regression_count"]),
                str(row["excluded_drift_count"]),
                str(row["interference_score"]),
                str(row.get("likely_driver") or "-"),
                row["notes"],
            ]
        )

    decision = summary["decision"]
    lines = [
        "# AlphaZero-lite Incumbent Proxy Teacher Interference Attribution Results",
        "",
        "## 1. Context",
        "",
        "- No arena was run.",
        "- No MCTS1200 benchmark lane was run.",
        "- No model was promoted.",
        "- Active references stayed read-only at `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.",
        "- PUCT rows were never used as Classic training targets.",
        "- Excluded rows were never used as training targets or gates.",
        "- Temporary artifacts and exports stayed under `/tmp/azlite_incumbent_proxy_teacher_interference_attribution/`.",
        "",
        "## 2. Why PR #48 blocked production training",
        "",
        "- PR #48 showed local Classic improvement but broad PUCT-bucket regression under tiny traces.",
        "- Its final classification was `cross_teacher_interference`.",
        "- This audit isolates which Classic row or training component causes that interference.",
        "",
        "## 3. Input and bucket validation",
        "",
        f"- Validation status: `{summary['input_validation']['status']}`.",
        f"- PUCT rows present in Classic artifact: `{json.dumps(summary['input_validation']['puct_rows_in_classic_artifact'])}`.",
        f"- Excluded rows present in Classic artifact: `{json.dumps(summary['input_validation']['excluded_rows_in_classic_artifact'])}`.",
        f"- Conflicting duplicate canonical roles: `{len(summary['input_validation']['conflicting_canonical_roles'])}`.",
        "",
        "## 4. State-similarity analysis",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "classic_row",
                "nearest_puct_rows",
                "distance",
                "shared_legal_mask",
                "shared_consequence_pattern",
                "notes",
            ],
            similarity_rows,
        )
    )
    lines.extend(["", "## 5. Artifact variants", ""])
    lines.extend(
        markdown_table(
            [
                "artifact_name",
                "included_classic_rows",
                "excluded_classic_rows",
                "row_count",
                "status",
                "notes",
            ],
            artifact_rows,
        )
    )
    lines.extend(["", "## 6. Training traces", ""])
    lines.append(
        f"- Phase-2 epoch-4 reruns: `{', '.join(summary['phase2_epoch4_variants'])}`."
    )
    lines.append(
        f"- Learning-rate ablations: `classic_all_low_lr`, `best_leave_one_out_low_lr` with lr={LOW_LR}."
    )
    lines.append(
        "- Value-only ablation skipped because train.py does not disable policy loss cleanly."
    )
    lines.extend(["", "## 7. Classic improvement results", ""])
    lines.extend(
        markdown_table(
            [
                "artifact_name",
                "epoch",
                "row_id",
                "selected_384",
                "selected_1200",
                "active_reference_move",
                "reference_visit_share_384",
                "reference_visit_share_1200",
                "improved_vs_current",
                "strict_pass",
                "notes",
            ],
            classic_result_rows,
        )
    )
    lines.extend(["", "## 8. PUCT interference results", ""])
    lines.extend(
        markdown_table(
            [
                "artifact_name",
                "epoch",
                "row_id",
                "puct_preferred_move",
                "selected_384",
                "selected_1200",
                "puct_preferred_visit_share_384",
                "puct_preferred_visit_share_1200",
                "selected_equals_puct_preferred_1200",
                "cross_teacher_regression",
                "notes",
            ],
            puct_result_rows,
        )
    )
    lines.extend(["", "## 9. Attribution analysis", ""])
    lines.extend(
        markdown_table(
            [
                "artifact_name",
                "classic_gain_count",
                "puct_damage_count",
                "excluded_drift_count",
                "interference_score",
                "likely_driver",
                "notes",
            ],
            attribution_rows,
        )
    )
    lines.extend(["", "## 10. Optional gradient attribution", ""])
    if summary["gradient_attribution"]["status"] == "completed":
        strongest = (
            summary["gradient_attribution"].get("strongest_negative_pairs") or []
        )
        if strongest:
            lines.append(
                "- Strongest negative policy-gradient cosine pairs at initialization:"
            )
            for row in strongest[:5]:
                lines.append(
                    f"- `{row['classic_row']}` vs `{row['puct_row']}`: preferred={row['preferred_policy_gradient_cosine']}, active_reference={row['active_reference_gradient_cosine']}"
                )
        else:
            lines.append(
                "- No strongly negative policy-gradient cosine pairs crossed the reporting threshold."
            )
    else:
        lines.append("- gradient_attribution_skipped_due_to_code_complexity")
    lines.extend(
        [
            "",
            "## 11. Interpretation",
            "",
            f"- Final classification: `{decision['classification']}`.",
            f"- Likely driver: `{decision['likely_driver']}`.",
            f"- Interference score formula: `{summary['interference_score_formula']}`.",
        ]
    )
    for bullet in decision["supporting_evidence"]:
        lines.append(f"- {bullet}")
    lines.extend(["", "## 12. Exactly one recommended next action", ""])
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
                    decision["classification"],
                    "; ".join(decision["supporting_evidence"]),
                    "; ".join(decision["rejected_alternatives"]),
                    decision["next_action"],
                ]
            ],
        )
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.export_root.mkdir(parents=True, exist_ok=True)

    classic_rows = read_jsonl(args.classic_rows_path)
    puct_rows = read_jsonl(args.puct_rows_path)
    excluded_rows = read_jsonl(args.excluded_rows_path)
    base_artifact_source_path = (
        args.pr48_artifact_path
        if args.pr48_artifact_path.exists()
        else args.output_dir / "rebuilt_pr48_artifact.jsonl"
    )
    base_artifact_rows = load_or_build_base_artifact(args)

    input_validation = validate_bucket_rows(
        suite_path=args.suite_path,
        reference_path=args.reference_path,
        classic_rows=classic_rows,
        puct_rows=puct_rows,
        excluded_rows=excluded_rows,
        base_artifact_rows=base_artifact_rows,
    )
    if input_validation["status"] != "ok":
        raise ValueError(f"input validation failed: {input_validation}")

    classic_specs = row_specs_from_bucket_rows(classic_rows, CLASSIC_ROW_IDS)
    puct_specs = row_specs_from_bucket_rows(puct_rows, PUCT_ROW_IDS)
    excluded_specs = row_specs_from_bucket_rows(excluded_rows, EXCLUDED_ROW_IDS)
    classic_spec_by_id = {row["row_id"]: row for row in classic_specs}
    puct_spec_by_id = {row["row_id"]: row for row in puct_specs}
    excluded_spec_by_id = {row["row_id"]: row for row in excluded_specs}

    similarity = build_similarity_matrix(
        classic_specs=classic_specs, puct_specs=puct_specs
    )
    write_json(args.similarity_out, similarity)

    variant_catalog = build_variant_catalog()
    phase1_specs = [
        spec
        for spec in variant_catalog
        if spec.artifact_name != "best_leave_one_out_low_lr"
    ]
    materialize_variant_artifacts(
        base_artifact_rows=base_artifact_rows,
        base_artifact_source_path=base_artifact_source_path,
        artifact_dir=args.artifact_dir,
        variant_specs=phase1_specs,
        selected_best_leave_one_out=None,
    )

    current_metadata = load_json(args.current_artifact / "metadata.json")
    base_model_spec = model_spec_from_metadata(current_metadata)
    init_checkpoint = current_checkpoint_path(args.current_artifact, args.output_dir)
    device = choose_device(args.device)

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

    traces: dict[str, dict[str, Any]] = {}
    for spec in phase1_specs:
        traces[spec.artifact_name] = run_variant_trace(
            spec=spec,
            classic_spec_by_id=classic_spec_by_id,
            puct_spec_by_id=puct_spec_by_id,
            excluded_spec_by_id=excluded_spec_by_id,
            classic_baseline_by_id=classic_baseline_by_id,
            puct_baseline_by_id=puct_baseline_by_id,
            excluded_baseline_by_id=excluded_baseline_by_id,
            init_checkpoint=init_checkpoint,
            current_metadata=current_metadata,
            base_model_spec=base_model_spec,
            export_root=args.export_root,
            seed=args.seed,
            device=device,
        )
        traces[spec.artifact_name]["classic_baseline_by_id"] = classic_baseline_by_id
        traces[spec.artifact_name]["puct_baseline_by_id"] = puct_baseline_by_id
        traces[spec.artifact_name]["excluded_baseline_by_id"] = excluded_baseline_by_id

    epoch2_rows = epoch2_attribution_map(traces)
    best_leave_one_out_name = select_best_leave_one_out(epoch2_rows)
    worst_single_row_name = select_worst_single_row(epoch2_rows)
    best_leave_one_out_removed_row = removed_row_id_from_leave_one_out_variant(
        best_leave_one_out_name
    )

    materialized_all = materialize_variant_artifacts(
        base_artifact_rows=base_artifact_rows,
        base_artifact_source_path=base_artifact_source_path,
        artifact_dir=args.artifact_dir,
        variant_specs=variant_catalog,
        selected_best_leave_one_out=best_leave_one_out_name,
    )

    best_loo_low_lr_spec = VariantSpec(
        artifact_name="best_leave_one_out_low_lr",
        included_row_ids=tuple(
            row_id
            for row_id in CLASSIC_ROW_IDS
            if row_id != best_leave_one_out_removed_row
        ),
        notes=f"same rows as {best_leave_one_out_name} with lower learning rate",
        lr=LOW_LR,
    )
    traces[best_loo_low_lr_spec.artifact_name] = run_variant_trace(
        spec=best_loo_low_lr_spec,
        classic_spec_by_id=classic_spec_by_id,
        puct_spec_by_id=puct_spec_by_id,
        excluded_spec_by_id=excluded_spec_by_id,
        classic_baseline_by_id=classic_baseline_by_id,
        puct_baseline_by_id=puct_baseline_by_id,
        excluded_baseline_by_id=excluded_baseline_by_id,
        init_checkpoint=init_checkpoint,
        current_metadata=current_metadata,
        base_model_spec=base_model_spec,
        export_root=args.export_root,
        seed=args.seed,
        device=device,
    )
    traces[best_loo_low_lr_spec.artifact_name]["classic_baseline_by_id"] = (
        classic_baseline_by_id
    )
    traces[best_loo_low_lr_spec.artifact_name]["puct_baseline_by_id"] = (
        puct_baseline_by_id
    )
    traces[best_loo_low_lr_spec.artifact_name]["excluded_baseline_by_id"] = (
        excluded_baseline_by_id
    )

    phase2_epoch4_variants = [
        "classic_all",
        best_leave_one_out_name,
        worst_single_row_name,
        "target_only",
        "preservation_only",
    ]
    phase2_override_specs = {
        "classic_all": VariantSpec(
            "classic_all",
            CLASSIC_ROW_IDS,
            "same rows as PR #48 full Classic artifact",
            TRACE_PHASE2_EPOCHS,
        ),
        best_leave_one_out_name: VariantSpec(
            best_leave_one_out_name,
            tuple(
                row_id
                for row_id in CLASSIC_ROW_IDS
                if row_id != best_leave_one_out_removed_row
            ),
            f"leave out Classic row {best_leave_one_out_removed_row}",
            TRACE_PHASE2_EPOCHS,
        ),
        "target_only": VariantSpec(
            "target_only",
            CLASSIC_TARGET_ROW_IDS,
            "all five Classic target rows without preservation control",
            TRACE_PHASE2_EPOCHS,
        ),
        "preservation_only": VariantSpec(
            "preservation_only",
            CLASSIC_CONTROL_ROW_IDS,
            "preservation control only",
            TRACE_PHASE2_EPOCHS,
        ),
    }
    if worst_single_row_name in SINGLE_ROW_ID_BY_VARIANT:
        phase2_override_specs[worst_single_row_name] = VariantSpec(
            worst_single_row_name,
            (SINGLE_ROW_ID_BY_VARIANT[worst_single_row_name],),
            f"single-row follow-up for {worst_single_row_name}",
            TRACE_PHASE2_EPOCHS,
        )
    for artifact_name in phase2_epoch4_variants:
        spec = phase2_override_specs[artifact_name]
        write_jsonl(
            spec.artifact_path,
            [
                clone_artifact_row(
                    {
                        **row,
                        "source_runs": [{**(row.get("source_runs") or [{}])[0]}],
                    },
                    base_artifact_source_path=base_artifact_source_path,
                )
                for row in base_artifact_rows
                if str(
                    (row.get("source_runs") or [{}])[0].get("id") or row.get("row_id")
                )
                in set(spec.included_row_ids)
            ],
        )
        traces[artifact_name] = run_variant_trace(
            spec=spec,
            classic_spec_by_id=classic_spec_by_id,
            puct_spec_by_id=puct_spec_by_id,
            excluded_spec_by_id=excluded_spec_by_id,
            classic_baseline_by_id=classic_baseline_by_id,
            puct_baseline_by_id=puct_baseline_by_id,
            excluded_baseline_by_id=excluded_baseline_by_id,
            init_checkpoint=init_checkpoint,
            current_metadata=current_metadata,
            base_model_spec=base_model_spec,
            export_root=args.export_root,
            seed=args.seed,
            device=device,
        )
        traces[artifact_name]["classic_baseline_by_id"] = classic_baseline_by_id
        traces[artifact_name]["puct_baseline_by_id"] = puct_baseline_by_id
        traces[artifact_name]["excluded_baseline_by_id"] = excluded_baseline_by_id

    gradient_attribution = compute_gradient_attribution(
        init_checkpoint=init_checkpoint,
        current_metadata=current_metadata,
        device=device,
        classic_spec_by_id=classic_spec_by_id,
        puct_spec_by_id=puct_spec_by_id,
    )

    classification, next_action, likely_driver, supporting, rejected = classify_results(
        traces=traces,
        similarity=similarity,
        gradient=gradient_attribution,
    )

    attribution_rows: list[dict[str, Any]] = []
    for trace in sorted(traces.values(), key=lambda row: row["artifact_name"]):
        row = attribution_row_for_trace(trace, latest_epoch(trace))
        row["likely_driver"] = (
            likely_driver
            if trace["artifact_name"]
            in {"classic_all", best_leave_one_out_name, worst_single_row_name}
            else "-"
        )
        row["notes"] = (
            f"lr={trace['lr']}, value_loss_weight={trace['value_loss_weight']}, epochs={trace['epochs']}"
        )
        attribution_rows.append(row)

    summary = {
        "schema": SCHEMA,
        "inputs": {
            "reference_path": str(args.reference_path),
            "suite_path": str(args.suite_path),
            "current_artifact": str(args.current_artifact),
            "classic_rows_path": str(args.classic_rows_path),
            "puct_rows_path": str(args.puct_rows_path),
            "excluded_rows_path": str(args.excluded_rows_path),
            "pr48_artifact_path": str(args.pr48_artifact_path),
            "init_checkpoint": str(init_checkpoint),
        },
        "input_validation": input_validation,
        "state_similarity": similarity,
        "artifact_variants": materialized_all,
        "phase2_epoch4_variants": phase2_epoch4_variants,
        "traces": [traces[name] for name in sorted(traces)],
        "attribution_rows": attribution_rows,
        "interference_score_formula": "puct_lost_selection_1200_count + heavy_regression_count + excluded_drift_count - classic_gain_count",
        "gradient_attribution": gradient_attribution,
        "decision": {
            "classification": classification,
            "likely_driver": likely_driver,
            "supporting_evidence": supporting,
            "rejected_alternatives": rejected,
            "next_action": next_action,
        },
        "acceptance": {
            "arena_run": False,
            "mcts1200_lane_run": False,
            "model_promoted": False,
            "active_references_mutated": False,
            "puct_rows_used_as_targets": False,
            "excluded_rows_used_as_targets": False,
            "broad_replay_sweep_run": False,
        },
    }
    write_json(args.summary_out, summary)
    args.report_path.write_text(build_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(args.summary_out),
                "similarity_path": str(args.similarity_out),
                "report_path": str(args.report_path),
                "classification": classification,
                "next_action": next_action,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
