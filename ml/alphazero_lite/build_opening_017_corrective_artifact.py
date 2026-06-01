#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.build_train_only_forensic_suite_from_selfplay import decode_state
from ml.alphazero_lite.build_tracked_opening_capture_policy_artifact import (
    encode_raw_state,
)
from ml.alphazero_lite.corrected_guard_kill_gate import (
    DEFAULT_FALLBACK_REFERENCE_ARTIFACT,
    DEFAULT_REFERENCE_ARTIFACT,
)
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state


DEFAULT_SELF_PLAY_PATH = Path(
    "/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_versions/"
    "exp-v3-guard-safe-opening-selected-w1-iter1/self_play.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path("/tmp/azlite_opening_017_corrective_patch")
DEFAULT_ARTIFACT_PATH = (
    DEFAULT_OUTPUT_ROOT / "opening_017_corrective_patch_artifact.jsonl"
)
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_ROOT / "opening_017_corrective_patch_summary.json"
DEFAULT_INPUT_ENCODING = "kalah_v3"
DEFAULT_POLICY_TARGET_MODE = "sharpened"
DEFAULT_VALUE_TARGET_MODE = "sharpened"
SCHEMA = "azlite_opening_017_corrective_patch_artifact_v1"

ROW_SPECS = (
    {
        "row_id": "opening_plies_1_8-017",
        "required": True,
        "reason": "predecessor_target_correction",
    },
    {
        "row_id": "capture_available-002",
        "required": True,
        "reason": "descendant_guard_correction",
    },
    {
        "row_id": "capture_available-003",
        "required": True,
        "reason": "descendant_guard_correction",
    },
    {
        "row_id": "capture_available-007",
        "required": True,
        "reason": "descendant_guard_correction",
    },
    {
        "row_id": "capture_available-006",
        "required": False,
        "reason": "guard_preservation_control",
    },
    {
        "row_id": "capture_available-008",
        "required": False,
        "reason": "guard_preservation_control",
    },
)
REQUIRED_ROW_IDS = tuple(spec["row_id"] for spec in ROW_SPECS if spec["required"])
TARGET_ROW_IDS = tuple(spec["row_id"] for spec in ROW_SPECS)
CORRECTED_POLICY_MASS = 0.85
NON_REFERENCE_POLICY_MASS = 0.15
SOURCE_NAME = "opening_017_corrective_patch"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-play", type=Path, default=DEFAULT_SELF_PLAY_PATH)
    parser.add_argument(
        "--reference-artifact", type=Path, default=DEFAULT_REFERENCE_ARTIFACT
    )
    parser.add_argument(
        "--fallback-reference-artifact",
        type=Path,
        default=DEFAULT_FALLBACK_REFERENCE_ARTIFACT,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--out-summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--input-encoding", default=DEFAULT_INPUT_ENCODING)
    parser.add_argument("--policy-target-mode", default=DEFAULT_POLICY_TARGET_MODE)
    parser.add_argument("--value-target-mode", default=DEFAULT_VALUE_TARGET_MODE)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(f"expected JSON object row in {path}")
                rows.append(payload)
    return rows


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


def rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("id")
        if isinstance(row_id, str) and row_id:
            indexed[row_id] = row
    return indexed


def merge_reference_rows(
    reference_artifact: dict[str, Any],
    fallback_reference_artifact: dict[str, Any],
    *,
    reference_path: Path,
    fallback_path: Path,
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row_id, row in rows_by_id(
        list(fallback_reference_artifact.get("rows") or [])
    ).items():
        merged[row_id] = {
            **row,
            "reference_artifact_path": str(fallback_path),
            "reference_artifact_kind": "fallback",
        }
    for row_id, row in rows_by_id(list(reference_artifact.get("rows") or [])).items():
        merged[row_id] = {
            **row,
            "reference_artifact_path": str(reference_path),
            "reference_artifact_kind": "primary",
        }
    missing = [row_id for row_id in TARGET_ROW_IDS if row_id not in merged]
    if missing:
        raise ValueError(f"missing corrected references for rows: {missing}")
    return merged


def self_play_teacher_source(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    if len(counter) == 1:
        return next(iter(counter))
    return ",".join(f"{name}:{counter[name]}" for name in sorted(counter))


def rounded_policy(values: list[float]) -> list[float]:
    return [round(float(value), 4) for value in values]


def build_failure_chain_rows(
    *,
    self_play_rows: list[dict[str, Any]],
    reference_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    canonical_to_row_id = {
        str(reference_rows[row_id]["canonical_state"]): row_id
        for row_id in TARGET_ROW_IDS
    }
    aggregates: dict[str, dict[str, Any]] = {
        row_id: {
            "count": 0,
            "policy_sum": [0.0] * 6,
            "teacher_sources": Counter(),
            "plys": Counter(),
        }
        for row_id in TARGET_ROW_IDS
    }

    for row in self_play_rows:
        raw_state = decode_state(list(row["state"]))
        canonical = canonical_state_key(raw_state)
        row_id = canonical_to_row_id.get(canonical)
        if row_id is None:
            continue
        aggregate = aggregates[row_id]
        aggregate["count"] += 1
        policy = list(row.get("policy") or [])
        if len(policy) != 6:
            raise ValueError(f"unexpected self-play policy width for {row_id}")
        for move, value in enumerate(policy):
            aggregate["policy_sum"][move] += float(value)
        teacher_source = row.get("teacher_source")
        if isinstance(teacher_source, str) and teacher_source:
            aggregate["teacher_sources"][teacher_source] += 1
        ply = row.get("move_index")
        if isinstance(ply, int) and not isinstance(ply, bool):
            aggregate["plys"][int(ply)] += 1

    extracted_rows: list[dict[str, Any]] = []
    missing_required = [
        row_id for row_id in REQUIRED_ROW_IDS if int(aggregates[row_id]["count"]) <= 0
    ]
    if missing_required:
        raise ValueError(
            "required rows missing from PR #36 self-play: "
            + ", ".join(missing_required)
        )

    for spec in ROW_SPECS:
        row_id = str(spec["row_id"])
        reference_row = reference_rows[row_id]
        state = dict(reference_row["state"])
        legal_moves = KalahGame.from_state(state).possible_moves()
        corrected_reference_move = int(reference_row["reference_move"])
        aggregate = aggregates[row_id]
        count = int(aggregate["count"])
        averaged_policy = None
        averaged_top_move = None
        corrected_reference_mass = None
        top_move_mass = None
        sibling_move = None
        sibling_move_target_mass = None
        if count > 0:
            averaged_policy = [value / count for value in aggregate["policy_sum"]]
            averaged_top_move = top_policy_move(averaged_policy, legal_moves)
            corrected_reference_mass = round(
                float(averaged_policy[corrected_reference_move]), 4
            )
            if averaged_top_move is not None:
                top_move_mass = round(float(averaged_policy[averaged_top_move]), 4)
            sibling_candidates = [
                move for move in legal_moves if move != corrected_reference_move
            ]
            if sibling_candidates and averaged_policy is not None:
                best_sibling = sibling_candidates[0]
                for candidate in sibling_candidates[1:]:
                    if float(averaged_policy[candidate]) > float(
                        averaged_policy[best_sibling]
                    ) or (
                        math.isclose(
                            float(averaged_policy[candidate]),
                            float(averaged_policy[best_sibling]),
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                        and candidate < best_sibling
                    ):
                        best_sibling = candidate
                sibling_move = best_sibling
                sibling_move_target_mass = round(
                    float(averaged_policy[sibling_move]), 4
                )
        plys = [ply for ply, _count in aggregate["plys"].most_common()]
        consequence_rows = []
        for move in legal_moves:
            consequence = move_consequence_for_state(state, move)
            consequence_rows.append(
                {
                    "move": int(move),
                    "gives_extra_turn": bool(consequence["gives_extra_turn"]),
                    "produces_capture": bool(consequence["produces_capture"]),
                    "capture_count": int(consequence["capture_count"]),
                    "immediate_store_delta": int(consequence["store_delta_immediate"]),
                    "side_to_move_after": int(consequence["resulting_side_to_move"]),
                }
            )
        if count <= 0:
            diagnosis = "control_reference_only"
        elif averaged_top_move == corrected_reference_move:
            if (
                sibling_move_target_mass is not None
                and corrected_reference_mass is not None
            ):
                if corrected_reference_mass - sibling_move_target_mass < 0.05:
                    diagnosis = "reference_move_still_top_but_noisy"
                else:
                    diagnosis = "reference_move_supported"
            else:
                diagnosis = "reference_move_supported"
        elif row_id == "opening_plies_1_8-017":
            diagnosis = "predecessor_target_drift"
        elif row_id in {"capture_available-002", "capture_available-007"}:
            diagnosis = "descendant_shift_to_move_1"
        else:
            diagnosis = "descendant_target_noise"
        extracted_rows.append(
            {
                "row_id": row_id,
                "canonical_state_hash": str(reference_row["canonical_state"]),
                "self_play_count": count,
                "averaged_policy_target": None
                if averaged_policy is None
                else rounded_policy(averaged_policy),
                "averaged_self_play_top_move": averaged_top_move,
                "corrected_reference_move": corrected_reference_move,
                "corrected_reference_mass": corrected_reference_mass,
                "top_move_mass": top_move_mass,
                "sibling_move": sibling_move,
                "sibling_move_target_mass": sibling_move_target_mass,
                "teacher_source": self_play_teacher_source(
                    aggregate["teacher_sources"]
                ),
                "ply": None if not plys else (plys[0] if len(plys) == 1 else plys),
                "legal_moves": legal_moves,
                "per_move_consequences": consequence_rows,
                "reference_teacher_value": reference_row.get("teacher_value"),
                "reference_source": reference_row.get("reference_source"),
                "diagnosis": diagnosis,
            }
        )
    return extracted_rows


def corrective_policy(legal_moves: list[int], reference_move: int) -> list[float]:
    policy = [0.0] * 6
    if reference_move not in legal_moves:
        raise ValueError(f"reference move {reference_move} is not legal")
    if len(legal_moves) == 1:
        policy[reference_move] = 1.0
        return policy
    residual = NON_REFERENCE_POLICY_MASS / float(len(legal_moves) - 1)
    for move in legal_moves:
        policy[move] = residual
    policy[reference_move] = CORRECTED_POLICY_MASS
    return policy


def build_corrective_rows(
    *,
    extracted_rows: list[dict[str, Any]],
    reference_rows: dict[str, dict[str, Any]],
    reference_artifact: dict[str, Any],
    fallback_reference_artifact: dict[str, Any],
    input_encoding: str,
    policy_target_mode: str,
    value_target_mode: str,
) -> list[dict[str, Any]]:
    extracted_by_id = {row["row_id"]: row for row in extracted_rows}
    corrective_rows: list[dict[str, Any]] = []
    for spec in ROW_SPECS:
        row_id = str(spec["row_id"])
        reference_row = reference_rows[row_id]
        extracted = extracted_by_id[row_id]
        state = dict(reference_row["state"])
        legal_moves = KalahGame.from_state(state).possible_moves()
        corrected_reference_move = int(reference_row["reference_move"])
        policy = corrective_policy(legal_moves, corrected_reference_move)
        teacher_value = reference_row.get("teacher_value")
        if teacher_value is None:
            raise ValueError(f"value target unavailable for {row_id}")
        source_reference_path = str(reference_row["reference_artifact_path"])
        source_reference_artifact = (
            reference_artifact
            if str(reference_row.get("reference_artifact_kind")) == "primary"
            else fallback_reference_artifact
        )
        sample_seeds = list(
            source_reference_artifact.get("reference", {}).get("sample_seeds") or []
        )
        corrective_rows.append(
            {
                "canonical_state": str(reference_row["canonical_state"]),
                "state": encode_raw_state(
                    raw_state=state,
                    input_encoding=input_encoding,
                ),
                "raw_state": state,
                "side_to_move": int(state["current_player"]),
                "legal_moves": legal_moves,
                "policy": policy,
                "value": float(teacher_value),
                "input_encoding": input_encoding,
                "policy_target_mode": policy_target_mode,
                "policy_target_actual_mode": policy_target_mode,
                "value_target_mode": value_target_mode,
                "source": SOURCE_NAME,
                "source_artifacts": [source_reference_path],
                "source_runs": [
                    {
                        "kind": SOURCE_NAME,
                        "id": row_id,
                        "reference_move": corrected_reference_move,
                    }
                ],
                "selection_reasons": [spec["reason"]],
                "priority_score": 100.0,
                "teacher_policy_simulations": int(
                    source_reference_artifact.get("reference", {}).get(
                        "policy_simulations", 0
                    )
                ),
                "teacher_value_simulations": int(
                    source_reference_artifact.get("reference", {}).get(
                        "value_simulations", 0
                    )
                ),
                "teacher_seed": int(sample_seeds[0]) if sample_seeds else 0,
                "teacher_policy_seed": int(sample_seeds[0]) if sample_seeds else 0,
                "teacher_value_seed": int(sample_seeds[0]) if sample_seeds else 0,
                "teacher_selected_move": corrected_reference_move,
                "teacher_child_stats": list(reference_row.get("child_stats") or []),
                "reference_move": corrected_reference_move,
                "corrected_reference_move": corrected_reference_move,
                "old_self_play_top_move": extracted.get("averaged_self_play_top_move"),
                "old_self_play_reference_mass": extracted.get(
                    "corrected_reference_mass"
                ),
                "corrected_policy_mass": round(
                    float(policy[corrected_reference_move]), 4
                ),
                "reason": spec["reason"],
                "train_only": True,
                "exclude_from_validation": True,
                "value_source": str(reference_row.get("reference_source") or "unknown"),
                "reference_artifact_kind": str(
                    reference_row.get("reference_artifact_kind") or "unknown"
                ),
            }
        )
    return corrective_rows


def validate_corrective_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing_required_rows = [
        row_id
        for row_id in REQUIRED_ROW_IDS
        if row_id not in {row["source_runs"][0]["id"] for row in rows}
    ]
    duplicate_conflicts = 0
    stale_reference_conflicts = 0
    canonical_targets: dict[str, tuple[int, tuple[float, ...]]] = {}
    notes: list[str] = []
    for row in rows:
        row_id = str(row["source_runs"][0]["id"])
        legal_moves = [int(move) for move in list(row.get("legal_moves") or [])]
        policy = [float(value) for value in list(row.get("policy") or [])]
        reference_move = int(row["corrected_reference_move"])
        if reference_move not in legal_moves:
            notes.append(f"illegal_reference_move_{row_id}")
        if not math.isclose(sum(policy), 1.0, rel_tol=0.0, abs_tol=1e-6):
            notes.append(f"policy_not_normalized_{row_id}")
        if top_policy_move(policy, legal_moves) != reference_move:
            notes.append(f"reference_not_top_{row_id}")
        if not bool(row.get("train_only")):
            notes.append(f"train_only_missing_{row_id}")
        if not bool(row.get("exclude_from_validation")):
            notes.append(f"exclude_from_validation_missing_{row_id}")
        if str(row.get("reference_artifact_kind") or "") not in {"primary", "fallback"}:
            stale_reference_conflicts += 1
        canonical_state = str(row.get("canonical_state", ""))
        target_key = (reference_move, tuple(round(value, 8) for value in policy))
        existing = canonical_targets.get(canonical_state)
        if existing is not None and existing != target_key:
            duplicate_conflicts += 1
        canonical_targets[canonical_state] = target_key
    if missing_required_rows:
        notes.append("missing_rows_" + ",".join(missing_required_rows))
    status = "ok"
    if (
        missing_required_rows
        or duplicate_conflicts > 0
        or stale_reference_conflicts > 0
    ):
        status = "invalid"
    return {
        "status": status,
        "all_required_rows_present": not missing_required_rows,
        "duplicate_conflicts": duplicate_conflicts,
        "stale_reference_conflicts": stale_reference_conflicts,
        "notes": notes or ["ok"],
    }


def build_artifact(
    *,
    self_play_path: Path,
    reference_artifact_path: Path,
    fallback_reference_artifact_path: Path,
    input_encoding: str,
    policy_target_mode: str,
    value_target_mode: str,
) -> dict[str, Any]:
    reference_artifact = load_json(reference_artifact_path)
    fallback_reference_artifact = load_json(fallback_reference_artifact_path)
    reference_rows = merge_reference_rows(
        reference_artifact,
        fallback_reference_artifact,
        reference_path=reference_artifact_path,
        fallback_path=fallback_reference_artifact_path,
    )
    self_play_rows = read_jsonl(self_play_path)
    extracted_rows = build_failure_chain_rows(
        self_play_rows=self_play_rows,
        reference_rows=reference_rows,
    )
    corrective_rows = build_corrective_rows(
        extracted_rows=extracted_rows,
        reference_rows=reference_rows,
        reference_artifact=reference_artifact,
        fallback_reference_artifact=fallback_reference_artifact,
        input_encoding=input_encoding,
        policy_target_mode=policy_target_mode,
        value_target_mode=value_target_mode,
    )
    validation = validate_corrective_rows(corrective_rows)
    if validation["status"] != "ok":
        raise ValueError(
            "corrective artifact validation failed: " + ", ".join(validation["notes"])
        )
    return {
        "schema": SCHEMA,
        "self_play_path": str(self_play_path),
        "reference_artifact": str(reference_artifact_path),
        "fallback_reference_artifact": str(fallback_reference_artifact_path),
        "failure_chain_rows": extracted_rows,
        "corrective_rows": corrective_rows,
        "validation": validation,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_artifact(
        self_play_path=args.self_play,
        reference_artifact_path=args.reference_artifact,
        fallback_reference_artifact_path=args.fallback_reference_artifact,
        input_encoding=str(args.input_encoding),
        policy_target_mode=str(args.policy_target_mode),
        value_target_mode=str(args.value_target_mode),
    )
    write_jsonl(args.out, list(summary["corrective_rows"]))
    summary = {
        **summary,
        "artifact_path": str(args.out),
        "summary_path": str(args.out_summary),
        "row_count": len(summary["corrective_rows"]),
    }
    write_json(args.out_summary, summary)
    print(
        json.dumps(
            {
                "artifact_path": str(args.out),
                "summary_path": str(args.out_summary),
                "row_count": len(summary["corrective_rows"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
