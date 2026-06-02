#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.forensic_suite import canonical_state_key, load_suite
from ml.alphazero_lite.run_incumbent_proxy_disagreement_value_backup_audit import (
    load_json,
    round_float,
    write_json,
)
from ml.alphazero_lite.run_incumbent_proxy_teacher_policy_decision_audit import (
    canonical_reference_rows,
)
from ml.alphazero_lite.self_play import encode_state


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
DEFAULT_REPORT_PATH = Path(
    "docs/alphazero-lite-incumbent-proxy-disagreement-teacher-bucket-split-results.md"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/azlite_incumbent_proxy_classic_teacher_diagnostic")
DEFAULT_ARTIFACT_OUT = DEFAULT_OUTPUT_DIR / "classic_teacher_diagnostic_artifact.jsonl"
DEFAULT_SUMMARY_OUT = (
    DEFAULT_OUTPUT_DIR / "classic_teacher_diagnostic_artifact_summary.json"
)
DEFAULT_TARGET_ONLY_OUT = DEFAULT_OUTPUT_DIR / "classic_teacher_target_candidates.jsonl"
DEFAULT_CONTROL_ONLY_OUT = (
    DEFAULT_OUTPUT_DIR / "classic_teacher_preservation_controls.jsonl"
)

SCHEMA = "azlite_incumbent_proxy_classic_teacher_diagnostic_artifact_v1"
SOURCE_NAME = "incumbent_proxy_classic_teacher_diagnostic"
FAMILY = "incumbent_proxy_disagreement"
BUCKET = "classic_teacher"
TARGET_ROW_IDS = (
    "incumbent_proxy_disagreement-014",
    "incumbent_proxy_disagreement-022",
    "incumbent_proxy_disagreement-024",
    "incumbent_proxy_disagreement-025",
    "incumbent_proxy_disagreement-035",
)
CONTROL_ROW_IDS = ("incumbent_proxy_disagreement-008",)
EXPECTED_ROW_IDS = (*TARGET_ROW_IDS, *CONTROL_ROW_IDS)
TARGET_ROLE_BY_ID = {
    **{row_id: "target_candidate" for row_id in TARGET_ROW_IDS},
    **{row_id: "preservation_control" for row_id in CONTROL_ROW_IDS},
}
REFERENCE_POLICY_MASS = 0.85
NON_REFERENCE_POLICY_MASS = 0.15


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
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-out", type=Path, default=DEFAULT_ARTIFACT_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--target-only-out", type=Path, default=DEFAULT_TARGET_ONLY_OUT)
    parser.add_argument(
        "--control-only-out", type=Path, default=DEFAULT_CONTROL_ONLY_OUT
    )
    parser.add_argument("--input-encoding", default="kalah_v3")
    parser.add_argument("--policy-target-mode", default="default")
    parser.add_argument("--value-target-mode", default="default")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path} contains a non-object JSONL row")
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


def parse_report_bucket_rows(report_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    in_table = False
    for raw_line in report_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("## Bucket Assignment Table"):
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        if not in_table or not line.startswith("| incumbent_proxy_disagreement-"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 8:
            continue
        rows.append(
            {
                "row_id": parts[0],
                "bucket": parts[1],
                "row_decision": parts[2],
                "preferred_teacher": parts[3],
                "preferred_move": int(parts[4]),
                "active_reference_move": int(parts[5]),
                "recommended_role": parts[6],
                "evidence_summary": parts[7],
                "status": "ok",
            }
        )
    return rows


def load_bucket_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def load_source_classic_rows(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], str]:
    if args.classic_rows_path.exists():
        return load_bucket_rows(args.classic_rows_path), str(args.classic_rows_path)
    report_rows = parse_report_bucket_rows(args.report_path)
    return [row for row in report_rows if row.get("bucket") == BUCKET], str(
        args.report_path
    )


def build_policy(legal_moves: list[int], reference_move: int) -> list[float]:
    policy = [0.0] * 6
    if reference_move not in legal_moves:
        raise ValueError(f"reference move {reference_move} is not legal")
    if len(legal_moves) == 1:
        policy[reference_move] = 1.0
        return policy
    residual = NON_REFERENCE_POLICY_MASS / float(len(legal_moves) - 1)
    for move in legal_moves:
        policy[move] = residual
    policy[reference_move] = REFERENCE_POLICY_MASS
    return policy


def duplicate_conflicts_by_canonical(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_canonical: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        canonical = str(
            row.get("canonical_state_hash") or row.get("canonical_state") or ""
        )
        if canonical:
            by_canonical.setdefault(canonical, []).append(row)
    return {key: value for key, value in by_canonical.items() if len(value) > 1}


def format_validation_note(reasons: list[str]) -> str:
    return ", ".join(reasons) if reasons else "ok"


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    source_rows, source_path = load_source_classic_rows(args)
    source_by_id = {str(row["row_id"]): row for row in source_rows if row.get("row_id")}
    puct_rows = load_bucket_rows(args.puct_rows_path)
    excluded_rows = load_bucket_rows(args.excluded_rows_path)
    puct_row_ids = {str(row["row_id"]) for row in puct_rows if row.get("row_id")}
    excluded_row_ids = {
        str(row["row_id"]) for row in excluded_rows if row.get("row_id")
    }
    other_bucket_duplicates = duplicate_conflicts_by_canonical(
        [*puct_rows, *excluded_rows]
    )

    suite = load_suite(args.suite_path)
    suite_by_id = {position.id: position for position in suite}
    reference_payload = load_json(args.reference_path)
    reference_by_id = canonical_reference_rows(reference_payload)
    reference_by_canonical = {
        str(row.get("canonical_state") or canonical_state_key(row["state"])): row
        for row in list(reference_payload.get("rows") or [])
        if isinstance(row, dict) and row.get("state") is not None
    }

    current_metadata = load_json(args.current_artifact / "metadata.json")
    input_encoding = str(current_metadata.get("input_encoding") or args.input_encoding)

    artifact_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    excluded_rows_for_artifact: list[dict[str, Any]] = []
    canonical_targets: dict[str, tuple[int, tuple[float, ...]]] = {}

    for row_id in EXPECTED_ROW_IDS:
        reasons: list[str] = []
        source_row = source_by_id.get(row_id, {})
        suite_row = suite_by_id.get(row_id)
        if suite_row is None:
            reasons.append("missing_canonical_state")
            canonical = None
            legal_moves: list[int] = []
            raw_state: dict[str, Any] | None = None
        else:
            canonical = canonical_state_key(suite_row.state)
            legal_moves = [int(move) for move in suite_row.legal_moves]
            raw_state = dict(suite_row.state)

        reference_row = None
        if row_id in reference_by_id:
            reference_row = reference_by_id[row_id]
        elif canonical is not None:
            reference_row = reference_by_canonical.get(canonical)
        if reference_row is None:
            reasons.append("missing_active_reference")

        active_reference_move = None
        active_reference_legal = False
        teacher_value = None
        value_source = None
        if reference_row is not None:
            active_reference_move = int(reference_row["reference_move"])
            active_reference_legal = active_reference_move in legal_moves
            teacher_value = reference_row.get("teacher_value")
            value_source = str(reference_row.get("reference_source") or "unknown")
            if not active_reference_legal:
                reasons.append("active_reference_move_illegal")
            if teacher_value is None:
                reasons.append("value_target_unavailable")

        preferred_teacher = str(source_row.get("preferred_teacher") or "classic_mcts")
        preferred_move = source_row.get("preferred_move")
        if preferred_move is None and active_reference_move is not None:
            preferred_move = int(active_reference_move)
        if preferred_move is None:
            reasons.append("preferred_move_missing")
        else:
            preferred_move = int(preferred_move)
            if preferred_move not in legal_moves:
                reasons.append("preferred_move_illegal")
        if preferred_teacher != "classic_mcts":
            reasons.append("preferred_teacher_not_classic")
        if (
            preferred_move is not None
            and active_reference_move is not None
            and preferred_move != active_reference_move
        ):
            reasons.append("preferred_move_differs_from_active_reference")

        if row_id in puct_row_ids:
            reasons.append("row_present_in_puct_bucket")
        if row_id in excluded_row_ids:
            reasons.append("row_present_in_excluded_bucket")

        if canonical is not None and canonical in other_bucket_duplicates:
            reasons.append("duplicate_canonical_present_in_other_bucket")

        role = TARGET_ROLE_BY_ID[row_id]
        evidence_summary = str(
            source_row.get("evidence_summary")
            or source_row.get("notes")
            or "validated against active references"
        )

        validation_payload = {
            "row_id": row_id,
            "role": role,
            "canonical_state_exists": suite_row is not None,
            "active_reference_exists": reference_row is not None,
            "active_reference_legal": active_reference_legal,
            "preferred_move_equals_active_reference": (
                preferred_move == active_reference_move
                if preferred_move is not None and active_reference_move is not None
                else False
            ),
            "in_puct_bucket": row_id in puct_row_ids,
            "in_excluded_bucket": row_id in excluded_row_ids,
            "status": "excluded" if reasons else "ok",
            "notes": format_validation_note(reasons),
        }

        if (
            reasons
            or raw_state is None
            or active_reference_move is None
            or preferred_move is None
        ):
            validation_rows.append(validation_payload)
            excluded_rows_for_artifact.append(
                {
                    "row_id": row_id,
                    "role": role,
                    "status": "excluded",
                    "notes": format_validation_note(reasons),
                }
            )
            continue

        assert teacher_value is not None
        teacher_value_float = float(teacher_value)
        policy = build_policy(legal_moves, active_reference_move)
        canonical_target_key = (
            active_reference_move,
            tuple(round(float(value), 8) for value in policy),
        )
        canonical_text = str(canonical)
        existing_target_key = canonical_targets.get(canonical_text)
        if (
            existing_target_key is not None
            and existing_target_key != canonical_target_key
        ):
            validation_payload["status"] = "excluded"
            validation_payload["notes"] = "duplicate_canonical_state_conflict"
            validation_rows.append(validation_payload)
            excluded_rows_for_artifact.append(
                {
                    "row_id": row_id,
                    "role": role,
                    "status": "excluded",
                    "notes": "duplicate_canonical_state_conflict",
                }
            )
            continue
        canonical_targets[canonical_text] = canonical_target_key

        artifact_row = {
            "canonical_state": canonical_text,
            "state": encode_state(raw_state, input_encoding=input_encoding),
            "raw_state": raw_state,
            "side_to_move": int(raw_state["current_player"]),
            "legal_moves": legal_moves,
            "policy": policy,
            "value": teacher_value_float,
            "input_encoding": input_encoding,
            "policy_target_mode": args.policy_target_mode,
            "policy_target_actual_mode": args.policy_target_mode,
            "value_target_mode": args.value_target_mode,
            "source": SOURCE_NAME,
            "family": FAMILY,
            "bucket": BUCKET,
            "role": role,
            "active_reference_move": active_reference_move,
            "preferred_teacher": preferred_teacher,
            "preferred_move": preferred_move,
            "do_not_mix_with_puct_teacher": True,
            "train_only": True,
            "exclude_from_validation": True,
            "evidence_summary": evidence_summary,
            "value_source": value_source,
            "reference_move": active_reference_move,
            "teacher_selected_move": active_reference_move,
            "teacher_child_stats": list((reference_row or {}).get("child_stats") or []),
            "teacher_policy_simulations": int(
                reference_payload.get("reference", {}).get("policy_simulations", 0)
            ),
            "teacher_value_simulations": int(
                reference_payload.get("reference", {}).get("value_simulations", 0)
            ),
            "source_artifacts": [str(args.reference_path), source_path],
            "source_runs": [
                {
                    "kind": SOURCE_NAME,
                    "id": row_id,
                    "role": role,
                    "bucket": BUCKET,
                }
            ],
            "selection_reasons": [role, "classic_teacher_diagnostic"],
            "policy_target_reference_mass": REFERENCE_POLICY_MASS,
            "policy_target_non_reference_mass": round_float(
                NON_REFERENCE_POLICY_MASS, 6
            ),
            "priority_score": 100.0 if role == "target_candidate" else 10.0,
        }
        artifact_rows.append(artifact_row)
        validation_rows.append(validation_payload)
        if role == "target_candidate":
            target_rows.append(artifact_row)
        else:
            control_rows.append(artifact_row)

    notes: list[str] = []
    duplicate_conflicts = 0
    for row in artifact_rows:
        policy = [float(value) for value in row["policy"]]
        legal_moves = [int(move) for move in row["legal_moves"]]
        reference_move = int(row["active_reference_move"])
        if not math.isclose(sum(policy), 1.0, rel_tol=0.0, abs_tol=1e-6):
            notes.append(f"policy_not_normalized_{row['source_runs'][0]['id']}")
        if top_policy_move(policy, legal_moves) != reference_move:
            notes.append(f"reference_not_top_policy_{row['source_runs'][0]['id']}")
    canonical_map: dict[str, tuple[int, tuple[float, ...]]] = {}
    for row in artifact_rows:
        canonical = str(row["canonical_state"])
        target_key = (
            int(row["active_reference_move"]),
            tuple(round(float(value), 8) for value in row["policy"]),
        )
        existing = canonical_map.get(canonical)
        if existing is not None and existing != target_key:
            duplicate_conflicts += 1
        canonical_map[canonical] = target_key

    validation_status = "ok"
    if duplicate_conflicts > 0 or notes:
        validation_status = "invalid"

    if len(target_rows) < 3:
        classification = "artifact_not_enough_signal"
        next_action = "stop before training trace; fewer than 3 Classic target candidates remained after validation"
    else:
        classification = "artifact_ready_for_trace"
        next_action = (
            "run the tiny Classic-only diagnostic traces against the current checkpoint"
        )

    summary = {
        "schema": SCHEMA,
        "artifact_path": str(args.artifact_out),
        "summary_path": str(args.summary_out),
        "target_only_path": str(args.target_only_out),
        "control_only_path": str(args.control_only_out),
        "inputs": {
            "reference_path": str(args.reference_path),
            "suite_path": str(args.suite_path),
            "current_artifact": str(args.current_artifact),
            "classic_rows_path": str(args.classic_rows_path),
            "puct_rows_path": str(args.puct_rows_path),
            "excluded_rows_path": str(args.excluded_rows_path),
            "bucket_summary_path": str(args.bucket_summary_path),
            "report_path": str(args.report_path),
            "source_rows_loaded_from": source_path,
        },
        "row_ids": list(EXPECTED_ROW_IDS),
        "artifact_rows": [
            {
                "row_id": row["source_runs"][0]["id"],
                "role": row["role"],
                "active_reference_move": row["active_reference_move"],
                "preferred_teacher": row["preferred_teacher"],
                "preferred_move": row["preferred_move"],
                "policy_target_reference_mass": row["policy_target_reference_mass"],
                "value_source": row["value_source"],
                "status": "ok",
                "notes": row["evidence_summary"],
            }
            for row in artifact_rows
        ],
        "validation_rows": validation_rows,
        "excluded_rows": excluded_rows_for_artifact,
        "validation": {
            "status": validation_status,
            "row_count": len(artifact_rows),
            "target_candidate_count": len(target_rows),
            "preservation_control_count": len(control_rows),
            "duplicate_conflicts": duplicate_conflicts,
            "all_active_reference_moves_legal": all(
                row["active_reference_move"] in row["legal_moves"]
                for row in artifact_rows
            ),
            "all_policy_targets_normalized": not any(
                note.startswith("policy_not_normalized_") for note in notes
            ),
            "all_reference_moves_top_policy": not any(
                note.startswith("reference_not_top_policy_") for note in notes
            ),
            "no_puct_rows_included": all(
                row["source_runs"][0]["id"] not in puct_row_ids for row in artifact_rows
            ),
            "no_excluded_rows_included": all(
                row["source_runs"][0]["id"] not in excluded_row_ids
                for row in artifact_rows
            ),
            "notes": notes or ["ok"],
        },
        "classification": classification,
        "next_action": next_action,
    }
    return {
        "summary": summary,
        "artifact_rows": artifact_rows,
        "target_rows": target_rows,
        "control_rows": control_rows,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    built = build_artifact(args)
    write_jsonl(args.artifact_out, built["artifact_rows"])
    write_jsonl(args.target_only_out, built["target_rows"])
    write_jsonl(args.control_only_out, built["control_rows"])
    write_json(args.summary_out, built["summary"])
    print(
        json.dumps(
            {
                "artifact_path": str(args.artifact_out),
                "summary_path": str(args.summary_out),
                "row_count": len(built["artifact_rows"]),
                "target_candidate_count": len(built["target_rows"]),
                "preservation_control_count": len(built["control_rows"]),
                "classification": built["summary"]["classification"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
