#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, cast

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.build_tracked_opening_capture_policy_artifact import (
    derive_policy,
    encode_raw_state,
    priority_score,
    replay_role_for_reference_move,
)


DEFAULT_INPUT_ENCODING = "kalah_v3"
DEFAULT_POLICY_TARGET_MODE = "sharpened"
DEFAULT_VALUE_TARGET_MODE = "sharpened"
OPENING_BUCKET = "opening_plies_1_8"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-forensics", required=True)
    parser.add_argument("--reference-artifact")
    parser.add_argument("--out", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--input-encoding", default=DEFAULT_INPUT_ENCODING)
    parser.add_argument("--policy-target-mode", default=DEFAULT_POLICY_TARGET_MODE)
    parser.add_argument("--value-target-mode", default=DEFAULT_VALUE_TARGET_MODE)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in rows}


def opening_replay_role(*, raw_state: dict[str, Any], reference_move: int) -> str:
    base_role = replay_role_for_reference_move(
        raw_state=raw_state,
        reference_move=reference_move,
    )
    if base_role.endswith("extra_turn_reference"):
        return "opening_plies_extra_turn_reference"
    return "opening_plies_no_extra_turn_reference"


def build_artifact_rows(
    *,
    candidate_forensics: dict,
    reference_artifact: dict,
    reference_artifact_path: str,
    input_encoding: str,
    policy_target_mode: str,
    value_target_mode: str,
    state_encoder: Callable[[dict[str, Any], str], Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_rows = rows_by_id(candidate_forensics["systems"]["current"]["rows"])
    reference_rows = rows_by_id(reference_artifact["rows"])
    artifact_rows: list[dict[str, Any]] = []
    replay_role_counts: Counter[str] = Counter()
    teacher_move_counts: Counter[int] = Counter()
    row_ids_by_replay_role: dict[str, list[str]] = {}
    teacher_move_counts_by_replay_role: dict[str, Counter[int]] = {}
    reference_extra_turn_by_row: dict[str, bool] = {}
    resolved_state_encoder = (
        (
            lambda raw_state, input_encoding: encode_raw_state(
                raw_state=raw_state,
                input_encoding=input_encoding,
            )
        )
        if state_encoder is None
        else state_encoder
    )

    for row_id in sorted(candidate_rows):
        row = candidate_rows[row_id]
        if row.get("bucket") != OPENING_BUCKET:
            continue
        reference_row = reference_rows.get(row_id)
        if reference_row is None:
            continue
        reference_move = row.get("reference_move")
        selected_move = row.get("selected_move")
        if (
            isinstance(reference_move, bool)
            or not isinstance(reference_move, int)
            or isinstance(selected_move, bool)
            or not isinstance(selected_move, int)
            or selected_move == reference_move
        ):
            continue

        teacher_child_stats = []
        for child in reference_row.get("child_stats") or []:
            if not isinstance(child, dict):
                continue
            teacher_child_stats.append(
                {
                    "move": int(child["move"]),
                    "visits": int(child.get("visits", 0)),
                    "win_rate": float(child.get("win_rate", 0.0)),
                }
            )
        if not teacher_child_stats:
            continue

        raw_state = dict(row["state"])
        replay_role = opening_replay_role(
            raw_state=raw_state,
            reference_move=reference_move,
        )
        reference_move_extra_turn_available = replay_role.endswith(
            "extra_turn_reference"
        )
        artifact_rows.append(
            {
                "canonical_state": str(row["canonical_state"]),
                "state": resolved_state_encoder(raw_state, input_encoding),
                "raw_state": raw_state,
                "side_to_move": int(row["side_to_move"]),
                "legal_moves": list(row["legal_moves"]),
                "policy": derive_policy(teacher_child_stats),
                "value": float(reference_row["teacher_value"]),
                "bucket": OPENING_BUCKET,
                "bucket_group": "preservation",
                "input_encoding": input_encoding,
                "policy_target_mode": policy_target_mode,
                "value_target_mode": value_target_mode,
                "selection_reasons": [
                    "opening_plies_policy_target",
                    "corrected_reference_failure",
                ],
                "source_artifacts": [
                    "forensic_suite_validation.json",
                    str(reference_artifact_path),
                ],
                "source_runs": [
                    {
                        "kind": "opening_plies_family",
                        "id": row_id,
                        "reference_move": reference_move,
                    }
                ],
                "priority_score": priority_score(row),
                "teacher_policy_simulations": int(
                    reference_artifact["reference"]["policy_simulations"]
                ),
                "teacher_value_simulations": int(
                    reference_artifact["reference"]["value_simulations"]
                ),
                "teacher_seed": int(reference_artifact["reference"]["sample_seeds"][0]),
                "teacher_policy_seed": int(
                    reference_artifact["reference"]["sample_seeds"][0]
                ),
                "teacher_value_seed": int(
                    reference_artifact["reference"]["sample_seeds"][0]
                ),
                "teacher_selected_move": reference_move,
                "teacher_child_stats": teacher_child_stats,
                "replay_role": replay_role,
                "reference_move_extra_turn_available": reference_move_extra_turn_available,
                "reference_move": reference_move,
                "baseline_search_move": int(selected_move),
            }
        )
        replay_role_counts[replay_role] += 1
        teacher_move_counts[reference_move] += 1
        row_ids_by_replay_role.setdefault(replay_role, []).append(row_id)
        teacher_move_counts_by_replay_role.setdefault(replay_role, Counter())[
            reference_move
        ] += 1
        reference_extra_turn_by_row[row_id] = reference_move_extra_turn_available

    summary = {
        "schema": "azlite_opening_plies_policy_artifact_summary_v1",
        "row_count": len(artifact_rows),
        "teacher_selected_move_distribution": dict(sorted(teacher_move_counts.items())),
        "row_ids": [row["source_runs"][0]["id"] for row in artifact_rows],
        "replay_role_counts": dict(sorted(replay_role_counts.items())),
        "row_ids_by_replay_role": {
            role: row_ids_by_replay_role[role]
            for role in sorted(row_ids_by_replay_role)
        },
        "teacher_selected_move_distribution_by_replay_role": {
            role: dict(sorted(teacher_move_counts_by_replay_role[role].items()))
            for role in sorted(teacher_move_counts_by_replay_role)
        },
        "reference_extra_turn_by_row": dict(
            sorted(reference_extra_turn_by_row.items())
        ),
    }
    return artifact_rows, summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    candidate_forensics_raw = load_json(Path(args.candidate_forensics))
    if not isinstance(candidate_forensics_raw, dict):
        raise SystemExit("candidate forensics must be a JSON object")
    inferred_reference_artifact_path = args.reference_artifact or str(
        candidate_forensics_raw["reference"]["artifact_path"]
    )
    reference_artifact_raw = load_json(Path(inferred_reference_artifact_path))
    if not isinstance(reference_artifact_raw, dict):
        raise SystemExit("reference artifact must be a JSON object")
    artifact_rows, summary = build_artifact_rows(
        candidate_forensics=cast(dict[str, Any], candidate_forensics_raw),
        reference_artifact=cast(dict[str, Any], reference_artifact_raw),
        reference_artifact_path=inferred_reference_artifact_path,
        input_encoding=args.input_encoding,
        policy_target_mode=args.policy_target_mode,
        value_target_mode=args.value_target_mode,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(row) for row in artifact_rows)
        + ("\n" if artifact_rows else ""),
        encoding="utf-8",
    )

    out_summary_path = Path(args.out_summary)
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    out_summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "rows": len(artifact_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
