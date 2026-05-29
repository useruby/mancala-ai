#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.build_capture_002_003_rule_collision_guard_artifact import (
    ROW_IDS,
    derive_policy,
    legal_moves_for_state,
)
from ml.alphazero_lite.build_tracked_opening_capture_policy_artifact import (
    encode_raw_state,
)


SCHEMA = "azlite_capture_002_003_guarded_w2_prior_calibration_artifact_summary_v1"
ROW_MULTIPLICITY = {
    "capture_available-002": 3,
    "capture_available-003": 1,
}
REPLAY_ROLE = "guarded_w2_prior_calibration"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-artifact", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--input-encoding", default="kalah_v3")
    parser.add_argument("--policy-target-mode", default="sharpened")
    parser.add_argument("--value-target-mode", default="sharpened")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in rows}


def build_rows(
    *,
    reference_artifact: dict,
    reference_artifact_path: str,
    input_encoding: str,
    policy_target_mode: str,
    value_target_mode: str,
    state_encoder: Callable[[dict[str, Any], str], Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_rows = rows_by_id(list(reference_artifact["rows"]))
    resolved_state_encoder = encode_raw_state if state_encoder is None else state_encoder

    rows: list[dict[str, Any]] = []
    repeated_row_ids: list[str] = []
    multiplicity_by_row = dict(ROW_MULTIPLICITY)
    teacher_move_counts: Counter[int] = Counter()
    row_counts: Counter[str] = Counter()

    for row_id in ROW_IDS:
        reference_row = reference_rows[row_id]
        raw_state = dict(reference_row["state"])
        reference_move = int(reference_row["reference_move"])
        multiplicity = int(ROW_MULTIPLICITY[row_id])

        base_payload = {
            "canonical_state": str(reference_row["canonical_state"]),
            "state": resolved_state_encoder(
                raw_state=raw_state, input_encoding=input_encoding
            ),
            "raw_state": raw_state,
            "side_to_move": int(raw_state["current_player"]),
            "legal_moves": legal_moves_for_state(raw_state),
            "policy": derive_policy(list(reference_row["child_stats"])),
            "value": float(reference_row["teacher_value"]),
            "bucket": "capture_available",
            "bucket_group": "tactical",
            "input_encoding": input_encoding,
            "policy_target_mode": policy_target_mode,
            "value_target_mode": value_target_mode,
            "selection_reasons": [
                "guarded_w2_prior_calibration",
                "policy_prior_sensitive_row_pair",
                "row_002_weighted_reference_support" if row_id == "capture_available-002" else "row_003_preservation_guard",
            ],
            "source_artifacts": [str(reference_artifact_path)],
            "source_runs": [
                {
                    "kind": "guarded_w2_prior_calibration",
                    "id": row_id,
                    "reference_move": reference_move,
                    "copy_count": multiplicity,
                }
            ],
            "priority_score": 30.0 if row_id == "capture_available-002" else 20.0,
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
            "teacher_child_stats": list(reference_row["child_stats"]),
            "replay_role": REPLAY_ROLE,
            "reference_move": reference_move,
            "row_weight_intent": "optimize" if row_id == "capture_available-002" else "preserve",
        }

        for copy_index in range(multiplicity):
            row_payload = json.loads(json.dumps(base_payload))
            row_payload["source_runs"][0]["copy_index"] = copy_index + 1
            rows.append(row_payload)
            repeated_row_ids.append(row_id)
            teacher_move_counts[reference_move] += 1
            row_counts[row_id] += 1

    summary = {
        "schema": SCHEMA,
        "row_count": len(rows),
        "unique_row_ids": list(ROW_IDS),
        "repeated_row_ids": repeated_row_ids,
        "multiplicity_by_row": multiplicity_by_row,
        "replay_role_counts": {REPLAY_ROLE: len(rows)},
        "row_counts": dict(sorted(row_counts.items())),
        "teacher_selected_move_distribution": dict(sorted(teacher_move_counts.items())),
        "reference_artifact_path": str(reference_artifact_path),
    }
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reference_artifact = load_json(Path(args.reference_artifact))
    rows, summary = build_rows(
        reference_artifact=reference_artifact,
        reference_artifact_path=args.reference_artifact,
        input_encoding=args.input_encoding,
        policy_target_mode=args.policy_target_mode,
        value_target_mode=args.value_target_mode,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )

    out_summary_path = Path(args.out_summary)
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    out_summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
