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

from ml.alphazero_lite.build_tracked_opening_capture_policy_artifact import (
    EXTRA_TURN_REPLAY_ROLE,
    NO_EXTRA_TURN_REPLAY_ROLE,
    encode_raw_state,
    replay_role_for_reference_move,
)
from ml.alphazero_lite.kalah_rules import KalahGame


SCHEMA = "azlite_capture_002_003_rule_collision_guard_artifact_summary_v1"
ROW_IDS = ["capture_available-002", "capture_available-003"]
RULE_COLLISION_EXTRA_TURN_GUARD_ROLE = "rule_collision_extra_turn_reference_guard"
RULE_COLLISION_NO_EXTRA_TURN_GUARD_ROLE = "rule_collision_no_extra_turn_reference_guard"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-artifact", required=True)
    parser.add_argument("--rule-collision-diagnostic", required=True)
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


def derive_policy(child_stats: list[dict[str, Any]]) -> list[float]:
    policy = [0.0] * 6
    total_visits = sum(int(child.get("visits", 0)) for child in child_stats)
    if total_visits <= 0:
        raise ValueError("reference child_stats must contain visits")
    for child in child_stats:
        move = int(child["move"])
        policy[move] = float(child["visits"]) / float(total_visits)
    return policy


def legal_moves_for_state(raw_state: dict[str, Any]) -> list[int]:
    return list(KalahGame.from_state(raw_state).possible_moves())


def guard_replay_role(*, raw_state: dict[str, Any], reference_move: int) -> str:
    replay_role = replay_role_for_reference_move(
        raw_state=raw_state, reference_move=reference_move
    )
    if replay_role == EXTRA_TURN_REPLAY_ROLE:
        return RULE_COLLISION_EXTRA_TURN_GUARD_ROLE
    return RULE_COLLISION_NO_EXTRA_TURN_GUARD_ROLE


def build_rows(
    *,
    reference_artifact: dict,
    rule_collision_diagnostic: dict,
    reference_artifact_path: str,
    rule_collision_diagnostic_path: str,
    input_encoding: str,
    policy_target_mode: str,
    value_target_mode: str,
    state_encoder: Callable[[dict[str, Any], str], Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_rows = rows_by_id(list(reference_artifact["rows"]))
    diagnostic_rows = rule_collision_diagnostic.get("rows") or {}
    resolved_state_encoder = encode_raw_state if state_encoder is None else state_encoder

    rows: list[dict[str, Any]] = []
    replay_role_counts: Counter[str] = Counter()
    teacher_move_counts: Counter[int] = Counter()
    row_ids_by_replay_role: dict[str, list[str]] = {}
    teacher_move_counts_by_replay_role: dict[str, Counter[int]] = {}
    reference_extra_turn_by_row: dict[str, bool] = {}

    for row_id in ROW_IDS:
        reference_row = reference_rows[row_id]
        diagnostic_row = diagnostic_rows[row_id]
        raw_state = dict(reference_row["state"])
        reference_move = int(reference_row["reference_move"])
        replay_role = guard_replay_role(raw_state=raw_state, reference_move=reference_move)
        reference_extra_turn_available = (
            replay_role == RULE_COLLISION_EXTRA_TURN_GUARD_ROLE
        )

        row_payload = {
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
                "capture_002_003_rule_collision_guard",
                "extra_turn_overvaluation_guard",
            ],
            "source_artifacts": [
                str(reference_artifact_path),
                str(rule_collision_diagnostic_path),
            ],
            "source_runs": [
                {
                    "kind": "rule_collision_guard",
                    "id": row_id,
                    "reference_move": reference_move,
                }
            ],
            "priority_score": 25.0,
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
            "replay_role": replay_role,
            "reference_move": reference_move,
            "reference_move_extra_turn_available": reference_extra_turn_available,
            "tracked_selected_move": int(
                diagnostic_row["search_runs"]["tracked"]["searched_selected_move"]
            ),
            "broader_selected_move": int(
                diagnostic_row["search_runs"]["broader"]["searched_selected_move"]
            ),
        }
        rows.append(row_payload)
        replay_role_counts[replay_role] += 1
        teacher_move_counts[reference_move] += 1
        row_ids_by_replay_role.setdefault(replay_role, []).append(row_id)
        teacher_move_counts_by_replay_role.setdefault(replay_role, Counter())[reference_move] += 1
        reference_extra_turn_by_row[row_id] = reference_extra_turn_available

    summary = {
        "schema": SCHEMA,
        "row_count": len(rows),
        "row_ids": list(ROW_IDS),
        "replay_role_counts": dict(sorted(replay_role_counts.items())),
        "teacher_selected_move_distribution": dict(sorted(teacher_move_counts.items())),
        "row_ids_by_replay_role": {
            role: row_ids_by_replay_role[role] for role in sorted(row_ids_by_replay_role)
        },
        "teacher_selected_move_distribution_by_replay_role": {
            role: dict(sorted(teacher_move_counts_by_replay_role[role].items()))
            for role in sorted(teacher_move_counts_by_replay_role)
        },
        "reference_extra_turn_by_row": dict(sorted(reference_extra_turn_by_row.items())),
    }
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reference_artifact = load_json(Path(args.reference_artifact))
    rule_collision_diagnostic = load_json(Path(args.rule_collision_diagnostic))
    rows, summary = build_rows(
        reference_artifact=reference_artifact,
        rule_collision_diagnostic=rule_collision_diagnostic,
        reference_artifact_path=args.reference_artifact,
        rule_collision_diagnostic_path=args.rule_collision_diagnostic,
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
