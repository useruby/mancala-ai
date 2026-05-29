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

from ml.alphazero_lite.capture_002_003_rule_collision_diagnostic import (
    simulate_move_rule_features,
)


DEFAULT_INPUT_ENCODING = "kalah_v3"
DEFAULT_POLICY_TARGET_MODE = "sharpened"
DEFAULT_VALUE_TARGET_MODE = "sharpened"
EXTRA_TURN_REPLAY_ROLE = "opening_capture_extra_turn_reference"
NO_EXTRA_TURN_REPLAY_ROLE = "opening_capture_no_extra_turn_reference"
TACTICAL_BUCKETS = frozenset(
    {
        "capture_available",
        "high_imbalance",
        "high_value_swing",
        "early_extra_turn",
    }
)
PRESERVATION_BUCKETS = frozenset(
    {
        "opening_plies_1_8",
        "sparse_endgame",
        "starvation_pressure",
    }
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-forensics", required=True)
    parser.add_argument("--reference-artifact", required=True)
    parser.add_argument("--move-selection-summary", required=True)
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


def derive_policy(child_stats: list[dict[str, Any]]) -> list[float]:
    policy = [0.0] * 6
    total_visits = sum(int(child.get("visits", 0)) for child in child_stats)
    if total_visits <= 0:
        raise ValueError("reference child_stats must contain visits")
    for child in child_stats:
        move = int(child["move"])
        policy[move] = float(child["visits"]) / float(total_visits)
    return policy


def priority_score(row: dict[str, Any]) -> float:
    regret = float(row.get("regret") or 0.0)
    value_error = float(row.get("value_error") or 0.0)
    return round(10.0 + (5.0 * regret) + (5.0 * value_error), 4)


def encode_raw_state(*, raw_state: dict[str, Any], input_encoding: str):
    from ml.alphazero_lite import self_play

    return self_play.encode_state(raw_state, input_encoding=input_encoding)


def replay_role_for_reference_move(*, raw_state: dict[str, Any], reference_move: int) -> str:
    move_features = simulate_move_rule_features(state=raw_state, move=reference_move)
    if move_features["extra_turn_available"]:
        return EXTRA_TURN_REPLAY_ROLE
    return NO_EXTRA_TURN_REPLAY_ROLE


def bucket_group(bucket: str) -> str:
    if bucket in TACTICAL_BUCKETS:
        return "tactical"
    if bucket in PRESERVATION_BUCKETS:
        return "preservation"
    raise ValueError(f"unsupported bucket for train split: {bucket}")


def build_artifact_rows(
    *,
    candidate_forensics: dict,
    reference_artifact: dict,
    move_selection_summary: list[dict[str, Any]],
    reference_artifact_path: str,
    input_encoding: str,
    policy_target_mode: str,
    value_target_mode: str,
    state_encoder: Callable[[dict[str, Any], str], Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_rows = rows_by_id(candidate_forensics["systems"]["challenger"]["rows"])
    reference_rows = rows_by_id(reference_artifact["rows"])
    artifact_rows: list[dict[str, Any]] = []
    mechanism_counts: Counter[str] = Counter()
    teacher_move_counts: Counter[int] = Counter()
    replay_role_counts: Counter[str] = Counter()
    row_ids_by_replay_role: dict[str, list[str]] = {
        EXTRA_TURN_REPLAY_ROLE: [],
        NO_EXTRA_TURN_REPLAY_ROLE: [],
    }
    teacher_move_counts_by_replay_role: dict[str, Counter[int]] = {
        EXTRA_TURN_REPLAY_ROLE: Counter(),
        NO_EXTRA_TURN_REPLAY_ROLE: Counter(),
    }
    reference_extra_turn_by_row: dict[str, bool] = {}
    resolved_state_encoder = encode_raw_state if state_encoder is None else state_encoder

    for summary_row in move_selection_summary:
        row_id = str(summary_row["id"])
        candidate_row = candidate_rows[row_id]
        reference_row = reference_rows[row_id]
        reference_move = int(reference_row["reference_move"])
        policy = derive_policy(list(reference_row["child_stats"]))
        bucket = str(candidate_row["bucket"])
        raw_state = dict(candidate_row["state"])
        replay_role = replay_role_for_reference_move(
            raw_state=raw_state, reference_move=reference_move
        )
        reference_extra_turn_available = replay_role == EXTRA_TURN_REPLAY_ROLE

        if summary_row["baseline_prior"] != reference_move and summary_row["baseline_search"] != reference_move:
            mechanism = "wrong_family_unrecovered"
        elif summary_row["baseline_prior"] != reference_move and summary_row["baseline_search"] == reference_move:
            mechanism = "search_rescued_wrong_family"
        elif summary_row["baseline_prior"] == reference_move and summary_row["baseline_search"] == reference_move:
            mechanism = "correct_family_preserved"
        else:
            mechanism = "other"

        artifact_rows.append(
            {
                "canonical_state": str(candidate_row["canonical_state"]),
                "state": resolved_state_encoder(
                    raw_state=raw_state, input_encoding=input_encoding
                ),
                "raw_state": raw_state,
                "side_to_move": int(candidate_row["side_to_move"]),
                "legal_moves": list(candidate_row["legal_moves"]),
                "policy": policy,
                "value": float(reference_row["teacher_value"]),
                "bucket": bucket,
                "bucket_group": bucket_group(bucket),
                "input_encoding": input_encoding,
                "policy_target_mode": policy_target_mode,
                "value_target_mode": value_target_mode,
                "selection_reasons": [
                    "tracked_opening_capture_policy_target",
                    mechanism,
                ],
                "source_artifacts": [
                    "batch1-opening-capture-diagnostic",
                    str(reference_artifact_path),
                ],
                "source_runs": [
                    {
                        "kind": "tracked_opening_capture_family",
                        "id": row_id,
                        "reference_move": reference_move,
                    }
                ],
                "priority_score": priority_score(candidate_row),
                "teacher_policy_simulations": int(reference_artifact["reference"]["policy_simulations"]),
                "teacher_value_simulations": int(reference_artifact["reference"]["value_simulations"]),
                "teacher_seed": int(reference_artifact["reference"]["sample_seeds"][0]),
                "teacher_policy_seed": int(reference_artifact["reference"]["sample_seeds"][0]),
                "teacher_value_seed": int(reference_artifact["reference"]["sample_seeds"][0]),
                "teacher_selected_move": reference_move,
                "teacher_child_stats": list(reference_row["child_stats"]),
                "replay_role": replay_role,
                "reference_move_extra_turn_available": reference_extra_turn_available,
                "reference_move": reference_move,
                "baseline_prior_move": int(summary_row["baseline_prior"]),
                "baseline_search_move": int(summary_row["baseline_search"]),
                "w2_prior_move": int(summary_row["w2_prior"]),
                "w2_search_move": int(summary_row["w2_search"]),
            }
        )
        mechanism_counts[mechanism] += 1
        teacher_move_counts[reference_move] += 1
        replay_role_counts[replay_role] += 1
        row_ids_by_replay_role[replay_role].append(row_id)
        teacher_move_counts_by_replay_role[replay_role][reference_move] += 1
        reference_extra_turn_by_row[row_id] = reference_extra_turn_available

    summary = {
        "schema": "azlite_tracked_opening_capture_policy_artifact_summary_v1",
        "row_count": len(artifact_rows),
        "teacher_selected_move_distribution": dict(sorted(teacher_move_counts.items())),
        "mechanism_counts": dict(sorted(mechanism_counts.items())),
        "row_ids": [row["source_runs"][0]["id"] for row in artifact_rows],
        "replay_role_counts": dict(sorted(replay_role_counts.items())),
        "row_ids_by_replay_role": {
            role: row_ids_by_replay_role[role]
            for role in sorted(row_ids_by_replay_role)
            if row_ids_by_replay_role[role]
        },
        "teacher_selected_move_distribution_by_replay_role": {
            role: dict(sorted(teacher_move_counts_by_replay_role[role].items()))
            for role in sorted(teacher_move_counts_by_replay_role)
            if teacher_move_counts_by_replay_role[role]
        },
        "reference_extra_turn_by_row": dict(sorted(reference_extra_turn_by_row.items())),
    }
    return artifact_rows, summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    candidate_forensics = load_json(Path(args.candidate_forensics))
    reference_artifact = load_json(Path(args.reference_artifact))
    move_selection_summary = load_json(Path(args.move_selection_summary))
    if not isinstance(move_selection_summary, list):
        raise SystemExit("move selection summary must be a JSON list")

    artifact_rows, summary = build_artifact_rows(
        candidate_forensics=candidate_forensics,
        reference_artifact=reference_artifact,
        move_selection_summary=move_selection_summary,
        reference_artifact_path=args.reference_artifact,
        input_encoding=args.input_encoding,
        policy_target_mode=args.policy_target_mode,
        value_target_mode=args.value_target_mode,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(row) for row in artifact_rows) + ("\n" if artifact_rows else ""),
        encoding="utf-8",
    )

    out_summary_path = Path(args.out_summary)
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    out_summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "rows": len(artifact_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
