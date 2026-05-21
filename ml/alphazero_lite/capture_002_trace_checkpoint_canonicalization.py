from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

SCHEMA = "azlite_capture_002_trace_checkpoint_canonicalization_v1"
SOURCE_SELECTION_SCORE_SCHEMA = "azlite_capture_002_selection_score_trace_v1"
ROW_ID = "capture_available-002"
EXPECTED_TRACE_CLASSIFICATION = "unresolved"
EXPECTED_TRACE_DECISION = "write_002_unresolved_trace_review_spec"
CLASSIFICATION_DECISIONS = {
    "duplicate_root_snapshot_only": "write_002_metric_audit_canonical_input_spec",
    "duplicate_equivalent_checkpoint": "write_002_metric_audit_canonical_input_spec",
    "duplicate_conflicting_checkpoint": "stop_002_duplicate_checkpoint_conflict",
    "checkpoint_shape_mismatch": "stop_002_checkpoint_shape_mismatch",
    "checkpoint_canonicalization_inconclusive": "stop_002_checkpoint_canonicalization_inconclusive",
}
ALLOWED_TRACE_POINT_KEYS = {
    "moves",
    "reference_move_by_prior",
    "selected_move",
    "simulation",
    "visits",
}
ALLOWED_MOVE_ENTRY_KEYS = {"move", "q_value", "selection_score"}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonicalize duplicate checkpoints for capture 002 traces"
    )
    parser.add_argument("--source-selection-score-artifact", type=Path, required=True)
    parser.add_argument(
        "--source-threshold-relaxed-selection-score-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def _classification_name(artifact: dict, *, context: str) -> str:
    classification = artifact.get("classification")
    if not isinstance(classification, dict):
        raise ValueError(f"{context} classification must be an object")
    name = classification.get("classification")
    if not isinstance(name, str):
        raise ValueError(f"{context} classification.classification must be a string")
    return name


def _finite_number(value, *, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{context} must be finite numeric")
    return float(value)


def _finite_non_negative_number(value, *, context: str) -> float:
    number = _finite_number(value, context=context)
    if number < 0:
        raise ValueError(f"{context} must be finite non-negative")
    return number


def _validate_thresholds(thresholds, *, context: str) -> dict:
    if not isinstance(thresholds, dict):
        raise ValueError(f"{context} thresholds must be an object")
    normalized = {}
    for key in (
        "meaningful_q_margin",
        "material_selection_score_margin",
        "material_visit_share_margin",
    ):
        if key not in thresholds:
            raise ValueError(f"{context} thresholds must contain {key}")
        normalized[key] = _finite_non_negative_number(
            thresholds[key],
            context=f"{context} thresholds.{key}",
        )
    return normalized


def _validate_source_artifact(source_artifact, *, context: str) -> dict:
    if not isinstance(source_artifact, dict):
        raise ValueError(f"{context} source_artifact must be an object")
    if source_artifact.get("row_id") != ROW_ID:
        raise ValueError(f"{context} source_artifact.row_id must be {ROW_ID}")
    for key in ("reference_move", "full_search_selected_move"):
        value = source_artifact.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{context} source_artifact.{key} must be an integer")
    selected_artifact = source_artifact.get("selected_artifact")
    if not isinstance(selected_artifact, dict):
        raise ValueError(
            f"{context} source_artifact.selected_artifact must be an object"
        )
    return copy.deepcopy(source_artifact)


def _normalize_move_entry(move_entry: dict, *, context: str) -> dict:
    if not isinstance(move_entry, dict):
        raise ValueError(f"{context} must be an object")
    unexpected_keys = set(move_entry) - ALLOWED_MOVE_ENTRY_KEYS
    if unexpected_keys:
        raise ValueError(
            f"{context} contains unsupported keys: {sorted(unexpected_keys)}"
        )
    if "move" not in move_entry:
        raise ValueError(f"{context}.move is required")
    move = move_entry["move"]
    if isinstance(move, bool) or not isinstance(move, int):
        raise ValueError(f"{context}.move must be an integer")
    normalized = {"move": int(move)}
    for metric_key in ("q_value", "selection_score"):
        if metric_key in move_entry:
            normalized[metric_key] = _finite_number(
                move_entry[metric_key],
                context=f"{context}.{metric_key}",
            )
    return normalized


def _normalize_trace_point(trace_point: dict, *, context: str) -> dict:
    if not isinstance(trace_point, dict):
        raise ValueError(f"{context} must be an object")
    unexpected_keys = set(trace_point) - ALLOWED_TRACE_POINT_KEYS
    if unexpected_keys:
        raise ValueError(
            f"{context} contains unsupported keys: {sorted(unexpected_keys)}"
        )
    for key in ("simulation", "selected_move", "visits", "moves"):
        if key not in trace_point:
            raise ValueError(f"{context}.{key} is required")

    simulation = _finite_number(
        trace_point["simulation"], context=f"{context}.simulation"
    )
    selected_move = trace_point["selected_move"]
    if isinstance(selected_move, bool) or not isinstance(selected_move, int):
        raise ValueError(f"{context}.selected_move must be an integer")

    normalized = {
        "simulation": simulation,
        "selected_move": int(selected_move),
    }

    reference_move_by_prior = trace_point.get("reference_move_by_prior")
    if reference_move_by_prior is not None:
        if isinstance(reference_move_by_prior, bool) or not isinstance(
            reference_move_by_prior, int
        ):
            raise ValueError(f"{context}.reference_move_by_prior must be an integer")
        normalized["reference_move_by_prior"] = int(reference_move_by_prior)

    visits = trace_point["visits"]
    if not isinstance(visits, list) or not visits:
        raise ValueError(f"{context}.visits must be a non-empty list")
    normalized["visits"] = [
        _finite_non_negative_number(visit, context=f"{context}.visits[{index}]")
        for index, visit in enumerate(visits)
    ]

    moves = trace_point["moves"]
    if not isinstance(moves, list) or not moves:
        raise ValueError(f"{context}.moves must be a non-empty list")
    normalized_moves = [
        _normalize_move_entry(move_entry, context=f"{context}.moves[{index}]")
        for index, move_entry in enumerate(moves)
    ]
    move_ids = [move_entry["move"] for move_entry in normalized_moves]
    if len(set(move_ids)) != len(move_ids):
        raise ValueError(f"{context}.moves must have unique move identifiers")
    normalized["moves"] = normalized_moves
    return normalized


def _validate_trace_artifact(
    artifact: dict, *, context: str
) -> tuple[dict, dict, str, list[dict]]:
    if artifact.get("schema") != SOURCE_SELECTION_SCORE_SCHEMA:
        raise ValueError(
            f"selection score artifact has wrong schema: expected {SOURCE_SELECTION_SCORE_SCHEMA}"
        )
    if _classification_name(artifact, context=context) != EXPECTED_TRACE_CLASSIFICATION:
        raise ValueError(f"{context} classification must be unresolved")
    if artifact.get("decision") != EXPECTED_TRACE_DECISION:
        raise ValueError(f"{context} decision must be {EXPECTED_TRACE_DECISION}")
    if artifact.get("insufficiency_reasons") != []:
        raise ValueError(f"{context} insufficiency_reasons must be []")
    trace_origin = artifact.get("trace_origin")
    if not isinstance(trace_origin, str) or not trace_origin:
        raise ValueError(f"{context} trace_origin must be a non-empty string")
    trace_points = artifact.get("trace_points")
    if not isinstance(trace_points, list) or not trace_points:
        raise ValueError(f"{context} trace_points must be a non-empty list")
    normalized_trace_points = [
        _normalize_trace_point(trace_point, context=f"{context} trace_points[{index}]")
        for index, trace_point in enumerate(trace_points)
    ]
    return (
        _validate_source_artifact(artifact.get("source_artifact"), context=context),
        _validate_thresholds(artifact.get("thresholds"), context=context),
        trace_origin,
        normalized_trace_points,
    )


def _group_trace_points_by_simulation(
    trace_points: list[dict],
) -> tuple[list[tuple[float, list[dict]]] | None, str | None]:
    grouped: list[tuple[float, list[dict]]] = []
    seen_simulations: set[float] = set()
    for trace_point in trace_points:
        simulation = trace_point["simulation"]
        if grouped and simulation == grouped[-1][0]:
            grouped[-1][1].append(trace_point)
            continue
        if simulation in seen_simulations:
            return None, "same simulation appears in multiple non-contiguous groups"
        grouped.append((simulation, [trace_point]))
        seen_simulations.add(simulation)
    return grouped, None


def _analyze_branch_duplicates(trace_points: list[dict], *, branch: str) -> dict:
    grouped_trace_points, grouping_error = _group_trace_points_by_simulation(
        trace_points
    )
    if grouping_error is not None:
        return {
            "duplicate_summary": {
                "duplicate_count": 0,
                "duplicate_simulations": [],
                "duplicate_groups": [],
            },
            "canonical_checkpoint_sequence": None,
            "status": "inconclusive",
            "reason": f"{branch} trace cannot be canonicalized safely because {grouping_error}",
        }

    canonical_checkpoint_sequence = [
        simulation for simulation, _group in grouped_trace_points
    ]
    if any(
        current <= previous
        for previous, current in zip(
            canonical_checkpoint_sequence, canonical_checkpoint_sequence[1:]
        )
    ):
        return {
            "duplicate_summary": {
                "duplicate_count": 0,
                "duplicate_simulations": [],
                "duplicate_groups": [],
            },
            "canonical_checkpoint_sequence": None,
            "status": "inconclusive",
            "reason": f"{branch} canonical checkpoint sequence is not strictly increasing after grouping duplicates",
        }

    duplicate_groups = []
    duplicate_simulations = []
    for simulation, group in grouped_trace_points:
        if len(group) == 1:
            continue
        duplicate_simulations.append(simulation)
        first = group[0]
        if all(candidate == first for candidate in group[1:]):
            duplicate_groups.append(
                {
                    "simulation": simulation,
                    "occurrence_count": len(group),
                    "classification": "equivalent",
                }
            )
            continue
        duplicate_groups.append(
            {
                "simulation": simulation,
                "occurrence_count": len(group),
                "classification": "conflicting",
            }
        )
        return {
            "duplicate_summary": {
                "duplicate_count": len(duplicate_groups),
                "duplicate_simulations": duplicate_simulations,
                "duplicate_groups": duplicate_groups,
            },
            "canonical_checkpoint_sequence": None,
            "status": "conflicting",
            "reason": f"{branch} duplicate checkpoint group at simulation {simulation} is not structurally equivalent",
        }

    return {
        "duplicate_summary": {
            "duplicate_count": len(duplicate_groups),
            "duplicate_simulations": duplicate_simulations,
            "duplicate_groups": duplicate_groups,
        },
        "canonical_checkpoint_sequence": canonical_checkpoint_sequence,
        "status": "safe",
        "reason": f"{branch} duplicate checkpoints are structurally equivalent"
        if duplicate_groups
        else f"{branch} trace has no duplicate checkpoints",
    }


def _classification_and_status(
    default_branch: dict, relaxed_branch: dict
) -> tuple[str, dict, bool]:
    branch_statuses = {default_branch["status"], relaxed_branch["status"]}
    if "conflicting" in branch_statuses:
        return (
            "duplicate_conflicting_checkpoint",
            {
                "safe_for_followup_spec": False,
                "reason": "At least one duplicate checkpoint group is not structurally equivalent within a branch.",
            },
            False,
        )
    if "inconclusive" in branch_statuses:
        return (
            "checkpoint_canonicalization_inconclusive",
            {
                "safe_for_followup_spec": False,
                "reason": "The trace cannot be canonicalized safely because at least one branch is not structurally well-formed for duplicate analysis.",
            },
            False,
        )

    default_sequence = default_branch["canonical_checkpoint_sequence"]
    relaxed_sequence = relaxed_branch["canonical_checkpoint_sequence"]
    if default_sequence != relaxed_sequence:
        return (
            "checkpoint_shape_mismatch",
            {
                "safe_for_followup_spec": False,
                "reason": "Default and threshold-relaxed canonical checkpoint sequences differ after safe duplicate grouping.",
            },
            False,
        )

    duplicate_simulations = (
        default_branch["duplicate_summary"]["duplicate_simulations"]
        + relaxed_branch["duplicate_summary"]["duplicate_simulations"]
    )
    unique_duplicate_simulations = sorted(set(duplicate_simulations))
    if not unique_duplicate_simulations:
        return (
            "checkpoint_canonicalization_inconclusive",
            {
                "safe_for_followup_spec": False,
                "reason": "No duplicate checkpoints were present to diagnose or canonicalize.",
            },
            True,
        )

    earliest_simulation = default_sequence[0]
    if any(
        simulation > earliest_simulation for simulation in unique_duplicate_simulations
    ):
        return (
            "duplicate_equivalent_checkpoint",
            {
                "safe_for_followup_spec": True,
                "reason": "Duplicate checkpoint groups are structurally equivalent, including at least one non-root simulation checkpoint.",
            },
            True,
        )
    return (
        "duplicate_root_snapshot_only",
        {
            "safe_for_followup_spec": True,
            "reason": "Only the earliest simulation checkpoint is duplicated, and each duplicate group is structurally equivalent.",
        },
        True,
    )


def build_payload(
    default_trace_artifact: dict,
    relaxed_trace_artifact: dict,
    *,
    source_selection_score_artifact_path: str,
    source_threshold_relaxed_selection_score_artifact_path: str,
) -> dict:
    (
        default_source_artifact,
        default_thresholds,
        default_trace_origin,
        default_trace_points,
    ) = _validate_trace_artifact(
        default_trace_artifact,
        context="default trace artifact",
    )
    (
        relaxed_source_artifact,
        relaxed_thresholds,
        relaxed_trace_origin,
        relaxed_trace_points,
    ) = _validate_trace_artifact(
        relaxed_trace_artifact,
        context="relaxed trace artifact",
    )

    if default_source_artifact != relaxed_source_artifact:
        raise ValueError(
            "default and relaxed source_artifact provenance chains must match"
        )
    if default_trace_origin != relaxed_trace_origin:
        raise ValueError("default and relaxed trace_origin must match")

    default_branch = _analyze_branch_duplicates(default_trace_points, branch="default")
    relaxed_branch = _analyze_branch_duplicates(relaxed_trace_points, branch="relaxed")
    classification_name, canonicalization_status, canonical_sequences_match = (
        _classification_and_status(
            default_branch,
            relaxed_branch,
        )
    )

    return {
        "schema": SCHEMA,
        "classification": {
            "classification": classification_name,
            "evidence_summary": canonicalization_status["reason"],
        },
        "decision": CLASSIFICATION_DECISIONS[classification_name],
        "input_artifacts": {
            "source_selection_score_artifact_path": source_selection_score_artifact_path,
            "source_threshold_relaxed_selection_score_artifact_path": source_threshold_relaxed_selection_score_artifact_path,
        },
        "source_artifact": default_source_artifact,
        "thresholds_evaluated": {
            "default": default_thresholds,
            "relaxed": relaxed_thresholds,
        },
        "trace_origin": default_trace_origin,
        "duplicate_summary": {
            "default": default_branch["duplicate_summary"],
            "relaxed": relaxed_branch["duplicate_summary"],
        },
        "canonical_checkpoint_sequences": {
            "default": default_branch["canonical_checkpoint_sequence"],
            "relaxed": relaxed_branch["canonical_checkpoint_sequence"],
        },
        "canonical_sequences_match": canonical_sequences_match,
        "canonicalization_status": canonicalization_status,
        "source_snapshots": {
            "default_trace_classification": copy.deepcopy(
                default_trace_artifact.get("classification")
            ),
            "relaxed_trace_classification": copy.deepcopy(
                relaxed_trace_artifact.get("classification")
            ),
            "default_trace_point_count": len(default_trace_points),
            "relaxed_trace_point_count": len(relaxed_trace_points),
            "default_first_simulation": default_trace_points[0]["simulation"],
            "relaxed_first_simulation": relaxed_trace_points[0]["simulation"],
            "default_last_simulation": default_trace_points[-1]["simulation"],
            "relaxed_last_simulation": relaxed_trace_points[-1]["simulation"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    default_trace_artifact = load_json(args.source_selection_score_artifact)
    relaxed_trace_artifact = load_json(
        args.source_threshold_relaxed_selection_score_artifact
    )
    payload = build_payload(
        default_trace_artifact,
        relaxed_trace_artifact,
        source_selection_score_artifact_path=str(args.source_selection_score_artifact),
        source_threshold_relaxed_selection_score_artifact_path=str(
            args.source_threshold_relaxed_selection_score_artifact
        ),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "artifact_path": str(args.out),
                "schema": payload["schema"],
                "classification": payload["classification"]["classification"],
                "decision": payload["decision"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
