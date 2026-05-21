from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

SCHEMA = "azlite_capture_002_selection_score_component_audit_v1"
SOURCE_METRIC_AUDIT_SCHEMA = "azlite_capture_002_metric_co_movement_audit_v1"
SOURCE_SELECTION_SCORE_SCHEMA = "azlite_capture_002_selection_score_trace_v1"
SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA = (
    "azlite_capture_002_trace_checkpoint_canonicalization_v1"
)
ROW_ID = "capture_available-002"
EXPECTED_METRIC_AUDIT_CLASSIFICATION = "early_selection_score_only"
EXPECTED_METRIC_AUDIT_DECISION = "write_002_selection_score_component_audit_spec"
EXPECTED_TRACE_CLASSIFICATION = "unresolved"
EXPECTED_TRACE_DECISION = "write_002_unresolved_trace_review_spec"
EXPECTED_CANONICALIZATION_DECISION = "write_002_metric_audit_canonical_input_spec"
FLOAT_TOLERANCE = 1e-12
CLASSIFICATION_DECISIONS = {
    "prior_pressure_lead": "write_002_prior_pressure_component_spec",
    "child_q_lift_lead": "write_002_child_q_lift_component_spec",
    "mixed_selection_score_signal": "write_002_mixed_selection_score_component_spec",
    "selection_score_component_inconclusive": "stop_002_selection_score_component_inconclusive",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit capture 002 selection-score support signatures"
    )
    parser.add_argument("--source-metric-audit-artifact", type=Path, required=True)
    parser.add_argument("--source-selection-score-artifact", type=Path, required=True)
    parser.add_argument(
        "--source-threshold-relaxed-selection-score-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-checkpoint-canonicalization-artifact", type=Path, required=False
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
    return {
        "meaningful_q_margin": _finite_non_negative_number(
            thresholds.get("meaningful_q_margin"),
            context=f"{context} thresholds.meaningful_q_margin",
        ),
        "material_selection_score_margin": _finite_non_negative_number(
            thresholds.get("material_selection_score_margin"),
            context=f"{context} thresholds.material_selection_score_margin",
        ),
        "material_visit_share_margin": _finite_non_negative_number(
            thresholds.get("material_visit_share_margin"),
            context=f"{context} thresholds.material_visit_share_margin",
        ),
    }


def _source_identity(artifact: dict, *, context: str) -> dict:
    source_artifact = artifact.get("source_artifact")
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
    return {
        "row_id": source_artifact["row_id"],
        "reference_move": source_artifact["reference_move"],
        "full_search_selected_move": source_artifact["full_search_selected_move"],
        "selected_artifact": copy.deepcopy(selected_artifact),
    }


def _validated_source_artifact(source_artifact, *, context: str) -> dict:
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


def _validate_metric_audit_artifact(
    artifact: dict,
    *,
    source_selection_score_artifact_path: str,
    source_threshold_relaxed_selection_score_artifact_path: str,
    source_checkpoint_canonicalization_artifact_path: str | None,
) -> tuple[dict, dict, dict]:
    if artifact.get("schema") != SOURCE_METRIC_AUDIT_SCHEMA:
        raise ValueError(
            f"metric audit artifact has wrong schema: expected {SOURCE_METRIC_AUDIT_SCHEMA}"
        )
    if (
        _classification_name(artifact, context="metric audit artifact")
        != EXPECTED_METRIC_AUDIT_CLASSIFICATION
    ):
        raise ValueError(
            f"metric audit artifact classification must be {EXPECTED_METRIC_AUDIT_CLASSIFICATION}"
        )
    if artifact.get("decision") != EXPECTED_METRIC_AUDIT_DECISION:
        raise ValueError(
            f"metric audit artifact decision must be {EXPECTED_METRIC_AUDIT_DECISION}"
        )
    input_artifacts = artifact.get("input_artifacts")
    if not isinstance(input_artifacts, dict):
        raise ValueError("metric audit input_artifacts must be an object")
    if (
        input_artifacts.get("source_selection_score_artifact_path")
        != source_selection_score_artifact_path
    ):
        raise ValueError(
            "metric audit input_artifacts source_selection_score_artifact_path must match source path"
        )
    if (
        input_artifacts.get("source_threshold_relaxed_selection_score_artifact_path")
        != source_threshold_relaxed_selection_score_artifact_path
    ):
        raise ValueError(
            "metric audit input_artifacts source_threshold_relaxed_selection_score_artifact_path must match source path"
        )
    metric_canonicalization_path = input_artifacts.get(
        "source_checkpoint_canonicalization_artifact_path"
    )
    if metric_canonicalization_path != source_checkpoint_canonicalization_artifact_path:
        raise ValueError(
            "metric audit input_artifacts source_checkpoint_canonicalization_artifact_path must match source path"
        )
    thresholds_evaluated = artifact.get("thresholds_evaluated")
    if not isinstance(thresholds_evaluated, dict):
        raise ValueError("metric audit thresholds_evaluated must be an object")
    default_thresholds = _validate_thresholds(
        thresholds_evaluated.get("default"),
        context="metric audit artifact thresholds_evaluated.default",
    )
    relaxed_thresholds = _validate_thresholds(
        thresholds_evaluated.get("relaxed"),
        context="metric audit artifact thresholds_evaluated.relaxed",
    )
    return (
        _source_identity(artifact, context="metric audit artifact"),
        _validated_source_artifact(
            artifact.get("source_artifact"), context="metric audit artifact"
        ),
        {"default": default_thresholds, "relaxed": relaxed_thresholds},
    )


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
    if not all(isinstance(point, dict) for point in trace_points):
        raise ValueError(f"{context} trace_points must contain objects")
    return (
        _source_identity(artifact, context=context),
        _validate_thresholds(artifact.get("thresholds"), context=context),
        trace_origin,
        trace_points,
    )


def _checkpoint_sequence(trace_points: list[dict], *, context: str) -> list[float]:
    sequence = [
        _finite_number(
            point.get("simulation"),
            context=f"{context} trace_points[{index}].simulation",
        )
        for index, point in enumerate(trace_points)
    ]
    if len(set(sequence)) != len(sequence):
        raise ValueError("checkpoint sequences must not contain duplicates")
    if any(current <= previous for previous, current in zip(sequence, sequence[1:])):
        raise ValueError("checkpoint sequences must be strictly increasing")
    return sequence


def _sequence_values(sequence, *, context: str) -> list[float]:
    if not isinstance(sequence, list) or not sequence:
        raise ValueError(f"{context} must be a non-empty list")
    normalized = [
        _finite_number(value, context=f"{context}[{index}]")
        for index, value in enumerate(sequence)
    ]
    if len(set(normalized)) != len(normalized):
        raise ValueError("checkpoint sequences must not contain duplicates")
    if any(
        current <= previous for previous, current in zip(normalized, normalized[1:])
    ):
        raise ValueError("checkpoint sequences must be strictly increasing")
    return normalized


def _trace_point_projection(trace_point: dict, *, context: str) -> tuple[float, str]:
    simulation = _finite_number(
        trace_point.get("simulation"), context=f"{context}.simulation"
    )
    projection = {
        "simulation": copy.deepcopy(trace_point.get("simulation")),
        "selected_move": copy.deepcopy(trace_point.get("selected_move")),
        "reference_move_by_prior": copy.deepcopy(
            trace_point.get("reference_move_by_prior")
        ),
        "visits": copy.deepcopy(trace_point.get("visits")),
        "moves": copy.deepcopy(trace_point.get("moves")),
    }
    return simulation, json.dumps(projection, sort_keys=True)


def _select_canonical_trace_points(
    trace_points: list[dict],
    *,
    canonical_sequence: list[float],
    context: str,
) -> list[dict]:
    selected_trace_points = []
    selected_projections = {}
    collapsed_sequence = []
    previous_simulation = None
    for index, trace_point in enumerate(trace_points):
        simulation, projection = _trace_point_projection(
            trace_point,
            context=f"{context} trace_points[{index}]",
        )
        if previous_simulation is not None and simulation < previous_simulation:
            raise ValueError("checkpoint sequences must be strictly increasing")
        previous_simulation = simulation
        if simulation in selected_projections:
            if selected_projections[simulation] != projection:
                raise ValueError(
                    "skipped duplicate checkpoint must match kept checkpoint contents"
                )
            continue
        selected_projections[simulation] = projection
        collapsed_sequence.append(simulation)
        selected_trace_points.append(trace_point)
    for simulation in collapsed_sequence:
        if simulation not in canonical_sequence:
            raise ValueError(
                "original trace contains non-duplicate checkpoint not present in canonical checkpoint sequence"
            )
    if collapsed_sequence != canonical_sequence:
        raise ValueError(
            "collapsed original checkpoint sequence must match canonical checkpoint sequence"
        )
    return selected_trace_points


def _validate_canonicalization_artifact(
    artifact: dict,
    *,
    source_artifact: dict,
    default_thresholds: dict,
    relaxed_thresholds: dict,
    trace_origin: str,
    source_selection_score_artifact_path: str,
    source_threshold_relaxed_selection_score_artifact_path: str,
    source_checkpoint_canonicalization_artifact_path: str | None,
) -> list[float]:
    if artifact.get("schema") != SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA:
        raise ValueError(
            f"canonicalization artifact has wrong schema: expected {SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA}"
        )
    if artifact.get("decision") != EXPECTED_CANONICALIZATION_DECISION:
        raise ValueError(
            f"canonicalization artifact decision must be {EXPECTED_CANONICALIZATION_DECISION}"
        )
    input_artifacts = artifact.get("input_artifacts")
    if not isinstance(input_artifacts, dict):
        raise ValueError("canonicalization input_artifacts must be an object")
    if (
        input_artifacts.get("source_selection_score_artifact_path")
        != source_selection_score_artifact_path
    ):
        raise ValueError(
            "canonicalization input_artifacts source_selection_score_artifact_path must match"
        )
    if (
        input_artifacts.get("source_threshold_relaxed_selection_score_artifact_path")
        != source_threshold_relaxed_selection_score_artifact_path
    ):
        raise ValueError(
            "canonicalization input_artifacts source_threshold_relaxed_selection_score_artifact_path must match"
        )
    if source_checkpoint_canonicalization_artifact_path is not None and artifact.get(
        "source_path"
    ) not in (
        None,
        source_checkpoint_canonicalization_artifact_path,
    ):
        raise ValueError("canonicalization source_path must match source path")
    if (
        _validated_source_artifact(
            artifact.get("source_artifact"), context="canonicalization artifact"
        )
        != source_artifact
    ):
        raise ValueError("canonicalization source_artifact must match source artifact")
    canonicalization_status = artifact.get("canonicalization_status")
    if not isinstance(canonicalization_status, dict):
        raise ValueError("canonicalization canonicalization_status must be an object")
    if canonicalization_status.get("safe_for_followup_spec") is not True:
        raise ValueError("canonicalization_status.safe_for_followup_spec must be true")
    if artifact.get("canonical_sequences_match") is not True:
        raise ValueError("canonical_sequences_match must be true")
    canonical_sequences = artifact.get("canonical_checkpoint_sequences")
    if not isinstance(canonical_sequences, dict):
        raise ValueError(
            "canonicalization canonical_checkpoint_sequences must be an object"
        )
    default_sequence = _sequence_values(
        canonical_sequences.get("default"),
        context="canonicalization canonical_checkpoint_sequences.default",
    )
    relaxed_sequence = _sequence_values(
        canonical_sequences.get("relaxed"),
        context="canonicalization canonical_checkpoint_sequences.relaxed",
    )
    if default_sequence != relaxed_sequence:
        raise ValueError("canonicalization canonical checkpoint sequences must match")
    thresholds_evaluated = artifact.get("thresholds_evaluated")
    if not isinstance(thresholds_evaluated, dict):
        raise ValueError("canonicalization thresholds_evaluated must be an object")
    if thresholds_evaluated.get("default") != default_thresholds:
        raise ValueError(
            "canonicalization thresholds_evaluated.default must match default thresholds"
        )
    if thresholds_evaluated.get("relaxed") != relaxed_thresholds:
        raise ValueError(
            "canonicalization thresholds_evaluated.relaxed must match relaxed thresholds"
        )
    if artifact.get("trace_origin") != trace_origin:
        raise ValueError("canonicalization trace_origin must match trace origin")
    return default_sequence


def validate_input_chain(
    metric_audit_artifact: dict,
    default_trace_artifact: dict,
    relaxed_trace_artifact: dict,
    *,
    source_selection_score_artifact_path: str,
    source_threshold_relaxed_selection_score_artifact_path: str,
    checkpoint_canonicalization_artifact: dict | None = None,
    source_checkpoint_canonicalization_artifact_path: str | None = None,
) -> dict:
    if (
        checkpoint_canonicalization_artifact is None
        and source_checkpoint_canonicalization_artifact_path is not None
    ):
        raise ValueError("canonical mode requires checkpoint_canonicalization_artifact")
    if (
        checkpoint_canonicalization_artifact is not None
        and source_checkpoint_canonicalization_artifact_path is None
    ):
        raise ValueError(
            "canonical mode requires source_checkpoint_canonicalization_artifact_path"
        )
    metric_audit_identity, full_metric_audit_source_artifact, metric_thresholds = (
        _validate_metric_audit_artifact(
            metric_audit_artifact,
            source_selection_score_artifact_path=source_selection_score_artifact_path,
            source_threshold_relaxed_selection_score_artifact_path=source_threshold_relaxed_selection_score_artifact_path,
            source_checkpoint_canonicalization_artifact_path=source_checkpoint_canonicalization_artifact_path,
        )
    )
    default_identity, default_thresholds, default_trace_origin, default_trace_points = (
        _validate_trace_artifact(
            default_trace_artifact,
            context="default trace artifact",
        )
    )
    relaxed_identity, relaxed_thresholds, relaxed_trace_origin, relaxed_trace_points = (
        _validate_trace_artifact(
            relaxed_trace_artifact,
            context="relaxed trace artifact",
        )
    )
    if (
        metric_audit_identity != default_identity
        or metric_audit_identity != relaxed_identity
    ):
        raise ValueError("source artifacts must match")
    full_default_source_artifact = _validated_source_artifact(
        default_trace_artifact.get("source_artifact"),
        context="default trace artifact",
    )
    full_relaxed_source_artifact = _validated_source_artifact(
        relaxed_trace_artifact.get("source_artifact"),
        context="relaxed trace artifact",
    )
    if full_metric_audit_source_artifact != full_default_source_artifact:
        raise ValueError("source artifacts must match")
    if full_default_source_artifact != full_relaxed_source_artifact:
        raise ValueError("source artifacts must match")
    if metric_thresholds["default"] != default_thresholds:
        raise ValueError(
            "metric audit thresholds_evaluated.default must match default thresholds"
        )
    if metric_thresholds["relaxed"] != relaxed_thresholds:
        raise ValueError(
            "metric audit thresholds_evaluated.relaxed must match relaxed thresholds"
        )
    if default_trace_origin != relaxed_trace_origin:
        raise ValueError("default and relaxed trace_origin must match")
    if checkpoint_canonicalization_artifact is None:
        default_sequence = _checkpoint_sequence(
            default_trace_points, context="default trace artifact"
        )
        relaxed_sequence = _checkpoint_sequence(
            relaxed_trace_points, context="relaxed trace artifact"
        )
        if default_sequence != relaxed_sequence:
            raise ValueError("checkpoint sequences must match")
    else:
        canonical_sequence = _validate_canonicalization_artifact(
            checkpoint_canonicalization_artifact,
            source_artifact=full_default_source_artifact,
            default_thresholds=default_thresholds,
            relaxed_thresholds=relaxed_thresholds,
            trace_origin=default_trace_origin,
            source_selection_score_artifact_path=source_selection_score_artifact_path,
            source_threshold_relaxed_selection_score_artifact_path=source_threshold_relaxed_selection_score_artifact_path,
            source_checkpoint_canonicalization_artifact_path=source_checkpoint_canonicalization_artifact_path,
        )
        default_trace_points = _select_canonical_trace_points(
            default_trace_points,
            canonical_sequence=canonical_sequence,
            context="default trace artifact",
        )
        relaxed_trace_points = _select_canonical_trace_points(
            relaxed_trace_points,
            canonical_sequence=canonical_sequence,
            context="relaxed trace artifact",
        )
        default_sequence = _checkpoint_sequence(
            default_trace_points, context="default trace artifact"
        )
        relaxed_sequence = _checkpoint_sequence(
            relaxed_trace_points, context="relaxed trace artifact"
        )
        if default_sequence != relaxed_sequence:
            raise ValueError("checkpoint sequences must match")
        if default_sequence != canonical_sequence:
            raise ValueError(
                "collapsed original checkpoint sequence must match canonical checkpoint sequence"
            )
    return {
        "source_artifact_identity": metric_audit_identity,
        "source_artifact": full_metric_audit_source_artifact,
        "default_thresholds": default_thresholds,
        "relaxed_thresholds": relaxed_thresholds,
        "trace_origin": default_trace_origin,
        "default_trace_points": default_trace_points,
        "relaxed_trace_points": relaxed_trace_points,
        "checkpoint_sequence": default_sequence,
        "preserve_original_simulation_representation": checkpoint_canonicalization_artifact
        is not None,
        "source_snapshots": {
            "metric_audit_classification": copy.deepcopy(
                metric_audit_artifact.get("classification")
            ),
            "default_trace_classification": copy.deepcopy(
                default_trace_artifact.get("classification")
            ),
            "relaxed_trace_classification": copy.deepcopy(
                relaxed_trace_artifact.get("classification")
            ),
            "default_trace_origin": default_trace_artifact.get("trace_origin"),
            "relaxed_trace_origin": relaxed_trace_artifact.get("trace_origin"),
        },
    }


def _move_entry(trace_point: dict, move: int) -> dict | None:
    moves = trace_point.get("moves")
    if not isinstance(moves, list):
        return None
    for entry in moves:
        if isinstance(entry, dict) and entry.get("move") == move:
            return entry
    return None


def _metric_margin(
    trace_point: dict, *, selected_move: int, reference_move: int, metric_key: str
) -> float | None:
    selected_entry = _move_entry(trace_point, selected_move)
    reference_entry = _move_entry(trace_point, reference_move)
    if selected_entry is None or reference_entry is None:
        return None
    selected_value = selected_entry.get(metric_key)
    reference_value = reference_entry.get(metric_key)
    if isinstance(selected_value, bool) or isinstance(reference_value, bool):
        return None
    if not isinstance(selected_value, (int, float)) or not isinstance(
        reference_value, (int, float)
    ):
        return None
    if not math.isfinite(selected_value) or not math.isfinite(reference_value):
        return None
    return float(selected_value) - float(reference_value)


def _selection_score_margin(
    trace_point: dict, *, selected_move: int, reference_move: int
) -> float | None:
    return _metric_margin(
        trace_point,
        selected_move=selected_move,
        reference_move=reference_move,
        metric_key="selection_score",
    )


def _q_margin(
    trace_point: dict, *, selected_move: int, reference_move: int
) -> float | None:
    return _metric_margin(
        trace_point,
        selected_move=selected_move,
        reference_move=reference_move,
        metric_key="q_value",
    )


def _prior_pressure_support(
    trace_point: dict, *, selected_move: int, reference_move: int, thresholds: dict
) -> dict | None:
    selection_score_margin = _selection_score_margin(
        trace_point,
        selected_move=selected_move,
        reference_move=reference_move,
    )
    q_margin = _q_margin(
        trace_point, selected_move=selected_move, reference_move=reference_move
    )
    if selection_score_margin is None:
        return None
    if (
        selection_score_margin
        < thresholds["material_selection_score_margin"] - FLOAT_TOLERANCE
    ):
        return None
    if (
        q_margin is not None
        and q_margin >= thresholds["meaningful_q_margin"] - FLOAT_TOLERANCE
    ):
        return None
    return {
        "simulation": trace_point["simulation"],
        "selection_score_margin": selection_score_margin,
        "q_margin": q_margin,
    }


def _child_q_lift_support(
    trace_point: dict, *, selected_move: int, reference_move: int, thresholds: dict
) -> dict | None:
    selection_score_margin = _selection_score_margin(
        trace_point,
        selected_move=selected_move,
        reference_move=reference_move,
    )
    q_margin = _q_margin(
        trace_point, selected_move=selected_move, reference_move=reference_move
    )
    if selection_score_margin is None or q_margin is None:
        return None
    if (
        selection_score_margin
        < thresholds["material_selection_score_margin"] - FLOAT_TOLERANCE
    ):
        return None
    if q_margin < thresholds["meaningful_q_margin"] - FLOAT_TOLERANCE:
        return None
    return {
        "simulation": trace_point["simulation"],
        "selection_score_margin": selection_score_margin,
        "q_margin": q_margin,
    }


def _positive_prior_pressure_support(
    trace_point: dict,
    *,
    selected_move: int,
    reference_move: int,
) -> dict | None:
    selection_score_margin = _selection_score_margin(
        trace_point,
        selected_move=selected_move,
        reference_move=reference_move,
    )
    q_margin = _q_margin(
        trace_point, selected_move=selected_move, reference_move=reference_move
    )
    if selection_score_margin is None or selection_score_margin <= FLOAT_TOLERANCE:
        return None
    if q_margin is not None and q_margin > FLOAT_TOLERANCE:
        return None
    return {
        "simulation": trace_point["simulation"],
        "selection_score_margin": selection_score_margin,
        "q_margin": q_margin,
    }


def _positive_child_q_lift_support(
    trace_point: dict,
    *,
    selected_move: int,
    reference_move: int,
) -> dict | None:
    selection_score_margin = _selection_score_margin(
        trace_point,
        selected_move=selected_move,
        reference_move=reference_move,
    )
    q_margin = _q_margin(
        trace_point, selected_move=selected_move, reference_move=reference_move
    )
    if selection_score_margin is None or selection_score_margin <= FLOAT_TOLERANCE:
        return None
    if q_margin is None or q_margin <= FLOAT_TOLERANCE:
        return None
    return {
        "simulation": trace_point["simulation"],
        "selection_score_margin": selection_score_margin,
        "q_margin": q_margin,
    }


def _support_summary_for_branch(
    trace_points: list[dict], *, source_artifact: dict, thresholds: dict
) -> dict:
    selected_move = source_artifact["full_search_selected_move"]
    reference_move = source_artifact["reference_move"]
    summary = {
        "first_positive_prior_pressure_support": None,
        "first_positive_child_q_lift_support": None,
        "first_material_prior_pressure_support": None,
        "first_material_child_q_lift_support": None,
    }
    for trace_point in trace_points:
        if summary["first_positive_prior_pressure_support"] is None:
            summary["first_positive_prior_pressure_support"] = (
                _positive_prior_pressure_support(
                    trace_point,
                    selected_move=selected_move,
                    reference_move=reference_move,
                )
            )
        if summary["first_positive_child_q_lift_support"] is None:
            summary["first_positive_child_q_lift_support"] = (
                _positive_child_q_lift_support(
                    trace_point,
                    selected_move=selected_move,
                    reference_move=reference_move,
                )
            )
        if summary["first_material_prior_pressure_support"] is None:
            summary["first_material_prior_pressure_support"] = _prior_pressure_support(
                trace_point,
                selected_move=selected_move,
                reference_move=reference_move,
                thresholds=thresholds,
            )
        if summary["first_material_child_q_lift_support"] is None:
            summary["first_material_child_q_lift_support"] = _child_q_lift_support(
                trace_point,
                selected_move=selected_move,
                reference_move=reference_move,
                thresholds=thresholds,
            )
    return summary


def _signature_from_checkpoints(summary: dict) -> str | None:
    prior = summary["prior_pressure"]
    child = summary["child_q_lift"]
    candidates = []
    if prior is not None:
        candidates.append((prior["simulation"], "prior_pressure_lead"))
    if child is not None:
        candidates.append((child["simulation"], "child_q_lift_lead"))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def build_checkpoint_audit(chain: dict) -> list[dict]:
    source_artifact = chain["source_artifact_identity"]
    if chain["preserve_original_simulation_representation"]:
        return [
            {
                "simulation": default_point.get("simulation"),
                "default": {
                    "selection_score_margin": _selection_score_margin(
                        default_point,
                        selected_move=source_artifact["full_search_selected_move"],
                        reference_move=source_artifact["reference_move"],
                    ),
                    "q_margin": _q_margin(
                        default_point,
                        selected_move=source_artifact["full_search_selected_move"],
                        reference_move=source_artifact["reference_move"],
                    ),
                },
                "relaxed": {
                    "selection_score_margin": _selection_score_margin(
                        relaxed_point,
                        selected_move=source_artifact["full_search_selected_move"],
                        reference_move=source_artifact["reference_move"],
                    ),
                    "q_margin": _q_margin(
                        relaxed_point,
                        selected_move=source_artifact["full_search_selected_move"],
                        reference_move=source_artifact["reference_move"],
                    ),
                },
            }
            for default_point, relaxed_point in zip(
                chain["default_trace_points"], chain["relaxed_trace_points"]
            )
        ]
    return [
        {
            "simulation": simulation,
            "default": {
                "selection_score_margin": _selection_score_margin(
                    default_point,
                    selected_move=source_artifact["full_search_selected_move"],
                    reference_move=source_artifact["reference_move"],
                ),
                "q_margin": _q_margin(
                    default_point,
                    selected_move=source_artifact["full_search_selected_move"],
                    reference_move=source_artifact["reference_move"],
                ),
            },
            "relaxed": {
                "selection_score_margin": _selection_score_margin(
                    relaxed_point,
                    selected_move=source_artifact["full_search_selected_move"],
                    reference_move=source_artifact["reference_move"],
                ),
                "q_margin": _q_margin(
                    relaxed_point,
                    selected_move=source_artifact["full_search_selected_move"],
                    reference_move=source_artifact["reference_move"],
                ),
            },
        }
        for simulation, default_point, relaxed_point in zip(
            chain["checkpoint_sequence"],
            chain["default_trace_points"],
            chain["relaxed_trace_points"],
        )
    ]


def first_positive_checkpoints(support_summary: dict) -> dict:
    return {
        branch: {
            "prior_pressure": copy.deepcopy(
                values["first_positive_prior_pressure_support"]
            ),
            "child_q_lift": copy.deepcopy(
                values["first_positive_child_q_lift_support"]
            ),
        }
        for branch, values in support_summary.items()
    }


def first_material_checkpoints(support_summary: dict) -> dict:
    return {
        branch: {
            "prior_pressure": copy.deepcopy(
                values["first_material_prior_pressure_support"]
            ),
            "child_q_lift": copy.deepcopy(
                values["first_material_child_q_lift_support"]
            ),
        }
        for branch, values in support_summary.items()
    }


def classify_component_audit(
    *, first_positive: dict, first_material: dict
) -> tuple[str, str, dict]:
    first_positive_signatures = {
        branch: _signature_from_checkpoints(values)
        for branch, values in first_positive.items()
    }
    first_material_signatures = {
        branch: _signature_from_checkpoints(values)
        for branch, values in first_material.items()
    }
    disagreement_summary = {
        "branch_signature_disagreement": (
            first_material_signatures["default"] != first_material_signatures["relaxed"]
        ),
        "first_positive_first_material_conflict": any(
            first_positive_signatures[branch] is not None
            and first_material_signatures[branch] is not None
            and first_positive_signatures[branch] != first_material_signatures[branch]
            for branch in ("default", "relaxed")
        ),
        "first_positive_signatures": first_positive_signatures,
        "first_material_signatures": first_material_signatures,
    }
    if disagreement_summary["branch_signature_disagreement"]:
        return (
            "mixed_selection_score_signal",
            "Default and relaxed branches disagree on the earliest material selection-score support signature.",
            disagreement_summary,
        )
    if disagreement_summary["first_positive_first_material_conflict"]:
        return (
            "mixed_selection_score_signal",
            "First-positive and first-material ordering imply conflicting selection-score support signatures.",
            disagreement_summary,
        )
    material_signatures = {
        signature
        for signature in first_material_signatures.values()
        if signature is not None
    }
    if material_signatures == {"prior_pressure_lead"}:
        return (
            "prior_pressure_lead",
            "The earliest material selection-score support appears before meaningful child-Q support across both branches.",
            disagreement_summary,
        )
    if material_signatures == {"child_q_lift_lead"}:
        return (
            "child_q_lift_lead",
            "The earliest material selection-score support arrives with meaningful child-Q lift across both branches.",
            disagreement_summary,
        )
    return (
        "selection_score_component_inconclusive",
        "The validated chain does not expose one stable earliest selection-score support signature.",
        disagreement_summary,
    )


def build_payload(
    metric_audit_artifact: dict,
    default_trace_artifact: dict,
    relaxed_trace_artifact: dict,
    *,
    source_metric_audit_artifact_path: str,
    source_selection_score_artifact_path: str,
    source_threshold_relaxed_selection_score_artifact_path: str,
    checkpoint_canonicalization_artifact: dict | None = None,
    source_checkpoint_canonicalization_artifact_path: str | None = None,
) -> dict:
    chain = validate_input_chain(
        metric_audit_artifact,
        default_trace_artifact,
        relaxed_trace_artifact,
        source_selection_score_artifact_path=source_selection_score_artifact_path,
        source_threshold_relaxed_selection_score_artifact_path=source_threshold_relaxed_selection_score_artifact_path,
        checkpoint_canonicalization_artifact=checkpoint_canonicalization_artifact,
        source_checkpoint_canonicalization_artifact_path=source_checkpoint_canonicalization_artifact_path,
    )
    checkpoint_audit = build_checkpoint_audit(chain)
    support_summary = {
        "default": _support_summary_for_branch(
            chain["default_trace_points"],
            source_artifact=chain["source_artifact_identity"],
            thresholds=chain["default_thresholds"],
        ),
        "relaxed": _support_summary_for_branch(
            chain["relaxed_trace_points"],
            source_artifact=chain["source_artifact_identity"],
            thresholds=chain["relaxed_thresholds"],
        ),
    }
    first_positive = first_positive_checkpoints(support_summary)
    first_material = first_material_checkpoints(support_summary)
    classification_name, evidence_summary, disagreement_summary = (
        classify_component_audit(
            first_positive=first_positive,
            first_material=first_material,
        )
    )
    input_artifacts = {
        "source_metric_audit_artifact_path": source_metric_audit_artifact_path,
        "source_selection_score_artifact_path": source_selection_score_artifact_path,
        "source_threshold_relaxed_selection_score_artifact_path": source_threshold_relaxed_selection_score_artifact_path,
    }
    if source_checkpoint_canonicalization_artifact_path is not None:
        input_artifacts["source_checkpoint_canonicalization_artifact_path"] = (
            source_checkpoint_canonicalization_artifact_path
        )
    return {
        "schema": SCHEMA,
        "hypothesis": "selection_score_component_audit",
        "classification": {
            "classification": classification_name,
            "evidence_summary": evidence_summary,
        },
        "decision": CLASSIFICATION_DECISIONS[classification_name],
        "input_artifacts": input_artifacts,
        "source_artifact": chain["source_artifact"],
        "thresholds_evaluated": {
            "selection_score": chain["default_thresholds"][
                "material_selection_score_margin"
            ],
            "meaningful_q": chain["default_thresholds"]["meaningful_q_margin"],
        },
        "checkpoint_audit": checkpoint_audit,
        "first_positive_checkpoints": first_positive,
        "first_material_checkpoints": first_material,
        "selection_score_support_signatures": {
            **copy.deepcopy(support_summary),
            "branch_level_disagreement": disagreement_summary,
        },
        "source_snapshots": chain["source_snapshots"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metric_audit_artifact = load_json(args.source_metric_audit_artifact)
    default_trace_artifact = load_json(args.source_selection_score_artifact)
    relaxed_trace_artifact = load_json(
        args.source_threshold_relaxed_selection_score_artifact
    )
    checkpoint_canonicalization_artifact = None
    if args.source_checkpoint_canonicalization_artifact is not None:
        checkpoint_canonicalization_artifact = load_json(
            args.source_checkpoint_canonicalization_artifact
        )
    payload = build_payload(
        metric_audit_artifact,
        default_trace_artifact,
        relaxed_trace_artifact,
        source_metric_audit_artifact_path=str(args.source_metric_audit_artifact),
        source_selection_score_artifact_path=str(args.source_selection_score_artifact),
        source_threshold_relaxed_selection_score_artifact_path=str(
            args.source_threshold_relaxed_selection_score_artifact
        ),
        checkpoint_canonicalization_artifact=checkpoint_canonicalization_artifact,
        source_checkpoint_canonicalization_artifact_path=(
            str(args.source_checkpoint_canonicalization_artifact)
            if args.source_checkpoint_canonicalization_artifact is not None
            else None
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
