from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

SCHEMA = "azlite_capture_002_prior_pressure_component_audit_v1"
SOURCE_SELECTION_SCORE_COMPONENT_AUDIT_SCHEMA = (
    "azlite_capture_002_selection_score_component_audit_v1"
)
SOURCE_METRIC_AUDIT_SCHEMA = "azlite_capture_002_metric_co_movement_audit_v1"
SOURCE_SELECTION_SCORE_SCHEMA = "azlite_capture_002_selection_score_trace_v1"
SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA = (
    "azlite_capture_002_trace_checkpoint_canonicalization_v1"
)
ROW_ID = "capture_available-002"
EXPECTED_COMPONENT_AUDIT_CLASSIFICATION = "prior_pressure_lead"
EXPECTED_COMPONENT_AUDIT_DECISION = "write_002_prior_pressure_component_spec"
EXPECTED_METRIC_AUDIT_CLASSIFICATION = "early_selection_score_only"
EXPECTED_METRIC_AUDIT_DECISION = "write_002_selection_score_component_audit_spec"
EXPECTED_TRACE_CLASSIFICATION = "unresolved"
EXPECTED_TRACE_DECISION = "write_002_unresolved_trace_review_spec"
EXPECTED_CANONICALIZATION_DECISION = "write_002_metric_audit_canonical_input_spec"
FLOAT_TOLERANCE = 1e-12
CLASSIFICATION_DECISIONS = {
    "selection_score_residual_lead": "write_002_selection_score_residual_spec",
    "visit_alignment_pressure": "write_002_visit_alignment_pressure_spec",
    "mixed_prior_pressure_signal": "write_002_prior_pressure_mixed_signal_spec",
    "prior_pressure_component_inconclusive": "stop_002_prior_pressure_component_inconclusive",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit capture 002 prior-pressure support signatures"
    )
    parser.add_argument(
        "--source-selection-score-component-audit-artifact", type=Path, required=True
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validated_source_artifact(artifact: dict, *, context: str) -> dict:
    source_artifact = artifact.get("source_artifact")
    if not isinstance(source_artifact, dict):
        raise ValueError(f"{context} source_artifact must be an object")
    if source_artifact.get("row_id") != ROW_ID:
        raise ValueError(f"{context} source_artifact.row_id must be {ROW_ID}")
    for field in ("reference_move", "full_search_selected_move"):
        value = source_artifact.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{context} source_artifact.{field} must be an integer")
    if not isinstance(source_artifact.get("selected_artifact"), dict):
        raise ValueError(
            f"{context} source_artifact.selected_artifact must be an object"
        )
    return source_artifact


def _validate_matching_source_identity(
    source_artifact: dict, *, expected_source_artifact: dict, context: str
) -> None:
    for field in ("reference_move", "full_search_selected_move"):
        if source_artifact.get(field) != expected_source_artifact.get(field):
            raise ValueError(
                f"{context} source_artifact.{field} must match prior-pressure component audit upstream source_artifact.{field}"
            )
    if source_artifact.get("selected_artifact") != expected_source_artifact.get(
        "selected_artifact"
    ):
        raise ValueError(
            f"{context} source_artifact.selected_artifact must match prior-pressure component audit upstream source_artifact.selected_artifact"
        )


def _validate_metric_audit_inputs(
    artifact: dict,
    *,
    source_selection_score_artifact_path: str,
    source_threshold_relaxed_selection_score_artifact_path: str,
    source_checkpoint_canonicalization_artifact_path: str | None,
) -> None:
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


def _validated_classification_name(artifact: dict, *, context: str) -> str:
    classification = artifact.get("classification")
    if not isinstance(classification, dict):
        raise ValueError(f"{context} classification must be an object")
    return classification.get("classification")


def _finite_number(value, *, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{context} must be finite numeric")
    return float(value)


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


def _validated_trace_origin(trace_artifact: dict, *, context: str) -> str:
    trace_origin = trace_artifact.get("trace_origin")
    if not isinstance(trace_origin, str) or not trace_origin.strip():
        raise ValueError(f"{context} trace_origin must be a non-empty string")
    return trace_origin


def _validated_trace_points(trace_artifact: dict, *, context: str) -> list[dict]:
    trace_points = trace_artifact.get("trace_points")
    if not isinstance(trace_points, list) or not trace_points:
        raise ValueError(f"{context} trace_points must be a non-empty list")
    if not all(isinstance(point, dict) for point in trace_points):
        raise ValueError(f"{context} trace_points must be a list of objects")
    return trace_points


def _validated_payload_metadata_thresholds(
    trace_artifact: dict, *, context: str
) -> dict:
    thresholds = trace_artifact.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError(f"{context} thresholds must be an object")
    return {
        "meaningful_q_margin": _validated_non_negative_threshold(
            thresholds.get("meaningful_q_margin"),
            context=f"{context} thresholds.meaningful_q_margin",
        ),
        "material_selection_score_margin": _validated_non_negative_threshold(
            thresholds.get("material_selection_score_margin"),
            context=f"{context} thresholds.material_selection_score_margin",
        ),
        "material_visit_share_margin": _validated_non_negative_threshold(
            thresholds.get("material_visit_share_margin"),
            context=f"{context} thresholds.material_visit_share_margin",
        ),
    }


def _validated_non_negative_threshold(value, *, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{context} must be a finite non-negative number")
    return float(value)


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
    default_trace_origin: str,
    relaxed_trace_origin: str,
    source_selection_score_artifact_path: str,
    source_threshold_relaxed_selection_score_artifact_path: str,
    source_checkpoint_canonicalization_artifact_path: str,
) -> list[float]:
    if not isinstance(artifact, dict):
        raise ValueError("checkpoint_canonicalization_artifact must be an object")
    if artifact.get("schema") != SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA:
        raise ValueError("canonicalization artifact has wrong schema")
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
    if artifact.get("source_path") not in (
        None,
        source_checkpoint_canonicalization_artifact_path,
    ):
        raise ValueError("canonicalization source_path must match source path")
    if artifact.get("source_artifact") != source_artifact:
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
    if default_trace_origin != relaxed_trace_origin:
        raise ValueError("default and relaxed trace_origin must match")
    if artifact.get("trace_origin") != default_trace_origin:
        raise ValueError("canonicalization trace_origin must match trace origin")
    return default_sequence


def _validate_component_audit_inputs(
    artifact: dict,
    *,
    source_metric_audit_artifact_path: str,
    source_selection_score_artifact_path: str,
    source_threshold_relaxed_selection_score_artifact_path: str,
    source_checkpoint_canonicalization_artifact_path: str | None,
) -> None:
    input_artifacts = artifact.get("input_artifacts")
    if not isinstance(input_artifacts, dict):
        raise ValueError(
            "prior-pressure component audit upstream input_artifacts must be an object"
        )
    if (
        input_artifacts.get("source_metric_audit_artifact_path")
        != source_metric_audit_artifact_path
    ):
        raise ValueError(
            "prior-pressure component audit upstream input_artifacts source_metric_audit_artifact_path must match source path"
        )
    if (
        input_artifacts.get("source_selection_score_artifact_path")
        != source_selection_score_artifact_path
    ):
        raise ValueError(
            "prior-pressure component audit upstream input_artifacts source_selection_score_artifact_path must match source path"
        )
    if (
        input_artifacts.get("source_threshold_relaxed_selection_score_artifact_path")
        != source_threshold_relaxed_selection_score_artifact_path
    ):
        raise ValueError(
            "prior-pressure component audit upstream input_artifacts source_threshold_relaxed_selection_score_artifact_path must match source path"
        )
    component_canonicalization_path = input_artifacts.get(
        "source_checkpoint_canonicalization_artifact_path"
    )
    if source_checkpoint_canonicalization_artifact_path is None:
        if component_canonicalization_path is not None:
            raise ValueError(
                "prior-pressure component audit upstream input_artifacts source_checkpoint_canonicalization_artifact_path must be absent outside canonical mode"
            )
    elif (
        component_canonicalization_path
        != source_checkpoint_canonicalization_artifact_path
    ):
        raise ValueError(
            "prior-pressure component audit upstream input_artifacts source_checkpoint_canonicalization_artifact_path must match source path"
        )


def _validate_required_prior_pressure_checkpoints(artifact: dict) -> None:
    first_material_checkpoints = artifact.get("first_material_checkpoints")
    if not isinstance(first_material_checkpoints, dict):
        raise ValueError(
            "prior-pressure component audit upstream first_material_checkpoints must be an object"
        )
    for branch in ("default", "relaxed"):
        branch_checkpoints = first_material_checkpoints.get(branch)
        if not isinstance(branch_checkpoints, dict):
            raise ValueError(
                f"prior-pressure component audit upstream {branch} first_material_checkpoints must be an object"
            )
        if not isinstance(branch_checkpoints.get("prior_pressure"), dict):
            raise ValueError(
                f"prior-pressure component audit upstream {branch} first_material_checkpoints.prior_pressure must be present"
            )


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


def _matching_trace_point(
    trace_points: list[dict], *, simulation: float
) -> dict | None:
    for trace_point in trace_points:
        if trace_point.get("simulation") == simulation:
            return trace_point
    return None


def _selected_and_reference_visits(
    trace_point: dict, *, selected_move: int, reference_move: int
) -> tuple[float | None, float | None]:
    visits = trace_point.get("visits")
    if not isinstance(visits, list):
        return None, None
    for move in (selected_move, reference_move):
        if move < 0 or move >= len(visits):
            return None, None
    selected_visits = visits[selected_move]
    reference_visits = visits[reference_move]
    if isinstance(selected_visits, bool) or isinstance(reference_visits, bool):
        return None, None
    if not isinstance(selected_visits, (int, float)) or not isinstance(
        reference_visits, (int, float)
    ):
        return None, None
    if not math.isfinite(selected_visits) or not math.isfinite(reference_visits):
        return None, None
    return float(selected_visits), float(reference_visits)


def _visit_shares(
    trace_point: dict, *, selected_move: int, reference_move: int
) -> tuple[float | None, float | None]:
    selected_visits, reference_visits = _selected_and_reference_visits(
        trace_point,
        selected_move=selected_move,
        reference_move=reference_move,
    )
    if selected_visits is None or reference_visits is None:
        return None, None
    visits = trace_point.get("visits")
    total_visits = 0.0
    for value in visits:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return None, None
        total_visits += float(value)
    if total_visits <= 0.0:
        return None, None
    return selected_visits / total_visits, reference_visits / total_visits


def _selection_score_residual_margin(
    selection_score_margin: float | None,
    q_margin: float | None,
) -> float | None:
    if selection_score_margin is None:
        return None
    if q_margin is None:
        return None
    return selection_score_margin - max(q_margin, 0.0)


def _validated_branch_threshold(
    trace_artifact: dict, *, branch: str, key: str
) -> float:
    thresholds = trace_artifact.get("thresholds")
    value = thresholds.get(key) if isinstance(thresholds, dict) else None
    return _validated_non_negative_threshold(
        value, context=f"{branch} branch evidence thresholds.{key}"
    )


def _branch_explanation_candidate(
    *,
    selection_score_residual_margin: float | None,
    selection_score_residual_threshold: float | None,
    selected_visit_share: float | None,
    reference_visit_share: float | None,
    visit_share_threshold: float | None,
) -> str:
    residual_signal_is_material = (
        selection_score_residual_margin is not None
        and selection_score_residual_threshold is not None
        and selection_score_residual_margin >= selection_score_residual_threshold
    )
    visit_signal_is_material = (
        selected_visit_share is not None
        and reference_visit_share is not None
        and visit_share_threshold is not None
        and (selected_visit_share - reference_visit_share) >= visit_share_threshold
    )
    if residual_signal_is_material and visit_signal_is_material:
        return "mixed_prior_pressure_signal"
    if residual_signal_is_material:
        return "selection_score_residual_lead"
    if visit_signal_is_material:
        return "visit_alignment_pressure"
    return "prior_pressure_component_inconclusive"


def _stronger_upstream_residual_threshold(
    component_audit_artifact: dict,
) -> float | None:
    thresholds_evaluated = component_audit_artifact.get("thresholds_evaluated")
    if not isinstance(thresholds_evaluated, dict):
        return None
    selection_score_threshold = thresholds_evaluated.get("selection_score")
    if selection_score_threshold is None:
        return None
    return _validated_non_negative_threshold(
        selection_score_threshold,
        context="prior-pressure component audit upstream thresholds_evaluated.selection_score",
    )


def _residual_threshold(component_audit_artifact: dict) -> float:
    upstream_residual_threshold = _stronger_upstream_residual_threshold(
        component_audit_artifact
    )
    if upstream_residual_threshold is not None:
        return upstream_residual_threshold
    return FLOAT_TOLERANCE


def _prior_pressure_checkpoint(
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
        "simulation": _finite_number(
            trace_point.get("simulation"),
            context="prior-pressure checkpoint simulation",
        ),
        "selection_score_margin": selection_score_margin,
        "q_margin": q_margin,
    }


def _checkpoint_audit(
    default_trace_points: list[dict],
    relaxed_trace_points: list[dict],
    *,
    selected_move: int,
    reference_move: int,
) -> list[dict]:
    audit = []
    for default_trace_point, relaxed_trace_point in zip(
        default_trace_points, relaxed_trace_points, strict=True
    ):
        default_simulation = _finite_number(
            default_trace_point.get("simulation"),
            context="checkpoint audit default simulation",
        )
        relaxed_simulation = _finite_number(
            relaxed_trace_point.get("simulation"),
            context="checkpoint audit relaxed simulation",
        )
        if default_simulation != relaxed_simulation:
            raise ValueError(
                "checkpoint audit default and relaxed trace simulations must align"
            )
        audit.append(
            {
                "simulation": default_simulation,
                "default": {
                    "selection_score_margin": _selection_score_margin(
                        default_trace_point,
                        selected_move=selected_move,
                        reference_move=reference_move,
                    ),
                    "q_margin": _q_margin(
                        default_trace_point,
                        selected_move=selected_move,
                        reference_move=reference_move,
                    ),
                },
                "relaxed": {
                    "selection_score_margin": _selection_score_margin(
                        relaxed_trace_point,
                        selected_move=selected_move,
                        reference_move=reference_move,
                    ),
                    "q_margin": _q_margin(
                        relaxed_trace_point,
                        selected_move=selected_move,
                        reference_move=reference_move,
                    ),
                },
            }
        )
    return audit


def _branch_has_material_residual_vs_visit_conflict(branch_evidence: dict) -> bool:
    selection_score_residual_margin = branch_evidence.get(
        "selection_score_residual_margin"
    )
    selection_score_residual_threshold = branch_evidence.get(
        "selection_score_residual_threshold"
    )
    selected_visit_share = branch_evidence.get("selected_visit_share")
    reference_visit_share = branch_evidence.get("reference_visit_share")
    visit_share_threshold = branch_evidence.get("visit_share_threshold")
    if (
        selection_score_residual_margin is None
        or selection_score_residual_threshold is None
        or selected_visit_share is None
        or reference_visit_share is None
        or visit_share_threshold is None
    ):
        return False
    return (
        selection_score_residual_margin >= selection_score_residual_threshold
        and (selected_visit_share - reference_visit_share) >= visit_share_threshold
    )


def _branch_disagreement_summary(branch_level_evidence: dict) -> dict:
    default_candidate = branch_level_evidence["default"]["explanation_candidate"]
    relaxed_candidate = branch_level_evidence["relaxed"]["explanation_candidate"]
    material_conflict_branches = [
        branch
        for branch, branch_evidence in branch_level_evidence.items()
        if _branch_has_material_residual_vs_visit_conflict(branch_evidence)
    ]
    return {
        "default_explanation_candidate": default_candidate,
        "relaxed_explanation_candidate": relaxed_candidate,
        "branches_disagree": default_candidate != relaxed_candidate,
        "residual_vs_visit_material_conflict": bool(material_conflict_branches),
        "material_conflict_branches": material_conflict_branches,
    }


def _overall_classification(branch_disagreement_summary: dict) -> str:
    if branch_disagreement_summary["branches_disagree"]:
        return "mixed_prior_pressure_signal"
    if branch_disagreement_summary["residual_vs_visit_material_conflict"]:
        return "mixed_prior_pressure_signal"
    return branch_disagreement_summary["default_explanation_candidate"]


def _overall_evidence_summary(
    classification: str, branch_disagreement_summary: dict
) -> str:
    if classification == "mixed_prior_pressure_signal":
        if branch_disagreement_summary["branches_disagree"]:
            return "branch explanation candidates disagree"
        if branch_disagreement_summary["residual_vs_visit_material_conflict"]:
            return "material residual and visit evidence conflict"
    return classification.replace("_", " ")


def _build_branch_level_evidence(
    checkpoint: dict,
    *,
    branch: str,
    trace_artifact: dict,
    component_audit_artifact: dict,
    selected_move: int,
    reference_move: int,
) -> dict:
    trace_point = _matching_trace_point(
        trace_artifact.get("trace_points", []), simulation=checkpoint.get("simulation")
    )
    selection_score_margin = _selection_score_margin(
        trace_point,
        selected_move=selected_move,
        reference_move=reference_move,
    )
    q_margin = _q_margin(
        trace_point,
        selected_move=selected_move,
        reference_move=reference_move,
    )
    selected_visits, reference_visits = _selected_and_reference_visits(
        trace_point,
        selected_move=selected_move,
        reference_move=reference_move,
    )
    selected_visit_share, reference_visit_share = _visit_shares(
        trace_point,
        selected_move=selected_move,
        reference_move=reference_move,
    )
    selection_score_residual_threshold = _residual_threshold(component_audit_artifact)
    visit_share_threshold = _validated_branch_threshold(
        trace_artifact,
        branch=branch,
        key="material_visit_share_margin",
    )
    selection_score_residual_margin = _selection_score_residual_margin(
        selection_score_margin, q_margin
    )
    return {
        "upstream_checkpoint_echo": copy.deepcopy(checkpoint),
        "selection_score_margin": selection_score_margin,
        "q_margin": q_margin,
        "selection_score_residual_margin": selection_score_residual_margin,
        "selected_visits": selected_visits,
        "reference_visits": reference_visits,
        "selected_visit_share": selected_visit_share,
        "reference_visit_share": reference_visit_share,
        "selection_score_residual_threshold": selection_score_residual_threshold,
        "visit_share_threshold": visit_share_threshold,
        "explanation_candidate": _branch_explanation_candidate(
            selection_score_residual_margin=selection_score_residual_margin,
            selection_score_residual_threshold=selection_score_residual_threshold,
            selected_visit_share=selected_visit_share,
            reference_visit_share=reference_visit_share,
            visit_share_threshold=visit_share_threshold,
        ),
    }


def _matches_expected_margin(expected, actual) -> bool:
    if expected is None or actual is None:
        return expected is actual
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        return False
    return math.isfinite(expected) and abs(float(expected) - actual) <= FLOAT_TOLERANCE


def _validate_prior_pressure_checkpoint_against_trace(
    checkpoint: dict,
    *,
    branch: str,
    trace_artifact: dict,
    selected_move: int,
    reference_move: int,
) -> None:
    simulation = checkpoint.get("simulation")
    trace_point = _matching_trace_point(
        trace_artifact.get("trace_points", []), simulation=simulation
    )
    if trace_point is None:
        raise ValueError(
            f"prior-pressure component audit upstream {branch} prior_pressure checkpoint simulation {simulation} is missing from validated trace"
        )
    selection_score_margin = _selection_score_margin(
        trace_point,
        selected_move=selected_move,
        reference_move=reference_move,
    )
    if not _matches_expected_margin(
        checkpoint.get("selection_score_margin"), selection_score_margin
    ):
        raise ValueError(
            f"prior-pressure component audit upstream {branch} prior_pressure checkpoint selection_score_margin must match validated trace"
        )
    q_margin = _q_margin(
        trace_point,
        selected_move=selected_move,
        reference_move=reference_move,
    )
    if not _matches_expected_margin(checkpoint.get("q_margin"), q_margin):
        raise ValueError(
            f"prior-pressure component audit upstream {branch} prior_pressure checkpoint q_margin must match validated trace"
        )
    thresholds = {
        "material_selection_score_margin": _validated_branch_threshold(
            trace_artifact,
            branch=branch,
            key="material_selection_score_margin",
        ),
        "meaningful_q_margin": _validated_branch_threshold(
            trace_artifact,
            branch=branch,
            key="meaningful_q_margin",
        ),
    }
    earliest_material_checkpoint = None
    for candidate_trace_point in trace_artifact.get("trace_points", []):
        earliest_material_checkpoint = _prior_pressure_checkpoint(
            candidate_trace_point,
            selected_move=selected_move,
            reference_move=reference_move,
            thresholds=thresholds,
        )
        if earliest_material_checkpoint is not None:
            break
    if earliest_material_checkpoint is None or not (
        _matches_expected_margin(
            checkpoint.get("simulation"), earliest_material_checkpoint["simulation"]
        )
        and _matches_expected_margin(
            checkpoint.get("selection_score_margin"),
            earliest_material_checkpoint["selection_score_margin"],
        )
        and _matches_expected_margin(
            checkpoint.get("q_margin"), earliest_material_checkpoint["q_margin"]
        )
    ):
        raise ValueError(
            f"prior-pressure component audit upstream {branch} prior_pressure checkpoint must match earliest material prior-pressure trace checkpoint"
        )
    return None


def _validate_prior_pressure_trace_cross_checks(
    component_audit_artifact: dict,
    *,
    default_trace_artifact: dict,
    relaxed_trace_artifact: dict,
) -> None:
    first_material_checkpoints = component_audit_artifact["first_material_checkpoints"]
    source_artifact = component_audit_artifact["source_artifact"]
    selected_move = source_artifact["full_search_selected_move"]
    reference_move = source_artifact["reference_move"]
    _validate_prior_pressure_checkpoint_against_trace(
        first_material_checkpoints["default"]["prior_pressure"],
        branch="default",
        trace_artifact=default_trace_artifact,
        selected_move=selected_move,
        reference_move=reference_move,
    )
    _validate_prior_pressure_checkpoint_against_trace(
        first_material_checkpoints["relaxed"]["prior_pressure"],
        branch="relaxed",
        trace_artifact=relaxed_trace_artifact,
        selected_move=selected_move,
        reference_move=reference_move,
    )


def build_payload(
    selection_score_component_audit_artifact: dict,
    metric_audit_artifact: dict,
    default_trace_artifact: dict,
    relaxed_trace_artifact: dict,
    *,
    source_selection_score_component_audit_artifact_path: str,
    source_metric_audit_artifact_path: str,
    source_selection_score_artifact_path: str,
    source_threshold_relaxed_selection_score_artifact_path: str,
    checkpoint_canonicalization_artifact: dict | None = None,
    source_checkpoint_canonicalization_artifact_path: str | None = None,
) -> dict:
    if not isinstance(selection_score_component_audit_artifact, dict):
        raise ValueError(
            "prior-pressure component audit upstream artifact must be an object"
        )
    if (
        selection_score_component_audit_artifact.get("schema")
        != SOURCE_SELECTION_SCORE_COMPONENT_AUDIT_SCHEMA
    ):
        raise ValueError(
            "prior-pressure component audit upstream artifact has wrong schema"
        )

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
    if checkpoint_canonicalization_artifact is not None:
        if not isinstance(checkpoint_canonicalization_artifact, dict):
            raise ValueError("checkpoint_canonicalization_artifact must be an object")
        if (
            checkpoint_canonicalization_artifact.get("schema")
            != SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA
        ):
            raise ValueError("canonicalization artifact has wrong schema")
    classification = selection_score_component_audit_artifact.get("classification")
    if not isinstance(classification, dict):
        raise ValueError(
            "prior-pressure component audit upstream classification must be an object"
        )
    if classification.get("classification") != EXPECTED_COMPONENT_AUDIT_CLASSIFICATION:
        raise ValueError(
            f"prior-pressure component audit upstream classification must be {EXPECTED_COMPONENT_AUDIT_CLASSIFICATION}"
        )
    if (
        selection_score_component_audit_artifact.get("decision")
        != EXPECTED_COMPONENT_AUDIT_DECISION
    ):
        raise ValueError(
            f"prior-pressure component audit upstream decision must be {EXPECTED_COMPONENT_AUDIT_DECISION}"
        )
    _validate_component_audit_inputs(
        selection_score_component_audit_artifact,
        source_metric_audit_artifact_path=source_metric_audit_artifact_path,
        source_selection_score_artifact_path=source_selection_score_artifact_path,
        source_threshold_relaxed_selection_score_artifact_path=(
            source_threshold_relaxed_selection_score_artifact_path
        ),
        source_checkpoint_canonicalization_artifact_path=source_checkpoint_canonicalization_artifact_path,
    )
    _validate_required_prior_pressure_checkpoints(
        selection_score_component_audit_artifact
    )
    component_source_artifact = _validated_source_artifact(
        selection_score_component_audit_artifact,
        context="prior-pressure component audit upstream",
    )

    if not isinstance(metric_audit_artifact, dict):
        raise ValueError("metric audit artifact must be an object")
    if metric_audit_artifact.get("schema") != SOURCE_METRIC_AUDIT_SCHEMA:
        raise ValueError("metric audit artifact has wrong schema")
    if (
        _validated_classification_name(
            metric_audit_artifact, context="metric audit artifact"
        )
        != EXPECTED_METRIC_AUDIT_CLASSIFICATION
    ):
        raise ValueError(
            f"metric audit artifact classification must be {EXPECTED_METRIC_AUDIT_CLASSIFICATION}"
        )
    if metric_audit_artifact.get("decision") != EXPECTED_METRIC_AUDIT_DECISION:
        raise ValueError(
            f"metric audit artifact decision must be {EXPECTED_METRIC_AUDIT_DECISION}"
        )
    _validate_metric_audit_inputs(
        metric_audit_artifact,
        source_selection_score_artifact_path=source_selection_score_artifact_path,
        source_threshold_relaxed_selection_score_artifact_path=source_threshold_relaxed_selection_score_artifact_path,
        source_checkpoint_canonicalization_artifact_path=source_checkpoint_canonicalization_artifact_path,
    )
    metric_source_artifact = _validated_source_artifact(
        metric_audit_artifact, context="metric audit artifact"
    )
    _validate_matching_source_identity(
        metric_source_artifact,
        expected_source_artifact=component_source_artifact,
        context="metric audit artifact",
    )

    if not isinstance(default_trace_artifact, dict):
        raise ValueError("default trace artifact must be an object")
    if default_trace_artifact.get("schema") != SOURCE_SELECTION_SCORE_SCHEMA:
        raise ValueError("default trace artifact has wrong schema")
    if (
        _validated_classification_name(
            default_trace_artifact, context="default trace artifact"
        )
        != EXPECTED_TRACE_CLASSIFICATION
    ):
        raise ValueError(
            f"default trace artifact classification must be {EXPECTED_TRACE_CLASSIFICATION}"
        )
    if default_trace_artifact.get("decision") != EXPECTED_TRACE_DECISION:
        raise ValueError(
            f"default trace artifact decision must be {EXPECTED_TRACE_DECISION}"
        )
    default_trace_source_artifact = _validated_source_artifact(
        default_trace_artifact, context="default trace artifact"
    )
    _validate_matching_source_identity(
        default_trace_source_artifact,
        expected_source_artifact=component_source_artifact,
        context="default trace artifact",
    )

    if not isinstance(relaxed_trace_artifact, dict):
        raise ValueError("relaxed trace artifact must be an object")
    if relaxed_trace_artifact.get("schema") != SOURCE_SELECTION_SCORE_SCHEMA:
        raise ValueError("relaxed trace artifact has wrong schema")
    if (
        _validated_classification_name(
            relaxed_trace_artifact, context="relaxed trace artifact"
        )
        != EXPECTED_TRACE_CLASSIFICATION
    ):
        raise ValueError(
            f"relaxed trace artifact classification must be {EXPECTED_TRACE_CLASSIFICATION}"
        )
    if relaxed_trace_artifact.get("decision") != EXPECTED_TRACE_DECISION:
        raise ValueError(
            f"relaxed trace artifact decision must be {EXPECTED_TRACE_DECISION}"
        )
    relaxed_trace_source_artifact = _validated_source_artifact(
        relaxed_trace_artifact, context="relaxed trace artifact"
    )
    _validate_matching_source_identity(
        relaxed_trace_source_artifact,
        expected_source_artifact=component_source_artifact,
        context="relaxed trace artifact",
    )

    default_trace_points = _validated_trace_points(
        default_trace_artifact, context="default trace artifact"
    )
    relaxed_trace_points = _validated_trace_points(
        relaxed_trace_artifact, context="relaxed trace artifact"
    )
    default_thresholds = default_trace_artifact.get("thresholds")
    relaxed_thresholds = relaxed_trace_artifact.get("thresholds")
    default_trace_origin = _validated_trace_origin(
        default_trace_artifact, context="default trace artifact"
    )
    relaxed_trace_origin = _validated_trace_origin(
        relaxed_trace_artifact, context="relaxed trace artifact"
    )

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
            source_artifact=default_trace_source_artifact,
            default_thresholds=default_thresholds,
            relaxed_thresholds=relaxed_thresholds,
            default_trace_origin=default_trace_origin,
            relaxed_trace_origin=relaxed_trace_origin,
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

    effective_default_trace_artifact = copy.deepcopy(default_trace_artifact)
    effective_default_trace_artifact["trace_points"] = default_trace_points
    effective_relaxed_trace_artifact = copy.deepcopy(relaxed_trace_artifact)
    effective_relaxed_trace_artifact["trace_points"] = relaxed_trace_points

    _validate_prior_pressure_trace_cross_checks(
        selection_score_component_audit_artifact,
        default_trace_artifact=effective_default_trace_artifact,
        relaxed_trace_artifact=effective_relaxed_trace_artifact,
    )

    input_artifacts = {
        "source_selection_score_component_audit_artifact_path": source_selection_score_component_audit_artifact_path,
        "source_metric_audit_artifact_path": source_metric_audit_artifact_path,
        "source_selection_score_artifact_path": source_selection_score_artifact_path,
        "source_threshold_relaxed_selection_score_artifact_path": source_threshold_relaxed_selection_score_artifact_path,
    }
    if source_checkpoint_canonicalization_artifact_path is not None:
        input_artifacts["source_checkpoint_canonicalization_artifact_path"] = (
            source_checkpoint_canonicalization_artifact_path
        )
    first_material_checkpoints = selection_score_component_audit_artifact[
        "first_material_checkpoints"
    ]
    selected_move = component_source_artifact["full_search_selected_move"]
    reference_move = component_source_artifact["reference_move"]
    branch_level_evidence = {
        "default": _build_branch_level_evidence(
            first_material_checkpoints["default"]["prior_pressure"],
            branch="default",
            trace_artifact=effective_default_trace_artifact,
            component_audit_artifact=selection_score_component_audit_artifact,
            selected_move=selected_move,
            reference_move=reference_move,
        ),
        "relaxed": _build_branch_level_evidence(
            first_material_checkpoints["relaxed"]["prior_pressure"],
            branch="relaxed",
            trace_artifact=effective_relaxed_trace_artifact,
            component_audit_artifact=selection_score_component_audit_artifact,
            selected_move=selected_move,
            reference_move=reference_move,
        ),
    }
    branch_disagreement_summary = _branch_disagreement_summary(branch_level_evidence)
    overall_classification = _overall_classification(branch_disagreement_summary)
    thresholds_evaluated = {
        "default": _validated_payload_metadata_thresholds(
            default_trace_artifact, context="default trace artifact"
        ),
        "relaxed": _validated_payload_metadata_thresholds(
            relaxed_trace_artifact, context="relaxed trace artifact"
        ),
    }
    checkpoint_audit = _checkpoint_audit(
        default_trace_points,
        relaxed_trace_points,
        selected_move=selected_move,
        reference_move=reference_move,
    )
    return {
        "schema": SCHEMA,
        "hypothesis": "prior_pressure_component_audit",
        "classification": {
            "classification": overall_classification,
            "evidence_summary": _overall_evidence_summary(
                overall_classification, branch_disagreement_summary
            ),
        },
        "decision": CLASSIFICATION_DECISIONS[overall_classification],
        "input_artifacts": input_artifacts,
        "source_artifact": copy.deepcopy(component_source_artifact),
        "thresholds_evaluated": thresholds_evaluated,
        "checkpoint_audit": checkpoint_audit,
        "branch_level_evidence": branch_level_evidence,
        "branch_disagreement_summary": branch_disagreement_summary,
        "source_snapshots": {
            "component_audit_classification": copy.deepcopy(
                selection_score_component_audit_artifact.get("classification")
            ),
            "metric_audit_classification": copy.deepcopy(
                metric_audit_artifact.get("classification")
            ),
            "default_trace_classification": copy.deepcopy(
                default_trace_artifact.get("classification")
            ),
            "relaxed_trace_classification": copy.deepcopy(
                relaxed_trace_artifact.get("classification")
            ),
            "default_trace_origin": default_trace_origin,
            "relaxed_trace_origin": relaxed_trace_origin,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selection_score_component_audit_artifact = load_json(
        args.source_selection_score_component_audit_artifact
    )
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
        selection_score_component_audit_artifact,
        metric_audit_artifact,
        default_trace_artifact,
        relaxed_trace_artifact,
        source_selection_score_component_audit_artifact_path=str(
            args.source_selection_score_component_audit_artifact
        ),
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
