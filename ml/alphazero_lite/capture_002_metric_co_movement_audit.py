from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

SCHEMA = "azlite_capture_002_metric_co_movement_audit_v1"
SOURCE_DECOMPOSITION_SCHEMA = "azlite_capture_002_nonseparable_decomposition_v1"
SOURCE_SELECTION_SCORE_SCHEMA = "azlite_capture_002_selection_score_trace_v1"
SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA = (
    "azlite_capture_002_trace_checkpoint_canonicalization_v1"
)
ROW_ID = "capture_available-002"
EXPECTED_DECOMPOSITION_CLASSIFICATION = "metric_co_movement"
EXPECTED_DECOMPOSITION_DECISION = "stop_002_mechanism_not_isolated"
EXPECTED_TRACE_CLASSIFICATION = "unresolved"
EXPECTED_TRACE_DECISION = "write_002_unresolved_trace_review_spec"
EXPECTED_CANONICALIZATION_DECISION = "write_002_metric_audit_canonical_input_spec"
FLOAT_TOLERANCE = 1e-12
METRIC_THRESHOLDS = {
    "q": "meaningful_q_margin",
    "selection_score": "material_selection_score_margin",
    "visit_share": "material_visit_share_margin",
}
CLASSIFICATION_DECISIONS = {
    "weak_aligned_drift": "write_002_low_confidence_policy_value_interaction_spec",
    "early_selection_score_only": "write_002_selection_score_component_audit_spec",
    "late_visit_share_only": "write_002_visit_accumulation_audit_spec",
    "mixed_low_confidence_signal": "write_002_low_confidence_trace_comparison_spec",
    "metric_audit_inconclusive": "stop_002_metric_audit_inconclusive",
}
MARGIN_KEYS = {
    "q": "selected_minus_reference_q",
    "selection_score": "selected_minus_reference_selection_score",
    "visit_share": "selected_minus_reference_visit_share",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit capture 002 metric co-movement")
    parser.add_argument("--source-decomposition-artifact", type=Path, required=True)
    parser.add_argument("--source-selection-score-artifact", type=Path, required=True)
    parser.add_argument(
        "--source-threshold-relaxed-selection-score-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-checkpoint-canonicalization-artifact",
        type=Path,
        required=False,
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
    for key in METRIC_THRESHOLDS.values():
        if key not in thresholds:
            raise ValueError(f"{context} thresholds must contain {key}")
        normalized[key] = _finite_non_negative_number(
            thresholds[key],
            context=f"{context} thresholds.{key}",
        )
    return normalized


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


def _validate_decomposition_artifact(
    artifact: dict,
    *,
    source_selection_score_artifact_path: str,
    source_threshold_relaxed_selection_score_artifact_path: str,
) -> tuple[dict, dict]:
    if artifact.get("schema") != SOURCE_DECOMPOSITION_SCHEMA:
        raise ValueError(
            f"decomposition artifact has wrong schema: expected {SOURCE_DECOMPOSITION_SCHEMA}"
        )
    if (
        _classification_name(artifact, context="decomposition artifact")
        != EXPECTED_DECOMPOSITION_CLASSIFICATION
    ):
        raise ValueError(
            "decomposition artifact classification must be metric_co_movement"
        )
    if artifact.get("decision") != EXPECTED_DECOMPOSITION_DECISION:
        raise ValueError(
            "decomposition artifact decision must be stop_002_mechanism_not_isolated"
        )
    input_artifacts = artifact.get("input_artifacts")
    if not isinstance(input_artifacts, dict):
        raise ValueError("decomposition input_artifacts must be an object")
    if (
        input_artifacts.get("source_selection_score_artifact_path")
        != source_selection_score_artifact_path
    ):
        raise ValueError(
            "decomposition input_artifacts source_selection_score_artifact_path must match source path"
        )
    if (
        input_artifacts.get("source_threshold_review_artifact_path")
        != source_threshold_relaxed_selection_score_artifact_path
    ):
        raise ValueError(
            "decomposition input_artifacts source_threshold_review_artifact_path must match source path"
        )
    thresholds_evaluated = artifact.get("thresholds_evaluated")
    if not isinstance(thresholds_evaluated, dict):
        raise ValueError("decomposition thresholds_evaluated must be an object")
    return (
        _source_identity(artifact, context="decomposition artifact"),
        _validated_source_artifact(
            artifact.get("source_artifact"), context="decomposition artifact"
        ),
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
    decomposition_artifact: dict,
    default_trace_artifact: dict,
    relaxed_trace_artifact: dict,
    *,
    source_selection_score_artifact_path: str,
    source_threshold_relaxed_selection_score_artifact_path: str,
    checkpoint_canonicalization_artifact: dict | None = None,
    source_checkpoint_canonicalization_artifact_path: str | None = None,
) -> dict:
    decomposition_identity, full_decomposition_source_artifact = (
        _validate_decomposition_artifact(
            decomposition_artifact,
            source_selection_score_artifact_path=source_selection_score_artifact_path,
            source_threshold_relaxed_selection_score_artifact_path=source_threshold_relaxed_selection_score_artifact_path,
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
        decomposition_identity != default_identity
        or decomposition_identity != relaxed_identity
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
    if full_decomposition_source_artifact != full_default_source_artifact:
        raise ValueError("source artifacts must match")
    if full_default_source_artifact != full_relaxed_source_artifact:
        raise ValueError("source artifacts must match")
    thresholds_evaluated = decomposition_artifact["thresholds_evaluated"]
    if thresholds_evaluated.get("default") != default_thresholds:
        raise ValueError(
            "decomposition thresholds_evaluated.default must match default thresholds"
        )
    if thresholds_evaluated.get("relaxed") != relaxed_thresholds:
        raise ValueError(
            "decomposition thresholds_evaluated.relaxed must match relaxed thresholds"
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
            source_threshold_relaxed_selection_score_artifact_path=(
                source_threshold_relaxed_selection_score_artifact_path
            ),
            source_checkpoint_canonicalization_artifact_path=(
                source_checkpoint_canonicalization_artifact_path
            ),
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
        "source_artifact": full_decomposition_source_artifact,
        "source_artifact_identity": decomposition_identity,
        "default_thresholds": default_thresholds,
        "relaxed_thresholds": relaxed_thresholds,
        "trace_origin": default_trace_origin,
        "default_trace_points": default_trace_points,
        "relaxed_trace_points": relaxed_trace_points,
        "checkpoint_sequence": default_sequence,
        "preserve_original_simulation_representation": checkpoint_canonicalization_artifact
        is not None,
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


def _visit_share_margin(
    trace_point: dict, *, selected_move: int, reference_move: int
) -> float | None:
    visits = trace_point.get("visits")
    if not isinstance(visits, list) or len(visits) <= max(
        selected_move, reference_move
    ):
        return None
    selected_visits = visits[selected_move]
    reference_visits = visits[reference_move]
    if isinstance(selected_visits, bool) or isinstance(reference_visits, bool):
        return None
    if not isinstance(selected_visits, (int, float)) or not isinstance(
        reference_visits, (int, float)
    ):
        return None
    if not math.isfinite(selected_visits) or not math.isfinite(reference_visits):
        return None
    total_visits = 0.0
    for visit in visits:
        if (
            isinstance(visit, bool)
            or not isinstance(visit, (int, float))
            or not math.isfinite(visit)
        ):
            return None
        total_visits += float(visit)
    if total_visits <= 0:
        return None
    return (float(selected_visits) / total_visits) - (
        float(reference_visits) / total_visits
    )


def _checkpoint_margins(trace_point: dict, *, source_artifact: dict) -> dict:
    selected_move = source_artifact["full_search_selected_move"]
    reference_move = source_artifact["reference_move"]
    return {
        "selected_minus_reference_q": _metric_margin(
            trace_point,
            selected_move=selected_move,
            reference_move=reference_move,
            metric_key="q_value",
        ),
        "selected_minus_reference_selection_score": _metric_margin(
            trace_point,
            selected_move=selected_move,
            reference_move=reference_move,
            metric_key="selection_score",
        ),
        "selected_minus_reference_visit_share": _visit_share_margin(
            trace_point,
            selected_move=selected_move,
            reference_move=reference_move,
        ),
    }


def build_checkpoint_audit(chain: dict) -> list[dict]:
    if chain["preserve_original_simulation_representation"]:
        return [
            {
                "simulation": default_point.get("simulation"),
                "default": _checkpoint_margins(
                    default_point, source_artifact=chain["source_artifact"]
                ),
                "relaxed": _checkpoint_margins(
                    relaxed_point, source_artifact=chain["source_artifact"]
                ),
            }
            for default_point, relaxed_point in zip(
                chain["default_trace_points"],
                chain["relaxed_trace_points"],
            )
        ]
    return [
        {
            "simulation": simulation,
            "default": _checkpoint_margins(
                default_point, source_artifact=chain["source_artifact"]
            ),
            "relaxed": _checkpoint_margins(
                relaxed_point, source_artifact=chain["source_artifact"]
            ),
        }
        for simulation, default_point, relaxed_point in zip(
            chain["checkpoint_sequence"],
            chain["default_trace_points"],
            chain["relaxed_trace_points"],
        )
    ]


def _first_positive_for_branch(checkpoint_audit: list[dict], *, branch: str) -> dict:
    result = {"q": None, "selection_score": None, "visit_share": None}
    for row in checkpoint_audit:
        for metric, margin_key in MARGIN_KEYS.items():
            if result[metric] is not None:
                continue
            margin = row[branch][margin_key]
            if margin is not None and margin > FLOAT_TOLERANCE:
                result[metric] = {"simulation": row["simulation"], "margin": margin}
    return result


def _first_material_for_branch(
    checkpoint_audit: list[dict], *, branch: str, thresholds: dict
) -> dict:
    result = {"q": None, "selection_score": None, "visit_share": None}
    for row in checkpoint_audit:
        for metric, margin_key in MARGIN_KEYS.items():
            if result[metric] is not None:
                continue
            margin = row[branch][margin_key]
            threshold = thresholds[METRIC_THRESHOLDS[metric]]
            if margin is not None and margin >= threshold - FLOAT_TOLERANCE:
                result[metric] = {"simulation": row["simulation"], "margin": margin}
    return result


def first_positive_checkpoints(checkpoint_audit: list[dict]) -> dict:
    return {
        "default": _first_positive_for_branch(checkpoint_audit, branch="default"),
        "relaxed": _first_positive_for_branch(checkpoint_audit, branch="relaxed"),
    }


def first_material_checkpoints(
    checkpoint_audit: list[dict],
    *,
    default_thresholds: dict,
    relaxed_thresholds: dict,
) -> dict:
    return {
        "default": _first_material_for_branch(
            checkpoint_audit,
            branch="default",
            thresholds=default_thresholds,
        ),
        "relaxed": _first_material_for_branch(
            checkpoint_audit,
            branch="relaxed",
            thresholds=relaxed_thresholds,
        ),
    }


def _final_margin_summary(checkpoint_audit: list[dict]) -> dict:
    final_row = checkpoint_audit[-1]
    return {
        "default_q_margin": final_row["default"]["selected_minus_reference_q"],
        "default_selection_score_margin": final_row["default"][
            "selected_minus_reference_selection_score"
        ],
        "default_visit_share_margin": final_row["default"][
            "selected_minus_reference_visit_share"
        ],
        "relaxed_q_margin": final_row["relaxed"]["selected_minus_reference_q"],
        "relaxed_selection_score_margin": final_row["relaxed"][
            "selected_minus_reference_selection_score"
        ],
        "relaxed_visit_share_margin": final_row["relaxed"][
            "selected_minus_reference_visit_share"
        ],
    }


def _first_material_metrics(first_material: dict) -> list[str]:
    material = {
        metric: value
        for metric, value in first_material["relaxed"].items()
        if value is not None
    }
    if not material:
        return []
    earliest = min(value["simulation"] for value in material.values())
    return [
        metric for metric, value in material.items() if value["simulation"] == earliest
    ]


def _non_material_at_first_lead(metric: str, first_material: dict) -> bool:
    first = first_material["relaxed"][metric]
    if first is None:
        return False
    for other_metric, other_first in first_material["relaxed"].items():
        if other_metric == metric:
            continue
        if other_first is not None and other_first["simulation"] <= first["simulation"]:
            return False
    return True


def _weak_aligned_drift(
    final_margin_summary: dict, *, relaxed_thresholds: dict, first_material: dict
) -> bool:
    relaxed_margins = {
        "q": final_margin_summary["relaxed_q_margin"],
        "selection_score": final_margin_summary["relaxed_selection_score_margin"],
        "visit_share": final_margin_summary["relaxed_visit_share_margin"],
    }
    if any(
        margin is None or margin <= FLOAT_TOLERANCE
        for margin in relaxed_margins.values()
    ):
        return False
    if any(
        relaxed_margins[metric] >= relaxed_thresholds[threshold_key] - FLOAT_TOLERANCE
        for metric, threshold_key in METRIC_THRESHOLDS.items()
    ):
        return False
    return not any(value is not None for value in first_material["relaxed"].values())


def classify_audit(
    *, final_margin_summary: dict, first_material: dict, relaxed_thresholds: dict
) -> tuple[str, str]:
    first_material_metrics = _first_material_metrics(first_material)
    if first_material_metrics == ["selection_score"] and _non_material_at_first_lead(
        "selection_score", first_material
    ):
        return (
            "early_selection_score_only",
            "Selection score becomes materially favorable before Q or visit share under the relaxed trace.",
        )
    if first_material_metrics == ["visit_share"] and _non_material_at_first_lead(
        "visit_share", first_material
    ):
        return (
            "late_visit_share_only",
            "Visit share is the unique first material metric, indicating delayed visit accumulation rather than early Q or selection-score support.",
        )
    if _weak_aligned_drift(
        final_margin_summary,
        relaxed_thresholds=relaxed_thresholds,
        first_material=first_material,
    ):
        return (
            "weak_aligned_drift",
            "All three final relaxed margins drift toward the selected move, but none reaches embedded material support strongly enough to isolate a narrower mechanism.",
        )
    if len(first_material_metrics) >= 2 or any(
        value is not None for value in first_material["relaxed"].values()
    ):
        return (
            "mixed_low_confidence_signal",
            "The trace contains material or split low-confidence signal but no unique first-material lead or clean weak-drift interpretation.",
        )
    return (
        "metric_audit_inconclusive",
        "The chain is valid, but the trace does not supply enough usable signal to justify a narrower follow-up.",
    )


def build_payload(
    decomposition_artifact: dict,
    default_trace_artifact: dict,
    relaxed_trace_artifact: dict,
    *,
    source_decomposition_artifact_path: str,
    source_selection_score_artifact_path: str,
    source_threshold_relaxed_selection_score_artifact_path: str,
    checkpoint_canonicalization_artifact: dict | None = None,
    source_checkpoint_canonicalization_artifact_path: str | None = None,
) -> dict:
    chain = validate_input_chain(
        decomposition_artifact,
        default_trace_artifact,
        relaxed_trace_artifact,
        source_selection_score_artifact_path=source_selection_score_artifact_path,
        source_threshold_relaxed_selection_score_artifact_path=source_threshold_relaxed_selection_score_artifact_path,
        checkpoint_canonicalization_artifact=checkpoint_canonicalization_artifact,
        source_checkpoint_canonicalization_artifact_path=source_checkpoint_canonicalization_artifact_path,
    )
    checkpoint_audit = build_checkpoint_audit(chain)
    first_positive = first_positive_checkpoints(checkpoint_audit)
    first_material = first_material_checkpoints(
        checkpoint_audit,
        default_thresholds=chain["default_thresholds"],
        relaxed_thresholds=chain["relaxed_thresholds"],
    )
    final_margin_summary = _final_margin_summary(checkpoint_audit)
    classification_name, evidence_summary = classify_audit(
        final_margin_summary=final_margin_summary,
        first_material=first_material,
        relaxed_thresholds=chain["relaxed_thresholds"],
    )
    input_artifacts = {
        "source_decomposition_artifact_path": source_decomposition_artifact_path,
        "source_selection_score_artifact_path": source_selection_score_artifact_path,
        "source_threshold_relaxed_selection_score_artifact_path": source_threshold_relaxed_selection_score_artifact_path,
    }
    if source_checkpoint_canonicalization_artifact_path is not None:
        input_artifacts["source_checkpoint_canonicalization_artifact_path"] = (
            source_checkpoint_canonicalization_artifact_path
        )
    return {
        "schema": SCHEMA,
        "hypothesis": "metric_co_movement_audit",
        "classification": {
            "classification": classification_name,
            "evidence_summary": evidence_summary,
        },
        "decision": CLASSIFICATION_DECISIONS[classification_name],
        "input_artifacts": input_artifacts,
        "source_artifact": chain["source_artifact"],
        "thresholds_evaluated": {
            "default": chain["default_thresholds"],
            "relaxed": chain["relaxed_thresholds"],
        },
        "checkpoint_audit": checkpoint_audit,
        "first_positive_checkpoints": first_positive,
        "first_material_checkpoints": first_material,
        "final_margin_summary": final_margin_summary,
        "source_snapshots": {
            "decomposition_classification": copy.deepcopy(
                decomposition_artifact.get("classification")
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    decomposition_artifact = load_json(args.source_decomposition_artifact)
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
        decomposition_artifact,
        default_trace_artifact,
        relaxed_trace_artifact,
        source_decomposition_artifact_path=str(args.source_decomposition_artifact),
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
