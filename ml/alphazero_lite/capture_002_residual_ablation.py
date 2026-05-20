from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

SCHEMA = "azlite_capture_002_residual_ablation_v1"
SOURCE_SELECTION_SCORE_RESIDUAL_AUDIT_SCHEMA = "azlite_capture_002_selection_score_residual_audit_v1"
SOURCE_PRIOR_PRESSURE_AUDIT_SCHEMA = "azlite_capture_002_prior_pressure_component_audit_v1"
SOURCE_SELECTION_SCORE_SCHEMA = "azlite_capture_002_selection_score_trace_v1"
SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA = "azlite_capture_002_trace_checkpoint_canonicalization_v1"
ROW_ID = "capture_available-002"
EXPECTED_RESIDUAL_AUDIT_CLASSIFICATION = "stable_selected_residual_advantage"
EXPECTED_RESIDUAL_AUDIT_DECISION = "write_002_residual_ablation_spec"
EXPECTED_PRIOR_PRESSURE_CLASSIFICATION = "selection_score_residual_lead"
EXPECTED_PRIOR_PRESSURE_DECISION = "write_002_selection_score_residual_spec"
EXPECTED_TRACE_CLASSIFICATION = "unresolved"
EXPECTED_TRACE_DECISION = "write_002_unresolved_trace_review_spec"
EXPECTED_CANONICALIZATION_DECISION = "write_002_metric_audit_canonical_input_spec"
EXPECTED_CANONICAL_SIMULATION = 2.0
MODES = (
    "baseline_replay",
    "selected_residual_neutralized",
    "all_residuals_flattened",
)
CLASSIFICATION_DECISIONS = {
    "selected_move_residual_sensitive": "write_002_residual_sensitive_intervention_spec",
    "selected_move_residual_insensitive": "write_002_non_residual_mechanism_review_spec",
    "selected_move_residual_ablation_inconclusive": "stop_002_residual_ablation_inconclusive",
}
FLOAT_TOLERANCE = 1e-12
EXPECTED_SELECTED_MOVE = 0
EXPECTED_REFERENCE_MOVE = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate residual effects for capture 002")
    parser.add_argument("--source-selection-score-residual-audit-artifact", type=Path, required=True)
    parser.add_argument("--source-prior-pressure-audit-artifact", type=Path, required=True)
    parser.add_argument("--source-selection-score-artifact", type=Path, required=True)
    parser.add_argument(
        "--source-threshold-relaxed-selection-score-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument("--source-checkpoint-canonicalization-artifact", type=Path, required=False)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_number(value, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{context} must be finite numeric")
    return float(value)


def _validated_source_artifact(source_artifact: dict, *, context: str) -> dict:
    if not isinstance(source_artifact, dict):
        raise ValueError(f"{context} must be an object")
    if source_artifact.get("row_id") != ROW_ID:
        raise ValueError(f"{context}.row_id must be {ROW_ID}")
    for field in ("reference_move", "full_search_selected_move"):
        value = source_artifact.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{context}.{field} must be an integer")
    if source_artifact.get("full_search_selected_move") != EXPECTED_SELECTED_MOVE:
        raise ValueError(f"{context}.full_search_selected_move must be {EXPECTED_SELECTED_MOVE}")
    if source_artifact.get("reference_move") != EXPECTED_REFERENCE_MOVE:
        raise ValueError(f"{context}.reference_move must be {EXPECTED_REFERENCE_MOVE}")
    selected_artifact = source_artifact.get("selected_artifact")
    if not isinstance(selected_artifact, dict):
        raise ValueError(f"{context}.selected_artifact must be an object")
    return source_artifact


def _validated_trace_points(trace_artifact: dict, *, branch: str) -> list[dict]:
    trace_points = trace_artifact.get("trace_points")
    if not isinstance(trace_points, list) or not trace_points:
        raise ValueError(f"{branch} trace_points must be a non-empty list")
    if not all(isinstance(trace_point, dict) for trace_point in trace_points):
        raise ValueError(f"{branch} trace_points must be a list of objects")
    return trace_points


def _normalized_projection_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _finite_number(value, context="duplicate-equivalence numeric value")
    if isinstance(value, list):
        return [_normalized_projection_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalized_projection_value(item) for key, item in value.items()}
    return value


def _trace_point_projection(trace_point: dict, *, context: str) -> tuple[float, str]:
    simulation = _finite_number(trace_point.get("simulation"), context=f"{context}.simulation")
    projection = {
        "simulation": copy.deepcopy(trace_point.get("simulation")),
        "selected_move": copy.deepcopy(trace_point.get("selected_move")),
        "reference_move_by_prior": copy.deepcopy(trace_point.get("reference_move_by_prior")),
        "visits": copy.deepcopy(trace_point.get("visits")),
        "moves": copy.deepcopy(trace_point.get("moves")),
    }
    return simulation, json.dumps(_normalized_projection_value(projection), sort_keys=True)


def _validated_canonical_trace_point_identity(trace_point: dict, *, branch: str) -> dict:
    selected_move = trace_point.get("selected_move")
    if isinstance(selected_move, bool) or not isinstance(selected_move, int):
        raise ValueError(f"{branch} canonical trace point.selected_move must be an integer")
    reference_move = trace_point.get("reference_move_by_prior")
    if isinstance(reference_move, bool) or not isinstance(reference_move, int):
        raise ValueError(f"{branch} canonical trace point.reference_move_by_prior must be an integer")
    if reference_move != EXPECTED_REFERENCE_MOVE:
        raise ValueError(
            f"{branch} canonical trace point.reference_move_by_prior must be {EXPECTED_REFERENCE_MOVE}"
        )
    return trace_point


def _move_lookup(trace_point: dict, *, move: int, context: str) -> dict:
    moves = trace_point.get("moves")
    if not isinstance(moves, list):
        raise ValueError(f"{context}.moves must be a list")
    for entry in moves:
        entry_move = entry.get("move") if isinstance(entry, dict) else None
        if (
            isinstance(entry, dict)
            and not isinstance(entry_move, bool)
            and isinstance(entry_move, int)
            and entry_move == move
        ):
            return entry
    raise ValueError(f"{context}.moves must include move {move}")


def _canonical_checkpoint_echo_from_trace_point(trace_point: dict, *, branch: str, source_artifact: dict) -> dict:
    selected_move = source_artifact["full_search_selected_move"]
    reference_move = source_artifact["reference_move"]
    selected_entry = _move_lookup(trace_point, move=selected_move, context=f"{branch} canonical trace point")
    reference_entry = _move_lookup(trace_point, move=reference_move, context=f"{branch} canonical trace point")
    selected_score = _finite_number(
        selected_entry.get("selection_score"),
        context=f"{branch} canonical trace point selected selection_score",
    )
    reference_score = _finite_number(
        reference_entry.get("selection_score"),
        context=f"{branch} canonical trace point reference selection_score",
    )
    selected_q = _finite_number(
        selected_entry.get("q_value"),
        context=f"{branch} canonical trace point selected q_value",
    )
    reference_q = _finite_number(
        reference_entry.get("q_value"),
        context=f"{branch} canonical trace point reference q_value",
    )
    return {
        "simulation": _finite_number(
            trace_point.get("simulation"),
            context=f"{branch} canonical trace point simulation",
        ),
        "selection_score_margin": selected_score - reference_score,
        "q_margin": selected_q - reference_q,
    }


def _checkpoint_payload(*, default_trace_point: dict, relaxed_trace_point: dict, source_artifact: dict) -> dict:
    return {
        "canonical_simulation": EXPECTED_CANONICAL_SIMULATION,
        "default_upstream_checkpoint_echo": _canonical_checkpoint_echo_from_trace_point(
            default_trace_point,
            branch="default",
            source_artifact=source_artifact,
        ),
        "relaxed_upstream_checkpoint_echo": _canonical_checkpoint_echo_from_trace_point(
            relaxed_trace_point,
            branch="relaxed",
            source_artifact=source_artifact,
        ),
    }


def _canonical_trace_point_at_simulation(
    trace_artifact: dict,
    *,
    branch: str,
    canonicalization_mode: bool,
) -> dict:
    trace_points = _validated_trace_points(trace_artifact, branch=branch)
    canonical_matches = []
    canonical_projections = set()
    raw_duplicate_projections = {}
    previous_simulation = None
    for index, trace_point in enumerate(trace_points):
        simulation, projection = _trace_point_projection(
            trace_point,
            context=f"{branch} trace trace_points[{index}]",
        )
        if previous_simulation is not None and simulation < previous_simulation:
            raise ValueError("checkpoint sequences must be strictly increasing")
        previous_projection = raw_duplicate_projections.get(simulation)
        if previous_projection is not None:
            if simulation != previous_simulation:
                raise ValueError("checkpoint sequences must be strictly increasing")
            if abs(simulation - EXPECTED_CANONICAL_SIMULATION) <= FLOAT_TOLERANCE:
                if not canonicalization_mode:
                    raise ValueError(
                        f"{branch} raw duplicate {EXPECTED_CANONICAL_SIMULATION} checkpoint requires canonicalization artifact"
                    )
                if previous_projection != projection:
                    raise ValueError(
                        f"{branch} raw duplicate {EXPECTED_CANONICAL_SIMULATION} checkpoint must match canonical projection"
                    )
            elif previous_projection != projection:
                raise ValueError("checkpoint sequences must be strictly increasing")
        else:
            raw_duplicate_projections[simulation] = projection
        previous_simulation = simulation
        if abs(simulation - EXPECTED_CANONICAL_SIMULATION) > FLOAT_TOLERANCE:
            continue
        canonical_matches.append(trace_point)
        canonical_projections.add(projection)
    if not canonical_matches:
        raise ValueError(f"{branch} trace must include canonical simulation {EXPECTED_CANONICAL_SIMULATION}")
    if len(canonical_projections) > 1 and not canonicalization_mode:
        raise ValueError(
            f"{branch} raw duplicate {EXPECTED_CANONICAL_SIMULATION} checkpoint requires canonicalization artifact"
        )
    canonical_trace_point = canonical_matches[0]
    if canonicalization_mode:
        _, canonical_projection = _trace_point_projection(
            canonical_trace_point,
            context=f"{branch} canonical trace point",
        )
        for index, trace_point in enumerate(canonical_matches):
            _, raw_projection = _trace_point_projection(
                trace_point,
                context=f"{branch} raw trace_points[{index}]",
            )
            if raw_projection != canonical_projection:
                raise ValueError(
                    f"{branch} raw duplicate {EXPECTED_CANONICAL_SIMULATION} checkpoint must match canonical projection"
                )
    return _validated_canonical_trace_point_identity(canonical_trace_point, branch=branch)


def _validated_residual_audit_artifact(residual_audit_artifact: dict) -> dict:
    if not isinstance(residual_audit_artifact, dict):
        raise ValueError("selection-score residual audit artifact must be an object")
    if residual_audit_artifact.get("schema") != SOURCE_SELECTION_SCORE_RESIDUAL_AUDIT_SCHEMA:
        raise ValueError("selection-score residual audit artifact has wrong schema")
    classification = residual_audit_artifact.get("classification")
    if not isinstance(classification, dict):
        raise ValueError("selection-score residual audit artifact classification must be an object")
    if classification.get("classification") != EXPECTED_RESIDUAL_AUDIT_CLASSIFICATION:
        raise ValueError(
            "selection-score residual audit artifact classification must be "
            f"{EXPECTED_RESIDUAL_AUDIT_CLASSIFICATION}"
        )
    if residual_audit_artifact.get("decision") != EXPECTED_RESIDUAL_AUDIT_DECISION:
        raise ValueError(
            f"selection-score residual audit artifact decision must be {EXPECTED_RESIDUAL_AUDIT_DECISION}"
        )

    checkpoint = residual_audit_artifact.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("selection-score residual audit artifact checkpoint must be an object")
    canonical_simulation = _finite_number(
        checkpoint.get("canonical_simulation"),
        context="selection-score residual audit artifact checkpoint.canonical_simulation",
    )
    if abs(canonical_simulation - EXPECTED_CANONICAL_SIMULATION) > FLOAT_TOLERANCE:
        raise ValueError(
            "selection-score residual audit artifact checkpoint.canonical_simulation must be "
            f"{EXPECTED_CANONICAL_SIMULATION}"
        )

    _validated_source_artifact(
        residual_audit_artifact.get("source_artifact"),
        context="selection-score residual audit artifact source_artifact",
    )
    return residual_audit_artifact


def _validated_residual_audit_input_artifacts(
    residual_audit_artifact: dict,
    *,
    source_prior_pressure_audit_artifact_path: str,
    source_selection_score_artifact_path: str,
    source_threshold_relaxed_selection_score_artifact_path: str,
) -> None:
    input_artifacts = residual_audit_artifact.get("input_artifacts")
    if not isinstance(input_artifacts, dict):
        raise ValueError("selection-score residual audit artifact input_artifacts must be an object")
    if input_artifacts.get("source_prior_pressure_audit_artifact_path") != source_prior_pressure_audit_artifact_path:
        raise ValueError(
            "selection-score residual audit artifact input_artifacts source_prior_pressure_audit_artifact_path must match source path"
        )
    if input_artifacts.get("source_selection_score_artifact_path") != source_selection_score_artifact_path:
        raise ValueError(
            "selection-score residual audit artifact input_artifacts source_selection_score_artifact_path must match source path"
        )
    if (
        input_artifacts.get("source_threshold_relaxed_selection_score_artifact_path")
        != source_threshold_relaxed_selection_score_artifact_path
    ):
        raise ValueError(
            "selection-score residual audit artifact input_artifacts source_threshold_relaxed_selection_score_artifact_path must match source path"
        )


def _thresholds_payload(default_trace_artifact: dict, relaxed_trace_artifact: dict) -> dict:
    return {
        "default": _validated_trace_thresholds(default_trace_artifact, context="default trace artifact"),
        "relaxed": _validated_trace_thresholds(relaxed_trace_artifact, context="relaxed trace artifact"),
        "float_tolerance": FLOAT_TOLERANCE,
    }


def _source_snapshots_payload(
    residual_audit_artifact: dict,
    *,
    default_trace_artifact: dict,
    relaxed_trace_artifact: dict,
) -> dict:
    source_snapshots = residual_audit_artifact.get("source_snapshots")
    if not isinstance(source_snapshots, dict):
        source_snapshots = {}

    payload = {}
    metric_audit_classification = source_snapshots.get("metric_audit_classification")
    if metric_audit_classification is not None:
        if not isinstance(metric_audit_classification, dict):
            raise ValueError(
                "selection-score residual audit artifact source_snapshots.metric_audit_classification must be an object"
            )
        payload["metric_audit_classification"] = copy.deepcopy(metric_audit_classification)

    payload["default_trace_classification"] = copy.deepcopy(default_trace_artifact.get("classification", {}))
    payload["relaxed_trace_classification"] = copy.deepcopy(relaxed_trace_artifact.get("classification", {}))
    payload["default_trace_origin"] = default_trace_artifact.get("trace_origin")
    payload["relaxed_trace_origin"] = relaxed_trace_artifact.get("trace_origin")
    return payload


def _validated_prior_pressure_artifact(prior_pressure_artifact: dict) -> dict:
    if not isinstance(prior_pressure_artifact, dict):
        raise ValueError("prior-pressure audit artifact must be an object")
    if prior_pressure_artifact.get("schema") != SOURCE_PRIOR_PRESSURE_AUDIT_SCHEMA:
        raise ValueError("prior-pressure audit artifact has wrong schema")
    classification = prior_pressure_artifact.get("classification")
    if not isinstance(classification, dict):
        raise ValueError("prior-pressure audit artifact classification must be an object")
    if classification.get("classification") != EXPECTED_PRIOR_PRESSURE_CLASSIFICATION:
        raise ValueError(
            f"prior-pressure audit artifact classification must be {EXPECTED_PRIOR_PRESSURE_CLASSIFICATION}"
        )
    if prior_pressure_artifact.get("decision") != EXPECTED_PRIOR_PRESSURE_DECISION:
        raise ValueError(f"prior-pressure audit artifact decision must be {EXPECTED_PRIOR_PRESSURE_DECISION}")
    _validated_source_artifact(
        prior_pressure_artifact.get("source_artifact"),
        context="prior-pressure audit artifact source_artifact",
    )
    return prior_pressure_artifact


def _validated_trace_artifact(trace_artifact: dict, *, branch: str) -> dict:
    if not isinstance(trace_artifact, dict):
        raise ValueError(f"{branch} trace artifact must be an object")
    if trace_artifact.get("schema") != SOURCE_SELECTION_SCORE_SCHEMA:
        raise ValueError(f"{branch} trace artifact has wrong schema; expected {SOURCE_SELECTION_SCORE_SCHEMA}")
    classification = trace_artifact.get("classification")
    if not isinstance(classification, dict):
        raise ValueError(f"{branch} trace artifact classification must be an object")
    if classification.get("classification") != EXPECTED_TRACE_CLASSIFICATION:
        raise ValueError(f"{branch} trace artifact classification must be {EXPECTED_TRACE_CLASSIFICATION}")
    if trace_artifact.get("decision") != EXPECTED_TRACE_DECISION:
        raise ValueError(f"{branch} trace artifact decision must be {EXPECTED_TRACE_DECISION}")
    _validated_source_artifact(
        trace_artifact.get("source_artifact"),
        context=f"{branch} trace artifact source_artifact",
    )
    _validated_trace_origin(trace_artifact, context=f"{branch} trace artifact")
    _validated_trace_thresholds(trace_artifact, context=f"{branch} trace artifact")
    _validated_trace_points(trace_artifact, branch=branch)
    return trace_artifact


def _validated_trace_origin(trace_artifact: dict, *, context: str) -> str:
    trace_origin = trace_artifact.get("trace_origin")
    if not isinstance(trace_origin, str) or not trace_origin.strip():
        raise ValueError(f"{context} trace_origin must be a non-empty string")
    return trace_origin


def _validated_trace_thresholds(trace_artifact: dict, *, context: str) -> dict:
    thresholds = trace_artifact.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError(f"{context} thresholds must be an object")
    validated_thresholds = {}
    for key in (
        "meaningful_q_margin",
        "material_selection_score_margin",
        "material_visit_share_margin",
    ):
        try:
            value = _finite_number(thresholds.get(key), context=f"{context} thresholds.{key}")
        except ValueError as error:
            raise ValueError(f"{context} thresholds.{key} must be finite non-negative numeric") from error
        if value < 0.0:
            raise ValueError(f"{context} thresholds.{key} must be finite non-negative numeric")
        validated_thresholds[key] = value
    return validated_thresholds


def _validated_canonicalization_artifact(
    checkpoint_canonicalization_artifact: dict | None,
    *,
    source_artifact_identity: dict,
    validated_default_trace_artifact: dict,
    validated_relaxed_trace_artifact: dict,
) -> dict | None:
    if checkpoint_canonicalization_artifact is None:
        return None
    if not isinstance(checkpoint_canonicalization_artifact, dict):
        raise ValueError("checkpoint canonicalization artifact must be an object")
    if checkpoint_canonicalization_artifact.get("schema") != SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA:
        raise ValueError("checkpoint canonicalization artifact has wrong schema")
    if checkpoint_canonicalization_artifact.get("decision") != EXPECTED_CANONICALIZATION_DECISION:
        raise ValueError(
            f"checkpoint canonicalization artifact decision must be {EXPECTED_CANONICALIZATION_DECISION}"
        )
    if _source_artifact_identity(
        checkpoint_canonicalization_artifact.get("source_artifact"),
        context="checkpoint canonicalization artifact source_artifact",
    ) != source_artifact_identity:
        raise ValueError(
            "checkpoint canonicalization artifact source_artifact must match selection-score residual audit artifact source_artifact"
        )
    canonical_checkpoint_sequences = checkpoint_canonicalization_artifact.get("canonical_checkpoint_sequences")
    if not isinstance(canonical_checkpoint_sequences, dict):
        raise ValueError("checkpoint canonicalization artifact canonical_checkpoint_sequences must be an object")
    canonicalization_status = checkpoint_canonicalization_artifact.get("canonicalization_status")
    if not isinstance(canonicalization_status, dict):
        raise ValueError("checkpoint canonicalization artifact canonicalization_status.safe_for_followup_spec must be true")
    if canonicalization_status.get("safe_for_followup_spec") is not True:
        raise ValueError("checkpoint canonicalization artifact canonicalization_status.safe_for_followup_spec must be true")
    if checkpoint_canonicalization_artifact.get("canonical_sequences_match") is not True:
        raise ValueError("checkpoint canonicalization artifact canonical_sequences_match must be true")
    trace_artifacts_by_branch = {
        "default": validated_default_trace_artifact,
        "relaxed": validated_relaxed_trace_artifact,
    }
    for branch in ("default", "relaxed"):
        branch_sequence = canonical_checkpoint_sequences.get(branch)
        if not isinstance(branch_sequence, list):
            raise ValueError(
                f"checkpoint canonicalization artifact canonical_checkpoint_sequences.{branch} must be a list"
            )
        normalized_branch_sequence = [
            _finite_number(
                value,
                context=f"checkpoint canonicalization artifact canonical_checkpoint_sequences.{branch}[{index}]",
            )
            for index, value in enumerate(branch_sequence)
        ]
        if EXPECTED_CANONICAL_SIMULATION not in normalized_branch_sequence:
            raise ValueError(
                "checkpoint canonicalization artifact canonical_checkpoint_sequences must include canonical simulation 2.0"
            )
        trace_simulations = []
        previous_simulation = None
        seen_simulations = set()
        for index, trace_point in enumerate(_validated_trace_points(trace_artifacts_by_branch[branch], branch=branch)):
            simulation, _ = _trace_point_projection(
                trace_point,
                context=f"{branch} trace trace_points[{index}]",
            )
            if previous_simulation is not None and simulation < previous_simulation:
                raise ValueError("checkpoint sequences must be strictly increasing")
            if simulation != previous_simulation and simulation in seen_simulations:
                raise ValueError("checkpoint sequences must be strictly increasing")
            seen_simulations.add(simulation)
            previous_simulation = simulation
            if simulation not in trace_simulations:
                trace_simulations.append(simulation)
        if normalized_branch_sequence != trace_simulations:
            raise ValueError(
                "checkpoint canonicalization artifact canonical checkpoint sequences must align with supplied traces"
            )
    return checkpoint_canonicalization_artifact


def _source_artifact_identity(source_artifact: dict, *, context: str) -> dict:
    return copy.deepcopy(_validated_source_artifact(source_artifact, context=context))


def _legal_moves(trace_point: dict, *, context: str) -> list[int]:
    moves = trace_point.get("moves")
    if not isinstance(moves, list) or not moves:
        raise ValueError(f"{context}.moves must be a non-empty list")

    legal_moves = []
    seen_moves = set()
    for entry in moves:
        move = entry.get("move") if isinstance(entry, dict) else None
        if isinstance(move, bool) or not isinstance(move, int):
            raise ValueError(f"{context}.moves entries must have integer move ids")
        if move in seen_moves:
            raise ValueError(f"{context}.moves must not contain duplicate move ids")
        seen_moves.add(move)
        legal_moves.append(move)
    return legal_moves


def _residual_components(trace_point: dict, *, move: int, context: str) -> dict:
    move_entry = _move_lookup(trace_point, move=move, context=context)
    selection_score = _finite_number(
        move_entry.get("selection_score"),
        context=f"{context} move {move} selection_score",
    )
    q_value = _finite_number(
        move_entry.get("q_value"),
        context=f"{context} move {move} q_value",
    )
    residual = selection_score - max(q_value, 0.0)
    return {
        "move": move,
        "selection_score": selection_score,
        "q_value": q_value,
        "residual": residual,
        "non_residual_component": max(q_value, 0.0),
    }


def _selected_move_from_scores(scores_by_move: dict[int, float], *, context: str) -> int:
    if not scores_by_move:
        raise ValueError(f"{context} must include at least one legal move")

    best_score = max(scores_by_move.values())
    candidate_moves = [
        move
        for move, score in scores_by_move.items()
        if abs(score - best_score) <= FLOAT_TOLERANCE
    ]
    return min(candidate_moves)


def _baseline_replay_selected_move(trace_point: dict, *, branch: str) -> int:
    scores_by_move = {}
    for move in _legal_moves(trace_point, context=f"{branch} canonical trace point"):
        move_entry = _move_lookup(trace_point, move=move, context=f"{branch} canonical trace point")
        scores_by_move[move] = _finite_number(
            move_entry.get("selection_score"),
            context=f"{branch} baseline_replay move {move} selection_score",
        )
    return _selected_move_from_scores(scores_by_move, context=f"{branch} baseline_replay")


def _selected_residual_neutralized_selected_move(
    trace_point: dict,
    *,
    branch: str,
) -> int:
    scores_by_move = {}
    for move in _legal_moves(trace_point, context=f"{branch} canonical trace point"):
        if move == EXPECTED_SELECTED_MOVE:
            components = _residual_components(
                trace_point,
                move=move,
                context=f"{branch} selected_residual_neutralized",
            )
            scores_by_move[move] = components["non_residual_component"]
        else:
            move_entry = _move_lookup(
                trace_point,
                move=move,
                context=f"{branch} selected_residual_neutralized",
            )
            scores_by_move[move] = _finite_number(
                move_entry.get("selection_score"),
                context=f"{branch} selected_residual_neutralized move {move} selection_score",
            )
    return _selected_move_from_scores(scores_by_move, context=f"{branch} selected_residual_neutralized")


def _all_residuals_flattened_selected_move(trace_point: dict, *, branch: str) -> int:
    scores_by_move = {}
    for move in _legal_moves(trace_point, context=f"{branch} canonical trace point"):
        try:
            components = _residual_components(
                trace_point,
                move=move,
                context=f"{branch} all_residuals_flattened",
            )
        except ValueError as error:
            raise ValueError(
                "all_residuals_flattened requires usable residual evidence for every legal move"
            ) from error
        scores_by_move[move] = components["non_residual_component"]
    return _selected_move_from_scores(scores_by_move, context=f"{branch} all_residuals_flattened")


def _mode_outcome(
    mode: str,
    *,
    default_trace_point: dict,
    relaxed_trace_point: dict,
) -> dict:
    if mode == "baseline_replay":
        selected_moves_by_branch = {
            "default": _baseline_replay_selected_move(default_trace_point, branch="default"),
            "relaxed": _baseline_replay_selected_move(relaxed_trace_point, branch="relaxed"),
        }
    elif mode == "selected_residual_neutralized":
        selected_moves_by_branch = {
            "default": _selected_residual_neutralized_selected_move(
                default_trace_point,
                branch="default",
            ),
            "relaxed": _selected_residual_neutralized_selected_move(
                relaxed_trace_point,
                branch="relaxed",
            ),
        }
    elif mode == "all_residuals_flattened":
        selected_moves_by_branch = {
            "default": _all_residuals_flattened_selected_move(default_trace_point, branch="default"),
            "relaxed": _all_residuals_flattened_selected_move(relaxed_trace_point, branch="relaxed"),
        }
    else:
        raise ValueError(f"unsupported mode: {mode}")

    branches_agree = selected_moves_by_branch["default"] == selected_moves_by_branch["relaxed"]
    selected_move = selected_moves_by_branch["default"] if branches_agree else None
    if mode == "baseline_replay":
        applied_edit_summary = "no ablation edit applied"
    elif mode == "selected_residual_neutralized":
        applied_edit_summary = "neutralized residual contribution for move 0 at simulation 2.0"
    else:
        applied_edit_summary = "flattened residual contribution across all legal moves with usable evidence at simulation 2.0"
    return {
        "mode": mode,
        "validation_status": "ok" if branches_agree else "inconclusive",
        "selected_move": selected_move,
        "preserved_move_zero": selected_move == EXPECTED_SELECTED_MOVE,
        "evidence_summary": (
            f"{mode} preserved move 0"
            if selected_move == EXPECTED_SELECTED_MOVE
            else (f"{mode} changed away from move 0" if selected_move is not None else f"{mode} branches disagreed")
        ),
        "failure_reason": None if branches_agree else "branch disagreement prevented a clean selected-move result",
        "applied_edit_summary": applied_edit_summary,
        "branch_selected_moves": selected_moves_by_branch,
        "branches_agree": branches_agree,
    }


def _classification_from_mode_results(mode_results: list[dict], *, validated_selected_move: int) -> str:
    outcomes_by_mode = {entry["mode"]: entry for entry in mode_results}
    baseline_selected_move = outcomes_by_mode["baseline_replay"]["selected_move"]
    if baseline_selected_move is None:
        return "selected_move_residual_ablation_inconclusive"
    if baseline_selected_move != validated_selected_move:
        return "selected_move_residual_ablation_inconclusive"

    for mode in ("selected_residual_neutralized", "all_residuals_flattened"):
        if outcomes_by_mode[mode]["selected_move"] is None:
            return "selected_move_residual_ablation_inconclusive"
        if outcomes_by_mode[mode]["selected_move"] != validated_selected_move:
            return "selected_move_residual_sensitive"
    return "selected_move_residual_insensitive"


def build_payload(
    residual_audit_artifact: dict,
    prior_pressure_artifact: dict,
    default_trace_artifact: dict,
    relaxed_trace_artifact: dict,
    *,
    source_selection_score_residual_audit_artifact_path: str,
    source_prior_pressure_audit_artifact_path: str,
    source_selection_score_artifact_path: str,
    source_threshold_relaxed_selection_score_artifact_path: str,
    checkpoint_canonicalization_artifact: dict | None = None,
    source_checkpoint_canonicalization_artifact_path: str | None = None,
) -> dict:
    canonicalization_mode = checkpoint_canonicalization_artifact is not None
    if not canonicalization_mode and source_checkpoint_canonicalization_artifact_path is not None:
        raise ValueError(
            "source_checkpoint_canonicalization_artifact_path requires checkpoint canonicalization artifact"
        )
    if canonicalization_mode and source_checkpoint_canonicalization_artifact_path is None:
        raise ValueError(
            "checkpoint canonicalization artifact requires source_checkpoint_canonicalization_artifact_path"
        )
    validated_residual_audit_artifact = _validated_residual_audit_artifact(residual_audit_artifact)
    validated_prior_pressure_artifact = _validated_prior_pressure_artifact(prior_pressure_artifact)
    validated_default_trace_artifact = _validated_trace_artifact(default_trace_artifact, branch="default")
    validated_relaxed_trace_artifact = _validated_trace_artifact(relaxed_trace_artifact, branch="relaxed")
    _validated_residual_audit_input_artifacts(
        validated_residual_audit_artifact,
        source_prior_pressure_audit_artifact_path=source_prior_pressure_audit_artifact_path,
        source_selection_score_artifact_path=source_selection_score_artifact_path,
        source_threshold_relaxed_selection_score_artifact_path=source_threshold_relaxed_selection_score_artifact_path,
    )
    source_artifact = validated_residual_audit_artifact["source_artifact"]
    source_artifact_identity = _source_artifact_identity(
        source_artifact,
        context="selection-score residual audit artifact source_artifact",
    )
    _validated_canonicalization_artifact(
        checkpoint_canonicalization_artifact,
        source_artifact_identity=source_artifact_identity,
        validated_default_trace_artifact=validated_default_trace_artifact,
        validated_relaxed_trace_artifact=validated_relaxed_trace_artifact,
    )
    if _source_artifact_identity(
        validated_prior_pressure_artifact.get("source_artifact"),
        context="prior-pressure audit artifact source_artifact",
    ) != source_artifact_identity:
        raise ValueError(
            "prior-pressure audit artifact source_artifact must match selection-score residual audit artifact source_artifact"
        )
    if _source_artifact_identity(
        validated_default_trace_artifact.get("source_artifact"),
        context="default trace artifact source_artifact",
    ) != source_artifact_identity:
        raise ValueError(
            "default trace artifact source_artifact must match selection-score residual audit artifact source_artifact"
        )
    if _source_artifact_identity(
        validated_relaxed_trace_artifact.get("source_artifact"),
        context="relaxed trace artifact source_artifact",
    ) != source_artifact_identity:
        raise ValueError(
            "relaxed trace artifact source_artifact must match selection-score residual audit artifact source_artifact"
        )
    default_canonical_trace_point = _canonical_trace_point_at_simulation(
        validated_default_trace_artifact,
        branch="default",
        canonicalization_mode=canonicalization_mode,
    )
    relaxed_canonical_trace_point = _canonical_trace_point_at_simulation(
        validated_relaxed_trace_artifact,
        branch="relaxed",
        canonicalization_mode=canonicalization_mode,
    )
    baseline_result = _mode_outcome(
        "baseline_replay",
        default_trace_point=default_canonical_trace_point,
        relaxed_trace_point=relaxed_canonical_trace_point,
    )
    validated_selected_move = source_artifact["full_search_selected_move"]

    mode_results = [
        baseline_result,
        _mode_outcome(
            "selected_residual_neutralized",
            default_trace_point=default_canonical_trace_point,
            relaxed_trace_point=relaxed_canonical_trace_point,
        ),
        _mode_outcome(
            "all_residuals_flattened",
            default_trace_point=default_canonical_trace_point,
            relaxed_trace_point=relaxed_canonical_trace_point,
        ),
    ]
    classification_name = _classification_from_mode_results(
        mode_results,
        validated_selected_move=validated_selected_move,
    )

    input_artifacts = {
        "source_selection_score_residual_audit_artifact_path": source_selection_score_residual_audit_artifact_path,
        "source_prior_pressure_audit_artifact_path": source_prior_pressure_audit_artifact_path,
        "source_selection_score_artifact_path": source_selection_score_artifact_path,
        "source_threshold_relaxed_selection_score_artifact_path": source_threshold_relaxed_selection_score_artifact_path,
    }
    if (
        checkpoint_canonicalization_artifact is not None
        and source_checkpoint_canonicalization_artifact_path is not None
    ):
        input_artifacts["source_checkpoint_canonicalization_artifact_path"] = (
            source_checkpoint_canonicalization_artifact_path
        )

    return {
        "schema": SCHEMA,
        "hypothesis": "residual_ablation",
        "classification": {
            "classification": classification_name,
            "evidence_summary": classification_name.replace("_", " "),
        },
        "decision": CLASSIFICATION_DECISIONS[classification_name],
        "input_artifacts": input_artifacts,
        "source_artifact": copy.deepcopy(source_artifact),
        "checkpoint": _checkpoint_payload(
            default_trace_point=default_canonical_trace_point,
            relaxed_trace_point=relaxed_canonical_trace_point,
            source_artifact=source_artifact,
        ),
        "thresholds_evaluated": _thresholds_payload(
            validated_default_trace_artifact,
            validated_relaxed_trace_artifact,
        ),
        "mode_results": mode_results,
        "mode_comparison": {
            "baseline_selected_move": mode_results[0]["selected_move"],
            "selected_residual_neutralized_selected_move": mode_results[1]["selected_move"],
            "selected_residual_neutralized_changed_away_from_baseline": (
                mode_results[0]["selected_move"] is not None
                and mode_results[1]["selected_move"] is not None
                and mode_results[1]["selected_move"] != mode_results[0]["selected_move"]
            ),
            "all_residuals_flattened_selected_move": mode_results[2]["selected_move"],
            "all_residuals_flattened_changed_away_from_baseline": (
                mode_results[0]["selected_move"] is not None
                and mode_results[2]["selected_move"] is not None
                and mode_results[2]["selected_move"] != mode_results[0]["selected_move"]
            ),
        },
        "source_snapshots": _source_snapshots_payload(
            validated_residual_audit_artifact,
            default_trace_artifact=validated_default_trace_artifact,
            relaxed_trace_artifact=validated_relaxed_trace_artifact,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    residual_audit_artifact = load_json(args.source_selection_score_residual_audit_artifact)
    prior_pressure_artifact = load_json(args.source_prior_pressure_audit_artifact)
    default_trace_artifact = load_json(args.source_selection_score_artifact)
    relaxed_trace_artifact = load_json(args.source_threshold_relaxed_selection_score_artifact)
    checkpoint_canonicalization_artifact = None
    if args.source_checkpoint_canonicalization_artifact is not None:
        checkpoint_canonicalization_artifact = load_json(args.source_checkpoint_canonicalization_artifact)

    payload = build_payload(
        residual_audit_artifact,
        prior_pressure_artifact,
        default_trace_artifact,
        relaxed_trace_artifact,
        source_selection_score_residual_audit_artifact_path=str(args.source_selection_score_residual_audit_artifact),
        source_prior_pressure_audit_artifact_path=str(args.source_prior_pressure_audit_artifact),
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
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact_path": str(args.out),
                "classification": payload["classification"]["classification"],
                "decision": payload["decision"],
                "schema": payload["schema"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
