from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

SCHEMA = "azlite_capture_002_selection_score_residual_audit_v1"
SOURCE_PRIOR_PRESSURE_AUDIT_SCHEMA = (
    "azlite_capture_002_prior_pressure_component_audit_v1"
)
SOURCE_SELECTION_SCORE_SCHEMA = "azlite_capture_002_selection_score_trace_v1"
SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA = (
    "azlite_capture_002_trace_checkpoint_canonicalization_v1"
)
ROW_ID = "capture_available-002"
EXPECTED_PRIOR_PRESSURE_CLASSIFICATION = "selection_score_residual_lead"
EXPECTED_PRIOR_PRESSURE_DECISION = "write_002_selection_score_residual_spec"
EXPECTED_CANONICAL_SIMULATION = 2.0
EXPECTED_TRACE_CLASSIFICATION = "unresolved"
EXPECTED_TRACE_DECISION = "write_002_unresolved_trace_review_spec"
EXPECTED_CANONICALIZATION_DECISION = "write_002_metric_audit_canonical_input_spec"
FLOAT_TOLERANCE = 1e-12
CLASSIFICATION_DECISIONS = {
    "stable_selected_residual_advantage": "write_002_residual_ablation_spec",
    "reference_or_other_move_residual_competes": "write_002_competing_residual_review_spec",
    "tiny_count_residual_ambiguous": "write_002_tiny_count_residual_review_spec",
    "selection_score_residual_inconclusive": "stop_002_selection_score_residual_inconclusive",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit capture 002 selection-score residual signals"
    )
    parser.add_argument(
        "--source-prior-pressure-audit-artifact", type=Path, required=True
    )
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


def _finite_number(value, *, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{context} must be finite numeric")
    return float(value)


def _validated_trace_points(trace_artifact: dict, *, context: str) -> list[dict]:
    trace_points = trace_artifact.get("trace_points")
    if not isinstance(trace_points, list) or not trace_points:
        raise ValueError(f"{context} trace_points must be a non-empty list")
    if not all(isinstance(trace_point, dict) for trace_point in trace_points):
        raise ValueError(f"{context} trace_points must be a list of objects")
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
    return simulation, json.dumps(
        _normalized_projection_value(projection), sort_keys=True
    )


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


def _source_selected_and_reference_moves(
    prior_pressure_source_artifact: dict,
) -> tuple[int, int]:
    selected_move = prior_pressure_source_artifact["full_search_selected_move"]
    reference_move = prior_pressure_source_artifact["reference_move"]
    if isinstance(selected_move, bool) or not isinstance(selected_move, int):
        raise ValueError(
            "prior-pressure audit artifact source_artifact.full_search_selected_move must be an integer"
        )
    if isinstance(reference_move, bool) or not isinstance(reference_move, int):
        raise ValueError(
            "prior-pressure audit artifact source_artifact.reference_move must be an integer"
        )
    return selected_move, reference_move


def _canonical_checkpoint_echo_from_trace_point(
    trace_point: dict,
    *,
    branch: str,
    prior_pressure_source_artifact: dict,
) -> dict:
    selected_move, reference_move = _source_selected_and_reference_moves(
        prior_pressure_source_artifact
    )

    selected_entry = _move_lookup(
        trace_point, move=selected_move, context=f"{branch} canonical trace point"
    )
    reference_entry = _move_lookup(
        trace_point, move=reference_move, context=f"{branch} canonical trace point"
    )
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


def _canonical_trace_point_at_simulation(trace_artifact: dict, *, branch: str) -> dict:
    trace_points = _validated_trace_points(trace_artifact, context=f"{branch} trace")
    matches = []
    for index, trace_point in enumerate(trace_points):
        simulation = _finite_number(
            trace_point.get("simulation"),
            context=f"{branch} trace trace_points[{index}].simulation",
        )
        if abs(simulation - EXPECTED_CANONICAL_SIMULATION) <= FLOAT_TOLERANCE:
            matches.append(trace_point)
    if not matches:
        raise ValueError(
            f"{branch} trace must include canonical simulation {EXPECTED_CANONICAL_SIMULATION}"
        )
    return matches[0]


def _validated_canonical_trace_point_move_ids(
    trace_point: dict,
    *,
    branch: str,
    prior_pressure_source_artifact: dict,
) -> None:
    selected_move, reference_move = _source_selected_and_reference_moves(
        prior_pressure_source_artifact
    )
    _move_lookup(
        trace_point, move=selected_move, context=f"{branch} canonical trace point"
    )
    _move_lookup(
        trace_point, move=reference_move, context=f"{branch} canonical trace point"
    )


def _validated_prior_pressure_source_artifact(prior_pressure_artifact: dict) -> dict:
    source_artifact = prior_pressure_artifact.get("source_artifact")
    if not isinstance(source_artifact, dict):
        raise ValueError(
            "prior-pressure audit artifact source_artifact must be an object"
        )
    if source_artifact.get("row_id") != ROW_ID:
        raise ValueError(
            f"prior-pressure audit artifact source_artifact.row_id must be {ROW_ID}"
        )
    for field in ("reference_move", "full_search_selected_move"):
        value = source_artifact.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"prior-pressure audit artifact source_artifact.{field} must be an integer"
            )
    selected_artifact = source_artifact.get("selected_artifact")
    if not isinstance(selected_artifact, dict):
        raise ValueError(
            "prior-pressure audit artifact source_artifact.selected_artifact must be an object"
        )
    return source_artifact


def _move_residual_evidence(
    trace_point: dict, *, move: int, context: str
) -> dict | None:
    try:
        move_entry = _move_lookup(trace_point, move=move, context=context)
    except ValueError:
        return None

    selection_score = move_entry.get("selection_score")
    q_value = move_entry.get("q_value")
    if (
        isinstance(selection_score, bool)
        or not isinstance(selection_score, (int, float))
        or not math.isfinite(selection_score)
        or isinstance(q_value, bool)
        or not isinstance(q_value, (int, float))
        or not math.isfinite(q_value)
    ):
        return None

    selection_score = float(selection_score)
    q_value = float(q_value)
    return {
        "move": move,
        "selection_score": selection_score,
        "q_value": q_value,
        "residual": selection_score - max(q_value, 0.0),
    }


def _visit_summary(
    trace_point: dict, *, selected_move: int, reference_move: int
) -> dict:
    moves = trace_point.get("moves")
    if not isinstance(moves, list):
        return {"usable": False, "legal_moves": []}

    legal_moves = []
    seen_moves = set()
    has_invalid_move_ids = False
    for entry in moves:
        move = entry.get("move") if isinstance(entry, dict) else None
        if isinstance(move, bool) or not isinstance(move, int) or move in seen_moves:
            has_invalid_move_ids = True
            continue
        seen_moves.add(move)
        legal_moves.append(move)

    visits = trace_point.get("visits")
    if not isinstance(visits, list) or not visits:
        return {"usable": False, "legal_moves": legal_moves}

    normalized_visits = []
    total_visits = 0.0
    for value in visits:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return {"usable": False, "legal_moves": legal_moves}
        normalized_value = float(value)
        if normalized_value < 0.0:
            return {"usable": False, "legal_moves": legal_moves}
        normalized_visits.append(normalized_value)
        total_visits += normalized_value

    if total_visits <= 0.0:
        return {"usable": False, "legal_moves": legal_moves}
    if selected_move >= len(normalized_visits) or reference_move >= len(
        normalized_visits
    ):
        return {"usable": False, "legal_moves": legal_moves}

    positive_visit_moves = [
        index
        for index, visit_count in enumerate(normalized_visits)
        if visit_count > 0.0
    ]
    available_moves = set(legal_moves)
    required_moves = set(positive_visit_moves)
    required_moves.add(selected_move)
    required_moves.add(reference_move)
    if not required_moves.issubset(available_moves):
        return {"usable": False, "legal_moves": legal_moves}

    return {
        "usable": not has_invalid_move_ids,
        "total_visits": total_visits,
        "legal_moves": legal_moves,
        "positive_visit_moves": positive_visit_moves,
    }


def _branch_candidate(
    trace_point: dict,
    *,
    branch: str,
    prior_pressure_source_artifact: dict,
    selection_score_residual_threshold: float,
) -> dict:
    selected_move, reference_move = _source_selected_and_reference_moves(
        prior_pressure_source_artifact
    )

    selected_evidence = _move_residual_evidence(
        trace_point,
        move=selected_move,
        context=f"{branch} canonical trace point",
    )
    reference_evidence = _move_residual_evidence(
        trace_point,
        move=reference_move,
        context=f"{branch} canonical trace point",
    )
    visit_summary = _visit_summary(
        trace_point,
        selected_move=selected_move,
        reference_move=reference_move,
    )

    per_move_residuals = {}
    competing_moves = []
    missing_residual_inputs_moves = []
    for move in visit_summary["legal_moves"]:
        move_evidence = _move_residual_evidence(
            trace_point,
            move=move,
            context=f"{branch} canonical trace point",
        )
        if move_evidence is None:
            per_move_residuals[str(move)] = None
            if move != selected_move:
                missing_residual_inputs_moves.append(move)
            continue
        per_move_residuals[str(move)] = {
            "selection_score": move_evidence["selection_score"],
            "q_value": move_evidence["q_value"],
            "residual": move_evidence["residual"],
        }
        if move == selected_move or selected_evidence is None:
            continue
        if move_evidence["residual"] >= selected_evidence["residual"] - FLOAT_TOLERANCE:
            competing_moves.append(move)

    competitor_evidence = [
        per_move_residuals[str(move)]["residual"]
        for move in visit_summary["legal_moves"]
        if move != selected_move and per_move_residuals[str(move)] is not None
    ]

    branch_evidence = {
        "selected_move": selected_move,
        "reference_move": reference_move,
        "selected_move_evidence": selected_evidence,
        "reference_move_evidence": reference_evidence,
        "selected_has_usable_evidence": selected_evidence is not None,
        "reference_has_usable_evidence": reference_evidence is not None,
        "selected_residual": None
        if selected_evidence is None
        else selected_evidence["residual"],
        "reference_residual": None
        if reference_evidence is None
        else reference_evidence["residual"],
        "selection_score_residual_threshold": selection_score_residual_threshold,
        "legal_moves": visit_summary["legal_moves"],
        "per_move_residuals": per_move_residuals,
        "missing_residual_inputs_moves": missing_residual_inputs_moves,
        "competing_moves": competing_moves,
        "visit_summary": visit_summary,
    }

    if competing_moves:
        branch_evidence["branch_candidate"] = (
            "reference_or_other_move_residual_competes"
        )
        return branch_evidence

    if (
        selected_evidence is None
        or not visit_summary["usable"]
        or bool(missing_residual_inputs_moves)
    ):
        branch_evidence["branch_candidate"] = "tiny_count_residual_ambiguous"
        return branch_evidence

    if competitor_evidence:
        best_competitor_residual = max(competitor_evidence)
    else:
        best_competitor_residual = None

    if (
        selected_evidence["residual"] > 0.0
        and selected_evidence["residual"] >= selection_score_residual_threshold
        and (
            best_competitor_residual is None
            or selected_evidence["residual"] - best_competitor_residual
            > FLOAT_TOLERANCE
        )
    ):
        branch_evidence["branch_candidate"] = "stable_selected_residual_advantage"
        return branch_evidence

    branch_evidence["branch_candidate"] = "selection_score_residual_inconclusive"
    return branch_evidence


def _top_level_classification(branch_residual_evidence: dict) -> str:
    branch_candidates = [
        branch_evidence.get("branch_candidate")
        for branch_evidence in branch_residual_evidence.values()
        if isinstance(branch_evidence, dict)
    ]
    if branch_candidates and all(
        branch_candidate == "stable_selected_residual_advantage"
        for branch_candidate in branch_candidates
    ):
        return "stable_selected_residual_advantage"
    if "reference_or_other_move_residual_competes" in branch_candidates:
        return "reference_or_other_move_residual_competes"
    if "tiny_count_residual_ambiguous" in branch_candidates:
        return "tiny_count_residual_ambiguous"
    return "selection_score_residual_inconclusive"


def _validated_canonical_trace_match(
    trace_artifact: dict,
    *,
    branch: str,
    upstream_checkpoint_echo: dict,
    prior_pressure_source_artifact: dict,
) -> None:
    canonical_trace_point = _canonical_trace_point_at_simulation(
        trace_artifact, branch=branch
    )
    canonical_checkpoint_echo = _canonical_checkpoint_echo_from_trace_point(
        canonical_trace_point,
        branch=branch,
        prior_pressure_source_artifact=prior_pressure_source_artifact,
    )
    upstream_projection = {
        "simulation": _finite_number(
            upstream_checkpoint_echo.get("simulation"),
            context=f"prior-pressure audit {branch} upstream checkpoint simulation",
        ),
        "selection_score_margin": _finite_number(
            upstream_checkpoint_echo.get("selection_score_margin"),
            context=f"prior-pressure audit {branch} upstream checkpoint selection_score_margin",
        ),
        "q_margin": _finite_number(
            upstream_checkpoint_echo.get("q_margin"),
            context=f"prior-pressure audit {branch} upstream checkpoint q_margin",
        ),
    }
    if (
        abs(upstream_projection["simulation"] - canonical_checkpoint_echo["simulation"])
        > FLOAT_TOLERANCE
        or abs(
            upstream_projection["selection_score_margin"]
            - canonical_checkpoint_echo["selection_score_margin"]
        )
        > FLOAT_TOLERANCE
        or abs(upstream_projection["q_margin"] - canonical_checkpoint_echo["q_margin"])
        > FLOAT_TOLERANCE
    ):
        raise ValueError(
            f"prior-pressure audit {branch} upstream checkpoint must match canonical 2.0 trace point"
        )


def _validated_upstream_branch_evidence(
    prior_pressure_artifact: dict, *, branch: str
) -> dict:
    branch_level_evidence = prior_pressure_artifact.get("branch_level_evidence")
    if not isinstance(branch_level_evidence, dict):
        raise ValueError("prior-pressure audit branch_level_evidence must be an object")
    evidence = branch_level_evidence.get(branch)
    if not isinstance(evidence, dict):
        raise ValueError(
            f"prior-pressure audit branch_level_evidence.{branch} must be an object"
        )
    _finite_number(
        evidence.get("selection_score_residual_threshold"),
        context=(
            f"prior-pressure audit branch_level_evidence.{branch}.selection_score_residual_threshold"
        ),
    )
    checkpoint = evidence.get("upstream_checkpoint_echo")
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"prior-pressure audit {branch} upstream checkpoint must be an object"
        )
    simulation = _finite_number(
        checkpoint.get("simulation"),
        context=f"prior-pressure audit {branch} upstream checkpoint simulation",
    )
    if abs(simulation - EXPECTED_CANONICAL_SIMULATION) > FLOAT_TOLERANCE:
        raise ValueError(
            f"prior-pressure audit {branch} upstream checkpoint simulation must be {EXPECTED_CANONICAL_SIMULATION}"
        )
    return evidence


def _validated_canonicalization_artifact(
    artifact: dict | None,
    *,
    source_selection_score_artifact_path: str,
    source_threshold_relaxed_selection_score_artifact_path: str,
    source_checkpoint_canonicalization_artifact_path: str | None,
    prior_pressure_source_artifact: dict,
    validated_thresholds_by_branch: dict,
    default_trace_origin: str,
    relaxed_trace_origin: str,
) -> dict | None:
    if artifact is None:
        return None
    if not isinstance(artifact, dict):
        raise ValueError("checkpoint canonicalization artifact must be an object")
    if artifact.get("schema") != SOURCE_CHECKPOINT_CANONICALIZATION_SCHEMA:
        raise ValueError("checkpoint canonicalization artifact has wrong schema")
    canonical_sequences = artifact.get("canonical_checkpoint_sequences")
    if not isinstance(canonical_sequences, dict):
        raise ValueError(
            "checkpoint canonicalization artifact canonical_checkpoint_sequences must be an object"
        )
    default_sequence = _sequence_values(
        canonical_sequences.get("default"),
        context="checkpoint canonicalization artifact canonical_checkpoint_sequences.default",
    )
    relaxed_sequence = _sequence_values(
        canonical_sequences.get("relaxed"),
        context="checkpoint canonicalization artifact canonical_checkpoint_sequences.relaxed",
    )
    if default_sequence != relaxed_sequence:
        raise ValueError(
            "checkpoint canonicalization artifact canonical checkpoint sequences must match"
        )
    input_artifacts = artifact.get("input_artifacts")
    if not isinstance(input_artifacts, dict):
        raise ValueError(
            "checkpoint canonicalization artifact input_artifacts must be an object"
        )
    if (
        input_artifacts.get("source_selection_score_artifact_path")
        != source_selection_score_artifact_path
    ):
        raise ValueError(
            "checkpoint canonicalization artifact input_artifacts source_selection_score_artifact_path must match source path"
        )
    if (
        input_artifacts.get("source_threshold_relaxed_selection_score_artifact_path")
        != source_threshold_relaxed_selection_score_artifact_path
    ):
        raise ValueError(
            "checkpoint canonicalization artifact input_artifacts source_threshold_relaxed_selection_score_artifact_path must match source path"
        )
    source_artifact = artifact.get("source_artifact")
    if source_artifact != prior_pressure_source_artifact:
        raise ValueError(
            "checkpoint canonicalization artifact source_artifact must match prior-pressure audit artifact source_artifact"
        )
    canonicalization_status = artifact.get("canonicalization_status")
    if not isinstance(canonicalization_status, dict):
        raise ValueError(
            "checkpoint canonicalization artifact canonicalization_status must be an object"
        )
    if canonicalization_status.get("safe_for_followup_spec") is not True:
        raise ValueError(
            "checkpoint canonicalization artifact canonicalization_status.safe_for_followup_spec must be true"
        )
    if artifact.get("canonical_sequences_match") is not True:
        raise ValueError(
            "checkpoint canonicalization artifact canonical_sequences_match must be true"
        )
    thresholds_evaluated = artifact.get("thresholds_evaluated")
    if not isinstance(thresholds_evaluated, dict):
        raise ValueError(
            "checkpoint canonicalization artifact thresholds_evaluated must be an object"
        )
    for branch in ("default", "relaxed"):
        if thresholds_evaluated.get(branch) != validated_thresholds_by_branch[branch]:
            raise ValueError(
                f"checkpoint canonicalization artifact thresholds_evaluated.{branch} must match validated trace thresholds"
            )
    if default_trace_origin != relaxed_trace_origin:
        raise ValueError("trace artifacts trace_origin must match each other")
    trace_origin = artifact.get("trace_origin")
    if trace_origin != default_trace_origin:
        raise ValueError(
            "checkpoint canonicalization artifact trace_origin must match trace artifacts"
        )
    source_path = artifact.get("source_path")
    if (
        source_path is not None
        and source_path != source_checkpoint_canonicalization_artifact_path
    ):
        raise ValueError(
            "checkpoint canonicalization artifact source_path must match source path"
        )
    decision = artifact.get("decision")
    if decision != EXPECTED_CANONICALIZATION_DECISION:
        raise ValueError(
            f"checkpoint canonicalization artifact decision must be {EXPECTED_CANONICALIZATION_DECISION}"
        )
    return artifact


def _validated_canonicalization_trace_context(
    checkpoint_canonicalization_artifact: dict | None,
    *,
    canonicalization_mode: bool,
    source_selection_score_artifact_path: str,
    source_threshold_relaxed_selection_score_artifact_path: str,
    source_checkpoint_canonicalization_artifact_path: str | None,
    prior_pressure_source_artifact: dict,
    validated_thresholds_by_branch: dict,
    default_trace_origin: str,
    relaxed_trace_origin: str,
) -> None:
    if not canonicalization_mode:
        return
    _validated_canonicalization_artifact(
        checkpoint_canonicalization_artifact,
        source_selection_score_artifact_path=source_selection_score_artifact_path,
        source_threshold_relaxed_selection_score_artifact_path=source_threshold_relaxed_selection_score_artifact_path,
        source_checkpoint_canonicalization_artifact_path=source_checkpoint_canonicalization_artifact_path,
        prior_pressure_source_artifact=prior_pressure_source_artifact,
        validated_thresholds_by_branch=validated_thresholds_by_branch,
        default_trace_origin=default_trace_origin,
        relaxed_trace_origin=relaxed_trace_origin,
    )


def _validated_trace_contract(trace_artifact: dict, *, branch: str) -> None:
    if not isinstance(trace_artifact, dict):
        raise ValueError(f"{branch} trace artifact must be an object")
    if trace_artifact.get("schema") != SOURCE_SELECTION_SCORE_SCHEMA:
        raise ValueError(
            f"{branch} trace artifact has wrong schema; expected {SOURCE_SELECTION_SCORE_SCHEMA}"
        )
    classification = trace_artifact.get("classification")
    if not isinstance(classification, dict):
        raise ValueError(f"{branch} trace artifact classification must be an object")
    if classification.get("classification") != EXPECTED_TRACE_CLASSIFICATION:
        raise ValueError(
            f"{branch} trace artifact classification must be {EXPECTED_TRACE_CLASSIFICATION}"
        )
    if trace_artifact.get("decision") != EXPECTED_TRACE_DECISION:
        raise ValueError(
            f"{branch} trace artifact decision must be {EXPECTED_TRACE_DECISION}"
        )
    trace_origin = trace_artifact.get("trace_origin")
    if not isinstance(trace_origin, str) or not trace_origin.strip():
        raise ValueError(
            f"{branch} trace artifact trace_origin must be a non-empty string"
        )


def _validated_thresholds(trace_artifact: dict, *, branch: str) -> dict:
    thresholds = trace_artifact.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError(f"{branch} trace artifact thresholds must be an object")

    validated_thresholds = {}
    for key in (
        "meaningful_q_margin",
        "material_selection_score_margin",
        "material_visit_share_margin",
    ):
        try:
            value = _finite_number(
                thresholds.get(key),
                context=f"{branch} trace artifact thresholds.{key}",
            )
        except ValueError as error:
            raise ValueError(
                f"{branch} trace artifact thresholds.{key} must be finite non-negative numeric"
            ) from error
        if value < 0.0:
            raise ValueError(
                f"{branch} trace artifact thresholds.{key} must be finite non-negative numeric"
            )
        validated_thresholds[key] = value
    return validated_thresholds


def _validate_duplicate_equivalence(
    trace_artifact: dict,
    *,
    branch: str,
    canonicalization_mode: bool,
    checkpoint_canonicalization_artifact: dict | None,
    source_selection_score_artifact_path: str,
    source_threshold_relaxed_selection_score_artifact_path: str,
    source_checkpoint_canonicalization_artifact_path: str | None,
    prior_pressure_source_artifact: dict,
) -> dict:
    raw_trace_points = _validated_trace_points(
        trace_artifact, context=f"{branch} trace"
    )
    canonical_trace_projections = []
    for index, trace_point in enumerate(raw_trace_points):
        simulation, projection = _trace_point_projection(
            trace_point,
            context=f"{branch} raw trace_points[{index}]",
        )
        if abs(simulation - EXPECTED_CANONICAL_SIMULATION) <= FLOAT_TOLERANCE:
            canonical_trace_projections.append(projection)

    if not canonicalization_mode:
        if len(set(canonical_trace_projections)) > 1:
            raise ValueError(
                f"{branch} raw duplicate {EXPECTED_CANONICAL_SIMULATION} checkpoint requires canonicalization artifact"
            )
        return {
            "canonical_simulation": EXPECTED_CANONICAL_SIMULATION,
            "raw_match_count": len(canonical_trace_projections),
            "all_raw_matches_canonical": True,
        }

    canonicalization_artifact = checkpoint_canonicalization_artifact
    if canonicalization_artifact is None:
        raise ValueError(
            "canonicalization path requires checkpoint_canonicalization_artifact"
        )
    if not isinstance(canonicalization_artifact, dict):
        raise ValueError("checkpoint canonicalization artifact must be an object")
    canonical_sequences = canonicalization_artifact.get(
        "canonical_checkpoint_sequences"
    )
    if not isinstance(canonical_sequences, dict):
        raise ValueError(
            "checkpoint canonicalization artifact canonical_checkpoint_sequences must be an object"
        )
    canonical_sequence = _sequence_values(
        canonical_sequences.get(branch),
        context=f"checkpoint canonicalization artifact canonical_checkpoint_sequences.{branch}",
    )
    try:
        canonical_trace_points = _select_canonical_trace_points(
            raw_trace_points,
            canonical_sequence=canonical_sequence,
            context=f"{branch} trace canonical selection",
        )
    except ValueError as error:
        if (
            str(error)
            == "skipped duplicate checkpoint must match kept checkpoint contents"
        ):
            raise ValueError(
                f"{branch} raw duplicate {EXPECTED_CANONICAL_SIMULATION} checkpoint must match canonical projection"
            ) from error
        raise

    canonical_projection = None
    for index, trace_point in enumerate(canonical_trace_points):
        simulation, projection = _trace_point_projection(
            trace_point,
            context=f"{branch} selected canonical trace_points[{index}]",
        )
        if abs(simulation - EXPECTED_CANONICAL_SIMULATION) <= FLOAT_TOLERANCE:
            canonical_projection = projection
            break
    if canonical_projection is None:
        raise ValueError(
            f"{branch} trace must include canonical simulation {EXPECTED_CANONICAL_SIMULATION}"
        )

    duplicate_count = 0
    for index, trace_point in enumerate(raw_trace_points):
        simulation, projection = _trace_point_projection(
            trace_point,
            context=f"{branch} raw trace_points[{index}]",
        )
        if abs(simulation - EXPECTED_CANONICAL_SIMULATION) > FLOAT_TOLERANCE:
            continue
        duplicate_count += 1
        if projection != canonical_projection:
            raise ValueError(
                f"{branch} raw duplicate {EXPECTED_CANONICAL_SIMULATION} checkpoint must match canonical projection"
            )
    if duplicate_count == 0:
        raise ValueError(
            f"{branch} trace must include canonical simulation {EXPECTED_CANONICAL_SIMULATION}"
        )
    return {
        "canonical_simulation": EXPECTED_CANONICAL_SIMULATION,
        "raw_match_count": duplicate_count,
        "all_raw_matches_canonical": True,
    }


def _checkpoint_payload(prior_pressure_artifact: dict) -> dict:
    return {
        "canonical_simulation": EXPECTED_CANONICAL_SIMULATION,
        "default_upstream_checkpoint_echo": copy.deepcopy(
            prior_pressure_artifact["branch_level_evidence"]["default"][
                "upstream_checkpoint_echo"
            ]
        ),
        "relaxed_upstream_checkpoint_echo": copy.deepcopy(
            prior_pressure_artifact["branch_level_evidence"]["relaxed"][
                "upstream_checkpoint_echo"
            ]
        ),
    }


def _branch_disagreement_summary(branch_residual_evidence: dict) -> dict:
    default_candidate = branch_residual_evidence["default"]["branch_candidate"]
    relaxed_candidate = branch_residual_evidence["relaxed"]["branch_candidate"]
    return {
        "all_branches_agree": default_candidate == relaxed_candidate,
        "default_branch_candidate": default_candidate,
        "relaxed_branch_candidate": relaxed_candidate,
    }


def build_payload(
    prior_pressure_artifact: dict,
    default_trace_artifact: dict,
    relaxed_trace_artifact: dict,
    *,
    source_prior_pressure_audit_artifact_path: str,
    source_selection_score_artifact_path: str,
    source_threshold_relaxed_selection_score_artifact_path: str,
    checkpoint_canonicalization_artifact=None,
    source_checkpoint_canonicalization_artifact_path=None,
) -> dict:
    input_artifacts = {
        "source_prior_pressure_audit_artifact_path": source_prior_pressure_audit_artifact_path,
        "source_selection_score_artifact_path": source_selection_score_artifact_path,
        "source_threshold_relaxed_selection_score_artifact_path": source_threshold_relaxed_selection_score_artifact_path,
    }
    if source_checkpoint_canonicalization_artifact_path is not None:
        input_artifacts["source_checkpoint_canonicalization_artifact_path"] = (
            source_checkpoint_canonicalization_artifact_path
        )
    canonicalization_mode = source_checkpoint_canonicalization_artifact_path is not None
    if checkpoint_canonicalization_artifact is not None and not canonicalization_mode:
        raise ValueError(
            "checkpoint_canonicalization_artifact requires canonicalization path"
        )
    if not isinstance(prior_pressure_artifact, dict):
        raise ValueError("prior-pressure audit artifact must be an object")

    if prior_pressure_artifact.get("schema") != SOURCE_PRIOR_PRESSURE_AUDIT_SCHEMA:
        raise ValueError("prior-pressure audit artifact has wrong schema")
    prior_pressure_classification = prior_pressure_artifact.get("classification")
    if not isinstance(prior_pressure_classification, dict):
        raise ValueError(
            "prior-pressure audit artifact classification must be an object"
        )
    if (
        prior_pressure_classification.get("classification")
        != EXPECTED_PRIOR_PRESSURE_CLASSIFICATION
    ):
        raise ValueError(
            f"prior-pressure audit artifact classification must be {EXPECTED_PRIOR_PRESSURE_CLASSIFICATION}"
        )
    if prior_pressure_artifact.get("decision") != EXPECTED_PRIOR_PRESSURE_DECISION:
        raise ValueError(
            f"prior-pressure audit artifact decision must be {EXPECTED_PRIOR_PRESSURE_DECISION}"
        )
    prior_pressure_input_artifacts = prior_pressure_artifact.get("input_artifacts")
    if not isinstance(prior_pressure_input_artifacts, dict):
        raise ValueError(
            "prior-pressure audit artifact input_artifacts must be an object"
        )
    if (
        prior_pressure_input_artifacts.get("source_selection_score_artifact_path")
        != source_selection_score_artifact_path
    ):
        raise ValueError(
            "prior-pressure audit upstream input_artifacts source_selection_score_artifact_path must match source path"
        )
    if (
        prior_pressure_input_artifacts.get(
            "source_threshold_relaxed_selection_score_artifact_path"
        )
        != source_threshold_relaxed_selection_score_artifact_path
    ):
        raise ValueError(
            "prior-pressure audit upstream input_artifacts source_threshold_relaxed_selection_score_artifact_path must match source path"
        )
    canonicalization_input_path = prior_pressure_input_artifacts.get(
        "source_checkpoint_canonicalization_artifact_path"
    )
    if canonicalization_mode:
        if (
            canonicalization_input_path
            != source_checkpoint_canonicalization_artifact_path
        ):
            raise ValueError(
                "prior-pressure audit upstream input_artifacts source_checkpoint_canonicalization_artifact_path must match source path"
            )
    elif canonicalization_input_path is not None:
        raise ValueError(
            "prior-pressure audit upstream input_artifacts source_checkpoint_canonicalization_artifact_path must be absent outside canonical mode"
        )

    prior_pressure_source_artifact = _validated_prior_pressure_source_artifact(
        prior_pressure_artifact
    )

    validated_thresholds_by_branch = {}
    duplicate_equivalence_audit = {}
    for branch, trace_artifact in (
        ("default", default_trace_artifact),
        ("relaxed", relaxed_trace_artifact),
    ):
        _validated_trace_contract(trace_artifact, branch=branch)
        validated_thresholds_by_branch[branch] = _validated_thresholds(
            trace_artifact, branch=branch
        )
        if trace_artifact.get("source_artifact") != prior_pressure_source_artifact:
            raise ValueError(
                f"{branch} trace artifact source_artifact must match prior-pressure audit artifact source_artifact"
            )
        canonical_trace_point = _canonical_trace_point_at_simulation(
            trace_artifact, branch=branch
        )
        _validated_canonical_trace_point_move_ids(
            canonical_trace_point,
            branch=branch,
            prior_pressure_source_artifact=prior_pressure_source_artifact,
        )
        branch_evidence = _validated_upstream_branch_evidence(
            prior_pressure_artifact, branch=branch
        )
        _validated_canonical_trace_match(
            trace_artifact,
            branch=branch,
            upstream_checkpoint_echo=branch_evidence["upstream_checkpoint_echo"],
            prior_pressure_source_artifact=prior_pressure_source_artifact,
        )
        duplicate_equivalence_audit[branch] = _validate_duplicate_equivalence(
            trace_artifact,
            branch=branch,
            canonicalization_mode=canonicalization_mode,
            checkpoint_canonicalization_artifact=checkpoint_canonicalization_artifact,
            source_selection_score_artifact_path=source_selection_score_artifact_path,
            source_threshold_relaxed_selection_score_artifact_path=source_threshold_relaxed_selection_score_artifact_path,
            source_checkpoint_canonicalization_artifact_path=source_checkpoint_canonicalization_artifact_path,
            prior_pressure_source_artifact=prior_pressure_source_artifact,
        )

    default_trace_origin = default_trace_artifact.get("trace_origin")
    relaxed_trace_origin = relaxed_trace_artifact.get("trace_origin")
    _validated_canonicalization_trace_context(
        checkpoint_canonicalization_artifact,
        canonicalization_mode=canonicalization_mode,
        source_selection_score_artifact_path=source_selection_score_artifact_path,
        source_threshold_relaxed_selection_score_artifact_path=source_threshold_relaxed_selection_score_artifact_path,
        source_checkpoint_canonicalization_artifact_path=source_checkpoint_canonicalization_artifact_path,
        prior_pressure_source_artifact=prior_pressure_source_artifact,
        validated_thresholds_by_branch=validated_thresholds_by_branch,
        default_trace_origin=default_trace_origin,
        relaxed_trace_origin=relaxed_trace_origin,
    )

    branch_residual_evidence = {}
    for branch, trace_artifact in (
        ("default", default_trace_artifact),
        ("relaxed", relaxed_trace_artifact),
    ):
        canonical_trace_point = _canonical_trace_point_at_simulation(
            trace_artifact, branch=branch
        )
        branch_residual_evidence[branch] = _branch_candidate(
            canonical_trace_point,
            branch=branch,
            prior_pressure_source_artifact=prior_pressure_source_artifact,
            selection_score_residual_threshold=_finite_number(
                prior_pressure_artifact["branch_level_evidence"][branch][
                    "selection_score_residual_threshold"
                ],
                context=(
                    f"prior-pressure audit branch_level_evidence.{branch}.selection_score_residual_threshold"
                ),
            ),
        )

    classification_name = _top_level_classification(branch_residual_evidence)
    return {
        "schema": SCHEMA,
        "hypothesis": "selection_score_residual_audit",
        "classification": {
            "classification": classification_name,
            "evidence_summary": classification_name.replace("_", " "),
        },
        "decision": CLASSIFICATION_DECISIONS[classification_name],
        "input_artifacts": input_artifacts,
        "source_artifact": copy.deepcopy(prior_pressure_artifact["source_artifact"]),
        "thresholds_evaluated": {
            "default": {
                **copy.deepcopy(validated_thresholds_by_branch["default"]),
                "selection_score_residual_threshold": _finite_number(
                    prior_pressure_artifact["branch_level_evidence"]["default"][
                        "selection_score_residual_threshold"
                    ],
                    context="prior-pressure audit branch_level_evidence.default.selection_score_residual_threshold",
                ),
            },
            "relaxed": {
                **copy.deepcopy(validated_thresholds_by_branch["relaxed"]),
                "selection_score_residual_threshold": _finite_number(
                    prior_pressure_artifact["branch_level_evidence"]["relaxed"][
                        "selection_score_residual_threshold"
                    ],
                    context="prior-pressure audit branch_level_evidence.relaxed.selection_score_residual_threshold",
                ),
            },
            "float_tolerance": FLOAT_TOLERANCE,
        },
        "checkpoint": _checkpoint_payload(prior_pressure_artifact),
        "duplicate_equivalence_audit": duplicate_equivalence_audit,
        "branch_residual_evidence": branch_residual_evidence,
        "branch_disagreement_summary": _branch_disagreement_summary(
            branch_residual_evidence
        ),
        "source_snapshots": copy.deepcopy(
            prior_pressure_artifact.get("source_snapshots", {})
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prior_pressure_artifact = load_json(args.source_prior_pressure_audit_artifact)
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
        prior_pressure_artifact,
        default_trace_artifact,
        relaxed_trace_artifact,
        source_prior_pressure_audit_artifact_path=str(
            args.source_prior_pressure_audit_artifact
        ),
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
                "classification": payload["classification"]["classification"],
                "decision": payload["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
