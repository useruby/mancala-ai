import argparse
import copy
import json
import math
from pathlib import Path


SCHEMA = "azlite_capture_002_selection_score_trace_v1"
SOURCE_SHARED_DRIFT_SCHEMA = "azlite_shared_full_search_drift_diagnostic_v1"
ROW_ID = "capture_available-002"
EXPECTED_SOURCE_DECISION = "write_row_split_followup_spec"
THRESHOLDS = {
    "meaningful_q_margin": 0.03,
    "material_selection_score_margin": 0.05,
    "material_visit_share_margin": 0.05,
}
REQUIRED_SEARCH_SETTINGS_KEYS = [
    "c_puct",
    "fpu_mode",
    "normalize_values",
    "reuse_subtree",
    "root_policy_mode",
    "tactical_root_bias",
]
CLASSIFICATION_DECISIONS = {
    "selection_score_pressure_confirmed": "write_002_selection_pressure_ablation_spec",
    "q_support_precedes_selection_score": "write_002_child_value_audit_spec",
    "trace_insufficient": "write_002_trace_capture_spec",
    "unresolved": "write_002_unresolved_trace_review_spec",
}
TRACE_ORIGINS = {"extracted", "rerun", "insufficient"}


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _finite_number(value, *, context: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"source shared drift artifact {context} must be finite numeric")
    return float(value)


def _finite_non_negative_number(value, *, context: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{context} must be finite non-negative")
    return float(value)


def _move_entry(trace_point, move):
    if not isinstance(trace_point, dict):
        return None
    for move_entry in trace_point.get("moves") or []:
        if move_entry.get("move") == move:
            return move_entry
    return None


def _validate_trace_point(point, *, context: str, legal_moves: list[int]):
    if not isinstance(point, dict):
        raise ValueError(f"source shared drift artifact {context} must be a dict")

    for required_key in ["selected_move", "simulation", "visits"]:
        if required_key not in point:
            raise ValueError(f"source shared drift artifact {context} missing {required_key}")

    selected_move = point["selected_move"]
    if isinstance(selected_move, bool) or not isinstance(selected_move, int) or selected_move not in legal_moves:
        raise ValueError(f"source shared drift artifact {context} selected_move must be legal int")

    _finite_number(point["simulation"], context=f"{context} simulation")

    visits = point["visits"]
    if not isinstance(visits, list):
        raise ValueError(f"source shared drift artifact {context} visits must be a list")
    for index, visit in enumerate(visits):
        _finite_number(visit, context=f"{context} visits[{index}]")
    if len(visits) <= max(legal_moves, default=-1):
        raise ValueError(f"source shared drift artifact {context} visits must cover legal_moves")

    reference_move_by_prior = point.get("reference_move_by_prior")
    if reference_move_by_prior is not None and (
        isinstance(reference_move_by_prior, bool)
        or not isinstance(reference_move_by_prior, int)
        or reference_move_by_prior not in legal_moves
    ):
        raise ValueError(f"source shared drift artifact {context} reference_move_by_prior must be legal int")

    moves = point.get("moves")
    normalized_moves = None
    if moves is not None:
        if not isinstance(moves, list):
            raise ValueError(f"source shared drift artifact {context} moves must be a list")
        normalized_moves = []
        for move_index, move_entry in enumerate(moves):
            if not isinstance(move_entry, dict):
                raise ValueError(f"source shared drift artifact {context} moves[{move_index}] must be a dict")
            move = move_entry.get("move")
            if isinstance(move, bool) or not isinstance(move, int) or move not in legal_moves:
                raise ValueError(f"source shared drift artifact {context} moves[{move_index}].move must be legal int")

            normalized_move_entry = {"move": int(move)}
            for metric_key in ["selection_score", "q_value"]:
                if metric_key in move_entry:
                    normalized_move_entry[metric_key] = _finite_number(
                        move_entry[metric_key],
                        context=f"{context} moves[{move_index}].{metric_key}",
                    )
            normalized_moves.append(normalized_move_entry)

    normalized_point = {
        "selected_move": selected_move,
        "simulation": float(point["simulation"]),
        "visits": [float(visit) for visit in visits],
    }
    if reference_move_by_prior is not None:
        normalized_point["reference_move_by_prior"] = int(reference_move_by_prior)
    if normalized_moves is not None:
        normalized_point["moves"] = normalized_moves
    return normalized_point


def adapt_trace_points_for_downstream_shared_drift_artifact(trace_points, *, legal_moves: list[int]):
    if not isinstance(trace_points, list):
        raise ValueError("trace_points must be a list")
    return [
        _validate_trace_point(trace_point, context=f"trace_points[{index}]", legal_moves=legal_moves)
        for index, trace_point in enumerate(trace_points)
    ]


def _validate_settings(settings, *, legal_moves: list[int]):
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise ValueError("source shared drift artifact settings must be a dict")

    search_settings = settings.get("search_settings")
    if not isinstance(search_settings, dict):
        raise ValueError("source shared drift artifact settings must include search_settings")
    for key in REQUIRED_SEARCH_SETTINGS_KEYS:
        if key not in search_settings:
            raise ValueError(f"source shared drift artifact settings search_settings missing required key: {key}")

    _finite_number(search_settings["c_puct"], context="settings search_settings c_puct")
    _finite_number(search_settings["tactical_root_bias"], context="settings search_settings tactical_root_bias")

    seed = settings.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("source shared drift artifact settings seed must be an int")

    simulation_count = settings.get("simulation_count")
    if isinstance(simulation_count, bool) or not isinstance(simulation_count, int) or simulation_count <= 0:
        raise ValueError("source shared drift artifact settings simulation_count must be positive int")

    rerun_trace_points = settings.get("rerun_trace_points")
    normalized_rerun_trace_points = None
    if rerun_trace_points is not None:
        if not isinstance(rerun_trace_points, list):
            raise ValueError("source shared drift artifact settings rerun_trace_points must be a list")
        normalized_rerun_trace_points = [
            _validate_trace_point(
                trace_point,
                context=f"settings rerun_trace_points[{index}]",
                legal_moves=legal_moves,
            )
            for index, trace_point in enumerate(rerun_trace_points)
        ]

    normalized_settings = dict(settings)
    normalized_settings["search_settings"] = dict(search_settings)
    normalized_settings["seed"] = int(seed)
    normalized_settings["simulation_count"] = int(simulation_count)
    if normalized_rerun_trace_points is not None:
        normalized_settings["rerun_trace_points"] = normalized_rerun_trace_points
    return normalized_settings


def _validate_row_move_identity(value, *, field: str, legal_moves: list[int]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in legal_moves:
        raise ValueError(f"source shared drift artifact row {ROW_ID} {field} must be a legal move int")
    return int(value)


def load_source_shared_drift_artifact_document(artifact, *, artifact_path: str):
    if not isinstance(artifact, dict):
        raise ValueError("source shared drift artifact must be a JSON object")
    if artifact.get("schema") != SOURCE_SHARED_DRIFT_SCHEMA:
        raise ValueError(
            f"source shared drift artifact has wrong schema: expected {SOURCE_SHARED_DRIFT_SCHEMA}"
        )

    decision = artifact.get("decision")
    if decision != EXPECTED_SOURCE_DECISION:
        raise ValueError(f"source shared drift artifact decision must be {EXPECTED_SOURCE_DECISION}")

    rows = artifact.get("rows")
    if not isinstance(rows, dict):
        raise ValueError("source shared drift artifact rows must be a dict")

    row = rows.get(ROW_ID)
    if not isinstance(row, dict):
        raise ValueError(f"source shared drift artifact missing row {ROW_ID}")
    if row.get("row_id") != ROW_ID:
        raise ValueError(f"source shared drift artifact row {ROW_ID} row_id must match row key")

    legal_moves = row.get("legal_moves")
    if not isinstance(legal_moves, list) or not legal_moves:
        raise ValueError(f"source shared drift artifact row {ROW_ID} legal_moves must be a non-empty list")
    if any(isinstance(move, bool) or not isinstance(move, int) or move < 0 for move in legal_moves):
        raise ValueError(f"source shared drift artifact row {ROW_ID} legal_moves must contain non-negative ints")
    if len(set(legal_moves)) != len(legal_moves):
        raise ValueError(f"source shared drift artifact row {ROW_ID} legal_moves must be unique")
    normalized_legal_moves = [int(move) for move in legal_moves]
    reference_move = _validate_row_move_identity(
        row.get("reference_move"),
        field="reference_move",
        legal_moves=normalized_legal_moves,
    )
    full_search_selected_move = _validate_row_move_identity(
        row.get("full_search_selected_move"),
        field="full_search_selected_move",
        legal_moves=normalized_legal_moves,
    )

    snapshots = row.get("snapshots")
    if not isinstance(snapshots, list):
        raise ValueError(f"source shared drift artifact row {ROW_ID} snapshots must be a list")

    root_start_value = row.get("root_start")
    if root_start_value is None:
        if snapshots:
            raise ValueError(f"source shared drift artifact row {ROW_ID} root_start must be a dict")
        root_start = None
    else:
        root_start = _validate_trace_point(
            root_start_value,
            context=f"row {ROW_ID} root_start",
            legal_moves=normalized_legal_moves,
        )

    normalized_snapshots = []
    previous_simulation = None
    for snapshot_index, snapshot in enumerate(snapshots):
        normalized_snapshot = _validate_trace_point(
            snapshot,
            context=f"row {ROW_ID} snapshots[{snapshot_index}]",
            legal_moves=normalized_legal_moves,
        )
        simulation = normalized_snapshot["simulation"]
        if previous_simulation is not None and simulation < previous_simulation:
            raise ValueError(f"source shared drift artifact row {ROW_ID} snapshots must be ordered by simulation")
        previous_simulation = simulation
        normalized_snapshots.append(normalized_snapshot)

    if root_start is not None and normalized_snapshots and root_start["simulation"] > normalized_snapshots[0]["simulation"]:
        raise ValueError(f"source shared drift artifact row {ROW_ID} root_start must be ordered by simulation")

    settings = _validate_settings(artifact.get("settings"), legal_moves=normalized_legal_moves)

    return {
        "artifact_path": artifact_path,
        "schema": artifact["schema"],
        "decision": decision,
        "classification": artifact.get("classification"),
        "selected_artifact": copy.deepcopy(artifact.get("selected_artifact")),
        "settings": copy.deepcopy(settings),
        "row": {
            **dict(row),
            "legal_moves": normalized_legal_moves,
            "reference_move": reference_move,
            "full_search_selected_move": full_search_selected_move,
            "root_start": root_start,
            "snapshots": normalized_snapshots,
        },
    }


def load_source_shared_drift_artifact(path):
    artifact = load_json(path)
    return load_source_shared_drift_artifact_document(artifact, artifact_path=str(path))


def classify_source_shared_drift_artifact_document(artifact, *, artifact_path: str):
    loaded_artifact = load_source_shared_drift_artifact_document(artifact, artifact_path=artifact_path)
    return build_payload(loaded_artifact)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-shared-drift-artifact",
        required=True,
        type=Path,
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-rerun-capture", action="store_true")
    parser.add_argument("--meaningful-q-margin", type=float, default=None)
    parser.add_argument("--material-selection-score-margin", type=float, default=None)
    parser.add_argument("--material-visit-share-margin", type=float, default=None)
    return parser.parse_args(argv)


def _target_minus_reference_metric(trace_point, metric_key, *, target_move, reference_move):
    selected_entry = _move_entry(trace_point, target_move)
    reference_entry = _move_entry(trace_point, reference_move)
    if selected_entry is None or reference_entry is None:
        return None

    selected_value = selected_entry.get(metric_key)
    reference_value = reference_entry.get(metric_key)
    if (
        isinstance(selected_value, bool)
        or not isinstance(selected_value, (int, float))
        or not math.isfinite(selected_value)
        or isinstance(reference_value, bool)
        or not isinstance(reference_value, (int, float))
        or not math.isfinite(reference_value)
    ):
        return None

    return float(selected_value) - float(reference_value)


def _first_snapshot_where_margin_at_least(trace_points, metric, threshold):
    for trace_point in trace_points:
        margin = metric(trace_point)
        if margin is not None and margin >= threshold:
            return trace_point
    return None


def _visit_share_delta(trace_point, *, target_move, reference_move):
    visits = trace_point.get("visits")
    if (
        not isinstance(visits, list)
        or isinstance(target_move, bool)
        or not isinstance(target_move, int)
        or isinstance(reference_move, bool)
        or not isinstance(reference_move, int)
        or target_move >= len(visits)
        or reference_move >= len(visits)
    ):
        return None

    if any(isinstance(visit, bool) or not isinstance(visit, (int, float)) or not math.isfinite(visit) for visit in visits):
        return None

    total_visits = float(sum(visits))
    if total_visits <= 0.0:
        return None

    return (float(visits[target_move]) / total_visits) - (float(visits[reference_move]) / total_visits)


def _insufficient_payload(source_artifact, trace_points, insufficiency_reasons, *, thresholds, settings=None):
    selected_artifact = source_artifact.get("selected_artifact")
    row = source_artifact.get("row") or {}
    source_artifact_payload = {
        "artifact_path": source_artifact.get("artifact_path"),
        "schema": source_artifact.get("schema"),
        "decision": source_artifact.get("decision"),
        "classification": copy.deepcopy(source_artifact.get("classification")),
        "row_id": row.get("row_id"),
        "reference_move": row.get("reference_move"),
        "full_search_selected_move": row.get("full_search_selected_move"),
    }
    if selected_artifact is not None:
        source_artifact_payload["selected_artifact"] = copy.deepcopy(selected_artifact)

    classification = _classify_payload({"trace_origin": "insufficient"})

    return {
        "schema": SCHEMA,
        "trace_origin": "insufficient",
        "thresholds": dict(thresholds),
        "source_artifact": source_artifact_payload,
        "settings": copy.deepcopy(settings),
        "trace_points": copy.deepcopy(trace_points),
        "first_selected_selection_score_overtake_snapshot": None,
        "first_selected_meaningful_q_support_snapshot": None,
        "first_selected_material_visit_share_snapshot": None,
        "final_selected_minus_reference_selection_score": None,
        "final_selected_minus_reference_q": None,
        "final_selected_minus_reference_visit_share": None,
        "classification": classification["classification"],
        "decision": classification["decision"],
        "insufficiency_reasons": list(insufficiency_reasons),
    }


def _trace_sufficiency(trace_points, *, reference_move, full_search_selected_move):
    insufficiency_reasons = []
    if not trace_points:
        insufficiency_reasons.append("missing_trace_points")
        return {
            "insufficiency_reasons": insufficiency_reasons,
            "final_trace_point": None,
            "final_selection_score": None,
            "final_q_margin": None,
            "final_visit_share": None,
        }

    if len(trace_points) < 2:
        insufficiency_reasons.append("too_few_trace_points")

    final_trace_point = trace_points[-1]
    final_selection_score = _target_minus_reference_metric(
        final_trace_point,
        "selection_score",
        target_move=full_search_selected_move,
        reference_move=reference_move,
    )
    final_q_margin = _target_minus_reference_metric(
        final_trace_point,
        "q_value",
        target_move=full_search_selected_move,
        reference_move=reference_move,
    )
    final_visit_share = _visit_share_delta(
        final_trace_point,
        target_move=full_search_selected_move,
        reference_move=reference_move,
    )

    if final_selection_score is None:
        insufficiency_reasons.append("missing_final_selection_score_margin")
    if final_q_margin is None:
        insufficiency_reasons.append("missing_final_q_margin")
    if final_visit_share is None:
        insufficiency_reasons.append("missing_final_visit_share_delta")

    return {
        "insufficiency_reasons": insufficiency_reasons,
        "final_trace_point": final_trace_point,
        "final_selection_score": final_selection_score,
        "final_q_margin": final_q_margin,
        "final_visit_share": final_visit_share,
    }


def _trace_ordering_reasons(trace_points, *, reason_key):
    reasons = []
    previous_simulation = None
    for trace_point in trace_points:
        simulation = trace_point.get("simulation")
        if previous_simulation is not None and simulation < previous_simulation:
            reasons.append(reason_key)
            break
        previous_simulation = simulation
    return reasons


def _trace_pair_alignment_reasons(
    trace_points,
    *,
    reference_move,
    full_search_selected_move,
    reason_key,
):
    reasons = []
    expected_pair = {reference_move, full_search_selected_move}
    for trace_point in trace_points:
        if trace_point.get("selected_move") not in expected_pair:
            reasons.append(reason_key)
            break
        if trace_point.get("reference_move_by_prior") != reference_move:
            reasons.append(reason_key)
            break
    return reasons


def _trace_pair_requirements_reasons(*, reference_move, full_search_selected_move):
    reasons = []
    if isinstance(reference_move, bool) or not isinstance(reference_move, int):
        reasons.append("missing_reference_move")
    if isinstance(full_search_selected_move, bool) or not isinstance(full_search_selected_move, int):
        reasons.append("missing_full_search_selected_move")
    return reasons


def _build_thresholds(*, meaningful_q_margin=None, material_selection_score_margin=None, material_visit_share_margin=None):
    return {
        "meaningful_q_margin": _finite_non_negative_number(
            THRESHOLDS["meaningful_q_margin"] if meaningful_q_margin is None else meaningful_q_margin,
            context="meaningful_q_margin",
        ),
        "material_selection_score_margin": _finite_non_negative_number(
            THRESHOLDS["material_selection_score_margin"]
            if material_selection_score_margin is None
            else material_selection_score_margin,
            context="material_selection_score_margin",
        ),
        "material_visit_share_margin": _finite_non_negative_number(
            THRESHOLDS["material_visit_share_margin"] if material_visit_share_margin is None else material_visit_share_margin,
            context="material_visit_share_margin",
        ),
    }


def _rerun_trace_points_or_reason(rerun_payload):
    if not isinstance(rerun_payload, dict):
        return [], "rerun_payload_malformed"

    rerun_trace_points = rerun_payload.get("trace_points")
    if rerun_trace_points is None:
        return [], None
    if not isinstance(rerun_trace_points, list):
        return [], "rerun_trace_points_malformed"
    return rerun_trace_points, None


def build_payload(
    source_artifact,
    *,
    meaningful_q_margin=None,
    material_selection_score_margin=None,
    material_visit_share_margin=None,
    allow_rerun_capture=False,
    rerun_capture=None,
):
    thresholds = _build_thresholds(
        meaningful_q_margin=meaningful_q_margin,
        material_selection_score_margin=material_selection_score_margin,
        material_visit_share_margin=material_visit_share_margin,
    )

    extracted_trace_points = []
    row = source_artifact.get("row") or {}
    root_start = row.get("root_start")
    if root_start is not None:
        extracted_trace_points.append(copy.deepcopy(root_start))
    extracted_trace_points.extend(copy.deepcopy(row.get("snapshots") or []))

    settings = None
    trace_origin = "extracted"
    trace_points = extracted_trace_points
    reference_move = row.get("reference_move")
    full_search_selected_move = row.get("full_search_selected_move")

    insufficiency_reasons = _trace_pair_requirements_reasons(
        reference_move=reference_move,
        full_search_selected_move=full_search_selected_move,
    )
    sufficiency = _trace_sufficiency(
        trace_points,
        reference_move=reference_move,
        full_search_selected_move=full_search_selected_move,
    )
    if not insufficiency_reasons:
        insufficiency_reasons.extend(
            _trace_pair_alignment_reasons(
                trace_points,
                reference_move=reference_move,
                full_search_selected_move=full_search_selected_move,
                reason_key="trace_points_pair_mismatch",
            )
        )
    insufficiency_reasons.extend(_trace_ordering_reasons(trace_points, reason_key="trace_points_out_of_order"))
    insufficiency_reasons.extend(sufficiency["insufficiency_reasons"])
    final_trace_point = sufficiency["final_trace_point"]
    final_selection_score = sufficiency["final_selection_score"]
    final_q_margin = sufficiency["final_q_margin"]
    final_visit_share = sufficiency["final_visit_share"]

    if not _trace_pair_requirements_reasons(
        reference_move=reference_move,
        full_search_selected_move=full_search_selected_move,
    ) and insufficiency_reasons and allow_rerun_capture and rerun_capture is not None:
        rerun_payload = rerun_capture(source_artifact) or {}
        rerun_trace_points, rerun_malformed_reason = _rerun_trace_points_or_reason(rerun_payload)
        if isinstance(rerun_payload, dict):
            settings = copy.deepcopy(rerun_payload.get("settings"))
        if rerun_trace_points:
            trace_origin = "rerun"
            trace_points = copy.deepcopy(rerun_trace_points)
            insufficiency_reasons = []
            sufficiency = _trace_sufficiency(
                trace_points,
                reference_move=reference_move,
                full_search_selected_move=full_search_selected_move,
            )
            insufficiency_reasons.extend(
                _trace_pair_alignment_reasons(
                    trace_points,
                    reference_move=reference_move,
                    full_search_selected_move=full_search_selected_move,
                    reason_key="rerun_trace_points_pair_mismatch",
                )
            )
            insufficiency_reasons.extend(_trace_ordering_reasons(trace_points, reason_key="rerun_trace_points_out_of_order"))
            insufficiency_reasons.extend(sufficiency["insufficiency_reasons"])
            final_trace_point = sufficiency["final_trace_point"]
            final_selection_score = sufficiency["final_selection_score"]
            final_q_margin = sufficiency["final_q_margin"]
            final_visit_share = sufficiency["final_visit_share"]
        elif rerun_malformed_reason is not None:
            trace_origin = "insufficient"
            trace_points = []
            insufficiency_reasons.append(rerun_malformed_reason)
        else:
            insufficiency_reasons.append("rerun_missing_trace_points")

    if insufficiency_reasons:
        return _insufficient_payload(
            source_artifact,
            trace_points,
            insufficiency_reasons,
            thresholds=thresholds,
            settings=settings,
        )

    selection_score_snapshot = _first_snapshot_where_margin_at_least(
        trace_points,
        lambda trace_point: _target_minus_reference_metric(
            trace_point,
            "selection_score",
            target_move=full_search_selected_move,
            reference_move=reference_move,
        ),
        thresholds["material_selection_score_margin"],
    )
    meaningful_q_snapshot = _first_snapshot_where_margin_at_least(
        trace_points,
        lambda trace_point: _target_minus_reference_metric(
            trace_point,
            "q_value",
            target_move=full_search_selected_move,
            reference_move=reference_move,
        ),
        thresholds["meaningful_q_margin"],
    )
    material_visit_share_snapshot = _first_snapshot_where_margin_at_least(
        trace_points,
        lambda trace_point: _visit_share_delta(
            trace_point,
            target_move=full_search_selected_move,
            reference_move=reference_move,
        ),
        thresholds["material_visit_share_margin"],
    )

    classification = _classify_payload(
        {
            "trace_origin": trace_origin,
            "first_selected_selection_score_overtake_snapshot": selection_score_snapshot,
            "first_selected_meaningful_q_support_snapshot": meaningful_q_snapshot,
            "first_selected_material_visit_share_snapshot": material_visit_share_snapshot,
        }
    )

    selected_artifact = source_artifact.get("selected_artifact")
    source_artifact_payload = {
        "artifact_path": source_artifact.get("artifact_path"),
        "schema": source_artifact.get("schema"),
        "decision": source_artifact.get("decision"),
        "classification": copy.deepcopy(source_artifact.get("classification")),
        "row_id": row.get("row_id"),
        "reference_move": row.get("reference_move"),
        "full_search_selected_move": row.get("full_search_selected_move"),
    }
    if selected_artifact is not None:
        source_artifact_payload["selected_artifact"] = copy.deepcopy(selected_artifact)

    return {
        "schema": SCHEMA,
        "trace_origin": trace_origin,
        "thresholds": dict(thresholds),
        "source_artifact": source_artifact_payload,
        "settings": settings,
        "trace_points": copy.deepcopy(trace_points),
        "first_selected_selection_score_overtake_snapshot": copy.deepcopy(selection_score_snapshot),
        "first_selected_meaningful_q_support_snapshot": copy.deepcopy(meaningful_q_snapshot),
        "first_selected_material_visit_share_snapshot": copy.deepcopy(material_visit_share_snapshot),
        "final_selected_minus_reference_selection_score": final_selection_score,
        "final_selected_minus_reference_q": final_q_margin,
        "final_selected_minus_reference_visit_share": final_visit_share,
        "classification": classification["classification"],
        "decision": classification["decision"],
        "insufficiency_reasons": [],
    }


def _classify_payload(payload):
    trace_origin = payload.get("trace_origin")
    if trace_origin == "insufficient":
        return {
            "classification": {
                "classification": "trace_insufficient",
                "evidence_summary": "Selection-score trace is incomplete for a machine-checkable 002 diagnostic.",
            },
            "decision": CLASSIFICATION_DECISIONS["trace_insufficient"],
        }

    selection_snapshot = payload.get("first_selected_selection_score_overtake_snapshot")
    meaningful_q_snapshot = payload.get("first_selected_meaningful_q_support_snapshot")
    material_visit_share_snapshot = payload.get("first_selected_material_visit_share_snapshot")

    selection_simulation = None if selection_snapshot is None else selection_snapshot.get("simulation")
    meaningful_q_simulation = None if meaningful_q_snapshot is None else meaningful_q_snapshot.get("simulation")
    visit_share_simulation = None if material_visit_share_snapshot is None else material_visit_share_snapshot.get("simulation")

    if (
        selection_simulation is not None
        and visit_share_simulation is not None
        and selection_simulation <= visit_share_simulation
        and (meaningful_q_simulation is None or selection_simulation < meaningful_q_simulation)
    ):
        return {
            "classification": {
                "classification": "selection_score_pressure_confirmed",
                "evidence_summary": "Selection-score pressure appears before meaningful child-Q support and visit share is already present or follows.",
            },
            "decision": CLASSIFICATION_DECISIONS["selection_score_pressure_confirmed"],
        }

    if (
        selection_simulation is not None
        and meaningful_q_simulation is not None
        and meaningful_q_simulation < selection_simulation
    ):
        return {
            "classification": {
                "classification": "q_support_precedes_selection_score",
                "evidence_summary": "Meaningful child-Q support appears before any material selection-score overtake.",
            },
            "decision": CLASSIFICATION_DECISIONS["q_support_precedes_selection_score"],
        }

    return {
        "classification": {
            "classification": "unresolved",
            "evidence_summary": "Selection-score and Q-support timing do not cleanly separate for capture 002.",
        },
        "decision": CLASSIFICATION_DECISIONS["unresolved"],
    }


def classify_fixture_payload(payload, thresholds):
    del thresholds
    return _classify_payload(payload)


def _thresholds_from_args(args):
    return _build_thresholds(
        meaningful_q_margin=args.meaningful_q_margin,
        material_selection_score_margin=args.material_selection_score_margin,
        material_visit_share_margin=args.material_visit_share_margin,
    )


def _rerun_capture_from_source_artifact(source_artifact):
    settings = source_artifact.get("settings")
    if not isinstance(settings, dict):
        return {}

    rerun_trace_points = settings.get("rerun_trace_points")
    if not rerun_trace_points:
        return {}

    return {
        "trace_points": copy.deepcopy(rerun_trace_points),
        "settings": {
            "search_settings": copy.deepcopy(settings.get("search_settings")),
            "seed": settings.get("seed"),
            "simulation_count": settings.get("simulation_count"),
        },
    }


def main(argv=None):
    args = parse_args(argv)
    thresholds = _thresholds_from_args(args)
    source_artifact = load_source_shared_drift_artifact(args.source_shared_drift_artifact)
    payload = build_payload(
        source_artifact,
        meaningful_q_margin=thresholds["meaningful_q_margin"],
        material_selection_score_margin=thresholds["material_selection_score_margin"],
        material_visit_share_margin=thresholds["material_visit_share_margin"],
        allow_rerun_capture=args.allow_rerun_capture,
        rerun_capture=_rerun_capture_from_source_artifact if args.allow_rerun_capture else None,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "artifact_path": str(args.out),
                "schema": payload["schema"],
                "decision": payload["decision"],
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
