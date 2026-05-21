from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from ml.alphazero_lite import capture_002_nonseparable_review as nonseparable_review
from ml.alphazero_lite import capture_002_selection_score_trace as selection_score_trace
from ml.alphazero_lite import capture_002_trace_cadence_review as trace_cadence_review
from ml.alphazero_lite import capture_002_trace_capture as trace_capture

SCHEMA = "azlite_capture_002_trace_cadence_capture_v1"
SOURCE_TRACE_CAPTURE_SCHEMA = "azlite_capture_002_trace_capture_v1"
SOURCE_SELECTION_SCORE_SCHEMA = "azlite_capture_002_selection_score_trace_v1"
SOURCE_TRACE_CADENCE_REVIEW_SCHEMA = "azlite_capture_002_trace_cadence_review_v1"
SOURCE_NONSEPARABLE_REVIEW_SCHEMA = "azlite_capture_002_nonseparable_review_v1"
ROW_ID = "capture_available-002"
CLASSIFICATION_DECISIONS = {
    "trace_cadence_unresolved": "stop_002_trace_cadence_unresolved",
    "selection_score_pressure_confirmed": "write_002_selection_pressure_ablation_spec",
    "q_support_precedes_selection_score": "write_002_child_value_audit_spec",
    "genuinely_not_separable": "stop_002_unresolved",
}
ALLOWED_INSTRUMENTATION_ONLY_SEARCH_SETTING_CHANGES = set()


def _value_or_fallback(value, fallback):
    return fallback if value is None else value


def load_baseline_inputs(
    *,
    trace_capture_artifact_path: Path,
    selection_score_artifact_path: Path,
    trace_cadence_review_artifact_path: Path,
    nonseparable_review_artifact_path: Path | None,
) -> dict:
    trace_capture_artifact = trace_cadence_review.load_trace_capture_artifact(
        trace_capture_artifact_path
    )
    selection_score_artifact = trace_cadence_review.load_selection_score_artifact(
        selection_score_artifact_path
    )
    cadence_review_artifact = trace_cadence_review.load_json(
        trace_cadence_review_artifact_path
    )

    if cadence_review_artifact.get("schema") != SOURCE_TRACE_CADENCE_REVIEW_SCHEMA:
        raise ValueError(
            f"trace cadence review artifact has wrong schema: expected {SOURCE_TRACE_CADENCE_REVIEW_SCHEMA}"
        )
    if (cadence_review_artifact.get("classification") or {}).get(
        "classification"
    ) != "trace_too_sparse":
        raise ValueError("trace cadence review artifact must classify trace_too_sparse")
    if (
        cadence_review_artifact.get("decision")
        != "write_002_trace_cadence_capture_spec"
    ):
        raise ValueError(
            "trace cadence review artifact must emit write_002_trace_cadence_capture_spec"
        )

    nonseparable_review_artifact = None
    if nonseparable_review_artifact_path is not None:
        nonseparable_review_artifact = nonseparable_review.load_json(
            nonseparable_review_artifact_path
        )
        if (
            nonseparable_review_artifact.get("schema")
            != SOURCE_NONSEPARABLE_REVIEW_SCHEMA
        ):
            raise ValueError(
                f"nonseparable review artifact has wrong schema: expected {SOURCE_NONSEPARABLE_REVIEW_SCHEMA}"
            )
        if (nonseparable_review_artifact.get("classification") or {}).get(
            "classification"
        ) != "prerequisite_preempted":
            raise ValueError(
                "nonseparable review artifact must classify prerequisite_preempted"
            )
        if (
            nonseparable_review_artifact.get("decision")
            != "write_002_trace_cadence_capture_spec"
        ):
            raise ValueError(
                "nonseparable review artifact must emit write_002_trace_cadence_capture_spec"
            )

    selection_source_artifact = copy.deepcopy(
        selection_score_artifact.get("source_artifact") or {}
    )
    trace_row_context = copy.deepcopy(trace_capture_artifact.get("row_context") or {})
    baseline_row_context = {
        "row_id": trace_row_context.get("row_id")
        or trace_capture_artifact.get("row_id"),
        "canonical_state": trace_row_context.get("canonical_state"),
        "legal_moves": copy.deepcopy(trace_row_context.get("legal_moves")),
        "reference_move": _value_or_fallback(
            trace_row_context.get("reference_move"),
            trace_capture_artifact.get("reference_move"),
        ),
        "full_search_selected_move": _value_or_fallback(
            trace_row_context.get("full_search_selected_move"),
            trace_capture_artifact.get("full_search_selected_move"),
        ),
    }
    if baseline_row_context.get("row_id") != ROW_ID:
        raise ValueError(f"trace capture artifact must target {ROW_ID}")
    if selection_source_artifact.get("row_id") != ROW_ID:
        raise ValueError(f"selection score artifact must target {ROW_ID}")
    for move_field in ("reference_move", "full_search_selected_move"):
        selection_move = selection_source_artifact.get(move_field)
        if selection_move is not None and selection_move != baseline_row_context.get(
            move_field
        ):
            raise ValueError(
                "selection score artifact source_artifact move pair must match trace capture baseline"
            )

    return {
        "trace_capture_artifact": trace_capture_artifact,
        "selection_score_artifact": selection_score_artifact,
        "trace_cadence_review_artifact": cadence_review_artifact,
        "nonseparable_review_artifact": nonseparable_review_artifact,
        "row_context": baseline_row_context,
        "source_selected_artifact": copy.deepcopy(
            selection_source_artifact.get("selected_artifact")
        ),
        "search_settings": copy.deepcopy(
            (trace_capture_artifact.get("upstream_inputs") or {}).get("search_settings")
        ),
        "baseline_trigger": {
            "required_input_schemas": [
                SOURCE_TRACE_CAPTURE_SCHEMA,
                SOURCE_SELECTION_SCORE_SCHEMA,
                SOURCE_TRACE_CADENCE_REVIEW_SCHEMA,
            ]
            + (
                []
                if nonseparable_review_artifact is None
                else [SOURCE_NONSEPARABLE_REVIEW_SCHEMA]
            ),
            "baseline_source_paths": {
                "trace_capture_artifact_path": str(trace_capture_artifact_path),
                "selection_score_artifact_path": str(selection_score_artifact_path),
                "trace_cadence_review_artifact_path": str(
                    trace_cadence_review_artifact_path
                ),
                "nonseparable_review_artifact_path": None
                if nonseparable_review_artifact_path is None
                else str(nonseparable_review_artifact_path),
            },
            "sparse_cadence_trigger_decision": cadence_review_artifact["decision"],
            "baseline_unique_checkpoint_list": copy.deepcopy(
                cadence_review_artifact.get("unique_simulation_checkpoints") or []
            ),
            "baseline_ambiguity_signals": copy.deepcopy(
                cadence_review_artifact.get("ambiguity_signals") or []
            ),
        },
    }


def build_provenance_guard(
    baseline: dict,
    *,
    dense_row_context: dict,
    dense_selected_artifact: dict | None,
    dense_search_settings: dict | None,
    checkpoint_capture_policy: dict,
) -> dict:
    failures = []
    baseline_row_context = baseline["row_context"]
    for field in (
        "row_id",
        "canonical_state",
        "legal_moves",
        "reference_move",
        "full_search_selected_move",
    ):
        if dense_row_context.get(field) != baseline_row_context.get(field):
            failures.append(f"{field}_mismatch")

    if dense_selected_artifact != baseline.get("source_selected_artifact"):
        failures.append("selected_artifact_provenance_mismatch")

    baseline_search_settings = baseline.get("search_settings") or {}
    dense_search_settings = dense_search_settings or {}
    for key in sorted(set(baseline_search_settings) | set(dense_search_settings)):
        if baseline_search_settings.get(key) == dense_search_settings.get(key):
            continue
        if key in ALLOWED_INSTRUMENTATION_ONLY_SEARCH_SETTING_CHANGES:
            continue
        failures.append(f"decision_relevant_search_setting_changed:{key}")

    return {
        "passed": not failures,
        "failures": failures,
        "baseline_row_context": copy.deepcopy(baseline_row_context),
        "dense_row_context": copy.deepcopy(dense_row_context),
        "baseline_selected_artifact": copy.deepcopy(
            baseline.get("source_selected_artifact")
        ),
        "dense_selected_artifact": copy.deepcopy(dense_selected_artifact),
        "baseline_search_settings": copy.deepcopy(baseline_search_settings),
        "dense_search_settings": copy.deepcopy(dense_search_settings),
        "checkpoint_capture_policy": copy.deepcopy(checkpoint_capture_policy),
    }


def _unique_simulation_checkpoints(trace_points: list[dict]) -> list[float]:
    checkpoints = []
    for trace_point in trace_points:
        simulation = float(trace_point["simulation"])
        if simulation not in checkpoints:
            checkpoints.append(simulation)
    return checkpoints


def _first_divergent_selected_move_index(
    trace_points: list[dict], *, reference_move: int, final_selected_move: int
) -> int | None:
    if reference_move == final_selected_move:
        return None
    for index, trace_point in enumerate(trace_points):
        if trace_point.get("selected_move") == final_selected_move:
            return index
    return None


def build_dense_trace(
    baseline: dict, *, dense_trace_points: list[dict], checkpoint_capture_policy: dict
) -> dict:
    row_context = baseline["row_context"]
    normalized_trace_points = copy.deepcopy(dense_trace_points)
    unique_checkpoints = _unique_simulation_checkpoints(normalized_trace_points)
    insufficiency_reasons = []

    if len(unique_checkpoints) <= 2:
        insufficiency_reasons.append(
            "dense trace still collapses to only root/final checkpoints after deduplication"
        )

    divergent_index = _first_divergent_selected_move_index(
        normalized_trace_points,
        reference_move=row_context["reference_move"],
        final_selected_move=row_context["full_search_selected_move"],
    )
    divergent_simulation = None
    if divergent_index is not None:
        divergent_simulation = float(
            normalized_trace_points[divergent_index]["simulation"]
        )

    final_unique_checkpoint = unique_checkpoints[-1] if unique_checkpoints else None
    has_additional_checkpoint_between_divergence_and_final = (
        divergent_simulation is not None
        and final_unique_checkpoint is not None
        and any(
            divergent_simulation < checkpoint < final_unique_checkpoint
            for checkpoint in unique_checkpoints
        )
    )
    if not has_additional_checkpoint_between_divergence_and_final:
        insufficiency_reasons.append(
            "no additional checkpoint exists between the first divergent selected move and the final snapshot"
        )

    return {
        "trace_origin": "dense_rerun",
        "trace_points": normalized_trace_points,
        "unique_simulation_checkpoints": unique_checkpoints,
        "unique_simulation_checkpoint_count": len(unique_checkpoints),
        "duplicate_root_snapshot_count": len(normalized_trace_points)
        - len(unique_checkpoints),
        "first_divergent_selected_move_index": divergent_index,
        "has_additional_checkpoint_between_divergence_and_final": has_additional_checkpoint_between_divergence_and_final,
        "insufficiency_reasons": insufficiency_reasons,
        "checkpoint_capture_policy": copy.deepcopy(checkpoint_capture_policy),
    }


def build_regenerated_shared_drift_artifact(
    baseline: dict,
    *,
    dense_trace: dict,
    source_shared_drift_artifact: dict | None = None,
    trace_cadence_capture_artifact_path: str,
    trace_cadence_capture_artifact_sha256: str,
) -> dict:
    source_shared_drift_artifact = (
        source_shared_drift_artifact
        or baseline["trace_capture_artifact"]["source_shared_drift_artifact"]
    )
    regenerated = copy.deepcopy(
        (source_shared_drift_artifact or {}).get("source_payload")
    )
    row = regenerated["rows"][ROW_ID]
    trace_points = copy.deepcopy(dense_trace["trace_points"])
    row["root_start"] = copy.deepcopy(trace_points[0])
    row["snapshots"] = copy.deepcopy(trace_points[1:])
    final_trace_point = trace_points[-1]
    reference_move = row["reference_move"]
    selected_move = row["full_search_selected_move"]
    row["final_deltas"] = {
        "selected_visits": float(final_trace_point["visits"][selected_move])
        - float(trace_points[0]["visits"][selected_move]),
        "reference_visits": float(final_trace_point["visits"][reference_move])
        - float(trace_points[0]["visits"][reference_move]),
    }
    regenerated["trace_capture_provenance"] = {
        "trace_capture_schema": SCHEMA,
        "trace_origin": dense_trace["trace_origin"],
        "row_id": ROW_ID,
        "trace_capture_artifact_path": trace_cadence_capture_artifact_path,
        "trace_capture_artifact_sha256": trace_cadence_capture_artifact_sha256,
        "source_shared_drift_artifact_path": source_shared_drift_artifact[
            "artifact_path"
        ],
        "source_shared_drift_artifact_sha256": source_shared_drift_artifact[
            "artifact_sha256"
        ],
    }
    return regenerated


def _stable_trace_capture_artifact_sha256(payload: dict) -> str:
    normalized_payload = copy.deepcopy(payload)
    normalized_payload["trace_capture_artifact_sha256"] = None
    artifact_write_summary = normalized_payload.get("artifact_write_summary") or {}
    artifact_write_summary["trace_capture_sha256"] = None
    normalized_payload["artifact_write_summary"] = artifact_write_summary
    regenerated_shared_drift_artifact = (
        normalized_payload.get("regenerated_shared_drift_artifact") or {}
    )
    trace_capture_provenance = (
        regenerated_shared_drift_artifact.get("trace_capture_provenance") or {}
    )
    trace_capture_provenance["trace_capture_artifact_sha256"] = None
    regenerated_shared_drift_artifact["trace_capture_provenance"] = (
        trace_capture_provenance
    )
    normalized_payload["regenerated_shared_drift_artifact"] = (
        regenerated_shared_drift_artifact
    )
    return hashlib.sha256(
        (json.dumps(normalized_payload, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()


def _classification_and_decision(
    *,
    provenance_guard: dict,
    dense_trace: dict,
    dense_selection_score_artifact: dict | None,
    dense_trace_cadence_review_artifact: dict | None,
    dense_threshold_relaxed_selection_score_artifact: dict | None,
    dense_nonseparable_review_artifact: dict | None,
) -> tuple[str, str]:
    classification = "trace_cadence_unresolved"
    decision = CLASSIFICATION_DECISIONS[classification]

    if not provenance_guard.get("passed") or dense_trace.get("insufficiency_reasons"):
        return classification, decision

    dense_cadence_decision = (dense_trace_cadence_review_artifact or {}).get("decision")
    if dense_cadence_decision == "write_002_trace_cadence_capture_spec":
        return classification, decision

    for artifact in (
        dense_selection_score_artifact,
        dense_threshold_relaxed_selection_score_artifact,
    ):
        artifact_classification = ((artifact or {}).get("classification") or {}).get(
            "classification"
        )
        artifact_decision = (artifact or {}).get("decision")
        if (
            artifact_classification
            in {
                "selection_score_pressure_confirmed",
                "q_support_precedes_selection_score",
            }
            and artifact_decision
        ):
            return artifact_classification, artifact_decision

    default_classification = (
        (dense_selection_score_artifact or {}).get("classification") or {}
    ).get("classification")
    relaxed_classification = (
        (dense_threshold_relaxed_selection_score_artifact or {}).get("classification")
        or {}
    ).get("classification")
    if (
        default_classification == "unresolved"
        and relaxed_classification == "unresolved"
        and dense_nonseparable_review_artifact is not None
    ):
        return "genuinely_not_separable", dense_nonseparable_review_artifact["decision"]

    return classification, decision


def build_payload(
    *,
    baseline: dict,
    dense_trace: dict,
    provenance_guard: dict,
    trace_cadence_capture_artifact_path: str,
    regenerated_shared_drift_artifact: dict | None = None,
    dense_selection_score_artifact: dict | None = None,
    dense_trace_cadence_review_artifact: dict | None = None,
    dense_threshold_relaxed_selection_score_artifact: dict | None = None,
    dense_nonseparable_review_artifact: dict | None = None,
    downstream_rerun_summary: dict | None = None,
) -> dict:
    classification, decision = _classification_and_decision(
        provenance_guard=provenance_guard,
        dense_trace=dense_trace,
        dense_selection_score_artifact=dense_selection_score_artifact,
        dense_trace_cadence_review_artifact=dense_trace_cadence_review_artifact,
        dense_threshold_relaxed_selection_score_artifact=dense_threshold_relaxed_selection_score_artifact,
        dense_nonseparable_review_artifact=dense_nonseparable_review_artifact,
    )

    payload = {
        "schema": SCHEMA,
        "hypothesis": "trace_too_sparse",
        "trace_origin": dense_trace["trace_origin"],
        "row_id": baseline["row_context"]["row_id"],
        "reference_move": baseline["row_context"]["reference_move"],
        "full_search_selected_move": baseline["row_context"][
            "full_search_selected_move"
        ],
        "trace_points": copy.deepcopy(dense_trace["trace_points"]),
        "insufficiency_reasons": copy.deepcopy(dense_trace["insufficiency_reasons"]),
        "checkpoint_capture_policy": copy.deepcopy(
            dense_trace["checkpoint_capture_policy"]
        ),
        "trace_points_summary": {
            "unique_simulation_checkpoints": copy.deepcopy(
                dense_trace["unique_simulation_checkpoints"]
            ),
            "unique_simulation_checkpoint_count": dense_trace[
                "unique_simulation_checkpoint_count"
            ],
            "duplicate_root_snapshot_count": dense_trace[
                "duplicate_root_snapshot_count"
            ],
            "first_divergent_selected_move_index": dense_trace[
                "first_divergent_selected_move_index"
            ],
            "has_additional_checkpoint_between_divergence_and_final": dense_trace[
                "has_additional_checkpoint_between_divergence_and_final"
            ],
        },
        "provenance_guard": copy.deepcopy(provenance_guard),
        "baseline_trigger": copy.deepcopy(baseline["baseline_trigger"]),
        "source_baseline_artifacts": {
            "trace_capture_artifact": copy.deepcopy(baseline["trace_capture_artifact"]),
            "selection_score_artifact": copy.deepcopy(
                baseline["selection_score_artifact"]
            ),
            "trace_cadence_review_artifact": copy.deepcopy(
                baseline["trace_cadence_review_artifact"]
            ),
            "nonseparable_review_artifact": copy.deepcopy(
                baseline["nonseparable_review_artifact"]
            ),
        },
        "row_context": copy.deepcopy(baseline["row_context"]),
        "source_selected_artifact": copy.deepcopy(
            baseline.get("source_selected_artifact")
        ),
        "search_settings": copy.deepcopy(baseline.get("search_settings")),
        "cadence_validation": {
            "passed": provenance_guard.get("passed")
            and not dense_trace.get("insufficiency_reasons"),
            "insufficiency_reasons": copy.deepcopy(
                dense_trace.get("insufficiency_reasons") or []
            )
            + copy.deepcopy(provenance_guard.get("failures") or []),
        },
        "downstream_artifacts": {
            "dense_selection_score_artifact": copy.deepcopy(
                dense_selection_score_artifact
            ),
            "dense_trace_cadence_review_artifact": copy.deepcopy(
                dense_trace_cadence_review_artifact
            ),
            "dense_threshold_relaxed_selection_score_artifact": copy.deepcopy(
                dense_threshold_relaxed_selection_score_artifact
            ),
            "dense_nonseparable_review_artifact": copy.deepcopy(
                dense_nonseparable_review_artifact
            ),
        },
        "downstream_rerun_summary": copy.deepcopy(downstream_rerun_summary),
        "classification": classification,
        "decision": decision,
        "artifact_write_summary": {
            "trace_cadence_capture_artifact_path": trace_cadence_capture_artifact_path,
            "trace_capture_sha256": None,
        },
    }
    payload["trace_capture_artifact_sha256"] = hashlib.sha256(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()
    payload["artifact_write_summary"]["trace_capture_sha256"] = payload[
        "trace_capture_artifact_sha256"
    ]
    payload["regenerated_shared_drift_artifact"] = copy.deepcopy(
        regenerated_shared_drift_artifact
    )
    if (
        payload["regenerated_shared_drift_artifact"] is None
        and payload["cadence_validation"]["passed"]
        and dense_trace.get("trace_points")
    ):
        payload["regenerated_shared_drift_artifact"] = (
            build_regenerated_shared_drift_artifact(
                baseline,
                dense_trace=dense_trace,
                trace_cadence_capture_artifact_path=trace_cadence_capture_artifact_path,
                trace_cadence_capture_artifact_sha256=payload[
                    "trace_capture_artifact_sha256"
                ],
            )
        )
    payload["trace_capture_artifact_sha256"] = _stable_trace_capture_artifact_sha256(
        payload
    )
    payload["artifact_write_summary"]["trace_capture_sha256"] = payload[
        "trace_capture_artifact_sha256"
    ]
    regenerated = payload.get("regenerated_shared_drift_artifact")
    if isinstance(regenerated, dict):
        provenance = regenerated.get("trace_capture_provenance")
        if isinstance(provenance, dict):
            provenance["trace_capture_artifact_sha256"] = payload[
                "trace_capture_artifact_sha256"
            ]
    return payload


def capture_dense_rerun_result(baseline: dict) -> dict:
    trace_capture_artifact = baseline["trace_capture_artifact"]
    source_artifact = trace_capture.load_source_shared_drift_artifact(
        Path(trace_capture_artifact["source_shared_drift_artifact"]["artifact_path"]),
        allow_fixture_provenance=False,
    )
    rerun_result = (
        trace_capture._default_rerun_result(
            source_artifact, capture_mode="extract_then_rerun"
        )
        or {}
    )
    return {
        "trace_points": copy.deepcopy(rerun_result.get("trace_points") or []),
        "row_context": trace_capture._row_context(source_artifact),
        "selected_artifact": copy.deepcopy(source_artifact.get("selected_artifact")),
        "search_settings": copy.deepcopy(
            (trace_capture._upstream_inputs(source_artifact) or {}).get(
                "search_settings"
            )
        ),
        "source_shared_drift_artifact": copy.deepcopy(source_artifact),
        "insufficiency_reasons": copy.deepcopy(
            rerun_result.get("insufficiency_reasons") or []
        ),
    }


def capture_dense_trace_points(baseline: dict) -> list[dict]:
    return capture_dense_rerun_result(baseline)["trace_points"]


def run_downstream_reruns(
    baseline: dict,
    *,
    dense_trace: dict,
    provenance_guard: dict,
    source_shared_drift_artifact: dict | None = None,
    out_path: Path,
) -> dict:
    if not provenance_guard.get("passed"):
        return {
            "regenerated_shared_drift_artifact": None,
            "dense_selection_score_artifact": None,
            "dense_trace_cadence_review_artifact": None,
            "dense_threshold_relaxed_selection_score_artifact": None,
            "dense_nonseparable_review_artifact": None,
            "downstream_rerun_summary": {
                "stopped_early": True,
                "stop_reason": "provenance_guard_failed",
            },
        }

    if dense_trace["insufficiency_reasons"]:
        return {
            "regenerated_shared_drift_artifact": None,
            "dense_selection_score_artifact": None,
            "dense_trace_cadence_review_artifact": None,
            "dense_threshold_relaxed_selection_score_artifact": None,
            "dense_nonseparable_review_artifact": None,
            "downstream_rerun_summary": {
                "stopped_early": True,
                "stop_reason": "trace_cadence_unresolved",
            },
        }

    dense_selection_score_path = str(
        out_path.with_name("capture_002_dense_selection_score_trace.json")
    )
    dense_trace_cadence_review_path = str(
        out_path.with_name("capture_002_dense_trace_cadence_review.json")
    )
    dense_threshold_relaxed_selection_score_path = str(
        out_path.with_name(
            "capture_002_dense_threshold_relaxed_selection_score_trace.json"
        )
    )
    dense_nonseparable_review_path = str(
        out_path.with_name("capture_002_dense_nonseparable_review.json")
    )

    regenerated_shared_drift_artifact = build_regenerated_shared_drift_artifact(
        baseline,
        dense_trace=dense_trace,
        source_shared_drift_artifact=source_shared_drift_artifact,
        trace_cadence_capture_artifact_path=str(out_path),
        trace_cadence_capture_artifact_sha256="",
    )
    dense_source_artifact = (
        selection_score_trace.load_source_shared_drift_artifact_document(
            regenerated_shared_drift_artifact,
            artifact_path=str(
                out_path.with_name(
                    "capture_002_trace_cadence_rehydrated_shared_drift.json"
                )
            ),
        )
    )
    dense_selection_score_artifact = selection_score_trace.build_payload(
        dense_source_artifact
    )

    dense_trace_capture_preview = {
        "schema": SCHEMA,
        "trace_origin": dense_trace["trace_origin"],
        "row_id": baseline["row_context"]["row_id"],
        "reference_move": baseline["row_context"]["reference_move"],
        "full_search_selected_move": baseline["row_context"][
            "full_search_selected_move"
        ],
        "trace_points": copy.deepcopy(dense_trace["trace_points"]),
        "insufficiency_reasons": [],
    }
    dense_trace_cadence_review_artifact = trace_cadence_review.build_payload(
        dense_trace_capture_preview,
        dense_selection_score_artifact,
        trace_capture_artifact_path=str(out_path),
        selection_score_artifact_path=dense_selection_score_path,
    )

    dense_threshold_relaxed_selection_score_artifact = None
    dense_nonseparable_review_artifact = None
    stopped_early = False
    stop_reason = None

    if (
        dense_selection_score_artifact["decision"]
        != "write_002_unresolved_trace_review_spec"
    ):
        stopped_early = True
        stop_reason = "dense_default_mechanism_resolved"
    elif (
        dense_trace_cadence_review_artifact["decision"]
        != "continue_002_threshold_too_strict_check"
    ):
        stopped_early = True
        stop_reason = "trace_cadence_unresolved"
    else:
        dense_threshold_relaxed_selection_score_artifact = (
            selection_score_trace.build_payload(
                dense_source_artifact,
                meaningful_q_margin=0.03,
                material_selection_score_margin=0.05,
                material_visit_share_margin=0.04,
            )
        )
        if (
            dense_threshold_relaxed_selection_score_artifact["decision"]
            != "write_002_unresolved_trace_review_spec"
        ):
            stopped_early = True
            stop_reason = "dense_threshold_mechanism_resolved"
        else:
            dense_nonseparable_review_artifact = nonseparable_review.build_payload(
                dense_selection_score_artifact,
                dense_trace_cadence_review_artifact,
                dense_threshold_relaxed_selection_score_artifact,
                source_selection_score_artifact_path=dense_selection_score_path,
                source_trace_cadence_review_artifact_path=dense_trace_cadence_review_path,
                source_threshold_review_artifact_path=dense_threshold_relaxed_selection_score_path,
            )

    return {
        "regenerated_shared_drift_artifact": regenerated_shared_drift_artifact,
        "dense_selection_score_artifact": dense_selection_score_artifact,
        "dense_trace_cadence_review_artifact": dense_trace_cadence_review_artifact,
        "dense_threshold_relaxed_selection_score_artifact": dense_threshold_relaxed_selection_score_artifact,
        "dense_nonseparable_review_artifact": dense_nonseparable_review_artifact,
        "downstream_rerun_summary": {
            "default_dense_selection_score_artifact_path": dense_selection_score_path,
            "dense_trace_cadence_review_artifact_path": dense_trace_cadence_review_path,
            "dense_threshold_relaxed_selection_score_artifact_path": None
            if dense_threshold_relaxed_selection_score_artifact is None
            else dense_threshold_relaxed_selection_score_path,
            "dense_nonseparable_review_artifact_path": None
            if dense_nonseparable_review_artifact is None
            else dense_nonseparable_review_path,
            "emitted_decisions": {
                "dense_default_selection_score": dense_selection_score_artifact[
                    "decision"
                ],
                "dense_trace_cadence_review": dense_trace_cadence_review_artifact[
                    "decision"
                ],
                "dense_threshold_relaxed_selection_score": None
                if dense_threshold_relaxed_selection_score_artifact is None
                else dense_threshold_relaxed_selection_score_artifact["decision"],
                "dense_nonseparable_review": None
                if dense_nonseparable_review_artifact is None
                else dense_nonseparable_review_artifact["decision"],
            },
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture dense cadence checkpoints and rerun 002 unresolved diagnostics"
    )
    parser.add_argument("--source-trace-capture-artifact", type=Path, required=True)
    parser.add_argument("--source-selection-score-artifact", type=Path, required=True)
    parser.add_argument(
        "--source-trace-cadence-review-artifact", type=Path, required=True
    )
    parser.add_argument("--source-nonseparable-review-artifact", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    baseline = load_baseline_inputs(
        trace_capture_artifact_path=args.source_trace_capture_artifact,
        selection_score_artifact_path=args.source_selection_score_artifact,
        trace_cadence_review_artifact_path=args.source_trace_cadence_review_artifact,
        nonseparable_review_artifact_path=args.source_nonseparable_review_artifact,
    )
    dense_rerun_result = capture_dense_rerun_result(baseline)
    dense_trace_points = dense_rerun_result["trace_points"]
    dense_trace = build_dense_trace(
        baseline,
        dense_trace_points=dense_trace_points,
        checkpoint_capture_policy={
            "capture_mode": "dense_full",
            "checkpoint_schedule": [
                trace_point["simulation"] for trace_point in dense_trace_points
            ],
            "root_snapshot_deduplicated": True,
            "source_artifact_path": str(args.source_trace_capture_artifact),
            "instrumentation_only_search_setting_overrides": {},
        },
    )
    provenance_guard = build_provenance_guard(
        baseline,
        dense_row_context=dense_rerun_result["row_context"],
        dense_selected_artifact=dense_rerun_result["selected_artifact"],
        dense_search_settings=dense_rerun_result["search_settings"],
        checkpoint_capture_policy=dense_trace["checkpoint_capture_policy"],
    )
    rerun_outputs = run_downstream_reruns(
        baseline,
        dense_trace=dense_trace,
        provenance_guard=provenance_guard,
        source_shared_drift_artifact=dense_rerun_result["source_shared_drift_artifact"],
        out_path=args.out,
    )
    payload = build_payload(
        baseline=baseline,
        dense_trace=dense_trace,
        provenance_guard=provenance_guard,
        trace_cadence_capture_artifact_path=str(args.out),
        regenerated_shared_drift_artifact=rerun_outputs[
            "regenerated_shared_drift_artifact"
        ],
        dense_selection_score_artifact=rerun_outputs["dense_selection_score_artifact"],
        dense_trace_cadence_review_artifact=rerun_outputs[
            "dense_trace_cadence_review_artifact"
        ],
        dense_threshold_relaxed_selection_score_artifact=rerun_outputs[
            "dense_threshold_relaxed_selection_score_artifact"
        ],
        dense_nonseparable_review_artifact=rerun_outputs[
            "dense_nonseparable_review_artifact"
        ],
        downstream_rerun_summary=rerun_outputs["downstream_rerun_summary"],
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
                "decision": payload["decision"],
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
