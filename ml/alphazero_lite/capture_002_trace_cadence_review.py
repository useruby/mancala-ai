from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

SCHEMA = "azlite_capture_002_trace_cadence_review_v1"
SOURCE_TRACE_CAPTURE_SCHEMA = "azlite_capture_002_trace_capture_v1"
SOURCE_TRACE_CAPTURE_SCHEMAS = {
    "azlite_capture_002_trace_capture_v1",
    "azlite_capture_002_trace_cadence_capture_v1",
}
SOURCE_SELECTION_SCORE_SCHEMA = "azlite_capture_002_selection_score_trace_v1"
ROW_ID = "capture_available-002"
CLASSIFICATION_DECISIONS = {
    "trace_too_sparse": "write_002_trace_cadence_capture_spec",
    "cadence_adequate": "continue_002_threshold_too_strict_check",
}


def _validate_integer(value, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _validate_row_id(value, *, context: str) -> str:
    if value != ROW_ID:
        raise ValueError(f"{context} must be {ROW_ID}")
    return value


def _validate_simulation(value, *, context: str) -> float:
    if value is None:
        raise ValueError(f"{context} is required")
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a finite number")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be numeric") from error
    if not math.isfinite(numeric_value):
        raise ValueError(f"{context} must be a finite number")
    return numeric_value


def _validate_required_finite_float(value, *, context: str) -> float:
    if value is None:
        raise ValueError(f"{context} must be a finite number")
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a finite number")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be a finite number") from error
    if not math.isfinite(numeric_value):
        raise ValueError(f"{context} must be a finite number")
    return numeric_value


def _validate_trace_points(trace_points: list[dict]) -> None:
    for index, trace_point in enumerate(trace_points):
        if not isinstance(trace_point, dict):
            raise ValueError(f"trace capture artifact trace_points[{index}] must be an object")
        _validate_simulation(trace_point.get("simulation"), context=f"trace capture artifact trace_points[{index}] simulation")
        if "selected_move" not in trace_point:
            raise ValueError(f"trace capture artifact trace_points[{index}] selected_move is required")
        _validate_integer(
            trace_point.get("selected_move"),
            context=f"trace capture artifact trace_points[{index}] selected_move",
        )


def _first_divergent_selected_move_index(trace_points: list[dict], *, final_selected_move: int) -> int | None:
    for index, trace_point in enumerate(trace_points):
        if trace_point.get("selected_move") == final_selected_move:
            return index
    return None


def _validate_matching_case(trace_capture_artifact: dict, selection_score_artifact: dict) -> None:
    selection_source_artifact = selection_score_artifact.get("source_artifact")
    if not isinstance(selection_source_artifact, dict):
        raise ValueError("selection score artifact source_artifact must be an object")

    selection_identity = _validate_selection_source_artifact_identity(selection_source_artifact)

    trace_row_id = _validate_row_id(
        trace_capture_artifact.get("row_id"),
        context="trace capture artifact row_id",
    )
    selection_row_id = _validate_row_id(
        selection_identity["row_id"],
        context="selection score artifact source_artifact.row_id",
    )
    if trace_row_id != selection_row_id:
        raise ValueError("trace capture artifact and selection score artifact must refer to the same row_id")

    for field in ("reference_move", "full_search_selected_move"):
        trace_value = _validate_integer(
            trace_capture_artifact.get(field),
            context=f"trace capture artifact {field}",
        )
        selection_value = _validate_integer(
            selection_identity[field],
            context=f"selection score artifact source_artifact.{field}",
        )
        if trace_value != selection_value:
            raise ValueError(f"trace capture artifact and selection score artifact must agree on {field}")


def _validate_selection_source_artifact_identity(selection_source_artifact: dict) -> dict:
    for field in ("row_id", "reference_move", "full_search_selected_move"):
        if field not in selection_source_artifact:
            raise ValueError(f"selection score artifact source_artifact.{field} is required")

    return {
        "row_id": _validate_row_id(
            selection_source_artifact.get("row_id"),
            context="selection score artifact source_artifact.row_id",
        ),
        "reference_move": _validate_integer(
            selection_source_artifact.get("reference_move"),
            context="selection score artifact source_artifact.reference_move",
        ),
        "full_search_selected_move": _validate_integer(
            selection_source_artifact.get("full_search_selected_move"),
            context="selection score artifact source_artifact.full_search_selected_move",
        ),
    }


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _normalized_trace_capture_input(artifact: dict) -> dict:
    normalized = copy.deepcopy(artifact)
    schema = normalized.get("schema")
    if schema not in SOURCE_TRACE_CAPTURE_SCHEMAS:
        raise ValueError(
            "trace capture artifact has wrong schema: expected one of "
            f"{sorted(SOURCE_TRACE_CAPTURE_SCHEMAS)}"
        )
    if schema == SOURCE_TRACE_CAPTURE_SCHEMA:
        expected_trace_origin = "extracted"
    else:
        expected_trace_origin = "dense_rerun"
    if normalized.get("trace_origin") != expected_trace_origin:
        raise ValueError(f"trace capture artifact trace_origin must be {expected_trace_origin}")
    return normalized


def _validate_dense_cadence_capture_artifact(artifact: dict) -> None:
    provenance_guard = artifact.get("provenance_guard")
    if not isinstance(provenance_guard, dict) or provenance_guard.get("passed") is not True:
        raise ValueError("trace capture artifact dense cadence capture provenance_guard must be an object with passed == True")

    trace_points_summary = artifact.get("trace_points_summary")
    if not isinstance(trace_points_summary, dict):
        raise ValueError("trace capture artifact dense cadence capture trace_points_summary must be an object")

    unique_simulation_checkpoints = trace_points_summary.get("unique_simulation_checkpoints")
    if not isinstance(unique_simulation_checkpoints, list):
        raise ValueError(
            "trace capture artifact dense cadence capture trace_points_summary.unique_simulation_checkpoints must be a list"
        )

    recomputed_unique_simulation_checkpoints = _unique_simulation_checkpoints(artifact["trace_points"])
    if unique_simulation_checkpoints != recomputed_unique_simulation_checkpoints:
        raise ValueError(
            "trace capture artifact dense cadence capture trace_points_summary.unique_simulation_checkpoints must match trace_points"
        )

    unique_simulation_checkpoint_count = trace_points_summary.get("unique_simulation_checkpoint_count")
    if unique_simulation_checkpoint_count != len(recomputed_unique_simulation_checkpoints):
        raise ValueError(
            "trace capture artifact dense cadence capture trace_points_summary.unique_simulation_checkpoint_count must match trace_points"
        )

    duplicate_root_snapshot_count = trace_points_summary.get("duplicate_root_snapshot_count")
    if duplicate_root_snapshot_count != len(artifact["trace_points"]) - len(recomputed_unique_simulation_checkpoints):
        raise ValueError(
            "trace capture artifact dense cadence capture trace_points_summary.duplicate_root_snapshot_count must match trace_points"
        )

    if unique_simulation_checkpoint_count < 4:
        raise ValueError(
            "trace capture artifact dense cadence capture must retain at least four unique simulation checkpoints"
        )

    recomputed_first_divergent_selected_move_index = _first_divergent_selected_move_index(
        artifact["trace_points"],
        final_selected_move=_validate_integer(
            artifact.get("full_search_selected_move"),
            context="trace capture artifact full_search_selected_move",
        ),
    )

    first_divergent_selected_move_index = trace_points_summary.get("first_divergent_selected_move_index")
    if recomputed_first_divergent_selected_move_index is None:
        raise ValueError(
            "trace capture artifact dense cadence capture trace_points must include the final selected move"
        )
    if isinstance(first_divergent_selected_move_index, bool) or not isinstance(first_divergent_selected_move_index, int):
        raise ValueError(
            "trace capture artifact dense cadence capture trace_points_summary.first_divergent_selected_move_index must be an integer"
        )
    if not 0 <= first_divergent_selected_move_index < len(artifact["trace_points"]):
        raise ValueError(
            "trace capture artifact dense cadence capture trace_points_summary.first_divergent_selected_move_index must point into trace_points"
        )
    if first_divergent_selected_move_index != recomputed_first_divergent_selected_move_index:
        raise ValueError(
            "trace capture artifact dense cadence capture trace_points_summary.first_divergent_selected_move_index must match trace_points"
        )

    divergent_simulation = _validate_simulation(
        artifact["trace_points"][recomputed_first_divergent_selected_move_index].get("simulation"),
        context=(
            "trace capture artifact "
            f"trace_points[{recomputed_first_divergent_selected_move_index}] simulation"
        ),
    )
    final_unique_checkpoint = recomputed_unique_simulation_checkpoints[-1]
    recomputed_has_additional_checkpoint_between_divergence_and_final = any(
        divergent_simulation < checkpoint < final_unique_checkpoint
        for checkpoint in recomputed_unique_simulation_checkpoints
    )
    if (
        trace_points_summary.get("has_additional_checkpoint_between_divergence_and_final")
        != recomputed_has_additional_checkpoint_between_divergence_and_final
    ):
        raise ValueError(
            "trace capture artifact dense cadence capture trace_points_summary.has_additional_checkpoint_between_divergence_and_final must match trace_points"
        )
    if not recomputed_has_additional_checkpoint_between_divergence_and_final:
        raise ValueError(
            "trace capture artifact dense cadence capture must retain an additional checkpoint between divergence and final"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review 002 trace cadence for unresolved downstream traces")
    parser.add_argument("--source-trace-capture-artifact", type=Path, required=True)
    parser.add_argument("--source-selection-score-artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def load_trace_capture_artifact(path: Path) -> dict:
    artifact = _normalized_trace_capture_input(load_json(path))
    if not isinstance(artifact.get("trace_points"), list) or not artifact["trace_points"]:
        raise ValueError("trace capture artifact must include non-empty trace_points")
    _validate_trace_points(artifact["trace_points"])
    if artifact.get("insufficiency_reasons") != []:
        raise ValueError("trace capture artifact must have no insufficiency_reasons")
    if artifact.get("schema") != SOURCE_TRACE_CAPTURE_SCHEMA:
        _validate_dense_cadence_capture_artifact(artifact)
    return artifact


def load_selection_score_artifact(path: Path) -> dict:
    artifact = load_json(path)
    if artifact.get("schema") != SOURCE_SELECTION_SCORE_SCHEMA:
        raise ValueError(
            f"selection score artifact has wrong schema: expected {SOURCE_SELECTION_SCORE_SCHEMA}"
        )
    if artifact.get("trace_origin") != "extracted":
        raise ValueError("selection score artifact trace_origin must be extracted")
    if artifact.get("insufficiency_reasons") != []:
        raise ValueError("selection score artifact must have no insufficiency_reasons")
    selection_source_artifact = artifact.get("source_artifact")
    if not isinstance(selection_source_artifact, dict):
        raise ValueError("selection score artifact source_artifact must be an object")
    _validate_selection_source_artifact_identity(selection_source_artifact)
    _validate_matching_case(
        {
            "row_id": (artifact.get("source_artifact") or {}).get("row_id"),
            "reference_move": (artifact.get("source_artifact") or {}).get("reference_move"),
            "full_search_selected_move": (artifact.get("source_artifact") or {}).get("full_search_selected_move"),
        },
        artifact,
    )
    classification = (artifact.get("classification") or {}).get("classification")
    if classification != "unresolved":
        raise ValueError("selection score artifact classification must be unresolved")
    _validate_required_finite_float(
        artifact.get("final_selected_minus_reference_visit_share"),
        context="selection score artifact final_selected_minus_reference_visit_share",
    )
    return artifact


def _unique_simulation_checkpoints(trace_points: list[dict]) -> list[float]:
    checkpoints = []
    for index, trace_point in enumerate(trace_points):
        simulation = _validate_simulation(
            trace_point.get("simulation"),
            context=f"trace capture artifact trace_points[{index}] simulation",
        )
        if simulation not in checkpoints:
            checkpoints.append(simulation)
    return checkpoints


def _selected_move_changes_without_captured_crossing(trace_points: list[dict]) -> bool:
    selected_moves = []
    for index, trace_point in enumerate(trace_points):
        if "selected_move" not in trace_point:
            raise ValueError(f"trace capture artifact trace_points[{index}] selected_move is required")
        selected_moves.append(
            _validate_integer(
                trace_point.get("selected_move"),
                context=f"trace capture artifact trace_points[{index}] selected_move",
            )
        )
    return len(set(selected_moves)) > 1 and len(_unique_simulation_checkpoints(trace_points)) < 4


def build_payload(
    trace_capture_artifact: dict,
    selection_score_artifact: dict,
    *,
    trace_capture_artifact_path: str,
    selection_score_artifact_path: str,
) -> dict:
    _validate_matching_case(trace_capture_artifact, selection_score_artifact)
    trace_points = copy.deepcopy(trace_capture_artifact["trace_points"])
    unique_simulations = _unique_simulation_checkpoints(trace_points)
    ambiguity_signals = []
    if _selected_move_changes_without_captured_crossing(trace_points):
        ambiguity_signals.append("selected_move_changed_without_captured_crossing")
    visit_share = _validate_required_finite_float(
        selection_score_artifact.get("final_selected_minus_reference_visit_share"),
        context="selection score artifact final_selected_minus_reference_visit_share",
    )
    if abs(0.05 - visit_share) <= 0.01:
        ambiguity_signals.append("near_material_visit_share_threshold")

    trace_too_sparse = len(unique_simulations) <= 2 and bool(ambiguity_signals)
    classification = "trace_too_sparse" if trace_too_sparse else "cadence_adequate"
    evidence_summary = (
        "Trace cadence is too sparse to localize the first crossing event for the validated 002 trace."
        if trace_too_sparse
        else "Trace cadence is adequate enough to evaluate threshold sensitivity before declaring non-separability."
    )

    return {
        "schema": SCHEMA,
        "hypothesis": "trace_too_sparse",
        "classification": {
            "classification": classification,
            "evidence_summary": evidence_summary,
        },
        "decision": CLASSIFICATION_DECISIONS[classification],
        "input_artifacts": {
            "trace_capture_artifact_path": trace_capture_artifact_path,
            "selection_score_artifact_path": selection_score_artifact_path,
        },
        "trace_origin": trace_capture_artifact["trace_origin"],
        "unique_simulation_checkpoints": unique_simulations,
        "unique_simulation_checkpoint_count": len(unique_simulations),
        "duplicate_root_snapshot_count": len(trace_points) - len(unique_simulations),
        "ambiguity_signals": ambiguity_signals,
        "trace_capture_excerpt": {
            "row_id": trace_capture_artifact.get("row_id"),
            "reference_move": trace_capture_artifact.get("reference_move"),
            "full_search_selected_move": trace_capture_artifact.get("full_search_selected_move"),
        },
        "selection_score_excerpt": {
            "final_selected_minus_reference_visit_share": selection_score_artifact.get(
                "final_selected_minus_reference_visit_share"
            ),
            "first_selected_material_visit_share_snapshot": selection_score_artifact.get(
                "first_selected_material_visit_share_snapshot"
            ),
            "first_selected_meaningful_q_support_snapshot": selection_score_artifact.get(
                "first_selected_meaningful_q_support_snapshot"
            ),
            "first_selected_selection_score_overtake_snapshot": selection_score_artifact.get(
                "first_selected_selection_score_overtake_snapshot"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trace_capture_artifact = load_trace_capture_artifact(args.source_trace_capture_artifact)
    selection_score_artifact = load_selection_score_artifact(args.source_selection_score_artifact)
    payload = build_payload(
        trace_capture_artifact,
        selection_score_artifact,
        trace_capture_artifact_path=str(args.source_trace_capture_artifact),
        selection_score_artifact_path=str(args.source_selection_score_artifact),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifact_path": str(args.out), "schema": payload["schema"], "decision": payload["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
