from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

SCHEMA = "azlite_capture_002_nonseparable_review_v1"
SOURCE_SELECTION_SCORE_SCHEMA = "azlite_capture_002_selection_score_trace_v1"
SOURCE_TRACE_CADENCE_REVIEW_SCHEMA = "azlite_capture_002_trace_cadence_review_v1"
ROW_ID = "capture_available-002"
CLASSIFICATION_DECISIONS = {
    "genuinely_not_separable": "stop_002_unresolved",
    "prerequisite_preempted": None,
}
THRESHOLD_REVIEW_SUPPORTED_DECISIONS = {
    "selection_score_pressure_confirmed": "write_002_selection_pressure_ablation_spec",
    "q_support_precedes_selection_score": "write_002_child_value_audit_spec",
}
DEFAULT_SELECTION_SCORE_UNRESOLVED_DECISION = "write_002_unresolved_trace_review_spec"
THRESHOLD_REVIEW_UNRESOLVED_DECISION = "write_002_unresolved_trace_review_spec"
BASELINE_THRESHOLDS = {
    "meaningful_q_margin": 0.03,
    "material_selection_score_margin": 0.05,
    "material_visit_share_margin": 0.05,
}
RELAXED_THRESHOLDS = {
    "meaningful_q_margin": 0.03,
    "material_selection_score_margin": 0.05,
    "material_visit_share_margin": 0.04,
}
RELAXED_MATERIAL_VISIT_SHARE_MARGIN = 0.04
CADENCE_CLASSIFICATION_DECISIONS = {
    "cadence_adequate": "continue_002_threshold_too_strict_check",
    "trace_too_sparse": "write_002_trace_cadence_capture_spec",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review whether the validated 002 trace is genuinely not separable"
    )
    parser.add_argument("--source-selection-score-artifact", type=Path, required=True)
    parser.add_argument(
        "--source-trace-cadence-review-artifact", type=Path, required=True
    )
    parser.add_argument("--source-threshold-review-artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def load_selection_score_artifact(path: Path) -> dict:
    artifact = load_json(path)
    validate_selection_score_artifact_contract(artifact)
    return artifact


def load_trace_cadence_review_artifact(path: Path) -> dict:
    artifact = load_json(path)
    validate_trace_cadence_review_artifact_contract(artifact)
    return artifact


def validate_trace_cadence_review_artifact(artifact: dict) -> None:
    classification = artifact.get("classification", {}).get("classification")
    expected_decision = CADENCE_CLASSIFICATION_DECISIONS.get(classification)
    if expected_decision is None or artifact.get("decision") != expected_decision:
        raise ValueError(
            "trace cadence review artifact must use an allowed classification/decision pair"
        )


def load_threshold_review_artifact(path: Path) -> dict:
    artifact = load_json(path)
    validate_selection_score_artifact_contract(artifact)
    validate_threshold_review_artifact(artifact)
    return artifact


def validate_selection_score_artifact_contract(artifact: dict) -> None:
    if artifact.get("schema") != SOURCE_SELECTION_SCORE_SCHEMA:
        raise ValueError(
            f"selection score artifact has wrong schema: expected {SOURCE_SELECTION_SCORE_SCHEMA}"
        )
    if artifact.get("insufficiency_reasons") != []:
        raise ValueError("selection score artifact must have no insufficiency_reasons")


def _validate_integer(value, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _selection_source_identity(artifact: dict, *, context: str) -> dict:
    source_artifact = artifact.get("source_artifact")
    if not isinstance(source_artifact, dict):
        raise ValueError(f"{context} source_artifact must be an object")
    row_id = source_artifact.get("row_id")
    if row_id != ROW_ID:
        raise ValueError(f"{context} source_artifact.row_id must be {ROW_ID}")
    return {
        "row_id": row_id,
        "reference_move": _validate_integer(
            source_artifact.get("reference_move"),
            context=f"{context} source_artifact.reference_move",
        ),
        "full_search_selected_move": _validate_integer(
            source_artifact.get("full_search_selected_move"),
            context=f"{context} source_artifact.full_search_selected_move",
        ),
        "selected_artifact": copy.deepcopy(source_artifact.get("selected_artifact")),
    }


def validate_input_chain_identity(
    default_selection_score_artifact: dict,
    trace_cadence_review_artifact: dict,
    threshold_review_artifact: dict,
) -> None:
    default_identity = _selection_source_identity(
        default_selection_score_artifact,
        context="default selection score artifact",
    )
    threshold_identity = _selection_source_identity(
        threshold_review_artifact,
        context="threshold review artifact",
    )
    if default_identity != threshold_identity:
        raise ValueError("selection score source identities must match")

    trace_capture_excerpt = trace_cadence_review_artifact.get("trace_capture_excerpt")
    if not isinstance(trace_capture_excerpt, dict):
        raise ValueError(
            "trace cadence review artifact trace_capture_excerpt must be an object"
        )
    if trace_capture_excerpt.get("row_id") != default_identity["row_id"]:
        raise ValueError(
            "trace cadence review artifact trace_capture_excerpt row_id must match selection source"
        )
    for field in ("reference_move", "full_search_selected_move"):
        excerpt_value = _validate_integer(
            trace_capture_excerpt.get(field),
            context=f"trace cadence review artifact trace_capture_excerpt.{field}",
        )
        if excerpt_value != default_identity[field]:
            raise ValueError(
                f"trace cadence review artifact trace_capture_excerpt {field} must match selection source"
            )

    selection_score_excerpt = trace_cadence_review_artifact.get(
        "selection_score_excerpt"
    )
    if not isinstance(selection_score_excerpt, dict):
        raise ValueError(
            "trace cadence review artifact selection_score_excerpt must be an object"
        )
    for field in (
        "final_selected_minus_reference_visit_share",
        "first_selected_material_visit_share_snapshot",
        "first_selected_meaningful_q_support_snapshot",
        "first_selected_selection_score_overtake_snapshot",
    ):
        if selection_score_excerpt.get(field) != default_selection_score_artifact.get(
            field
        ):
            raise ValueError(
                f"trace cadence review artifact selection_score_excerpt {field} must match default selection score artifact"
            )


def validate_trace_cadence_review_artifact_contract(artifact: dict) -> None:
    if artifact.get("schema") != SOURCE_TRACE_CADENCE_REVIEW_SCHEMA:
        raise ValueError(
            f"trace cadence review artifact has wrong schema: expected {SOURCE_TRACE_CADENCE_REVIEW_SCHEMA}"
        )
    validate_trace_cadence_review_artifact(artifact)


def validate_default_selection_score_artifact(artifact: dict) -> None:
    classification = artifact.get("classification", {}).get("classification")
    decision = artifact.get("decision")
    if artifact.get("thresholds") != BASELINE_THRESHOLDS:
        raise ValueError(
            "default selection score artifact must use the baseline threshold configuration"
        )
    if (
        classification != "unresolved"
        or decision != DEFAULT_SELECTION_SCORE_UNRESOLVED_DECISION
    ):
        raise ValueError(
            "default selection score artifact must be the unresolved baseline artifact"
        )


def validate_threshold_review_artifact(artifact: dict) -> None:
    classification = artifact.get("classification", {}).get("classification")
    decision = artifact.get("decision")
    if artifact.get("thresholds") != RELAXED_THRESHOLDS:
        raise ValueError(
            "threshold review artifact must use the relaxed threshold configuration"
        )
    if classification == "unresolved":
        if decision != THRESHOLD_REVIEW_UNRESOLVED_DECISION:
            raise ValueError(
                "threshold review artifact unresolved classification must keep the unresolved decision"
            )
        return
    expected_decision = THRESHOLD_REVIEW_SUPPORTED_DECISIONS.get(classification)
    if expected_decision is not None:
        if decision != expected_decision:
            raise ValueError(
                "threshold review artifact classification must match its supported-mechanism decision"
            )
        return
    raise ValueError(
        "threshold review artifact must represent a threshold-review prerequisite state"
    )


def build_payload(
    default_selection_score_artifact: dict,
    trace_cadence_review_artifact: dict,
    threshold_review_artifact: dict,
    *,
    source_selection_score_artifact_path: str,
    source_trace_cadence_review_artifact_path: str,
    source_threshold_review_artifact_path: str,
) -> dict:
    validate_selection_score_artifact_contract(default_selection_score_artifact)
    validate_trace_cadence_review_artifact_contract(trace_cadence_review_artifact)
    validate_selection_score_artifact_contract(threshold_review_artifact)
    validate_default_selection_score_artifact(default_selection_score_artifact)
    validate_threshold_review_artifact(threshold_review_artifact)
    validate_input_chain_identity(
        default_selection_score_artifact,
        trace_cadence_review_artifact,
        threshold_review_artifact,
    )

    cadence_decision = trace_cadence_review_artifact.get("decision")
    threshold_decision = threshold_review_artifact.get("decision")
    if cadence_decision != "continue_002_threshold_too_strict_check":
        return {
            "schema": SCHEMA,
            "hypothesis": "genuinely_not_separable",
            "classification": {
                "classification": "prerequisite_preempted",
                "evidence_summary": "Cadence review already supported a narrower branch before threshold or non-separable review.",
            },
            "decision": cadence_decision,
            "input_artifacts": {
                "source_selection_score_artifact_path": source_selection_score_artifact_path,
                "source_trace_cadence_review_artifact_path": source_trace_cadence_review_artifact_path,
                "source_threshold_review_artifact_path": source_threshold_review_artifact_path,
            },
        }
    if threshold_decision != THRESHOLD_REVIEW_UNRESOLVED_DECISION:
        return {
            "schema": SCHEMA,
            "hypothesis": "genuinely_not_separable",
            "classification": {
                "classification": "prerequisite_preempted",
                "evidence_summary": "Threshold review already supported a more specific mechanism branch.",
            },
            "decision": threshold_decision,
            "input_artifacts": {
                "source_selection_score_artifact_path": source_selection_score_artifact_path,
                "source_trace_cadence_review_artifact_path": source_trace_cadence_review_artifact_path,
                "source_threshold_review_artifact_path": source_threshold_review_artifact_path,
            },
        }

    return {
        "schema": SCHEMA,
        "hypothesis": "genuinely_not_separable",
        "classification": {
            "classification": "genuinely_not_separable",
            "evidence_summary": "Cadence is adequate and minimal visit-share threshold relaxation still does not isolate a supported mechanism for capture 002.",
        },
        "decision": CLASSIFICATION_DECISIONS["genuinely_not_separable"],
        "input_artifacts": {
            "source_selection_score_artifact_path": source_selection_score_artifact_path,
            "source_trace_cadence_review_artifact_path": source_trace_cadence_review_artifact_path,
            "source_threshold_review_artifact_path": source_threshold_review_artifact_path,
        },
        "thresholds_evaluated": {
            "default_material_visit_share_margin": default_selection_score_artifact[
                "thresholds"
            ]["material_visit_share_margin"],
            "relaxed_material_visit_share_margin": threshold_review_artifact[
                "thresholds"
            ]["material_visit_share_margin"],
        },
        "final_margin_summary": {
            "default_q_margin": default_selection_score_artifact.get(
                "final_selected_minus_reference_q"
            ),
            "default_selection_score_margin": default_selection_score_artifact.get(
                "final_selected_minus_reference_selection_score"
            ),
            "default_visit_share_margin": default_selection_score_artifact.get(
                "final_selected_minus_reference_visit_share"
            ),
            "relaxed_q_margin": threshold_review_artifact.get(
                "final_selected_minus_reference_q"
            ),
            "relaxed_selection_score_margin": threshold_review_artifact.get(
                "final_selected_minus_reference_selection_score"
            ),
            "relaxed_visit_share_margin": threshold_review_artifact.get(
                "final_selected_minus_reference_visit_share"
            ),
        },
        "source_snapshots": {
            "default_classification": copy.deepcopy(
                default_selection_score_artifact.get("classification")
            ),
            "cadence_classification": copy.deepcopy(
                trace_cadence_review_artifact.get("classification")
            ),
            "threshold_classification": copy.deepcopy(
                threshold_review_artifact.get("classification")
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    default_selection_score_artifact = load_selection_score_artifact(
        args.source_selection_score_artifact
    )
    trace_cadence_review_artifact = load_trace_cadence_review_artifact(
        args.source_trace_cadence_review_artifact
    )
    threshold_review_artifact = load_threshold_review_artifact(
        args.source_threshold_review_artifact
    )
    payload = build_payload(
        default_selection_score_artifact,
        trace_cadence_review_artifact,
        threshold_review_artifact,
        source_selection_score_artifact_path=str(args.source_selection_score_artifact),
        source_trace_cadence_review_artifact_path=str(
            args.source_trace_cadence_review_artifact
        ),
        source_threshold_review_artifact_path=str(
            args.source_threshold_review_artifact
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
                "decision": payload["decision"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
